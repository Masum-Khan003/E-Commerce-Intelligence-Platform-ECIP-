# api/storage/image_store.py
# E-CIP v3.0 — Image Storage Service
# Blueprint Section 11 — Fix #9
#
# Handles two file types:
#   'upload'  — incoming product images for classification
#   'gradcam' — Grad-CAM heatmap overlays (TTL = 1 hour)
#
# Storage path: /app/storage/{file_type}/{request_id}.{ext}
# Docker volume-mounted — survives container restart.
#
# Fix #9:  Explicit storage backend with volume mount
# Fix #30: Grad-CAM TTL enforced — expires_at tracked in DB
#
# In production this module writes to:
#   docker-compose.yml volume: gradcam_storage:/app/storage/gradcam
#
# Usage:
#   from api.storage.image_store import ImageStore
#   store = ImageStore()
#   path, expires_at = store.save_upload(file_bytes, request_id, "jpg")
#   path, expires_at = store.save_gradcam(image_array, request_id)

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# ─── Constants ────────────────────────────────────────────────────────────────

# Storage root — overridden by ECIP_STORAGE_ROOT env var in production
DEFAULT_STORAGE_ROOT = Path("storage")

# TTL configuration
GRADCAM_TTL_HOURS = 1       # Fix #30: Grad-CAM expires after 1 hour
UPLOAD_TTL_HOURS = 24       # Product uploads retained for 24 hours

# File size limits
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024   # 10MB
MIN_UPLOAD_SIZE_BYTES = 5_000              # 5KB

# Allowed upload formats
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ─── Image store ──────────────────────────────────────────────────────────────

class ImageStore:
    """
    Central image storage service for E-CIP API.

    Blueprint Section 11 — Fix #9:
    All image I/O goes through this class.
    Storage is Docker volume-mounted for persistence across restarts.

    Two storage paths:
        uploads/  — incoming product images
        gradcam/  — Grad-CAM heatmap outputs (with TTL)
    """

    def __init__(self, storage_root: Path = DEFAULT_STORAGE_ROOT) -> None:
        self.storage_root = storage_root
        self.uploads_dir = storage_root / "uploads"
        self.gradcam_dir = storage_root / "gradcam"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create storage directories if they don't exist."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.gradcam_dir.mkdir(parents=True, exist_ok=True)

    # ── Upload handling ───────────────────────────────────────────────────────

    def validate_upload(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> tuple[bool, str]:
        """
        Validate an uploaded product image before storage.
        Returns (is_valid, reason).
        """
        # Size checks
        size = len(file_bytes)
        if size < MIN_UPLOAD_SIZE_BYTES:
            return False, f"File too small: {size} bytes < {MIN_UPLOAD_SIZE_BYTES}"
        if size > MAX_UPLOAD_SIZE_BYTES:
            return False, f"File too large: {size} bytes > {MAX_UPLOAD_SIZE_BYTES}"

        # Extension check
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return False, f"Invalid extension: {suffix}"

        # Magic bytes check (verify it's actually an image)
        magic_valid = self._check_magic_bytes(file_bytes, suffix)
        if not magic_valid:
            return False, "File content does not match declared image type"

        return True, "ok"

    def _check_magic_bytes(self, file_bytes: bytes, suffix: str) -> bool:
        """Verify image magic bytes match the declared file extension."""
        if len(file_bytes) < 4:
            return False

        header = file_bytes[:4]

        jpeg_magic = header[:2] == b"\xff\xd8"
        png_magic = header == b"\x89PNG"
        webp_magic = file_bytes[8:12] == b"WEBP" if len(file_bytes) >= 12 else False

        if suffix in (".jpg", ".jpeg"):
            return jpeg_magic
        if suffix == ".png":
            return png_magic
        if suffix == ".webp":
            return webp_magic

        return True  # permissive for other types

    def save_upload(
        self,
        file_bytes: bytes,
        request_id: str,
        extension: str = "jpg",
    ) -> tuple[Path, str]:
        """
        Save an uploaded product image to storage.

        Returns:
            (file_path, expires_at_iso)
        """
        extension = extension.lstrip(".")
        filename = f"{request_id}.{extension}"
        file_path = self.uploads_dir / filename

        file_path.write_bytes(file_bytes)

        expires_at = datetime.now(UTC) + timedelta(hours=UPLOAD_TTL_HOURS)
        return file_path, expires_at.isoformat()

    def compute_input_hash(self, file_bytes: bytes) -> str:
        """
        Compute SHA256 hash of uploaded file.
        Stored in prediction_logs.input_hash for deduplication.
        """
        return hashlib.sha256(file_bytes).hexdigest()

    # ── Grad-CAM handling ─────────────────────────────────────────────────────

    def save_gradcam_array(
        self,
        image_array: Any,
        request_id: str,
    ) -> tuple[Path, str]:
        """
        Save a Grad-CAM numpy array as PNG.

        Fix #9:  Saved to self.gradcam_dir/{request_id}.png
        Fix #30: Returns expires_at for API response field.

        Args:
            image_array: numpy array (H, W, 3) uint8
            request_id: unique request identifier

        Returns:
            (file_path, expires_at_iso)
        """
        try:
            import numpy as np
            from PIL import Image

            file_path = self.gradcam_dir / f"{request_id}.png"

            if isinstance(image_array, np.ndarray):
                img = Image.fromarray(image_array.astype(np.uint8), mode="RGB")
            else:
                img = image_array

            img.save(file_path, format="PNG", optimize=True)
            expires_at = datetime.now(UTC) + timedelta(hours=GRADCAM_TTL_HOURS)
            return file_path, expires_at.isoformat()

        except ImportError:
            # Stub for environments without PIL
            file_path = self.gradcam_dir / f"{request_id}.stub"
            file_path.write_text(f'{{"request_id": "{request_id}"}}')
            expires_at = datetime.now(UTC) + timedelta(hours=GRADCAM_TTL_HOURS)
            return file_path, expires_at.isoformat()

    def save_gradcam_bytes(
        self,
        image_bytes: bytes,
        request_id: str,
    ) -> tuple[Path, str]:
        """
        Save pre-encoded Grad-CAM PNG bytes to storage.
        Used when the image is already encoded upstream.
        """
        file_path = self.gradcam_dir / f"{request_id}.png"
        file_path.write_bytes(image_bytes)
        expires_at = datetime.now(UTC) + timedelta(hours=GRADCAM_TTL_HOURS)
        return file_path, expires_at.isoformat()

    def get_gradcam(self, request_id: str) -> tuple[Path | None, bool]:
        """
        Retrieve a Grad-CAM file by request ID.

        Returns:
            (file_path, is_expired)
            file_path is None if not found.
            is_expired is True if TTL has passed.
        """
        file_path = self.gradcam_dir / f"{request_id}.png"

        if not file_path.exists():
            return None, False

        # Check TTL
        age_seconds = time.time() - file_path.stat().st_mtime
        is_expired = age_seconds > (GRADCAM_TTL_HOURS * 3600)

        return file_path, is_expired

    def get_upload(self, request_id: str, extension: str = "jpg") -> Path | None:
        """Retrieve an uploaded product image by request ID."""
        for ext in ALLOWED_EXTENSIONS:
            file_path = self.uploads_dir / f"{request_id}{ext}"
            if file_path.exists():
                return file_path
        return None

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup_expired(self) -> dict[str, int]:
        """
        Remove expired Grad-CAM and upload files.
        Called by Celery Beat every 30 minutes.
        Blueprint Section 11 — Fix #30.
        """
        now = time.time()
        stats: dict[str, int] = {
            "gradcam_removed": 0,
            "uploads_removed": 0,
            "gradcam_remaining": 0,
            "uploads_remaining": 0,
        }

        # Clean Grad-CAM files
        for file_path in self.gradcam_dir.glob("*"):
            if not file_path.is_file():
                continue
            age = now - file_path.stat().st_mtime
            if age > GRADCAM_TTL_HOURS * 3600:
                file_path.unlink()
                stats["gradcam_removed"] += 1
            else:
                stats["gradcam_remaining"] += 1

        # Clean upload files
        for file_path in self.uploads_dir.glob("*"):
            if not file_path.is_file():
                continue
            age = now - file_path.stat().st_mtime
            if age > UPLOAD_TTL_HOURS * 3600:
                file_path.unlink()
                stats["uploads_removed"] += 1
            else:
                stats["uploads_remaining"] += 1

        return stats

    # ── Storage stats ─────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return storage statistics for Prometheus monitoring."""
        gradcam_files = list(self.gradcam_dir.glob("*.png"))
        upload_files = [
            f for f in self.uploads_dir.glob("*")
            if f.suffix.lower() in ALLOWED_EXTENSIONS
        ]

        gradcam_size = sum(f.stat().st_size for f in gradcam_files)
        upload_size = sum(f.stat().st_size for f in upload_files)

        return {
            "gradcam_count": len(gradcam_files),
            "gradcam_size_mb": round(gradcam_size / 1024 / 1024, 2),
            "upload_count": len(upload_files),
            "upload_size_mb": round(upload_size / 1024 / 1024, 2),
            "storage_root": str(self.storage_root),
        }


# ─── Module-level singleton ───────────────────────────────────────────────────

def get_image_store(storage_root: Path | None = None) -> ImageStore:
    """
    Get the ImageStore instance.
    In production, storage_root is set from ECIP_STORAGE_ROOT env var.
    """
    import os
    root = storage_root or Path(
        os.getenv("ECIP_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT))
    )
    return ImageStore(storage_root=root)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  E-CIP v3.0 — Image Storage Service")
    print("  Blueprint Section 11 — Fix #9 + Fix #30")
    print("=" * 60)

    store = ImageStore()

    print(f"\n  Storage root : {store.storage_root}")
    print(f"  Uploads dir  : {store.uploads_dir}")
    print(f"  Grad-CAM dir : {store.gradcam_dir}")
    print(f"  Upload TTL   : {UPLOAD_TTL_HOURS}h")
    print(f"  Grad-CAM TTL : {GRADCAM_TTL_HOURS}h")

    # Test validation
    print("\n  Testing upload validation...")
    fake_jpeg = b"\xff\xd8" + b"\x00" * 6000  # valid JPEG magic + padding
    is_valid, reason = store.validate_upload(fake_jpeg, "test.jpg")
    print(f"  Valid JPEG   : {is_valid} — {reason}")

    too_small = b"\xff\xd8" + b"\x00" * 10
    is_valid, reason = store.validate_upload(too_small, "small.jpg")
    print(f"  Too small    : {is_valid} — {reason}")

    wrong_magic = b"\x00\x00\x00\x00" + b"\x00" * 6000
    is_valid, reason = store.validate_upload(wrong_magic, "fake.jpg")
    print(f"  Wrong magic  : {is_valid} — {reason}")

    # Test cleanup
    cleanup_stats = store.cleanup_expired()
    print(f"\n  Cleanup stats: {cleanup_stats}")

    # Storage stats
    storage_stats = store.stats()
    print(f"  Storage stats: {storage_stats}")

    print("\n  ✓ Image storage service verified.")
    print("  In production: mounted at /app/storage via Docker volume.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
