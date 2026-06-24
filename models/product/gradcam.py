# models/product/gradcam.py
# E-CIP v3.0 — Grad-CAM Explainability for EfficientNet-B3
# Blueprint Section 11 — Fix #9 and Fix #30
#
# Generates class activation heatmaps overlaid on product images.
# Saved as PNG to Docker volume mount with explicit TTL.
# Served via /v1/explain/gradcam/{request_id}
#
# Fix #9:  Storage path: /app/storage/gradcam/{request_id}.png
#          Docker volume-mounted — survives container restart.
# Fix #30: gradcam_expires_at returned in API response.
#          Cleanup via Celery Beat every 30 min.
#
# Verification requirement (Week 6):
#   Heatmaps must highlight product regions, NOT background.
#   Spot-check ≥ 20 images per class before Phase 2 gate.
#
# Usage (Colab/Kaggle after training):
#   from models.product.gradcam import GradCAMGenerator
#   gen = GradCAMGenerator(model, target_layer="blocks[-1]")
#   heatmap_path = gen.generate(image_tensor, class_idx, request_id)

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# ─── Constants ────────────────────────────────────────────────────────────────

# Fix #9: Grad-CAM storage path (Docker volume mount in production)
GRADCAM_STORAGE_DIR = Path("storage/gradcam")
GRADCAM_TTL_HOURS = 1  # Fix #30: explicit TTL — 404 after expiry

# Heatmap overlay parameters
HEATMAP_ALPHA = 0.4       # transparency of heatmap overlay
HEATMAP_COLORMAP = "jet"  # colour map for activation intensity


# ─── Grad-CAM generator ───────────────────────────────────────────────────────

class GradCAMGenerator:
    """
    Grad-CAM class activation map generator for EfficientNet-B3.

    Blueprint Section 11:
    Class activation heatmap overlaid on product image.
    Saved as PNG to Docker volume mount with explicit TTL.

    Spot-check requirement:
    Heatmaps must highlight product regions, not background.
    Verify on ≥ 20 images per class before Phase 2 gate.
    """

    def __init__(
        self,
        model: Any,
        target_layer: str = "auto",
        storage_dir: Path = GRADCAM_STORAGE_DIR,
    ) -> None:
        self.model = model
        self.target_layer_name = target_layer
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._gradients: Any = None
        self._activations: Any = None
        self._hooks: list[Any] = []

    def _get_target_layer(self) -> Any:
        """
        Get the target convolutional layer for Grad-CAM.
        For EfficientNet-B3 (timm), use the last block's conv layer.
        """
        if self.model is None:
            return None

        if self.target_layer_name == "auto":
            # EfficientNet-B3 in timm: last block before global pool
            try:
                return self.model.blocks[-1][-1].conv_pwl
            except (AttributeError, IndexError):
                pass
            # Fallback: find last Conv2d layer
            last_conv = None
            try:
                import torch.nn as nn
                for module in self.model.modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module
                return last_conv
            except ImportError:
                return None

        # Named layer lookup
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                return module
        return None

    def _register_hooks(self, target_layer: Any) -> None:
        """Register forward and backward hooks for gradient capture."""
        def forward_hook(module: Any, input: Any, output: Any) -> None:
            self._activations = output.detach()

        def backward_hook(module: Any, grad_input: Any, grad_output: Any) -> None:
            self._gradients = grad_output[0].detach()

        self._hooks = [
            target_layer.register_forward_hook(forward_hook),
            target_layer.register_full_backward_hook(backward_hook),
        ]

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate_heatmap(
        self,
        image_tensor: Any,
        class_idx: int,
        device: Any,
    ) -> Any:
        """
        Generate raw Grad-CAM heatmap tensor for a single image.

        Args:
            image_tensor: Preprocessed image tensor (1, C, H, W)
            class_idx: Target class index for activation map
            device: torch device

        Returns:
            numpy array of shape (H, W) with values in [0, 1]
        """
        try:
            import torch
            import torch.nn.functional as functional

            target_layer = self._get_target_layer()
            if target_layer is None:
                print("  ✗ Target layer not found — Grad-CAM unavailable")
                return None

            self._register_hooks(target_layer)

            # Forward pass
            self.model.eval()
            image_tensor = image_tensor.to(device)
            output = self.model(image_tensor)

            # Backward pass for target class
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, class_idx] = 1.0
            output.backward(gradient=one_hot, retain_graph=True)

            self._remove_hooks()

            # Compute Grad-CAM
            gradients = self._gradients    # (1, C, H, W)
            activations = self._activations  # (1, C, H, W)

            if gradients is None or activations is None:
                return None

            # Global average pooling of gradients
            weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

            # Weighted combination of activation maps
            cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
            cam = functional.relu(cam)  # keep only positive activations

            # Normalise to [0, 1]
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            # Resize to image dimensions
            h = image_tensor.shape[2]
            w = image_tensor.shape[3]
            cam = functional.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

            return cam.squeeze().cpu().numpy()

        except Exception as e:
            self._remove_hooks()
            print(f"  ✗ Grad-CAM generation failed: {e}")
            return None

    def overlay_heatmap(
        self,
        original_image: Any,
        heatmap: Any,
        alpha: float = HEATMAP_ALPHA,
    ) -> Any:
        """
        Overlay Grad-CAM heatmap on original image.

        Args:
            original_image: PIL Image or numpy array (H, W, 3)
            heatmap: numpy array (H, W) in [0, 1]
            alpha: heatmap transparency (0=invisible, 1=opaque)

        Returns:
            numpy array (H, W, 3) with heatmap overlay
        """
        try:
            import cv2  # type: ignore
            import numpy as np

            # Convert PIL to numpy if needed
            if hasattr(original_image, "convert"):
                original_image = np.array(original_image.convert("RGB"))

            img = original_image.astype(np.float32) / 255.0

            # Apply colormap to heatmap
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            heatmap_float = heatmap_colored.astype(np.float32) / 255.0

            # Resize heatmap to match image if needed
            if heatmap_float.shape[:2] != img.shape[:2]:
                heatmap_float = cv2.resize(
                    heatmap_float, (img.shape[1], img.shape[0])
                )

            # Overlay
            overlaid = (1 - alpha) * img + alpha * heatmap_float
            overlaid = np.clip(overlaid, 0, 1)
            return (overlaid * 255).astype(np.uint8)

        except ImportError:
            # cv2 not available — return heatmap as grayscale overlay
            try:
                import numpy as np
                heatmap_rgb = np.stack([heatmap, heatmap, heatmap], axis=-1)
                return (heatmap_rgb * 255).astype(np.uint8)
            except ImportError:
                return None

    def save_gradcam(
        self,
        overlaid_image: Any,
        request_id: str,
    ) -> tuple[Path, str]:
        """
        Save Grad-CAM overlay to storage with TTL.

        Fix #9:  Storage at self.storage_dir/{request_id}.png
        Fix #30: Returns expires_at ISO timestamp for API response.

        Returns:
            (file_path, expires_at_iso_string)
        """
        try:
            import numpy as np
            from PIL import Image

            output_path = self.storage_dir / f"{request_id}.png"

            if isinstance(overlaid_image, np.ndarray):
                img = Image.fromarray(overlaid_image.astype(np.uint8))
            else:
                img = overlaid_image

            img.save(output_path, format="PNG", optimize=True)

            expires_at = datetime.now(UTC) + timedelta(hours=GRADCAM_TTL_HOURS)
            expires_at_iso = expires_at.isoformat()

            print(f"  ✓ Grad-CAM saved: {output_path}")
            print(f"    Expires at    : {expires_at_iso}")

            return output_path, expires_at_iso

        except ImportError as e:
            print(f"  ✗ Grad-CAM save failed — missing dependency: {e}")
            placeholder = self.storage_dir / f"{request_id}.json"
            placeholder.write_text(json.dumps({
                "status": "stub",
                "request_id": request_id,
            }))
            expires_at = datetime.now(UTC) + timedelta(hours=GRADCAM_TTL_HOURS)
            return placeholder, expires_at.isoformat()

    def generate(
        self,
        image_tensor: Any,
        original_image: Any,
        class_idx: int,
        request_id: str,
        device: Any,
    ) -> dict[str, Any]:
        """
        Full Grad-CAM pipeline: generate → overlay → save.

        Returns dict with path, expires_at, and quality metrics.
        """
        t0 = time.time()

        heatmap = self.generate_heatmap(image_tensor, class_idx, device)
        if heatmap is None:
            return {"error": "Grad-CAM generation failed"}

        overlaid = self.overlay_heatmap(original_image, heatmap)
        if overlaid is None:
            return {"error": "Heatmap overlay failed"}

        file_path, expires_at = self.save_gradcam(overlaid, request_id)

        return {
            "request_id": request_id,
            "gradcam_path": str(file_path),
            "gradcam_url": f"/v1/explain/gradcam/{request_id}",
            "gradcam_expires_at": expires_at,
            "generation_ms": int((time.time() - t0) * 1000),
        }


# ─── Cleanup utilities ────────────────────────────────────────────────────────

def cleanup_expired_gradcam(
    storage_dir: Path = GRADCAM_STORAGE_DIR,
    ttl_hours: int = GRADCAM_TTL_HOURS,
) -> dict[str, int]:
    """
    Remove expired Grad-CAM files.
    Called by Celery Beat every 30 minutes (mlops/beat_schedule.py).
    Blueprint Section 11 — Fix #30.
    """
    if not storage_dir.exists():
        return {"removed": 0, "remaining": 0}

    now = time.time()
    ttl_seconds = ttl_hours * 3600
    removed = 0
    remaining = 0

    for file_path in storage_dir.glob("*.png"):
        age_seconds = now - file_path.stat().st_mtime
        if age_seconds > ttl_seconds:
            file_path.unlink()
            removed += 1
        else:
            remaining += 1

    # Also clean stub JSON files
    for file_path in storage_dir.glob("*.json"):
        age_seconds = now - file_path.stat().st_mtime
        if age_seconds > ttl_seconds:
            file_path.unlink()

    if removed > 0:
        print(f"  Grad-CAM cleanup: removed {removed}, remaining {remaining}")

    return {"removed": removed, "remaining": remaining}


def generate_request_id(prefix: str = "req") -> str:
    """Generate a unique request ID for Grad-CAM file naming."""
    timestamp = str(time.time()).encode()
    hash_suffix = hashlib.sha256(timestamp).hexdigest()[:8]
    return f"{prefix}_{hash_suffix}"


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  E-CIP v3.0 — Grad-CAM Generator")
    print("  Blueprint Section 11 — Fix #9 + Fix #30")
    print("=" * 60)

    # Verify storage directory
    GRADCAM_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Storage dir : {GRADCAM_STORAGE_DIR}")
    print(f"  TTL         : {GRADCAM_TTL_HOURS} hour(s)")

    # Test cleanup utility
    stats = cleanup_expired_gradcam()
    print(f"  Cleanup test: {stats}")

    # Test request ID generation
    req_id = generate_request_id()
    print(f"  Sample req ID: {req_id}")

    print("\n  GradCAMGenerator class ready.")
    print("  Full pipeline runs in Colab/Kaggle after training (Phase 2, Week 6).")
    print("\n  Spot-check requirement:")
    print("  Heatmaps must highlight product regions, NOT background.")
    print("  Verify ≥ 20 images/class before Phase 2 gate.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
