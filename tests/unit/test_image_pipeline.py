# tests/unit/test_image_pipeline.py
# E-CIP v3.0 — Unit tests for data/pipelines/image_pipeline.py

from __future__ import annotations

from pathlib import Path

from data.pipelines.image_pipeline import (
    MAX_FILE_SIZE_BYTES,
    MIN_DIMENSION_PX,
    MIN_FILE_SIZE_BYTES,
    validate_image,
)


def _make_image(path: Path, width: int, height: int) -> None:
    """
    Random-noise pixels, saved as PNG — a flat-color JPEG compresses well
    below MIN_FILE_SIZE_BYTES even at moderate dimensions, which would
    make the small-dimension test accidentally exercise the file-size
    rejection path instead of the dimension check it's meant to test.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, mode="RGB")
    img.save(path, format="PNG")


class TestValidateImage:
    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        is_valid, reason = validate_image(tmp_path / "nope.jpg")
        assert is_valid is False
        assert "not found" in reason

    def test_rejects_invalid_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "file.gif"
        path.write_bytes(b"x" * (MIN_FILE_SIZE_BYTES + 1))
        is_valid, reason = validate_image(path)
        assert is_valid is False
        assert "extension" in reason

    def test_rejects_file_too_small(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.jpg"
        path.write_bytes(b"x" * (MIN_FILE_SIZE_BYTES - 1))
        is_valid, reason = validate_image(path)
        assert is_valid is False
        assert "too small" in reason
        # Fix #3: this specifically exercises the corrected `<` comparison —
        # the v2 blueprint bug was a syntactically broken comparison that
        # silently accepted every file regardless of size.
        assert (MIN_FILE_SIZE_BYTES - 1) < MIN_FILE_SIZE_BYTES

    def test_rejects_file_too_large(self, tmp_path: Path) -> None:
        path = tmp_path / "huge.jpg"
        path.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))
        is_valid, reason = validate_image(path)
        assert is_valid is False
        assert "too large" in reason

    def test_rejects_corrupt_image_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.jpg"
        # Right size range, valid extension, but not a real image.
        path.write_bytes(b"not a real jpeg" * 1000)
        is_valid, reason = validate_image(path)
        assert is_valid is False
        assert "corrupt" in reason or "unreadable" in reason

    def test_rejects_image_below_min_dimension(self, tmp_path: Path) -> None:
        path = tmp_path / "small.png"
        _make_image(path, MIN_DIMENSION_PX - 1, MIN_DIMENSION_PX - 1)
        assert path.stat().st_size >= MIN_FILE_SIZE_BYTES  # confirm this tests dimension, not size
        is_valid, reason = validate_image(path)
        assert is_valid is False
        assert "too small" in reason

    def test_accepts_valid_image(self, tmp_path: Path) -> None:
        path = tmp_path / "valid.png"
        _make_image(path, 256, 256)
        is_valid, reason = validate_image(path)
        assert is_valid is True
        assert reason == "ok"
