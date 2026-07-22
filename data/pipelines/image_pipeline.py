# data/pipelines/image_pipeline.py
# E-CIP v3.0 — Image Data Pipeline
# Blueprint Section 03 (Fix #3) + Section 07
#
# Responsibilities:
#   - Validate images (syntax-corrected from v2 blueprint)
#   - SHA256 deduplication across train/val/test splits
#   - Stratified train/val/test split (70/15/15)
#   - Apply transforms (TRAIN_TRANSFORM, VAL_TRANSFORM)
#   - Save processed split manifests to data/processed/images/
#
# Usage:
#   python data/pipelines/image_pipeline.py
#   python data/pipelines/image_pipeline.py --sample-only
#   python data/pipelines/image_pipeline.py --input data/raw/fashion_product_images_fullres/by_category

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw/fashion_product_images_fullres/by_category")
PROCESSED_DIR = Path("data/processed/images")
SAMPLES_DIR = Path("data/samples/images")
FEATURE_STORE_DIR = Path("data/feature_store/product_features")

# ─── Constants ────────────────────────────────────────────────────────────────

# Blueprint Section 07: dev sample — 2K images, 250 per class
DEV_SAMPLE_SIZE_PER_CLASS = 250

# Train/val/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Image validation thresholds (Blueprint Section 03 — Fix #3)
MIN_FILE_SIZE_BYTES = 5_000       # reject tiny/corrupt files
MAX_FILE_SIZE_BYTES = 10_000_000  # reject files > 10MB
MIN_DIMENSION_PX = 64            # reject images too small to be useful

# Supported extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Target categories (blueprint Section 03)
TARGET_CATEGORIES = [
    "Electronics",
    "Fashion",
    "Home & Kitchen",
    "Sports",
    "Furniture",
    "Beauty",
    "Books",
    "Toys",
]


# ─── Transforms specification ─────────────────────────────────────────────────
# NOTE: Actual torchvision transforms are applied during training in
# models/product/train.py — not at pipeline time.
# The pipeline outputs file paths + metadata; transforms are applied
# by the DataLoader. This avoids storing augmented copies of 10K images.
#
# Defined here as documentation of the transform specification.
# Blueprint Section 03.

TRANSFORM_SPEC = {
    "train": {
        "resize": (300, 300),
        "random_horizontal_flip": 0.5,
        "color_jitter": {
            "brightness": 0.3,
            "contrast": 0.3,
            "saturation": 0.2,
        },
        "random_rotation_degrees": 15,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
    "val": {
        "resize": (300, 300),
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
    "test": {
        "resize": (300, 300),
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
}


# ─── Image validation ─────────────────────────────────────────────────────────

def validate_image(path: Path) -> tuple[bool, str]:
    """
    Validate a single image file.

    Blueprint Section 03 — Critical Fix #3:
    v2 blueprint had broken comparison: 'st_size 5000' (missing operator).
    Corrected to: path.stat().st_size < MIN_FILE_SIZE_BYTES

    Returns (is_valid, reason) tuple.
    """
    # Check extension
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return False, f"invalid extension: {path.suffix}"

    # Check file exists
    if not path.exists():
        return False, "file not found"

    # Check file size bounds — FIX #3: correct comparison operator
    file_size = path.stat().st_size
    if file_size < MIN_FILE_SIZE_BYTES:
        return False, f"file too small: {file_size} bytes < {MIN_FILE_SIZE_BYTES}"
    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"file too large: {file_size} bytes > {MAX_FILE_SIZE_BYTES}"

    # Check image dimensions and integrity using PIL
    try:
        # Import here so pipeline can be imported without PIL installed
        from PIL import Image  # type: ignore

        # First open: verify header integrity
        with Image.open(path) as img:
            img.verify()

        # Second open: check dimensions (verify() closes the file)
        with Image.open(path) as img:
            width, height = img.size
            if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
                return False, f"image too small: {width}x{height}px"

    except Exception as e:
        return False, f"corrupt or unreadable: {e}"

    return True, "ok"


# ─── SHA256 deduplication ─────────────────────────────────────────────────────

def compute_sha256(path: Path) -> str:
    """
    Compute SHA256 hash for deduplication.
    Blueprint Section 03: identical images must not appear in
    both train and val splits.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


# ─── Dataset discovery ────────────────────────────────────────────────────────

def discover_images(raw_dir: Path) -> dict[str, list[Path]]:
    """
    Discover images organised by category.
    Expects directory structure: raw_dir/{category}/{image_files}
    Returns dict mapping category name to list of image paths.
    """
    category_images: dict[str, list[Path]] = defaultdict(list)

    if not raw_dir.exists():
        print(f"  Raw directory not found: {raw_dir}")
        print("  Run data/scripts/download.py first.")
        return {}

    for category_dir in sorted(raw_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        images = [
            p for p in category_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
        if images:
            category_images[category] = sorted(images)

    return dict(category_images)


# ─── Stratified split ─────────────────────────────────────────────────────────

def stratified_split(
    images: list[Path],
    seen_hashes: set[str],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """
    Validate, deduplicate, and split images into train/val/test.

    Returns:
        train_records, val_records, test_records, stats
        Each record: {path, sha256, split}
    """
    import random
    random.seed(seed)

    valid_records = []
    stats: dict[str, int] = {
        "total": len(images),
        "valid": 0,
        "invalid_size": 0,
        "invalid_corrupt": 0,
        "duplicates": 0,
    }

    for path in images:
        is_valid, reason = validate_image(path)
        if not is_valid:
            if "too small" in reason or "too large" in reason:
                stats["invalid_size"] += 1
            else:
                stats["invalid_corrupt"] += 1
            continue

        sha256 = compute_sha256(path)
        if sha256 in seen_hashes:
            stats["duplicates"] += 1
            continue

        seen_hashes.add(sha256)
        valid_records.append({"path": str(path), "sha256": sha256})
        stats["valid"] += 1

    # Shuffle deterministically
    random.shuffle(valid_records)

    # Split
    n = len(valid_records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = [{**r, "split": "train"} for r in valid_records[:n_train]]
    val = [{**r, "split": "val"} for r in valid_records[n_train:n_train + n_val]]
    test = [{**r, "split": "test"} for r in valid_records[n_train + n_val:]]

    return train, val, test, stats


# ─── Dev sample generation ────────────────────────────────────────────────────

def generate_dev_sample(
    category_splits: dict[str, tuple[list, list, list]],
    samples_dir: Path,
    per_class: int = DEV_SAMPLE_SIZE_PER_CLASS,
) -> None:
    """
    Generate deterministic stratified dev sample.
    Blueprint Section 07: 2K images, 250/class for fast local iteration.
    Copies image files to data/samples/images/{split}/{category}/
    """
    print(f"\n  Generating dev sample ({per_class} images/class)...")
    samples_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        split_ratio = (
            TRAIN_RATIO if split == "train"
            else VAL_RATIO if split == "val"
            else TEST_RATIO
        )
        n_split = max(1, int(per_class * split_ratio))

        for category, (train_r, val_r, test_r) in category_splits.items():
            records = (
                train_r if split == "train"
                else val_r if split == "val"
                else test_r
            )
            sample_records = records[:n_split]
            sample_dir = samples_dir / split / category
            sample_dir.mkdir(parents=True, exist_ok=True)

            for record in sample_records:
                src = Path(record["path"])
                dst = sample_dir / src.name
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)

    print(f"  ✓ Dev sample written to {samples_dir}")


# ─── Manifest output ──────────────────────────────────────────────────────────

def save_split_manifest(
    records: list[dict],
    output_path: Path,
) -> None:
    """Save split records as CSV manifest for DataLoader consumption."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(raw_dir: Path, sample_only: bool = False) -> None:
    """
    Full image pipeline:
    1. Discover images by category
    2. Validate + deduplicate
    3. Stratified split
    4. Save manifests
    5. Generate dev sample
    6. Save transform spec
    """
    print("=" * 60)
    print("  E-CIP v3.0 — Image Pipeline")
    print("  Blueprint Section 03 + Section 07")
    print("=" * 60)

    # Discover
    print(f"\n  Discovering images in: {raw_dir}")
    category_images = discover_images(raw_dir)

    if not category_images:
        print("\n  No images found. Pipeline will run fully in Phase 2")
        print("  after data/scripts/download.py downloads Products-10K.")
        _save_transform_spec()
        print("\n  ✓ Transform spec saved — pipeline structure verified.")
        return

    print(f"  Found {len(category_images)} categories:")
    for cat, imgs in category_images.items():
        print(f"    {cat}: {len(imgs)} images")

    # Validate + split per category
    all_train, all_val, all_test = [], [], []
    seen_hashes: set[str] = set()
    category_splits: dict[str, tuple] = {}
    total_stats: dict[str, int] = defaultdict(int)

    print("\n  Validating and splitting...")
    for category, images in category_images.items():
        train, val, test, stats = stratified_split(images, seen_hashes)

        # Tag records with category
        for r in train:
            r["category"] = category
        for r in val:
            r["category"] = category
        for r in test:
            r["category"] = category

        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)
        category_splits[category] = (train, val, test)

        for k, v in stats.items():
            total_stats[k] += v

        print(f"    {category}: {stats['valid']} valid "
              f"({stats['duplicates']} dupes, "
              f"{stats['invalid_corrupt']} corrupt, "
              f"{stats['invalid_size']} wrong size) "
              f"→ {len(train)}/{len(val)}/{len(test)} train/val/test")

    # Summary
    print("\n  Pipeline summary:")
    print(f"    Total images   : {total_stats['total']:,}")
    print(f"    Valid          : {total_stats['valid']:,}")
    print(f"    Duplicates     : {total_stats['duplicates']:,}")
    print(f"    Invalid size   : {total_stats['invalid_size']:,}")
    print(f"    Corrupt        : {total_stats['invalid_corrupt']:,}")
    print(f"    Train split    : {len(all_train):,}")
    print(f"    Val split      : {len(all_val):,}")
    print(f"    Test split     : {len(all_test):,}")

    if not sample_only:
        # Save manifests
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        save_split_manifest(all_train, PROCESSED_DIR / "train_manifest.csv")
        save_split_manifest(all_val,   PROCESSED_DIR / "val_manifest.csv")
        save_split_manifest(all_test,  PROCESSED_DIR / "test_manifest.csv")
        print(f"\n  ✓ Manifests saved to {PROCESSED_DIR}")

    # Dev sample
    generate_dev_sample(category_splits, SAMPLES_DIR)

    # Transform spec
    _save_transform_spec()

    print("\n" + "=" * 60)
    print("  Image pipeline complete.")
    print("  Next: python data/pipelines/text_pipeline.py")
    print("=" * 60 + "\n")


def _save_transform_spec() -> None:
    """Save transform specification to feature store for reproducibility."""
    FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    spec_path = FEATURE_STORE_DIR / "transform_spec.json"
    spec_path.write_text(json.dumps(TRANSFORM_SPEC, indent=2))
    print(f"  ✓ Transform spec saved: {spec_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E-CIP v3.0 — Image Data Pipeline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIR,
        help=f"Raw images directory (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Generate dev sample only, skip full manifest",
    )
    args = parser.parse_args()

    run_pipeline(raw_dir=args.input, sample_only=args.sample_only)


if __name__ == "__main__":
    main()
