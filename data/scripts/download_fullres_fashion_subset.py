# data/scripts/download_fullres_fashion_subset.py
# E-CIP v3.0 — Selective full-resolution download for Module 1
#
# paramaggarwal/fashion-product-images-dataset (24.7GB) is too large to move
# to a free-tier Colab whole, but paramaggarwal/fashion-product-images-small
# (already downloaded to data/raw/fashion_product_images_small/) uses the
# same product IDs at 60x80 thumbnail resolution — too small for
# EfficientNet-B3 (image_pipeline.py's MIN_DIMENSION_PX=64 gate, and 300x300
# training input). Both datasets share IDs, so this script reads the
# existing styles.csv label file, picks a capped sample per subCategory
# class, and pulls just those IDs at full resolution via per-file Kaggle
# downloads — keeping total size small while getting real image quality.
#
# Usage:
#   python data/scripts/download_fullres_fashion_subset.py
#   python data/scripts/download_fullres_fashion_subset.py --per-class 150

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path

STYLES_CSV = Path("data/raw/fashion_product_images_small/styles.csv")
OUTPUT_DIR = Path("data/raw/fashion_product_images_fullres/by_category")
DATASET = "paramaggarwal/fashion-product-images-dataset"
DATASET_IMAGE_PREFIX = "fashion-dataset/fashion-dataset/images"

MIN_IMAGES_PER_CATEGORY = 100
DEFAULT_PER_CLASS = 80


def select_ids(per_class: int) -> dict[str, list[str]]:
    """Group product IDs by subCategory, keeping classes with enough samples
    and capping the count per class."""
    by_category: dict[str, list[str]] = defaultdict(list)
    with open(STYLES_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            by_category[row["subCategory"]].append(row["id"])

    kept = {
        category: ids[:per_class]
        for category, ids in by_category.items()
        if len(ids) >= MIN_IMAGES_PER_CATEGORY
    }
    return kept


def download_one(product_id: str, dest_dir: Path) -> bool:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f"{product_id}.jpg"
    if dest_file.exists():
        return True

    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET,
            "-f", f"{DATASET_IMAGE_PREFIX}/{product_id}.jpg",
            "-p", str(dest_dir),
            "--force",
        ],
        capture_output=True,
        timeout=60,
    )
    return result.returncode == 0 and dest_file.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    args = parser.parse_args()

    if not STYLES_CSV.exists():
        print(f"  {STYLES_CSV} not found — run the Small dataset download first.")
        return

    kept = select_ids(args.per_class)
    total = sum(len(ids) for ids in kept.values())
    print(f"  {len(kept)} categories, up to {args.per_class} images/class, "
          f"{total} images to fetch")

    done = 0
    failed: list[str] = []
    for category, ids in sorted(kept.items()):
        dest_dir = OUTPUT_DIR / category
        for product_id in ids:
            ok = download_one(product_id, dest_dir)
            done += 1
            if not ok:
                failed.append(f"{category}/{product_id}")
            if done % 100 == 0:
                print(f"    {done}/{total} fetched ({len(failed)} failed)")

    print(f"\n  ✓ Done: {done - len(failed)}/{total} images downloaded")
    if failed:
        print(f"  ✗ {len(failed)} failed: {failed[:10]}{'...' if len(failed) > 10 else ''}")


if __name__ == "__main__":
    main()
