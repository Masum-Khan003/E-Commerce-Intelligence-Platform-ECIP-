# data/scripts/download.py
# E-CIP v3.0 — Dataset Download Script
# Blueprint Section 07 + Section 20
#
# Called by: dvc repro download_datasets
# Downloads all datasets to data/raw/ with MD5 verification.
#
# Usage:
#   python data/scripts/download.py              # all datasets
#   python data/scripts/download.py --module 1   # Module 1 only
#   python data/scripts/download.py --dry-run    # show what would download

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ─── Dataset registry ─────────────────────────────────────────────────────────

@dataclass
class Dataset:
    name: str
    method: Literal["kaggle", "wget", "git_clone"]
    target: str
    dest: Path
    module: int
    required: bool = True
    post_extract: list[str] = field(default_factory=list)


RAW = Path("data/raw")

DATASETS: list[Dataset] = [
    # ── Module 1 — Product Intelligence ──────────────────────────────────────
    # Products-10K (18.3GB) was the original blueprint choice but is too large
    # to move to a free-tier Colab (15GB Drive quota). Its full-resolution
    # replacement (paramaggarwal/fashion-product-images-dataset, 24.7GB) has
    # the same problem. Fix: this "Small" variant shares the same product IDs
    # and subCategory/masterCategory/articleType labels but ships 60x80px
    # thumbnails (too small for image_pipeline.py's MIN_DIMENSION_PX=64 gate
    # and EfficientNet-B3's 300x300 input) — it's downloaded only to read
    # styles.csv. Actual training images are then pulled at full resolution,
    # per-ID, capped per class, via
    # data/scripts/download_fullres_fashion_subset.py (run after this).
    Dataset(
        name="Fashion Product Images (Small) — labels only, see script header",
        method="kaggle",
        target="paramaggarwal/fashion-product-images-small",
        dest=RAW / "fashion_product_images_small",
        module=1,
        required=True,
    ),
    Dataset(
        name="Products-10K (original, too large for free-tier Colab)",
        method="kaggle",
        target="hirune924/products10k",
        dest=RAW / "products10k",
        module=1,
        required=False,
    ),
    Dataset(
        name="FEIDEGGER (Zalando)",
        method="git_clone",
        target="https://github.com/zalandoresearch/feidegger.git",
        dest=RAW / "feidegger",
        module=1,
        required=False,  # Single-category (dresses), no class labels — text/image research only
    ),

    # ── Module 2 — Sentiment Intelligence ────────────────────────────────────
    Dataset(
        name="Amazon Reviews 2023 — Electronics",
        method="wget",
        target=(
            "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
            "/raw/review_categories/Electronics.jsonl.gz"
        ),
        dest=RAW / "amazon_reviews",
        module=2,
        required=True,
    ),
    Dataset(
        name="Amazon Reviews 2023 — Fashion",
        method="wget",
        target=(
            "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
            "/raw/review_categories/Amazon_Fashion.jsonl.gz"
        ),
        dest=RAW / "amazon_reviews",
        module=2,
        required=True,
    ),

    # ── Module 3 — Retention Intelligence ────────────────────────────────────
    Dataset(
        name="UCI Online Retail II",
        method="wget",
        target=(
            "https://archive.ics.uci.edu/static/public/502"
            "/online+retail+ii.zip"
        ),
        dest=RAW / "online_retail2",
        module=3,
        required=True,
        post_extract=["unzip -o online+retail+ii.zip -d ."],
    ),
    Dataset(
        name="E-Commerce Behavior (backup)",
        method="kaggle",
        target="mkechinov/ecommerce-behavior-data-from-multi-category-store",
        dest=RAW / "ecommerce_behavior",
        module=3,
        required=False,
    ),
]


# ─── Download methods ─────────────────────────────────────────────────────────

def download_kaggle(dataset: Dataset) -> bool:
    """Download a Kaggle dataset using the Kaggle CLI."""
    dataset.dest.mkdir(parents=True, exist_ok=True)
    print(f"    kaggle datasets download -d {dataset.target} -p {dataset.dest} --unzip")
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", dataset.target,
            "-p", str(dataset.dest),
            "--unzip",
        ],
        capture_output=False,
        timeout=3600,
    )
    return result.returncode == 0


def download_wget(dataset: Dataset) -> bool:
    """Download a file via URL to destination directory."""
    dataset.dest.mkdir(parents=True, exist_ok=True)
    filename = dataset.target.split("/")[-1]
    dest_file = dataset.dest / filename

    print(f"    wget {dataset.target}")
    print(f"      → {dest_file}")

    try:
        req = urllib.request.Request(dataset.target)
        req.add_header("User-Agent", "Mozilla/5.0 (ecip-download/1.0)")
        with urllib.request.urlopen(req, timeout=3600) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            with open(dest_file, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb = downloaded / 1024 / 1024
                        print(f"\r      {mb:.1f} MB / {total/1024/1024:.1f} MB "
                              f"({pct:.0f}%)", end="", flush=True)
            print()
    except Exception as e:
        print(f"\n    ✗ Download failed: {e}")
        return False

    # Run post-extraction commands (e.g. unzip)
    if dataset.post_extract:
        for cmd in dataset.post_extract:
            print(f"    Extracting: {cmd}")
            result = subprocess.run(
                cmd, shell=True, cwd=dataset.dest, capture_output=False
            )
            if result.returncode != 0:
                print("    ✗ Extraction failed")
                return False

    return dest_file.exists()


def download_git_clone(dataset: Dataset) -> bool:
    """Clone a git repository to destination."""
    if dataset.dest.exists() and any(dataset.dest.iterdir()):
        print(f"    Already cloned at {dataset.dest} — skipping")
        return True
    dataset.dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    git clone {dataset.target} {dataset.dest}")
    result = subprocess.run(
        ["git", "clone", "--depth=1", dataset.target, str(dataset.dest)],
        capture_output=False,
        timeout=300,
    )
    return result.returncode == 0


# ─── MD5 fingerprint ──────────────────────────────────────────────────────────

def compute_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute MD5 hash of a file for DVC fingerprinting."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def log_fingerprints(dest: Path) -> None:
    """Log MD5 fingerprints of downloaded files for DVC verification."""
    fingerprint_file = dest / "fingerprints.txt"
    lines = []
    for f in sorted(dest.glob("*")):
        if f.is_file() and f.name != "fingerprints.txt":
            md5 = compute_md5(f)
            lines.append(f"{md5}  {f.name}")
            print(f"    MD5: {md5}  {f.name}")
    fingerprint_file.write_text("\n".join(lines) + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E-CIP v3.0 — Dataset Download Script"
    )
    parser.add_argument(
        "--module",
        type=int,
        choices=[1, 2, 3],
        help="Download only datasets for a specific module",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading",
    )
    parser.add_argument(
        "--required-only",
        action="store_true",
        default=True,
        help="Skip optional backup datasets (default: True)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  E-CIP v3.0 — Dataset Download")
    print("  Blueprint Section 07 + Section 20")
    print("=" * 60)

    # Filter datasets
    targets = DATASETS
    if args.module:
        targets = [d for d in targets if d.module == args.module]
    if args.required_only:
        targets = [d for d in targets if d.required]

    print(f"\n  Datasets to download: {len(targets)}")
    for d in targets:
        status = "required" if d.required else "optional"
        print(f"    [{status}] Module {d.module}: {d.name}")

    if args.dry_run:
        print("\n  Dry run complete — no files downloaded.")
        return

    # Download loop
    results: dict[str, bool] = {}
    for dataset in targets:
        print(f"\n  Downloading: {dataset.name}")
        print(f"  Method     : {dataset.method}")
        print(f"  Destination: {dataset.dest}")

        if dataset.method == "kaggle":
            ok = download_kaggle(dataset)
        elif dataset.method == "wget":
            ok = download_wget(dataset)
        elif dataset.method == "git_clone":
            ok = download_git_clone(dataset)
        else:
            print(f"  ✗ Unknown method: {dataset.method}")
            ok = False

        if ok:
            print(f"  ✓ {dataset.name} — download complete")
            log_fingerprints(dataset.dest)
        else:
            print(f"  ✗ {dataset.name} — download FAILED")

        results[dataset.name] = ok

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results.values())
    total = len(results)
    print(f"  Result: {passed}/{total} datasets downloaded")

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print("\n  Failed:")
        for name in failed:
            print(f"    ✗ {name}")
        sys.exit(1)

    print("\n  All datasets ready. Next: dvc repro image_pipeline")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
