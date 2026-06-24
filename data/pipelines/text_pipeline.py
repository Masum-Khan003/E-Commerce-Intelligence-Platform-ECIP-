# data/pipelines/text_pipeline.py
# E-CIP v3.0 — Text Data Pipeline
# Blueprint Section 04 (Fix #6, Fix #12) + Section 07
#
# Responsibilities:
#   - Load Amazon Reviews 2023 (Electronics + Fashion)
#   - Clean and normalise review text
#   - Apply head+tail truncation strategy (Fix #12)
#   - Save tokenizer artifact to feature store (Fix #6)
#   - Generate stratified train/val/test split manifests
#   - Generate dev sample (10K reviews, stratified by star rating)
#
# Fix #6:  Tokenizer saved to artifacts/ — NEVER re-initialised from Hub
#          at inference time. Load from artifact path only.
# Fix #12: Head+tail truncation — first 128 + last 382 tokens.
#          Simple tail truncation loses review conclusions.
#
# Usage:
#   python data/pipelines/text_pipeline.py
#   python data/pipelines/text_pipeline.py --sample-only
#   python data/pipelines/text_pipeline.py --max-reviews 50000

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path
from typing import Any

# ─── Paths ────────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw/amazon_reviews")
PROCESSED_DIR = Path("data/processed/reviews")
SAMPLES_DIR = Path("data/samples/reviews")
TOKENIZER_ARTIFACT_DIR = Path("data/feature_store/artifacts/tokenizer_v1")

# ─── Constants ────────────────────────────────────────────────────────────────

# Blueprint Section 07: dev sample — 10K reviews, 2K per star rating
DEV_SAMPLE_SIZE = 10_000
DEV_SAMPLE_PER_RATING = 2_000

# Train/val/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Head+tail truncation parameters (Blueprint Section 04 — Fix #12)
MAX_TOKENS = 512
HEAD_TOKENS = 128   # first 128 tokens — product/brand context
TAIL_TOKENS = 382   # last 382 tokens — conclusion and rating summary

# Minimum review length to be useful
MIN_REVIEW_CHARS = 10
MAX_REVIEW_CHARS = 50_000

# Star rating → sentiment label mapping
RATING_TO_LABEL: dict[int, str] = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

# Source files to process
REVIEW_FILES = [
    "Electronics.jsonl.gz",
    "Amazon_Fashion.jsonl.gz",
]


# ─── Text cleaning ────────────────────────────────────────────────────────────

def clean_review_text(text: str) -> str:
    """
    Normalise review text for tokenisation.
    Preserves sentiment signal — does not strip negations or punctuation
    that carry meaning (e.g. "not good", "terrible!!!").
    """
    if not isinstance(text, str):
        return ""

    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove HTML tags (common in older Amazon reviews)
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalise repeated punctuation (!!!!! → !)
    text = re.sub(r"([!?.]){3,}", r"\1\1", text)

    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    # Final whitespace normalisation
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_valid_review(text: str, rating: int) -> tuple[bool, str]:
    """
    Validate a single review record.
    Returns (is_valid, reason).
    """
    if not text or len(text) < MIN_REVIEW_CHARS:
        return False, f"too short: {len(text)} chars"
    if len(text) > MAX_REVIEW_CHARS:
        return False, f"too long: {len(text)} chars"
    if rating not in range(1, 6):
        return False, f"invalid rating: {rating}"
    return True, "ok"


# ─── Head+tail truncation ─────────────────────────────────────────────────────

def head_tail_truncate(
    text: str,
    tokenizer: Any,
    max_length: int = MAX_TOKENS,
    head_tokens: int = HEAD_TOKENS,
    tail_tokens: int = TAIL_TOKENS,
) -> dict[str, Any]:
    """
    Blueprint Section 04 — Critical Fix #12.

    Head+tail truncation for reviews longer than max_length tokens.
    Keeps first {head_tokens} tokens (product/brand context) and last
    {tail_tokens} tokens (conclusion and rating summary — strongest
    sentiment signal).

    Simple tail-truncation loses the review's conclusion where the
    customer's final verdict is usually most explicit.

    Args:
        text: Raw review text (cleaned)
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum token sequence length (default: 512)
        head_tokens: Tokens to keep from start (default: 128)
        tail_tokens: Tokens to keep from end (default: 382)

    Returns:
        HuggingFace tokenizer output dict (input_ids, attention_mask)
    """
    # Encode without special tokens first to check length
    tokens = tokenizer(text, add_special_tokens=False)
    input_ids = tokens["input_ids"]

    # If fits within budget, standard tokenisation
    if len(input_ids) <= max_length - 2:  # -2 for [CLS] and [SEP]
        result: dict[str, Any] = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        return result

    # Head+tail strategy
    head_ids = input_ids[:head_tokens]
    tail_ids = input_ids[-tail_tokens:]

    truncated_ids = (
        [tokenizer.cls_token_id]
        + head_ids
        + tail_ids
        + [tokenizer.sep_token_id]
    )

    attention_mask = [1] * len(truncated_ids)

    return {
        "input_ids": truncated_ids,
        "attention_mask": attention_mask,
        "truncation_applied": True,
    }


# ─── Tokenizer artifact ───────────────────────────────────────────────────────

def save_tokenizer_artifact(tokenizer: Any, artifact_dir: Path) -> Path:
    """
    Blueprint Section 04 — Critical Fix #6.

    Save tokenizer alongside model weights to a versioned artifact path.
    At inference time, load from this path ONLY.
    NEVER call AutoTokenizer.from_pretrained(hub_name) at inference.

    Re-downloading from Hub risks version drift if the upstream
    tokenizer is updated, silently corrupting predictions.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(artifact_dir))  # type: ignore[attr-defined]

    # Write a manifest so the loader knows what's here
    manifest = {
        "tokenizer_name": "distilbert-base-uncased",
        "artifact_version": "v1",
        "max_length": MAX_TOKENS,
        "head_tokens": HEAD_TOKENS,
        "tail_tokens": TAIL_TOKENS,
        "truncation_strategy": "head_tail",
        "saved_by": "data/pipelines/text_pipeline.py",
        "load_instruction": (
            "Load via: AutoTokenizer.from_pretrained(str(artifact_dir)) "
            "NEVER via: AutoTokenizer.from_pretrained('distilbert-base-uncased')"
        ),
    }
    manifest_path = artifact_dir / "ecip_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  ✓ Tokenizer artifact saved: {artifact_dir}")
    print(f"  ✓ Manifest written        : {manifest_path}")
    return artifact_dir


# ─── JSONL loading ────────────────────────────────────────────────────────────

def load_reviews_jsonl(file_path: Path, max_reviews: int | None = None) -> list[dict]:
    """
    Load Amazon Reviews 2023 from .jsonl.gz format.
    Each line is a JSON object with fields:
        rating, title, text, asin, parent_asin, user_id,
        timestamp, helpful_vote, verified_purchase
    """
    records: list[dict] = []
    open_fn = gzip.open if file_path.suffix == ".gz" else open

    print(f"  Loading: {file_path.name}")
    with open_fn(file_path, "rt", encoding="utf-8") as f:  # type: ignore
        for i, line in enumerate(f):
            if max_reviews and len(records) >= max_reviews:
                break
            try:
                obj = json.loads(line.strip())
                text = clean_review_text(obj.get("text", "") or "")
                rating = int(float(obj.get("rating", 0)))
                is_valid, _ = is_valid_review(text, rating)
                if not is_valid:
                    continue
                records.append({
                    "review_id": f"{obj.get('asin', 'unk')}_{i}",
                    "text": text,
                    "rating": rating,
                    "label": RATING_TO_LABEL.get(rating, "neutral"),
                    "source": file_path.stem,
                    "verified_purchase": obj.get("verified_purchase", False),
                    "char_length": len(text),
                })
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

    print(f"    Loaded {len(records):,} valid reviews from {file_path.name}")
    return records


# ─── Stratified split ─────────────────────────────────────────────────────────

def stratified_split_by_label(
    records: list[dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified split preserving label distribution across splits.
    Blueprint Section 07: split is stratified by star rating.
    """
    import random
    random.seed(seed)

    # Group by label
    by_label: dict[str, list[dict]] = {}
    for record in records:
        label = record["label"]
        by_label.setdefault(label, []).append(record)

    train, val, test = [], [], []
    for label, label_records in by_label.items():
        random.shuffle(label_records)
        n = len(label_records)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for r in label_records[:n_train]:
            train.append({**r, "split": "train"})
        for r in label_records[n_train:n_train + n_val]:
            val.append({**r, "split": "val"})
        for r in label_records[n_train + n_val:]:
            test.append({**r, "split": "test"})

    # Shuffle within splits
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# ─── Dev sample ───────────────────────────────────────────────────────────────

def generate_dev_sample(
    records: list[dict],
    samples_dir: Path,
    per_rating: int = DEV_SAMPLE_PER_RATING,
    seed: int = 42,
) -> list[dict]:
    """
    Blueprint Section 07: 10K review dev sample, 2K per star rating.
    Deterministic — same seed always produces same sample.
    """
    import random
    random.seed(seed)

    by_rating: dict[int, list[dict]] = {}
    for r in records:
        by_rating.setdefault(r["rating"], []).append(r)

    sample = []
    for rating in sorted(by_rating.keys()):
        rating_records = by_rating[rating]
        random.shuffle(rating_records)
        sample.extend(rating_records[:per_rating])

    random.shuffle(sample)

    # Save sample
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_path = samples_dir / "reviews_dev_10k.csv"
    if sample:
        _save_csv(sample, sample_path)
        print(f"  ✓ Dev sample saved: {sample_path} ({len(sample):,} reviews)")

    return sample


# ─── CSV output ───────────────────────────────────────────────────────────────

def _save_csv(records: list[dict], path: Path) -> None:
    """Save records to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    raw_dir: Path,
    max_reviews: int | None = None,
    sample_only: bool = False,
    save_tokenizer: bool = True,
) -> None:
    """
    Full text pipeline:
    1. Load reviews from JSONL.GZ files
    2. Clean and validate
    3. Stratified split by label
    4. Save manifests to processed/reviews/
    5. Generate dev sample
    6. Save tokenizer artifact (Fix #6)
    """
    print("=" * 60)
    print("  E-CIP v3.0 — Text Pipeline")
    print("  Blueprint Section 04 — Fix #6 (tokenizer) + Fix #12 (truncation)")
    print("=" * 60)

    # Check for raw data
    available_files = [
        raw_dir / fname for fname in REVIEW_FILES
        if (raw_dir / fname).exists()
    ]

    if not available_files:
        print(f"\n  No review files found in: {raw_dir}")
        print("  Expected files:")
        for fname in REVIEW_FILES:
            print(f"    {raw_dir / fname}")
        print("\n  Run data/scripts/download.py --module 2 first.")
        print("\n  Saving tokenizer artifact stub for pipeline verification...")
        _save_tokenizer_stub()
        print("\n  ✓ Pipeline structure verified — runs fully in Phase 3.")
        return

    # Load all available review files
    all_records: list[dict] = []
    for file_path in available_files:
        records = load_reviews_jsonl(file_path, max_reviews=max_reviews)
        all_records.extend(records)

    print(f"\n  Total valid reviews loaded: {len(all_records):,}")

    # Label distribution
    label_counts: dict[str, int] = {}
    for r in all_records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    print("  Label distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / len(all_records) * 100
        print(f"    {label}: {count:,} ({pct:.1f}%)")

    # Dev sample — always generated
    print("\n  Generating dev sample...")
    generate_dev_sample(all_records, SAMPLES_DIR)

    if not sample_only:
        # Stratified split
        print("\n  Splitting into train/val/test...")
        train, val, test = stratified_split_by_label(all_records)
        print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

        # Save manifests
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        _save_csv(train, PROCESSED_DIR / "train_reviews.csv")
        _save_csv(val,   PROCESSED_DIR / "val_reviews.csv")
        _save_csv(test,  PROCESSED_DIR / "test_reviews.csv")
        print(f"  ✓ Review manifests saved to {PROCESSED_DIR}")

    # Save tokenizer artifact (Fix #6)
    if save_tokenizer:
        print("\n  Saving tokenizer artifact (Fix #6)...")
        _save_tokenizer_to_artifact()

    print("\n" + "=" * 60)
    print("  Text pipeline complete.")
    print("  Next: python data/pipelines/tabular_pipeline.py")
    print("=" * 60 + "\n")


def _save_tokenizer_to_artifact() -> None:
    """
    Download and save the DistilBERT tokenizer to the artifact store.
    Fix #6: This is the ONLY place the tokenizer is downloaded from Hub.
    All subsequent loads use the saved artifact path.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
        print("  Downloading distilbert-base-uncased tokenizer from Hub...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        save_tokenizer_artifact(tokenizer, TOKENIZER_ARTIFACT_DIR)
    except ImportError:
        print("  transformers not installed — saving stub manifest only.")
        print("  Install [train] extras in Colab/Kaggle for full tokenizer save.")
        _save_tokenizer_stub()


def _save_tokenizer_stub() -> None:
    """
    Save a stub manifest when transformers is not installed.
    Ensures the artifact directory exists and is documented.
    """
    TOKENIZER_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stub = {
        "tokenizer_name": "distilbert-base-uncased",
        "artifact_version": "v1",
        "status": "stub — run with [train] extras to download full tokenizer",
        "max_length": MAX_TOKENS,
        "head_tokens": HEAD_TOKENS,
        "tail_tokens": TAIL_TOKENS,
        "truncation_strategy": "head_tail",
        "fix_reference": "Blueprint Section 04 — Fix #6",
        "load_instruction": (
            "Load via: AutoTokenizer.from_pretrained(str(artifact_dir)) "
            "NEVER via: AutoTokenizer.from_pretrained('distilbert-base-uncased')"
        ),
    }
    stub_path = TOKENIZER_ARTIFACT_DIR / "ecip_manifest.json"
    stub_path.write_text(json.dumps(stub, indent=2))
    print(f"  ✓ Tokenizer stub manifest saved: {stub_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E-CIP v3.0 — Text Data Pipeline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIR,
        help=f"Raw reviews directory (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=None,
        help="Max reviews to load per file (default: all)",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Generate dev sample only, skip full manifests",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip tokenizer artifact saving",
    )
    args = parser.parse_args()

    run_pipeline(
        raw_dir=args.input,
        max_reviews=args.max_reviews,
        sample_only=args.sample_only,
        save_tokenizer=not args.no_tokenizer,
    )


if __name__ == "__main__":
    main()
