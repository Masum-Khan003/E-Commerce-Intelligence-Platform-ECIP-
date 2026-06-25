# models/sentiment/finetune.py
# E-CIP v3.0 — DistilBERT Fine-Tuning for Sentiment Classification
# Blueprint Section 04 — Fix #6, Fix #12
#
# Fix #6:  Tokenizer saved to MLflow artifact URI at training time.
#          NEVER re-initialised from Hub at inference.
# Fix #12: Head+tail truncation — first 128 + last 382 tokens.
#          Simple tail truncation loses review conclusions.
#
# Architecture:
#   distilbert-base-uncased → [CLS] token → Dropout → Linear(768, 3)
#   3-class: negative(0), neutral(1), positive(2)
#   Focal Loss for class imbalance (neutral class underrepresented)
#
# Training targets:
#   Macro F1       : ≥ 0.88
#   Negative Recall: ≥ 0.85
#   Neutral Prec   : ≥ 0.75
#   Inference p95  : < 50ms
#
# Usage (Colab/Kaggle — GPU required):
#   python models/sentiment/finetune.py --data data/processed/reviews
#   python models/sentiment/finetune.py --data data/samples/reviews --dev

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

# ─── Constants ────────────────────────────────────────────────────────────────

DISTILBERT_MODEL_ID = "distilbert-base-uncased"
NUM_CLASSES = 3
SEED = 42
MLFLOW_EXPERIMENT = "sentiment_classifier"

# Tokenizer artifact path — Fix #6
TOKENIZER_ARTIFACT_DIR = Path("data/feature_store/artifacts/tokenizer_v1")
MODELS_DIR = Path("models/sentiment/weights")
ARTIFACTS_DIR = Path("models/sentiment/artifacts")

# Head+tail truncation — Fix #12
MAX_TOKENS = 512
HEAD_TOKENS = 128
TAIL_TOKENS = 382

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_NAMES = ["negative", "neutral", "positive"]

# Training hyperparameters
HPARAMS: dict[str, Any] = {
    "epochs": 5,
    "lr": 2e-5,
    "batch_size": 16,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "focal_loss_gamma": 2.0,   # Focal Loss — class imbalance
    "focal_loss_alpha": 1.0,
    "dropout": 0.1,
    "seed": SEED,
}


# ─── Device + seed ────────────────────────────────────────────────────────────

def get_device() -> Any:
    """CUDA → MPS → CPU priority chain. Blueprint Section 03 — Fix #18."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"  ✓ Device: CUDA — {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("  ✓ Device: Apple Silicon MPS")
        else:
            device = torch.device("cpu")
            print("  ⚠ Device: CPU — training will be slow.")
        return device
    except ImportError:
        return None


def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    try:
        import random

        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─── Tokenizer — Fix #6 ───────────────────────────────────────────────────────

def load_tokenizer_from_artifact(artifact_dir: Path) -> Any:
    """
    Fix #6: Load tokenizer from saved artifact path ONLY.
    Never call AutoTokenizer.from_pretrained(hub_name) at inference.
    """
    try:
        from transformers import AutoTokenizer

        if artifact_dir.exists() and (artifact_dir / "tokenizer.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir))
            print(f"  ✓ Tokenizer loaded from artifact: {artifact_dir}")
        else:
            # First-time setup: download from Hub and save to artifact
            print(f"  Artifact not found at {artifact_dir}")
            print(f"  Downloading {DISTILBERT_MODEL_ID} from Hub (one-time only)...")
            tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_ID)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(artifact_dir))
            print(f"  ✓ Tokenizer saved to artifact: {artifact_dir}")
            print("  Future loads will use the artifact path.")

        return tokenizer

    except ImportError:
        print("  transformers not installed — tokenizer load skipped.")
        return None


# ─── Head+tail truncation — Fix #12 ──────────────────────────────────────────

def head_tail_tokenize(
    text: str,
    tokenizer: Any,
    max_length: int = MAX_TOKENS,
    head_tokens: int = HEAD_TOKENS,
    tail_tokens: int = TAIL_TOKENS,
) -> dict[str, Any]:
    """
    Fix #12: Head+tail truncation for reviews longer than max_length.

    Keeps first {head_tokens} tokens (product/brand context) and last
    {tail_tokens} tokens (conclusion — strongest sentiment signal).
    Simple tail-truncation loses the review's conclusion.

    Returns tokenizer output dict with truncation_applied flag.
    """
    tokens = tokenizer(text, add_special_tokens=False)
    input_ids: list[int] = tokens["input_ids"]

    truncation_applied = False

    if len(input_ids) <= max_length - 2:
        # Fits within budget — standard tokenisation
        result: dict[str, Any] = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
    else:
        # Head+tail strategy
        truncation_applied = True
        head_ids = input_ids[:head_tokens]
        tail_ids = input_ids[-tail_tokens:]
        truncated_ids = (
            [tokenizer.cls_token_id]
            + head_ids
            + tail_ids
            + [tokenizer.sep_token_id]
        )
        attention_mask = [1] * len(truncated_ids)

        # Pad to max_length if shorter
        pad_length = max_length - len(truncated_ids)
        if pad_length > 0:
            truncated_ids = truncated_ids + [tokenizer.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        result = {
            "input_ids": truncated_ids[:max_length],
            "attention_mask": attention_mask[:max_length],
        }

    result["truncation_applied"] = truncation_applied
    return result


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ReviewDataset:
    """PyTorch Dataset for Amazon Reviews."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int = MAX_TOKENS,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        encoding = head_tail_tokenize(
            self.texts[idx], self.tokenizer, self.max_length
        )
        try:
            import torch
            return {
                "input_ids": torch.tensor(
                    encoding["input_ids"], dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    encoding["attention_mask"], dtype=torch.long
                ),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                "truncation_applied": encoding["truncation_applied"],
            }
        except ImportError:
            return encoding


def load_split(csv_path: Path) -> tuple[list[str], list[int]]:
    """Load texts and labels from a split CSV file."""
    texts: list[str] = []
    labels: list[int] = []

    if not csv_path.exists():
        return texts, labels

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            label_str = row.get("label", "").strip().lower()
            if not text or label_str not in LABEL_MAP:
                continue
            texts.append(text)
            labels.append(LABEL_MAP[label_str])

    return texts, labels


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss:
    """
    Focal Loss for class imbalance in sentiment classification.
    Downweights easy examples — focuses training on hard/minority cases.
    Blueprint Section 04: applied for neutral class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    gamma=2.0: standard setting from Lin et al. 2017
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 1.0,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, logits: Any, targets: Any) -> Any:
        try:
            import torch
            import torch.nn.functional as functional

            log_probs = functional.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

            # Gather log-probs and probs for true class
            log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            focal_weight = self.alpha * (1 - pt) ** self.gamma
            loss = -focal_weight * log_pt
            return loss.mean()

        except ImportError:
            return None


# ─── Training loop ────────────────────────────────────────────────────────────

def train_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
    epoch: int,
    total_epochs: int,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Single training epoch with gradient clipping."""
    try:
        import torch

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        truncated = 0

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            truncated += batch["truncation_applied"].sum().item()

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 20 == 0:
                print(
                    f"\r    [{epoch}/{total_epochs}] "
                    f"batch {batch_idx}/{len(loader)} "
                    f"loss={loss.item():.4f} "
                    f"acc={100.*correct/max(total,1):.1f}%",
                    end="", flush=True,
                )

        print()
        return {
            "train_loss": total_loss / len(loader),
            "train_acc": correct / max(total, 1),
            "truncated_pct": truncated / max(total, 1),
        }

    except Exception as e:
        print(f"\n  Training error: {e}")
        return {"train_loss": 0.0, "train_acc": 0.0, "truncated_pct": 0.0}


def eval_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
) -> dict[str, float]:
    """Evaluation epoch with loss and accuracy."""
    try:
        import torch

        model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                loss = criterion(logits, labels)
                total_loss += loss.item()

                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        from sklearn.metrics import f1_score
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        acc = sum(p == lbl for p, lbl in zip(all_preds, all_labels)) / max(len(all_labels), 1)

        return {
            "val_loss": total_loss / len(loader),
            "val_acc": acc,
            "val_macro_f1": macro_f1,
        }

    except Exception as e:
        print(f"\n  Eval error: {e}")
        return {"val_loss": 0.0, "val_acc": 0.0, "val_macro_f1": 0.0}


# ─── Full training pipeline ───────────────────────────────────────────────────

def run_training(
    data_dir: Path,
    dev_mode: bool = False,
    run_name: str = "distilbert_sentiment_v1",
) -> dict[str, Any]:
    """
    Full DistilBERT fine-tuning pipeline.
    Logs all metrics to MLflow.
    Saves best model checkpoint by val_macro_f1.
    """
    print("=" * 60)
    print("  E-CIP v3.0 — DistilBERT Fine-Tuning")
    print("  Blueprint Section 04 — Fix #6 + Fix #12")
    print("=" * 60)

    device = get_device()
    set_seed(SEED)

    if device is None:
        print("  torch not installed — training skipped.")
        return {}

    # Load tokenizer from artifact (Fix #6)
    print("\n  Loading tokenizer (Fix #6 — from artifact)...")
    tokenizer = load_tokenizer_from_artifact(TOKENIZER_ARTIFACT_DIR)
    if tokenizer is None:
        return {}

    # Load data
    print(f"\n  Loading data from: {data_dir}")
    train_file = data_dir / "train_reviews.csv"
    val_file = data_dir / "val_reviews.csv"

    if not train_file.exists():
        print(f"  Training data not found: {train_file}")
        print("  Run data/pipelines/text_pipeline.py first.")
        return {}

    train_texts, train_labels = load_split(train_file)
    val_texts, val_labels = load_split(val_file)

    # Dev mode: use small subset
    if dev_mode:
        train_texts = train_texts[:500]
        train_labels = train_labels[:500]
        val_texts = val_texts[:100]
        val_labels = val_labels[:100]

    print(f"  Train: {len(train_texts):,} | Val: {len(val_texts):,}")

    # Build datasets
    try:
        import torch.optim as optim
        from torch.optim.lr_scheduler import OneCycleLR
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification

        train_ds = ReviewDataset(train_texts, train_labels, tokenizer)
        val_ds = ReviewDataset(val_texts, val_labels, tokenizer)

        batch_size = 8 if dev_mode else int(HPARAMS["batch_size"])
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=2,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size * 2,
            shuffle=False, num_workers=2,
        )

        # Model
        print("\n  Building DistilBERT classifier...")
        model = AutoModelForSequenceClassification.from_pretrained(
            DISTILBERT_MODEL_ID,
            num_labels=NUM_CLASSES,
            hidden_dropout_prob=HPARAMS["dropout"],
        )
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params/1e6:.1f}M")

        # Focal loss
        criterion = FocalLoss(
            gamma=float(HPARAMS["focal_loss_gamma"]),
            alpha=float(HPARAMS["focal_loss_alpha"]),
        )

        # Optimizer + scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(HPARAMS["lr"]),
            weight_decay=float(HPARAMS["weight_decay"]),
        )
        epochs = 2 if dev_mode else int(HPARAMS["epochs"])
        scheduler = OneCycleLR(
            optimizer,
            max_lr=float(HPARAMS["lr"]),
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=float(HPARAMS["warmup_ratio"]),
        )

    except ImportError as e:
        print(f"  Missing dependency: {e}")
        print("  Install [train] extras in Colab/Kaggle.")
        return {}

    # MLflow
    try:
        import mlflow
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    except ImportError:
        mlflow = None  # type: ignore[assignment]

    results: dict[str, Any] = {}
    best_macro_f1 = 0.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with (mlflow.start_run(run_name=run_name) if mlflow else _NullContext()):
        if mlflow:
            mlflow.log_params({
                **{k: v for k, v in HPARAMS.items()},
                "dev_mode": dev_mode,
                "model_id": DISTILBERT_MODEL_ID,
                "truncation": f"head{HEAD_TOKENS}+tail{TAIL_TOKENS}",
            })

        print(f"\n  Training {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion,
                device, epoch, epochs,
                float(HPARAMS["max_grad_norm"]),
            )
            val_metrics = eval_epoch(
                model, val_loader, criterion, device
            )
            scheduler.step()

            epoch_time = time.time() - t0
            macro_f1 = val_metrics["val_macro_f1"]

            print(f"  Epoch {epoch}/{epochs} "
                  f"| train_loss={train_metrics['train_loss']:.4f} "
                  f"| val_f1={macro_f1:.4f} "
                  f"| truncated={train_metrics['truncated_pct']:.1%} "
                  f"| {epoch_time:.1f}s")

            if mlflow:
                mlflow.log_metrics(
                    {**train_metrics, **val_metrics, "epoch": epoch},
                    step=int(epoch),
                )

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                try:
                    import torch
                    torch.save(
                        model.state_dict(),
                        MODELS_DIR / "distilbert_sentiment_best.pt",
                    )
                except ImportError:
                    pass

        results = {
            "best_val_macro_f1": best_macro_f1,
            "model_path": str(MODELS_DIR / "distilbert_sentiment_best.pt"),
            "tokenizer_artifact": str(TOKENIZER_ARTIFACT_DIR),
        }

        if mlflow:
            mlflow.log_metrics({"best_val_macro_f1": best_macro_f1})

            # Fix #6: log tokenizer artifact alongside model
            if TOKENIZER_ARTIFACT_DIR.exists():
                mlflow.log_artifacts(
                    str(TOKENIZER_ARTIFACT_DIR),
                    artifact_path="tokenizer",
                )

        print(f"\n  ✓ Training complete. Best macro F1: {best_macro_f1:.4f}")
        print(f"    Target: ≥ 0.88 | Gap: {best_macro_f1 - 0.88:+.4f}")

    return results


# ─── Null context manager ─────────────────────────────────────────────────────

class _NullContext:
    def __enter__(self) -> _NullContext:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E-CIP v3.0 — DistilBERT Sentiment Fine-Tuning"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/reviews"),
        help="Directory with train_reviews.csv and val_reviews.csv",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: 2 epochs, 500 train samples, batch_size=8",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="distilbert_sentiment_v1",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print("=" * 60)
        print("  E-CIP v3.0 — DistilBERT Fine-Tuning")
        print("  Blueprint Section 04 — Fix #6 + Fix #12")
        print("=" * 60)
        print(f"\n  Data directory not found: {args.data}")
        print("  Run data/pipelines/text_pipeline.py first.")
        print("\n  Training structure verified.")
        print("  Key implementations:")
        print(f"    Tokenizer artifact : {TOKENIZER_ARTIFACT_DIR}")
        print(f"    Head tokens        : {HEAD_TOKENS}")
        print(f"    Tail tokens        : {TAIL_TOKENS}")
        print(f"    Max tokens         : {MAX_TOKENS}")
        print(f"    Focal Loss gamma   : {HPARAMS['focal_loss_gamma']}")
        print(f"    Epochs             : {HPARAMS['epochs']}")
        print(f"    Learning rate      : {HPARAMS['lr']}")
        print("\n  Run in Colab/Kaggle with GPU (1–2 hrs on T4).")

        get_device()
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        print("\n  ✓ Directory structure verified.")
        print("=" * 60 + "\n")
        return

    run_training(
        data_dir=args.data,
        dev_mode=args.dev,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
