# mlops/promotion_gate.py
# E-CIP v3.0 — Model Promotion Gate Check
# Blueprint Section 10 — Staging -> Production promotion gate
#
# Gate: ensemble CV ROC-AUC >= 0.87 AND calibrated ECE < 0.05.
# Reads the real artifacts models/retention/train.py and calibrate.py
# produce — no separate MLflow Model Registry wrapper exists in this
# build (out of scope; this script is the concrete, testable substitute
# .github/workflows/model_retrain.yml needs to branch on).
#
# Usage:
#   python mlops/promotion_gate.py --module retention
#   python mlops/promotion_gate.py --module retention --github-output  (writes to $GITHUB_OUTPUT)

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

AUC_FLOOR = 0.87
ECE_CEILING = 0.05

TRAINING_METRICS_PATH = Path("models/retention/artifacts/training_metrics.json")
CALIBRATION_METRICS_PATH = Path("models/retention/artifacts/calibration_metrics.json")


def check_promotion_gate(
    training_metrics_path: Path = TRAINING_METRICS_PATH,
    calibration_metrics_path: Path = CALIBRATION_METRICS_PATH,
) -> tuple[bool, dict[str, object]]:
    """Returns (approved, details) — details always includes the metrics checked."""
    if not training_metrics_path.exists() or not calibration_metrics_path.exists():
        return False, {
            "reason": "training or calibration metrics missing — run train.py and calibrate.py first",
        }

    training = json.loads(training_metrics_path.read_text())
    calibration = json.loads(calibration_metrics_path.read_text())

    auc = float(training.get("ensemble_cv_auc", 0.0))
    ece = float(calibration.get("ece_after", 1.0))

    auc_pass = auc >= AUC_FLOOR
    ece_pass = ece < ECE_CEILING
    approved = auc_pass and ece_pass

    return approved, {
        "ensemble_cv_auc": auc,
        "auc_floor": AUC_FLOOR,
        "auc_pass": auc_pass,
        "ece_after": ece,
        "ece_ceiling": ECE_CEILING,
        "ece_pass": ece_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E-CIP v3.0 — Model Promotion Gate Check")
    parser.add_argument("--module", type=str, default="retention")
    parser.add_argument(
        "--github-output", action="store_true",
        help="Write promotion_approved=true|false to $GITHUB_OUTPUT for workflow branching",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  E-CIP v3.0 — Model Promotion Gate Check")
    print(f"  Module: {args.module}")
    print("=" * 60)

    approved, details = check_promotion_gate()

    for key, value in details.items():
        print(f"  {key}: {value}")

    print()
    print("  ✓ PROMOTION APPROVED" if approved else "  ✗ PROMOTION BLOCKED")
    print("=" * 60)

    if args.github_output:
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"promotion_approved={'true' if approved else 'false'}\n")

    sys.exit(0 if approved else 1)


if __name__ == "__main__":
    main()
