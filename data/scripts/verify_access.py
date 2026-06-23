# data/scripts/verify_access.py
# E-CIP v3.0 — Dataset Access Verification Script
# Phase 0, Week 1 — Run this before any pipeline work begins.
# Blueprint Section 20: All datasets verified for public access.
#
# Usage: python data/scripts/verify_access.py

import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from typing import Literal


@dataclass
class DatasetCheck:
    name: str
    method: Literal["url", "kaggle"]
    target: str
    module: str
    license: str


DATASET_CHECKS: list[DatasetCheck] = [
    DatasetCheck(
        name="Products-10K",
        method="kaggle",
        target="hirune924/products10k",
        module="Module 1 — Product Intelligence",
        license="CC0 Public Domain",
    ),
    DatasetCheck(
        name="FEIDEGGER (Zalando)",
        method="url",
        target="https://raw.githubusercontent.com/zalandoresearch/feidegger/master/README.md",
        module="Module 1 — Product Intelligence (backup)",
        license="CC BY 4.0",
    ),
    DatasetCheck(
        name="Fashion-MNIST",
        method="url",
        target="https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/README.md",
        module="Module 1 — Sanity check",
        license="MIT",
    ),
    DatasetCheck(
        name="Amazon Reviews 2023 (McAuley Lab)",
        method="url",
        target="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz",
        module="Module 2 — Sentiment Intelligence",
        license="Non-commercial research",
    ),
    DatasetCheck(
        name="SemEval-2014 Task 4",
        method="url",
        target="https://alt.qcri.org/semeval2014/task4/",
        module="Module 2 — ABSA evaluation",
        license="Open research",
    ),
    DatasetCheck(
        name="UCI Online Retail II",
        method="url",
        target="https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        module="Module 3 — Retention Intelligence",
        license="CC BY 4.0",
    ),
    DatasetCheck(
        name="E-Commerce Behavior (Kaggle)",
        method="kaggle",
        target="mkechinov/ecommerce-behavior-data-from-multi-category-store",
        module="Module 3 — Retention Intelligence (backup)",
        license="CC0 Public Domain",
    ),
]


def check_url(target: str, timeout: int = 15) -> tuple[bool, str]:
    """Check if a URL is reachable with a HEAD request first, GET as fallback."""
    try:
        req = urllib.request.Request(target, method="HEAD")
        req.add_header("User-Agent", "Mozilla/5.0 (ecip-verify/1.0)")
        with urllib.request.urlopen(req, timeout=timeout):
            return True, "HTTP 200 OK"
    except urllib.error.HTTPError as e:
        # Some servers reject HEAD — try GET
        if e.code in (405, 403):
            try:
                req = urllib.request.Request(target, method="GET")
                req.add_header("User-Agent", "Mozilla/5.0 (ecip-verify/1.0)")
                with urllib.request.urlopen(req, timeout=timeout):
                    return True, "HTTP 200 OK (GET fallback)"
            except Exception as e2:
                return False, str(e2)
        return False, f"HTTP {e.code} {e.reason}"
    except Exception as e:
        return False, str(e)


def check_kaggle(target: str, timeout: int = 20) -> tuple[bool, str]:
    """Check Kaggle dataset accessibility via the Kaggle CLI."""
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "files", target],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, "Kaggle API OK"
        # Check for common failure modes
        stderr = result.stderr.lower()
        if "401" in stderr or "unauthorized" in stderr:
            return False, "Kaggle API key not configured — run: kaggle datasets list"
        if "404" in stderr or "not found" in stderr:
            return False, "Dataset not found — slug may have changed"
        return False, result.stderr.strip() or "Unknown Kaggle error"
    except FileNotFoundError:
        return False, "Kaggle CLI not installed — run: pip install kaggle"
    except subprocess.TimeoutExpired:
        return False, "Timeout — check network connectivity"
    except Exception as e:
        return False, str(e)


def run_checks(checks: list[DatasetCheck]) -> dict[str, bool]:
    """Run all dataset checks and print a formatted report."""
    print("\n" + "=" * 60)
    print("  E-CIP v3.0 — Dataset Access Verification")
    print("  Blueprint Section 20 — Phase 0, Week 1")
    print("=" * 60)

    results: dict[str, bool] = {}

    for check in checks:
        print(f"\n  [{check.module}]")
        print(f"  Dataset : {check.name}")
        print(f"  License : {check.license}")

        if check.method == "url":
            ok, message = check_url(check.target)
        else:
            ok, message = check_kaggle(check.target)

        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  Status  : {status} — {message}")
        results[check.name] = ok

    print("\n" + "=" * 60)
    passed = sum(results.values())
    total = len(results)
    print(f"  Result  : {passed}/{total} datasets accessible")

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print("\n  Failed datasets:")
        for name in failed:
            print(f"    ✗ {name}")
        print("\n  Action required before proceeding to Phase 1.")
    else:
        print("\n  All datasets accessible. Proceed to dvc repro.")

    print("=" * 60 + "\n")
    return results


def main() -> None:
    results = run_checks(DATASET_CHECKS)
    # Exit with error code if any critical dataset failed
    # Kaggle failures are expected without CLI setup — warn but don't block
    url_failures = [
        name for name, ok in results.items()
        if not ok and any(
            c.name == name and c.method == "url"
            for c in DATASET_CHECKS
        )
    ]
    if url_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()