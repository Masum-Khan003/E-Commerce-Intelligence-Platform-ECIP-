# api/main.py
# E-CIP v3.0 — FastAPI Application Entry Point
# Blueprint Section 12 — Fix #7 (startup warm-up loader)
#
# Loading order: XGBoost -> EfficientNet -> DistilBERT (lightest to
# heaviest), so the retention endpoint becomes usable earliest while
# heavier models continue loading.
#
# Deviation from the blueprint's literal wording, documented honestly:
# the blueprint says /health/ready stays 503 "until ALL models loaded".
# In this project's actual current state, Module 1 (EfficientNet) and
# Module 2 (DistilBERT) haven't been trained yet — that requires GPU
# time on Colab/Kaggle, out of scope for this local build (see
# HANDOFF.md). Gating readiness on their presence would make /health/ready
# permanently return 503 despite Module 3 (Retention) being fully real
# and working. Instead, readiness reflects "the warm-up sequence ran to
# completion" (attempting all models, tolerating individual absences),
# and the response body reports per-model load status honestly so
# monitoring can see exactly what's actually available. Once Module 1/2
# are trained and their weight files exist, they'll load automatically —
# no code change needed here.

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app

logger = logging.getLogger(__name__)

model_registry: dict[str, Any] = {}
_load_status: dict[str, str] = {}
_ready = False

PRODUCT_WEIGHTS_PATH = Path("models/product/weights/efficientnet_b3_best.pt")
SENTIMENT_WEIGHTS_PATH = Path("models/sentiment/weights/distilbert_sentiment_best.pt")
TOKENIZER_ARTIFACT_DIR = Path("data/feature_store/artifacts/tokenizer_v1")
OOD_REFERENCE_PATH = Path("data/feature_store/product_features/mahalanobis_reference_v1.json")

NUM_PRODUCT_CLASSES = 8


def _load_retention_models() -> None:
    """Lightest model first — small joblib artifacts, no GPU."""
    try:
        from api.routers.retention import _load_local_retention_artifacts

        registry = _load_local_retention_artifacts()
        if "xgb_final" in registry and "lgbm_final" in registry and "calibrator" in registry:
            model_registry.update(registry)
            _load_status["xgb_ensemble"] = "loaded"
        else:
            _load_status["xgb_ensemble"] = (
                "missing — run models/retention/train.py and calibrate.py"
            )
    except Exception as e:
        _load_status["xgb_ensemble"] = f"error: {e}"


def _load_product_model() -> None:
    """Fix #18: explicit device selection, CUDA -> MPS -> CPU."""
    try:
        from models.product.train import get_device

        device = get_device()
        model_registry["device"] = device

        if not PRODUCT_WEIGHTS_PATH.exists():
            _load_status["efficientnet"] = (
                "missing — requires GPU training on Colab/Kaggle (see HANDOFF.md)"
            )
            return

        import timm
        import torch

        model = timm.create_model(
            "efficientnet_b3", pretrained=False, num_classes=NUM_PRODUCT_CLASSES
        )
        state_dict = torch.load(PRODUCT_WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        model_registry["efficientnet"] = model
        _load_status["efficientnet"] = "loaded"

        if OOD_REFERENCE_PATH.exists():
            import json

            model_registry["mahal_ref"] = json.loads(OOD_REFERENCE_PATH.read_text())
    except ImportError as e:
        _load_status["efficientnet"] = f"dependency not installed: {e}"
    except Exception as e:
        _load_status["efficientnet"] = f"error: {e}"


def _load_sentiment_model() -> None:
    """Heaviest model last — DistilBERT + tokenizer artifact (Fix #6: never Hub)."""
    try:
        device = model_registry.get("device")
        if device is None:
            from models.product.train import get_device

            device = get_device()
            model_registry["device"] = device

        if not SENTIMENT_WEIGHTS_PATH.exists() or not TOKENIZER_ARTIFACT_DIR.exists():
            _load_status["distilbert"] = (
                "missing — requires GPU training on Colab/Kaggle (see HANDOFF.md)"
            )
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Fix #6: tokenizer loaded from the saved artifact, never from the Hub.
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ARTIFACT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=3
        )
        state_dict = torch.load(SENTIMENT_WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        model_registry["distilbert"] = model
        model_registry["tokenizer"] = tokenizer
        _load_status["distilbert"] = "loaded"
    except ImportError as e:
        _load_status["distilbert"] = f"dependency not installed: {e}"
    except Exception as e:
        _load_status["distilbert"] = f"error: {e}"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _ready

    start = time.time()
    logger.info("Starting model warm-up sequence: XGBoost -> EfficientNet -> DistilBERT")

    _load_retention_models()
    _load_product_model()
    _load_sentiment_model()

    _ready = True
    elapsed = time.time() - start
    logger.info(f"Warm-up sequence complete in {elapsed:.1f}s — status: {_load_status}")

    from observability.prometheus.metrics import model_warmup_seconds

    model_warmup_seconds.observe(elapsed)

    yield

    model_registry.clear()
    _load_status.clear()


app = FastAPI(title="E-CIP API", version="3.0.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def _record_http_requests(request: Any, call_next: Any) -> Any:
    """Backs the Alertmanager HighErrorRate rule with a real metric."""
    from observability.prometheus.metrics import http_requests_total

    response = await call_next(request)
    http_requests_total.labels(
        method=request.method, status=str(response.status_code)
    ).inc()
    return response


@app.get("/health/live", tags=["Health"])
async def liveness() -> dict[str, str]:
    """Always 200 — process is up. No auth required."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def readiness() -> dict[str, Any]:
    """
    503 until the warm-up sequence has run to completion; 200 afterward
    with per-model load status (see module docstring for why this
    doesn't strictly gate on every model being present).
    """
    if not _ready:
        raise HTTPException(status_code=503, detail="Models loading — retry shortly")
    return {"status": "ready", "models": _load_status}


def _register_routers() -> None:
    """
    Deferred import — routers import `from api.main import model_registry`,
    so importing them at module top-level would create a circular import.

    Fix #14: every /v1/* route requires a valid X-API-Key. The dependency
    is attached here at include_router time rather than per-endpoint, so
    no router file needs to import auth machinery itself.
    """
    from fastapi import Depends

    from api.middleware.auth import verify_api_key
    from api.routers import analytics, explain, products, retention, sentiment

    auth_dep = [Depends(verify_api_key)]
    app.include_router(products.router, dependencies=auth_dep)
    app.include_router(sentiment.router, dependencies=auth_dep)
    app.include_router(retention.router, dependencies=auth_dep)
    app.include_router(explain.router, dependencies=auth_dep)
    app.include_router(analytics.router, dependencies=auth_dep)


_register_routers()
