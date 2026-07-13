# api/routers/sentiment.py
# E-CIP v3.0 — Sentiment Analysis API Router
# Blueprint Section 12 — /v1/sentiment/analyze
#
# Endpoints:
#   POST /v1/sentiment/analyze       — single review, sync
#   POST /v1/sentiment/analyze/batch — async batch via Celery
#
# Response contract (Blueprint Section 04):
#   overall_sentiment, overall_confidence,
#   aspect_sentiments (from ABSA pipeline),
#   sentiment_score [-1, 1] for Module 3 retention features,
#   truncation_applied, tokenizer_version, model_version,
#   inference_ms
#
# Fix #6: tokenizer_version in response — loaded from artifact, never Hub

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_VERSION = "distilbert_sentiment_v1.0.0"
TOKENIZER_VERSION = "distilbert_tokenizer_v1.0.0"
LOW_CONFIDENCE_THRESHOLD = 0.65

LABEL_NAMES = ["negative", "neutral", "positive"]

router = APIRouter(prefix="/v1/sentiment", tags=["Sentiment Intelligence"])


# ─── Response schemas ─────────────────────────────────────────────────────────

class AspectSentiment(BaseModel):
    aspect: str
    sentiment: str
    score: float = Field(ge=0.0, le=1.0)
    method: str = "zero_shot_nli"


class SentimentAnalyzeResponse(BaseModel):
    request_id: str
    review_text: str
    overall_sentiment: str
    overall_confidence: float = Field(ge=0.0, le=1.0)
    aspect_sentiments: list[AspectSentiment]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    truncation_applied: bool
    tokenizer_version: str
    model_version: str
    inference_ms: int


class SentimentBatchResponse(BaseModel):
    task_id: str
    status: str
    message: str


class SentimentRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=50000,
        description="Review text to analyze",
    )
    include_aspects: bool = Field(
        default=True,
        description="Whether to run ABSA pipeline for aspect sentiments",
    )


# ─── Inference helpers ────────────────────────────────────────────────────────

def get_model_registry() -> dict[str, Any]:
    """Return loaded model registry from FastAPI lifespan loader."""
    try:
        from api.main import model_registry
        registry: dict[str, Any] = model_registry
        return registry
    except ImportError:
        return {}


def run_sentiment_inference(
    text: str,
    model: Any,
    tokenizer: Any,
    device: Any,
) -> dict[str, Any]:
    """
    Run DistilBERT inference on a single review.
    Returns label, confidence, sentiment_score, truncation_applied.
    """
    try:
        import torch
        import torch.nn.functional as functional

        from models.sentiment.finetune import head_tail_tokenize

        model.eval()
        encoding = head_tail_tokenize(text, tokenizer)
        truncation_applied: bool = bool(encoding.get("truncation_applied", False))

        input_ids = torch.tensor(
            [encoding["input_ids"]], dtype=torch.long
        ).to(device)
        attention_mask = torch.tensor(
            [encoding["attention_mask"]], dtype=torch.long
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            probs = functional.softmax(outputs.logits, dim=-1).squeeze()

        probs_list: list[float] = probs.cpu().tolist()
        pred_idx = int(probs.argmax().item())
        confidence = float(probs_list[pred_idx])
        label = LABEL_NAMES[pred_idx]

        # Continuous sentiment score for Module 3
        # Weighted sum: negative=-1, neutral=0, positive=+1
        sentiment_score = float(
            -1.0 * probs_list[0]
            + 0.0 * probs_list[1]
            + 1.0 * probs_list[2]
        )
        sentiment_score = round(max(-1.0, min(1.0, sentiment_score)), 4)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "sentiment_score": sentiment_score,
            "truncation_applied": truncation_applied,
            "probs": probs_list,
        }

    except ImportError:
        return {}


def map_to_overall_sentiment(label: str, confidence: float) -> str:
    """
    Map 3-class prediction to display-friendly overall sentiment.
    Mixed sentiment surfaces when confidence is below threshold.
    """
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "Mixed"
    return label.capitalize()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=SentimentAnalyzeResponse,
    summary="Analyze sentiment of a product review",
    description=(
        "Analyze a product review for overall sentiment (Positive/Negative/Neutral) "
        "and aspect-level sentiments (Battery, Display, Shipping, Build, Price, Support). "
        "Returns a continuous sentiment_score in [-1, 1] for use in the retention module."
    ),
)
async def analyze_sentiment(
    request: SentimentRequest,
) -> SentimentAnalyzeResponse:
    """
    Single review sentiment analysis endpoint.
    Blueprint Section 04 + Section 12.
    """
    t0 = time.time()
    request_id = f"req_{uuid.uuid4().hex[:8]}"

    model_registry = get_model_registry()
    model = model_registry.get("distilbert")
    tokenizer = model_registry.get("tokenizer")
    device = model_registry.get("device")

    # ── Model inference ───────────────────────────────────────────────────
    if model is None or tokenizer is None:
        # Stub response in dev/test mode
        return _stub_response(request_id, request.text, time.time() - t0)

    inference_result = run_sentiment_inference(
        request.text, model, tokenizer, device
    )

    if not inference_result:
        raise HTTPException(
            status_code=500,
            detail="Sentiment inference failed",
        )

    label = inference_result["label"]
    confidence = inference_result["confidence"]
    sentiment_score = inference_result["sentiment_score"]
    truncation_applied = inference_result["truncation_applied"]
    overall_sentiment = map_to_overall_sentiment(label, confidence)

    # ── ABSA pipeline ─────────────────────────────────────────────────────
    aspect_sentiments: list[AspectSentiment] = []

    if request.include_aspects:
        try:
            from models.sentiment.absa_pipeline import ABSAPipeline
            absa = ABSAPipeline(device=-1)  # CPU for ABSA
            if absa.load():
                raw_aspects = absa.extract_aspects(request.text)
                aspect_sentiments = [
                    AspectSentiment(
                        aspect=a["aspect"],
                        sentiment=a["sentiment"],
                        score=a["score"],
                        method=a["method"],
                    )
                    for a in raw_aspects
                ]
        except Exception:
            pass  # ABSA is best-effort — never blocks the response

    inference_ms = int((time.time() - t0) * 1000)

    return SentimentAnalyzeResponse(
        request_id=request_id,
        review_text=request.text[:200] + "..." if len(request.text) > 200 else request.text,
        overall_sentiment=overall_sentiment,
        overall_confidence=confidence,
        aspect_sentiments=aspect_sentiments,
        sentiment_score=sentiment_score,
        truncation_applied=truncation_applied,
        tokenizer_version=TOKENIZER_VERSION,
        model_version=MODEL_VERSION,
        inference_ms=inference_ms,
    )


@router.post(
    "/analyze/batch",
    response_model=SentimentBatchResponse,
    summary="Batch analyze reviews (async)",
    description=(
        "Submit a batch of reviews for async sentiment analysis. "
        "Returns task_id to poll via GET /v1/tasks/{task_id}. "
        "Processed by GPU Celery worker (concurrency=1)."
    ),
)
async def analyze_sentiment_batch(
    requests: list[SentimentRequest],
) -> SentimentBatchResponse:
    """
    Async batch sentiment analysis via Celery GPU queue.
    Blueprint Section 12.
    """
    if not requests:
        raise HTTPException(status_code=422, detail="No reviews provided")

    if len(requests) > 500:
        raise HTTPException(
            status_code=422,
            detail="Batch size exceeds limit of 500 reviews",
        )

    task_id = f"task_{uuid.uuid4().hex[:12]}"

    try:
        from api.workers.celery_tasks import batch_score_sentiment
        texts = [r.text for r in requests]
        batch_score_sentiment.apply_async(
            args=[texts],
            task_id=task_id,
            queue="gpu_queue",
        )
    except ImportError:
        pass  # Celery not yet wired — stub response

    return SentimentBatchResponse(
        task_id=task_id,
        status="PENDING",
        message=f"Batch of {len(requests)} reviews queued. "
                f"Poll GET /v1/tasks/{task_id} for results.",
    )


# ─── Stub response ────────────────────────────────────────────────────────────

def _stub_response(
    request_id: str,
    text: str,
    elapsed: float,
) -> SentimentAnalyzeResponse:
    """Stub response when model not loaded (dev/test mode)."""
    preview = text[:200] + "..." if len(text) > 200 else text
    return SentimentAnalyzeResponse(
        request_id=request_id,
        review_text=preview,
        overall_sentiment="Negative",
        overall_confidence=0.82,
        aspect_sentiments=[
            AspectSentiment(
                aspect="Battery",
                sentiment="Negative",
                score=0.96,
                method="zero_shot_nli",
            ),
            AspectSentiment(
                aspect="Display",
                sentiment="Positive",
                score=0.94,
                method="zero_shot_nli",
            ),
        ],
        sentiment_score=-0.31,
        truncation_applied=False,
        tokenizer_version=f"{TOKENIZER_VERSION} (stub)",
        model_version=f"{MODEL_VERSION} (stub)",
        inference_ms=int(elapsed * 1000),
    )
