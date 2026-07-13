# Model Card — Sentiment Intelligence Engine
**Module:** 2 — Sentiment Intelligence  
**Version:** distilbert_sentiment_v1.0.0  
**Blueprint Reference:** Section 04, Section 18, Section 22  
**Status:** Training pending (Phase 3, Week 8–10)

---

## Model Details

| Field | Value |
|---|---|
| Architecture | DistilBERT (distilbert-base-uncased) |
| Pretrained on | English Wikipedia + BookCorpus |
| Framework | PyTorch 2.2 + HuggingFace Transformers 4.39 |
| Parameters | ~67M |
| Max input tokens | 512 (head+tail truncation — Fix #12) |
| Output | 3-class softmax: negative / neutral / positive |
| Artifact URI | `models:/sentiment_model/Production` (MLflow) |
| Tokenizer URI | `models:/sentiment_tokenizer/Production` (MLflow) |
| Training script | `models/sentiment/finetune.py` |
| Baseline model | SetFit (`models/sentiment/setfit_baseline.py`) |
| ABSA model | cross-encoder/nli-deberta-v3-small (zero-shot) |

---

## Intended Use

**Primary use case:**  
Automated sentiment classification of e-commerce product reviews
at scale. Provides document-level sentiment (positive/negative/neutral)
and aspect-level sentiment (Battery, Display, Shipping, Build, Price,
Support) for downstream retention risk scoring.

**Intended users:**  
E-commerce platform ML engineers and product analytics teams.

**Out of scope:**  
- Non-English reviews
- Reviews from categories not in training data without domain-shift
  validation (see Known Limitations)
- Emotion classification (anger, joy, etc.) — only polarity
- Sarcasm detection — model may misclassify sarcastic positive text

---

## Training Data

| Field | Value |
|---|---|
| Dataset | Amazon Reviews 2023 (McAuley Lab, UCSD) |
| Categories | Electronics + Fashion (in-domain) |
| License | Non-commercial research |
| DVC version | Pinned via `data/raw/amazon_reviews.dvc` |
| Sample size | ≥ 100K reviews (dev: 10K) |
| Split | 70% train / 15% val / 15% test (stratified by rating) |
| Label mapping | 1–2★ → negative, 3★ → neutral, 4–5★ → positive |

**Preprocessing:**  
- Text cleaning: whitespace normalisation, HTML tag removal
- Truncation: head+tail strategy — first 128 + last 382 tokens (Fix #12)
- Tokenizer: distilbert-base-uncased saved to MLflow artifact (Fix #6)
- Class imbalance: Focal Loss (γ=2.0) — downweights easy examples

**Loss function:**  
Focal Loss — addresses neutral class underrepresentation.
FL(p_t) = -α(1-p_t)^γ log(p_t), γ=2.0, α=1.0

---

## Evaluation Data

| Field | Value |
|---|---|
| In-domain test set | 15% of Amazon Reviews 2023 Electronics + Fashion |
| OOD test sets | 100 reviews × 4 categories (manually labeled — Fix #32) |
| ABSA evaluation | SemEval-2014 Task 4 (laptop domain) |

---

## Performance Metrics

> **Note:** Metrics below are targets. Populate with actual values
> after training run in Phase 3, Week 9–10.

### Overall Metrics

| Metric | Target | Actual | Status |
|---|---|---|---|
| Macro F1 | ≥ 0.88 | TBD | Pending training |
| Negative Recall | ≥ 0.85 | TBD | Pending training |
| Neutral Precision | ≥ 0.75 | TBD | Pending training |
| Inference p95 | < 50ms | TBD | Pending training |
| vs SetFit Δ | documented | TBD | Pending training |

### Per-Class F1

| Class | Target F1 | Actual F1 | vs SetFit Δ |
|---|---|---|---|
| Negative | ≥ 0.85 | TBD | TBD |
| Neutral | ≥ 0.75 | TBD | TBD |
| Positive | ≥ 0.90 | TBD | TBD |

### SetFit Baseline Comparison

| Metric | SetFit (100-shot) | DistilBERT | Delta |
|---|---|---|---|
| Macro F1 | TBD | TBD | TBD |
| Negative F1 | TBD | TBD | TBD |
| Neutral F1 | TBD | TBD | TBD |
| Positive F1 | TBD | TBD | TBD |

*SetFit baseline trained on same held-out set (Phase 3, Week 8).*

---

## Subgroup Analysis (Blueprint Section 18)

### By Review Length

| Length Band | Token Range | Sample Size | Macro F1 |
|---|---|---|---|
| Short | < 50 tokens | TBD | TBD |
| Medium | 50–300 tokens | TBD | TBD |
| Long | > 300 tokens | TBD | TBD |

### By Star Rating

| Rating | Label | Sample Size | Macro F1 |
|---|---|---|---|
| 1★ | Negative | TBD | TBD |
| 3★ | Neutral | TBD | TBD |
| 5★ | Positive | TBD | TBD |

### Domain-Shift Evaluation (Fix #32)

> Categories where F1 < 0.78 are flagged as out-of-scope.

| Category | In/Out Domain | F1 | Flagged |
|---|---|---|---|
| Electronics | In-domain | TBD | — |
| Fashion | In-domain | TBD | — |
| Home & Kitchen | Out-of-domain | TBD | TBD |
| Sports | Out-of-domain | TBD | TBD |
| Toys | Out-of-domain | TBD | TBD |
| Beauty | Out-of-domain | TBD | TBD |

---

## ABSA Pipeline (Blueprint Section 22)

| Field | Value |
|---|---|
| Model | cross-encoder/nli-deberta-v3-small |
| Method | Zero-shot NLI — no annotations required (Fix #3) |
| Aspects | Battery, Display, Shipping, Build, Price, Support |
| Confidence threshold | 0.70 |
| Evaluation | SemEval-2014 Task 4 (laptop domain) |
| Target F1 | > 0.72 on laptop domain |
| Actual F1 | TBD |

**Retention feature mapping:**

| Aspect | Retention Feature |
|---|---|
| Battery | avg_battery_sentiment |
| Shipping | avg_shipping_sentiment |
| Price | avg_price_sentiment |

---

## Ethical Considerations

- Training data (Amazon Reviews 2023) is English-only — non-English
  reviews may produce unreliable predictions
- Star rating → sentiment label mapping is a heuristic; 3★ reviews
  contain mixed sentiment that the neutral label may not capture
- Focal Loss mitigates class imbalance but neutral class recall
  may still be lower than positive/negative
- Model is not tested for demographic bias in review language

---

## Known Limitations

1. **Tokenizer artifact dependency (Fix #6):** Tokenizer must be
   loaded from the MLflow artifact URI — never re-initialised from
   Hub. Version mismatch silently corrupts predictions.

2. **Head+tail truncation (Fix #12):** Reviews longer than 512 tokens
   use head+tail strategy. Middle sections (tokens 129–130 to end-383)
   are discarded. This is acceptable for most reviews but may lose
   context in highly structured long-form reviews.

3. **Domain scope:** Trained on Electronics + Fashion only. Out-of-domain
   categories flagged after domain-shift evaluation (Fix #32).
   Flagged categories documented above after evaluation.

4. **Zero-shot ABSA ceiling:** The NLI approach has lower F1 than
   supervised ABSA models. Target F1 > 0.72 on SemEval-2014.
   Improvement path: few-shot fine-tuning with aspect annotations.

5. **Sarcasm:** Sarcastic reviews ("Great, another product that breaks
   in a week!") may be misclassified as positive.

6. **Sentiment score range:** sentiment_score ∈ [-1, 1] is a weighted
   sum of class probabilities — not a calibrated probability.
   Use for ranking/feature engineering only, not threshold decisions.

---

## API Response Contract

```json
{
  "request_id": "req_a4c882e1",
  "review_text": "Battery dies in 2 hours but the display is gorgeous",
  "overall_sentiment": "Mixed",
  "overall_confidence": 0.82,
  "aspect_sentiments": [
    {"aspect": "Battery", "sentiment": "Negative", "score": 0.96, "method": "zero_shot_nli"},
    {"aspect": "Display", "sentiment": "Positive", "score": 0.94, "method": "zero_shot_nli"}
  ],
  "sentiment_score": -0.31,
  "truncation_applied": false,
  "tokenizer_version": "distilbert_tokenizer_v1.0.0",
  "model_version": "distilbert_sentiment_v1.0.0",
  "inference_ms": 43
}
```

---

## Changelog

| Version | Date | Notes |
|---|---|---|
| v1.0.0 | TBD | Initial training on Amazon Reviews 2023 |
| v1.0.0-stub | 2026-07-14 | Model card created, training pending |