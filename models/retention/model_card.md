# Model Card — Retention Prediction Engine
**Module:** 3 — Retention Intelligence
**Version:** retention_ensemble_v1.0.0
**Blueprint Reference:** Section 05, Section 18, Section 21
**Status:** Trained on real data (Phase 4, Weeks 11–13)

---

## Model Details

| Field | Value |
|---|---|
| Architecture | XGBoost + LightGBM ensemble (weighted average, 0.5/0.5) |
| Framework | scikit-learn API (xgboost 3.3, lightgbm 4.7) |
| Hyperparameters | Optuna-tuned, 50 trials, SQLite-persisted study (`mlops/optuna_studies.db`) |
| Calibration | Isotonic Regression (n_calibration=3,923 ≥ 1,000 threshold) |
| Explainability | TreeSHAP (`shap.TreeExplainer`) on the XGBoost half, margin (log-odds) space |
| Training script | `models/retention/train.py` |
| Calibration script | `models/retention/calibrate.py` |
| Explanation script | `models/retention/shap_explain.py` |
| Artifacts | `models/retention/weights/{xgb,lgbm}_final.joblib`, `models/retention/artifacts/calibrator.joblib` |

---

## Intended Use

**Primary use case:**
90-day churn risk scoring for UK e-commerce customers, using RFM (recency/
frequency/monetary), behavioural, temporal, and sentiment-derived features.
Drives a retention-offer decision (`recommended_action`) and a SHAP-based
explanation of the top risk factors per customer.

**Intended users:**
E-commerce retention/CRM teams deciding which customers to target with
retention offers; ML engineers monitoring the model in production.

**Out of scope:**
- Non-UK customers (mixed-currency `monetary_value` would corrupt RFM features)
- Guest checkouts (no `CustomerID` — ~22.8% of raw transactions, excluded)
- Customers with tenure > 363 days (the training data's 12-month observation
  window caps tenure by construction — see Known Limitations)
- Real-time sentiment fusion (see Known Limitations — sentiment inputs are
  synthetic for this dataset)

---

## Training Data

| Field | Value |
|---|---|
| Dataset | UCI Online Retail II (real download, `archive.ics.uci.edu`) |
| Raw rows | 1,067,371 (both sheets, 2009-12-01 → 2011-12-09) |
| After guest-checkout exclusion | 824,364 (22.8% removed — null `CustomerID`) |
| After UK-only scoping | 741,301 (removed 83,063 international rows) |
| After cancellation/negative-value removal | 725,250 |
| Observation window | 2009-12-01 → 2010-11-30 (12 months) |
| Prediction horizon | 2010-12-01 → 2011-02-28 (90 days) |
| Snapshot date | 2010-11-30 |
| Customers (labeled) | 3,923 |
| Churn rate | 67.8% (2,658 churned / 1,265 retained) — within the [15%, 80%] validity band |

**Churn label definition** (`models/retention/churn_label_engineer.py`, ADR-001):
churned = 1 if the customer made zero purchases in the 90 days following the
observation window; retained = 0 otherwise.

**Sentiment features** — see **Known Limitations** below; this is the one
place this model card asks for extra scrutiny.

**Preprocessing:**
- RFM + behavioural + temporal features computed by `data/pipelines/tabular_pipeline.py`
- `purchase_gap_cv` sentinel = -1.0 for single-order customers (82 customers, 2.1%),
  paired with an `is_single_purchase` binary flag
- All 17 continuous features scaled with a `StandardScaler` fit once at training
  time and reused at inference (`data/feature_store/artifacts/scaler_v1.joblib`) —
  never re-fit (training-serving consistency)
- SMOTE (`imblearn.over_sampling.SMOTE`) applied **only** to each CV fold's
  training split, never to a validation/test split — see `train.py` module
  docstring for why this ordering matters

---

## Evaluation Data

Two distinct held-out mechanisms, matched to what each metric needs:

1. **5-fold stratified CV** (`train.py`) — SMOTE re-applied fresh inside each
   fold — used for the headline AUC and all subgroup metrics below (subgroup
   metrics are computed on out-of-fold predictions, not in-sample ones).
2. **50/50 split of the CV out-of-fold pool** (`calibrate.py`) — one half
   fits the Platt/Isotonic calibrator, the other half reports ECE. A
   calibrator fit and scored on the same data would otherwise report a
   trivially optimistic ECE (isotonic regression can fit a monotonic step
   function almost perfectly to its own training points).

---

## Performance Metrics

### Overall (5-fold CV, out-of-fold)

| Metric | Target | Actual |
|---|---|---|
| XGBoost CV ROC-AUC | — | 0.9138 ± 0.0149 |
| LightGBM CV ROC-AUC | — | 0.9055 ± 0.0152 |
| **Ensemble CV ROC-AUC** | **≥ 0.87** | **0.9118** |
| Optuna best single-trial CV AUC (trial #25/50) | — | 0.9146 |
| ECE before calibration (held-out half) | — | 0.0444 |
| **ECE after calibration (held-out half)** | **< 0.05** | **0.0141** |
| Decision threshold (F-beta, β=2) | — | 0.33 |
| Precision at threshold | ≥ 0.80 | 0.8271 |
| Recall at threshold | ≥ 0.75 | 0.9789 |
| SHAP sum-consistency (margin space) | < 0.05 | ~0.0000 (machine precision) |

All figures above are real, computed against the real UCI Online Retail II
download — not placeholders.

### Subgroup Analysis (Blueprint Section 18) — out-of-fold predictions

**ROC-AUC by tenure quartile** (raw `tenure_days`, inverse-scaled from the
stored standardized feature — the parquet feature table holds scaled values):

| Tenure band | n | ROC-AUC | Churn rate |
|---|---|---|---|
| 0–90d | 863 | 0.8709 | 76.1% |
| 91–180d | 547 | 0.8995 | 76.8% |
| 181–365d | 2,513 | 0.9236 | 62.9% |
| 365d+ | 0 | — | n/a — 12-month observation window caps tenure at 363 days |

The model is measurably more reliable for longer-tenured customers — makes
sense, they have more behavioural signal (more orders, longer history) for
RFM features to differentiate on.

**Precision/Recall by frequency band** (raw order count):

| Frequency band | n | Precision | Recall | Churn rate |
|---|---|---|---|---|
| 1 order | 1,318 | 0.9186 | 0.9966 | 88.5% |
| 2–5 orders | 1,776 | 0.7939 | 0.9847 | 70.0% |
| 6–20 orders | 751 | 0.6193 | 0.8436 | 32.4% |
| 20+ orders | 78 | 0.6667 | 0.4000 | 6.4% |

Recall drops sharply in the 20+ order band (0.40) — this is the smallest,
most churn-resistant segment (6.4% churn rate); the low-threshold-optimized
model (tuned for overall recall via F-beta β=2) under-flags this segment's
rare churners. Worth a segment-specific threshold if 20+-order customers
become a target audience.

**Single- vs multi-purchase customers:**

| Segment | n | Churn rate | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| Single-purchase | 82 | 91.5% | 0.8724 | 0.9146 | 1.0000 |
| Multi-purchase | 3,841 | 67.2% | 0.9137 | 0.8232 | 0.9752 |

Single-purchase customers churn at a much higher rate (91.5% vs 67.2%) and
the model catches essentially all of them (recall 1.0) — consistent with
`purchase_gap_cv`'s -1.0 sentinel and `is_single_purchase` flag giving the
model a clean, unambiguous signal for this segment.

---

## Explainability

TreeSHAP (`models/retention/shap_explain.py`) on the XGBoost half of the
ensemble, top-10 features by `|shap_value|`, each tagged
`increases_churn`/`decreases_churn`.

**Important scoping note:** SHAP values here are in the model's raw margin
(log-odds) space, not probability space. The installed `shap`/`xgboost`
version pairing doesn't support probability-space TreeSHAP output for this
model (`feature_perturbation="interventional"` raises `NotImplementedError`
on this XGBoost version's tree encoding; `model_output="probability"` is
only supported together with `interventional`). The sum-consistency check
(`|sum(shap) + expected_value − prediction| < 0.05`) is therefore verified
against the model's raw margin score, not `predict_proba`'s probability —
this is what TreeSHAP's additivity guarantee actually promises, and it
passes at machine precision on real data. `churn_probability` in both the
`/v1/retention/score` and `/v1/explain/shap/{request_id}` API responses is
still a proper `predict_proba`-derived probability; only the internal SHAP
math runs in margin space.

**Verified direction:** `avg_sentiment_score` and the three aspect-sentiment
features consistently rank in the top SHAP features with `increases_churn`
direction for negative-sentiment synthetic customers — confirmed by
`tests/model_tests/test_cross_module.py::test_negative_sentiment_increases_churn_risk`
and a live smoke test through the `/v1/retention/score` → `/v1/explain/shap`
API round-trip.

---

## Ethical Considerations

- **UK-only scope**: this model has never seen non-UK customer behaviour.
  Deploying it against other markets without retraining on that market's
  data would be inappropriate.
- **Decision consequences**: `recommended_action: RETENTION_OFFER` drives a
  real business action (a discount/offer). A false positive costs a
  discount; a false negative costs a customer. The threshold (0.33, F-beta
  β=2) is deliberately biased toward recall — the business judgment
  encoded here is that missing a churner is worse than over-offering.

---

## Known Limitations

1. **Sentiment fusion is SYNTHETIC for this dataset.** UCI Online Retail II
   has no review text, and Module 2 (DistilBERT sentiment) was trained on a
   completely separate corpus (Amazon Reviews 2023) with no shared customer
   identifiers. There is no real linkage between the two datasets. Rather
   than leaving the sentiment feature columns at a neutral-prior placeholder
   (0.0 for every customer), `data/scripts/synthesize_demo_sentiment.py`
   generates a seeded, weakly churn-correlated synthetic review-sentiment
   dataset so the merge logic (`merge_sentiment_features()` in
   `tabular_pipeline.py`, Gate G8 causal-integrity filter) and the
   cross-module invariant tests have real, non-degenerate signal to
   exercise. **The reported SHAP ranking of `avg_sentiment_score` as a top
   churn driver reflects this synthetic signal, not a real customer
   sentiment effect.** A production deployment would replace this generator
   with live Module 2 output keyed by a real, shared customer/order
   identifier.
2. **Tenure capped at 363 days.** The 12-month observation window means no
   customer in this dataset has tenure beyond 365 days — the "365d+"
   subgroup in the blueprint's specified tenure bands is empty by
   construction, not a modeling gap.
3. **Guest checkouts excluded** (~22.8% of raw transactions had no
   `CustomerID`) — this model has no signal about guest-checkout behaviour.
4. **Single-purchase customers are a small, extreme segment** (82 of 3,923,
   2.1%) — the near-perfect recall (1.0) on this segment should be read with
   that small `n` in mind.
5. **LightGBM's SHAP explanations are not implemented** — only the XGBoost
   half of the ensemble is explained via TreeSHAP. The blueprint's
   `ShapExplanationResponse` schema describes a single top-10 feature list
   per prediction, not a dual-model breakdown, so this is a scope match,
   not a shortfall — but it's worth knowing the SHAP narrative doesn't
   capture LightGBM's share of the ensemble's decision.
6. **Optuna tuning covers XGBoost only** — `mlops/optuna_search.py`'s
   `lgbm_objective` exists but isn't wired into the CLI yet; LightGBM runs
   with reasonable defaults (search-space midpoints), not a tuned optimum.

---

## API Response Contract

`POST /v1/retention/score` → `RetentionScoreResponse`
(`api/routers/retention.py`): `request_id`, `customer_id`,
`churn_probability`, `risk_band` (LOW <0.3 / MEDIUM 0.3–0.6 / HIGH >0.6),
`recommended_action`, `top_risk_factors` (top-3 SHAP features),
`churn_label_definition`, `is_single_purchase_customer`, `model_version`,
`calibration_method`, `decision_threshold`, `inference_ms`.

`GET /v1/explain/shap/{request_id}` → `ShapExplanationResponse`
(`api/schemas/explain.py`): full top-10 SHAP feature attribution for a
prior `/v1/retention/score` call.

---

## Changelog

- **v1.0.0** (Phase 4, Weeks 11–13): first real training run against
  downloaded UCI Online Retail II data. Optuna 50-trial search, SMOTE-in-CV
  ensemble training, Isotonic calibration, TreeSHAP explanations, full
  cross-module test suite passing.
