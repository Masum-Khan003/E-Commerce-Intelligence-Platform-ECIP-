# ADR-001: Churn Label Definition

**Date:** 2026-06-23  
**Status:** Accepted  
**Module:** Module 3 — Retention Prediction Engine  
**Blueprint Reference:** Section 21 — Critical Fix #2

---

## Context

The UCI Online Retail II dataset contains transactional records but has
no pre-existing churn label. The label must be engineered explicitly.
The choice of observation window, snapshot date, and prediction horizon
directly affects class balance, model performance, and business meaning.
This is the most consequential modelling decision in Module 3.

---

## Decision

A customer is defined as **churned (label = 1)** if they made **zero
purchases** in the **90 days** following the observation window end date.

| Parameter | Value | Rationale |
|---|---|---|
| Observation window | 2009-12-01 → 2010-11-30 | 12 months of behavioural history |
| Snapshot date | 2010-11-30 | Last date with sufficient history before dataset end |
| Prediction horizon | 2010-12-01 → 2011-02-28 | 90 days post-snapshot |
| Churn definition | Zero purchases in horizon | Binary, unambiguous, measurable |
| Scope | UK customers only | Single-currency RFM integrity (Fix #24) |
| Guest checkouts | Excluded (null CustomerID) | Cannot track repeat behaviour (Fix #5) |

---

## Rationale for 90-Day Horizon

- The UCI dataset spans Dec 2009 – Dec 2011, leaving adequate label
  period before dataset end
- 90 days aligns with typical e-commerce repurchase cycles (60–120 days)
- Avoids class imbalance extremes: 30-day horizons produce too few
  churned customers; 180-day horizons produce too few retained customers
- Expected churn rate: 15–80% (validated by `churn_label_engineer.py`)

---

## Alternatives Considered

| Horizon | Expected Churn Rate | Reason Rejected |
|---|---|---|
| 30 days | < 15% | Too few churned — severe class imbalance |
| 60 days | ~20–35% | Viable fallback if 90-day rate is out of range |
| 90 days | ~35–55% | **Selected** |
| 180 days | > 70% | Too few retained — opposite imbalance problem |

If `churn_label_engineer.py` reports a rate outside [15%, 80%], the
fallback sequence is: try 60 days, then 120 days, then re-evaluate
dataset scope.

---

## Consequences

- `models/retention/churn_label_engineer.py` encodes this definition
- The churn label definition is surfaced in every API response via
  `churn_label_definition: "no_purchase_90d"` (Blueprint Section 05)
- Model card must document this definition and its business meaning
- Sensitivity analysis across horizons is logged to MLflow in Phase 4

---

## Known Limitations

- UK-only scope excludes international customers (~15–20% of dataset)
- Guest checkouts (~25% of transactions) cannot be labelled and are
  excluded — documented as a known limitation in the model card
- The 90-day horizon is a portfolio decision; production systems should
  validate against actual business repurchase cycle data