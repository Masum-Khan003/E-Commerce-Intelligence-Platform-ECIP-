# Model Card — Product Intelligence Engine
**Module:** 1 — Product Intelligence  
**Version:** efficientnet_b3_v1.0.0  
**Blueprint Reference:** Section 03, Section 18  
**Status:** Training pending (Phase 2, Week 5–7)

---

## Model Details

| Field | Value |
|---|---|
| Architecture | EfficientNet-B3 (timm library) |
| Pretrained on | ImageNet-21K (14M images) |
| Framework | PyTorch 2.2 + timm 0.9 |
| Parameters | ~12M |
| Input size | 300 × 300 × 3 (RGB) |
| Output | 8-class softmax |
| Artifact URI | `models:/product_classifier/Production` (MLflow) |
| Training script | `models/product/train.py` |
| Baseline model | ResNet-18 (`models/product/baseline_resnet18.py`) |

---

## Intended Use

**Primary use case:**  
Automated product category classification from product images
at scale — replacing manual categorisation for e-commerce catalogs
exceeding 10K SKUs.

**Intended users:**  
E-commerce platform ML engineers and catalog operations teams.

**Out of scope:**  
- Fine-grained attribute prediction (colour, size, brand)
- Multi-label classification
- Non-product images (people, landscapes, documents)
- Categories outside the 8 trained classes
- Non-UK or non-English product listings (not tested)

---

## Training Data

| Field | Value |
|---|---|
| Dataset | Products-10K (Kaggle — hirune924/products10k) |
| License | CC0 Public Domain |
| DVC version | Pinned via `data/raw/products10k.dvc` |
| Total images | ~10,000 |
| Dev sample | 2,000 images (250/class) |
| Split | 70% train / 15% val / 15% test (stratified) |
| Deduplication | SHA256 hash — no duplicates across splits |
| Sampling | Stratified by category |

**Preprocessing:**  
- Resize to 300×300
- ImageNet normalisation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Training augmentation: random horizontal flip, colour jitter, rotation ±15°

---

## Evaluation Data

| Field | Value |
|---|---|
| Held-out test set | 15% of Products-10K (stratified) |
| Dev test set | 300 images (37–38/class) |
| Domain-shift test | Not applicable (single-domain dataset) |

---

## Performance Metrics

> **Note:** Metrics below are targets. Populate with actual values
> after training run in Phase 2, Week 7.

### Overall Metrics

| Metric | Target | Actual | Status |
|---|---|---|---|
| Top-1 Accuracy | ≥ 92% | TBD | Pending training |
| Macro F1 | ≥ 0.90 | TBD | Pending training |
| Calibration ECE | < 0.05 | TBD | Pending training |
| Inference p95 | < 120ms | TBD | Pending training |
| vs ResNet-18 Δ | documented | TBD | Pending training |

### Per-Class F1 (Blueprint Section 18 — Subgroup Specification)

| Category | Target F1 | Actual F1 | vs ResNet-18 Δ |
|---|---|---|---|
| Electronics | ≥ 0.90 | TBD | TBD |
| Fashion | ≥ 0.90 | TBD | TBD |
| Home & Kitchen | ≥ 0.90 | TBD | TBD |
| Sports | ≥ 0.90 | TBD | TBD |
| Furniture | ≥ 0.90 | TBD | TBD |
| Beauty | ≥ 0.90 | TBD | TBD |
| Books | ≥ 0.90 | TBD | TBD |
| Toys | ≥ 0.90 | TBD | TBD |

### Performance by Image Resolution (Subgroup)

| Resolution Band | Sample Size | Top-1 Acc | Notes |
|---|---|---|---|
| < 100px | TBD | TBD | Excluded by validate_image() |
| 100–300px | TBD | TBD | Pending |
| > 300px | TBD | TBD | Pending |

### ResNet-18 Baseline Comparison

| Metric | ResNet-18 | EfficientNet-B3 | Delta |
|---|---|---|---|
| Top-1 Accuracy | TBD | TBD | TBD |
| Macro F1 | TBD | TBD | TBD |
| Inference p95 | TBD | TBD | TBD |

*Baseline trained on identical splits (Phase 2, Week 5).*

---

## Top-3 Confusion Pairs (Known Failure Modes)

> Populate after training from `models/product/artifacts/top_confusion_pairs.json`

| Rank | True Class | Predicted Class | Count | Business Impact |
|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD |

**Business cost note:**  
Electronics ↔ Home & Kitchen confusion carries highest business cost
(search ranking degradation, catalog integrity risk).
Fashion ↔ Sports confusion carries medium cost (user experience only).

---

## OOD Detection

| Parameter | Value |
|---|---|
| Method | Mahalanobis distance on penultimate layer features |
| Threshold | 99th percentile of training set distances |
| Reference | `data/feature_store/product_features/mahalanobis_reference_v1.npy` |
| API field | `ood_risk_score` (normalised, > 1.0 = flagged) |
| Action | Routes to review_queue table + Prometheus counter |

---

## Ethical Considerations

- Dataset (Products-10K) is CC0 — no copyright concerns
- Model may underperform on product images from non-Western markets
  not represented in training data
- Confidence scores are calibrated (ECE < 0.05) — safe for
  automated decisions at the stated threshold
- Low-confidence predictions (< 0.65) are always routed to human review

---

## Known Limitations

1. **8-class scope:** Only classifies into the 8 trained categories.
   Products outside these categories will be misclassified with
   potentially high confidence — OOD detection is the mitigation.

2. **Image quality dependence:** Performance degrades significantly
   for images below 100px in either dimension. Rejected by
   `validate_image()` at pipeline time.

3. **Single dataset:** Trained on Products-10K only. Domain shift
   to other e-commerce catalogs (eBay, Etsy, etc.) not evaluated.

4. **No attribute prediction:** Cannot predict colour, size, brand,
   material, or other product attributes — only top-level category.

5. **Dev sample gap:** On the 2K dev sample, target accuracy is 85%+
   (not 92%). Full 10K dataset required for production target.
   This is a compute constraint, not an architecture limitation.

---

## Caveats

- OOD detection threshold (99th percentile) is a portfolio choice.
  Production systems should validate threshold against actual
  out-of-distribution rejection rate requirements.
- Grad-CAM heatmaps verified on ≥20 images/class (spot-check).
  Systematic evaluation of explanation quality is future work.

---

## API Response Contract

```json
{
  "request_id": "req_8f2a91c3",
  "product_category": "Electronics",
  "confidence": 0.947,
  "top_3_predictions": [...],
  "is_confident": true,
  "ood_risk_score": 0.12,
  "ood_flagged": false,
  "gradcam_url": "/v1/explain/gradcam/req_8f2a91c3",
  "gradcam_expires_at": "2026-06-23T12:00:00Z",
  "low_confidence_flag": false,
  "human_review_queued": false,
  "model_version": "efficientnet_b3_v1.0.0",
  "baseline_comparison": {"resnet18_top1": 0.874, "delta": "+0.073"},
  "inference_ms": 87
}
```

---

## Changelog

| Version | Date | Notes |
|---|---|---|
| v1.0.0 | TBD | Initial training on Products-10K |
| v1.0.0-stub | 2026-06-23 | Model card created, training pending |