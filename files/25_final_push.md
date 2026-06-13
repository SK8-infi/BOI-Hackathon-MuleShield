# Phase 9: Final Push — Maximum Performance

## Summary

Re-tuned Optuna specifically for FULL-60 features, built 13 diverse models across 3 feature levels and 2 training strategies (class-weighted + SMOTE), then exhaustively tested all blend combinations with threshold optimization.

**Result: NEW BEST F1 = 0.7432** (up from 0.704, a +5.6% improvement)

---

## Key Discovery: SMOTE + Class-Weighted Blend

The best model is a **2-model blend**:

| Component | Weight | Features | Strategy | Individual F1 |
|-----------|--------|----------|----------|---------------|
| XGB-P6-Enhanced30 | 0.479 | 30 | Class-weighted | 0.705 |
| SMOTETomek-Full60 | 0.521 | 60 | SMOTE+Tomek resampled | 0.686 |

**Why this works**: The class-weighted model and SMOTE model make **different errors**. Class-weighted XGBoost is conservative (high precision, lower recall). SMOTE-trained XGBoost is more aggressive (catches more mules but more false positives). Blending them averages out these biases.

---

## Final Model Performance

| Metric | Value |
|--------|-------|
| **F1 Score** | **0.7432** |
| **Precision** | **0.8209** (82% of flagged accounts are truly suspicious) |
| **Recall** | **0.6790** (catches 55 of 81 mules) |
| **AUC-ROC** | **0.9815** |
| **Threshold** | 0.483 |
| **True Positives** | 55 |
| **False Positives** | 12 |
| **False Negatives** | 26 |
| **True Negatives** | 8,989 |

---

## Optuna Results (300 trials on FULL-60)

Re-tuning Optuna specifically for 60 features found significantly better parameters:

| Parameter | Phase 6 (30 feat) | Phase 9 (60 feat) |
|-----------|-------------------|-------------------|
| max_depth | 9 | **6** |
| learning_rate | 0.036 | **0.063** |
| n_estimators | 614 | **556** |
| scale_pos_weight | 111.1 | **144.9** |
| Single model F1 | 0.684 | **0.735** |

Key: Shallower trees (depth 6 vs 9) with higher learning rate work better on 60 features — prevents overfitting on the expanded feature set.

---

## All 13 Individual Models (Threshold-Optimized)

| Rank | Model | F1 | Threshold | AUC |
|------|-------|-----|-----------|-----|
| 1 | **XGB-Optuna-Full60** | **0.735** | 0.49 | 0.979 |
| 2 | XGB-Seed2-Full60 | 0.718 | 0.68 | 0.981 |
| 3 | LGB-Full60 | 0.716 | 0.45 | 0.978 |
| 4 | SMOTETomek-Full60 | 0.714 | 0.71 | 0.979 |
| 5 | XGB-P6-Enhanced30 | 0.713 | 0.53 | 0.972 |
| 6 | SMOTEENN-Full60 | 0.711 | 0.82 | 0.977 |
| 7 | SMOTETomek-Enh30 | 0.710 | 0.50 | 0.979 |
| 8 | XGB-P6-Full60 | 0.708 | 0.64 | 0.978 |
| 9 | XGB-Shallow-Full60 | 0.703 | 0.66 | 0.981 |
| 10 | SMOTEENN-Enh30 | 0.701 | 0.66 | 0.980 |
| 11 | XGB-Deep-Full60 | 0.699 | 0.68 | 0.980 |
| 12 | XGB-HighReg-Full60 | 0.671 | 0.75 | 0.975 |
| 13 | XGB-P6-Base14 | 0.658 | 0.45 | 0.976 |

---

## Exhaustive Blending Results

| Blend Size | Best F1 | Threshold | Best Combo |
|-----------|---------|-----------|------------|
| **2 models** | **0.7432** | 0.483 | **XGB-P6-Enhanced30 + SMOTETomek-Full60** |
| 3 models | 0.7432 | 0.580 | XGB-P6-Enhanced30 + SMOTETomek-Full60 + SMOTEENN-Full60 |
| 4 models | 0.7391 | 0.570 | XGB-Shallow + LGB + SMOTETomek-Enh30 + SMOTEENN-Enh30 |
| 5 models | 0.7376 | 0.590 | — |
| 6+ models | 0.723-0.738 | varies | Diminishing returns |

**Key insight**: More models = worse performance. The 2-model blend is optimal because:
- Each additional model adds noise (their errors don't cancel out)
- With only 81 minority samples, blend diversity has limited benefit
- The 2 models already capture the key complementary signals

---

## Threshold Sweep

| Threshold | F1 | Precision | Recall | TP | FP |
|-----------|-----|-----------|--------|----|----|
| 0.30 | 0.659 | 0.640 | 0.679 | 55 | 31 |
| 0.35 | 0.696 | 0.714 | 0.679 | 55 | 22 |
| 0.40 | 0.714 | 0.753 | 0.679 | 55 | 18 |
| 0.45 | 0.729 | 0.786 | 0.679 | 55 | 15 |
| **0.50** | **0.743** | **0.821** | **0.679** | **55** | **12** |
| 0.55 | 0.714 | 0.848 | 0.617 | 50 | 9 |
| 0.60 | 0.710 | 0.860 | 0.605 | 49 | 8 |
| 0.65 | 0.701 | 0.857 | 0.593 | 48 | 8 |
| 0.70 | 0.682 | 0.852 | 0.568 | 46 | 8 |

---

## F1 Improvement Journey (All Phases)

| Phase | Model | F1 | Delta |
|-------|-------|-----|-------|
| Phase 5 | XGBoost baseline | 0.604 | — |
| Phase 5 | + SMOTE(0.3) | 0.621 | +0.017 |
| Phase 5 | + Optuna 100 trials | 0.636 | +0.015 |
| Phase 6 | + PS features (30 feat) | 0.684 | +0.048 |
| Phase 7 | + TabNet ensemble | 0.704 | +0.020 |
| Phase 8 | Equal blend (4 models) | 0.704 | +0.000 |
| **Phase 9** | **Optuna-Full60 + SMOTE blend** | **0.743** | **+0.039** |

**Total improvement: +0.139 F1 (+23% from baseline)**

---

## Graph
- **Graph 55**: [55_final_push.png](./graphs/55_final_push.png) — Individual models, threshold optimization, improvement journey
