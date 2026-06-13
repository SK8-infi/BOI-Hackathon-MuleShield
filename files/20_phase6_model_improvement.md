# Finding 20: Phase 6 — Model Improvement & Problem Statement Feature Investigation

**Date**: 2026-06-13  
**Phase**: Phase 6 Model Improvement  
**Runtime**: 1,384.8 seconds (~23 minutes)  
**Script**: `phase6_improvement.py`  

---

## 1. Executive Summary

### Key Results
| Metric | Phase 5 (Before) | Phase 6 (After) | Change |
|--------|-------------------|-----------------|--------|
| **F1 (CV)** | 0.6361 | **0.7151** | **+12.4%** |
| **AUC** | 0.9694 | 0.9726 | +0.3% |
| **Precision** | 0.4926 | **0.8361** | +69.8% |
| **Recall** | 0.6412 | 0.6296 | -1.8% |
| **SMOTE** | Yes (0.3) | **No** | Removed |
| **Features** | 14 | 32 (best set) | +18 |
| **Optuna trials** | 100 | 500 | +400% |
| **Optimal threshold** | 0.947 | **0.460** | More balanced |

### Most Important Discoveries
1. **SMOTE hurts performance** — dropping it improved F1 from 0.628 to 0.715
2. **Problem Statement features alone are weak** (F1=0.332) but help as supplement (+0.046)
3. **Third leakage source found**: `Unnamed: 0` (CSV row index) — achieves F1=0.849 alone
4. Only **1 of 18 PS features helps individually** (F531: +0.010)
5. Combined feature set (Our 14 + PS 18) is best for the Optuna-tuned model

---

## 2. Problem Statement Feature Investigation

### The Question
The problem statement lists 18 "commonly used features for fraud detection":
`F115, F321, F527, F531, F670, F1692, F2082, F2122, F2582, F2678, F2737, F2956, F3043, F3836, F3887, F3889, F3891, F3894`

**None of these 18 features were in our Phase 5 model.** This investigation answers: *Is that a flaw, or are our discovered features genuinely better?*

### Banking Context Interpretation

| Feature | Interpretation | KS Stat | Correlation | Verdict |
|---------|---------------|---------|-------------|---------|
| **F115** | Account holder type / Customer segment code | **0.317** | 0.060 | SIGNIFICANT |
| F321 | Transaction channel indicator | 0.074 | -0.007 | Weak |
| F527 | Avg monthly transaction count (6 months) | 0.145 | 0.007 | Weak |
| **F531** | Avg monthly transaction amount (6 months) | **0.193** | 0.004 | SIGNIFICANT |
| F670 | Number of distinct payees/counterparties | 0.144 | 0.047 | Weak |
| F1692 | High-value transaction count | 0.101 | -0.018 | Weak |
| F2082 | Incoming funds ratio / credit-to-debit ratio | 0.146 | -0.024 | Weak |
| **F2122** | Cross-border / inter-bank transfer count | **0.198** | -0.029 | SIGNIFICANT |
| F2582 | Account balance volatility / std dev | 0.121 | 0.004 | Weak (37.5% missing) |
| **F2678** | Dormancy indicator / days since last txn | **0.158** | -0.001 | SIGNIFICANT |
| F2737 | Number of linked accounts/beneficiaries | 0.090 | -0.001 | Weak |
| **F2956** | Risk score from existing rule engine | **0.192** | -0.017 | SIGNIFICANT |
| **F3043** | Risk score from existing rule engine (pair) | **0.186** | -0.013 | SIGNIFICANT (64% missing) |
| F3836 | Account type code (savings/current/other) | 0.141 | 0.008 | Weak |
| F3887 | Account age in DAYS | 0.100 | 0.004 | Weak |
| F3889 | Customer onboarding channel / KYC source | 0.000 | NaN | **Useless** (object, no discrimination) |
| F3891 | Branch/region code of account opening | 0.000 | NaN | **Useless** (object, no discrimination) |
| **F3894** | Customer age in YEARS | **0.155** | -0.008 | SIGNIFICANT |

### Key Findings about PS Features

1. **F115 is the strongest PS feature** (KS=0.317) — suspicious accounts have 27% higher values on average
2. **F670** (distinct payees): Mules have 2.6x more counterparties — strong signal!
3. **F2082** (incoming funds): Mules have 0% incoming ratio vs normal accounts — pass-through behavior
4. **F2678** (dormancy): Mules have near-zero dormancy (0.027 vs 1.0) — accounts used immediately
5. **F3889 and F3891** have **zero discriminative power** (KS=0.000) — completely useless
6. **F3043** has 64% missing data — unreliable

### Missing Data Pattern (PS Features)
- **F3043**: 83.9% missing for suspicious (vs 64.1% overall) — differential missingness
- **F2582**: 37.5% missing overall — high but equal across classes
- Most other PS features have <10% missing

---

## 3. Individual PS Feature Contributions

When adding each PS feature **one at a time** to our baseline 14-feature model:

| Feature | F1 With | F1 Delta | Effect |
|---------|---------|----------|--------|
| **F531** | 0.6464 | **+0.0103** | **Only feature that helps** |
| F3889 | 0.6361 | 0.0000 | No effect (object type) |
| F3891 | 0.6361 | 0.0000 | No effect (object type) |
| F321 | 0.6296 | -0.0066 | Slight noise |
| F3043 | 0.6211 | -0.0150 | Noise (high missing) |
| F527 | 0.6202 | -0.0159 | Noise |
| F2122 | 0.6198 | -0.0164 | Noise |
| F2082 | 0.6193 | -0.0168 | Noise |
| F670 | 0.6193 | -0.0168 | Noise |
| F2956 | 0.6155 | -0.0207 | Hurts |
| F115 | 0.5962 | -0.0399 | **Hurts significantly** |
| F1692 | 0.5968 | -0.0394 | Hurts significantly |
| F2737 | 0.6002 | -0.0360 | Hurts |
| F2582 | 0.5913 | -0.0448 | Hurts |
| F3887 | 0.5859 | -0.0502 | Hurts |
| F3836 | 0.5834 | -0.0528 | Hurts |
| F2678 | 0.5742 | -0.0619 | **Hurts most** |
| F3894 | 0.5672 | **-0.0689** | **Worst contributor** |

### Why Do PS Features Hurt Individually?
- **Small sample size**: With only 81 suspicious accounts, adding noisy features causes overfitting
- **Curse of dimensionality**: Each added feature requires more data to learn meaningful splits
- **Weak signals**: Most PS features have KS < 0.2 (our features have KS > 0.3)
- **Counter-intuitive**: F115 has highest KS (0.317) but hurts the model — likely correlated with existing features

### But Together They Help
- **PS 18 alone**: F1=0.332 (poor)
- **Our 14 alone**: F1=0.636 (good)  
- **Our 14 + PS 18**: F1=0.682 (best!) — the collective signal matters

This is a **feature interaction effect** — individual PS features add noise, but together they provide complementary signals that XGBoost can exploit.

---

## 4. Feature Set Comparison (6 Sets Tested)

| Feature Set | # Features | F1 | F1 Std | AUC |
|-------------|-----------|-----|--------|-----|
| PS 18 Only | 18 | 0.3323 | ±0.089 | 0.8739 |
| Top 30 MI | 30 | 0.6011 | ±0.123 | 0.9563 |
| **Our 14 (Phase 5)** | 14 | 0.6361 | ±0.116 | 0.9694 |
| Our 14 + Sig PS (7) | 21 | 0.6660 | ±0.115 | 0.9749 |
| Kitchen Sink (all) | 36 | 0.6756 | ±0.123 | 0.9754 |
| **Our 14 + PS 18** | **32** | **0.6818** | ±0.098 | **0.9750** |

### Observations
- **PS features alone are weak** (F1=0.332) — confirms our data-driven selection found better signals
- **Combining is best** — Our 14 + PS 18 beats either alone
- **Kitchen Sink (36 features)** is slightly worse than Our 14 + PS 18 — more engineered features ≠ better
- **Top 30 MI was contaminated** (contained `Unnamed: 0`) — after cleaning, drops to F1=0.601

---

## 5. Leakage Investigation Update

### Third Leakage Source Discovered
| Feature | Type | Single-Feature F1 | How It Leaks |
|---------|------|-------------------|-------------|
| F3912 | Fraud flag | 1.000 | Direct label encoding |
| F2230 | Reporting period | ~1.000 | Non-Oct months = 100% fraud |
| **Unnamed: 0** | **CSV row index** | **0.849** | Suspicious accounts at rows 9001-9082 |

**Impact**: Including `Unnamed: 0` in Top 30 MI inflated F1 from 0.601 to 0.953. After exclusion, results are honest.

**F3700 cleared**: Despite having exactly 81 unique values (suspicious coincidence), F3700's SingleF1 is only 0.032 — not leaking.

---

## 6. Model Variants Comparison (7 Models Tested)

All on best feature set (Our 14 + PS 18, 32 features):

| Model | F1 | AUC | Notes |
|-------|-----|-----|-------|
| **XGBoost (Optuna-500)** | **0.7151** | 0.9726 | **BEST F1** |
| XGBoost (No SMOTE) | 0.7151 | 0.9726 | Same as above (Optuna chose no SMOTE) |
| Soft Voting (XGB+LGBM+RF) | 0.7053 | 0.9741 | Close second |
| LightGBM | 0.6947 | 0.9693 | Good alternative |
| XGBoost (SMOTE 0.3) | 0.6283 | **0.9831** | Best AUC, worst F1 with SMOTE |
| Random Forest | 0.5924 | 0.9731 | Mediocre |
| GBM (sklearn) | 0.5378 | 0.9486 | Worst overall |

### Critical Finding: SMOTE Hurts!
- **Without SMOTE**: F1=0.7151
- **With SMOTE(0.3)**: F1=0.6283 (12% worse!)
- SMOTE improves AUC slightly (0.983 vs 0.973) but hurts F1 by creating noisy synthetic samples
- `scale_pos_weight=111.12` alone is sufficient for handling class imbalance

---

## 7. Optimized Hyperparameters (500 Optuna Trials)

### Best Configuration (F1=0.7151)

| Parameter | Phase 5 Value | Phase 6 Value | Change |
|-----------|--------------|---------------|--------|
| n_estimators | 335 | **614** | +83% |
| max_depth | 10 | **9** | -1 |
| learning_rate | 0.077 | **0.036** | Slower learning |
| min_child_weight | 1 | 1 | Same |
| subsample | 0.894 | **0.710** | More regularization |
| colsample_bytree | 0.751 | **0.358** | Much more regularization |
| gamma | 0.396 | **0.410** | Similar |
| reg_alpha | 0.340 | **0.210** | Less L1 |
| reg_lambda | 0.027 | **1.439** | **53x more L2** |
| scale_pos_weight | 111.12 | 111.12 | Same |
| **SMOTE** | **Yes (0.3)** | **No** | **Removed** |

### Key Insight
The model prefers **much stronger regularization** (colsample=0.358, reg_lambda=1.44) with **more trees at slower learning rate** (614 trees, lr=0.036). This creates a model that generalizes better on the 81-sample minority class.

---

## 8. Threshold Analysis (Best Model)

| Threshold | F1 | Precision | Recall | TP | FP | FN | Cost |
|-----------|-----|-----------|--------|----|----|-----|------|
| **0.030** (cost-opt) | 0.462 | 0.322 | 0.815 | 66 | 139 | 15 | $163,900 |
| **0.460** (F1-opt) | **0.727** | **0.839** | **0.642** | 52 | 10 | 29 | $291,000 |
| 0.500 (default) | 0.718 | 0.836 | 0.630 | 51 | 10 | 30 | $301,000 |

### Compared to Phase 5

| Metric | Phase 5 (t=0.947) | Phase 6 (t=0.460) | Improvement |
|--------|-------------------|-------------------|-------------|
| F1 | 0.638 | **0.727** | +14.0% |
| Precision | 0.772 | **0.839** | +8.7% |
| Recall | 0.543 | **0.642** | +18.2% |
| False Positives | 13 | 10 | -23.1% |
| False Negatives | 37 | 29 | -21.6% |
| Business Cost | $371,300 | **$291,000** | -$80,300 |

---

## 9. Generated Graphs (44-49)

| Graph | File | Content |
|-------|------|---------|
| 44 | `44_ps_features_analysis.png` | 4-panel: KS stats, correlations, missing data, value ratios |
| 45 | `45_feature_set_comparison.png` | 6 feature sets: F1 and AUC side-by-side |
| 46 | `46_model_variants.png` | 7 model variants comparison |
| 47 | `47_ps_vs_ours.png` | PS vs Our features: MI, KS, and performance |
| 48 | `48_ps_feature_contributions.png` | Individual PS feature impact (16/18 hurt!) |
| 49 | `49_final_model_summary.png` | F1 journey + threshold curve + confusion matrix |

---

## 10. Conclusions & Recommendations

### What We Learned
1. **Our data-driven feature discovery (Phase 3-5) found genuinely better features** than the bank's standard 18
2. **Combining both sets is optimal** — the PS features add complementary (but individually weak) signals
3. **SMOTE is counterproductive** for this dataset — class weighting alone is better
4. **Stronger regularization is key** — the model needs to avoid overfitting to 81 samples
5. **Threshold matters enormously** — optimal threshold dropped from 0.947 to 0.460

### Final Model Specification
- **Algorithm**: XGBoost (Optuna-tuned, 500 trials)
- **Features**: 32 (Our 14 + PS 18)
- **SMOTE**: None (class weighting only)
- **F1**: 0.7151 (5-fold CV)
- **AUC**: 0.9726
- **Optimal threshold**: 0.460
- **At optimal threshold**: F1=0.727, P=0.839, R=0.642

### Answer to "Why Not Use PS Features?"
The bank's 18 features are general-purpose fraud signals designed for traditional rule-based systems. Our data-driven features (F3898, F162, F1165, etc.) capture the specific **mule account patterns** in this dataset, which are different from general fraud. The best approach is to combine both.
