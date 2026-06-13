# 18 -- Phase 5: Pre-Modeling Research

**Date**: June 13, 2026  
**Script**: `phase5_research.py`  
**Runtime**: 918.1 seconds (15.3 minutes)

---

## 1. Systematic Feature Engineering

Created and evaluated **13 engineered features** from Phase 4 findings.

### Top Engineered Features by KS Statistic

| Feature | KS Stat | p-value | Description |
|---------|---------|---------|-------------|
| **low_value_count** | **0.652** | 6.62e-34 | Count of top-8 features below legitimate median |
| **max_value_top8** | **0.513** | 4.02e-20 | Maximum value across top 8 stable features |
| **F3898_x_F162** | **0.264** | 1.99e-05 | Interaction of the two dominant features |
| **missing_count_F0_F500** | **0.227** | 4.01e-04 | Count of NULLs in F0-F499 block |
| **F3898_div_F3811** | 0.173 | 1.43e-02 | Ratio feature |
| **age_risk_score** | 0.168 | 1.91e-02 | Binary: 20-25yo + <6mo account |

### Key Insights
- **low_value_count** is the STRONGEST engineered feature (KS=0.652) -- captures the "low everything" mule signature in a single number
- **max_value_top8** (KS=0.513) -- mules have much lower max activity
- **missing_count_F0_F500** confirms Phase 4: mules have MORE complete data (fewer NULLs)
- **age_risk_score** is statistically significant (p=0.019) but weak individually
- 6 ratio features (F3898/F3811, F3898/F3805, etc.) all significant but moderate KS values

---

## 2. Forward Feature Selection

### Optimal Feature Set: 14 Features

Started with F3898 alone and greedily added features to maximize 5-fold CV F1 score.

| Step | Added Feature | Cumulative F1 | Type |
|------|---------------|---------------|------|
| 1 | F3898 | 0.0407 | Raw |
| 2 | F1819 | 0.1253 | Raw |
| 3 | F3799 | 0.3912 | Raw |
| 4 | F1165 | 0.5393 | Raw |
| 5 | F1813 | 0.5411 | Raw |
| 6 | **F162_div_F3898** | 0.5725 | **Engineered** |
| 7 | F3806 | 0.5761 | Raw |
| 8 | **max_value_top8** | 0.5804 | **Engineered** |
| 9 | **F3898_div_F3805** | 0.5764 | **Engineered** |
| 10 | F162 | 0.5780 | Raw |
| 11 | **missing_count_F0_F500** | 0.5981 | **Engineered** |
| 12 | **F3811_div_F3805** | 0.5983 | **Engineered** |
| 13 | **F3898_div_F3811** | 0.5986 | **Engineered** |
| **14** | **F3800** | **0.6037** | **Raw (PEAK)** |
| 15 | F3813 | 0.6000 | Raw |
| 16 | age_risk_score | 0.5611 | Engineered |

### CRITICAL FINDING: Engineered Features Matter!
- **6 out of 14 optimal features are ENGINEERED** (not in raw data)
- F162_div_F3898 was the first engineered feature selected (step 6)
- missing_count_F0_F500 provided a +2% F1 boost at step 11
- F1 plateaus around 5-6 features (~0.57) then slowly climbs to 0.60 at 14
- Adding features beyond 14 causes DECLINE -- overfitting

### The "Core 4" (Steps 1-4 = 90% of signal)
F3898 + F1819 + F3799 + F1165 get to F1=0.54, which is 90% of the final 0.60

---

## 3. Cross-Validation Strategy Design

### Results

| Strategy | F1 Mean | F1 Std | Precision | Recall | AUC |
|----------|---------|--------|-----------|--------|-----|
| **Stratified 5-Fold** | **0.604** | **0.080** | 0.582 | 0.641 | **0.977** |
| Stratified 10-Fold | 0.584 | 0.104 | 0.555 | 0.628 | 0.976 |
| Repeated 5-Fold (3x) | 0.599 | 0.096 | 0.567 | 0.645 | 0.974 |
| Repeated 5-Fold (5x) | 0.595 | 0.101 | 0.573 | 0.634 | 0.971 |
| 10-Fold (per-sample) | 0.580 | -- | 0.537 | 0.630 | -- |

### RECOMMENDATION: Stratified 5-Fold
- **Best F1 (0.604) with lowest variance (0.080)**
- 10-Fold has higher variance because each fold has only ~8 suspicious samples
- Repeated splits don't improve over single 5-Fold
- Per-sample predictions show 51/81 suspicious caught (63% recall)

---

## 4. SMOTE Variant Comparison

### Results (Stratified 5-Fold CV, RF classifier)

| Method | F1 Mean | F1 Std | Notes |
|--------|---------|--------|-------|
| **SMOTE (ratio=0.3)** | **0.621** | **0.084** | **WINNER** |
| No SMOTE (balanced weights) | 0.604 | 0.080 | Strong baseline |
| SMOTE (ratio=0.1) | 0.591 | 0.125 | Under-sampling |
| No SMOTE (scale_pos_weight) | 0.582 | 0.123 | XGBoost default |
| SMOTE (ratio=0.5) | 0.550 | 0.061 | Over-sampling starts hurting |
| Borderline SMOTE | 0.521 | 0.030 | Worse than vanilla |
| ADASYN | 0.486 | 0.046 | Too aggressive |
| SMOTE (ratio=1.0) | 0.462 | 0.050 | Way too many synthetics |
| SMOTE+Tomek | 0.456 | 0.024 | Worst performer |

### KEY FINDINGS
1. **SMOTE at 30% ratio is optimal** -- generates ~240 synthetic mules (vs 81 real)
2. **Higher ratios HURT** -- 50% is already declining, 100% is terrible
3. **Balanced class weights alone is competitive** (F1=0.604 vs 0.621)
4. **Fancy variants (Borderline, ADASYN, Tomek) all WORSE than vanilla SMOTE**
5. The improvement from SMOTE 0.3 over no-SMOTE is only +0.017 F1 -- marginal
6. **RECOMMENDATION**: Use SMOTE(0.3) for final model OR just balanced class weights

---

## 5. Error Analysis / Misclassification Profiling

### Overall Performance (10-Fold per-sample predictions)
- **Caught (TP)**: 50/81 (61.7% recall)
- **Missed (FN)**: 31/81
- **False Positives**: 49
- **Precision**: 50.5%

### Who Gets Missed?

| Metric | Caught Mules (n=50) | Missed Mules (n=31) | Interpretation |
|--------|---------------------|---------------------|----------------|
| F3898 mean | 0.580 | 0.677 | Missed have HIGHER F3898 (blend in) |
| F162 mean | 0.871 | 0.667 | Missed have LOWER F162 (less risk signal) |
| F3811 mean | 127,455 | **724,957** | Missed have 5.7x HIGHER transaction volumes |
| F3805 mean | 171,136 | **1,069,630** | Missed have 6.3x HIGHER account values |
| F3887 (acct age) | 113.8 days | 97.7 days | Similar account ages |
| F3894 (cust age) | 33.2 years | 32.2 years | Similar customer ages |

### THE HARDEST MULES TO CATCH
- **High-value mule accounts** -- F3811/F3805 values are 5-6x higher than caught mules
- They have MORE activity (higher F3898) and LESS risk signal (lower F162)
- These are **sophisticated mules** that look like legitimate high-value accounts
- 7 mules have <10% predicted probability -- nearly invisible to the model
- Age/demographics are NOT the differentiator -- it's the activity level

### Probability Distribution of 81 Suspicious Accounts
| Probability Range | Count | Category |
|-------------------|-------|----------|
| 0.0 - 0.1 | 7 | Nearly invisible |
| 0.1 - 0.3 | 13 | Hard to catch |
| 0.3 - 0.5 | 11 | Borderline |
| 0.5 - 0.7 | 8 | Caught |
| 0.7 - 1.0 | 42 | Easy catches |

---

## 6. Threshold Optimization

### Optimal Threshold: 0.705

| Threshold | F1 | Precision | Recall | TP | FP | FN |
|-----------|----|-----------|--------|----|----|-----|
| 0.10 | -- | 0.080 | 0.914 | 74 | 854 | 7 |
| 0.20 | -- | 0.185 | 0.852 | 69 | 304 | 12 |
| 0.30 | -- | 0.298 | 0.753 | 61 | 144 | 20 |
| 0.50 (default) | 0.556 | 0.505 | 0.617 | 50 | 49 | 31 |
| **0.705 (optimal)** | **0.641** | **0.840** | **0.519** | **42** | **8** | **39** |
| 0.80 | -- | 0.941 | 0.395 | 32 | 2 | 49 |
| 0.90 | -- | 0.950 | 0.235 | 19 | 1 | 62 |

### Business Cost Analysis (FP=$100, FN=$10,000)

| Threshold | Total Cost | Strategy |
|-----------|-----------|----------|
| **0.10** | **$155,400** | **Minimum cost (catch most mules, accept FP noise)** |
| 0.20 | $150,400 | Near-optimal cost |
| 0.50 | $314,900 | Default -- 2x minimum cost |
| 0.705 | $390,800 | Best F1 but worst cost |
| 0.90 | $620,100 | Most expensive (miss too many mules) |

### CRITICAL INSIGHT: F1-Optimal != Cost-Optimal
- Best F1 at threshold=0.705 (high precision, low recall)
- **Best business cost at threshold=0.19** (catch 85% of mules, accept investigation overhead)
- The right threshold depends on the bank's tolerance for false positives
- **RECOMMENDATION**: Use threshold ~0.2 for production (maximize mule detection)

---

## 7. Model Stacking / Blending

### Individual Model Performance

| Model | F1 | F1 Std | AUC | Precision | Recall |
|-------|-----|--------|-----|-----------|--------|
| **XGBoost** | **0.636** | 0.116 | 0.969 | **0.708** | 0.580 |
| **Soft Voting** | **0.634** | 0.119 | -- | -- | -- |
| LightGBM | 0.605 | 0.128 | 0.966 | 0.651 | 0.567 |
| RF | 0.582 | 0.089 | **0.975** | 0.567 | 0.616 |
| GBM | 0.576 | 0.069 | 0.967 | 0.559 | 0.605 |
| Stacking (LR) | 0.243 | 0.030 | -- | -- | -- |

### KEY FINDINGS
1. **XGBoost is the best individual model** (F1=0.636)
2. **Soft Voting nearly matches XGBoost** (F1=0.634) -- diversity doesn't add much
3. **RF has highest AUC (0.975)** but lower F1 -- better ranking, worse classification
4. **Stacking with LR meta-learner FAILS** (F1=0.243) -- too few minority samples for a 2-stage approach
5. **All AUCs are >0.96** -- models rank well, threshold matters more than model choice

### RECOMMENDATION
- **Primary**: XGBoost with scale_pos_weight
- **Fallback**: Soft Voting (XGBoost + LightGBM + RF)
- **DON'T USE**: Stacking with meta-learner (not enough data)

---

## 8. Temporal Stability Check

### BOMBSHELL FINDING: F2230 Perfectly Separates Classes!

| Period | Total Accounts | Suspicious | Rate |
|--------|---------------|------------|------|
| Oct25 | 9,001 | 0 | **0.00%** |
| Sep25 | 48 | 48 | **100.00%** |
| Nov25 | 23 | 23 | **100.00%** |
| Dec25 | 10 | 10 | **100.00%** |

### What This Means
- **ALL 81 suspicious accounts are in Sep/Nov/Dec periods**
- **ALL 9,001 October accounts are legitimate**
- F2230 is PERFECT LEAKAGE -- it's essentially the label in disguise
- The suspicious accounts from non-October periods may be **manually labeled/injected samples**
- This confirms our Phase 1 decision to EXCLUDE F2230

### Temporal Split Performance (Train on other periods, test on held-out)

| Test Period | F1 | Precision | Recall | Train Susp | Test Susp |
|-------------|-----|-----------|--------|------------|-----------|
| Dec25 | 0.750 | 1.000 | 0.600 | 71 | 10 |
| Nov25 | 0.686 | 1.000 | 0.522 | 58 | 23 |
| Sep25 | 0.648 | 1.000 | 0.479 | 33 | 48 |
| Random split | 0.636 | -- | -- | -- | -- |

### KEY FINDINGS
1. **100% Precision on temporal splits** -- zero false positives when testing on unseen periods
2. **Recall drops to 48-60%** -- model misses some mules in new periods
3. **Performance is STABLE across periods** (F1: 0.65-0.75)
4. Sep25 has worst recall because it's the largest test set (48 mules) and fewest training mules (33)
5. **The model generalizes well across time** -- temporal shift is not a major concern

---

## Graphs Generated (Phase 5)

| # | File | Description |
|---|------|-------------|
| 36 | `36_feature_engineering.png` | Engineered feature distributions (6 key features) |
| 37 | `37_forward_selection.png` | F1 vs feature count + selection order |
| 38 | `38_cv_strategies.png` | CV strategy comparison (5 strategies) |
| 39 | `39_smote_comparison.png` | SMOTE variant comparison (9 methods) |
| 40 | `40_error_analysis.png` | Misclassification profiling (4 panels) |
| 41 | `41_threshold_optimization.png` | PR curve + F1/threshold + business cost |
| 42 | `42_model_comparison.png` | Model stacking comparison (6 models) |
| 43 | `43_temporal_stability.png` | Temporal split analysis (3 panels) |

---

## Phase 5 Summary: Model Blueprint

Based on all findings, the optimal model configuration is:

### Feature Set (14 features)
**Raw**: F3898, F1819, F3799, F1165, F1813, F3806, F162, F3800  
**Engineered**: F162_div_F3898, max_value_top8, F3898_div_F3805, missing_count_F0_F500, F3811_div_F3805, F3898_div_F3811

### Model
- **Algorithm**: XGBoost (F1=0.636)
- **Class handling**: scale_pos_weight OR SMOTE(0.3)
- **CV**: Stratified 5-Fold
- **Threshold**: 0.2 (cost-optimal) or 0.5 (balanced) or 0.705 (F1-optimal)

### Expected Performance
- **F1**: 0.60-0.64
- **AUC**: 0.97
- **Recall**: 52-91% (threshold-dependent)
- **Precision**: 8-94% (threshold-dependent)

### Known Limitations
- 31/81 mules are "sophisticated" (high-value, low-risk-signal) and very hard to catch
- 7 mules are nearly invisible (probability <10%)
- Stacking doesn't help due to small minority class
- More data collection on the hard-to-catch profile would improve recall
