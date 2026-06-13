# Finding 19: Initial Model Training — Production Pipeline Results

**Date**: 2026-06-13  
**Phase**: Initial Production Pipeline (model/pipeline.py)  
**Runtime**: 213.5 seconds  

---

## 1. Pipeline Architecture

### Overview
Built an end-to-end production pipeline (`model/pipeline.py`, 891 lines) that automates:
1. Data loading & validation
2. Feature engineering (14 features)
3. Hyperparameter tuning via Optuna (100 trials)
4. Stratified 5-Fold cross-validation
5. Final model training with SMOTE(0.3)
6. Evaluation report generation (7 plots + CSV exports)
7. SHAP explainability analysis (KernelExplainer)
8. Model serialization (joblib)

### Prediction Script
`model/predict.py` — standalone inference script for scoring new data:
```
python model/predict.py new_data.csv --threshold 0.947
```

---

## 2. Model Configuration

### Algorithm: XGBoost (Gradient Boosted Trees)
- Chosen for: best F1 in Phase 5 research (0.636 vs RF=0.582, LightGBM=0.605)
- Native missing value handling
- Built-in regularization (gamma, L1/L2)

### Optuna-Tuned Hyperparameters (100 trials)

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_estimators | 335 | Number of boosting rounds |
| max_depth | 10 | Maximum tree depth |
| learning_rate | 0.0767 | Step size shrinkage |
| min_child_weight | 1 | Minimum sum of instance weight |
| subsample | 0.894 | Row sampling ratio |
| colsample_bytree | 0.751 | Feature sampling ratio |
| gamma | 0.396 | Minimum loss reduction for split |
| reg_alpha | 0.340 | L1 regularization |
| reg_lambda | 0.027 | L2 regularization |
| scale_pos_weight | 111.12 | Class imbalance compensation |

### Imbalance Handling
- **SMOTE ratio**: 0.3 (minority upsampled to 30% of majority)
- **SMOTE k_neighbors**: 3 (reduced from default 5 due to small minority class)
- Applied INSIDE each CV fold to prevent data leakage

---

## 3. Feature Engineering

### 14 Features Total (8 Raw + 6 Engineered)

#### Raw Features (from Forward Feature Selection)
| Feature | Interpretation | Importance (Gain) |
|---------|---------------|-------------------|
| F3898 | Account balance/status indicator | **0.42** (dominant) |
| F1819 | Transaction value metric | 0.03 |
| F3799 | Financial activity score | 0.06 |
| F1165 | Transaction frequency metric | 0.04 |
| F1813 | Payment pattern indicator | 0.05 |
| F3806 | Account activity score | 0.04 |
| F162 | Binary behavioral flag | 0.12 |
| F3800 | Account utilization metric | 0.01 |

#### Engineered Features
| Feature | Formula | Rationale |
|---------|---------|-----------|
| F162_div_F3898 | F162 / F3898 | Interaction between top 2 features |
| max_value_top8 | max(STABLE_8) | Peak value across stable features |
| F3898_div_F3805 | F3898 / F3805 | Balance ratio feature |
| missing_count_F0_F500 | count NULLs in F0-F499 | Mules have 97.5% coverage vs 63% |
| F3811_div_F3805 | F3811 / F3805 | Transaction ratio |
| F3898_div_F3811 | F3898 / F3811 | Balance-to-transaction ratio |

### Data Cleaning
- Bracket-wrapped values (e.g., `[8.68012E-1]`) cleaned via `_clean_col()` method
- `pd.to_numeric(errors='coerce')` for safe numeric conversion
- All features cast to `np.float64` to prevent SHAP compatibility issues

---

## 4. Cross-Validation Results

### Stratified 5-Fold CV (honest evaluation)

| Fold | F1 | Precision | Recall | AUC |
|------|-----|-----------|--------|-----|
| 1 | 0.5405 | 0.4762 | 0.6250 | 0.9602 |
| 2 | 0.6154 | 0.5455 | 0.7059 | 0.9877 |
| 3 | 0.5556 | 0.5000 | 0.6250 | 0.9875 |
| 4 | 0.6111 | 0.5500 | 0.6875 | 0.9724 |
| 5 | 0.4615 | 0.3913 | 0.5625 | 0.9491 |
| **MEAN** | **0.5568** | **0.4926** | **0.6412** | **0.9714** |
| Std | ±0.058 | ±0.061 | ±0.052 | ±0.016 |

### Key Observations
- **AUC is excellent** (0.9714) — the model ranks suspicious accounts very well
- **F1 variance** is moderate (0.46-0.62 across folds) — expected with only 16 positives per fold
- **Fold 5 is weakest** (F1=0.46) — likely due to harder-to-detect mules in that split

---

## 5. Threshold Analysis

### Three Operating Points

| Threshold | F1 | Precision | Recall | TP | FP | FN | Business Cost |
|-----------|-----|-----------|--------|----|----|-----|---------------|
| **0.200** (cost-optimal) | 0.470 | 0.349 | 0.716 | 58 | 108 | 23 | $240,800 |
| **0.500** (balanced) | 0.556 | 0.491 | 0.642 | 52 | 54 | 29 | $295,400 |
| **0.947** (F1-optimal) | **0.638** | **0.772** | 0.543 | 44 | 13 | 37 | $371,300 |

### Recommendation
- **For production deployment**: Use t=0.947 (highest precision, fewer false alarms)
- **For investigation queues**: Use t=0.2 (catches 72% of mules, generates 166 alerts)
- **Tiered approach**: Flag all accounts with score > 0.2, prioritize by risk score

---

## 6. SHAP Explainability

### Compatibility Issue & Resolution
- **Problem**: SHAP 0.49.1 `TreeExplainer` is incompatible with XGBoost 3.2.0
  - XGBoost 3.x stores tree node values in bracket format `[4.997221E-1]`
  - SHAP's tree parser fails with `ValueError: could not convert string to float`
- **Solution**: Switched to `KernelExplainer` (model-agnostic, ~2 seconds for 90 accounts)

### SHAP Findings
- **F3898** dominates: low values (0-1) strongly predict mule accounts
- **F1165** is second: high values push prediction toward suspicious
- **F162_div_F3898** (engineered): provides meaningful lift despite lower raw importance
- **missing_count_F0_F500**: low missing count is a mule indicator
- Top 5 mule accounts all have `F3898 = 0 or 1` combined with high F1165 values

---

## 7. Generated Artifacts

### Model Files (model/)
| File | Size | Description |
|------|------|-------------|
| xgboost_model.joblib | 598 KB | Trained model |
| feature_engineer.joblib | 5 KB | Feature pipeline |
| model_metadata.json | 1.4 KB | Params & metrics |
| pipeline.py | 38.5 KB | Full training code |
| predict.py | 5.7 KB | Inference script |

### Evaluation Files (model/evaluation/)
| File | Description |
|------|-------------|
| confusion_matrices.png | 3 thresholds side-by-side |
| roc_pr_curves.png | ROC (AUC=0.972) + PR (AP=0.600) |
| feature_importance.png | XGBoost gain + feature direction |
| risk_distribution.png | Score density + risk tiers |
| threshold_analysis.png | F1/cost/TP-FP-FN curves |
| shap_summary.png | Beeswarm plot (all 14 features) |
| shap_top_accounts.png | Per-account explanations (top 5) |
| risk_scores.csv | 9,082 accounts with scores |
| flagged_accounts.csv | 166 flagged at t=0.2 |
| classification_report.txt | Full sklearn reports |
| evaluation_summary.md | Metrics summary |

---

## 8. Known Limitations

1. **Small minority class**: Only 81 suspicious accounts → high F1 variance across folds
2. **SMOTE artifacts**: Synthetic samples may not perfectly represent real mule patterns
3. **Feature anonymization**: Cannot verify if features make business sense
4. **Temporal validation**: No time-based train/test split (data appears to be Oct 2024 snapshot)
5. **SHAP approximation**: KernelExplainer uses sampling (nsamples=200) rather than exact tree-based computation

---

## 9. Risk Tier Distribution

| Risk Tier | Score Range | Count | % of Total |
|-----------|-------------|-------|------------|
| Very High | > 0.8 | 77 | 0.85% |
| High | 0.5 - 0.8 | 29 | 0.32% |
| Medium | 0.2 - 0.5 | 60 | 0.66% |
| Low | 0.1 - 0.2 | 43 | 0.47% |
| Minimal | < 0.1 | 8,873 | 97.70% |
