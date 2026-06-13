# 12 — Recommendations for ML Pipeline

Based on all research findings, here is the recommended approach for building 
the mule account classification model.

---

## Pre-Processing Pipeline

### Step 1: Column Removal
```
DROP:
├── Unnamed: 0          (row index)
├── F3912               (data leakage — fraud flag)
├── F2230               (data leakage — temporal separator)
├── F3888               (raw date — use F3887 instead)
├── 63 all-null columns (zero information)
├── 359 constant columns (zero information)
└── Total removed: ~425 columns
    Remaining: ~3,500 columns
```

### Step 2: Feature Selection (Reduce from ~3,500 to ~100-300)
Options (in order of preference):
1. **Tree-based importance**: Train a quick XGBoost/LightGBM, extract feature importances, keep top N
2. **Mutual information**: sklearn's `mutual_info_classif` — captures non-linear relationships
3. **Variance threshold**: Remove features with near-zero variance
4. **Correlation filtering**: Among highly correlated pairs, keep the one with higher target correlation

### Step 3: Missing Value Handling
```
IF using XGBoost/LightGBM:
    → These handle missing values natively — minimal imputation needed
    
IF using other models:
    → Numeric: Impute with MEDIAN
    → Categorical: Impute with MODE
    → Consider: "is_missing" binary indicator for columns with 10-90% missing
```

### Step 4: Categorical Encoding
```
F3886 (Account Type):   Target encoding or one-hot (17 categories)
F3889 (Account Scheme):  One-hot encoding (7 categories)
F3890 (Area Code):       One-hot encoding (4 categories)
F3891 (Occupation):      One-hot encoding (7 categories)
F3892 (Gender):          One-hot encoding (3 categories)
F3893 (Customer Segment): Binary encoding (2 categories)
```

### Step 5: Class Imbalance Handling
Options:
1. **SMOTE** (Synthetic Minority Oversampling) — generate synthetic suspicious samples
2. **ADASYN** — adaptive version of SMOTE, focuses on harder-to-learn samples
3. **Class weights** — `scale_pos_weight=111` in XGBoost
4. **Combination**: SMOTE + Tomek links (clean decision boundary)
5. **Threshold tuning**: Adjust classification threshold post-training

---

## Recommended Models

### Primary: XGBoost / LightGBM
**Why**:
- Handles missing values natively
- Handles sparse features well
- Captures non-linear relationships (critical — individual correlations are weak)
- Feature importance built-in
- Fast training, GPU-accelerable
- Works well with class imbalance (via `scale_pos_weight`)

### Secondary: Random Forest
**Why**:
- Robust baseline
- Less prone to overfitting than XGBoost
- Good feature importance via permutation importance

### Tertiary: Neural Network (if time permits)
**Why**:
- Can capture complex interactions
- Auto-encoder approach for anomaly detection
- But: 81 positive samples may be too few for deep learning

### Baseline: Logistic Regression
**Why**:
- Simple, interpretable
- Provides a performance floor
- Works with regularization (L1 for feature selection)

---

## Evaluation Strategy

### Metrics (DO NOT use accuracy)
| Metric | Why |
|--------|-----|
| **F1-Score** | Primary metric — balances precision and recall |
| **AUC-ROC** | Overall discriminative ability |
| **PR-AUC** | Better than ROC for imbalanced data |
| **Precision** | % of flagged accounts that are truly suspicious |
| **Recall** | % of suspicious accounts that are correctly caught |
| **Confusion Matrix** | Understand false positive/negative trade-offs |

### Validation
- **Stratified 5-Fold Cross-Validation**
  - Ensures ~16 positive samples per fold
  - More robust than single train/test split
  - Report mean ± std for all metrics

### Threshold Optimization
- Default threshold (0.5) may not be optimal for imbalanced data
- Use PR curve to find optimal threshold
- Consider business cost: missing a mule account may be more costly than false alarms

---

## Feature Engineering Ideas

### From existing features
1. **Interaction features**: F115 × F670, F2956 × F2122 (combine risk score with activity)
2. **Ratio features**: F2956 / F3887 (transactions per day of account age)
3. **Aggregations**: Mean of proportions (F1–F299), max of deltas (F2500–F2799)
4. **Missingness features**: Count of missing values per row as a feature

### From domain knowledge
1. **Account recency**: How recently the account was opened (from F3887)
2. **Activity intensity**: Transaction count / account age
3. **Channel diversity**: Number of different channels used
4. **Beneficiary concentration**: Are transactions going to few or many recipients?

---

## Suggested Experiments (Priority Order)

| # | Experiment | Purpose |
|---|-----------|---------|
| 1 | XGBoost with all 2,520 usable features + class weights | Baseline with full feature set |
| 2 | XGBoost with top 100 features (by importance from #1) | Reduced, more robust model |
| 3 | XGBoost with 18 suggested features only | Problem statement compliance |
| 4 | LightGBM comparison | Speed and performance comparison |
| 5 | SMOTE + XGBoost | Impact of oversampling |
| 6 | Feature engineering + best model from above | Enhanced features |
| 7 | Ensemble: XGBoost + RF + LR voting | Robust final model |

---

## GPU Utilization Plan

The user has offered GPU resources. Use for:
1. **XGBoost**: `tree_method='gpu_hist'` for GPU-accelerated training
2. **LightGBM**: `device='gpu'` for GPU training
3. **Neural Network**: PyTorch with CUDA for any deep learning experiments
4. **Hyperparameter search**: GPU accelerates grid/random search significantly

### Required packages
```
pip install xgboost lightgbm scikit-learn imbalanced-learn matplotlib seaborn
# For GPU:
pip install xgboost --upgrade  # GPU support built-in
pip install lightgbm --install-option=--gpu
```
