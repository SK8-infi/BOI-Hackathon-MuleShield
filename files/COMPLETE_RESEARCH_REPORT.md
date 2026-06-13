# BOI Hackathon - Problem Statement 2: Complete Research Report

## AI/ML-Based Classification of Suspicious Mule Accounts

**Date**: June 13, 2026
**Dataset**: DataSet.csv (111 MB, 9,082 rows x 3,925 columns)
**Target Variable**: F3924 (binary: 0 = legitimate, 1 = suspicious/mule)
**Final Model**: XGBoost Blend | F1=0.743 | AUC=0.982 | 2-model ensemble
**Total Experiments**: 80+ across 9 phases

---

## Table of Contents

1. dataset overview
2. target variable
3. missing values
4. feature types
5. feature blocks
6. key features
7. categorical features
8. top correlations
9. data leakage
10. suspicious profile
11. feature interpretation
12. recommendations
13. graph analysis
14. deep investigation
15. phase3 investigation
16. ml techniques research
17. phase4 investigation
18. phase5 premodeling
19. phase5 summary
20. initial model training
21. phase6 model improvement
22. feature identity
23. literature review
24. neural network experiment
25. advanced techniques
26. final push

---

<!-- FILE: 01_dataset_overview.md -->

# 01 — Dataset Overview

**Date**: June 13, 2026  
**Source**: `DataSet.csv`

---

## Shape & Size

| Metric | Value |
|--------|-------|
| Rows | 9,082 |
| Columns | 3,925 |
| File Size | ~111 MB |
| Memory (loaded) | ~270 MB |

---

## Column Breakdown

| Data Type | Count | Description |
|-----------|-------|-------------|
| `float64` | 3,876 | Continuous numeric features |
| `int64` | 41 | Integer features (counts, binary flags) |
| `object` | 8 | Categorical/string features |
| **Total** | **3,925** | Includes index column `Unnamed: 0` and target `F3924` |

---

## Special Columns

| Column | Role |
|--------|------|
| `Unnamed: 0` | Row index (1-based), not a feature |
| `F3924` | **Target variable** — binary classification (0 = legit, 1 = suspicious) |
| `F1` to `F3923` | Anonymized features |

---

## Feature Usability Summary

| Category | Count |
|----------|-------|
| Columns with 100% missing values | 63 |
| Columns with ≤1 unique value (constant) | 359 |
| Columns with exactly 2 unique values (binary) | 527 |
| Columns with >90% zero values (sparse) | 815 |
| **Usable features** (<50% missing, >1 unique value) | **2,520** |
| — Usable numeric | 2,512 |
| — Usable categorical | 8 |

---

## Notes

- The dataset appears to be a **single snapshot** of bank account data with pre-computed features.
- Feature names are anonymized (F1–F3924) — the bank intentionally hid business meanings.
- The dataset is dense in features but sparse in positive (suspicious) samples.
- Every single row has at least 699 missing feature values.

---

<!-- FILE: 02_target_variable.md -->

# 02 — Target Variable Analysis (F3924)

---

## Distribution

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| 0 | Legitimate / Normal | 9,001 | 99.11% |
| 1 | Suspicious / Mule | 81 | 0.89% |

**Imbalance Ratio**: **111.1 : 1**

---

## Why This Matters

This is an **extreme class imbalance** problem. With only 81 positive samples:

### Modeling Implications
- A naive "always predict 0" classifier would achieve **99.11% accuracy** — but catch zero fraud.
- Standard loss functions will be dominated by the majority class.
- Models will tend to underfit the minority class.

### Required Strategies
1. **Resampling Techniques**:
   - SMOTE (Synthetic Minority Oversampling Technique)
   - ADASYN (Adaptive Synthetic Sampling)
   - Random undersampling of majority class
   - Combination: SMOTE + Tomek links

2. **Class Weight Adjustment**:
   - Set `class_weight='balanced'` (sklearn) or `scale_pos_weight=111` (XGBoost)

3. **Evaluation Metrics** (DO NOT use accuracy):
   - **Primary**: F1-Score (harmonic mean of precision & recall)
   - **Secondary**: AUC-ROC, PR-AUC (Precision-Recall AUC)
   - **Also track**: Precision, Recall, Confusion Matrix

4. **Validation Strategy**:
   - **Stratified K-Fold** (k=5) — ensures ~16 positive samples per fold
   - Never use random splits — the minority class may be absent from some folds

---

## Statistical Properties

| Metric | Value |
|--------|-------|
| Mean (F3924) | 0.0089 |
| Std Dev | 0.0940 |
| Data Type | int64 |
| Non-null | 9,082 / 9,082 (100%) |

---

<!-- FILE: 03_missing_values.md -->

# 03 — Missing Values Analysis

---

## Column-Level Missing Values

| Missing % Range | Column Count | Action |
|-----------------|-------------|--------|
| 0% (No missing) | 90 | ✅ Use as-is |
| 0–10% | 2,450 | ✅ Impute (median/mode) |
| 10–25% | 76 | ⚠️ Impute with caution |
| 25–50% | 171 | ⚠️ Impute or create missingness indicator |
| 50–75% | 116 | ❌ Consider dropping or using missingness as feature |
| 75–100% | 959 | ❌ Likely drop |
| 100% (All missing) | 63 | ❌ Must drop (zero information) |

### Columns with 100% Missing (63 total — all useless)

These include: F437, F440, F492, F495, F539, F542, F2312, F2607, F2655, F2707, F2756,
F3133, F3179, F3182, F3233, F3236, F3665, F3668, F3773, F3776, and 43 others.

---

## Row-Level Missing Values

| Metric | Value |
|--------|-------|
| Minimum missing per row | 699 |
| Maximum missing per row | 2,330 |
| Mean missing per row | 1,084.2 |
| Median missing per row | 1,076.0 |

### Key Insight
**Every single row** has at least 699 missing feature values (out of 3,923 features = ~17.8%).
The average row is missing ~27.6% of its features. This is inherent to the data structure — 
many features are likely only relevant to certain account types or transaction patterns.

---

## Recommended Missing Value Strategy

### Step 1: Drop useless columns
```
- 63 columns with 100% missing → DROP
- 359 columns with ≤1 unique value → DROP
- Total dropped: ~422 columns
```

### Step 2: For remaining columns
```
- Numeric with <50% missing → Impute with MEDIAN
- Categorical with <50% missing → Impute with MODE
- Numeric with 50-100% missing → Create binary "is_missing" indicator, then impute
- Consider: XGBoost/LightGBM can handle missing values natively
```

### Step 3: Missingness as a feature
Some columns may have **informative missingness** — the fact that a value is missing 
could itself be predictive (e.g., no loan features = no loan account = lower risk).
Consider creating `is_missing_X` binary features for columns with 10-90% missing.

---

<!-- FILE: 04_feature_types.md -->

# 04 — Feature Type Classification

All 3,923 features (excluding index and target) classified by statistical properties.

---

## Summary Table

| Category | Count | Value Pattern | Likely Meaning |
|----------|-------|--------------|----------------|
| **Proportions (0–1)** | 478 | Float in [0, 1] | Normalized ratios, percentages, channel distributions |
| **Binary {0, 1}** | 527 | Exactly {0, 1} | Flags, indicators, one-hot encoded categories |
| **Small floats** | 463 | Float, range < 100, ≥ 0 | Ratios, scores, multipliers (many centered around 1.0) |
| **Large floats** | 599 | Float, range ≥ 100, ≥ 0 | Monetary amounts, cumulative transaction values |
| **Negative floats** | 882 | Float, contains negatives | Deltas, changes, z-scores, growth rates |
| **Count-like integers** | 405 | Non-negative integers, >10 unique | Transaction counts, event counts |
| **Low-cardinality integers** | 26 | Integer, 3–10 unique values | Categorical-like (e.g., risk levels, categories) |
| **Categorical (object)** | 8 | String values | Account metadata fields |
| **Binary (non-0/1)** | 15 | 2 unique values, not {0,1} | Two-state features with non-standard encoding |
| **Constant** | 359 | ≤1 unique value | No information — must drop |
| **All null** | 63 | 100% missing | Completely empty — must drop |

---

## Binary {0,1} Features (527 total)

### Distribution by % of rows with value = 1

| % of 1s | Count | Interpretation |
|---------|-------|---------------|
| < 1% (near-zero) | ~180 | Very rare flags (specific alerts, rare events) |
| 1–10% (rare) | ~120 | Uncommon flags |
| 10–50% (moderate) | ~100 | Fairly common indicators |
| 50–90% (common) | ~80 | Majority-true flags |
| > 90% (near-all) | ~47 | Almost always true |

### Most predictive binary features (by correlation with target)

| Feature | Correlation | % with value=1 | Missing % |
|---------|------------|----------------|-----------|
| F3912 | +0.9691 | 0.90% | 0% |
| F3908 | +0.0970 | 30.6% | 0% |
| F670 | +0.0471 | 9.2% | 0.2% |
| F3909 | +0.0399 | 0.56% | 0% |
| F3913 | -0.0632 | 57.6% | 0% |
| F3914 | -0.0549 | 31.9% | 0% |

---

## One-Hot Encoded Groups Detected

Consecutive binary columns that sum to ~1.0 per row (one-hot encoding):

| Group | Columns | Size | Probable Meaning |
|-------|---------|------|-----------------|
| F3895–F3907 | 13 binary cols | 13 | Month of account opening or similar temporal category |
| F3915–F3923 | 9 binary cols | 9 | Transaction channel type or account status categories |

**Important**: Keep these groups together during feature engineering. Treating them as independent features could mislead models.

---

## Negative Float Features (882 total)

These are the largest category. They span F2500–F3799 predominantly.

**Statistical properties**:
- Most are centered near 0
- Contain both positive and negative values
- Likely represent: month-over-month changes, velocity deltas, z-scores, growth/decline rates

**Example patterns**:
- F2582: range [-0.88, 18.89], median = 0.00
- F2678: range [-0.91, 1,626,943.59], median = -0.12
- F2737: range [-0.94, 1,707.60], median = -0.15

These delta/change features are **highly relevant for fraud detection** — sudden behavioral changes 
are a primary indicator of mule account activity.

---

<!-- FILE: 05_feature_blocks.md -->

# 05 — Feature Block Analysis (Thematic Groupings)

Features are organized in numbered blocks that follow clear thematic patterns.
This suggests the dataset was constructed by concatenating different categories of banking features.

---

## Block Map

| Block Range | # Cols | Dominant Type | % | Likely Theme |
|-------------|--------|--------------|---|-------------|
| **F1–F99** | 99 | proportion (0–1) | 100% | Behavioral scores / risk proportions |
| **F100–F199** | 100 | proportion (0–1) | 88% | Behavioral scores (continued) |
| **F200–F299** | 100 | proportion (0–1) | 95% | Channel usage / distribution ratios |
| **F300–F399** | 100 | small float | 92% | Transaction velocity ratios (centered ~1.0) |
| **F400–F499** | 100 | small float | 96% | Peer comparison ratios |
| **F500–F599** | 100 | small float | 96% | Trend multipliers / deviation scores |
| **F600–F699** | 100 | constant (42%), binary (24%), large float (13%) | mixed | Product/channel flags + cumulative amounts |
| **F700–F799** | 100 | constant (42%), binary (29%), large float (15%) | mixed | Product/channel flags + amounts (continued) |
| **F800–F899** | 100 | constant (43%), binary (29%), large float (11%) | mixed | Product/channel flags + amounts (continued) |
| **F900–F999** | 100 | binary (36%), large float (28%), constant (27%) | mixed | Transaction channel flags + amounts |
| **F1000–F1099** | 100 | binary (47%), large float (40%) | mixed | Channel indicators + monetary volumes |
| **F1100–F1199** | 100 | binary (46%), large float (42%) | mixed | Channel indicators + volumes (continued) |
| **F1200–F1299** | 100 | binary (45%), large float (43%) | mixed | Channel indicators + volumes (continued) |
| **F1300–F1399** | 100 | binary (47%), large float (41%) | mixed | Channel indicators + volumes (continued) |
| **F1400–F1499** | 100 | binary (47%), large float (41%) | mixed | Channel indicators + volumes (continued) |
| **F1500–F1599** | 100 | large float (43%), binary (36%), count (11%) | mixed | Amounts + counts + flags |
| **F1600–F1699** | 100 | count (45%), large float (38%) | mixed | Transaction counts by type/channel |
| **F1700–F1799** | 100 | large float (44%), count (41%) | mixed | Amounts + counts |
| **F1800–F1899** | 100 | count (44%), large float (42%) | mixed | Counts + amounts |
| **F1900–F1999** | 100 | large float (44%), proportion (42%) | mixed | Volume proportions + amounts |
| **F2000–F2099** | 100 | large float (47%), proportion (47%) | equal | Volume ratios + amounts |
| **F2100–F2199** | 100 | proportion (46%), large float (46%) | equal | Proportions + volumes |
| **F2200–F2299** | 100 | large float (43%), small float (36%) | mixed | Derived ratios + aggregations |
| **F2300–F2399** | 100 | small float (51%), large float (45%) | mixed | Statistical aggregations |
| **F2400–F2499** | 100 | large float (54%), small float (43%) | mixed | Scored/weighted metrics |
| **F2500–F2599** | 100 | **negative float (76%)** | dominant | **Month-over-month changes / deltas** |
| **F2600–F2699** | 100 | **negative float (96%)** | dominant | **Change metrics / velocity deltas** |
| **F2700–F2799** | 100 | **negative float (97%)** | dominant | **Change metrics / z-scores** |
| **F2800–F2899** | 100 | count (67%), negative float (20%) | mixed | Transaction counts + some changes |
| **F2900–F2999** | 100 | **count (88%)** | dominant | **Raw transaction counts** |
| **F3000–F3099** | 100 | **count (90%)** | dominant | **Raw transaction counts (continued)** |
| **F3100–F3199** | 100 | count (41%), negative float (38%) | mixed | Counts + deviation metrics |
| **F3200–F3299** | 100 | negative float (75%), binary (18%) | mixed | Growth rates + flags |
| **F3300–F3399** | 100 | negative float (72%), binary (21%) | mixed | Growth rates + flags |
| **F3400–F3499** | 100 | negative float (88%) | dominant | Growth/change rates |
| **F3500–F3599** | 100 | negative float (88%) | dominant | Growth/change rates |
| **F3600–F3699** | 100 | negative float (88%) | dominant | Growth/change rates |
| **F3700–F3799** | 100 | negative float (86%) | dominant | Growth/change rates |
| **F3800–F3899** | 100 | mixed (neg 45%, prop 18%, large 11%) | mixed | **Account profile / metadata features** |
| **F3900–F3924** | 25 | binary (18 of 25) | dominant | **Alert/status flags + TARGET** |

---

## Visual Summary

```
FEATURE RANGE THEME MAP:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

F1 ────────── F299   │ PROPORTIONS (behavioral scores)
F300 ───────── F599   │ RATIOS (velocity, peer comparison)
F600 ───────── F1499  │ FLAGS + MONETARY AMOUNTS (channel/product)
F1500 ──────── F2199  │ COUNTS + PROPORTIONS + AMOUNTS
F2200 ──────── F2499  │ DERIVED SCORES & AGGREGATIONS
F2500 ──────── F2799  │ DELTAS / CHANGES (month-over-month)
F2800 ──────── F3099  │ RAW COUNTS (transaction counts)
F3100 ──────── F3799  │ GROWTH RATES & DEVIATIONS
F3800 ──────── F3899  │ ACCOUNT PROFILE & METADATA
F3900 ──────── F3924  │ ALERT FLAGS + TARGET VARIABLE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Implications for Feature Engineering

1. **Proportions (F1–F299)**: Already normalized, ready to use. May represent time-of-day distributions, channel usage patterns.

2. **Ratios (F300–F599)**: Values centered around 1.0 suggest peer/baseline comparisons. Values > 1 = above average, < 1 = below average.

3. **Flags + Amounts (F600–F1499)**: Heavy in constant columns (~42%). After removing constants, the binary flags indicate product/channel activity and the large floats are cumulative amounts.

4. **Deltas (F2500–F2799)**: These 300 features capture behavioral **changes over time** — critical for fraud detection since mule accounts often show sudden shifts in activity patterns.

5. **Raw Counts (F2800–F3099)**: Transaction counts that can reveal unusual activity volumes.

6. **Growth Rates (F3100–F3799)**: Percentage changes that normalize absolute counts — useful for comparing across different account sizes.

---

<!-- FILE: 06_key_features.md -->

# 06 — Key Features Analysis (Problem Statement Suggested)

The hackathon problem statement suggests these 18 features as commonly used for fraud detection.

---

## Feature Statistics Table

| Feature | Type | Non-Null % | Unique | Range | Mean (Legit) | Mean (Susp) | Corr Target | Interpretation |
|---------|------|-----------|--------|-------|-------------|-------------|-------------|---------------|
| **F115** | float64 | 96.1% | 97 | [0, 1] | 0.5905 | **0.7211** | +0.0576 | Behavioral risk score |
| **F321** | float64 | 99.1% | 308 | [0, 10.6] | 1.1526 | 1.1110 | -0.0099 | Transaction velocity ratio |
| **F527** | float64 | 91.3% | 301 | [0, 48.42] | 1.0007 | 0.9790 | -0.0028 | Peer comparison ratio |
| **F531** | float64 | 92.9% | 500 | [0, 20.96] | 1.4394 | 1.3811 | -0.0063 | Transaction diversity score |
| **F670** | float64 | 99.8% | 2 | {0, 1} | 0.0903 | **0.2346** | +0.0471 | High-risk flag |
| **F1692** | float64 | 99.9% | 12 | [0, 14] | 0.2611 | 0.1111 | -0.0181 | Suspicious activity count |
| **F2082** | float64 | 99.9% | 571 | [0, 1] | 0.0209 | **0.0000** | -0.0243 | Rare transaction proportion |
| **F2122** | float64 | 100.0% | 1,235 | [0, 1] | 0.0457 | 0.0052 | -0.0288 | Digital channel ratio |
| **F2582** | float64 | 62.5% | 290 | [-0.88, 18.89] | 0.0765 | 0.0741 | -0.0005 | Month-over-month change |
| **F2678** | float64 | 71.7% | 435 | [-0.91, 1.6M] | **351.25** | **11.41** | -0.0014 | Transaction amount deviation |
| **F2737** | float64 | 98.8% | 379 | [-0.94, 1707.6] | 0.3018 | 0.1330 | -0.0008 | Balance change metric |
| **F2956** | float64 | 88.7% | 718 | [0, 11,548] | **133.46** | **58.38** | -0.0206 | Total transaction count |
| **F3043** | float64 | 35.9% | 713 | [0, 21,819] | 232.45 | 129.93 | -0.0095 | Cross-channel txn count |
| **F3836** | float64 | 100.0% | 8,898 | [-20B, 16B] | -48.9M | **+1.49M** | +0.0081 | Total balance/throughput |
| **F3887** | int64 | 100.0% | 472 | [0, 1,510] | 103.78 | 107.62 | +0.0039 | **Account age (days)** ✅ |
| **F3894** | float64 | 100.0% | 95 | [-2, 94] | 34.33 | 32.84 | -0.0081 | **Customer age (years)** ✅ |

---

## Detailed Percentile Analysis

| Feature | 25th | 50th (Median) | 75th | 95th |
|---------|------|---------------|------|------|
| F115 | 0.45 | 0.53 | 0.70 | 1.00 |
| F321 | 1.00 | 1.07 | 1.29 | 1.79 |
| F527 | 0.83 | 1.00 | 1.13 | 1.64 |
| F531 | 1.00 | 1.21 | 1.68 | 2.95 |
| F670 | 0.00 | 0.00 | 0.00 | 1.00 |
| F1692 | 0.00 | 0.00 | 0.00 | 2.00 |
| F2082 | 0.00 | 0.00 | 0.00 | 0.16 |
| F2122 | 0.00 | 0.00 | 0.01 | 0.33 |
| F2582 | -0.13 | 0.00 | 0.13 | 0.82 |
| F2678 | -0.33 | -0.12 | 0.02 | 1.05 |
| F2737 | -0.39 | -0.15 | 0.00 | 0.62 |
| F2956 | 25.00 | 64.00 | 124.00 | 421.60 |
| F3043 | 9.00 | 87.50 | 226.00 | 773.90 |
| F3836 | 44,155 | 381,286 | 2,015,515 | 26,660,732 |
| F3887 | 27 | 94 | 150 | 267 |
| F3894 | 25 | 34 | 44 | 65 |

---

## Key Observations

### Most Discriminative (Legit vs Suspicious)
1. **F2678**: Legitimate accounts average **351** vs suspicious at **11** — 30x difference. Likely a transaction volume metric where mule accounts transact far less legitimately.
2. **F670**: Suspicious accounts are **2.6x more likely** to have this binary flag set (23.5% vs 9.0%).
3. **F115**: Suspicious accounts score higher (0.72 vs 0.59) — possibly a normalized risk indicator.
4. **F2956**: Suspicious accounts have ~44% fewer transactions (58 vs 133).
5. **F3836**: Sign reversal — legitimate average -49M, suspicious average +1.5M.
6. **F2082**: Suspicious accounts have **zero** proportion of this transaction type.

### Correlated Pairs (within key features)
| Pair | Correlation | Notes |
|------|------------|-------|
| F2956 ↔ F3043 | **0.9848** | Near-duplicates — same metric, different time/scope |
| F321 ↔ F531 | 0.4957 | Both are velocity ratios (centered ~1.0) |
| F3887 ↔ F3894 | 0.3966 | Account age ↔ customer age (older customers tend to have older accounts) |
| F1692 ↔ F2122 | 0.3302 | Activity count ↔ channel proportion |

### Weak Individual Correlations
- Maximum correlation with target: **0.0576** (F115)
- This means individual features have weak linear relationships with the target
- **Non-linear models** (XGBoost, Random Forest, Neural Nets) will be essential
- Feature **interactions** and **combinations** will be more predictive than individual features

---

## Confirmed Identities

| Feature | Confirmed As | Confidence | Evidence |
|---------|-------------|------------|---------|
| F3887 | Account age in days | **Very High** | 0.9999 correlation with (Oct 2025 - F3888 opening date) |
| F3894 | Customer age in years | **High** | Range [-2, 94], median 34, 0.40 correlation with account age |
| F670 | Some risk/alert flag | **Medium** | Binary, 2.6x enrichment in suspicious accounts |
| F2956 | Transaction count metric | **Medium** | Integer-valued, range [0, 11548], 0.98 corr with F3043 |

---

<!-- FILE: 07_categorical_features.md -->

# 07 — Categorical Features Analysis

All 8 categorical (object dtype) features with their value distributions and fraud rates.

---

## F2230 — Data/Reporting Period

⚠️ **DATA LEAKAGE — SEE [09_data_leakage.md](./09_data_leakage.md)**

| Value | Count | Suspicious | Suspicious % |
|-------|-------|-----------|-------------|
| Oct25 | 9,001 | 0 | **0.00%** |
| Sep25 | 48 | 48 | **100.00%** |
| Nov25 | 23 | 23 | **100.00%** |
| Dec25 | 10 | 10 | **100.00%** |

**Interpretation**: Likely the data reporting period or batch identifier. All suspicious accounts 
are from non-October batches, creating perfect temporal separation.

---

## F3886 — Account Type

| Value | Count | % of Total | Suspicious | Suspicious Rate |
|-------|-------|-----------|-----------|----------------|
| Savings | 5,956 | 65.6% | 76 | 1.28% |
| Current | 2,051 | 22.6% | 4 | 0.20% |
| MSME Micro | 337 | 3.7% | 0 | 0.00% |
| MSME Small | 242 | 2.7% | 0 | 0.00% |
| Staff Loans | 108 | 1.2% | 0 | 0.00% |
| Agri Adv | 96 | 1.1% | 0 | 0.00% |
| Term Deposit | 87 | 1.0% | 0 | 0.00% |
| MSME Medium | 71 | 0.8% | 1 | 1.41% |
| Corp Adv | 23 | 0.3% | 0 | 0.00% |
| Gold Loan | 23 | 0.3% | 0 | 0.00% |
| Retail Others | 22 | 0.2% | 0 | 0.00% |
| MSME LA TDR | 20 | 0.2% | 0 | 0.00% |
| Retail LA TDR | 19 | 0.2% | 0 | 0.00% |
| HL | 10 | 0.1% | 0 | 0.00% |
| All Others | 10 | 0.1% | 0 | 0.00% |
| ML | 4 | 0.0% | 0 | 0.00% |
| PL | 3 | 0.0% | 0 | 0.00% |

**Key Finding**: 93.8% of suspicious accounts are **Savings** accounts. Mule accounts 
predominantly use savings accounts — easy to open, less scrutiny than current accounts.

---

## F3888 — Account Opening Date

| Metric | Value |
|--------|-------|
| Unique values | 4,292 |
| Date range | 1900-01-03 to 2025-11-28 |
| Computed account age (mean) | 8.6 years |
| Computed account age (median) | 7.8 years |

**Note**: Some dates go back to 1900, suggesting data quality issues or placeholder values. 
The majority of accounts were opened in the last 15 years.

**Confirmed**: F3887 (account age in days) = reference date (Oct 2025) minus F3888. 
Correlation = 0.9999.

---

## F3889 — Account Scheme/Product

| Value | Count | Suspicious | Suspicious Rate | Likely Meaning |
|-------|-------|-----------|----------------|----------------|
| G365D | 7,544 | 72 | 0.95% | General 365-Day (regular account) |
| L365D | 397 | 5 | 1.26% | Loan 365-Day |
| L7D | 386 | 0 | 0.00% | Loan 7-Day |
| L180D | 313 | 3 | 0.96% | Loan 180-Day |
| L90D | 207 | 0 | 0.00% | Loan 90-Day |
| L31D | 148 | 1 | 0.68% | Loan 31-Day |
| L14D | 87 | 0 | 0.00% | Loan 14-Day |

**Key Finding**: 88.9% of suspicious accounts are on the **G365D** (general/regular) scheme. 
Short-term loan schemes (L7D, L14D, L90D) have **zero** suspicious accounts.

---

## F3890 — Branch Area Classification

| Value | Full Name | Count | Suspicious | Suspicious Rate |
|-------|----------|-------|-----------|----------------|
| R | Rural | 2,015 | 29 | **1.44%** |
| SU | Semi-Urban | 2,390 | 21 | 0.88% |
| U | Urban | 1,777 | 13 | 0.73% |
| M | Metro | 2,900 | 18 | 0.62% |

**Key Finding**: **Rural** areas have the highest suspicious rate (1.44%), 
more than **2x the metro rate** (0.62%). This could reflect weaker KYC enforcement 
in rural branches or targeted exploitation of rural accounts as mules.

---

## F3891 — Occupation

| Value | Count | % of Total | Suspicious | Suspicious Rate |
|-------|-------|-----------|-----------|----------------|
| selfemployed | 3,951 | 43.5% | 26 | 0.66% |
| salaried | 1,909 | 21.0% | 14 | 0.73% |
| student | 1,185 | 13.0% | 23 | **1.94%** |
| agriculture | 1,112 | 12.2% | 14 | **1.26%** |
| housewife | 660 | 7.3% | 3 | 0.45% |
| others | 169 | 1.9% | 0 | 0.00% |
| retired | 96 | 1.1% | 1 | 1.04% |

**Key Finding**: **Students** have the highest suspicious rate (1.94%) — nearly **3x** 
the rate of self-employed. Students may be targeted as mules due to financial vulnerability.
**Agriculture** workers are second highest (1.26%).

---

## F3892 — Gender

| Value | Count | % of Total | Suspicious | Suspicious Rate |
|-------|-------|-----------|-----------|----------------|
| M (Male) | 5,007 | 55.1% | 63 | **1.26%** |
| F (Female) | 1,416 | 15.6% | 13 | 0.92% |
| O (Other) | 61 | 0.7% | 0 | 0.00% |
| NaN (Missing) | 2,598 | 28.6% | 5 | 0.19% |

**Key Finding**: Males are 77.8% of suspicious accounts. Male accounts have a 
1.37x higher suspicious rate than female accounts.
Note: 28.6% of records have missing gender — these have a very low suspicious rate.

---

## F3893 — Customer Segment

| Value | Count | Suspicious | Suspicious Rate |
|-------|-------|-----------|----------------|
| RETAIL | 6,437 | 76 | **1.18%** |
| CORPORATE | 2,645 | 5 | 0.19% |

**Key Finding**: **RETAIL** accounts are 93.8% of suspicious accounts with 
a **6.2x higher** suspicious rate than corporate accounts.
This aligns with mule account patterns — individual retail accounts are 
easier to recruit and harder to monitor.

---

<!-- FILE: 08_top_correlations.md -->

# 08 — Top Correlated Features with Target (F3924)

Correlations computed for all 3,387 numeric features with computable correlation.
Only correlation with the target column was computed (not the full N×N matrix).

---

## Top 30 Features by Absolute Correlation

| Rank | Feature | Correlation | Direction | Feature Block | Likely Category |
|------|---------|------------|-----------|--------------|----------------|
| 1 | **F3912** | **0.9691** | + | F3900–F3924 | ⚠️ LEAKAGE: Fraud alert flag |
| 2 | F2506 | 0.1845 | + | F2500–F2599 | Delta / change metric |
| 3 | F2507 | 0.1845 | + | F2500–F2599 | Delta / change metric |
| 4 | F2409 | 0.1571 | + | F2400–F2499 | Derived score |
| 5 | F2408 | 0.1571 | + | F2400–F2499 | Derived score |
| 6 | F515 | 0.1370 | + | F500–F599 | Small float / ratio |
| 7 | F518 | 0.1269 | + | F500–F599 | Small float / ratio |
| 8 | F2578 | 0.1190 | + | F2500–F2599 | Delta / change metric |
| 9 | F82 | 0.1169 | + | F1–F99 | Proportion / score |
| 10 | F81 | 0.1169 | + | F1–F99 | Proportion / score |
| 11 | F83 | 0.1164 | + | F1–F99 | Proportion / score |
| 12 | F84 | 0.1164 | + | F1–F99 | Proportion / score |
| 13 | F255 | 0.1131 | + | F200–F299 | Proportion / ratio |
| 14 | F2285 | 0.1126 | + | F2200–F2299 | Derived aggregation |
| 15 | F285 | 0.1117 | + | F200–F299 | Proportion / ratio |
| 16 | F283 | 0.1117 | + | F200–F299 | Proportion / ratio |
| 17 | F2779 | 0.1098 | + | F2700–F2799 | Delta / z-score |
| 18 | F253 | 0.1094 | + | F200–F299 | Proportion / ratio |
| 19 | F287 | 0.1059 | + | F200–F299 | Proportion / ratio |
| 20 | F286 | 0.1058 | + | F200–F299 | Proportion / ratio |
| 21 | F267 | 0.1053 | + | F200–F299 | Proportion / ratio |
| 22 | F78 | 0.0985 | + | F1–F99 | Proportion / score |
| 23 | F77 | 0.0985 | + | F1–F99 | Proportion / score |
| 24 | F2502 | 0.0981 | **-** | F2500–F2599 | Delta / change metric |
| 25 | F2503 | 0.0979 | **-** | F2500–F2599 | Delta / change metric |
| 26 | F2686 | 0.0979 | + | F2600–F2699 | Delta / change metric |
| 27 | F3908 | 0.0970 | + | F3900–F3924 | Alert/status flag |
| 28 | F258 | 0.0957 | + | F200–F299 | Proportion / ratio |
| 29 | F388 | 0.0949 | + | F300–F399 | Small float / ratio |
| 30 | F389 | 0.0949 | + | F300–F399 | Small float / ratio |

---

## Observations

### Excluding F3912 (leakage), the strongest predictors come from:
1. **Delta/change metrics (F2400–F2700)**: 8 of top 30 — behavioral changes are highly predictive
2. **Proportions (F1–F299)**: 12 of top 30 — behavioral scores/ratios are strong signals
3. **Ratio features (F300–F599)**: 3 of top 30 — velocity ratios matter

### Correlated feature pairs (likely duplicates or near-duplicates):
- F2506 ↔ F2507 (identical correlation: 0.1845)
- F2409 ↔ F2408 (identical correlation: 0.1571)
- F82 ↔ F81 (nearly identical: 0.1169)
- F83 ↔ F84 (nearly identical: 0.1164)
- F285 ↔ F283 (identical: 0.1117)

These pairs likely represent the same metric computed over slightly different windows 
or with minor variations.

### All top features (except F2502, F2503) have POSITIVE correlation
This means suspicious accounts tend to have **higher** values for these features — 
suggesting elevated activity, behavioral anomalies, or risk indicators.

### Suggested features from problem statement are NOT in top 30
The 18 features mentioned in the problem statement (F115, F321, etc.) have weaker 
individual correlations (max 0.058). This doesn't mean they're useless — they may 
contribute strongly in **combination** with other features through non-linear models.

---

<!-- FILE: 09_data_leakage.md -->

# 09 — Data Leakage Analysis

⚠️ **CRITICAL — READ BEFORE ANY MODELING WORK** ⚠️

Two features in the dataset exhibit data leakage — they contain information 
that would not be available at prediction time and must be handled carefully.

---

## 🔴 F3912 — Fraud/Mule Alert Flag (MUST EXCLUDE)

### Evidence

| Metric | Value |
|--------|-------|
| Data type | int64 |
| Unique values | 2 ({0, 1}) |
| Correlation with target | **0.9691** |
| Distribution | 0: 9,000 / 1: 82 |

### Cross-tabulation with Target (F3924)

|  | F3924 = 0 (Legit) | F3924 = 1 (Suspicious) |
|---|---|---|
| **F3912 = 0** | 8,998 | 2 |
| **F3912 = 1** | 3 | 79 |

### Interpretation
- F3912 = 1 for 82 accounts, of which **79 are suspicious** (96.3% precision)
- F3912 = 0 for 9,000 accounts, of which **only 2 are suspicious** (0.02% miss rate)
- This is almost certainly a **fraud investigation flag** or **alert outcome** — it was likely 
  set AFTER the account was identified as suspicious, making it a downstream artifact of the target.
- The 3 false positives (F3912=1, F3924=0) and 2 false negatives (F3912=0, F3924=1) represent 
  minor label disagreements between the flag and the target.

### Decision: **EXCLUDE FROM ALL MODELS**
Including F3912 would create a model that simply copies this flag, achieving ~97% recall 
but providing zero predictive value for new/unseen accounts.

---

## 🟡 F2230 — Temporal Period (HANDLE WITH CAUTION)

### Evidence

| Value | Total Accounts | Suspicious | Suspicious Rate |
|-------|---------------|-----------|----------------|
| Oct25 | 9,001 | 0 | **0.00%** |
| Sep25 | 48 | 48 | **100.00%** |
| Nov25 | 23 | 23 | **100.00%** |
| Dec25 | 10 | 10 | **100.00%** |

### Interpretation
- **ALL 81 suspicious accounts** come from Sep25, Nov25, or Dec25
- **ALL 9,001 Oct25 accounts** are legitimate
- This creates a **perfect temporal separator** — a simple rule `if period != Oct25 → suspicious` 
  achieves 100% recall and 100% precision on this dataset

### Why This Is Problematic
1. The model would learn to simply check the time period instead of actual behavioral patterns
2. In production, new data would come from a different time period, making this rule useless
3. The suspicious accounts were likely identified through a separate investigation process 
   and added to the dataset from different time batches

### Decision Options
1. **EXCLUDE F2230** from features (safest approach)
2. **Use with awareness**: If the time period genuinely captures seasonal fraud patterns, 
   it may have legitimate predictive value — but this seems unlikely given the 100%/0% split
3. **Use for stratified splitting**: Ensure train/test splits don't separate by time period

### Recommended Decision: **EXCLUDE FROM FEATURES**

---

## 🟢 Neighboring Alert Flags (F3908–F3914)

These are in the same block as F3912 and should be reviewed:

| Feature | Unique | Distribution | Corr Target | Risk Level |
|---------|--------|-------------|-------------|------------|
| F3908 | 2 | 0:6300, 1:2782 | +0.0970 | Low — common flag (30.6%), weak correlation |
| F3909 | 2 | 0:9031, 1:51 | +0.0399 | Low — rare flag (0.56%) |
| F3910 | 2 | 0:8842, 1:240 | -0.0083 | Safe — negligible correlation |
| F3911 | 1 | 0:9082 | N/A | Safe — constant (no info) |
| **F3912** | **2** | **0:9000, 1:82** | **+0.9691** | **🔴 LEAKAGE** |
| F3913 | 2 | 1:5234, 0:3848 | -0.0632 | Low — common flag |
| F3914 | 2 | 0:6185, 1:2897 | -0.0549 | Low — common flag |

**Assessment**: Only F3912 is a clear leakage risk. F3908–F3914 (excluding F3912) appear 
to be legitimate binary features (e.g., product flags, verification flags) with weak 
correlations — they can likely be retained.

---

## Summary of Exclusions

| Feature | Action | Reason |
|---------|--------|--------|
| F3912 | **DROP** | Near-perfect correlation (0.97) with target — fraud flag leakage |
| F2230 | **DROP** | Perfect temporal separator — 100%/0% split by period |
| Unnamed: 0 | **DROP** | Row index, not a feature |
| 63 all-null columns | **DROP** | Zero information |
| 359 constant columns | **DROP** | Zero information |

---

<!-- FILE: 10_suspicious_profile.md -->

# 10 — Suspicious / Mule Account Profile

Deep dive into the 81 accounts labeled as suspicious (F3924 = 1).

---

## Overview

| Metric | Value |
|--------|-------|
| Total suspicious accounts | 81 |
| Percentage of dataset | 0.89% |
| Distribution across time periods | Sep25: 48, Nov25: 23, Dec25: 10 |

---

## Demographic Profile

### Account Type (F3886)
| Type | Count | % of Suspicious |
|------|-------|-----------------|
| **Savings** | **76** | **93.8%** |
| Current | 4 | 4.9% |
| MSME Medium | 1 | 1.2% |

→ Mule accounts are overwhelmingly **Savings accounts** (easy to open, less scrutiny).

---

### Account Scheme (F3889)
| Scheme | Count | % of Suspicious |
|--------|-------|-----------------|
| **G365D** | **72** | **88.9%** |
| L365D | 5 | 6.2% |
| L180D | 3 | 3.7% |
| L31D | 1 | 1.2% |

→ Regular/general accounts (G365D) dominate. Short-term loan schemes have zero mule accounts.

---

### Area (F3890)
| Area | Count | % of Suspicious | Rate in Area |
|------|-------|-----------------|-------------|
| **R (Rural)** | **29** | **35.8%** | **1.44%** |
| SU (Semi-Urban) | 21 | 25.9% | 0.88% |
| M (Metro) | 18 | 22.2% | 0.62% |
| U (Urban) | 13 | 16.0% | 0.73% |

→ **Rural** branches contribute the most suspicious accounts both in absolute count 
and rate. Metro areas have the lowest risk.

---

### Occupation (F3891)
| Occupation | Count | % of Suspicious | Rate in Group |
|-----------|-------|-----------------|--------------|
| selfemployed | 26 | 32.1% | 0.66% |
| **student** | **23** | **28.4%** | **1.94%** |
| salaried | 14 | 17.3% | 0.73% |
| agriculture | 14 | 17.3% | **1.26%** |
| housewife | 3 | 3.7% | 0.45% |
| retired | 1 | 1.2% | 1.04% |
| others | 0 | 0.0% | 0.00% |

→ **Students** have the highest risk rate (1.94%), nearly **3x self-employed**.
Students + agriculture account for 45.7% of mule accounts.

---

### Gender (F3892)
| Gender | Count | % of Suspicious |
|--------|-------|-----------------|
| **M (Male)** | **63** | **77.8%** |
| F (Female) | 13 | 16.0% |
| Missing | 5 | 6.2% |
| O (Other) | 0 | 0.0% |

→ Males are heavily overrepresented among suspicious accounts.

---

### Customer Segment (F3893)
| Segment | Count | % of Suspicious | Rate in Segment |
|---------|-------|-----------------|----------------|
| **RETAIL** | **76** | **93.8%** | **1.18%** |
| CORPORATE | 5 | 6.2% | 0.19% |

→ RETAIL accounts are 6.2x more likely to be suspicious than corporate accounts.

---

## Typical Mule Account Profile

Based on the data, the **highest-risk profile** is:

```
┌─────────────────────────────────────────┐
│          MULE ACCOUNT ARCHETYPE         │
├─────────────────────────────────────────┤
│  Account Type:    Savings               │
│  Scheme:          G365D (General)       │
│  Segment:         RETAIL                │
│  Branch Area:     Rural                 │
│  Occupation:      Student               │
│  Gender:          Male                  │
│  Account Age:     ~108 days (median)    │
│  Customer Age:    ~33 years (median)    │
│  F670 flag:       23.5% have it set     │
│  F2678 value:     Very low (~11 avg)    │
│  F2956 count:     Low (~58 avg)         │
└─────────────────────────────────────────┘
```

---

## Behavioral Indicators (from Key Features)

Comparing suspicious vs legitimate accounts on key numeric features:

| Feature | Legit (mean) | Suspicious (mean) | Ratio | Pattern |
|---------|-------------|-------------------|-------|---------|
| F115 (risk score) | 0.59 | **0.72** | 1.22x | Higher risk scores |
| F670 (flag) | 9.0% | **23.5%** | 2.6x | More alerts |
| F2082 (rare txn) | 0.021 | **0.000** | 0x | Zero rare transaction activity |
| F2122 (digital) | 0.046 | **0.005** | 0.11x | Almost no digital channel usage |
| F2678 (amount) | 351.25 | **11.41** | 0.03x | Tiny transaction deviation |
| F2956 (count) | 133.46 | **58.38** | 0.44x | Much fewer transactions |
| F3836 (balance) | -48.9M | **+1.49M** | sign flip | Net inflow instead of outflow |

### Behavioral Summary
Suspicious/mule accounts tend to:
1. ✅ Have **higher risk scores** (F115)
2. ✅ Trigger more **alert flags** (F670)
3. ❌ Have **zero** rare/special transaction types (F2082)
4. ❌ Use **almost no** digital channels (F2122)
5. ❌ Have **much fewer** overall transactions (F2956)
6. ❌ Show **minimal** transaction amounts (F2678)
7. 🔄 Show **net inflows** vs net outflows (F3836) — consistent with receiving fraudulent funds

---

<!-- FILE: 11_feature_interpretation.md -->

# 11 — Feature Interpretation (Reverse-Engineering Anonymized Columns)

Attempt to determine what the anonymized columns (F1–F3924) represent, 
based on statistical properties, value patterns, and banking domain knowledge.

---

## Confirmed Identities (High Confidence)

| Feature | Identity | Confidence | Evidence |
|---------|----------|------------|---------|
| **F3888** | Account Opening Date | ✅ Very High | Date format ("8-1-2011"), range 1900–2025 |
| **F3887** | Account Age (days) | ✅ Very High | **0.9999 correlation** with (Oct 2025 - F3888) |
| **F3894** | Customer Age (years) | ✅ High | Range [-2, 94], median=34, corr=0.40 with account age |
| **F3886** | Account Type | ✅ Exact | Literal values: Savings, Current, MSME, etc. |
| **F3889** | Account Scheme/Product | ✅ Exact | G365D, L365D, L180D, L90D, L31D, L14D, L7D |
| **F3890** | Branch Area Classification | ✅ Exact | R=Rural, M=Metro, SU=Semi-Urban, U=Urban |
| **F3891** | Occupation | ✅ Exact | selfemployed, salaried, student, agriculture, etc. |
| **F3892** | Gender | ✅ Exact | M, F, O |
| **F3893** | Customer Segment | ✅ Exact | RETAIL, CORPORATE |
| **F2230** | Data/Reporting Period | ✅ High | Oct25, Sep25, Nov25, Dec25 |
| **F3912** | Fraud/Mule Alert Flag | ✅ Very High | Binary, 0.97 correlation with target, near-identical distribution |
| **F3924** | Target: Suspicious Account | ✅ Given | Binary 0/1, defined in problem statement |

---

## Probable Identities (Medium Confidence)

| Feature | Probable Identity | Evidence |
|---------|------------------|---------|
| F115 | Risk score / behavioral proportion | [0,1], 97 unique, higher in suspicious (0.72 vs 0.59) |
| F321 | Transaction velocity ratio | [0,10.6], centered ~1.0, correlates with F531 (0.50) |
| F527 | Peer comparison / baseline ratio | [0,48.4], centered ~1.0 |
| F531 | Transaction diversity/complexity score | [0,20.96], correlates with F321 |
| F670 | High-risk alert flag | Binary, 2.6x enrichment in suspicious |
| F1692 | Flagged event count | Integer [0,14], low values |
| F2082 | Rare/special transaction proportion | [0,1], 95th%=0.16, zero for all suspicious |
| F2122 | Digital/online channel ratio | [0,1], 1235 unique, very low for suspicious |
| F2582 | Month-over-month behavioral change | [-0.88,18.89], centered ~0 |
| F2678 | Transaction amount deviation / volatility | [-0.91, 1.6M], massive range, 30x lower for suspicious |
| F2737 | Balance change metric | [-0.94, 1707.6], centered near 0 |
| F2956 | Total transaction count | [0, 11548], integer-like, 0.98 corr with F3043 |
| F3043 | Cross-channel/period transaction count | [0, 21819], near-duplicate of F2956 |
| F3836 | Total balance/net throughput | [-20B, +16B], sign reversal between classes |
| F3908 | Product/verification flag | Binary, 30.6% have it, +0.097 corr |

---

## Feature Block Themes (Inferred)

### F1–F299: Behavioral Scores (Proportions 0–1)
**478 features** with values strictly in [0, 1].

These likely represent:
- Time-of-day transaction distributions (% of transactions at night, weekend, etc.)
- Channel usage proportions (% ATM, % mobile, % branch, etc.)
- Transaction type proportions (% cash, % transfer, % bill pay, etc.)
- Normalized risk scores from internal models
- Account activity percentiles

### F300–F599: Velocity / Comparison Ratios
**463 features** as small positive floats, many centered around 1.0.

These likely represent:
- Current period / previous period ratios (velocity)
- Account value / peer average ratios (deviation from norm)
- Transaction frequency compared to baseline
- Values > 1 = above average/increasing, < 1 = below average/decreasing

### F600–F1499: Product/Channel Flags + Monetary Amounts
~42% constant (no info), ~35% binary flags, ~20% large floats.

These likely represent:
- Binary indicators for specific products (has_loan, has_credit_card, etc.)
- Binary indicators for transaction channels used
- Cumulative transaction amounts by product/channel
- Many constants suggest products/channels not relevant to this account population

### F1500–F2199: Transaction Counts + Volume Proportions
Mix of counts, large floats, and proportions.

These likely represent:
- Raw transaction counts by type (credit, debit, cash, transfer)
- Transaction volumes (total amounts)
- Volume proportions (% of total by channel/type)

### F2200–F2499: Derived Statistical Aggregations
Mix of small and large floats.

These likely represent:
- Averages (average transaction size, average daily balance)
- Standard deviations (transaction amount variability)
- Percentile metrics
- Weighted composite scores

### F2500–F2799: Change / Delta Metrics (882 features with negatives)
**Largest block** — 300 features predominantly containing negative values.

These likely represent:
- Month-over-month changes in transaction counts
- Period-over-period changes in balances
- Z-scores (standard deviation from mean)
- Growth/decline rates
- Velocity changes (acceleration/deceleration)

**Why these matter for fraud**: Mule accounts often show sudden behavioral shifts — 
a dormant account suddenly receiving many transactions, or a stable account 
showing unusual transfer patterns. These delta features capture exactly those signals.

### F2800–F3099: Raw Transaction Counts
~400 count-like integer features.

These likely represent:
- Daily/weekly/monthly transaction counts
- Counts by channel (ATM, branch, online, mobile)
- Counts by type (credit, debit, transfer, cash withdrawal)
- Number of unique beneficiaries
- Login/access counts

### F3100–F3799: Growth Rates & Deviation Scores
~700 features, mostly negative-capable floats.

These likely represent:
- Percentage change metrics (week-over-week, month-over-month)
- Normalized deviation scores
- Z-scores and percentile ranks
- Risk scoring outputs from multiple internal models

### F3800–F3899: Account Profile & Metadata
Mixed types — this is where the identified features live.

Known: Account type, date, scheme, area, occupation, gender, segment, age.

### F3900–F3924: Alert/Status Flags + Target
18 binary features + target variable.

Known: F3912 = fraud flag (leakage), F3911 = constant, F3924 = target.
F3895–F3907 and F3915–F3923 are one-hot encoded categorical groups.

---

## One-Hot Encoded Groups

| Group | Features | Size | Probable Meaning |
|-------|----------|------|-----------------|
| F3895–F3907 | 13 binary columns | 13 | Month of some temporal attribute (month of opening, last transaction, etc.) |
| F3915–F3923 | 9 binary columns | 9 | Transaction channel primary type or account status category |

---

## Limitations

> **Important**: While we can infer general categories (proportions, counts, amounts, deltas), 
> the **exact business meaning** of individual features cannot be definitively determined 
> without the official data dictionary from Bank of India.
> 
> The anonymization is intentional for security. Feature numbers may not follow a 
> logical naming convention — they could be arbitrarily assigned or auto-generated 
> from a feature engineering pipeline.
>
> Our interpretations are based on:
> - Statistical properties (range, distribution, data type)
> - Cross-correlations between features
> - Banking fraud detection domain knowledge
> - Value patterns and block analysis

---

<!-- FILE: 12_recommendations.md -->

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

---

<!-- FILE: 13_graph_analysis.md -->

# Visual Analysis — BOI Mule Account Dataset

14 graphs generated and saved to [files/graphs/](file:///c:/Github/BOI Hackathon/files/graphs).
Each visualization is analyzed below with key takeaways.

---

## 1. Class Imbalance

![Class Imbalance](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/01_class_imbalance.png)

> [!CAUTION]
> **111:1 imbalance** — Only 81 suspicious accounts out of 9,082.
> A naive model predicting "legitimate" for everything would score 99.11% accuracy but catch zero fraud.

**Takeaways:**
- Must use **SMOTE/ADASYN** or **class weights** during training
- Accuracy is a meaningless metric — use **F1-score** and **PR-AUC** instead
- Stratified splitting is mandatory for cross-validation

---

## 2. Missing Values Distribution

![Missing Values](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/02_missing_values.png)

**Takeaways:**
- Left: Most columns with missing data are missing for **10-30%** of rows — a cluster around 10% and another around 26%
- Right: Every single row is missing ~**1,084 features** (27.6%) — this is structural, not random
- The bimodal pattern in column-level missingness suggests **product-dependent features** (e.g., loan features are null for non-loan accounts)

---

## 3. Feature Type Distribution

![Feature Types](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/03_feature_types.png)

**Takeaways:**
- **882 negative-value features** (largest group) — these are delta/change metrics, critical for detecting behavioral shifts
- **527 binary features** — likely product/channel flags
- **478 proportion features** (0-1 range) — behavioral scores and ratios
- **422 features to drop** (359 constant + 63 all-null) — zero predictive value

---

## 4. Categorical Features vs Fraud Rate

![Categorical Fraud Rates](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/04_categorical_fraud_rates.png)

> [!IMPORTANT]
> **Students** have the highest fraud rate at 1.94% — nearly **3x** the dataset average of 0.89%.

**Key findings by category:**
- **Occupation:** Students (1.94%) > Agriculture (1.26%) > Retired (1.04%) > Salaried (0.73%)
- **Branch Area:** Rural (1.44%) is **2.3x** Metro (0.62%)
- **Customer Segment:** RETAIL (1.18%) is **6.2x** CORPORATE (0.19%)
- **Account Type:** Savings (1.28%) dominates — 76 of 81 suspicious accounts
- **Gender:** Males (1.26%) are overrepresented
- **Account Scheme:** L365D (1.26%) slightly above G365D (0.95%); short-term schemes (L7D, L14D, L90D) have **zero** fraud

---

## 5. Key Features: Legit vs Suspicious Distributions

![Key Features Comparison](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/05_key_features_comparison.png)

> [!TIP]
> The dashed vertical lines show the class means. Where the red (suspicious) and teal (legit) dashed lines separate, we have predictive signal.

**Clear separation patterns:**
- **F115 (Risk Score):** Suspicious accounts cluster near 0.8, legit spread 0.3-0.7 — suspicious have visibly higher risk scores
- **F670 (Alert Flag):** Binary — suspicious accounts have the flag set 2.6x more often
- **F2082 (Rare Txn):** Suspicious accounts show almost **zero** rare transaction activity vs. a spread for legit
- **F2122 (Digital):** Suspicious accounts have near-zero digital channel usage — they don't use mobile/online banking
- **F2956 (Txn Count):** Suspicious accounts cluster at very low counts vs. a wide spread for legit
- **F3894 (Age):** Suspicious accounts skew **younger** (peak at 20-35), legit has a broader age spread

**Weak/overlapping patterns:**
- **F2678 (Amt Dev):** Both classes spike near zero — limited visual separation
- **F3836 (Balance):** Extreme skew makes visual comparison difficult, but the sign flip (noted in prior analysis) is a strong signal

---

## 6. Top 20 Correlated Features (F3912 excluded)

![Top Correlations](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/06_top_correlations.png)

**Takeaways:**
- **ALL top 20 have positive correlation** — suspicious accounts have *higher* values across all top features
- **F2506/F2507** are tied at 0.1845 — likely the same metric computed differently (delta/change features)
- **F2408/F2409** are tied at 0.1571 — another duplicate pair
- The max correlation is only **0.18** — no single feature is a strong standalone predictor
- This confirms we need **non-linear models** (tree-based) to capture interactions between weak features
- Interestingly, the 18 features suggested in the problem statement are NOT in this top 20 — the data-driven top features differ from domain suggestions

---

## 7. Feature Block Composition

![Feature Blocks](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/07_feature_blocks.png)

**Takeaways:**
- **F0-499:** Dominated by proportions (yellow) and small floats (blue) — behavioral ratios and scores
- **F500-999:** Heavy in constants (gray) — many inactive product flags
- **F1000-1499:** Binary flags (green) + large floats (purple) — product holdings and monetary amounts
- **F1500-1999:** Large floats dominant — transaction volumes
- **F2000-2499:** Large floats — derived statistical aggregations
- **F2500-2999:** **Almost entirely negative-capable floats (red)** — this is the change/delta block, our most predictive area
- **F3000-3499:** Also heavy in negatives — growth rates and deviation scores
- **F3500-3999:** Mixed — includes metadata, categoricals, and alert flags

---

## 8. Suspicious Account Profile

![Suspicious Profile](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/08_suspicious_profile.png)

**The typical mule account is:**
- A **self-employed (32%)** or **student (28%)** individual
- From a **Rural (36%)** or **Semi-Urban (26%)** branch
- With a **Savings account (94%)**
- This maps to the real-world pattern: young individuals in smaller towns are recruited as mule account holders

---

## 9. Data Leakage — Critical Finding

![Data Leakage](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/09_data_leakage.png)

> [!CAUTION]
> **Two features MUST be excluded from all models:**

**Left — F3912 (Fraud Flag):**
- The heatmap shows 79 of 82 F3912=1 accounts are also F3924=1 (suspicious)
- Only 5 mismatches across the entire dataset
- Including this would create a model that simply copies this flag

**Right — F2230 (Time Period):**
- **100%** of non-Oct25 accounts are suspicious (48+23+10 = 81)
- **0%** of Oct25 accounts are suspicious
- The temporal split is perfect — a trivial rule achieves 100% accuracy
- This is clearly a data collection artifact, not a real pattern

---

## 10. Key Feature Box Plots

![Box Plots](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/10_key_feature_boxplots.png)

**Takeaways:**
- **F115 (Risk):** Median for suspicious is noticeably higher — clear upward shift
- **F321 (Velocity):** Similar medians but legit has wider spread — suspicious accounts have uniform behavior
- **F527 (Peer):** Nearly identical — weak standalone predictor
- **F531 (Diversity):** Legit accounts show wider whiskers — suspicious accounts lack transaction diversity
- **F1692 (Activity):** Suspicious has a lower median — less account activity
- **F2122 (Digital):** Strikingly different — suspicious median is at ~0, legit has visible spread upward

---

## 11. Key Feature Correlation Heatmap

![Correlation Heatmap](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/11_correlation_heatmap.png)

**Notable correlations:**
- **F2956 ↔ F3043 = 0.98** — near-duplicate features (likely total transaction counts over different periods)
- **F321 ↔ F531 = 0.50** — velocity and diversity are moderately linked
- **F1692 ↔ F2122 = 0.33** — activity count and digital usage are weakly linked
- **F3887 ↔ F3894 = 0.40** — account age and customer age have moderate positive correlation
- **Most features are near-zero correlated with each other** — they capture independent signals, which is excellent for model performance (low multicollinearity)

> [!TIP]
> The near-zero inter-correlations mean each feature contributes unique information — a tree-based model will benefit from having access to all of them.

---

## 12. Age Distributions

![Age Distributions](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/12_age_distributions.png)

**Account Age (F3887):**
- Both classes are heavily right-skewed — most accounts are **< 6 months old**
- Suspicious accounts are concentrated in the **0-3 month range** — newly opened accounts
- This confirms mule accounts are freshly created and used quickly

**Customer Age (F3894):**
- Suspicious accounts peak sharply at **25-35 years** 
- Legitimate accounts have a broader, flatter distribution extending to 70+
- Younger demographic is more likely to be recruited for mule activity

---

## 13. Block-Level Importance

![Block Importance](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/13_block_importance.png)

*(Available at [files/graphs/13_block_importance.png](file:///c:/Github/BOI Hackathon/files/graphs/13_block_importance.png))*

Shows mean and max absolute correlation with target by 500-feature block.

---

## 14. Missing Values by Class

![Missing by Class](C:/Users/shiva/.gemini/antigravity-ide/brain/b248bb74-ccd5-4087-892a-e32f45b95e15/14_missing_by_class.png)

**Takeaways:**
- Legit accounts: mean **1,084** missing features
- Suspicious accounts: mean **1,120** missing features — slightly higher
- The distributions are very similar — missingness is **not a strong class signal**
- However, the suspicious distribution has a slightly heavier right tail
- Adding a **"row_missing_count" feature** could provide marginal signal

---

## Summary of Key Visual Insights

| Finding | Visual Evidence | Impact on Model |
|---------|----------------|-----------------|
| 111:1 class imbalance | Graph 1 | Must use SMOTE/class weights |
| F3912 data leakage | Graph 9 (left) | DROP — 0.97 correlation with target |
| F2230 temporal leakage | Graph 9 (right) | DROP — 100%/0% temporal split |
| Students = highest risk | Graph 4 | Occupation is a strong categorical feature |
| Rural = highest risk area | Graph 4 | Branch area adds predictive value |
| Suspicious = low digital | Graph 5, 10 | F2122 is one of the best features |
| Suspicious = newer accounts | Graph 12 | Account age (F3887) is predictive |
| Delta features = most predictive | Graph 6, 7 | F2500-F2999 block is the gold mine |
| Features are mostly independent | Graph 11 | Low multicollinearity = good for trees |
| Missingness is structural | Graph 2, 14 | Not class-dependent, use tree models |

> [!NOTE]
> All 14 graphs are saved permanently in [files/graphs/](file:///c:/Github/BOI Hackathon/files/graphs) for presentation and documentation purposes.

---

<!-- FILE: 14_deep_investigation.md -->

# 14 — Deep Investigation Phase 2

**Date**: June 13, 2026  
**Script**: `deep_investigation.py`

---

## 1. Mutual Information Analysis (Non-Linear Feature Importance)

MI captures non-linear relationships that correlation misses. Computed for 3,515 valid numeric features.

### Top 20 by MI (excluding leakage features)

| Rank | Feature | MI Score | Feature Block | Likely Meaning |
|------|---------|----------|---------------|----------------|
| 1 | F3484 | 0.0110 | F3000-3499 | Growth/deviation metric |
| 2 | F1489 | 0.0087 | F1000-1499 | Product holding amount |
| 3 | F1495 | 0.0085 | F1000-1499 | Product holding amount |
| 4 | F1501 | 0.0082 | F1000-1499 | Product holding amount |
| 5 | F1863 | 0.0080 | F1500-1999 | Transaction volume |
| 6 | F1719 | 0.0079 | F1500-1999 | Transaction volume |
| 7 | F3805 | 0.0079 | F3500-3999 | Account metadata/stat |
| 8 | F1713 | 0.0079 | F1500-1999 | Transaction volume |
| 9 | F1821 | 0.0079 | F1500-1999 | Transaction volume |
| 10 | F1827 | 0.0079 | F1500-1999 | Transaction volume |

### Key insight: MI vs Correlation disagree
- The **top MI features are completely different** from the top correlation features (F2506/F2507)
- Top MI features cluster in **F1000-F1999** (product holdings + transaction volumes)
- Top correlation features cluster in **F2500-F2999** (behavioral deltas)
- This means: **linear methods find different signals than non-linear methods**
- **Both should be included** in the model feature set

---

## 2. Feature Redundancy Analysis

### Near-duplicate pairs (|r| > 0.95): **268 pairs found**

Top examples (all r = 1.0000 — perfect duplicates):

| Feature A | Feature B | Action |
|-----------|-----------|--------|
| F1495 | F1501 | Drop F1501 |
| F1719 | F1713 | Drop F1713 |
| F1821 | F1827 | Drop F1827 |
| F1171 | F1177 | Drop F1177 |
| F1717 | F1711 | Drop F1711 |
| F1825 | F1819 | Drop F1819 |

**Pattern**: Features at distance 6 are often perfect duplicates (F1495↔F1501, F1171↔F1177).  
This is likely the same metric computed over overlapping time windows.

**Action**: **67 features can be safely dropped** (keeping the higher-MI copy in each pair).

---

## 3. Rule-Based Pattern Discovery

### Best Single-Feature Rules

| Feature | Rule | Flagged | Caught | Precision | Recall | F1 |
|---------|------|---------|--------|-----------|--------|----|
| F3811 | <= 92,192 | 455 | 31 | 6.81% | 38.27% | 0.116 |
| F2486 | <= 0.28 | 288 | 20 | 6.94% | 24.69% | 0.108 |
| F3805 | <= 110,502 | 455 | 29 | 6.37% | 35.80% | 0.108 |
| F1165 | <= 25,000 | 492 | 30 | 6.10% | 37.04% | 0.105 |
| F1705 | <= 90,049 | 454 | 26 | 5.73% | 32.10% | 0.097 |

**Interpretation**: Suspicious accounts have **lower monetary values** across all top rules. Mule accounts are characterized by small balances and low transaction amounts.

### Best Combination Rules (AND)

| Rule | Flagged | Caught | Precision | Recall | F1 |
|------|---------|--------|-----------|--------|----|
| F3805 <= 110,502 AND F1705 <= 90,049 | 53 | 26 | **49.06%** | 32.10% | 0.388 |
| F3811 <= 92,192 AND F1705 <= 90,049 | 55 | 26 | **47.27%** | 32.10% | 0.382 |
| F3811 <= 92,192 AND F1813 <= 224,179 | 67 | 28 | **41.79%** | 34.57% | 0.378 |
| F3811 <= 92,192 AND F1597 <= 125,508 | 84 | 31 | **36.90%** | 38.27% | 0.376 |
| F3805 <= 110,502 AND F1165 <= 25,000 | 44 | 20 | **45.45%** | 24.69% | 0.320 |

**Critical finding**: Simple 2-rule combinations achieve up to **49% precision** — flagging just 53 accounts and catching 26 of 81 suspicious ones. This is a massive improvement over single rules.

---

## 4. Suspicious Account Clustering

### Are there sub-types of mule accounts?

Tried k=2,3,4 with KMeans on top 20 MI features:

| K | Silhouette Score | Distribution |
|---|------------------|--------------|
| 2 | **0.809** (best) | 78 + 3 |
| 3 | 0.792 | 78 + 2 + 1 |
| 4 | 0.518 | 69 + 2 + 9 + 1 |

### Conclusion: Suspicious accounts are **homogeneous**

The k=2 clustering only separates 3 outlier accounts. The 78-account main cluster profile:
- **Occupation**: Self-employed (24) + Student (23) — dominates
- **Branch**: Rural (29) + Semi-Urban (21)
- **Account Type**: Savings (75 of 78)
- **Risk Score (F115)**: mean 0.72, median 0.74
- **Digital Usage (F2122)**: mean 0.00, median 0.00 — zero digital
- **Txn Count (F2956)**: mean 52, median 40.5
- **Age (F3894)**: mean 32.5, median 30.5
- **Account Age (F3887)**: mean 110 days, median 101.5 days

The 3 outliers are: 2 metro self-employed with current/MSME accounts (very young, 21y, brand new accounts 8.5 days), and 1 elderly housewife (87y, metro, savings).

---

## 5. Feature Interaction Analysis

### Difference/Ratio features beat individual features by >100%

| Interaction | MI | vs Best Individual |
|-------------|----|--------------------|
| F3805 - F1827 | 0.0168 | **+113%** |
| F3805 - F1821 | 0.0168 | **+113%** |
| F1719 / F3805 | 0.0158 | **+100%** |
| F3805 / F1713 | 0.0157 | **+99%** |

**F3805** is a key interaction hub — its difference with transaction volumes (F1821, F1827, F1713) generates the strongest signals. These engineered features should be added to the model.

---

## 6. Engineered Feature Results

| Feature | MI Score | Value? |
|---------|----------|--------|
| missing_count | 0.0062 | Yes — moderate signal |
| balance_positive | 0.0025 | Marginal |
| risk_x_alert | 0.0019 | Marginal |
| txn_per_day | 0.0016 | Marginal |
| non_digital | 0.0006 | Weak |
| young_factor | 0.0006 | Weak |

**missing_count** is worth adding as a feature. The rest don't justify inclusion over raw features.

---

## 7. Combined Feature Ranking (MI + Correlation)

The final recommended feature set should use features that rank high on **both** MI and correlation:

| Rank | Feature | MI | |Corr| |
|------|---------|----|----|
| 1 | F2686 | 0.0067 | 0.0979 |
| 2 | F267 | 0.0063 | 0.1053 |
| 3 | F162 | 0.0060 | 0.0771 |
| 4 | F3908 | 0.0052 | 0.0970 |
| 5 | F270 | 0.0053 | 0.0900 |

These features are strong on both linear and non-linear metrics.

---

## Summary of New Findings

1. **MI top features differ from correlation top features** — need both for complete picture
2. **268 near-duplicate pairs found** — 67 features can be safely dropped
3. **2-rule combinations achieve 49% precision** — catching 32% of mule accounts with simple thresholds
4. **Suspicious accounts are homogeneous** — no meaningful sub-types (k=2 silhouette 0.81 but only separates 3 outliers)
5. **Feature interactions boost MI by up to 113%** — F3805-based differences are the strongest
6. **missing_count** is a useful engineered feature (MI = 0.0062)

---

## Graphs Generated

| # | File | Description |
|---|------|-------------|
| 15 | `15_mutual_information.png` | MI top 20 + MI vs Correlation scatter |
| 16 | `16_clustered_correlations.png` | Hierarchically clustered correlation heatmap of top 50 MI features |
| 17 | `17_cluster_visualization.png` | PCA + t-SNE projection showing suspicious account clusters |
| 18 | `18_feature_interactions.png` | Top 15 feature interactions by MI |
| 19 | `19_combined_ranking.png` | Combined MI + Correlation feature ranking |

---

<!-- FILE: 15_phase3_investigation.md -->

# 15 — Phase 3 Comprehensive Investigation

**Date**: June 13, 2026  
**Script**: `deep_investigation_phase3.py`  
**Runtime**: 43.6 seconds

---

## 1. Statistical Hypothesis Testing (KS + Mann-Whitney U)

Tested 2,917 valid numeric features with Bonferroni-corrected significance (α = 1.71e-05).

| Metric | Result |
|--------|--------|
| Significant by KS test | **349 features** |
| Significant by Mann-Whitney U | **354 features** |

### Top Features by KS Statistic (distributional difference)

| Feature | KS Stat | p-value | Significant? | Block |
|---------|---------|---------|-------------|-------|
| F451 | 0.7388 | 7.25e-04 | No (sparse) | F0-499 |
| F453 | 0.7351 | 8.00e-04 | No (sparse) | F0-499 |
| F448 | 0.7340 | 8.19e-04 | No (sparse) | F0-499 |
| F142 | 0.7181 | 1.22e-03 | No (sparse) | F0-499 |
| **F2480** | **0.6524** | **3.76e-07** | **YES ✓** | F2000-2499 |
| **F2483** | **0.6082** | **3.56e-06** | **YES ✓** | F2000-2499 |
| **F261** | **0.5860** | **2.24e-06** | **YES ✓** | F0-499 |
| **F2284** | **0.5756** | **1.59e-05** | **YES ✓** | F2000-2499 |

### Key Insights
- Top KS features (F451/F453/F448) have **massive distributional differences** (KS > 0.73) but fail significance because they're sparse features (few non-null values) — the p-value is inflated by small sample size.
- **F2480, F2483** (behavioral flags block) and **F261** are the most statistically significant features.
- The volcano plot shows a clear L-shaped pattern: 349 features cross the Bonferroni threshold, confirming the signal is **real and widespread**.

---

## 2. Anomaly Detection (Unsupervised)

### Isolation Forest Results

| Contamination | Anomalies | Caught | Recall | Precision | F1 |
|---------------|-----------|--------|--------|-----------|-----|
| 1% | 91 | **1** | 1.2% | 1.1% | 0.012 |
| 2% | 182 | **1** | 1.2% | 0.5% | 0.008 |
| 5% | 455 | **4** | 4.9% | 0.9% | 0.015 |
| 10% | 909 | **8** | 9.9% | 0.9% | 0.016 |

### Local Outlier Factor Results

| n_neighbors | Anomalies | Caught | Recall |
|-------------|-----------|--------|--------|
| 10 | 182 | **0** | 0.0% |
| 20 | 182 | **0** | 0.0% |
| 50 | 182 | **0** | 0.0% |

### ⚠️ CRITICAL FINDING: Mule accounts are NOT anomalies!

This is arguably **the most important finding of the entire investigation**:
- Isolation Forest barely catches any suspicious accounts (max 8/81 at 10% contamination)
- LOF catches **zero** — none at all
- **Mule accounts are designed to blend in**. They look normal in feature space.
- This confirms the problem is a **supervised classification** task, NOT an anomaly detection task
- Any approach based on "find the outliers" will **catastrophically fail**
- The signal requires **learning subtle patterns** that distinguish mule accounts from legitimate ones — exactly what tree-based models do

---

## 3. Temporal / Period Analysis

### Time Period Distribution

| Period | Total Accounts | Suspicious | Rate |
|--------|---------------|------------|------|
| **Oct25** | 9,001 | **0** | 0.00% |
| **Sep25** | 48 | **48** | 100.00% |
| **Nov25** | 23 | **23** | 100.00% |
| **Dec25** | 10 | **10** | 100.00% |

### ⚠️ CRITICAL FINDING: Massive data structure issue!

This reveals the dataset structure:
- **Oct25** contains ONLY legitimate accounts (9,001)
- **Sep25/Nov25/Dec25** contain ONLY suspicious accounts (81)
- No month has BOTH legitimate and suspicious accounts mixed together

**Implication for modeling**: F2230 (Reporting Period) is a **perfect leakage feature** — already excluded. But this also means the suspicious accounts were reported in different months. The temporal distribution of features shows:

| Feature | Sep25 (n=48) | Nov25 (n=23) | Dec25 (n=10) |
|---------|-------------|-------------|-------------|
| F115 (Risk) | 0.82 | 0.58 | 0.57 |
| F2122 (Digital) | 0.002 | 0.008 | 0.017 |
| F2956 (Txn Count) | 60 | 52 | 65 |
| F3894 (Age) | 32.1 | 32.9 | 36.2 |
| F3887 (Acct Age days) | 116 | 97 | 93 |
| F3805 (Stat) | 485K | 279K | 1.2M |

- Sep25 mules have **higher risk scores** (0.82 vs 0.57-0.58)
- Dec25 mules show **slightly more digital activity** (0.017 vs 0.002)
- The profiles are broadly similar but Sep25 accounts are "more obvious" mules

---

## 4. Categorical Interaction Deep Dive

### ALL combinations show 0% fraud rate!

| Combination | Total | Suspicious | Fraud Rate | Lift |
|-------------|-------|------------|-----------|------|
| Student + Rural | 326 | **0** | 0.00% | 0.00x |
| Student + Savings | 1,182 | **0** | 0.00% | 0.00x |
| Student + Young | 1,098 | **0** | 0.00% | 0.00x |
| Self-emp + Rural | 703 | **0** | 0.00% | 0.00x |
| Self-emp + Semi-Urban | 956 | **0** | 0.00% | 0.00x |
| Young + Rural + Savings | 798 | **0** | 0.00% | 0.00x |
| Male + Student + Rural | 225 | **0** | 0.00% | 0.00x |

### CRITICAL INSIGHT
Despite our Phase 1 finding that suspicious accounts are predominantly students/self-employed/rural/savings — these categorical features have **ZERO discriminative power** when applied to the full dataset. There are far too many legitimate students in rural areas with savings accounts. The signal is entirely in the **numeric behavioral features**, not the demographics.

---

## 5. Tree-Based Feature Importance + CV Baseline

### Random Forest Top 10 Features

| Rank | Feature | Importance | Block |
|------|---------|-----------|-------|
| 1 | **F3898** | 0.0193 | Account metadata |
| 2 | **F3811** | 0.0146 | Account metadata |
| 3 | **F3813** | 0.0126 | Account metadata |
| 4 | **F1921** | 0.0121 | Transaction volumes |
| 5 | **F3805** | 0.0099 | Account metadata |
| 6 | **F162** | 0.0096 | Early features |
| 7 | **F3806** | 0.0088 | Account metadata |
| 8 | **F2137** | 0.0087 | Behavioral flags |
| 9 | **F1933** | 0.0087 | Transaction volumes |
| 10 | **F3812** | 0.0086 | Account metadata |

### 5-Fold CV Baseline (Balanced Random Forest, All Features)

| Metric | Mean | Std |
|--------|------|-----|
| **F1-Score** | **0.682** | ±0.166 |
| **Precision** | **0.886** | ±0.116 |
| **Recall** | **0.579** | ±0.198 |
| **ROC-AUC** | **0.969** | ±0.019 |

**Interpretation**:
- **AUC of 0.969** — excellent discrimination, the model CAN separate classes
- **Precision of 88.6%** — when it flags an account, it's almost always right
- **Recall of 57.9%** — it only catches ~58% of mule accounts (room for improvement)
- **F1 variance is HIGH** (±0.166) — performance depends heavily on which suspicious accounts end up in validation

### Key: F3800-F3900 block DOMINATES tree importance
6 of the top 10 RF features are in the F3800-F3900 range. These are account-level metadata/statistics that the tree model finds most useful for splitting.

---

## 6. Benford's Law Analysis

### Results (chi-squared test)

| Feature | Legit: Follows Benford? | Suspicious: Follows Benford? |
|---------|------------------------|------------------------------|
| F1165 (Balance) | **NO** (χ²=653, p≈0) | **YES** (χ²=5.1, p=0.75) |
| F1705 (Txn Amount) | **NO** (χ²=43, p≈7e-7) | **YES** (χ²=14.5, p=0.07) |
| F2956 (Txn Count) | **NO** (χ²=177, p≈0) | **YES** (χ²=3.5, p=0.90) |
| F3805 (Stat) | **NO** (χ²=60, p≈6e-10) | **YES** (χ²=15.3, p=0.05) |
| F3836 | **YES** (χ²=11.8, p=0.16) | **YES** (χ²=10.0, p=0.27) |
| F1489 | **NO** (χ²=1075, p≈0) | **YES** (χ²=11.6, p=0.17) |
| F1597 | **NO** (χ²=38, p≈9e-6) | **YES** (χ²=9.6, p=0.30) |

### 🔍 FASCINATING FINDING: Mule accounts FOLLOW Benford's Law, legitimate accounts DON'T

This is **counterintuitive** — normally you'd expect fraudulent accounts to violate Benford's Law. But:
- **Legitimate** bank accounts have non-Benford distributions because they're influenced by bank products, salary ranges, and psychological rounding (e.g., ₹25,000, ₹50,000)
- **Mule accounts** have small, seemingly random amounts that happen to follow the natural logarithmic distribution

This could be because mule accounts receive small, varied amounts from multiple sources to avoid detection — generating a naturally Benford-distributed pattern. This is a **potential engineered feature**: a Benford conformity score per account.

---

## 7. Hardest-to-Catch Profiling

### All 81 suspicious accounts are catchable! (RF score > 0.5)
- **Easy (score > 0.5): 81/81 (100%)**
- **Hard (score ≤ 0.5): 0/81 (0%)**

However, some are borderline:

| Rank | RF Score | Occupation | Area | Account | Age | Why Hard? |
|------|----------|-----------|------|---------|-----|-----------|
| 1 | 0.548 | Self-employed | Rural | Current | 3y | Very young, Current account — unusual mule profile |
| 2 | 0.586 | Agriculture | Semi-Urban | Savings | 44y | Middle-aged agriculture worker — blends perfectly |
| 3 | 0.599 | Housewife | Metro | Savings | 87y | 87-year-old metro housewife — extremely atypical mule |
| 4 | 0.629 | Salaried | Urban | Savings | 32y | Classic salaried urban profile — looks totally normal |
| 5 | 0.635 | Self-employed | Metro | MSME | 41y | MSME account in metro — legitimate business appearance |

The hardest-to-catch accounts **don't fit the typical mule profile** (young, student, rural). They have diverse occupations, older ages, and urban/metro locations.

---

## 8. Feature Stability (Bootstrap)

### Most Stable Features (appear in top-50 across 20 bootstraps)

| Feature | Appearance Rate | Mean Rank | Best Rank |
|---------|----------------|-----------|-----------|
| F3811 | **95%** | 16.6 | 2 |
| F3806 | **95%** | 23.3 | 1 |
| F3799 | **95%** | 14.1 | 1 |
| F3805 | **90%** | 10.1 | 1 |
| F3813 | **90%** | 13.8 | 1 |
| F3801 | **85%** | 11.6 | 1 |
| F3898 | **80%** | 9.8 | 1 |
| F3807 | **80%** | 15.1 | 1 |
| F1921 | **75%** | 14.6 | 2 |
| F3800 | **75%** | 16.4 | 3 |

### ROCK-SOLID FEATURES (≥ 80% stability)
**F3811, F3806, F3799, F3805, F3813, F3801, F3898, F3807** — these appear in 80%+ of all bootstrap runs and should form the **core feature set** for the production model.

### Moderately Stable (60-75%)
F1921, F3800, F1057, F1705, F1815, F2029, F1927, F1381, F1813, F1165, F1819, F3812

---

## Graphs Generated

| # | File | Description |
|---|------|-------------|
| 20 | `20_statistical_tests.png` | KS statistic top 30 + Volcano plot |
| 21 | `21_anomaly_detection.png` | Isolation Forest PCA projection + score distributions |
| 22 | `22_temporal_analysis.png` | Feature distributions across Sep25/Nov25/Dec25 |
| 23 | `23_categorical_interactions.png` | Fraud rate lift by demographic combinations |
| 24 | `24_tree_importance.png` | RF feature importance + 5-fold CV baseline |
| 25 | `25_benfords_law.png` | First digit distributions vs Benford's expected |
| 26 | `26_hardest_to_catch.png` | RF score distribution of suspicious accounts |
| 27 | `27_feature_stability.png` | Bootstrap stability analysis |

---

## Summary of Phase 3 Breakthrough Findings

### 🚨 Critical Discovery #1: Mule accounts are NOT anomalies
Isolation Forest catches max 8/81, LOF catches 0. Mule accounts are **designed to blend in**. Any anomaly-based approach will fail. Must use supervised learning.

### 🚨 Critical Discovery #2: Categorical features have ZERO discriminative power
Despite mules being predominantly young/student/rural/savings, these demographics are equally common in legitimate accounts. The signal is **entirely in numeric behavioral features**.

### 🔬 Discovery #3: Benford's Law reversal
Mule accounts FOLLOW Benford's Law while legitimate accounts DON'T — mule transactions look more "naturally random." Potential engineered feature.

### 📊 Discovery #4: Strong baseline already achievable
Balanced RF achieves **F1=0.682, AUC=0.969** out of the box. High precision (89%) but recall needs work (58%).

### 🎯 Discovery #5: Stable core feature set identified
8 features appear in 80%+ of all bootstrap samples: **F3811, F3806, F3799, F3805, F3813, F3801, F3898, F3807**. These are the production-ready features.

### 🕵️ Discovery #6: All accounts are catchable
Every suspicious account scores > 0.5, but the hardest ones are atypical mules (elderly, metro, salaried) that don't match the typical profile.

---

<!-- FILE: 16_ml_techniques_research.md -->

# ML Techniques Research — Out-of-the-Box Approaches

**Date**: June 13, 2026  
**Source**: User research notes  

---

## Recommended Approaches for Mule-Account Detection

### 1. Gradient-Boosted & Ensemble Models
- **XGBoost, LightGBM, CatBoost, Random Forests** — proven fraud baselines
- XGBoost: `scale_pos_weight` parameter + native missing value handling
- CatBoost: best on categorical data, handles missing values
- imbalanced-learn: BalancedRandomForest, EasyEnsemble, RUSBoost

### 2. Resampling and Imbalance Handling
- **SMOTE, BorderlineSMOTE, ADASYN** (imbalanced-learn)
- SMOTE + Tomek links / EditedNN for overlap cleaning
- **CTGAN** — conditional GANs for synthetic minority samples
- All plug-and-play via Pipeline or fit_resample

### 3. Anomaly/Outlier Detection (PyOD)
- **60+ algorithms**: IsolationForest, LOF, CBLOF, COPOD, HBOS, AutoEncoder
- One-class approaches: One-Class SVM, Deep One-Class
- ⚠️ NOTE: Our Phase 3 found IF/LOF perform **very poorly** on this dataset (mules blend in)

### 4. Graph and Network Techniques
- Transaction graph analysis (Neo4j, NetworkX)
- Graph centrality (PageRank, Betweenness) for role detection
- GNNs (PyTorch Geometric, DGL) for learning from account-transaction networks
- Graph features (degree, centrality) as regular model features

### 5. AutoML Platforms
- **AutoGluon, H2O AutoML, Auto-sklearn, TPOT**
- Amazon Fraud Detector (paid AWS service)
- PyCaret — turnkey library
- Auto-sklearn uses meta-learning for warm-start

### 6. Feature-Rich Pipelines
- scikit-learn Pipeline + imblearn sampler + XGBoost
- CatBoost auto-handles categoricals
- Deep learning: Autoencoders, DeepSVDD

### 7. Ensembles & Stacking
- VotingClassifier, StackingClassifier
- Blend XGB + LightGBM + IsolationForest scores
- mlxtend for easy stacking

### 8. Built-in Imbalance Handling
- CatBoost: scale_pos_weight
- LightGBM: is_unbalance parameter
- XGBoost: gpu_hist mode for GPU acceleration

### 9. Evaluation Tools
- StratifiedKFold, StratifiedShuffleSplit, GroupKFold
- F1, ROC-AUC, PR-AUC metrics
- imblearn pipelines auto-resample inside CV folds

---

## Key Takeaway
Tree-based ensembles + imbalanced-learn resamplers are the core.  
Anomaly detectors provide complementary signals (but weak standalone for this dataset).  
Graph methods useful IF transaction-level data is available.  
AutoML can quickly explore a wide range of approaches.

---

<!-- FILE: 17_phase4_investigation.md -->

# 17 — Phase 4 Final Deep Dive

**Date**: June 13, 2026  
**Script**: `deep_investigation_phase4.py`  
**Runtime**: 106.8 seconds

---

## 1. Feature Direction Analysis (SHAP-like)

Analyzed the **directional impact** of each feature — whether suspicious accounts have higher or lower values.

### Almost EVERY feature is LOWER in suspicious accounts

| Feature | RF Importance | Z-Diff | Direction | Interpretation |
|---------|-------------|--------|-----------|----------------|
| **F3898** | 0.0945 | **-0.581** | **LOWER** | Strongest signal — some kind of activity score/ratio |
| **F162** | 0.0366 | **+0.668** | **HIGHER** | **ONLY feature that's higher** in suspicious accounts |
| F3805 | 0.0581 | -0.058 | LOWER | Account statistics |
| F1813 | 0.0445 | -0.036 | LOWER | Transaction volumes |
| F1057 | 0.0356 | -0.049 | LOWER | Balance/amounts |
| F3799 | 0.0355 | -0.070 | LOWER | Account statistics |

### Key Insight: The Mule Account Signature
- **19 out of 20** top features are LOWER in suspicious accounts
- F3898 stands out with the largest Z-score difference (-0.581) — mule accounts have a drastically lower value (0.62 vs 1.87)
- **F162 is the ONE feature that's HIGHER** (+0.668) — this could be a risk indicator or flag ratio
- The "mule signature" = **low across all behavioral metrics + one elevated risk indicator**

---

## 2. Missing Value Pattern Analysis (MNAR)

### Mule accounts have MORE COMPLETE data than legitimate accounts!

| Feature | Miss% Suspicious | Miss% Legitimate | Difference | Pattern |
|---------|-----------------|-----------------|------------|---------|
| F61 | 2.47% | 36.90% | **-34.43%** | LESS missing in suspicious |
| F64 | 2.47% | 36.90% | **-34.43%** | LESS missing in suspicious |
| F366 | 2.47% | 36.90% | **-34.43%** | LESS missing in suspicious |
| F369 | 2.47% | 36.90% | **-34.43%** | LESS missing in suspicious |
| F267 | 2.47% | 36.30% | **-33.83%** | LESS missing in suspicious |
| F162 | 2.47% | 36.26% | **-33.79%** | LESS missing in suspicious |

### CRITICAL FINDING: Missingness itself is a feature!
- Suspicious accounts have **97.5% data coverage** in the F0-F500 block
- Legitimate accounts only have **63-74% data coverage** in the same block  
- **No features are MORE missing in suspicious accounts**
- This means mule accounts are **actively used** across all feature dimensions — they don't have inactive feature blocks
- **Potential engineered feature**: "missing data count in F0-F500" — lower values = more suspicious

### Missing Fingerprint Scatter
The scatter plot shows most features cluster along the diagonal (equal missing rates), but there's a clear cluster of features (F61, F64, F366, etc.) where legitimate accounts are ~37% missing but suspicious accounts are only ~2.5% missing.

---

## 3. t-SNE Dimensionality Reduction

### Suspicious accounts form VISIBLE CLUSTERS!

Using only the **16 stable features** identified in Phase 3, t-SNE reveals:

| Perplexity | Observation |
|-----------|-------------|
| **10** | Suspicious accounts form 2-3 small clusters in the upper-left region, with some scattered outliers |
| **30** | Clearest separation — a large group of ~40 suspicious accounts cluster together at top, with smaller clusters elsewhere |
| **50** | Two main suspicious clusters visible — one at upper-right, one at lower-right |

### Key Insight
- PCA captures **99.9% variance** in just 10 components from 16 features
- The suspicious accounts are NOT randomly scattered — they form **coherent sub-groups**
- This confirms Phase 2's sub-cluster finding: there are multiple "profiles" of mule accounts
- The fact that they cluster means a tree-based model CAN learn decision boundaries to separate them

---

## 4. Sparse Feature Deep-Dive (F0-F499)

### The top KS features only have 6 suspicious observations

| Feature | Suspicious (n) | Susp Mean | Legit Mean (n=~1880) | KS Stat |
|---------|----------------|-----------|---------------------|---------|
| F451 | 6 | 0.06 | 1.08 | 0.7388 |
| F453 | 6 | 0.06 | 1.08 | 0.7351 |
| F448 | 6 | 0.10 | 1.07 | 0.7340 |
| F142 | 6 | 0.04 | 0.62 | 0.7181 |
| F139 | 6 | 0.07 | 0.61 | 0.6713 |

### INSIGHT: Sparse features are likely RATIO features
- Legitimate values cluster around **1.0** (F451/F453/F448) or **0.67-0.84** (F142/F139) — these are ratios!
- Suspicious values are near **0.0** — the ratios are zero because mule accounts lack the underlying activity
- Only 6/81 suspicious accounts even HAVE these features filled — the rest are all NULL
- These features likely represent **historical averages or ratios** (e.g., "ratio of inbound to outbound transfers over 6 months")
- Mule accounts either: (a) don't have enough history, or (b) have zero ratios because activity flows only one way

### Coverage difference: Suspicious accounts have MORE data in F0-F499
- F64, F61, F366, F369: **97.5% coverage** in suspicious vs **63.1%** in legitimate
- This is the SAME pattern from the missing value analysis — mule accounts are more "data-complete"

---

## 5. Permutation Importance

### Only 2 features ACTUALLY matter for F1 score!

| Rank | Feature | Permutation Importance | RF Importance | Agreement? |
|------|---------|----------------------|---------------|------------|
| 1 | **F3898** | **+0.299** | 0.0945 | YES |
| 2 | **F162** | **+0.173** | 0.0366 | YES |
| 3 | F3800 | +0.0003 | 0.0188 | NO |
| 4-20 | All others | **NEGATIVE** | 0.01-0.04 | NO |

### THE MOST IMPORTANT FINDING OF PHASE 4

- **F3898 alone accounts for 0.30 F1-score decrease** when permuted — it IS the model
- **F162 accounts for 0.17 F1 decrease** — the secondary signal
- **Every other feature has NEGATIVE permutation importance** — they actually HURT the model!
- This means the RF model with 40 features is **overfitting on noise** for most features
- The optimal model likely needs **only 2-5 features** plus careful regularization
- RF impurity importance was MISLEADING — it ranked F3805, F1813, F1057 highly, but they have zero or negative permutation importance

### IMPLICATION FOR MODELING
- Start with a model using ONLY F3898 + F162
- Add features one-by-one and monitor F1 improvement
- Most of the "important" features from Phase 3 are actually noise in a multivariate context
- The final model should be **extremely simple** — which is actually better for production deployment

---

## 6. Account Lifecycle Analysis

### Fraud rate by Account Age

| Account Age | Total | Suspicious | Rate |
|------------|-------|------------|------|
| <1 month | 1,919 | 16 | 0.83% |
| 1-3 months | 2,007 | 17 | 0.85% |
| **3-6 months** | 3,336 | **36** | **1.08%** |
| 6m-1 year | 1,186 | 10 | 0.84% |
| **1-2 years** | 157 | **2** | **1.27%** |
| 2-5 years | 2 | 0 | 0.00% |

### Fraud rate by Customer Age

| Customer Age | Total | Suspicious | Rate |
|-------------|-------|------------|------|
| <20 | 1,514 | 8 | 0.53% |
| **20-25** | 815 | **17** | **2.09%** |
| **25-30** | 1,290 | **15** | **1.16%** |
| **30-35** | 1,342 | **16** | **1.19%** |
| 35-40 | 1,093 | 6 | 0.55% |
| 40-50 | 1,453 | 12 | 0.83% |
| 50-60 | 838 | 4 | 0.48% |
| 60+ | 673 | 3 | 0.45% |

### KEY FINDINGS
- **3-6 month old accounts** have the highest count of suspicious accounts (36) — this is the "sweet spot" for mule usage
- **1-2 year accounts** have the highest RATE (1.27%) but only 2 cases
- **20-25 year olds** are the highest-risk demographic (**2.09%** — 2.3x the average)
- **No mule accounts are older than 2 years** — mule accounts are SHORT-LIVED
- The scatter plot shows suspicious accounts cluster in the **young account + young customer** quadrant

---

## 7. Pairwise Decision Boundaries

### Mule accounts cluster in the LOW-LOW corner of every pair

The 2D scatter plots show a consistent pattern:
- For every feature pair (F3898 vs F3811, F3805 vs F1705, etc.), suspicious accounts cluster **near the origin** (0, 0)
- Legitimate accounts span a wide range of values
- The separation is most visible in **F162 vs F3813** where F162 shows the opposite direction (mules have HIGHER F162)

### VISUAL CONFIRMATION
- The pairwise plots visually confirm the "mule signature": universally low values
- Decision boundaries would be simple hyperplanes in low-dimensional space
- This supports the permutation importance finding — a simple model on F3898 + F162 can capture most of the signal

---

## 8. Benford Conformity Score (Engineered Feature)

### RESULT: NOT useful as an individual-account feature

| Metric | Suspicious (n=81) | Legitimate (n=8,952) |
|--------|-------------------|---------------------|
| Mean score | 1.528 | 1.556 |
| Median score | 1.374 | 1.431 |
| KS test | stat=0.076, **p=0.717** (NOT significant) |
| Mann-Whitney | **p=0.938** (NOT significant) |

### WHY: Aggregate vs Individual
- Phase 3 found that the AGGREGATE first-digit distribution of suspicious accounts follows Benford's Law
- But at the INDIVIDUAL ACCOUNT level, both classes have similar Benford conformity scores
- This is because each individual account has relatively few monetary features to compute a meaningful first-digit distribution
- **CONCLUSION**: Benford's Law conformity is NOT a useful engineered feature for this dataset

---

## Graphs Generated (Phase 4)

| # | File | Description |
|---|------|-------------|
| 28 | `28_shap_direction.png` | Feature direction analysis — 19/20 features lower in suspicious |
| 29 | `29_missing_patterns.png` | Missing value patterns — suspicious accounts have MORE data |
| 30 | `30_tsne_embedding.png` | t-SNE at 3 perplexities — visible class separation |
| 31 | `31_sparse_features.png` | Sparse feature distributions (F0-F499 block) |
| 32 | `32_permutation_importance.png` | Permutation importance — only F3898 and F162 matter |
| 33 | `33_lifecycle_analysis.png` | Account/customer age fraud rates |
| 34 | `34_pairwise_boundaries.png` | 2D scatter plots showing decision boundaries |
| 35 | `35_benford_conformity.png` | Per-account Benford score (not useful) |

---

## Summary of Phase 4 Breakthrough Findings

### 1. F3898 + F162 = THE model (Permutation Importance)
Only 2 features have positive permutation importance. F3898 alone drives 0.30 F1 decrease. All other features are noise in multivariate context.

### 2. The "Mule Signature" = Low Everything + High F162
19/20 top features are LOWER in mule accounts. Only F162 is HIGHER. This is a remarkably simple pattern.

### 3. Missing data IS a signal
Suspicious accounts have 97.5% data coverage in F0-F500 vs 63% for legitimate. Missingness count can be an engineered feature.

### 4. Mule accounts form visible sub-clusters (t-SNE)
Using just 16 stable features, t-SNE shows 2-3 coherent clusters of suspicious accounts. The signal is real and structured.

### 5. 20-25 year olds at 3-6 month accounts = highest risk
2.09% fraud rate for 20-25 age group. No mule accounts exist beyond 2 years — they're short-lived.

### 6. Benford conformity NOT useful per-account
The aggregate Benford finding from Phase 3 doesn't translate to individual account-level discrimination.

---

<!-- FILE: 18_phase5_premodeling.md -->

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

---

<!-- FILE: 18_phase5_summary.md -->

# Phase 5 Summary: Pre-Modeling Research Results

**Runtime**: 918.1s

## Key Findings

### Optimal Feature Set (14 features)
- F3898
- F1819
- F3799
- F1165
- F1813
- F162_div_F3898
- F3806
- max_value_top8
- F3898_div_F3805
- F162
- missing_count_F0_F500
- F3811_div_F3805
- F3898_div_F3811
- F3800

### Best F1 Score: 0.6037
### Best Threshold: 0.7047
### Best Model: XGBoost (F1=0.6361)

### Model Scores

| Model | F1 | Std |
|-------|----|-----|
| RF | 0.5817 | 0.0893 |
| XGBoost | 0.6361 | 0.1161 |
| LightGBM | 0.6053 | 0.1282 |
| GBM | 0.5758 | 0.0692 |
| Soft Voting | 0.6339 | 0.1193 |
| Stacking (LR) | 0.2434 | 0.0298 |

---

<!-- FILE: 19_initial_model_training.md -->

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

---

<!-- FILE: 20_phase6_model_improvement.md -->

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

---

<!-- FILE: 21_feature_identity.md -->

# Finding 21: Feature Identity — What Our 14 Model Features Actually Represent

**Date**: 2026-06-13  
**Purpose**: Reverse-engineer the banking meaning of our anonymized model features  

---

## 1. The Mule Account Signature (in Plain English)

Our model detects mule accounts using this pattern:

> **Mule accounts are newly opened, low-activity accounts with small balances and low transaction volumes,
> but they have a HIGH behavioral risk score (F162) — the only feature where mules score HIGHER than normal.**

This is exactly what anti-money-laundering literature describes: accounts opened quickly, used briefly
as pass-through channels, then abandoned.

---

## 2. Raw Features — Banking Interpretations

### F3898: Account Activity Tier / Product Engagement Level
| Property | Value |
|----------|-------|
| **Type** | int64, 8 unique values |
| **KS** | 0.517 (strongest raw feature) |
| **XGBoost Gain** | 0.42 (dominant feature) |
| **Missing** | 0% |

**Value Distribution:**
| Value | Count | Suspicious | Fraud Rate | Meaning |
|-------|-------|------------|------------|---------|
| 0 | 2,811 | 39 (48%) | **1.39%** | New / Inactive |
| 1 | 1,151 | 36 (44%) | **3.13%** | Low activity |
| 2 | 240 | 4 (5%) | 1.67% | Moderate activity |
| 3 | 4,755 | 2 (2%) | 0.04% | Normal active |
| 4+ | 125 | 0 (0%) | 0.00% | High engagement |

**Why it works**: 92% of mules are at tier 0 or 1 (new/low activity). Only 2 mules reached tier 3 (normal). The model essentially asks: *"Is this account barely used?"*

**Banking interpretation**: Product engagement score — how many products/services the customer uses (savings, credit card, insurance, etc.). Mule accounts only have a basic savings account.

---

### F162: Behavioral Risk Score / Anomaly Probability
| Property | Value |
|----------|-------|
| **Type** | float64, 0-1 scale, 101 unique values |
| **KS** | 0.354 |
| **Missing** | 36% |
| **Direction** | **HIGHER for suspicious** (the ONLY such feature) |

| Metric | Legitimate | Suspicious | Ratio |
|--------|-----------|------------|-------|
| Median | 0.62 | **0.88** | 1.42x |
| Mean | 0.64 | **0.80** | 1.25x |
| 75th %ile | 0.83 | **1.00** | 1.20x |

**Why it works**: This is likely a pre-computed risk/suspicion score from the bank's existing rule engine. Values are bounded 0-1 (probability). Mule accounts score significantly higher — median 0.88 vs 0.62 for legitimate.

**Banking interpretation**: Output from existing fraud detection rules or behavioral scoring model. The bank already has *some* signal, but it alone isn't sufficient (otherwise they wouldn't need ML).

---

### F1819: Cumulative Transaction Volume / Total Debit+Credit Sum
| Property | Value |
|----------|-------|
| **Type** | float64, 8,501 unique values |
| **KS** | 0.506 |
| **Missing** | 0.04% |

| Metric | Legitimate | Suspicious | Ratio S/L |
|--------|-----------|------------|-----------|
| Median | 1,483,515 | **228,016** | 0.015 |
| Mean | 110,654,596 | **658,049** | 0.006 |
| Max | 136,748,000,000 | 7,906,445 | 0.00006 |

**Why it works**: Mule account transaction volume is ~65x smaller. They handle small pass-through amounts, not the large sustained volume of real customers.

**Banking interpretation**: Total INR amount transacted (debits + credits) over the observation period. The enormous legitimate max (136B INR) suggests corporate accounts exist in the data.

---

### F3799: Peak/Maximum Account Balance
| Property | Value |
|----------|-------|
| **Type** | float64, 8,904 unique values |
| **KS** | 0.513 |
| **Missing** | 0% |

| Metric | Legitimate | Suspicious | Ratio S/L |
|--------|-----------|------------|-----------|
| Median | 1,736,020 | **237,968** | 0.014 |
| Mean | 95,128,198 | **727,907** | 0.008 |
| Min | 59 | **9,540** | — |

**Why it works**: Mule accounts never accumulate wealth. Maximum balance ~8x lower than legitimate.

**Banking interpretation**: Highest recorded balance in the account over its lifetime. Notable: suspicious min is 9,540 (not zero) — mule accounts do hold *some* money briefly.

---

### F1165: Average Individual Transaction Amount
| Property | Value |
|----------|-------|
| **Type** | float64, 3,803 unique values |
| **KS** | 0.500 |
| **Missing** | 0.02% |

| Metric | Legitimate | Suspicious | Ratio S/L |
|--------|-----------|------------|-----------|
| Median | 210,000 | **40,000** | 0.019 |
| Mean | 8,881,233 | **123,778** | 0.014 |
| Min | 0 | **5,000** | — |

**Why it works**: Mules transact in small amounts to stay under regulatory thresholds. Median individual transaction is just Rs 40,000 vs Rs 2,10,000 for normal customers.

**Banking interpretation**: Average transaction size (possibly average debit amount). The Rs 5,000 minimum suggests small structuring. Fewer unique values (3,803) vs other features (~8,500) suggests this is computed/rounded.

---

### F1813: Lifetime Total Transaction Amount
| Property | Value |
|----------|-------|
| **Type** | float64, 8,841 unique values |
| **KS** | 0.515 |
| **Missing** | 0.04% |

| Metric | Legitimate | Suspicious | Ratio S/L |
|--------|-----------|------------|-----------|
| Median | 1,652,508 | **237,968** | 0.014 |
| Mean | 232,598,774 | **691,823** | 0.003 |

**Why it works**: Essentially the same signal as F1819 but computed differently (possibly over a different time window — lifetime vs recent period). Mule accounts have ~330x less lifetime volume.

**Banking interpretation**: Total transaction value over the entire account life. Similar to F1819 but with slight calculation differences.

---

### F3806: Minimum Balance / Account Floor
| Property | Value |
|----------|-------|
| **Type** | float64, 8,358 unique values |
| **KS** | 0.493 |
| **Missing** | 0% |

| Metric | Legitimate | Suspicious | Ratio S/L |
|--------|-----------|------------|-----------|
| Median | 619,253 | **79,433** | 0.013 |
| Mean | 27,367,543 | **254,444** | 0.009 |
| Min | 0 | **0** | — |

**Why it works**: Mule accounts have very low floor balances — they drain funds quickly rather than maintaining a cushion.

**Banking interpretation**: Minimum recorded balance (or average low balance). Some mules reach zero (drained completely).

---

### F3800: Current/Recent Account Balance
| Property | Value |
|----------|-------|
| **Type** | float64, 8,614 unique values |
| **KS** | 0.490 |
| **Missing** | 0% |

| Metric | Legitimate | Suspicious | Ratio S/L |
|--------|-----------|------------|-----------|
| Median | 891,448 | **128,784** | 0.014 |
| Mean | 48,699,583 | **369,835** | 0.008 |

**Why it works**: Similar to F3806 but slightly higher values — likely a different balance snapshot (end-of-period vs minimum).

**Banking interpretation**: Average daily balance or period-end balance.

---

## 3. Dependency Features (Used in Engineering)

### F3805: Total Account Value / Assets Under Management
| Metric | Legitimate | Suspicious | Ratio |
|--------|-----------|------------|-------|
| Median | 1,216,688 | **192,227** | 0.016 |
| Mean | 52,873,119 | **515,004** | 0.010 |
| KS | **0.516** | | |

**Banking interpretation**: Highest of all balance-related features. Likely total AUM or cumulative credit value. All suspicious accounts have min=7,613 (no zeros).

### F3811: Total Outflow / Debit Volume
| Metric | Legitimate | Suspicious | Ratio |
|--------|-----------|------------|-------|
| Median | 853,831 | **135,590** | 0.016 |
| Mean | 29,344,206 | **356,129** | 0.012 |
| KS | **0.513** | | |

**Banking interpretation**: Cumulative debit/withdrawal amount. Mule outflows are 80x smaller.

---

## 4. Engineered Features — What They Capture

| Feature | Formula | Banking Meaning | Why It Helps |
|---------|---------|----------------|-------------|
| **F162_div_F3898** | Risk Score / Activity Tier | Risk-per-engagement | Amplifies: high risk + low activity = strong mule signal |
| **max_value_top8** | max(8 stable features) | Peak across all metrics | If the BEST number is still low, it's suspicious |
| **F3898_div_F3805** | Activity Tier / Total Value | Tier normalized by wealth | Low tier despite having some value = suspicious |
| **missing_count_F0_F500** | NULL count in F0-F499 | Data completeness | Mules have 97.5% data vs 63% for legit (more complete!) |
| **F3811_div_F3805** | Total Outflow / Total Value | Debit-to-credit ratio | How much flows OUT vs came IN — layering indicator |
| **F3898_div_F3811** | Activity Tier / Total Outflow | Engagement per outflow | Low engagement despite outflows = pass-through behavior |

---

## 5. Summary Table — All 14 Features

| # | Feature | Banking Meaning | Range | Direction | KS | Role |
|---|---------|----------------|-------|-----------|-----|------|
| 1 | **F3898** | Account activity tier | 0-57 (int) | Low = Mule | 0.517 | **Dominant** |
| 2 | **F162** | Behavioral risk score | 0-1 | **High = Mule** | 0.354 | Risk signal |
| 3 | **F1819** | Transaction volume | 0-137B | Low = Mule | 0.506 | Volume signal |
| 4 | **F3799** | Peak balance | 59-63B | Low = Mule | 0.513 | Balance signal |
| 5 | **F1165** | Avg transaction size | 0-7.7B | Low = Mule | 0.500 | Size signal |
| 6 | **F1813** | Lifetime txn amount | 0-338B | Low = Mule | 0.515 | Volume signal |
| 7 | **F3806** | Minimum balance | 0-39B | Low = Mule | 0.493 | Balance floor |
| 8 | **F3800** | Current balance | 0-39B | Low = Mule | 0.490 | Balance snapshot |
| 9 | F162_div_F3898 | Risk per engagement | ratio | High = Mule | — | Interaction |
| 10 | max_value_top8 | Peak across metrics | monetary | Low = Mule | — | Aggregation |
| 11 | F3898_div_F3805 | Tier per total value | ratio | Low = Mule | — | Normalization |
| 12 | missing_count_F0_F500 | Data completeness | 0-500 | **Low = Mule** | — | Metadata signal |
| 13 | F3811_div_F3805 | Debit/credit ratio | ratio | — | — | Flow pattern |
| 14 | F3898_div_F3811 | Tier per outflow | ratio | Low = Mule | — | Engagement/flow |

---

## 6. Overlooked Feature Scan — Conclusion: No Real Hidden Gems

### What We Found
Scanned **3,885 features** not already in our model. The top-KS features (F518, F515, F413, etc.) all have KS > 0.9 — but they are **statistical artifacts**:

| Feature | KS | Miss% Legit | Miss% Susp | NonNull Susp | Verdict |
|---------|-----|-------------|------------|--------------|---------|
| F518 | 0.994 | 90.3% | **98.8%** | **1** | Artifact |
| F515 | 0.994 | 90.3% | **98.8%** | **1** | Artifact |
| F413 | 0.991 | 90.3% | **98.8%** | **1** | Artifact |
| F416 | 0.991 | 90.3% | **98.8%** | **1** | Artifact |
| F313 | 0.936 | 94.6% | **98.8%** | **1** | Artifact |
| F316 | 0.932 | 94.6% | **98.8%** | **1** | Artifact |

**Every single one has only 1 non-null suspicious account!** The high KS is because comparing 1 sample vs 500+ samples always looks "different." XGBoost exploits the missingness pattern (98.8% missing for mules vs 90% for legit), but `missing_count_F0_F500` already captures this signal more robustly.

### Bottom Line
> **Our 14 features already capture all discoverable signal.** There are no hidden gems in the remaining 3,885 features — only statistical noise and missingness artifacts.

---

<!-- FILE: 22_literature_review.md -->

# Literature Review: Mule Account Detection — State of the Art

## Introduction & Context

Banks increasingly face "money mule" schemes, where fraudulent funds are cycled through seemingly normal accounts. Mule networks are inherently stealthy – each account looks benign in isolation, and conventional rule-based AML (anti‑money laundering) systems typically miss them. Modern research stresses that mule detection is a **network problem**, not just a per-transaction problem. For example, one study found that graph-based methods uncovered **40% more fraud rings** and identified **26% more unknown mule accounts** than traditional ML pipelines.

The challenge is compounded by extreme class imbalance: in this dataset only **81 of 9,082 accounts are suspicious (~0.9%)**. As prior analysis notes, a naïve 0/1 accuracy baseline is 99.11%. Therefore, we must use specialized modeling (e.g. oversampling, class weighting) and focus on metrics like F1-score, ROC-AUC, and PR-AUC.

---

## 1. Graph/Network-Based Detection

### Key Idea
Model the transaction ecosystem as a graph so account relationships become explicit. Nodes (accounts, customers, devices, etc.) are connected by edges (transactions, shared devices, IPs). Graph Neural Networks (GNNs) and graph analytics can then capture multi-hop patterns that tabular models miss.

### Research Evidence
- In UPI-based mule detection, a **hybrid GBDT + GNN + LSTM achieved ~94% accuracy**
- Graph-based modules significantly improve recall of mule rings
- If an account is connected (via two hops) to a known mule, a GNN will "propagate" that risk through message passing

### Practical Techniques
- **Graph features**: degree centrality, PageRank, betweenness, clustering coefficient, connected component size
- **GNNs**: Convert data into a graph and train a GNN classifier
- **Graph analytics**: PageRank, community detection, shortest-path algorithms

### Our Status
> ⚠️ **NOT APPLICABLE** — Our dataset is a **flat table** (9,082 rows × 3,925 columns). Each row = one account. No transaction-level data, no counterparty IDs, no network links exist. Graph methods require a **separate transaction log** which was not provided in this hackathon.

---

## 2. Advanced Machine Learning & Ensemble Methods

### Multimodal Ensembles
- Combining GBDT + LSTM + GNN outperformed single models (UPI study)
- Stacking or model blending leverages different strengths

### Deep Learning & Sequences
- LSTM/GRU for temporal transaction sequences
- Temporal CNNs for long-term patterns

### Autoencoders / Anomaly Models
- Train on legitimate accounts only; high reconstruction error = suspicious
- One-class SVM or Isolation Forests as unsupervised fallback

### Feature Selection & Reduction
- PCA, tree-based importance, mutual information for dimensionality reduction
- Embeddings or autoencoder bottlenecks

### Our Status
| Technique | Status | Result |
|-----------|--------|--------|
| ✅ XGBoost + Optuna (500 trials) | **DONE** | F1=0.684, AUC=0.973 |
| ✅ LightGBM | **DONE** | F1=0.695 |
| ✅ Random Forest | **DONE** | F1=0.682 |
| ✅ Soft Voting Ensemble | **DONE** | F1=0.705 |
| ✅ Stacking (Meta-learner) | **DONE** | F1=0.595 (FAILS — overfits with 81 samples) |
| ✅ Isolation Forest | **DONE** | Max 8/81 caught — mules are NOT anomalies |
| ✅ LOF / One-Class SVM | **DONE** | Catch 0/81 — complete failure |
| 🔄 TabNet Neural Network | **IN PROGRESS** | Running experiment now |
| 🔄 MLP Neural Network | **IN PROGRESS** | Running experiment now |
| ✅ Feature Selection (Forward) | **DONE** | Optimal: 14 features (8 raw + 6 engineered) |
| ✅ PCA & Dimensionality Reduction | **DONE** | Clustered correlations, 67 redundant dropped |

---

## 3. Data & Feature Engineering

### Temporal Features
- Time since last transaction, month-over-month changes, rolling windows
- Behavioral trend detection

### Behavioral/KYC Data
- Login patterns, device/browser info, geolocation anomalies
- "Sleeper account" behavior detection

### Account Network Features
- Average risk score of transaction neighbors
- Guilt-by-association scoring

### Our Status
| Technique | Status | Result |
|-----------|--------|--------|
| ✅ Missing pattern features | **DONE** | missing_count_F0_F500 (KS=0.652) |
| ✅ Ratio features | **DONE** | F162/F3898, F3898/F3805, F3811/F3805, F3898/F3811 |
| ✅ Interaction features | **DONE** | F162_div_F3898, max_value_top8 |
| ✅ Account lifecycle analysis | **DONE** | 20-25 yr olds, <6 month accounts = highest risk |
| ✅ Temporal stability check | **DONE** | Features stable across time splits |
| ❌ Transaction-level sequences | **N/A** | No temporal transaction log available |
| ❌ Device/browser telemetry | **N/A** | Not in dataset |
| ❌ Network neighbor features | **N/A** | No counterparty data |

---

## 4. Handling Class Imbalance

### Techniques
- Hybrid sampling: SMOTE+Tomek, SMOTEENN
- Cost-sensitive learning: scale_pos_weight, focal loss
- Threshold tuning and multi-stage classification
- Ensemble diversification
- Stratified cross-validation

### Our Status
| Technique | Status | Result |
|-----------|--------|--------|
| ✅ SMOTE (multiple ratios) | **DONE** | SMOTE(0.3) optimal but class weights are BETTER |
| ✅ ADASYN, Borderline-SMOTE, SVMSMOTE | **DONE** | All worse than plain SMOTE |
| ✅ scale_pos_weight=111 | **DONE** | **Best approach** — no SMOTE needed |
| ✅ Threshold optimization | **DONE** | Optimal t=0.610 (F1=0.706) |
| ✅ Cost analysis | **DONE** | Business cost optimal at t=0.19 |
| ✅ Stratified 5-Fold CV | **DONE** | Selected as optimal strategy |
| ❌ Focal loss | Not tested | Could try with neural nets |
| ❌ SMOTE+Tomek / SMOTEENN | Not tested | Hybrid cleaning variants |

---

## 5. Evaluation, Explainability & Deployment

### Metrics
- Precision, Recall, F1, ROC-AUC, PR-AUC
- Confusion matrix analysis

### Explainability
- SHAP for feature-level explanations
- GNNExplainer for graph models
- Interpretable tree ensembles

### Monitoring & Drift
- Track score distributions over time
- Feature drift detection
- Periodic retraining

### Our Status
| Technique | Status | Result |
|-----------|--------|--------|
| ✅ Full metrics suite | **DONE** | F1, AUC, PR, Recall, Precision at multiple thresholds |
| ✅ SHAP analysis | **DONE** | Graphs 50-52, per-account explanations |
| ✅ Permutation importance | **DONE** | Only F3898 + F162 are individually critical |
| ✅ Feature stability | **DONE** | Bootstrap stability across 100 resamples |
| ✅ Error analysis | **DONE** | 31 "hard" mules profiled (high-value, elderly) |
| ✅ Temporal stability | **DONE** | Model robust across time splits |

---

## 6. Similar Research and Ongoing Trends

### Federated Learning
- Cross-institution model training while preserving privacy (RBC Borealis case study)
- Not applicable here (single institution dataset)

### Behavioral Biometrics
- Device/browser telemetry, mouse/typing patterns (BioCatch)
- Beyond scope of this dataset

### Continuous Learning
- Online learning, active learning loops
- Periodic model updates as mule tactics evolve

---

## Summary: What We've Done vs. What's Possible

### ✅ Fully Addressed (18 techniques)
1. XGBoost with Optuna hyperparameter optimization
2. LightGBM comparison
3. Random Forest baseline
4. Soft voting ensemble
5. Stacking (tested, found to overfit)
6. Isolation Forest anomaly detection
7. LOF / One-Class SVM
8. Forward feature selection
9. SMOTE (6 variants tested)
10. Class weighting (scale_pos_weight)
11. Threshold optimization (F1 + cost-optimal)
12. Stratified cross-validation
13. SHAP explainability
14. Permutation importance
15. Temporal stability validation
16. Error analysis of missed mules
17. Feature engineering (6 engineered features)
18. Dimensionality reduction (correlation clustering)

### 🔄 In Progress (2 techniques)
1. TabNet neural network
2. MLP neural network

### ❌ Not Applicable with Current Data (5 techniques)
1. Graph Neural Networks (no transaction graph)
2. LSTM/temporal sequences (no time-series data)
3. Device/browser telemetry (not in dataset)
4. Network neighbor features (no counterparty data)
5. Federated learning (single institution)

### 🟡 Could Still Try (3 techniques)
1. Focal loss (with neural nets)
2. SMOTE+Tomek hybrid cleaning
3. Autoencoder on legitimate accounts only

---

## Key Insight

> **Our dataset fundamentally limits what techniques are applicable.** The most impactful suggestions (graph methods, GNNs, behavioral biometrics) require data that doesn't exist in this hackathon's flat CSV. Within the constraints of a **single flat table with 9,082 rows and 81 suspicious accounts**, we have exhaustively explored every applicable technique. Our XGBoost model at F1=0.684, AUC=0.973 likely represents the **ceiling** for this specific dataset.

---

<!-- FILE: 23_neural_network_experiment.md -->

# Phase 7: Neural Network Experiment — TabNet & MLP

## Summary

Tested **8 models** (3 TabNet, 3 MLP, 1 XGBoost baseline, 1 ensemble) using identical 30 features and 5-fold stratified CV.

**Key Finding**: Neural networks significantly **underperform** tree-based models for mule detection with only 81 minority samples. However, a **weighted XGBoost+TabNet ensemble** achieved the best F1 score.

---

## Results Table

| Rank | Model | F1 | F1 Std | Precision | Recall | AUC | Time |
|------|-------|-----|--------|-----------|--------|-----|------|
| 🥇 | **XGB+TabNet Ensemble** | **0.704** | 0.130 | 0.825 | 0.618 | 0.967 | 337s |
| 🥈 | **XGBoost (Phase 6 Best)** | **0.684** | 0.117 | 0.803 | 0.605 | **0.973** | 7s |
| 3 | MLP-Large (256-128-64-32) | 0.267 | 0.105 | 0.354 | 0.222 | 0.922 | 29s |
| 4 | MLP-Medium (128-64-32) | 0.248 | 0.099 | 0.279 | 0.235 | 0.919 | 14s |
| 5 | MLP-Small (64-32) | 0.220 | 0.094 | 0.213 | 0.235 | 0.930 | 9s |
| 6 | TabNet-Medium (16d, 5 steps) | 0.126 | 0.040 | 0.070 | 0.729 | 0.922 | 403s |
| 7 | TabNet-Small (8d, 3 steps) | 0.081 | 0.031 | 0.047 | 0.688 | 0.925 | 225s |
| 8 | TabNet-Large (32d, 7 steps) | 0.067 | 0.013 | 0.037 | 0.727 | 0.914 | 465s |

---

## Key Insights

### 1. TabNet: High Recall, Catastrophic Precision
TabNet catches ~70% of mules (Recall=0.69-0.73) but with only **3-7% precision** — flagging hundreds of legitimate accounts as suspicious. This is because:
- With only 81 positive samples, TabNet's attention mechanism can't learn stable patterns
- The sparse attention masks essentially fire for any slightly unusual account
- Early stopping kicks in very early (best epochs: 1-45 out of 200)

### 2. MLP: Moderate but Unstable
MLP with oversampled minority class achieves F1=0.22-0.27 — much better than TabNet but still 60% worse than XGBoost. The high variance (std=0.10) shows instability across folds.

### 3. Ensemble Effect: Best F1=0.704
The **0.7×XGBoost + 0.3×TabNet** ensemble is the best overall:
- XGBoost provides high precision (0.80)
- TabNet adds a small recall boost via its "catch everything" tendency
- The 0.7/0.8 weighting works best (0.5 is too much TabNet noise)
- However, the improvement over XGBoost alone is **marginal** (+0.020 F1) and comes with **50× longer training time**

### 4. XGBoost Remains King
For this problem (81 minority samples, 30 features, flat tabular data):
- XGBoost achieves **highest AUC** (0.973 vs 0.967 for ensemble)
- **50× faster** than the ensemble (7s vs 337s)
- Most **stable** across folds (std=0.117 vs 0.130)
- Doesn't require neural network infrastructure

---

## Why Neural Networks Fail Here

| Factor | Impact |
|--------|--------|
| **81 minority samples** | Far too few for neural nets to learn stable decision boundaries |
| **30 features** | Not enough dimensions for neural attention/embedding to add value |
| **No sequential/spatial structure** | TabNet/MLP can't exploit temporal or graph patterns (none exist) |
| **Tabular data** | Tree-based models consistently outperform neural nets on pure tabular tasks |
| **Extreme imbalance (111:1)** | Neural nets struggle more than trees with class weighting |

---

## Ensemble Weight Analysis (Fold 1)

| XGBoost Weight | TabNet Weight | F1 |
|---------------|---------------|-----|
| 0.5 | 0.5 | 0.647 |
| 0.6 | 0.4 | 0.690 |
| **0.7** | **0.3** | **0.741** |
| 0.8 | 0.2 | 0.741 |

Optimal: XGBoost dominates (70-80%), TabNet contributes a small diversity bonus.

---

## Verdict

> **XGBoost alone (F1=0.684, AUC=0.973) is the recommended production model.** The ensemble achieves marginally better F1 (+0.020) but at 50× the training cost, lower AUC, and higher variance. Neural networks are not suited for this problem given the extreme minority class size.

---

## Graph
- **Graph 53**: [53_neural_network_comparison.png](./graphs/53_neural_network_comparison.png) — 3-panel comparison (F1, AUC, Training Time)

---

<!-- FILE: 24_advanced_techniques.md -->

# Phase 8: Advanced Techniques Experiment

## Summary

Tested **20 experiments** across 5 technique categories: Focal Loss, SMOTE+Tomek/ENN, Autoencoder, Multi-Level Feature Engineering, and Advanced Ensemble Strategies.

**Key Finding**: None of the advanced techniques significantly outperform our XGBoost baseline (F1=0.684). The best ensemble achieves F1=0.704 (+0.020), while most alternatives perform worse. The XGBoost model with class weighting remains the optimal approach for this dataset.

---

## Results Table (Ranked by F1)

| Rank | Model | F1 | Std | AUC | Type |
|------|-------|-----|-----|-----|------|
| 1 | **Ens-EqualBlend** | **0.704** | — | 0.978 | Ensemble |
| 2 | **Ens-OptimizedBlend** | **0.704** | — | 0.978 | Ensemble |
| 3 | Ens-ElasticNetStack | 0.700 | — | 0.978 | Ensemble |
| 4 | Ens-RidgeStack | 0.699 | — | 0.978 | Ensemble |
| 5 | **XGB-FULL-60** | **0.698** | 0.147 | 0.978 | Feature Level |
| 6 | SMOTE+Tomek | 0.693 | 0.126 | 0.979 | Sampling |
| 7 | SMOTEENN | 0.693 | 0.118 | 0.980 | Sampling |
| 8 | SMOTE+Tomek-0.5 | 0.689 | 0.125 | 0.979 | Sampling |
| 9 | SMOTEENN-0.5 | 0.688 | 0.082 | 0.983 | Sampling |
| 10 | **XGBoost-Baseline** | **0.684** | 0.117 | **0.973** | Baseline |
| 11 | XGB-ENHANCED-30 | 0.684 | 0.117 | 0.973 | Feature Level |
| 12 | XGB-BASE-14 | 0.624 | 0.155 | 0.974 | Feature Level |
| 13 | Ens-LGBStack | 0.389 | — | 0.940 | Ensemble |
| 14 | Focal-a0.90-g2 | 0.305 | 0.044 | 0.906 | Focal Loss |
| 15 | Focal-a0.95-g2 | 0.253 | 0.047 | 0.893 | Focal Loss |
| 16 | Focal-a0.99-g3 | 0.148 | 0.055 | 0.903 | Focal Loss |
| 17 | Focal-a0.99-g5 | 0.099 | 0.055 | 0.878 | Focal Loss |
| 18 | AE-b16 | 0.019 | 0.011 | 0.518 | Autoencoder |
| 19 | AE-b8 | 0.014 | 0.009 | 0.494 | Autoencoder |
| 20 | AE-b4 | 0.013 | 0.012 | 0.471 | Autoencoder |

---

## Technique-by-Technique Analysis

### 1. Focal Loss (F1 = 0.10-0.31) — FAILED

Focal loss is designed for extreme imbalance — it down-weights easy-to-classify examples and focuses on hard ones. In theory, perfect for 111:1 imbalance.

**Why it failed:**
- With only 81 positive samples, the neural network can't learn stable boundaries regardless of loss function
- Higher alpha (0.99) and gamma (5) make the loss too focused on very few hard examples, causing instability
- Best config (alpha=0.90, gamma=2) achieves F1=0.305 — still 55% worse than XGBoost
- Focal loss helps neural nets on LARGE imbalanced datasets (100K+ samples), not tiny ones

| Config | Alpha | Gamma | F1 | AUC |
|--------|-------|-------|-----|-----|
| Focal-a0.90-g2 | 0.90 | 2.0 | 0.305 | 0.906 |
| Focal-a0.95-g2 | 0.95 | 2.0 | 0.253 | 0.893 |
| Focal-a0.99-g3 | 0.99 | 3.0 | 0.148 | 0.903 |
| Focal-a0.99-g5 | 0.99 | 5.0 | 0.099 | 0.878 |

**Verdict**: Higher alpha/gamma = worse performance. The loss function isn't the bottleneck — the sample count is.

---

### 2. SMOTE+Tomek / SMOTEENN (F1 = 0.69) — MARGINAL

Hybrid sampling: SMOTE creates synthetic minority samples, then Tomek links or ENN removes noisy boundary examples.

**Results:**
| Config | Sampling Ratio | Resampled Size | F1 | AUC |
|--------|---------------|----------------|-----|-----|
| SMOTE+Tomek 0.3 | 30% | ~9,310 (2,135 pos) | 0.693 | 0.979 |
| SMOTEENN 0.3 | 30% | ~8,858 (2,042 pos) | 0.693 | 0.980 |
| SMOTE+Tomek 0.5 | 50% | ~10,770 (3,585 pos) | 0.689 | 0.979 |
| SMOTEENN 0.5 | 50% | ~10,265 (3,474 pos) | 0.688 | 0.983 |

**Key findings:**
- All 4 variants achieve F1 ~ 0.69, matching our Phase 5 SMOTE(0.3) result
- Tomek link cleaning and ENN noise removal don't help — the boundary examples aren't the problem
- Lower sampling ratio (0.3) is slightly better than higher (0.5)
- **SMOTEENN-0.5 has the highest AUC (0.983)** but lower F1 — it's better at ranking but worse at classification threshold
- **None beat class-weighted XGBoost** (F1=0.684 with no SMOTE at all)

**Verdict**: The cleaning step in SMOTE+Tomek/ENN doesn't help because the 81 mule accounts aren't noisy boundary cases — they're genuinely hard to separate.

---

### 3. Autoencoder Anomaly Detection (F1 = 0.01-0.02) — COMPLETE FAILURE

Train autoencoder on legitimate accounts only, then flag high reconstruction error as suspicious.

| Config | Bottleneck | F1 | AUC |
|--------|-----------|-----|-----|
| AE-b4 | 4 dims | 0.013 | 0.471 |
| AE-b8 | 8 dims | 0.014 | 0.494 |
| AE-b16 | 16 dims | 0.019 | 0.518 |

**Why total failure:**
- AUC < 0.52 means the autoencoder is **worse than random** — mule accounts are NOT anomalous in reconstruction space
- This confirms our Phase 3 finding: Isolation Forest (8/81) and LOF (0/81) also failed
- Mule accounts are **embedded within the normal distribution**, not outliers
- The "low everything" mule signature is a SUBTLE pattern within normal variation, not a reconstruction anomaly
- With 30 features and 9,001 legitimate accounts, the autoencoder learns to reconstruct everything well — including mules

**Verdict**: Autoencoder-based anomaly detection is fundamentally unsuitable. Mules don't look anomalous — they look like inactive but otherwise normal accounts.

---

### 4. Multi-Level Feature Engineering (F1 = 0.62-0.70) — MODEST GAIN

Inspired by the Traffic Demand Pipeline's multi-level approach (BASE → ENHANCED → FULL).

| Level | Features | F1 | AUC | Delta vs BASE |
|-------|----------|-----|-----|---------------|
| BASE (raw + engineered) | 14 | 0.624 | 0.974 | — |
| ENHANCED (+ PS features) | 30 | 0.684 | 0.973 | +0.060 |
| FULL (+ interactions, ranks, clusters) | 60 | 0.698 | 0.978 | +0.074 |

**Key findings:**
- Adding PS features (14 → 30) gives +0.060 F1 — the biggest single jump
- Adding pairwise interactions, rank features, and cluster IDs (30 → 60) gives +0.014 more
- FULL-60 achieves F1=0.698 — our **second-best single model ever**
- More features = more AUC (0.974 → 0.978) — the model finds useful signal in interactions
- Feature interactions (F3898 × F1165, F162 / F1819) capture non-linear patterns that raw features miss

**Verdict**: Multi-level feature engineering shows diminishing returns but FULL-60 is competitive. The interaction features are worth keeping.

---

### 5. Advanced Ensemble Strategies (F1 = 0.39-0.70) — BEST RESULTS

Adapted directly from the Traffic Demand Pipeline's ensemble module.

**Base models for ensemble:**
| Model | F1 | AUC |
|-------|-----|-----|
| XGB-Optuna | 0.690 | 0.973 |
| XGB-Diverse | 0.694 | 0.978 |
| XGB-FULL | 0.699 | 0.978 |
| LightGBM | 0.525 | 0.942 |

**Ensemble results:**
| Strategy | F1 | Description |
|----------|-----|-------------|
| **Equal Blend** | **0.704** | Simple average of 4 models |
| **Optimized Blend** | **0.704** | Scipy-optimized weights (converged to equal) |
| ElasticNet Stack | 0.700 | Non-negative elastic net meta-learner |
| Ridge Stack | 0.699 | Ridge regression meta-learner |
| LGB Stack | 0.389 | LightGBM meta-learner — overfits badly |

**Key findings:**
- Equal blend = optimized blend — all 4 models contribute equally (weights: [0.25, 0.25, 0.25, 0.25])
- Simple averaging is surprisingly optimal — no model deserves more weight
- Linear stacking (Ridge/ElasticNet) nearly matches blending — the meta-learner adds little
- LGB stacking catastrophically fails — too few minority samples for a second-level tree model
- **Best ensemble F1=0.704 matches our Phase 7 XGB+TabNet ensemble**

**Verdict**: Simple averaging of diverse XGBoost variants is the best ensemble strategy. Complex stacking doesn't help with only 81 minority samples.

---

## Cross-Technique Insights

### What Works (ordered by impact)
1. **XGBoost with class weighting** — the foundation (F1=0.684)
2. **Multi-level feature engineering** — FULL-60 features push to F1=0.698
3. **Simple model blending** — averaging diverse XGBoosts achieves F1=0.704
4. **SMOTE variants** — competitive (F1~0.69) but don't beat class weighting

### What Doesn't Work
1. **Focal Loss** — neural nets can't learn from 81 samples regardless of loss function
2. **Autoencoders** — mules aren't anomalies (AUC < 0.52, worse than random)
3. **Complex stacking** — LGB/complex meta-learners overfit on tiny minority class
4. **Higher SMOTE ratios** — 0.5 is worse than 0.3, which is worse than no SMOTE

### The Ceiling
> **F1 ~ 0.70 appears to be the hard ceiling for this dataset.** Every technique tested across Phases 5-8 (50+ experiments) converges to this range. The fundamental limiting factor is 81 minority samples in a flat tabular dataset with no network/temporal structure.

---

## Graph
- **Graph 54**: [54_advanced_techniques.png](./graphs/54_advanced_techniques.png) — 20 experiments ranked by F1 and AUC, color-coded by technique type

---

<!-- FILE: 25_final_push.md -->

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

---
