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
