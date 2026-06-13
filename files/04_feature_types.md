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
