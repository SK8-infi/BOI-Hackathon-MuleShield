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
