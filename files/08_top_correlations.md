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
