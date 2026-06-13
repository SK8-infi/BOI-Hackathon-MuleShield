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
