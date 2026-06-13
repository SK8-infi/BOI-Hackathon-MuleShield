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
