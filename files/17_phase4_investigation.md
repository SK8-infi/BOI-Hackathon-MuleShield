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
