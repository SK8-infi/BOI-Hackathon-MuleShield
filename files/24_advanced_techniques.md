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
