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
