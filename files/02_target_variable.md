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
