# Mule Account Detection - Evaluation Summary

**Model**: XGBoost (Optuna-tuned)
**Features**: 14 (8 raw + 6 engineered)
**ROC-AUC**: 0.9720
**Average Precision**: 0.6003

## Performance at Different Thresholds

| Threshold | F1 | Precision | Recall | TP | FP | FN |
|-----------|-----|-----------|--------|----|----|-----|
| cost_optimal (0.2) | 0.470 | 0.349 | 0.716 | 58 | 108 | 23 |
| balanced (0.5) | 0.556 | 0.491 | 0.642 | 52 | 54 | 29 |
| f1_optimal (0.947) | 0.638 | 0.772 | 0.543 | 44 | 13 | 37 |

## Features Used

### Raw Features
- F3898
- F1819
- F3799
- F1165
- F1813
- F3806
- F162
- F3800

### Engineered Features
- F162_div_F3898
- max_value_top8
- F3898_div_F3805
- missing_count_F0_F500
- F3811_div_F3805
- F3898_div_F3811