# Phase 5 Summary: Pre-Modeling Research Results

**Runtime**: 918.1s

## Key Findings

### Optimal Feature Set (14 features)
- F3898
- F1819
- F3799
- F1165
- F1813
- F162_div_F3898
- F3806
- max_value_top8
- F3898_div_F3805
- F162
- missing_count_F0_F500
- F3811_div_F3805
- F3898_div_F3811
- F3800

### Best F1 Score: 0.6037
### Best Threshold: 0.7047
### Best Model: XGBoost (F1=0.6361)

### Model Scores

| Model | F1 | Std |
|-------|----|-----|
| RF | 0.5817 | 0.0893 |
| XGBoost | 0.6361 | 0.1161 |
| LightGBM | 0.6053 | 0.1282 |
| GBM | 0.5758 | 0.0692 |
| Soft Voting | 0.6339 | 0.1193 |
| Stacking (LR) | 0.2434 | 0.0298 |
