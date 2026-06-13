# BOI Hackathon — Problem Statement 2: Research Findings

## AI/ML-Based Classification of Suspicious Mule Accounts

**Date**: June 13, 2026  
**Dataset**: `DataSet.csv` (111 MB, 9,082 rows × 3,925 columns)  
**Target Variable**: `F3924` (binary: 0 = legitimate, 1 = suspicious/mule)

---

## 📁 Files Index

| File | Description |
|------|-------------|
| [01_dataset_overview.md](./01_dataset_overview.md) | Dataset shape, types, memory, and high-level statistics |
| [02_target_variable.md](./02_target_variable.md) | Class imbalance analysis and implications |
| [03_missing_values.md](./03_missing_values.md) | Column-level and row-level missing value analysis |
| [04_feature_types.md](./04_feature_types.md) | Classification of all 3,925 columns by statistical properties |
| [05_feature_blocks.md](./05_feature_blocks.md) | Thematic groupings of features by number range |
| [06_key_features.md](./06_key_features.md) | Deep analysis of the 18 features suggested in the problem statement |
| [07_categorical_features.md](./07_categorical_features.md) | All 8 categorical features with value distributions and fraud rates |
| [08_top_correlations.md](./08_top_correlations.md) | Top 30 features most correlated with the target |
| [09_data_leakage.md](./09_data_leakage.md) | F3912 and F2230 leakage analysis — MUST READ before modeling |
| [10_suspicious_profile.md](./10_suspicious_profile.md) | Profile of the 81 suspicious/mule accounts |
| [11_feature_interpretation.md](./11_feature_interpretation.md) | Reverse-engineered meanings of anonymized columns |
| [12_recommendations.md](./12_recommendations.md) | Recommended ML pipeline, models, and evaluation strategy |
| [13_graph_analysis.md](./13_graph_analysis.md) | Detailed analysis of 14 visualizations (Phase 1 graphs) |
| [14_deep_investigation.md](./14_deep_investigation.md) | Phase 2: MI, redundancy, rule discovery, clustering, interactions |
| [15_phase3_investigation.md](./15_phase3_investigation.md) | Phase 3: Stat tests, anomaly detection, Benford's law, RF baseline |
| [16_ml_techniques_research.md](./16_ml_techniques_research.md) | ML techniques research: XGBoost, SMOTE, PyOD, GNNs, AutoML |
| [17_phase4_investigation.md](./17_phase4_investigation.md) | Phase 4: SHAP, missing patterns, t-SNE, permutation importance, lifecycle |
| [18_phase5_premodeling.md](./18_phase5_premodeling.md) | Phase 5: Feature engineering, forward selection, CV strategy, SMOTE, error analysis, threshold, stacking, temporal |
| [18_phase5_summary.md](./18_phase5_summary.md) | Phase 5: Summary of optimal feature set and model scores |
| [19_initial_model_training.md](./19_initial_model_training.md) | Initial production pipeline: Optuna tuning, SHAP, evaluation, model artifacts |
| [20_phase6_model_improvement.md](./20_phase6_model_improvement.md) | Phase 6: PS feature investigation, leakage discovery, 500 Optuna trials, model comparison |
| [21_feature_identity.md](./21_feature_identity.md) | Feature identity: reverse-engineered banking meanings of all 14 model features |
| [22_literature_review.md](./22_literature_review.md) | Literature review: state-of-the-art mule detection techniques vs our implementation |
| [23_neural_network_experiment.md](./23_neural_network_experiment.md) | Phase 7: TabNet, MLP, ensemble experiment — neural networks vs XGBoost |
| [24_advanced_techniques.md](./24_advanced_techniques.md) | Phase 8: Focal loss, SMOTE+Tomek, autoencoder, multi-level features, advanced ensembles |
| [25_final_push.md](./25_final_push.md) | Phase 9: Final push — Optuna on FULL-60, 13-model blend, **NEW BEST F1=0.743** |

---

## 📊 Graphs (55 total in `graphs/` folder)

- **01-14**: Phase 1 — Class imbalance, missing values, feature types, correlations, profiles, boxplots, heatmaps
- **15-19**: Phase 2 — Mutual information, clustered correlations, clustering, interactions, combined ranking
- **20-27**: Phase 3 — Statistical tests, anomaly detection, temporal, categorical, tree importance, Benford's law, hardest-to-catch, feature stability
- **28-35**: Phase 4 — SHAP direction, missing patterns, t-SNE embedding, sparse features, permutation importance, lifecycle, pairwise boundaries, Benford conformity
- **36-43**: Phase 5 — Feature engineering, forward selection, CV strategies, SMOTE comparison, error analysis, threshold optimization, model stacking, temporal stability
- **44-49**: Phase 6 — PS features analysis, feature set comparison, model variants, PS vs Ours, PS contributions, final model summary
- **50-52**: Final Model — SHAP beeswarm, SHAP bar (color-coded by feature source), top-5 account explanations
- **53**: Phase 7 — Neural network vs tree-based model comparison (F1, AUC, training time)
- **54**: Phase 8 — Advanced techniques comparison (20 experiments, 6 technique types)
- **55**: Phase 9 — Final push: individual models, threshold optimization, F1 improvement journey

---

## ⚡ Quick Summary of Critical Findings

### Phase 1 — Dataset Understanding
1. **Extreme class imbalance**: 111:1 ratio (81 suspicious out of 9,082)
2. **Data leakage**: F3912 (fraud flag, 0.97 corr) and F2230 (temporal separator) MUST be excluded
3. **F3887 = Account age in days** (confirmed: 0.9999 corr with computed age from F3888)
4. **F3894 = Customer age in years** (range -2 to 94, median 34)
5. **2,520 usable features** after filtering (out of 3,925)
6. **Features follow thematic blocks**: proportions → ratios → flags → amounts → counts → deltas
7. **F2956 and F3043 are near-duplicates** (0.98 correlation)
8. **Students** (1.94%) and **rural areas** (1.44%) have the highest suspicious rates

### Phase 2 — Deep Investigation
9. **MI vs correlation disagree** — top MI features (F1000-1999) differ from top correlation features (F2500-2999)
10. **268 near-duplicate pairs** — 67 redundant features can be safely dropped
11. **2-rule combos achieve 49% precision** — F3805 ≤ 110K AND F1705 ≤ 90K flags 53 accounts, catches 26/81
12. **Feature interactions boost MI by +113%** — F3805-based differences are the strongest signal

### Phase 3 — Comprehensive Analysis
13. **🚨 Mule accounts are NOT anomalies** — Isolation Forest catches max 8/81, LOF catches 0. Must use supervised learning
14. **🚨 Categorical features have ZERO discriminative power** — Student+Rural+Savings combination has 0% fraud in the dataset
15. **Benford's Law reversal** — Mule accounts FOLLOW Benford's Law, legitimate accounts DON'T
16. **RF baseline: F1=0.682, AUC=0.969** — Strong discrimination, precision 89%, recall 58%
17. **8 rock-solid features** appear in 80%+ of bootstrap samples: F3811, F3806, F3799, F3805, F3813, F3801, F3898, F3807
18. **All 81 accounts are catchable** (RF score > 0.5), but hardest are atypical mules (elderly, metro, salaried)

### Phase 4 — Final Deep Dive
19. **F3898 + F162 = THE model** — Only 2 features have positive permutation importance. F3898 alone drives 0.30 F1 decrease
20. **The Mule Signature** — 19/20 top features are LOWER in suspicious. Only F162 is HIGHER (risk indicator)
21. **Missing data IS a signal** — Suspicious accounts have 97.5% data coverage in F0-F500 vs 63% for legitimate
22. **Mule accounts form visible sub-clusters** — t-SNE with 16 features shows 2-3 coherent suspicious clusters
23. **20-25 year olds at 3-6 month accounts = highest risk** — 2.09% fraud rate; no mules exist beyond 2 years
24. **Benford conformity NOT useful per-account** — Aggregate finding doesn't translate to individual-level discrimination

### Phase 5 — Pre-Modeling Research
25. **14 optimal features** (8 raw + 6 engineered) — Forward selection peaked at F1=0.604; engineered features account for 43% of optimal set
26. **low_value_count is strongest engineered feature** (KS=0.652) — Captures the "low everything" mule signature
27. **SMOTE(0.3) is optimal oversampling** (F1=0.621) — Higher ratios HURT; fancy variants (ADASYN, Borderline) all worse
28. **Stratified 5-Fold is best CV** (F1=0.604, std=0.080) — 10-Fold has too few mules per fold; repeated splits don't help
29. **XGBoost is best model** (F1=0.636) — Soft voting matches it; stacking FAILS with only 81 minority samples
30. **31/81 missed mules are high-value accounts** — F3811 5.7x higher, F3805 6.3x higher than caught mules
31. **Optimal threshold = 0.2 for cost, 0.705 for F1** — Business cost minimum at threshold 0.19 ($150K vs $315K at default)
32. **F2230 PERFECTLY separates classes** — Oct25=0% fraud, Sep/Nov/Dec=100% fraud. All 81 mules in non-October periods

### Phase 6 — Model Improvement (Production Pipeline)
33. **🚨 Third leakage source found**: `Unnamed: 0` (CSV row index) achieves F1=0.849 alone — mules clustered at rows 9001-9082
34. **SMOTE is counterproductive** — Removing SMOTE improved F1 from 0.628 to 0.715 (+13.8%)
35. **PS features alone are weak** (F1=0.332) — Our data-driven features are 2x better (F1=0.636)
36. **Only F531 helps individually** (+0.01) — 16 of 18 PS features HURT the model when added one at a time
37. **Combined features are best** — Our 14 + PS 18 = F1=0.682 (best of 6 feature sets tested)
38. **XGBoost with 500 Optuna trials = F1=0.7151** — 12.4% improvement over Phase 5
39. **Stronger regularization is key** — colsample=0.358, reg_lambda=1.44 (53x more than Phase 5)
40. **Final model at t=0.460**: F1=0.727, P=0.839, R=0.642, only 10 false positives

