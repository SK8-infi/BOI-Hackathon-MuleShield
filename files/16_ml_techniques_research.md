# ML Techniques Research — Out-of-the-Box Approaches

**Date**: June 13, 2026  
**Source**: User research notes  

---

## Recommended Approaches for Mule-Account Detection

### 1. Gradient-Boosted & Ensemble Models
- **XGBoost, LightGBM, CatBoost, Random Forests** — proven fraud baselines
- XGBoost: `scale_pos_weight` parameter + native missing value handling
- CatBoost: best on categorical data, handles missing values
- imbalanced-learn: BalancedRandomForest, EasyEnsemble, RUSBoost

### 2. Resampling and Imbalance Handling
- **SMOTE, BorderlineSMOTE, ADASYN** (imbalanced-learn)
- SMOTE + Tomek links / EditedNN for overlap cleaning
- **CTGAN** — conditional GANs for synthetic minority samples
- All plug-and-play via Pipeline or fit_resample

### 3. Anomaly/Outlier Detection (PyOD)
- **60+ algorithms**: IsolationForest, LOF, CBLOF, COPOD, HBOS, AutoEncoder
- One-class approaches: One-Class SVM, Deep One-Class
- ⚠️ NOTE: Our Phase 3 found IF/LOF perform **very poorly** on this dataset (mules blend in)

### 4. Graph and Network Techniques
- Transaction graph analysis (Neo4j, NetworkX)
- Graph centrality (PageRank, Betweenness) for role detection
- GNNs (PyTorch Geometric, DGL) for learning from account-transaction networks
- Graph features (degree, centrality) as regular model features

### 5. AutoML Platforms
- **AutoGluon, H2O AutoML, Auto-sklearn, TPOT**
- Amazon Fraud Detector (paid AWS service)
- PyCaret — turnkey library
- Auto-sklearn uses meta-learning for warm-start

### 6. Feature-Rich Pipelines
- scikit-learn Pipeline + imblearn sampler + XGBoost
- CatBoost auto-handles categoricals
- Deep learning: Autoencoders, DeepSVDD

### 7. Ensembles & Stacking
- VotingClassifier, StackingClassifier
- Blend XGB + LightGBM + IsolationForest scores
- mlxtend for easy stacking

### 8. Built-in Imbalance Handling
- CatBoost: scale_pos_weight
- LightGBM: is_unbalance parameter
- XGBoost: gpu_hist mode for GPU acceleration

### 9. Evaluation Tools
- StratifiedKFold, StratifiedShuffleSplit, GroupKFold
- F1, ROC-AUC, PR-AUC metrics
- imblearn pipelines auto-resample inside CV folds

---

## Key Takeaway
Tree-based ensembles + imbalanced-learn resamplers are the core.  
Anomaly detectors provide complementary signals (but weak standalone for this dataset).  
Graph methods useful IF transaction-level data is available.  
AutoML can quickly explore a wide range of approaches.
