"""
Phase 6: Model Improvement & Problem Statement Feature Investigation
=====================================================================
1. Deep analysis of the 18 bank-recommended features
2. Test if adding them improves the model
3. Extended Optuna tuning (500 trials)
4. No-SMOTE variants
5. Soft voting ensemble
6. Combined feature sets

Generates graphs 44-49 in files/graphs/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, classification_report, confusion_matrix,
    average_precision_score, mutual_info_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from scipy import stats
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.style.use('dark_background')
C = {
    'primary': '#ff006e', 'secondary': '#00f5d4', 'accent': '#8338ec',
    'warn': '#ffbe0b', 'blue': '#3a86ff', 'bg': '#0a0a23',
    'card': '#161b22', 'text': '#e6edf3', 'grid': '#21262d',
    'green': '#2ed573', 'orange': '#ff7f50'
}

GRAPH_DIR = 'files/graphs'
os.makedirs(GRAPH_DIR, exist_ok=True)

RANDOM_STATE = 42
LEAKAGE = ['F3912', 'F2230', 'Unnamed: 0']

# ======================================================================
# Problem Statement Features
# ======================================================================
PS_FEATURES = ['F115', 'F321', 'F527', 'F531', 'F670', 'F1692',
               'F2082', 'F2122', 'F2582', 'F2678', 'F2737',
               'F2956', 'F3043', 'F3836', 'F3887', 'F3889', 'F3891', 'F3894']

# Our discovered features (Phase 5 optimal)
OUR_RAW = ['F3898', 'F1819', 'F3799', 'F1165', 'F1813', 'F3806', 'F162', 'F3800']
STABLE_8 = ['F3811', 'F3806', 'F3799', 'F3805', 'F3813', 'F3801', 'F3898', 'F3807']


def clean_col(series):
    """Clean bracket-wrapped values."""
    if series.dtype == object:
        series = series.astype(str).str.strip('[]').str.strip()
    return pd.to_numeric(series, errors='coerce').fillna(0)


def engineer_features(df, y=None):
    """Create all engineered features."""
    features = pd.DataFrame(index=df.index)
    
    # Raw features
    for f in OUR_RAW:
        features[f] = clean_col(df[f]).values
    
    # Engineered
    f162 = clean_col(df['F162']).values
    f3898 = clean_col(df['F3898']).values
    features['F162_div_F3898'] = np.where(f3898 != 0, f162 / (f3898 + 1e-10), 0)
    
    stable_clean = pd.DataFrame({c: clean_col(df[c]) for c in STABLE_8}, index=df.index)
    features['max_value_top8'] = stable_clean.max(axis=1).values
    
    f3805 = clean_col(df['F3805']).values
    features['F3898_div_F3805'] = np.where(f3805 != 0, f3898 / (f3805 + 1e-10), 0)
    
    f0_f499 = [f'F{i}' for i in range(500) if f'F{i}' in df.columns]
    features['missing_count_F0_F500'] = df[f0_f499].isnull().sum(axis=1).values
    
    f3811 = clean_col(df['F3811']).values
    features['F3811_div_F3805'] = np.where(f3805 != 0, f3811 / (f3805 + 1e-10), 0)
    features['F3898_div_F3811'] = np.where(f3811 != 0, f3898 / (f3811 + 1e-10), 0)
    
    # Clip ratios
    for col in ['F162_div_F3898', 'F3898_div_F3805', 'F3811_div_F3805', 'F3898_div_F3811']:
        features[col] = np.clip(features[col], -1e6, 1e6)
    
    return features


def cv_evaluate(X, y, model_fn, n_splits=5, smote_ratio=None):
    """Cross-validate and return metrics."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    f1s, precs, recs, aucs = [], [], [], []
    y_prob_all = np.zeros(len(y))
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        if smote_ratio:
            try:
                sm = SMOTE(sampling_strategy=smote_ratio, k_neighbors=3, random_state=RANDOM_STATE)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except:
                pass
        
        clf = model_fn()
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        y_prob_all[val_idx] = probs
        
        f1s.append(f1_score(y_val, preds, zero_division=0))
        precs.append(precision_score(y_val, preds, zero_division=0))
        recs.append(recall_score(y_val, preds, zero_division=0))
        aucs.append(roc_auc_score(y_val, probs))
    
    return {
        'f1': np.mean(f1s), 'f1_std': np.std(f1s),
        'precision': np.mean(precs), 'recall': np.mean(recs),
        'auc': np.mean(aucs), 'y_prob': y_prob_all
    }


# ======================================================================
# MAIN
# ======================================================================
def main():
    start = time.time()
    print("=" * 70)
    print("PHASE 6: MODEL IMPROVEMENT & FEATURE INVESTIGATION")
    print("=" * 70)
    
    # Load data
    print("\n[1/8] Loading data...")
    df = pd.read_csv('DataSet.csv', low_memory=False)
    y = df['F3924'].values.astype(int)
    pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"  {len(df)} accounts, {y.sum()} suspicious")
    
    # ======================================================================
    # PART 1: Deep Investigation of 18 Problem Statement Features
    # ======================================================================
    print("\n" + "=" * 70)
    print("[2/8] INVESTIGATING 18 PROBLEM STATEMENT FEATURES")
    print("=" * 70)
    
    ps_stats = []
    for feat in PS_FEATURES:
        col = df[feat]
        clean = clean_col(col)
        
        susp_vals = clean[y == 1]
        legit_vals = clean[y == 0]
        
        # KS test
        if susp_vals.std() > 0 or legit_vals.std() > 0:
            ks_stat, ks_p = stats.ks_2samp(susp_vals, legit_vals)
        else:
            ks_stat, ks_p = 0, 1
        
        # Correlation with target
        try:
            corr = np.corrcoef(clean.values, y)[0, 1]
        except:
            corr = 0
        
        # Missing rate
        missing_total = col.isnull().sum() / len(col) * 100
        missing_susp = col[y == 1].isnull().sum() / y.sum() * 100
        missing_legit = col[y == 0].isnull().sum() / (y == 0).sum() * 100
        
        ps_stats.append({
            'feature': feat,
            'dtype': str(col.dtype),
            'n_unique': col.nunique(),
            'missing_pct': missing_total,
            'missing_susp': missing_susp,
            'missing_legit': missing_legit,
            'susp_mean': susp_vals.mean(),
            'legit_mean': legit_vals.mean(),
            'susp_median': susp_vals.median(),
            'legit_median': legit_vals.median(),
            'ks_stat': ks_stat,
            'ks_pvalue': ks_p,
            'corr_target': corr,
            'mean_ratio': susp_vals.mean() / (legit_vals.mean() + 1e-10),
        })
    
    ps_df = pd.DataFrame(ps_stats)
    
    # Print detailed analysis
    print("\n  === Problem Statement Feature Analysis ===")
    print(f"  {'Feature':<8} {'Type':<8} {'Unique':>7} {'Miss%':>6} {'KS':>6} {'Corr':>7} {'Susp/Legit':>12} {'Verdict'}")
    print("  " + "-" * 80)
    
    for _, row in ps_df.iterrows():
        verdict = ""
        if row['ks_pvalue'] < 0.05:
            verdict = "SIGNIFICANT"
        elif row['missing_pct'] > 90:
            verdict = "MOSTLY NULL"
        else:
            verdict = "weak"
        
        if row['feature'] in LEAKAGE:
            verdict = "LEAKAGE!"
        
        print(f"  {row['feature']:<8} {row['dtype']:<8} {row['n_unique']:>7} "
              f"{row['missing_pct']:>5.1f}% {row['ks_stat']:>6.3f} {row['corr_target']:>7.4f} "
              f"{row['mean_ratio']:>12.3f} {verdict}")
    
    # ======================================================================
    # PART 2: Hypothesize what these features mean in banking context
    # ======================================================================
    print("\n\n  === Banking Feature Interpretation ===")
    interpretations = {
        'F115': 'Account holder type / Customer segment code',
        'F321': 'Transaction channel indicator (ATM/Branch/Online)',
        'F527': 'Average monthly transaction count (last 6 months)',
        'F531': 'Average monthly transaction amount (last 6 months)',
        'F670': 'Number of distinct payees / counterparties',
        'F1692': 'High-value transaction count (above threshold)',
        'F2082': 'Incoming funds ratio / credit-to-debit ratio',
        'F2122': 'Cross-border or inter-bank transfer count',
        'F2582': 'Account balance volatility / standard deviation',
        'F2678': 'Dormancy indicator / days since last transaction',
        'F2737': 'Number of linked accounts or beneficiaries',
        'F2956': 'Risk score from existing rule engine (correlated pair 1)',
        'F3043': 'Risk score from existing rule engine (correlated pair 2)',
        'F3836': 'Account type code (savings/current/other)',
        'F3887': 'Account age in DAYS (confirmed: 0.9999 corr with computed age)',
        'F3889': 'Customer onboarding channel / KYC source',
        'F3891': 'Branch/region code of account opening',
        'F3894': 'Customer age in YEARS (range: -2 to 94, median 34)',
    }
    
    for feat, interp in interpretations.items():
        ks = ps_df[ps_df['feature'] == feat]['ks_stat'].values[0]
        print(f"  {feat}: {interp} (KS={ks:.3f})")
    
    # ======================================================================
    # GRAPH 44: Problem Statement Features Analysis
    # ======================================================================
    print("\n  Generating Graph 44...")
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle('Problem Statement Features: Deep Analysis', fontsize=20, fontweight='bold', color='white')
    
    # Panel 1: KS statistics comparison
    ax = axes[0, 0]
    ps_sorted = ps_df.sort_values('ks_stat', ascending=True)
    colors_ks = [C['primary'] if p < 0.05 else C['blue'] for p in ps_sorted['ks_pvalue']]
    ax.barh(range(len(ps_sorted)), ps_sorted['ks_stat'].values, color=colors_ks, alpha=0.85)
    ax.set_yticks(range(len(ps_sorted)))
    ax.set_yticklabels(ps_sorted['feature'].values, fontsize=10)
    ax.set_xlabel('KS Statistic', fontsize=12)
    ax.set_title('Discriminative Power (KS Test)', fontsize=14, color='white')
    ax.axvline(x=0.1, color=C['warn'], linestyle='--', alpha=0.5, label='Weak threshold')
    ax.legend(fontsize=9)
    ax.set_facecolor(C['bg'])
    
    # Panel 2: Correlation with target
    ax = axes[0, 1]
    ps_corr_sorted = ps_df.sort_values('corr_target', key=abs, ascending=True)
    colors_corr = [C['secondary'] if c > 0 else C['primary'] for c in ps_corr_sorted['corr_target']]
    ax.barh(range(len(ps_corr_sorted)), ps_corr_sorted['corr_target'].values, color=colors_corr, alpha=0.85)
    ax.set_yticks(range(len(ps_corr_sorted)))
    ax.set_yticklabels(ps_corr_sorted['feature'].values, fontsize=10)
    ax.set_xlabel('Correlation with Target (F3924)', fontsize=12)
    ax.set_title('Correlation Strength', fontsize=14, color='white')
    ax.axvline(x=0, color='white', linewidth=0.5, alpha=0.5)
    ax.set_facecolor(C['bg'])
    
    # Panel 3: Missing data pattern
    ax = axes[1, 0]
    x_pos = np.arange(len(PS_FEATURES))
    width = 0.35
    ax.bar(x_pos - width/2, ps_df['missing_susp'].values, width, color=C['primary'], alpha=0.85, label='Suspicious')
    ax.bar(x_pos + width/2, ps_df['missing_legit'].values, width, color=C['secondary'], alpha=0.85, label='Legitimate')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(PS_FEATURES, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Missing Rate (%)', fontsize=12)
    ax.set_title('Missing Values: Suspicious vs Legitimate', fontsize=14, color='white')
    ax.legend(fontsize=10)
    ax.set_facecolor(C['bg'])
    
    # Panel 4: Mean ratio (suspicious / legitimate)
    ax = axes[1, 1]
    ratios = ps_df['mean_ratio'].clip(-5, 5)  # Clip for display
    colors_ratio = [C['primary'] if r < 0.8 else (C['secondary'] if r > 1.2 else C['blue']) for r in ratios]
    ax.barh(range(len(ps_df)), ratios.values, color=colors_ratio, alpha=0.85)
    ax.set_yticks(range(len(ps_df)))
    ax.set_yticklabels(ps_df['feature'].values, fontsize=10)
    ax.set_xlabel('Mean Ratio (Suspicious / Legitimate)', fontsize=12)
    ax.set_title('Feature Value Ratio by Class', fontsize=14, color='white')
    ax.axvline(x=1, color=C['warn'], linestyle='--', linewidth=2, label='Equal')
    ax.legend(fontsize=10)
    ax.set_facecolor(C['bg'])
    
    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/44_ps_features_analysis.png', dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print("  [OK] 44_ps_features_analysis.png")
    
    # ======================================================================
    # PART 3: Test models with different feature sets
    # ======================================================================
    print("\n" + "=" * 70)
    print("[3/8] COMPARING FEATURE SETS")
    print("=" * 70)
    
    def make_xgb():
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=pos_weight, verbosity=0, random_state=RANDOM_STATE, n_jobs=-1
        )
    
    # Prepare different feature sets
    feature_sets = {}
    
    # Set 1: Our Phase 5 optimal (14 features)
    X_ours = engineer_features(df, y).values.astype(np.float64)
    feature_sets['Our 14 (Phase 5)'] = X_ours
    
    # Set 2: Problem Statement 18 features only
    ps_clean = pd.DataFrame({f: clean_col(df[f]) for f in PS_FEATURES})
    X_ps = ps_clean.values.astype(np.float64)
    feature_sets['PS 18 Only'] = X_ps
    
    # Set 3: Our 14 + PS 18 (merged, no duplicates)
    combined_feats = list(engineer_features(df, y).columns) + [f for f in PS_FEATURES if f not in OUR_RAW]
    X_combined = pd.concat([engineer_features(df, y), ps_clean[[f for f in PS_FEATURES if f not in OUR_RAW]]], axis=1)
    X_combined = X_combined.values.astype(np.float64)
    feature_sets['Our 14 + PS 18'] = X_combined
    
    # Set 4: Our 14 + ONLY significant PS features (KS p < 0.05)
    sig_ps = ps_df[ps_df['ks_pvalue'] < 0.05]['feature'].tolist()
    sig_ps_clean = pd.DataFrame({f: clean_col(df[f]) for f in sig_ps if f not in OUR_RAW})
    if len(sig_ps_clean.columns) > 0:
        X_ours_sigps = pd.concat([engineer_features(df, y), sig_ps_clean], axis=1).values.astype(np.float64)
    else:
        X_ours_sigps = X_ours
    feature_sets[f'Our 14 + Sig PS ({len(sig_ps)})'] = X_ours_sigps
    
    # Set 5: Top 30 features by MI (broad feature scan)
    print("  Computing MI for all numeric features...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['F3924'] + LEAKAGE]
    X_all_numeric = df[numeric_cols].fillna(0).values.astype(np.float64)
    mi_scores = mutual_info_classif(X_all_numeric, y, random_state=RANDOM_STATE, n_neighbors=5)
    mi_ranking = sorted(zip(numeric_cols, mi_scores), key=lambda x: -x[1])
    top30_mi_feats = [f for f, _ in mi_ranking[:30]]
    X_top30_mi = df[top30_mi_feats].fillna(0).values.astype(np.float64)
    feature_sets['Top 30 MI'] = X_top30_mi
    
    # Set 6: Our 14 + PS 18 + engineered from PS features
    ps_eng = pd.DataFrame(index=df.index)
    ps_eng['F527_x_F531'] = clean_col(df['F527']).values * clean_col(df['F531']).values  # txn_count x txn_amount
    f3887 = clean_col(df['F3887']).values
    f3894 = clean_col(df['F3894']).values
    ps_eng['age_acct_ratio'] = np.where(f3894 > 0, f3887 / (f3894 * 365 + 1e-10), 0)
    ps_eng['F2956_minus_F3043'] = clean_col(df['F2956']).values - clean_col(df['F3043']).values
    ps_eng['F670_per_age'] = np.where(f3887 > 0, clean_col(df['F670']).values / (f3887 + 1e-10), 0)
    X_kitchen_sink = pd.concat([
        engineer_features(df, y),
        ps_clean[[f for f in PS_FEATURES if f not in OUR_RAW]],
        ps_eng
    ], axis=1).values.astype(np.float64)
    feature_sets['Kitchen Sink (all)'] = X_kitchen_sink
    
    print(f"\n  Feature sets to test:")
    for name, X in feature_sets.items():
        print(f"    {name}: {X.shape[1]} features")
    
    # Evaluate all
    print("\n  Evaluating feature sets (5-fold CV)...")
    results = {}
    for name, X in feature_sets.items():
        print(f"    Testing: {name}...", end=' ')
        r = cv_evaluate(X, y, make_xgb)
        results[name] = r
        print(f"F1={r['f1']:.4f} (±{r['f1_std']:.3f}), AUC={r['auc']:.4f}")
    
    # ======================================================================
    # GRAPH 45: Feature Set Comparison
    # ======================================================================
    print("\n  Generating Graph 45...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Feature Set Comparison', fontsize=18, fontweight='bold', color='white')
    
    names = list(results.keys())
    f1s = [results[n]['f1'] for n in names]
    aucs = [results[n]['auc'] for n in names]
    f1_stds = [results[n]['f1_std'] for n in names]
    
    best_idx = np.argmax(f1s)
    colors_bar = [C['primary'] if i == best_idx else C['blue'] for i in range(len(names))]
    
    ax1.barh(range(len(names)), f1s, xerr=f1_stds, color=colors_bar, alpha=0.85, capsize=5)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=11)
    ax1.set_xlabel('F1 Score', fontsize=13)
    ax1.set_title(f'F1 Comparison (Best: {names[best_idx]})', fontsize=14, color=C['warn'])
    for i, (f1, std) in enumerate(zip(f1s, f1_stds)):
        ax1.text(f1 + std + 0.005, i, f'{f1:.4f}', va='center', fontsize=10, color='white')
    ax1.set_facecolor(C['bg'])
    
    ax2.barh(range(len(names)), aucs, color=[C['accent']] * len(names), alpha=0.85)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=11)
    ax2.set_xlabel('ROC-AUC', fontsize=13)
    ax2.set_title('AUC Comparison', fontsize=14, color='white')
    for i, auc in enumerate(aucs):
        ax2.text(auc + 0.002, i, f'{auc:.4f}', va='center', fontsize=10, color='white')
    ax2.set_facecolor(C['bg'])
    
    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/45_feature_set_comparison.png', dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print("  [OK] 45_feature_set_comparison.png")
    
    # ======================================================================
    # PART 4: Find the BEST feature set, then optimize model
    # ======================================================================
    best_set_name = names[best_idx]
    X_best = feature_sets[best_set_name]
    print(f"\n  BEST FEATURE SET: {best_set_name} (F1={f1s[best_idx]:.4f})")
    
    # ======================================================================
    # PART 5: Extended Optuna Tuning (500 trials)
    # ======================================================================
    print("\n" + "=" * 70)
    print("[4/8] EXTENDED OPTUNA TUNING (500 trials)")
    print("=" * 70)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 100, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 100, log=True),
            'scale_pos_weight': pos_weight,
            'verbosity': 0,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
        }
        
        use_smote = trial.suggest_categorical('use_smote', [True, False])
        smote_ratio = trial.suggest_float('smote_ratio', 0.1, 0.5) if use_smote else None
        
        f1_scores = []
        for train_idx, val_idx in cv.split(X_best, y):
            X_tr, X_val = X_best[train_idx], X_best[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            if use_smote and smote_ratio:
                try:
                    sm = SMOTE(sampling_strategy=smote_ratio, k_neighbors=3, random_state=RANDOM_STATE)
                    X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                except:
                    pass
            
            clf = XGBClassifier(**params)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            f1_scores.append(f1_score(y_val, preds, zero_division=0))
        
        return np.mean(f1_scores)
    
    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=500, timeout=600)  # 10 min max
    
    print(f"  Best F1: {study.best_value:.4f}")
    best_params = study.best_params
    use_smote_best = best_params.pop('use_smote')
    smote_ratio_best = best_params.pop('smote_ratio', None) if use_smote_best else None
    best_params['scale_pos_weight'] = pos_weight
    best_params['verbosity'] = 0
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = -1
    
    print(f"  Use SMOTE: {use_smote_best}" + (f" (ratio={smote_ratio_best:.3f})" if smote_ratio_best else ""))
    print(f"  Params: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()}, indent=4)}")
    
    # ======================================================================
    # PART 6: Model Variants Comparison  
    # ======================================================================
    print("\n" + "=" * 70)
    print("[5/8] MODEL VARIANTS COMPARISON")
    print("=" * 70)
    
    model_variants = {}
    
    # Variant 1: Optuna-tuned XGBoost
    def make_optuna_xgb():
        return XGBClassifier(**best_params)
    r = cv_evaluate(X_best, y, make_optuna_xgb, smote_ratio=smote_ratio_best if use_smote_best else None)
    model_variants['XGBoost (Optuna-500)'] = r
    print(f"  XGBoost (Optuna-500):    F1={r['f1']:.4f}, AUC={r['auc']:.4f}")
    
    # Variant 2: XGBoost no SMOTE
    r = cv_evaluate(X_best, y, make_optuna_xgb)
    model_variants['XGBoost (No SMOTE)'] = r
    print(f"  XGBoost (No SMOTE):      F1={r['f1']:.4f}, AUC={r['auc']:.4f}")
    
    # Variant 3: XGBoost + SMOTE(0.3)
    r = cv_evaluate(X_best, y, make_optuna_xgb, smote_ratio=0.3)
    model_variants['XGBoost (SMOTE 0.3)'] = r
    print(f"  XGBoost (SMOTE 0.3):     F1={r['f1']:.4f}, AUC={r['auc']:.4f}")
    
    # Variant 4: LightGBM
    def make_lgbm():
        return LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=pos_weight, verbosity=-1, random_state=RANDOM_STATE, n_jobs=-1
        )
    r = cv_evaluate(X_best, y, make_lgbm)
    model_variants['LightGBM'] = r
    print(f"  LightGBM:                F1={r['f1']:.4f}, AUC={r['auc']:.4f}")
    
    # Variant 5: Random Forest  
    def make_rf():
        return RandomForestClassifier(
            n_estimators=500, max_depth=10, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1
        )
    r = cv_evaluate(X_best, y, make_rf)
    model_variants['Random Forest'] = r
    print(f"  Random Forest:           F1={r['f1']:.4f}, AUC={r['auc']:.4f}")
    
    # Variant 6: Soft Voting (XGB + LGBM + RF)
    def make_voting():
        return VotingClassifier(estimators=[
            ('xgb', XGBClassifier(**best_params)),
            ('lgbm', LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                     scale_pos_weight=pos_weight, verbosity=-1, random_state=RANDOM_STATE, n_jobs=-1)),
            ('rf', RandomForestClassifier(n_estimators=500, max_depth=10, class_weight='balanced',
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ], voting='soft')
    r = cv_evaluate(X_best, y, make_voting)
    model_variants['Soft Voting (3)'] = r
    print(f"  Soft Voting (3):         F1={r['f1']:.4f}, AUC={r['auc']:.4f}")
    
    # Variant 7: GBM (sklearn)
    def make_gbm():
        return GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=RANDOM_STATE
        )
    r = cv_evaluate(X_best, y, make_gbm)
    model_variants['GBM (sklearn)'] = r
    print(f"  GBM (sklearn):           F1={r['f1']:.4f}, AUC={r['auc']:.4f}")
    
    # ======================================================================
    # GRAPH 46: Model Variants Comparison
    # ======================================================================
    print("\n  Generating Graph 46...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Model Variants: Final Comparison', fontsize=18, fontweight='bold', color='white')
    
    vnames = list(model_variants.keys())
    vf1s = [model_variants[n]['f1'] for n in vnames]
    vaucs = [model_variants[n]['auc'] for n in vnames]
    vstds = [model_variants[n]['f1_std'] for n in vnames]
    
    best_v = np.argmax(vf1s)
    vcolors = [C['primary'] if i == best_v else C['blue'] for i in range(len(vnames))]
    
    sorted_idx = np.argsort(vf1s)
    ax1.barh(range(len(vnames)), [vf1s[i] for i in sorted_idx],
             xerr=[vstds[i] for i in sorted_idx],
             color=[vcolors[i] for i in sorted_idx], alpha=0.85, capsize=5)
    ax1.set_yticks(range(len(vnames)))
    ax1.set_yticklabels([vnames[i] for i in sorted_idx], fontsize=11)
    ax1.set_xlabel('F1 Score', fontsize=13)
    ax1.set_title(f'Best: {vnames[best_v]} (F1={vf1s[best_v]:.4f})', fontsize=13, color=C['warn'])
    for i, idx in enumerate(sorted_idx):
        ax1.text(vf1s[idx] + vstds[idx] + 0.005, i, f'{vf1s[idx]:.4f}', va='center', fontsize=10, color='white')
    ax1.set_facecolor(C['bg'])
    
    sorted_idx_auc = np.argsort(vaucs)
    ax2.barh(range(len(vnames)), [vaucs[i] for i in sorted_idx_auc],
             color=[C['accent']] * len(vnames), alpha=0.85)
    ax2.set_yticks(range(len(vnames)))
    ax2.set_yticklabels([vnames[i] for i in sorted_idx_auc], fontsize=11)
    ax2.set_xlabel('ROC-AUC', fontsize=13)
    ax2.set_title('AUC Comparison', fontsize=14, color='white')
    for i, idx in enumerate(sorted_idx_auc):
        ax2.text(vaucs[idx] + 0.002, i, f'{vaucs[idx]:.4f}', va='center', fontsize=10, color='white')
    ax2.set_facecolor(C['bg'])
    
    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/46_model_variants.png', dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print("  [OK] 46_model_variants.png")
    
    # ======================================================================
    # PART 7: Find optimal threshold for best model
    # ======================================================================
    print("\n" + "=" * 70)
    print("[6/8] OPTIMAL THRESHOLD FOR BEST MODEL")
    print("=" * 70)
    
    best_model_name = vnames[best_v]
    y_prob_best = model_variants[best_model_name]['y_prob']
    
    threshs = np.arange(0.01, 0.99, 0.005)
    f1_at_t = []
    for t in threshs:
        preds_t = (y_prob_best >= t).astype(int)
        f1_at_t.append(f1_score(y, preds_t, zero_division=0))
    
    opt_t = threshs[np.argmax(f1_at_t)]
    opt_f1 = max(f1_at_t)
    
    # Also find cost-optimal
    costs = []
    for t in threshs:
        preds_t = (y_prob_best >= t).astype(int)
        fp = ((preds_t == 1) & (y == 0)).sum()
        fn = ((preds_t == 0) & (y == 1)).sum()
        costs.append(fp * 100 + fn * 10000)
    cost_opt_t = threshs[np.argmin(costs)]
    
    print(f"  Best model: {best_model_name}")
    print(f"  F1-optimal threshold: {opt_t:.3f} (F1={opt_f1:.4f})")
    print(f"  Cost-optimal threshold: {cost_opt_t:.3f}")
    
    # Performance at optimal threshold
    for t_name, t_val in [('Default (0.5)', 0.5), (f'F1-opt ({opt_t:.3f})', opt_t), (f'Cost-opt ({cost_opt_t:.3f})', cost_opt_t)]:
        preds_t = (y_prob_best >= t_val).astype(int)
        tp = ((preds_t == 1) & (y == 1)).sum()
        fp = ((preds_t == 1) & (y == 0)).sum()
        fn = ((preds_t == 0) & (y == 1)).sum()
        f1_t = f1_score(y, preds_t, zero_division=0)
        p_t = precision_score(y, preds_t, zero_division=0)
        r_t = recall_score(y, preds_t, zero_division=0)
        cost_t = fp * 100 + fn * 10000
        print(f"    {t_name:<25} F1={f1_t:.4f} P={p_t:.4f} R={r_t:.4f} TP={tp} FP={fp} FN={fn} Cost=${cost_t:,}")
    
    # ======================================================================
    # GRAPH 47: PS Features vs Our Features - Head-to-Head
    # ======================================================================
    print("\n  Generating Graph 47...")
    
    # Compute MI for PS features vs our features
    X_ps_mi = ps_clean.values.astype(np.float64)
    mi_ps = mutual_info_classif(X_ps_mi, y, random_state=RANDOM_STATE, n_neighbors=5)
    
    X_ours_raw_only = pd.DataFrame({f: clean_col(df[f]) for f in OUR_RAW}).values.astype(np.float64)
    mi_ours = mutual_info_classif(X_ours_raw_only, y, random_state=RANDOM_STATE, n_neighbors=5)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Problem Statement Features vs Our Discovered Features', fontsize=18, fontweight='bold', color='white')
    
    # MI comparison
    ax = axes[0]
    mi_ps_sorted = sorted(zip(PS_FEATURES, mi_ps), key=lambda x: -x[1])
    mi_ours_sorted = sorted(zip(OUR_RAW, mi_ours), key=lambda x: -x[1])
    
    all_mi = mi_ps_sorted + mi_ours_sorted
    all_mi_sorted = sorted(all_mi, key=lambda x: -x[1])
    names_mi = [x[0] for x in all_mi_sorted]
    vals_mi = [x[1] for x in all_mi_sorted]
    colors_mi = [C['primary'] if n in PS_FEATURES else C['secondary'] for n in names_mi]
    
    ax.barh(range(len(names_mi)), vals_mi, color=colors_mi, alpha=0.85)
    ax.set_yticks(range(len(names_mi)))
    ax.set_yticklabels(names_mi, fontsize=9)
    ax.set_xlabel('Mutual Information', fontsize=12)
    ax.set_title('MI with Target (Red=PS, Green=Ours)', fontsize=13, color='white')
    ax.set_facecolor(C['bg'])
    
    # KS comparison
    ax = axes[1]
    ks_ours = []
    for f in OUR_RAW:
        sv = clean_col(df[f])[y == 1]
        lv = clean_col(df[f])[y == 0]
        ks, _ = stats.ks_2samp(sv, lv)
        ks_ours.append((f, ks))
    
    all_ks = [(f, row['ks_stat']) for _, row in ps_df.iterrows()] + ks_ours
    all_ks_sorted = sorted(all_ks, key=lambda x: -x[1])
    names_ks = [x[0] for x in all_ks_sorted]
    vals_ks = [x[1] for x in all_ks_sorted]
    colors_ks2 = [C['primary'] if n in PS_FEATURES else C['secondary'] for n in names_ks]
    
    ax.barh(range(len(names_ks)), vals_ks, color=colors_ks2, alpha=0.85)
    ax.set_yticks(range(len(names_ks)))
    ax.set_yticklabels(names_ks, fontsize=9)
    ax.set_xlabel('KS Statistic', fontsize=12)
    ax.set_title('Discriminative Power (Red=PS, Green=Ours)', fontsize=13, color='white')
    ax.set_facecolor(C['bg'])
    
    # Feature set F1 comparison (summary)
    ax = axes[2]
    compare_names = ['PS 18 Only', 'Our 14 (Phase 5)', 'Our 14 + PS 18', f'Our 14 + Sig PS ({len(sig_ps)})', 'Kitchen Sink (all)']
    compare_f1 = [results[n]['f1'] for n in compare_names if n in results]
    compare_names = [n for n in compare_names if n in results]
    
    sorted_c = np.argsort(compare_f1)
    ax.barh(range(len(compare_names)), [compare_f1[i] for i in sorted_c],
            color=[C['warn']] * len(compare_names), alpha=0.85)
    ax.set_yticks(range(len(compare_names)))
    ax.set_yticklabels([compare_names[i] for i in sorted_c], fontsize=11)
    ax.set_xlabel('F1 Score (5-Fold CV)', fontsize=12)
    ax.set_title('Feature Set Model Performance', fontsize=13, color='white')
    for i, idx in enumerate(sorted_c):
        ax.text(compare_f1[idx] + 0.005, i, f'{compare_f1[idx]:.4f}', va='center', fontsize=11, color='white')
    ax.set_facecolor(C['bg'])
    
    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/47_ps_vs_ours.png', dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print("  [OK] 47_ps_vs_ours.png")
    
    # ======================================================================
    # PART 8: Individual PS feature importance when added to our model
    # ======================================================================
    print("\n" + "=" * 70)
    print("[7/8] INDIVIDUAL PS FEATURE CONTRIBUTION")
    print("=" * 70)
    
    base_f1 = results['Our 14 (Phase 5)']['f1']
    print(f"  Baseline (Our 14): F1={base_f1:.4f}")
    print(f"  Adding each PS feature one at a time...")
    
    ps_contributions = []
    for feat in PS_FEATURES:
        if feat in OUR_RAW:
            continue
        
        X_plus = np.column_stack([X_ours, clean_col(df[feat]).values.astype(np.float64)])
        r = cv_evaluate(X_plus, y, make_xgb)
        delta = r['f1'] - base_f1
        ps_contributions.append({
            'feature': feat,
            'f1_with': r['f1'],
            'f1_delta': delta,
            'auc_with': r['auc'],
        })
        marker = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
        print(f"    + {feat}: F1={r['f1']:.4f} ({marker}{abs(delta):.4f})")
    
    contrib_df = pd.DataFrame(ps_contributions).sort_values('f1_delta', ascending=False)
    
    # ======================================================================
    # GRAPH 48: PS Feature Individual Contributions
    # ======================================================================
    print("\n  Generating Graph 48...")
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle('Adding Each PS Feature to Our Model (One at a Time)', fontsize=18, fontweight='bold', color='white')
    
    contrib_sorted = contrib_df.sort_values('f1_delta', ascending=True)
    colors_contrib = [C['green'] if d > 0.001 else (C['primary'] if d < -0.001 else C['blue']) 
                      for d in contrib_sorted['f1_delta']]
    
    ax.barh(range(len(contrib_sorted)), contrib_sorted['f1_delta'].values, color=colors_contrib, alpha=0.85)
    ax.set_yticks(range(len(contrib_sorted)))
    ax.set_yticklabels(contrib_sorted['feature'].values, fontsize=11)
    ax.set_xlabel('F1 Change (vs Our 14 baseline)', fontsize=13)
    ax.axvline(x=0, color='white', linewidth=1, alpha=0.5)
    ax.set_title(f'Baseline F1={base_f1:.4f} | Green=Helps, Red=Hurts, Blue=Neutral', fontsize=13, color='white')
    
    for i, (_, row) in enumerate(contrib_sorted.iterrows()):
        delta = row['f1_delta']
        ax.text(delta + 0.002 * np.sign(delta), i, f'{delta:+.4f}', va='center', fontsize=9, color='white')
    
    ax.set_facecolor(C['bg'])
    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/48_ps_feature_contributions.png', dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print("  [OK] 48_ps_feature_contributions.png")
    
    # ======================================================================
    # PART 9: Build the ABSOLUTE BEST model
    # ======================================================================
    print("\n" + "=" * 70)
    print("[8/8] BUILDING THE BEST POSSIBLE MODEL")
    print("=" * 70)
    
    # Identify PS features that help
    helpful_ps = contrib_df[contrib_df['f1_delta'] > 0.001]['feature'].tolist()
    print(f"  Helpful PS features: {helpful_ps}")
    
    if helpful_ps:
        X_optimal = np.column_stack([X_ours] + [clean_col(df[f]).values.astype(np.float64) for f in helpful_ps])
        opt_feat_names = list(engineer_features(df, y).columns) + helpful_ps
    else:
        X_optimal = X_ours
        opt_feat_names = list(engineer_features(df, y).columns)
    
    print(f"  Optimal feature count: {X_optimal.shape[1]}")
    
    # Final Optuna on optimal features
    print("  Running final Optuna tuning on optimal features (300 trials)...")
    
    def objective_final(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 100, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 100, log=True),
            'scale_pos_weight': pos_weight,
            'verbosity': 0,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
        }
        
        use_smote = trial.suggest_categorical('use_smote', [True, False])
        sm_ratio = trial.suggest_float('smote_ratio', 0.1, 0.5) if use_smote else None
        
        f1_scores = []
        for train_idx, val_idx in cv.split(X_optimal, y):
            X_tr, X_val = X_optimal[train_idx], X_optimal[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            if use_smote and sm_ratio:
                try:
                    sm = SMOTE(sampling_strategy=sm_ratio, k_neighbors=3, random_state=RANDOM_STATE)
                    X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                except:
                    pass
            
            clf = XGBClassifier(**params)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            f1_scores.append(f1_score(y_val, preds, zero_division=0))
        
        return np.mean(f1_scores)
    
    study_final = optuna.create_study(direction='maximize',
                                       sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + 1))
    study_final.optimize(objective_final, n_trials=300, timeout=600)
    
    final_params = study_final.best_params
    final_use_smote = final_params.pop('use_smote')
    final_smote_ratio = final_params.pop('smote_ratio', None) if final_use_smote else None
    final_params['scale_pos_weight'] = pos_weight
    final_params['verbosity'] = 0
    final_params['random_state'] = RANDOM_STATE
    final_params['n_jobs'] = -1
    
    print(f"  Final best F1: {study_final.best_value:.4f}")
    print(f"  SMOTE: {final_use_smote}" + (f" (ratio={final_smote_ratio:.3f})" if final_smote_ratio else ""))
    
    # Final CV evaluation
    def make_final():
        return XGBClassifier(**final_params)
    
    r_final = cv_evaluate(X_optimal, y, make_final, smote_ratio=final_smote_ratio if final_use_smote else None)
    print(f"\n  FINAL MODEL PERFORMANCE:")
    print(f"    F1:        {r_final['f1']:.4f} (±{r_final['f1_std']:.4f})")
    print(f"    Precision: {r_final['precision']:.4f}")
    print(f"    Recall:    {r_final['recall']:.4f}")
    print(f"    AUC:       {r_final['auc']:.4f}")
    
    # Optimal threshold
    y_prob_final = r_final['y_prob']
    f1_at_t_final = []
    for t in threshs:
        preds_t = (y_prob_final >= t).astype(int)
        f1_at_t_final.append(f1_score(y, preds_t, zero_division=0))
    
    final_opt_t = threshs[np.argmax(f1_at_t_final)]
    final_opt_f1 = max(f1_at_t_final)
    print(f"    Optimal threshold: {final_opt_t:.3f} (F1={final_opt_f1:.4f})")
    
    # ======================================================================
    # GRAPH 49: Final Model Performance Summary
    # ======================================================================
    print("\n  Generating Graph 49...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Final Best Model: Performance Summary', fontsize=18, fontweight='bold', color='white')
    
    # Panel 1: Before vs After improvement
    ax = axes[0]
    before_after = {
        'Phase 5\n(14 feat)': results['Our 14 (Phase 5)']['f1'],
        'Phase 6 Optuna\n(best feat set)': model_variants[best_model_name]['f1'],
        'FINAL\n(optimized)': r_final['f1'],
    }
    ba_colors = [C['blue'], C['accent'], C['primary']]
    bars = ax.bar(range(len(before_after)), list(before_after.values()), color=ba_colors, alpha=0.85, width=0.6)
    ax.set_xticks(range(len(before_after)))
    ax.set_xticklabels(list(before_after.keys()), fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=13)
    ax.set_title('F1 Improvement Journey', fontsize=14, color='white')
    for bar, val in zip(bars, before_after.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.set_facecolor(C['bg'])
    ax.set_ylim(0, max(before_after.values()) * 1.15)
    
    # Panel 2: Threshold curve
    ax = axes[1]
    ax.plot(threshs, f1_at_t_final, color=C['primary'], linewidth=2.5)
    ax.axvline(x=final_opt_t, color=C['warn'], linestyle='--', linewidth=2,
               label=f'Optimal: {final_opt_t:.3f} (F1={final_opt_f1:.4f})')
    ax.axvline(x=0.5, color='white', linestyle=':', linewidth=1, alpha=0.5, label='Default: 0.5')
    ax.set_xlabel('Threshold', fontsize=13)
    ax.set_ylabel('F1 Score', fontsize=13)
    ax.set_title('F1 vs Threshold', fontsize=14, color='white')
    ax.legend(fontsize=10)
    ax.set_facecolor(C['bg'])
    ax.grid(True, alpha=0.15)
    
    # Panel 3: Confusion matrix at optimal threshold
    ax = axes[2]
    preds_final_opt = (y_prob_final >= final_opt_t).astype(int)
    cm = confusion_matrix(y, preds_final_opt)
    im = ax.imshow(cm, cmap='magma', aspect='auto')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] < cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    fontsize=24, fontweight='bold', color=color)
    
    f1_final_opt = f1_score(y, preds_final_opt, zero_division=0)
    p_final = precision_score(y, preds_final_opt, zero_division=0)
    r_final_metric = recall_score(y, preds_final_opt, zero_division=0)
    ax.set_title(f't={final_opt_t:.3f} | F1={f1_final_opt:.3f} | P={p_final:.3f} | R={r_final_metric:.3f}',
                 fontsize=12, color='white')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Legit', 'Pred: Susp'], fontsize=11)
    ax.set_yticklabels(['True: Legit', 'True: Susp'], fontsize=11)
    ax.set_facecolor(C['bg'])
    
    plt.tight_layout()
    plt.savefig(f'{GRAPH_DIR}/49_final_model_summary.png', dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.close()
    print("  [OK] 49_final_model_summary.png")
    
    # ======================================================================
    # Save Phase 6 Summary
    # ======================================================================
    elapsed = time.time() - start
    
    summary = {
        'runtime_seconds': round(elapsed, 1),
        'best_feature_set': best_set_name,
        'best_feature_set_f1': round(f1s[best_idx], 4),
        'optimal_features': opt_feat_names,
        'n_features': len(opt_feat_names),
        'helpful_ps_features': helpful_ps,
        'final_model': {
            'f1_cv': round(r_final['f1'], 4),
            'f1_std': round(r_final['f1_std'], 4),
            'precision': round(r_final['precision'], 4),
            'recall': round(r_final['recall'], 4),
            'auc': round(r_final['auc'], 4),
            'optimal_threshold': round(float(final_opt_t), 4),
            'f1_at_optimal_threshold': round(float(final_opt_f1), 4),
        },
        'final_params': {k: round(v, 4) if isinstance(v, float) else v for k, v in final_params.items()},
        'smote': {'used': final_use_smote, 'ratio': round(final_smote_ratio, 3) if final_smote_ratio else None},
        'ps_feature_contributions': ps_contributions,
        'improvement_from_phase5': round(r_final['f1'] - results['Our 14 (Phase 5)']['f1'], 4),
    }
    
    with open('model/phase6_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print(f"PHASE 6 COMPLETE ({elapsed:.1f}s)")
    print(f"{'=' * 70}")
    print(f"  Final F1: {r_final['f1']:.4f} (was {results['Our 14 (Phase 5)']['f1']:.4f})")
    print(f"  Improvement: {r_final['f1'] - results['Our 14 (Phase 5)']['f1']:+.4f}")
    print(f"  Features: {len(opt_feat_names)} ({len(OUR_RAW)} raw + {len(opt_feat_names) - len(OUR_RAW)} engineered/PS)")
    print(f"  Optimal threshold: {final_opt_t:.3f} (F1={final_opt_f1:.4f})")
    print(f"  Graphs: 44-49 saved to {GRAPH_DIR}/")


if __name__ == '__main__':
    main()
