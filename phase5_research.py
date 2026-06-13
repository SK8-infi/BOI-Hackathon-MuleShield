"""
Phase 5: Pre-Modeling Research
==============================
1. Systematic Feature Engineering
2. Formal Forward Feature Selection
3. Cross-Validation Strategy Design
4. SMOTE Variant Comparison
5. Error Analysis / Misclassification Profiling
6. Threshold Optimization
7. Model Stacking / Blending
8. Temporal Stability Check
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, LeaveOneOut, cross_val_predict, cross_validate
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                             precision_recall_curve, classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Try importing optional packages
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    print("[OK] XGBoost available")
except ImportError:
    HAS_XGB = False
    print("[!!] XGBoost not available - will use GradientBoosting as fallback")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
    print("[OK] LightGBM available")
except ImportError:
    HAS_LGBM = False
    print("[!!] LightGBM not available")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
    print("[OK] CatBoost available")
except ImportError:
    HAS_CAT = False
    print("[!!] CatBoost not available")

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SMOTENC
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
    print("[OK] imbalanced-learn available")
except ImportError:
    HAS_IMBLEARN = False
    print("[!!] imbalanced-learn not available")

# ======================================================================
# Setup
# ======================================================================
plt.style.use('dark_background')
COLORS = {'legit': '#00f5d4', 'susp': '#ff006e', 'accent': '#8338ec',
           'warn': '#ffbe0b', 'blue': '#3a86ff', 'bg': '#0a0a23'}

OUT = 'files/graphs'
os.makedirs(OUT, exist_ok=True)

start_time = time.time()
print("\n" + "=" * 70)
print("PHASE 5: PRE-MODELING RESEARCH")
print("=" * 70)

# ======================================================================
# Load Data
# ======================================================================
print("\nLoading dataset...")
df = pd.read_csv('DataSet.csv', low_memory=False)
TARGET = 'F3924'
LEAK = ['F3912', 'F2230']  # Must exclude

# Separate target
y = df[TARGET].values

# Get numeric features only, exclude target + leakage
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude = [TARGET] + LEAK
num_cols = [c for c in num_cols if c not in exclude]
print(f"  {len(num_cols)} numeric features available (excluding target + leakage)")
print(f"  Class distribution: {np.bincount(y.astype(int))} (legit vs suspicious)")

# Top 8 stable features from Phase 3
STABLE_8 = ['F3811', 'F3806', 'F3799', 'F3805', 'F3813', 'F3801', 'F3898', 'F3807']
# Top permutation-important features from Phase 4
PERM_TOP = ['F3898', 'F162']

# ======================================================================
# 1. SYSTEMATIC FEATURE ENGINEERING
# ======================================================================
print("\n" + "=" * 70)
print("1. SYSTEMATIC FEATURE ENGINEERING")
print("=" * 70)

# Get F0-F499 columns
f0_f499 = [f'F{i}' for i in range(500) if f'F{i}' in df.columns]
print(f"  F0-F499 block: {len(f0_f499)} features")

# --- Feature 1: Missing count in F0-F499 ---
eng_missing = df[f0_f499].isnull().sum(axis=1).values
print(f"\n  [ENG] missing_count_F0_F500:")
susp_mask = y == 1
print(f"    Suspicious: mean={eng_missing[susp_mask].mean():.1f}, median={np.median(eng_missing[susp_mask]):.1f}")
print(f"    Legitimate: mean={eng_missing[~susp_mask].mean():.1f}, median={np.median(eng_missing[~susp_mask]):.1f}")
ks_stat, ks_p = stats.ks_2samp(eng_missing[susp_mask], eng_missing[~susp_mask])
print(f"    KS test: stat={ks_stat:.4f}, p={ks_p:.2e}")

# --- Feature 2: F3898 x F162 interaction ---
f3898 = df['F3898'].fillna(0).values
f162 = df['F162'].fillna(0).values
eng_interact = f3898 * f162
print(f"\n  [ENG] F3898_x_F162:")
print(f"    Suspicious: mean={eng_interact[susp_mask].mean():.4f}")
print(f"    Legitimate: mean={eng_interact[~susp_mask].mean():.4f}")
ks_stat, ks_p = stats.ks_2samp(eng_interact[susp_mask], eng_interact[~susp_mask])
print(f"    KS test: stat={ks_stat:.4f}, p={ks_p:.2e}")

# --- Feature 3: Ratio features between top stable ---
ratios_eng = {}
ratio_pairs = [('F3898', 'F3811'), ('F3898', 'F3805'), ('F3898', 'F3806'),
               ('F3811', 'F3805'), ('F3799', 'F3801'), ('F162', 'F3898')]
print(f"\n  [ENG] Ratio features:")
for f1, f2 in ratio_pairs:
    v1 = df[f1].fillna(0).values.astype(float)
    v2 = df[f2].fillna(0).values.astype(float)
    ratio = np.where(v2 != 0, v1 / (v2 + 1e-10), 0)
    # Clip extreme values
    ratio = np.clip(ratio, -1e6, 1e6)
    name = f"{f1}_div_{f2}"
    ratios_eng[name] = ratio
    ks_stat, ks_p = stats.ks_2samp(ratio[susp_mask], ratio[~susp_mask])
    mean_s = np.mean(ratio[susp_mask])
    mean_l = np.mean(ratio[~susp_mask])
    print(f"    {name}: susp_mean={mean_s:.4f}, legit_mean={mean_l:.4f}, KS={ks_stat:.4f}, p={ks_p:.2e}")

# --- Feature 4: Age risk score ---
acct_age = df['F3887'].fillna(9999).values  # account age in days
cust_age = df['F3894'].fillna(99).values    # customer age in years
# 20-25 year olds with <6 month accounts
eng_age_risk = ((cust_age >= 20) & (cust_age <= 25) & (acct_age < 180)).astype(int)
print(f"\n  [ENG] age_risk_score (20-25yo + <6mo account):")
print(f"    Flagged: {eng_age_risk.sum()} accounts ({eng_age_risk.sum()/len(eng_age_risk)*100:.1f}%)")
print(f"    Suspicious flagged: {eng_age_risk[susp_mask].sum()}/{susp_mask.sum()} ({eng_age_risk[susp_mask].mean()*100:.1f}%)")
print(f"    Legitimate flagged: {eng_age_risk[~susp_mask].sum()}/{(~susp_mask).sum()} ({eng_age_risk[~susp_mask].mean()*100:.1f}%)")

# --- Feature 5: Low value count (features below class-median) ---
# Use top 8 stable features for this
top8_df = df[STABLE_8].copy()
class_medians = top8_df[~susp_mask].median()  # legitimate class medians
eng_low_count = (top8_df < class_medians).sum(axis=1).values
print(f"\n  [ENG] low_value_count (features below legit median, top 8):")
print(f"    Suspicious: mean={eng_low_count[susp_mask].mean():.2f}, median={np.median(eng_low_count[susp_mask]):.1f}")
print(f"    Legitimate: mean={eng_low_count[~susp_mask].mean():.2f}, median={np.median(eng_low_count[~susp_mask]):.1f}")
ks_stat, ks_p = stats.ks_2samp(eng_low_count[susp_mask], eng_low_count[~susp_mask])
print(f"    KS test: stat={ks_stat:.4f}, p={ks_p:.2e}")

# --- Feature 6: F3898 deviation from mean (z-score) ---
f3898_z = (f3898 - np.nanmean(f3898)) / (np.nanstd(f3898) + 1e-10)
print(f"\n  [ENG] F3898_zscore:")
print(f"    Suspicious: mean={f3898_z[susp_mask].mean():.4f}")
print(f"    Legitimate: mean={f3898_z[~susp_mask].mean():.4f}")

# --- Feature 7: Max value across top 8 (signals overall activity level) ---
eng_max_top8 = top8_df.max(axis=1).fillna(0).values
print(f"\n  [ENG] max_value_top8:")
print(f"    Suspicious: mean={eng_max_top8[susp_mask].mean():.2f}")
print(f"    Legitimate: mean={eng_max_top8[~susp_mask].mean():.2f}")
ks_stat, ks_p = stats.ks_2samp(eng_max_top8[susp_mask], eng_max_top8[~susp_mask])
print(f"    KS test: stat={ks_stat:.4f}, p={ks_p:.2e}")

# Collect all engineered features
eng_features = {
    'missing_count_F0_F500': eng_missing,
    'F3898_x_F162': eng_interact,
    'age_risk_score': eng_age_risk,
    'low_value_count': eng_low_count,
    'F3898_zscore': f3898_z,
    'max_value_top8': eng_max_top8,
}
for name, vals in ratios_eng.items():
    eng_features[name] = vals

print(f"\n  Total engineered features: {len(eng_features)}")

# --- GRAPH 36: Feature Engineering Evaluation ---
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Engineered Feature Evaluation', fontsize=18, fontweight='bold', color='white', y=0.98)

plot_features = ['missing_count_F0_F500', 'F3898_x_F162', 'age_risk_score',
                 'low_value_count', 'F3898_div_F3811', 'max_value_top8']
for idx, fname in enumerate(plot_features):
    ax = axes[idx // 3, idx % 3]
    vals = eng_features[fname]
    ax.hist(vals[~susp_mask], bins=50, alpha=0.6, color=COLORS['legit'],
            label=f'Legit (n={int((~susp_mask).sum())})', density=True)
    ax.hist(vals[susp_mask], bins=30, alpha=0.7, color=COLORS['susp'],
            label=f'Susp (n={int(susp_mask.sum())})', density=True)
    ks_stat, ks_p = stats.ks_2samp(vals[susp_mask], vals[~susp_mask])
    ax.set_title(f'{fname}\nKS={ks_stat:.3f}, p={ks_p:.2e}', fontsize=11, color='white')
    ax.legend(fontsize=8)
    ax.set_facecolor(COLORS['bg'])

plt.tight_layout()
plt.savefig(f'{OUT}/36_feature_engineering.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a23', edgecolor='none')
plt.close()
print(f"  [OK] {OUT}/36_feature_engineering.png")


# ======================================================================
# 2. FORMAL FORWARD FEATURE SELECTION
# ======================================================================
print("\n" + "=" * 70)
print("2. FORMAL FORWARD FEATURE SELECTION")
print("=" * 70)

# Build the feature matrix with top candidates + engineered features
candidate_features = STABLE_8 + ['F162', 'F1057', 'F1813', 'F1815', 'F1819',
                                  'F1705', 'F1707', 'F949', 'F1165', 'F1921',
                                  'F2029', 'F3800', 'F3812']

# Create feature matrix
X_candidates = df[candidate_features].fillna(0).values
candidate_names = list(candidate_features)

# Add engineered features
for name, vals in eng_features.items():
    X_candidates = np.column_stack([X_candidates, vals])
    candidate_names.append(name)

print(f"  Total candidate features: {len(candidate_names)}")

# Forward selection using stratified 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_features(X, y, cv):
    """Evaluate feature set with stratified CV, returns mean F1."""
    f1_scores = []
    for train_idx, test_idx in cv.split(X, y):
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        f1_scores.append(f1_score(y[test_idx], preds, zero_division=0))
    return np.mean(f1_scores), np.std(f1_scores)

# Start with F3898 (index 0 = F3811... F3898 is index 6)
f3898_idx = candidate_names.index('F3898')
selected = [f3898_idx]
remaining = list(range(len(candidate_names)))
remaining.remove(f3898_idx)

selection_results = []
f1_mean, f1_std = evaluate_features(X_candidates[:, selected], y, cv)
selection_results.append({
    'n_features': 1,
    'features': ['F3898'],
    'f1_mean': f1_mean,
    'f1_std': f1_std,
    'added': 'F3898'
})
print(f"  Step 1: F3898 alone -> F1={f1_mean:.4f} (+/-{f1_std:.4f})")

# Forward selection
max_features = min(20, len(candidate_names))
for step in range(2, max_features + 1):
    best_f1 = -1
    best_idx = -1
    best_std = 0

    for idx in remaining:
        trial = selected + [idx]
        f1_mean, f1_std = evaluate_features(X_candidates[:, trial], y, cv)
        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_idx = idx
            best_std = f1_std

    selected.append(best_idx)
    remaining.remove(best_idx)
    added_name = candidate_names[best_idx]
    selected_names = [candidate_names[i] for i in selected]
    selection_results.append({
        'n_features': step,
        'features': selected_names.copy(),
        'f1_mean': best_f1,
        'f1_std': best_std,
        'added': added_name
    })
    print(f"  Step {step}: +{added_name} -> F1={best_f1:.4f} (+/-{best_std:.4f})")

    # Early stopping if F1 plateaus
    if step > 5 and best_f1 < selection_results[-3]['f1_mean'] - 0.01:
        print(f"  [STOP] F1 declining, stopping at {step} features")
        break

# Find optimal
optimal_idx = np.argmax([r['f1_mean'] for r in selection_results])
optimal = selection_results[optimal_idx]
print(f"\n  OPTIMAL: {optimal['n_features']} features -> F1={optimal['f1_mean']:.4f}")
print(f"  Features: {optimal['features']}")

OPTIMAL_FEATURES = optimal['features']
OPTIMAL_INDICES = [candidate_names.index(f) for f in OPTIMAL_FEATURES]

# --- GRAPH 37: Forward Feature Selection Curve ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Forward Feature Selection', fontsize=18, fontweight='bold', color='white')

ns = [r['n_features'] for r in selection_results]
f1s = [r['f1_mean'] for r in selection_results]
stds = [r['f1_std'] for r in selection_results]
labels = [r['added'] for r in selection_results]

ax1.plot(ns, f1s, 'o-', color=COLORS['susp'], linewidth=2, markersize=8)
ax1.fill_between(ns, [f-s for f,s in zip(f1s, stds)], [f+s for f,s in zip(f1s, stds)],
                 alpha=0.2, color=COLORS['susp'])
ax1.axvline(x=optimal['n_features'], color=COLORS['warn'], linestyle='--', linewidth=2,
            label=f'Optimal: {optimal["n_features"]} features')
ax1.set_xlabel('Number of Features', fontsize=13)
ax1.set_ylabel('Mean F1 Score (5-Fold CV)', fontsize=13)
ax1.set_title('F1 vs Feature Count', fontsize=14, color='white')
ax1.legend(fontsize=11)
ax1.set_facecolor(COLORS['bg'])
ax1.grid(True, alpha=0.2)

# Bar chart of added features
colors_bar = [COLORS['susp'] if i == optimal_idx else COLORS['blue'] for i in range(len(labels))]
ax2.barh(range(len(labels)), f1s, color=colors_bar, alpha=0.8)
ax2.set_yticks(range(len(labels)))
ax2.set_yticklabels([f'+{l}' for l in labels], fontsize=10)
ax2.set_xlabel('Cumulative F1 Score', fontsize=13)
ax2.set_title('Feature Added at Each Step', fontsize=14, color='white')
ax2.set_facecolor(COLORS['bg'])
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{OUT}/37_forward_selection.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a23', edgecolor='none')
plt.close()
print(f"  [OK] {OUT}/37_forward_selection.png")


# ======================================================================
# 3. CROSS-VALIDATION STRATEGY DESIGN
# ======================================================================
print("\n" + "=" * 70)
print("3. CROSS-VALIDATION STRATEGY DESIGN")
print("=" * 70)

# Use the optimal feature set
X_opt = X_candidates[:, OPTIMAL_INDICES]

cv_strategies = {
    'Stratified 5-Fold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'Stratified 10-Fold': StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    'Repeated 5-Fold (3x)': RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42),
    'Repeated 5-Fold (5x)': RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42),
}

cv_results = {}
for name, cv_strat in cv_strategies.items():
    print(f"\n  Testing: {name}...")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    scores = cross_validate(clf, X_opt, y, cv=cv_strat,
                            scoring=['f1', 'precision', 'recall', 'roc_auc'],
                            return_train_score=False)
    cv_results[name] = {
        'f1_mean': np.mean(scores['test_f1']),
        'f1_std': np.std(scores['test_f1']),
        'precision_mean': np.mean(scores['test_precision']),
        'precision_std': np.std(scores['test_precision']),
        'recall_mean': np.mean(scores['test_recall']),
        'recall_std': np.std(scores['test_recall']),
        'auc_mean': np.mean(scores['test_roc_auc']),
        'auc_std': np.std(scores['test_roc_auc']),
        'n_splits': len(scores['test_f1']),
    }
    print(f"    F1={cv_results[name]['f1_mean']:.4f} +/- {cv_results[name]['f1_std']:.4f}")
    print(f"    Precision={cv_results[name]['precision_mean']:.4f}, Recall={cv_results[name]['recall_mean']:.4f}, AUC={cv_results[name]['auc_mean']:.4f}")

# LOO for the 81 suspicious only (full LOO is expensive)
print(f"\n  Testing: Leave-One-Out (on minority only)...")
# Use stratified 5-fold but report per-sample predictions
clf_loo = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=3,
    class_weight='balanced', random_state=42, n_jobs=-1
)
y_pred_cv = cross_val_predict(clf_loo, X_opt, y,
                               cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42))
f1_loo = f1_score(y, y_pred_cv)
prec_loo = precision_score(y, y_pred_cv, zero_division=0)
rec_loo = recall_score(y, y_pred_cv, zero_division=0)
susp_caught = y_pred_cv[susp_mask].sum()
print(f"    10-Fold per-sample predictions: F1={f1_loo:.4f}, Precision={prec_loo:.4f}, Recall={rec_loo:.4f}")
print(f"    Suspicious caught: {int(susp_caught)}/{int(susp_mask.sum())}")

cv_results['10-Fold (per-sample)'] = {
    'f1_mean': f1_loo, 'f1_std': 0.0,
    'precision_mean': prec_loo, 'precision_std': 0.0,
    'recall_mean': rec_loo, 'recall_std': 0.0,
    'auc_mean': 0.0, 'auc_std': 0.0,
    'n_splits': 1,
}

# --- GRAPH 38: CV Strategy Comparison ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Cross-Validation Strategy Comparison', fontsize=18, fontweight='bold', color='white')

strat_names = list(cv_results.keys())
f1_means = [cv_results[s]['f1_mean'] for s in strat_names]
f1_stds = [cv_results[s]['f1_std'] for s in strat_names]

bars = ax1.barh(range(len(strat_names)), f1_means, xerr=f1_stds,
                color=COLORS['blue'], alpha=0.8, capsize=5, ecolor='white')
ax1.set_yticks(range(len(strat_names)))
ax1.set_yticklabels(strat_names, fontsize=11)
ax1.set_xlabel('F1 Score', fontsize=13)
ax1.set_title('F1 Score by CV Strategy', fontsize=14, color='white')
ax1.set_facecolor(COLORS['bg'])
ax1.invert_yaxis()
for i, (m, s) in enumerate(zip(f1_means, f1_stds)):
    ax1.text(m + s + 0.005, i, f'{m:.3f}', va='center', fontsize=10, color='white')

# Precision vs Recall by strategy
prec_means = [cv_results[s]['precision_mean'] for s in strat_names[:-1]]
rec_means = [cv_results[s]['recall_mean'] for s in strat_names[:-1]]
short_names = strat_names[:-1]
x_pos = np.arange(len(short_names))
width = 0.35
ax2.bar(x_pos - width/2, prec_means, width, label='Precision', color=COLORS['legit'], alpha=0.8)
ax2.bar(x_pos + width/2, rec_means, width, label='Recall', color=COLORS['susp'], alpha=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(short_names, fontsize=9, rotation=15, ha='right')
ax2.set_ylabel('Score', fontsize=13)
ax2.set_title('Precision vs Recall by Strategy', fontsize=14, color='white')
ax2.legend(fontsize=11)
ax2.set_facecolor(COLORS['bg'])

plt.tight_layout()
plt.savefig(f'{OUT}/38_cv_strategies.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a23', edgecolor='none')
plt.close()
print(f"  [OK] {OUT}/38_cv_strategies.png")


# ======================================================================
# 4. SMOTE VARIANT COMPARISON
# ======================================================================
print("\n" + "=" * 70)
print("4. SMOTE VARIANT COMPARISON")
print("=" * 70)

smote_results = {}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Baseline: No oversampling, balanced class weights
print("\n  Testing: No SMOTE (class_weight=balanced)...")
f1s_baseline = []
for train_idx, test_idx in cv5.split(X_opt, y):
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3,
                                  class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_opt[train_idx], y[train_idx])
    preds = clf.predict(X_opt[test_idx])
    f1s_baseline.append(f1_score(y[test_idx], preds, zero_division=0))
smote_results['No SMOTE\n(balanced weights)'] = {'f1_mean': np.mean(f1s_baseline), 'f1_std': np.std(f1s_baseline)}
print(f"    F1={np.mean(f1s_baseline):.4f} +/- {np.std(f1s_baseline):.4f}")

# Baseline: No oversampling, scale_pos_weight
print("  Testing: No SMOTE (scale_pos_weight)...")
f1s_spw = []
if HAS_XGB:
    pos_weight = (~susp_mask).sum() / susp_mask.sum()
    for train_idx, test_idx in cv5.split(X_opt, y):
        clf = XGBClassifier(n_estimators=200, max_depth=6, scale_pos_weight=pos_weight,
                            random_state=42, use_label_encoder=False, eval_metric='logloss',
                            verbosity=0, n_jobs=-1)
        clf.fit(X_opt[train_idx], y[train_idx])
        preds = clf.predict(X_opt[test_idx])
        f1s_spw.append(f1_score(y[test_idx], preds, zero_division=0))
else:
    for train_idx, test_idx in cv5.split(X_opt, y):
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
        # Manually weight samples
        w = np.ones(len(train_idx))
        w[y[train_idx] == 1] = (~susp_mask).sum() / susp_mask.sum()
        clf.fit(X_opt[train_idx], y[train_idx], sample_weight=w)
        preds = clf.predict(X_opt[test_idx])
        f1s_spw.append(f1_score(y[test_idx], preds, zero_division=0))

smote_results['No SMOTE\n(scale_pos_weight)'] = {'f1_mean': np.mean(f1s_spw), 'f1_std': np.std(f1s_spw)}
print(f"    F1={np.mean(f1s_spw):.4f} +/- {np.std(f1s_spw):.4f}")

if HAS_IMBLEARN:
    # SMOTE variants at different ratios
    smote_configs = {
        'SMOTE\n(ratio=0.1)': SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=3),
        'SMOTE\n(ratio=0.3)': SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=3),
        'SMOTE\n(ratio=0.5)': SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=3),
        'SMOTE\n(ratio=1.0)': SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=3),
        'Borderline\nSMOTE': BorderlineSMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=3),
        'ADASYN': ADASYN(sampling_strategy=0.3, random_state=42, n_neighbors=3),
        'SMOTE\n+Tomek': SMOTETomek(sampling_strategy=0.3, random_state=42,
                                     smote=SMOTE(k_neighbors=3, random_state=42)),
    }

    for name, sampler in smote_configs.items():
        print(f"  Testing: {name.replace(chr(10), ' ')}...")
        f1s = []
        for train_idx, test_idx in cv5.split(X_opt, y):
            try:
                X_res, y_res = sampler.fit_resample(X_opt[train_idx], y[train_idx])
                clf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3,
                                              random_state=42, n_jobs=-1)
                clf.fit(X_res, y_res)
                preds = clf.predict(X_opt[test_idx])
                f1s.append(f1_score(y[test_idx], preds, zero_division=0))
            except Exception as e:
                print(f"    Error: {e}")
                f1s.append(0.0)
        smote_results[name] = {'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s)}
        print(f"    F1={np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
else:
    print("  [SKIP] imbalanced-learn not installed, skipping SMOTE variants")

# --- GRAPH 39: SMOTE Comparison ---
fig, ax = plt.subplots(figsize=(14, 8))
fig.suptitle('SMOTE Variant Comparison (5-Fold CV)', fontsize=18, fontweight='bold', color='white')

smote_names = list(smote_results.keys())
smote_f1s = [smote_results[s]['f1_mean'] for s in smote_names]
smote_stds = [smote_results[s]['f1_std'] for s in smote_names]

best_smote_idx = np.argmax(smote_f1s)
colors_smote = [COLORS['susp'] if i == best_smote_idx else COLORS['blue'] for i in range(len(smote_names))]

bars = ax.barh(range(len(smote_names)), smote_f1s, xerr=smote_stds,
               color=colors_smote, alpha=0.8, capsize=5, ecolor='white')
ax.set_yticks(range(len(smote_names)))
ax.set_yticklabels(smote_names, fontsize=11)
ax.set_xlabel('F1 Score', fontsize=13)
ax.set_title(f'Best: {smote_names[best_smote_idx].replace(chr(10), " ")} (F1={smote_f1s[best_smote_idx]:.4f})',
             fontsize=14, color=COLORS['warn'])
ax.set_facecolor(COLORS['bg'])
ax.invert_yaxis()
ax.grid(True, alpha=0.2, axis='x')
for i, (m, s) in enumerate(zip(smote_f1s, smote_stds)):
    ax.text(m + s + 0.005, i, f'{m:.3f}', va='center', fontsize=10, color='white')

plt.tight_layout()
plt.savefig(f'{OUT}/39_smote_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a23', edgecolor='none')
plt.close()
print(f"  [OK] {OUT}/39_smote_comparison.png")


# ======================================================================
# 5. ERROR ANALYSIS / MISCLASSIFICATION PROFILING
# ======================================================================
print("\n" + "=" * 70)
print("5. ERROR ANALYSIS / MISCLASSIFICATION PROFILING")
print("=" * 70)

# Get per-sample predictions using 10-fold CV
clf_err = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=3,
    class_weight='balanced', random_state=42, n_jobs=-1
)
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_pred_all = cross_val_predict(clf_err, X_opt, y, cv=cv10)
y_prob_all = cross_val_predict(clf_err, X_opt, y, cv=cv10, method='predict_proba')[:, 1]

# Focus on the 81 suspicious accounts
susp_indices = np.where(susp_mask)[0]
susp_preds = y_pred_all[susp_indices]
susp_probs = y_prob_all[susp_indices]
caught = susp_preds == 1
missed = susp_preds == 0

print(f"\n  Suspicious accounts: {len(susp_indices)}")
print(f"  Caught (TP): {caught.sum()}")
print(f"  Missed (FN): {missed.sum()}")
print(f"  Recall: {caught.sum()/len(susp_indices)*100:.1f}%")

# False positives
fp_mask = (y_pred_all == 1) & (y == 0)
print(f"  False Positives: {fp_mask.sum()}")
print(f"  Precision: {caught.sum()/(caught.sum() + fp_mask.sum())*100:.1f}%")

# Profile missed mules
if missed.sum() > 0:
    missed_idx = susp_indices[missed]
    caught_idx = susp_indices[caught]
    print(f"\n  MISSED MULE PROFILES (n={missed.sum()}):")
    print(f"  {'':>5} {'AcctAge':>10} {'CustAge':>10} {'F3898':>10} {'F162':>10} {'Prob':>8}")
    print(f"  {'-'*55}")
    for i, idx in enumerate(missed_idx[:20]):
        acct = df.iloc[idx]['F3887'] if not pd.isna(df.iloc[idx]['F3887']) else 'N/A'
        cust = df.iloc[idx]['F3894'] if not pd.isna(df.iloc[idx]['F3894']) else 'N/A'
        f3898_v = df.iloc[idx]['F3898'] if not pd.isna(df.iloc[idx]['F3898']) else 'N/A'
        f162_v = df.iloc[idx]['F162'] if not pd.isna(df.iloc[idx]['F162']) else 'N/A'
        print(f"  {i+1:>5} {str(acct):>10} {str(cust):>10} {str(f3898_v):>10} {str(f162_v):>10} {susp_probs[missed][i]:>8.4f}")

    # Compare caught vs missed
    print(f"\n  CAUGHT vs MISSED comparison:")
    for feat in ['F3898', 'F162', 'F3887', 'F3894', 'F3811', 'F3805']:
        if feat in df.columns:
            caught_vals = df.iloc[caught_idx][feat].dropna()
            missed_vals = df.iloc[missed_idx][feat].dropna()
            if len(caught_vals) > 0 and len(missed_vals) > 0:
                print(f"    {feat}: Caught mean={caught_vals.mean():.4f}, Missed mean={missed_vals.mean():.4f}")

# Probability distribution
print(f"\n  Probability distribution of suspicious accounts:")
prob_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(len(prob_bins) - 1):
    count = ((susp_probs >= prob_bins[i]) & (susp_probs < prob_bins[i+1])).sum()
    if count > 0:
        print(f"    [{prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}): {count} accounts")

# --- GRAPH 40: Error Analysis ---
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Error Analysis: Misclassification Profiling', fontsize=18, fontweight='bold', color='white', y=0.98)

# Plot 1: Probability histogram of suspicious accounts
ax = axes[0, 0]
ax.hist(susp_probs[caught], bins=20, alpha=0.7, color=COLORS['legit'], label=f'Caught (n={caught.sum()})')
ax.hist(susp_probs[missed], bins=20, alpha=0.7, color=COLORS['susp'], label=f'Missed (n={missed.sum()})')
ax.axvline(x=0.5, color=COLORS['warn'], linestyle='--', linewidth=2, label='Threshold=0.5')
ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Suspicious Account Probability Distribution', fontsize=13, color='white')
ax.legend(fontsize=10)
ax.set_facecolor(COLORS['bg'])

# Plot 2: Caught vs Missed feature comparison
ax = axes[0, 1]
compare_feats = ['F3898', 'F162', 'F3811', 'F3805', 'F3887']
compare_feats = [f for f in compare_feats if f in df.columns]
caught_means = [df.iloc[caught_idx][f].fillna(0).mean() for f in compare_feats]
missed_means = [df.iloc[missed_idx][f].fillna(0).mean() for f in compare_feats] if missed.sum() > 0 else [0]*len(compare_feats)
# Normalize for display
max_vals = [max(abs(c), abs(m), 1) for c, m in zip(caught_means, missed_means)]
caught_norm = [c/m for c, m in zip(caught_means, max_vals)]
missed_norm = [c/m for c, m in zip(missed_means, max_vals)]
x_comp = np.arange(len(compare_feats))
ax.bar(x_comp - 0.2, caught_norm, 0.4, label='Caught', color=COLORS['legit'], alpha=0.8)
ax.bar(x_comp + 0.2, missed_norm, 0.4, label='Missed', color=COLORS['susp'], alpha=0.8)
ax.set_xticks(x_comp)
ax.set_xticklabels(compare_feats, fontsize=10)
ax.set_ylabel('Normalized Mean Value', fontsize=12)
ax.set_title('Caught vs Missed: Feature Comparison', fontsize=13, color='white')
ax.legend(fontsize=10)
ax.set_facecolor(COLORS['bg'])

# Plot 3: Customer age distribution (caught vs missed)
ax = axes[1, 0]
if missed.sum() > 0:
    caught_ages = df.iloc[caught_idx]['F3894'].dropna()
    missed_ages = df.iloc[missed_idx]['F3894'].dropna()
    ax.hist(caught_ages, bins=15, alpha=0.6, color=COLORS['legit'], label=f'Caught (n={len(caught_ages)})', density=True)
    ax.hist(missed_ages, bins=10, alpha=0.6, color=COLORS['susp'], label=f'Missed (n={len(missed_ages)})', density=True)
    ax.set_xlabel('Customer Age', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Customer Age: Caught vs Missed Mules', fontsize=13, color='white')
    ax.legend(fontsize=10)
ax.set_facecolor(COLORS['bg'])

# Plot 4: Confusion matrix heatmap
ax = axes[1, 1]
cm = confusion_matrix(y, y_pred_all)
im = ax.imshow(cm, cmap='magma', aspect='auto')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i,j]}', ha='center', va='center', fontsize=20,
                color='white' if cm[i,j] < cm.max()/2 else 'black', fontweight='bold')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred: Legit', 'Pred: Susp'], fontsize=12)
ax.set_yticklabels(['True: Legit', 'True: Susp'], fontsize=12)
ax.set_title('Confusion Matrix (10-Fold CV)', fontsize=13, color='white')
fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(f'{OUT}/40_error_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a23', edgecolor='none')
plt.close()
print(f"  [OK] {OUT}/40_error_analysis.png")


# ======================================================================
# 6. THRESHOLD OPTIMIZATION
# ======================================================================
print("\n" + "=" * 70)
print("6. THRESHOLD OPTIMIZATION")
print("=" * 70)

precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y, y_prob_all)
f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-10)

# Best threshold by F1
best_thresh_idx = np.argmax(f1_curve)
best_threshold = thresholds_pr[best_thresh_idx] if best_thresh_idx < len(thresholds_pr) else 0.5
best_f1_thresh = f1_curve[best_thresh_idx]
print(f"\n  Optimal threshold by F1: {best_threshold:.4f}")
print(f"  F1 at optimal: {best_f1_thresh:.4f}")
print(f"  Precision at optimal: {precision_curve[best_thresh_idx]:.4f}")
print(f"  Recall at optimal: {recall_curve[best_thresh_idx]:.4f}")

# Compare with default 0.5
y_pred_05 = (y_prob_all >= 0.5).astype(int)
y_pred_opt = (y_prob_all >= best_threshold).astype(int)
print(f"\n  Default (0.5): F1={f1_score(y, y_pred_05):.4f}, P={precision_score(y, y_pred_05, zero_division=0):.4f}, R={recall_score(y, y_pred_05):.4f}")
print(f"  Optimal ({best_threshold:.3f}): F1={f1_score(y, y_pred_opt):.4f}, P={precision_score(y, y_pred_opt, zero_division=0):.4f}, R={recall_score(y, y_pred_opt):.4f}")

# Business cost analysis
print(f"\n  Business Cost Analysis (example costs):")
cost_fp = 100  # Cost of investigating a false positive
cost_fn = 10000  # Cost of missing a mule account
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, best_threshold, 0.6, 0.7, 0.8, 0.9]:
    y_t = (y_prob_all >= thresh).astype(int)
    fp = ((y_t == 1) & (y == 0)).sum()
    fn = ((y_t == 0) & (y == 1)).sum()
    tp = ((y_t == 1) & (y == 1)).sum()
    total_cost = fp * cost_fp + fn * cost_fn
    print(f"    Threshold={thresh:.3f}: TP={tp:3d}, FP={fp:4d}, FN={fn:3d}, Cost=${total_cost:>10,}")

# --- GRAPH 41: Threshold Optimization ---
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('Threshold Optimization', fontsize=18, fontweight='bold', color='white')

# PR Curve
ax = axes[0]
ax.plot(recall_curve, precision_curve, color=COLORS['susp'], linewidth=2)
ax.scatter([recall_curve[best_thresh_idx]], [precision_curve[best_thresh_idx]],
           color=COLORS['warn'], s=150, zorder=5, label=f'Best (t={best_threshold:.3f})')
ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Precision-Recall Curve', fontsize=14, color='white')
ax.legend(fontsize=11)
ax.set_facecolor(COLORS['bg'])
ax.grid(True, alpha=0.2)

# F1 vs Threshold
ax = axes[1]
valid_thresh = thresholds_pr[:len(f1_curve)-1] if len(thresholds_pr) < len(f1_curve) else thresholds_pr
ax.plot(thresholds_pr, f1_curve[:-1], color=COLORS['blue'], linewidth=2, label='F1')
ax.plot(thresholds_pr, precision_curve[:-1], color=COLORS['legit'], linewidth=1.5, alpha=0.7, label='Precision')
ax.plot(thresholds_pr, recall_curve[:-1], color=COLORS['susp'], linewidth=1.5, alpha=0.7, label='Recall')
ax.axvline(x=best_threshold, color=COLORS['warn'], linestyle='--', linewidth=2,
           label=f'Optimal={best_threshold:.3f}')
ax.axvline(x=0.5, color='white', linestyle=':', linewidth=1, alpha=0.5, label='Default=0.5')
ax.set_xlabel('Threshold', fontsize=13)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('F1/Precision/Recall vs Threshold', fontsize=14, color='white')
ax.legend(fontsize=9)
ax.set_facecolor(COLORS['bg'])
ax.grid(True, alpha=0.2)

# Business cost curve
ax = axes[2]
threshs = np.arange(0.05, 0.95, 0.01)
costs = []
tps, fps, fns = [], [], []
for t in threshs:
    y_t = (y_prob_all >= t).astype(int)
    fp = ((y_t == 1) & (y == 0)).sum()
    fn = ((y_t == 0) & (y == 1)).sum()
    tp = ((y_t == 1) & (y == 1)).sum()
    costs.append(fp * cost_fp + fn * cost_fn)
    tps.append(tp)
    fps.append(fp)
    fns.append(fn)

min_cost_idx = np.argmin(costs)
ax.plot(threshs, costs, color=COLORS['accent'], linewidth=2)
ax.scatter([threshs[min_cost_idx]], [costs[min_cost_idx]], color=COLORS['warn'], s=150, zorder=5,
           label=f'Min cost at t={threshs[min_cost_idx]:.2f}')
ax.set_xlabel('Threshold', fontsize=13)
ax.set_ylabel('Total Business Cost ($)', fontsize=13)
ax.set_title(f'Business Cost (FP=${cost_fp}, FN=${cost_fn:,})', fontsize=14, color='white')
ax.legend(fontsize=10)
ax.set_facecolor(COLORS['bg'])
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUT}/41_threshold_optimization.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a23', edgecolor='none')
plt.close()
print(f"  [OK] {OUT}/41_threshold_optimization.png")


# ======================================================================
# 7. MODEL STACKING / BLENDING
# ======================================================================
print("\n" + "=" * 70)
print("7. MODEL STACKING / BLENDING")
print("=" * 70)

# Build individual models
models = {}
models['RF'] = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=3,
    class_weight='balanced', random_state=42, n_jobs=-1
)

if HAS_XGB:
    pos_weight = (~susp_mask).sum() / susp_mask.sum()
    models['XGBoost'] = XGBClassifier(
        n_estimators=300, max_depth=6, scale_pos_weight=pos_weight,
        learning_rate=0.05, random_state=42, use_label_encoder=False,
        eval_metric='logloss', verbosity=0, n_jobs=-1
    )

if HAS_LGBM:
    models['LightGBM'] = LGBMClassifier(
        n_estimators=300, max_depth=6, class_weight='balanced',
        learning_rate=0.05, random_state=42, verbosity=-1, n_jobs=-1
    )

if HAS_CAT:
    models['CatBoost'] = CatBoostClassifier(
        iterations=300, depth=6, auto_class_weights='Balanced',
        learning_rate=0.05, random_seed=42, verbose=0
    )

models['GBM'] = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
)

# Evaluate each model individually
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_scores = {}

for name, model in models.items():
    print(f"\n  Evaluating: {name}...")
    f1s, precs, recs, aucs = [], [], [], []
    for train_idx, test_idx in cv5.split(X_opt, y):
        model_clone = type(model)(**model.get_params())
        if name == 'GBM':
            # Manual sample weighting for GBM
            w = np.ones(len(train_idx))
            w[y[train_idx] == 1] = (~susp_mask).sum() / susp_mask.sum()
            model_clone.fit(X_opt[train_idx], y[train_idx], sample_weight=w)
        else:
            model_clone.fit(X_opt[train_idx], y[train_idx])
        preds = model_clone.predict(X_opt[test_idx])
        probs = model_clone.predict_proba(X_opt[test_idx])[:, 1]
        f1s.append(f1_score(y[test_idx], preds, zero_division=0))
        precs.append(precision_score(y[test_idx], preds, zero_division=0))
        recs.append(recall_score(y[test_idx], preds, zero_division=0))
        aucs.append(roc_auc_score(y[test_idx], probs))

    model_scores[name] = {
        'f1': np.mean(f1s), 'f1_std': np.std(f1s),
        'precision': np.mean(precs), 'recall': np.mean(recs),
        'auc': np.mean(aucs), 'auc_std': np.std(aucs)
    }
    print(f"    F1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}, AUC={np.mean(aucs):.4f}, P={np.mean(precs):.4f}, R={np.mean(recs):.4f}")

# Soft Voting Ensemble
print(f"\n  Evaluating: Soft Voting Ensemble...")
voting_estimators = [(name, model) for name, model in models.items() if name != 'GBM']
if len(voting_estimators) >= 2:
    f1s_vote = []
    for train_idx, test_idx in cv5.split(X_opt, y):
        # Manual soft voting
        probs_list = []
        for mname, model in voting_estimators:
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_opt[train_idx], y[train_idx])
            probs_list.append(model_clone.predict_proba(X_opt[test_idx])[:, 1])
        avg_probs = np.mean(probs_list, axis=0)
        preds = (avg_probs >= 0.5).astype(int)
        f1s_vote.append(f1_score(y[test_idx], preds, zero_division=0))
    model_scores['Soft Voting'] = {
        'f1': np.mean(f1s_vote), 'f1_std': np.std(f1s_vote),
        'precision': 0.0, 'recall': 0.0, 'auc': 0.0, 'auc_std': 0.0
    }
    print(f"    F1={np.mean(f1s_vote):.4f}+/-{np.std(f1s_vote):.4f}")

# Stacking with Logistic Regression meta-learner
print(f"\n  Evaluating: Stacking (LR meta-learner)...")
stack_estimators = [(name, model) for name, model in models.items() if name != 'GBM']
if len(stack_estimators) >= 2:
    f1s_stack = []
    for train_idx, test_idx in cv5.split(X_opt, y):
        # Generate meta-features
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        meta_train = np.zeros((len(train_idx), len(stack_estimators)))
        meta_test = np.zeros((len(test_idx), len(stack_estimators)))

        for m_idx, (mname, model) in enumerate(stack_estimators):
            model_clone = type(model)(**model.get_params())
            # Inner CV for meta-features on train
            inner_preds = np.zeros(len(train_idx))
            for it, (it_train, it_val) in enumerate(inner_cv.split(X_opt[train_idx], y[train_idx])):
                mc = type(model)(**model.get_params())
                mc.fit(X_opt[train_idx][it_train], y[train_idx][it_train])
                inner_preds[it_val] = mc.predict_proba(X_opt[train_idx][it_val])[:, 1]
            meta_train[:, m_idx] = inner_preds
            # Full train for test meta-features
            model_clone.fit(X_opt[train_idx], y[train_idx])
            meta_test[:, m_idx] = model_clone.predict_proba(X_opt[test_idx])[:, 1]

        # Meta-learner
        meta_clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        meta_clf.fit(meta_train, y[train_idx])
        preds = meta_clf.predict(meta_test)
        f1s_stack.append(f1_score(y[test_idx], preds, zero_division=0))

    model_scores['Stacking (LR)'] = {
        'f1': np.mean(f1s_stack), 'f1_std': np.std(f1s_stack),
        'precision': 0.0, 'recall': 0.0, 'auc': 0.0, 'auc_std': 0.0
    }
    print(f"    F1={np.mean(f1s_stack):.4f}+/-{np.std(f1s_stack):.4f}")

# --- GRAPH 42: Model Comparison ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Model Stacking / Blending Comparison', fontsize=18, fontweight='bold', color='white')

m_names = list(model_scores.keys())
m_f1s = [model_scores[m]['f1'] for m in m_names]
m_stds = [model_scores[m]['f1_std'] for m in m_names]

best_model_idx = np.argmax(m_f1s)
colors_model = [COLORS['susp'] if i == best_model_idx else COLORS['blue'] for i in range(len(m_names))]

ax1.barh(range(len(m_names)), m_f1s, xerr=m_stds, color=colors_model, alpha=0.8, capsize=5, ecolor='white')
ax1.set_yticks(range(len(m_names)))
ax1.set_yticklabels(m_names, fontsize=12)
ax1.set_xlabel('F1 Score', fontsize=13)
ax1.set_title(f'Best: {m_names[best_model_idx]} (F1={m_f1s[best_model_idx]:.4f})', fontsize=14, color=COLORS['warn'])
ax1.set_facecolor(COLORS['bg'])
ax1.invert_yaxis()
ax1.grid(True, alpha=0.2, axis='x')
for i, (m, s) in enumerate(zip(m_f1s, m_stds)):
    ax1.text(m + s + 0.005, i, f'{m:.3f}', va='center', fontsize=10, color='white')

# AUC comparison (only for individual models)
indiv = [m for m in m_names if model_scores[m]['auc'] > 0]
ax2.barh(range(len(indiv)), [model_scores[m]['auc'] for m in indiv],
         color=COLORS['accent'], alpha=0.8)
ax2.set_yticks(range(len(indiv)))
ax2.set_yticklabels(indiv, fontsize=12)
ax2.set_xlabel('ROC-AUC', fontsize=13)
ax2.set_title('AUC Comparison (Individual Models)', fontsize=14, color='white')
ax2.set_facecolor(COLORS['bg'])
ax2.invert_yaxis()
for i, m in enumerate(indiv):
    ax2.text(model_scores[m]['auc'] + 0.002, i,
             f'{model_scores[m]["auc"]:.3f}', va='center', fontsize=10, color='white')

plt.tight_layout()
plt.savefig(f'{OUT}/42_model_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a23', edgecolor='none')
plt.close()
print(f"  [OK] {OUT}/42_model_comparison.png")


# ======================================================================
# 8. TEMPORAL STABILITY CHECK
# ======================================================================
print("\n" + "=" * 70)
print("8. TEMPORAL STABILITY CHECK")
print("=" * 70)

# F2230 is the reporting period — use it to check temporal stability
if 'F2230' in df.columns:
    periods = df['F2230'].dropna().unique()
    periods.sort()
    print(f"\n  Reporting periods (F2230): {periods}")
    print(f"  Period value counts:")
    for p in periods:
        mask_p = df['F2230'] == p
        n_total = mask_p.sum()
        n_susp = (mask_p & susp_mask).sum()
        print(f"    Period {p}: {n_total} total, {n_susp} suspicious ({n_susp/n_total*100:.2f}%)")

    if len(periods) >= 2:
        # Train on first period(s), test on last period
        temporal_results = []

        for i, test_period in enumerate(periods):
            train_periods = [p for p in periods if p != test_period]
            train_mask = df['F2230'].isin(train_periods)
            test_mask = df['F2230'] == test_period

            X_train_t = X_opt[train_mask]
            y_train_t = y[train_mask]
            X_test_t = X_opt[test_mask]
            y_test_t = y[test_mask]

            if y_train_t.sum() < 3 or y_test_t.sum() < 1:
                print(f"    Period {test_period}: Insufficient suspicious samples, skipping")
                continue

            clf_t = RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=3,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            clf_t.fit(X_train_t, y_train_t)
            preds_t = clf_t.predict(X_test_t)
            probs_t = clf_t.predict_proba(X_test_t)[:, 1]

            f1_t = f1_score(y_test_t, preds_t, zero_division=0)
            p_t = precision_score(y_test_t, preds_t, zero_division=0)
            r_t = recall_score(y_test_t, preds_t, zero_division=0)
            try:
                auc_t = roc_auc_score(y_test_t, probs_t)
            except:
                auc_t = 0.0

            temporal_results.append({
                'test_period': test_period,
                'train_periods': train_periods,
                'train_size': len(X_train_t),
                'test_size': len(X_test_t),
                'train_susp': int(y_train_t.sum()),
                'test_susp': int(y_test_t.sum()),
                'f1': f1_t, 'precision': p_t, 'recall': r_t, 'auc': auc_t
            })
            print(f"\n    Train on {train_periods} -> Test on {test_period}:")
            print(f"      Train: {len(X_train_t)} samples ({int(y_train_t.sum())} susp)")
            print(f"      Test:  {len(X_test_t)} samples ({int(y_test_t.sum())} susp)")
            print(f"      F1={f1_t:.4f}, P={p_t:.4f}, R={r_t:.4f}, AUC={auc_t:.4f}")

        # Also check: random split vs temporal split
        print(f"\n  Comparison: Random split vs Temporal split")
        # Random split (same proportions)
        from sklearn.model_selection import train_test_split
        X_rtrain, X_rtest, y_rtrain, y_rtest = train_test_split(
            X_opt, y, test_size=0.3, random_state=42, stratify=y
        )
        clf_r = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        clf_r.fit(X_rtrain, y_rtrain)
        preds_r = clf_r.predict(X_rtest)
        probs_r = clf_r.predict_proba(X_rtest)[:, 1]
        f1_r = f1_score(y_rtest, preds_r, zero_division=0)
        auc_r = roc_auc_score(y_rtest, probs_r)
        print(f"    Random 70/30 split: F1={f1_r:.4f}, AUC={auc_r:.4f}")

        # --- GRAPH 43: Temporal Stability ---
        if len(temporal_results) >= 2:
            fig, axes = plt.subplots(1, 3, figsize=(22, 7))
            fig.suptitle('Temporal Stability Check', fontsize=18, fontweight='bold', color='white')

            # Plot 1: F1 by test period
            ax = axes[0]
            test_periods = [str(r['test_period']) for r in temporal_results]
            f1s_temp = [r['f1'] for r in temporal_results]
            ax.bar(range(len(test_periods)), f1s_temp, color=COLORS['blue'], alpha=0.8)
            ax.axhline(y=f1_r, color=COLORS['warn'], linestyle='--', linewidth=2, label=f'Random split: {f1_r:.3f}')
            ax.set_xticks(range(len(test_periods)))
            ax.set_xticklabels([f'Test: {p}' for p in test_periods], fontsize=10, rotation=15)
            ax.set_ylabel('F1 Score', fontsize=13)
            ax.set_title('F1 by Test Period (Temporal Split)', fontsize=14, color='white')
            ax.legend(fontsize=10)
            ax.set_facecolor(COLORS['bg'])

            # Plot 2: Precision vs Recall by period
            ax = axes[1]
            precs_temp = [r['precision'] for r in temporal_results]
            recs_temp = [r['recall'] for r in temporal_results]
            x_pos = np.arange(len(test_periods))
            ax.bar(x_pos - 0.2, precs_temp, 0.4, label='Precision', color=COLORS['legit'], alpha=0.8)
            ax.bar(x_pos + 0.2, recs_temp, 0.4, label='Recall', color=COLORS['susp'], alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'Test: {p}' for p in test_periods], fontsize=10, rotation=15)
            ax.set_ylabel('Score', fontsize=13)
            ax.set_title('Precision vs Recall by Period', fontsize=14, color='white')
            ax.legend(fontsize=10)
            ax.set_facecolor(COLORS['bg'])

            # Plot 3: Fraud rate by period
            ax = axes[2]
            period_rates = []
            period_labels = []
            for p in periods:
                mask_p = df['F2230'] == p
                rate = (mask_p & susp_mask).sum() / mask_p.sum() * 100
                period_rates.append(rate)
                period_labels.append(str(p))
            ax.bar(range(len(period_labels)), period_rates, color=COLORS['accent'], alpha=0.8)
            ax.set_xticks(range(len(period_labels)))
            ax.set_xticklabels([f'Period {p}' for p in period_labels], fontsize=10, rotation=15)
            ax.set_ylabel('Suspicious Rate (%)', fontsize=13)
            ax.set_title('Fraud Rate by Reporting Period', fontsize=14, color='white')
            ax.set_facecolor(COLORS['bg'])

            plt.tight_layout()
            plt.savefig(f'{OUT}/43_temporal_stability.png', dpi=150, bbox_inches='tight',
                        facecolor='#0a0a23', edgecolor='none')
            plt.close()
            print(f"  [OK] {OUT}/43_temporal_stability.png")
        else:
            print("  [SKIP] Not enough temporal periods for graph")
    else:
        print("  [SKIP] Only 1 reporting period found")
else:
    print("  [SKIP] F2230 not found in dataset")


# ======================================================================
# FINAL SUMMARY
# ======================================================================
elapsed = time.time() - start_time
print("\n" + "=" * 70)
print("PHASE 5 COMPLETE")
print("=" * 70)
print(f"Total time: {elapsed:.1f}s")
print(f"\nGraphs generated:")
print(f"  36. Feature Engineering Evaluation")
print(f"  37. Forward Feature Selection Curve")
print(f"  38. CV Strategy Comparison")
print(f"  39. SMOTE Variant Comparison")
print(f"  40. Error Analysis / Misclassification")
print(f"  41. Threshold Optimization")
print(f"  42. Model Stacking / Blending")
print(f"  43. Temporal Stability")

print(f"\nKey Results Summary:")
print(f"  Optimal features: {OPTIMAL_FEATURES}")
print(f"  Optimal feature count: {optimal['n_features']}")
print(f"  Best CV F1: {optimal['f1_mean']:.4f}")
print(f"  Best threshold: {best_threshold:.4f}")
best_model_name = m_names[best_model_idx]
print(f"  Best model: {best_model_name} (F1={m_f1s[best_model_idx]:.4f})")

# Save summary to file
with open('files/18_phase5_summary.md', 'w', encoding='utf-8') as f:
    f.write("# Phase 5 Summary: Pre-Modeling Research Results\n\n")
    f.write(f"**Runtime**: {elapsed:.1f}s\n\n")
    f.write("## Key Findings\n\n")
    f.write(f"### Optimal Feature Set ({optimal['n_features']} features)\n")
    for feat in OPTIMAL_FEATURES:
        f.write(f"- {feat}\n")
    f.write(f"\n### Best F1 Score: {optimal['f1_mean']:.4f}\n")
    f.write(f"### Best Threshold: {best_threshold:.4f}\n")
    f.write(f"### Best Model: {best_model_name} (F1={m_f1s[best_model_idx]:.4f})\n")
    f.write(f"\n### Model Scores\n\n")
    f.write(f"| Model | F1 | Std |\n")
    f.write(f"|-------|----|-----|\n")
    for m in m_names:
        f.write(f"| {m} | {model_scores[m]['f1']:.4f} | {model_scores[m]['f1_std']:.4f} |\n")

print(f"\n  [OK] files/18_phase5_summary.md (auto-generated)")
print("\nDone!")
