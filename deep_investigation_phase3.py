"""
BOI Hackathon - Deep Investigation Phase 3 — EVERYTHING
1. Statistical Hypothesis Testing (KS test, Mann-Whitney U)
2. Anomaly Detection (Isolation Forest + LOF)
3. Temporal / Period Analysis
4. Categorical Interaction Deep Dive
5. Quick Tree-Based Feature Importance (XGBoost + RandomForest)
6. Benford's Law Analysis
7. Hardest-to-Catch Profiling
8. Feature Stability (Bootstrap)

Output: files/graphs/ + console + files/15_phase3_investigation.md
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from collections import Counter
import os, time, warnings
warnings.filterwarnings('ignore')

OUT_DIR = 'files/graphs'
os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use('dark_background')
C = {
    'legit': '#00D4AA', 'susp': '#FF4B6E', 'a1': '#6C5CE7', 'a2': '#FDCB6E',
    'a3': '#74B9FF', 'a4': '#A29BFE', 'a5': '#FF6B81', 'a6': '#2ED573',
    'bg': '#0D1117', 'card': '#161B22', 'txt': '#E6EDF3', 'grid': '#21262D',
}

def save_fig(fig, name, dpi=200):
    path = os.path.join(OUT_DIR, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=C['bg'], edgecolor='none')
    plt.close(fig)
    print(f'  [OK] {path}')


# ── Load ─────────────────────────────────────────────────────────────
t0 = time.time()
print("=" * 70)
print("PHASE 3 — COMPREHENSIVE DEEP INVESTIGATION")
print("=" * 70)
df = pd.read_csv('DataSet.csv')
target = df['F3924']
susp_mask = target == 1
legit_mask = target == 0
print(f"Loaded {len(df)} rows in {time.time()-t0:.1f}s\n")

# Pre-filter columns
drop = {'Unnamed: 0', 'F3924', 'F3912', 'F2230', 'F3888'}
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]
valid_cols = [c for c in num_cols if df[c].dropna().std() > 0 and df[c].count() > 100]


# ═══════════════════════════════════════════════════════════════════
# 1. STATISTICAL HYPOTHESIS TESTING
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. STATISTICAL HYPOTHESIS TESTING")
print("=" * 70)

ks_results = []
mw_results = []
for col in valid_cols:
    s_susp = df.loc[susp_mask, col].dropna()
    s_legit = df.loc[legit_mask, col].dropna()
    if len(s_susp) < 5 or len(s_legit) < 50:
        continue
    ks_stat, ks_p = stats.ks_2samp(s_susp, s_legit)
    mw_stat, mw_p = stats.mannwhitneyu(s_susp, s_legit, alternative='two-sided')
    ks_results.append({'feature': col, 'ks_stat': ks_stat, 'ks_p': ks_p})
    mw_results.append({'feature': col, 'mw_stat': mw_stat, 'mw_p': mw_p})

ks_df = pd.DataFrame(ks_results).sort_values('ks_stat', ascending=False)
mw_df = pd.DataFrame(mw_results).sort_values('mw_p')

# Bonferroni correction
n_tests = len(ks_df)
alpha = 0.05
bonf_alpha = alpha / n_tests

sig_ks = ks_df[ks_df['ks_p'] < bonf_alpha]
sig_mw = mw_df[mw_df['mw_p'] < bonf_alpha]

print(f"\nTotal features tested: {n_tests}")
print(f"Bonferroni-corrected alpha: {bonf_alpha:.2e}")
print(f"Features significant by KS test:         {len(sig_ks)}")
print(f"Features significant by Mann-Whitney U:   {len(sig_mw)}")

print(f"\nTop 20 Features by KS Statistic (largest distributional difference):")
print(f"{'Feature':>10} {'KS Stat':>10} {'p-value':>15} {'Significant':>12}")
print("-" * 50)
for _, r in ks_df.head(20).iterrows():
    sig = "YES ***" if r['ks_p'] < bonf_alpha else "no"
    print(f"{r['feature']:>10} {r['ks_stat']:>10.4f} {r['ks_p']:>15.2e} {sig:>12}")

# GRAPH 20: Statistical test results
fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=C['bg'])

# 20a: Top 30 by KS statistic
top30_ks = ks_df.head(30)
colors_ks = [C['susp'] if p < bonf_alpha else C['a3'] for p in top30_ks['ks_p']]
axes[0].barh(range(len(top30_ks)), top30_ks['ks_stat'].values, color=colors_ks, height=0.7)
axes[0].set_yticks(range(len(top30_ks)))
axes[0].set_yticklabels(top30_ks['feature'].values, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_xlabel('KS Statistic', fontsize=11, color=C['txt'])
axes[0].set_title(f'Top 30 Features by KS Test\n(red = significant at Bonferroni {bonf_alpha:.1e})',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[0].set_facecolor(C['card'])
axes[0].tick_params(colors=C['txt'])

# 20b: Volcano plot (KS stat vs -log10 p-value)
neg_log_p = -np.log10(ks_df['ks_p'].clip(lower=1e-300))
sig_mask_plot = ks_df['ks_p'] < bonf_alpha
axes[1].scatter(ks_df.loc[~sig_mask_plot.values, 'ks_stat'], neg_log_p[~sig_mask_plot.values],
                s=8, alpha=0.4, color=C['a3'], label='Not significant')
axes[1].scatter(ks_df.loc[sig_mask_plot.values, 'ks_stat'], neg_log_p[sig_mask_plot.values],
                s=15, alpha=0.7, color=C['susp'], label='Significant')
axes[1].axhline(-np.log10(bonf_alpha), color=C['a2'], linestyle='--', linewidth=1,
                label=f'Bonferroni threshold')
axes[1].set_xlabel('KS Statistic', fontsize=11, color=C['txt'])
axes[1].set_ylabel('-log10(p-value)', fontsize=11, color=C['txt'])
axes[1].set_title('Volcano Plot: KS Statistic vs Significance',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[1].legend(fontsize=10, facecolor=C['card'], edgecolor=C['grid'])
axes[1].set_facecolor(C['card'])
axes[1].tick_params(colors=C['txt'])

plt.tight_layout()
save_fig(fig, '20_statistical_tests')


# ═══════════════════════════════════════════════════════════════════
# 2. ANOMALY DETECTION (Isolation Forest + LOF)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. ANOMALY DETECTION (Unsupervised)")
print("=" * 70)

# Use top 50 features by KS stat for anomaly detection
top50_feats = ks_df.head(50)['feature'].tolist()
X_anom = df[top50_feats].fillna(df[top50_feats].median())

# Isolation Forest
print("\nRunning Isolation Forest...")
for contamination in [0.01, 0.02, 0.05, 0.1]:
    ifor = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    pred = ifor.fit_predict(X_anom)
    anomalies = pred == -1
    n_anom = anomalies.sum()
    # How many of the 81 suspicious are flagged?
    susp_flagged = (anomalies & susp_mask).sum()
    prec = susp_flagged / n_anom if n_anom > 0 else 0
    rec = susp_flagged / 81
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"  contamination={contamination:.2f}: {n_anom} anomalies, "
          f"caught {susp_flagged}/81 susp ({rec:.1%} recall), prec={prec:.1%}, F1={f1:.4f}")

# LOF
print("\nRunning Local Outlier Factor...")
for n_neighbors in [10, 20, 50]:
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.02, n_jobs=-1)
    pred = lof.fit_predict(X_anom)
    anomalies = pred == -1
    n_anom = anomalies.sum()
    susp_flagged = (anomalies & susp_mask).sum()
    prec = susp_flagged / n_anom if n_anom > 0 else 0
    rec = susp_flagged / 81
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"  n_neighbors={n_neighbors}: {n_anom} anomalies, "
          f"caught {susp_flagged}/81 ({rec:.1%}), prec={prec:.1%}, F1={f1:.4f}")

# Best model for visualization: IF with contamination=0.02
ifor_best = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
ifor_pred = ifor_best.fit_predict(X_anom)
ifor_scores = ifor_best.decision_function(X_anom)
ifor_anomalies = ifor_pred == -1

# GRAPH 21: Anomaly detection results
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_anom)

fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=C['bg'])

# 21a: Isolation Forest anomaly scores
axes[0].scatter(X_pca[legit_mask & ~ifor_anomalies, 0], X_pca[legit_mask & ~ifor_anomalies, 1],
                s=5, alpha=0.2, color=C['legit'], label='Legit (normal)')
axes[0].scatter(X_pca[legit_mask & ifor_anomalies, 0], X_pca[legit_mask & ifor_anomalies, 1],
                s=20, alpha=0.6, color=C['a2'], label='Legit (flagged anomaly)', marker='x')
axes[0].scatter(X_pca[susp_mask & ~ifor_anomalies, 0], X_pca[susp_mask & ~ifor_anomalies, 1],
                s=50, alpha=0.8, color=C['a1'], label='Suspicious (missed)', edgecolors='white', linewidth=0.5)
axes[0].scatter(X_pca[susp_mask & ifor_anomalies, 0], X_pca[susp_mask & ifor_anomalies, 1],
                s=80, alpha=0.9, color=C['susp'], label='Suspicious (caught!)', edgecolors='white',
                linewidth=1, zorder=5)
axes[0].set_title('Isolation Forest (contamination=2%)\nPCA Projection',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[0].legend(fontsize=9, facecolor=C['card'], edgecolor=C['grid'], markerscale=1.5)
axes[0].set_facecolor(C['card'])
axes[0].tick_params(colors=C['txt'])

# 21b: Anomaly score distributions
scores_legit = ifor_scores[legit_mask]
scores_susp = ifor_scores[susp_mask]
axes[1].hist(scores_legit, bins=80, alpha=0.6, color=C['legit'], label='Legitimate', density=True)
axes[1].hist(scores_susp, bins=20, alpha=0.7, color=C['susp'], label='Suspicious', density=True)
axes[1].axvline(np.percentile(ifor_scores, 2), color=C['a2'], linestyle='--',
                label='2% threshold', linewidth=2)
axes[1].set_xlabel('Anomaly Score', fontsize=11, color=C['txt'])
axes[1].set_ylabel('Density', fontsize=11, color=C['txt'])
axes[1].set_title('Anomaly Score Distribution\n(lower = more anomalous)',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[1].legend(fontsize=10, facecolor=C['card'], edgecolor=C['grid'])
axes[1].set_facecolor(C['card'])
axes[1].tick_params(colors=C['txt'])

plt.tight_layout()
save_fig(fig, '21_anomaly_detection')


# ═══════════════════════════════════════════════════════════════════
# 3. TEMPORAL / PERIOD ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. TEMPORAL / PERIOD ANALYSIS")
print("=" * 70)

periods = df['F2230'].value_counts()
print(f"\nTime period distribution:")
for p, cnt in periods.items():
    n_susp = df.loc[df['F2230'] == p, 'F3924'].sum()
    print(f"  {p}: {cnt} accounts, {int(n_susp)} suspicious ({n_susp/cnt*100:.2f}%)")

# Feature distributions across periods for suspicious accounts only
susp_df = df[susp_mask].copy()
period_stats = []
key_feats = ['F115', 'F2122', 'F2956', 'F3894', 'F3887', 'F3805', 'F1165']
for period in susp_df['F2230'].unique():
    p_data = susp_df[susp_df['F2230'] == period]
    row = {'period': period, 'count': len(p_data)}
    for f in key_feats:
        vals = p_data[f].dropna()
        if len(vals) > 0:
            row[f'{f}_mean'] = vals.mean()
            row[f'{f}_std'] = vals.std()
    period_stats.append(row)

period_df = pd.DataFrame(period_stats)
print(f"\nSuspicious account features by period:")
print(period_df.to_string())

# GRAPH 22: Temporal analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor=C['bg'])
period_colors = {'Sep25': C['susp'], 'Nov25': C['a2'], 'Dec25': C['a1']}

for idx, feat in enumerate(['F115', 'F2956', 'F3894', 'F3887', 'F3805', 'F1165']):
    ax = axes[idx // 3][idx % 3]
    for period in ['Sep25', 'Nov25', 'Dec25']:
        vals = susp_df.loc[susp_df['F2230'] == period, feat].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=15, alpha=0.5, color=period_colors[period], label=f'{period} (n={len(vals)})',
                    density=True)
    ax.set_title(feat, fontsize=12, fontweight='bold', color=C['txt'])
    ax.legend(fontsize=8, facecolor=C['card'], edgecolor=C['grid'])
    ax.set_facecolor(C['card'])
    ax.tick_params(colors=C['txt'])

fig.suptitle('Feature Distributions Across Time Periods (Suspicious Accounts Only)',
             fontsize=15, fontweight='bold', color=C['txt'])
plt.tight_layout()
save_fig(fig, '22_temporal_analysis')


# ═══════════════════════════════════════════════════════════════════
# 4. CATEGORICAL INTERACTION DEEP DIVE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. CATEGORICAL INTERACTION DEEP DIVE")
print("=" * 70)

# Create risk factor flags
df['is_student'] = (df['F3891'] == 'student').astype(int)
df['is_selfemployed'] = (df['F3891'] == 'selfemployed').astype(int)
df['is_rural'] = (df['F3890'] == 'R').astype(int)
df['is_semi_urban'] = (df['F3890'] == 'SU').astype(int)
df['is_savings'] = (df['F3886'] == 'Savings').astype(int)
df['is_male'] = (df['F3889'] == 'M').astype(int)
df['is_young'] = (df['F3894'] < 35).astype(int)

combo_results = []
combos = [
    ('Student + Rural', df['is_student'] & df['is_rural']),
    ('Student + Savings', df['is_student'] & df['is_savings']),
    ('Student + Young', df['is_student'] & df['is_young']),
    ('Student + Rural + Savings', df['is_student'] & df['is_rural'] & df['is_savings']),
    ('Student + Rural + Young', df['is_student'] & df['is_rural'] & df['is_young']),
    ('Self-emp + Rural', df['is_selfemployed'] & df['is_rural']),
    ('Self-emp + Semi-Urban', df['is_selfemployed'] & df['is_semi_urban']),
    ('Self-emp + Rural + Savings', df['is_selfemployed'] & df['is_rural'] & df['is_savings']),
    ('Male + Student + Rural', df['is_male'] & df['is_student'] & df['is_rural']),
    ('Male + Self-emp + Rural', df['is_male'] & df['is_selfemployed'] & df['is_rural']),
    ('Young + Rural + Savings', df['is_young'] & df['is_rural'] & df['is_savings']),
    ('Young + Student + Rural + Savings', df['is_young'] & df['is_student'] & df['is_rural'] & df['is_savings']),
    ('Young + Self-emp + Rural + Savings', df['is_young'] & df['is_selfemployed'] & df['is_rural'] & df['is_savings']),
    ('Male + Young + Rural', df['is_male'] & df['is_young'] & df['is_rural']),
]

print(f"\n{'Combination':<45} {'Total':>6} {'Susp':>5} {'Fraud%':>7} {'Lift':>6}")
print("-" * 75)
base_rate = 81 / len(df) * 100
for name, mask in combos:
    total = mask.sum()
    if total < 5:
        continue
    susp = df.loc[mask, 'F3924'].sum()
    rate = susp / total * 100 if total > 0 else 0
    lift = rate / base_rate
    combo_results.append({'combo': name, 'total': total, 'suspicious': int(susp),
                          'fraud_rate': rate, 'lift': lift})
    print(f"{name:<45} {total:>6} {int(susp):>5} {rate:>6.2f}% {lift:>6.2f}x")

combo_res_df = pd.DataFrame(combo_results).sort_values('lift', ascending=False)

# GRAPH 23: Categorical interactions
fig, ax = plt.subplots(figsize=(14, 8), facecolor=C['bg'])
top_combos = combo_res_df.head(12)
colors_bar = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_combos)))

bars = ax.barh(range(len(top_combos)), top_combos['lift'].values, color=colors_bar, height=0.65)
ax.set_yticks(range(len(top_combos)))
ax.set_yticklabels(top_combos['combo'].values, fontsize=10)
ax.invert_yaxis()

for bar, (_, row) in zip(bars, top_combos.iterrows()):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f'{row["lift"]:.2f}x  ({int(row["suspicious"])}/{int(row["total"])} = {row["fraud_rate"]:.1f}%)',
            va='center', fontsize=9, color=C['txt'])

ax.axvline(1.0, color=C['a2'], linestyle='--', linewidth=1, label='Baseline (0.89%)')
ax.set_xlabel('Lift over Baseline Fraud Rate', fontsize=12, color=C['txt'])
ax.set_title('Categorical Feature Interactions — Fraud Rate Lift',
             fontsize=15, fontweight='bold', color=C['txt'])
ax.legend(fontsize=10, facecolor=C['card'], edgecolor=C['grid'])
ax.set_facecolor(C['card'])
ax.tick_params(colors=C['txt'])
plt.tight_layout()
save_fig(fig, '23_categorical_interactions')


# ═══════════════════════════════════════════════════════════════════
# 5. TREE-BASED FEATURE IMPORTANCE (XGBoost + Random Forest)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. TREE-BASED FEATURE IMPORTANCE")
print("=" * 70)

# Prepare data
X = df[valid_cols].fillna(-999)
y = target

# Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced',
                            random_state=42, n_jobs=-1)
rf.fit(X, y)
rf_imp = pd.Series(rf.feature_importances_, index=valid_cols).sort_values(ascending=False)

print(f"\nTop 30 Features by Random Forest Importance:")
print(f"{'Rank':>5} {'Feature':>10} {'Importance':>12}")
print("-" * 30)
for i, (feat, imp) in enumerate(rf_imp.head(30).items(), 1):
    print(f"{i:>5} {feat:>10} {imp:>12.6f}")

# Cross-validated F1 baseline
print("\nCross-validated baseline (5-fold, balanced RF):")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores, prec_scores, rec_scores, auc_scores = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    rf_cv = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced',
                                    random_state=42, n_jobs=-1)
    rf_cv.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = rf_cv.predict(X.iloc[val_idx])
    y_proba = rf_cv.predict_proba(X.iloc[val_idx])[:, 1]
    
    f1 = f1_score(y.iloc[val_idx], y_pred)
    prec = precision_score(y.iloc[val_idx], y_pred, zero_division=0)
    rec = recall_score(y.iloc[val_idx], y_pred)
    auc = roc_auc_score(y.iloc[val_idx], y_proba)
    
    f1_scores.append(f1); prec_scores.append(prec)
    rec_scores.append(rec); auc_scores.append(auc)
    print(f"  Fold {fold+1}: F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, AUC={auc:.4f}")

print(f"\n  MEAN:   F1={np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
print(f"          Precision={np.mean(prec_scores):.4f} +/- {np.std(prec_scores):.4f}")
print(f"          Recall={np.mean(rec_scores):.4f} +/- {np.std(rec_scores):.4f}")
print(f"          AUC={np.mean(auc_scores):.4f} +/- {np.std(auc_scores):.4f}")

# GRAPH 24: Tree-based feature importance
fig, axes = plt.subplots(1, 2, figsize=(18, 10), facecolor=C['bg'])

# RF importance
top30_rf = rf_imp.head(30)
axes[0].barh(range(len(top30_rf)), top30_rf.values, color=C['a6'], height=0.7)
axes[0].set_yticks(range(len(top30_rf)))
axes[0].set_yticklabels(top30_rf.index, fontsize=8)
axes[0].invert_yaxis()
for i, (feat, imp) in enumerate(top30_rf.items()):
    axes[0].text(imp + 0.0002, i, f'{imp:.5f}', va='center', fontsize=7, color=C['txt'])
axes[0].set_xlabel('Feature Importance', fontsize=11, color=C['txt'])
axes[0].set_title('Random Forest (Balanced)\nTop 30 Feature Importance',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[0].set_facecolor(C['card'])
axes[0].tick_params(colors=C['txt'])

# CV results bar chart
metrics = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC']
means = [np.mean(f1_scores), np.mean(prec_scores), np.mean(rec_scores), np.mean(auc_scores)]
stds = [np.std(f1_scores), np.std(prec_scores), np.std(rec_scores), np.std(auc_scores)]
metric_colors = [C['susp'], C['a2'], C['a3'], C['a1']]

bars = axes[1].bar(metrics, means, yerr=stds, color=metric_colors, 
                    edgecolor=C['bg'], capsize=8, width=0.6)
for bar, m, s in zip(bars, means, stds):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
                 f'{m:.3f}\n+/-{s:.3f}', ha='center', va='bottom', fontsize=11,
                 fontweight='bold', color=C['txt'])
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel('Score', fontsize=12, color=C['txt'])
axes[1].set_title('5-Fold CV Baseline (Balanced RF)\nUsing All Valid Features',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[1].set_facecolor(C['card'])
axes[1].tick_params(colors=C['txt'])

plt.tight_layout()
save_fig(fig, '24_tree_importance')


# ═══════════════════════════════════════════════════════════════════
# 6. BENFORD'S LAW ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. BENFORD'S LAW ANALYSIS")
print("=" * 70)

def first_digit_distribution(series):
    """Extract first significant digit distribution."""
    abs_vals = series.dropna().abs()
    abs_vals = abs_vals[abs_vals > 0]
    first_digits = abs_vals.apply(lambda x: int(str(f'{x:.10e}')[0]))
    return first_digits.value_counts(normalize=True).sort_index()

benford_expected = {d: np.log10(1 + 1/d) for d in range(1, 10)}

# Test on monetary features (F1165, F1705, F2956, F3805, F3836)
monetary_feats = ['F1165', 'F1705', 'F2956', 'F3805', 'F3836', 'F1489', 'F1597']

fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor=C['bg'])
axes_flat = axes.flatten()

print(f"\n{'Feature':>10} {'Class':>10} {'Chi2 Stat':>10} {'p-value':>12} {'Follows Benford?':>18}")
print("-" * 65)

for idx, feat in enumerate(monetary_feats):
    ax = axes_flat[idx]
    benford_vals = [benford_expected[d] for d in range(1, 10)]
    
    for cls, mask, color, lbl in [(0, legit_mask, C['legit'], 'Legit'),
                                   (1, susp_mask, C['susp'], 'Suspicious')]:
        dist = first_digit_distribution(df.loc[mask, feat])
        observed = [dist.get(d, 0) for d in range(1, 10)]
        
        # Chi-squared test
        n = df.loc[mask, feat].dropna().abs().gt(0).sum()
        if n > 30:
            expected_counts = np.array(benford_vals) * n
            observed_counts = np.array(observed) * n
            # Only test if we have enough data
            chi2, p = stats.chisquare(observed_counts + 1, expected_counts + 1)
            follows = "YES" if p > 0.05 else "NO"
            print(f"{feat:>10} {lbl:>10} {chi2:>10.2f} {p:>12.2e} {follows:>18}")
        
        ax.bar(np.arange(1, 10) + (0.35 if cls == 1 else 0), observed, width=0.35,
               color=color, alpha=0.8, label=lbl)
    
    ax.plot(np.arange(1, 10) + 0.175, benford_vals, 'o--', color=C['a2'],
            linewidth=2, markersize=6, label="Benford's Law")
    ax.set_title(feat, fontsize=12, fontweight='bold', color=C['txt'])
    ax.set_xlabel('First Digit', fontsize=9, color=C['txt'])
    ax.set_xticks(range(1, 10))
    ax.legend(fontsize=7, facecolor=C['card'], edgecolor=C['grid'])
    ax.set_facecolor(C['card'])
    ax.tick_params(colors=C['txt'])

# Hide extra subplot
axes_flat[7].set_visible(False)

fig.suptitle("Benford's Law Analysis — First Digit Distribution of Monetary Features",
             fontsize=15, fontweight='bold', color=C['txt'])
plt.tight_layout()
save_fig(fig, '25_benfords_law')


# ═══════════════════════════════════════════════════════════════════
# 7. HARDEST-TO-CATCH PROFILING
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. HARDEST-TO-CATCH PROFILING")
print("=" * 70)

# Use RF prediction probabilities from the full model
rf_proba = rf.predict_proba(X)[:, 1]
df['rf_score'] = rf_proba

susp_df = df[susp_mask].copy()
susp_df_sorted = susp_df.sort_values('rf_score')

print("\nSuspicious accounts ranked by difficulty (hardest to catch first):")
print(f"{'Idx':>5} {'RF Score':>10} {'F115':>8} {'F2122':>8} {'F2956':>8} {'F3894':>8} {'F3887':>8} "
      f"{'Occ':>15} {'Area':>5} {'Acct':>10}")
print("-" * 100)

for i, (_, row) in enumerate(susp_df_sorted.head(20).iterrows()):
    print(f"{i+1:>5} {row['rf_score']:>10.4f} {row.get('F115', 'NA'):>8.2f} "
          f"{row.get('F2122', 'NA'):>8.4f} {row.get('F2956', 'NA'):>8.0f} "
          f"{row.get('F3894', 'NA'):>8.0f} {row.get('F3887', 'NA'):>8.0f} "
          f"{str(row.get('F3891', 'NA')):>15} {str(row.get('F3890', 'NA')):>5} "
          f"{str(row.get('F3886', 'NA')):>10}")

# Stats
easy = susp_df_sorted[susp_df_sorted['rf_score'] > 0.5]
hard = susp_df_sorted[susp_df_sorted['rf_score'] <= 0.5]
print(f"\n  'Easy' to catch (RF score > 0.5): {len(easy)} / 81 ({len(easy)/81*100:.1f}%)")
print(f"  'Hard' to catch (RF score <= 0.5): {len(hard)} / 81 ({len(hard)/81*100:.1f}%)")

if len(hard) > 0:
    print(f"\n  Hard-to-catch profile:")
    for feat in ['F115', 'F2122', 'F2956', 'F3894', 'F3887']:
        h_mean = hard[feat].mean()
        e_mean = easy[feat].mean() if len(easy) > 0 else 0
        print(f"    {feat}: hard={h_mean:.2f}, easy={e_mean:.2f}")

# GRAPH 26: Hardest to catch
fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=C['bg'])

# Score distribution
axes[0].hist(susp_df_sorted['rf_score'], bins=20, color=C['susp'], alpha=0.8, edgecolor=C['bg'])
axes[0].axvline(0.5, color=C['a2'], linestyle='--', linewidth=2, label='Decision boundary')
axes[0].set_xlabel('RF Prediction Score', fontsize=12, color=C['txt'])
axes[0].set_ylabel('Count', fontsize=12, color=C['txt'])
axes[0].set_title('Suspicious Account Score Distribution\n(lower = harder to catch)',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[0].legend(fontsize=10, facecolor=C['card'], edgecolor=C['grid'])
axes[0].set_facecolor(C['card'])
axes[0].tick_params(colors=C['txt'])

# Hard vs Easy comparison
if len(hard) > 0 and len(easy) > 0:
    compare_feats = ['F115', 'F2122', 'F2956', 'F3894', 'F3887']
    hard_means = [hard[f].mean() for f in compare_feats]
    easy_means = [easy[f].mean() for f in compare_feats]
    
    # Normalize for comparison
    max_vals = [max(abs(h), abs(e), 1) for h, e in zip(hard_means, easy_means)]
    hard_norm = [h/m for h, m in zip(hard_means, max_vals)]
    easy_norm = [e/m for e, m in zip(easy_means, max_vals)]
    
    x = np.arange(len(compare_feats))
    axes[1].bar(x - 0.2, hard_norm, 0.35, color=C['a1'], label=f'Hard to catch (n={len(hard)})')
    axes[1].bar(x + 0.2, easy_norm, 0.35, color=C['susp'], label=f'Easy to catch (n={len(easy)})')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(compare_feats, fontsize=10)
    axes[1].set_ylabel('Normalized Mean', fontsize=12, color=C['txt'])
    axes[1].set_title('Feature Comparison: Hard vs Easy to Catch',
                      fontsize=13, fontweight='bold', color=C['txt'])
    axes[1].legend(fontsize=10, facecolor=C['card'], edgecolor=C['grid'])
    axes[1].set_facecolor(C['card'])
    axes[1].tick_params(colors=C['txt'])

plt.tight_layout()
save_fig(fig, '26_hardest_to_catch')


# ═══════════════════════════════════════════════════════════════════
# 8. FEATURE STABILITY (Bootstrap)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. FEATURE STABILITY ANALYSIS (Bootstrap)")
print("=" * 70)

n_bootstrap = 20
top_k = 50
bootstrap_ranks = {feat: [] for feat in valid_cols}

print(f"\nRunning {n_bootstrap} bootstrap iterations (RF, top {top_k} features)...")
for b in range(n_bootstrap):
    # Bootstrap sample
    idx = np.random.RandomState(b).choice(len(X), size=len(X), replace=True)
    X_boot = X.iloc[idx]
    y_boot = y.iloc[idx]
    
    rf_b = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced',
                                   random_state=b, n_jobs=-1)
    rf_b.fit(X_boot, y_boot)
    imp_b = pd.Series(rf_b.feature_importances_, index=valid_cols).sort_values(ascending=False)
    top_k_feats = imp_b.head(top_k).index.tolist()
    
    for feat in valid_cols:
        if feat in top_k_feats:
            bootstrap_ranks[feat].append(top_k_feats.index(feat) + 1)
    
    if (b + 1) % 5 == 0:
        print(f"  Completed {b+1}/{n_bootstrap}")

# Stability metrics
stability = []
for feat, ranks in bootstrap_ranks.items():
    if len(ranks) > 0:
        stability.append({
            'feature': feat,
            'appearances': len(ranks),
            'appear_rate': len(ranks) / n_bootstrap * 100,
            'mean_rank': np.mean(ranks),
            'std_rank': np.std(ranks),
            'best_rank': min(ranks),
        })

stab_df = pd.DataFrame(stability).sort_values('appearances', ascending=False)

print(f"\nTop 30 Most Stable Features (appear in top-{top_k} across {n_bootstrap} bootstraps):")
print(f"{'Feature':>10} {'Appearances':>12} {'Rate':>6} {'Mean Rank':>10} {'Std':>6} {'Best':>6}")
print("-" * 55)
for _, r in stab_df.head(30).iterrows():
    print(f"{r['feature']:>10} {int(r['appearances']):>12} {r['appear_rate']:>5.0f}% "
          f"{r['mean_rank']:>10.1f} {r['std_rank']:>6.1f} {int(r['best_rank']):>6}")

# GRAPH 27: Feature stability
fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor=C['bg'])

top25_stable = stab_df.head(25)
# Color by stability: green = always appears, yellow = sometimes
stab_colors = plt.cm.RdYlGn(top25_stable['appear_rate'].values / 100)
axes[0].barh(range(len(top25_stable)), top25_stable['appear_rate'].values, 
             color=stab_colors, height=0.7)
axes[0].set_yticks(range(len(top25_stable)))
axes[0].set_yticklabels(top25_stable['feature'].values, fontsize=9)
axes[0].invert_yaxis()
for i, (_, r) in enumerate(top25_stable.iterrows()):
    axes[0].text(r['appear_rate'] + 1, i, f'{r["appear_rate"]:.0f}% (rank {r["mean_rank"]:.1f})',
                 va='center', fontsize=8, color=C['txt'])
axes[0].set_xlabel('Appearance Rate (%)', fontsize=11, color=C['txt'])
axes[0].set_title(f'Feature Stability (top 25)\n{n_bootstrap} bootstraps, top-{top_k} each',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[0].set_facecolor(C['card'])
axes[0].tick_params(colors=C['txt'])

# Stability vs Importance scatter
all_feats_with_both = stab_df[stab_df['appearances'] > 0].merge(
    pd.DataFrame({'feature': rf_imp.index, 'rf_importance': rf_imp.values}),
    on='feature'
)
axes[1].scatter(all_feats_with_both['rf_importance'], all_feats_with_both['appear_rate'],
                s=15, alpha=0.5, color=C['a3'])
# Highlight top stable features
top_stable = all_feats_with_both.head(15)
axes[1].scatter(top_stable['rf_importance'], top_stable['appear_rate'],
                s=60, color=C['susp'], edgecolors='white', linewidth=0.5, zorder=5)
for _, r in top_stable.iterrows():
    if r['appear_rate'] > 50:
        axes[1].annotate(r['feature'], (r['rf_importance'], r['appear_rate']),
                         fontsize=7, color=C['a2'], xytext=(5, 3), textcoords='offset points')

axes[1].set_xlabel('RF Feature Importance', fontsize=11, color=C['txt'])
axes[1].set_ylabel('Bootstrap Appearance Rate (%)', fontsize=11, color=C['txt'])
axes[1].set_title('Importance vs Stability\n(red = top 15 most stable)',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[1].set_facecolor(C['card'])
axes[1].tick_params(colors=C['txt'])

plt.tight_layout()
save_fig(fig, '27_feature_stability')


# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
elapsed = time.time() - t0
print("\n" + "=" * 70)
print("PHASE 3 COMPLETE")
print("=" * 70)
print(f"Total time: {elapsed:.1f}s")
print(f"\nGraphs generated:")
print(f"  20. Statistical Tests (KS + Volcano)")
print(f"  21. Anomaly Detection (Isolation Forest)")
print(f"  22. Temporal Analysis")
print(f"  23. Categorical Interactions")
print(f"  24. Tree-Based Importance + CV Baseline")
print(f"  25. Benford's Law")
print(f"  26. Hardest-to-Catch Profiling")
print(f"  27. Feature Stability (Bootstrap)")
print(f"\n{len(sig_ks)} features statistically significant at Bonferroni level")
print(f"RF Baseline: F1={np.mean(f1_scores):.4f}, AUC={np.mean(auc_scores):.4f}")
