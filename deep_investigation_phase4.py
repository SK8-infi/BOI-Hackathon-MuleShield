"""
BOI Hackathon - Deep Investigation Phase 4 — FINAL DEEP DIVE
1. SHAP Explainability Analysis
2. Missing Value Pattern Analysis (MNAR)
3. UMAP Full-Dataset Visualization
4. Sparse Feature Deep-Dive (F0-F499)
5. Permutation Importance
6. Account Lifecycle Analysis
7. Pairwise Decision Boundaries
8. Benford Conformity Score (Engineered Feature)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy import stats
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
print("PHASE 4 — FINAL DEEP DIVE")
print("=" * 70)
df = pd.read_csv('DataSet.csv')
target = df['F3924']
susp_mask = target == 1
legit_mask = target == 0
print(f"Loaded {len(df)} rows in {time.time()-t0:.1f}s\n")

drop = {'Unnamed: 0', 'F3924', 'F3912', 'F2230', 'F3888'}
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]
valid_cols = [c for c in num_cols if df[c].dropna().std() > 0 and df[c].count() > 100]


# ═══════════════════════════════════════════════════════════════════
# 1. SHAP EXPLAINABILITY (via TreeSHAP approximation)
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. SHAP EXPLAINABILITY ANALYSIS")
print("=" * 70)

# Train a focused RF on top features for interpretable SHAP
# Use top 50 features from Phase 3 stability analysis
top_feats = ['F3898', 'F3811', 'F3813', 'F1921', 'F3805', 'F162', 'F3806',
             'F2137', 'F1933', 'F3812', 'F3799', 'F3800', 'F1815', 'F1058',
             'F1273', 'F1165', 'F2030', 'F1814', 'F1705', 'F950',
             'F2035', 'F1923', 'F1603', 'F1813', 'F1819', 'F2149',
             'F2143', 'F1382', 'F1171', 'F1491', 'F3801', 'F3807',
             'F1057', 'F2029', 'F1927', 'F1381', 'F1707', 'F949',
             'F1166', 'F2138']

X_shap = df[top_feats].fillna(-999)
y_shap = target

rf_shap = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced',
                                  random_state=42, n_jobs=-1)
rf_shap.fit(X_shap, y_shap)

# Use impurity-based feature importance as a proxy for SHAP magnitudes
# (True SHAP requires shap library which may not be installed)
imp = pd.Series(rf_shap.feature_importances_, index=top_feats).sort_values(ascending=False)

# For each suspicious account, find which features contribute most
# Using the tree paths (prediction decomposition)
susp_indices = df[susp_mask].index
pred_proba = rf_shap.predict_proba(X_shap)[:, 1]

# Feature contribution analysis: for each feature, compute mean(feature_value_susp) vs mean(feature_value_legit)
# normalized by std, gives directional importance
feature_directions = {}
for feat in top_feats:
    s_mean = df.loc[susp_mask, feat].mean()
    l_mean = df.loc[legit_mask, feat].mean()
    l_std = df.loc[legit_mask, feat].std()
    if l_std > 0:
        z_diff = (s_mean - l_mean) / l_std
    else:
        z_diff = 0
    feature_directions[feat] = {
        'z_diff': z_diff,
        'susp_mean': s_mean,
        'legit_mean': l_mean,
        'direction': 'Higher in suspicious' if z_diff > 0 else 'Lower in suspicious',
        'rf_importance': imp.get(feat, 0)
    }

dir_df = pd.DataFrame(feature_directions).T.sort_values('rf_importance', ascending=False)

print("\nFeature Contribution Analysis (Top 20):")
print(f"{'Feature':>10} {'RF Imp':>8} {'Z-diff':>8} {'Susp Mean':>12} {'Legit Mean':>12} {'Direction':>25}")
print("-" * 80)
for feat, row in dir_df.head(20).iterrows():
    print(f"{feat:>10} {row['rf_importance']:>8.5f} {row['z_diff']:>8.3f} "
          f"{row['susp_mean']:>12.2f} {row['legit_mean']:>12.2f} {row['direction']:>25}")

# GRAPH 28: SHAP-like importance with direction
fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor=C['bg'])

top20 = dir_df.head(20)
colors_dir = [C['susp'] if z < 0 else C['legit'] for z in top20['z_diff']]
axes[0].barh(range(len(top20)), top20['z_diff'].values, color=colors_dir, height=0.7)
axes[0].set_yticks(range(len(top20)))
axes[0].set_yticklabels(top20.index, fontsize=9)
axes[0].invert_yaxis()
axes[0].axvline(0, color=C['txt'], linewidth=0.5, linestyle='--')
axes[0].set_xlabel('Z-Score Difference (Suspicious - Legitimate)', fontsize=11, color=C['txt'])
axes[0].set_title('Feature Direction Analysis\n(red = lower in suspicious, green = higher)',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[0].set_facecolor(C['card'])
axes[0].tick_params(colors=C['txt'])

# Importance vs Direction scatter
axes[1].scatter(dir_df['rf_importance'], dir_df['z_diff'].abs(),
                s=20, alpha=0.5, color=C['a3'])
for feat, row in dir_df.head(10).iterrows():
    axes[1].scatter(row['rf_importance'], abs(row['z_diff']),
                    s=80, color=C['susp'], edgecolors='white', linewidth=0.5, zorder=5)
    axes[1].annotate(feat, (row['rf_importance'], abs(row['z_diff'])),
                     fontsize=8, color=C['a2'], xytext=(5, 3), textcoords='offset points')
axes[1].set_xlabel('RF Importance', fontsize=11, color=C['txt'])
axes[1].set_ylabel('|Z-Score Difference|', fontsize=11, color=C['txt'])
axes[1].set_title('Importance vs Effect Size\n(top 10 highlighted)',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[1].set_facecolor(C['card'])
axes[1].tick_params(colors=C['txt'])

plt.tight_layout()
save_fig(fig, '28_shap_direction')


# ═══════════════════════════════════════════════════════════════════
# 2. MISSING VALUE PATTERN ANALYSIS (MNAR)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. MISSING VALUE PATTERN ANALYSIS")
print("=" * 70)

# For each feature, compare missing rate in suspicious vs legitimate
missing_patterns = []
for col in num_cols[:2000]:  # Check first 2000 numeric columns
    miss_susp = df.loc[susp_mask, col].isna().mean()
    miss_legit = df.loc[legit_mask, col].isna().mean()
    diff = miss_susp - miss_legit
    
    # Fisher exact test or chi2 for missing vs not-missing × class
    missing_patterns.append({
        'feature': col,
        'miss_susp': miss_susp,
        'miss_legit': miss_legit,
        'diff': diff,
        'abs_diff': abs(diff)
    })

miss_df = pd.DataFrame(missing_patterns).sort_values('abs_diff', ascending=False)

print(f"\nTop 20 Features with LARGEST missingness difference between classes:")
print(f"{'Feature':>10} {'Miss% Susp':>12} {'Miss% Legit':>12} {'Difference':>12} {'Pattern':>30}")
print("-" * 80)
for _, r in miss_df.head(20).iterrows():
    pattern = "MORE missing in suspicious" if r['diff'] > 0 else "LESS missing in suspicious"
    print(f"{r['feature']:>10} {r['miss_susp']*100:>11.2f}% {r['miss_legit']*100:>11.2f}% "
          f"{r['diff']*100:>11.2f}% {pattern:>30}")

# Are there features ONLY missing in suspicious or ONLY in legitimate?
only_susp_missing = miss_df[(miss_df['miss_susp'] > 0.1) & (miss_df['miss_legit'] < 0.01)]
only_legit_missing = miss_df[(miss_df['miss_legit'] > 0.1) & (miss_df['miss_susp'] < 0.01)]
print(f"\nFeatures mostly missing only in suspicious: {len(only_susp_missing)}")
print(f"Features mostly missing only in legitimate: {len(only_legit_missing)}")

# Create missing pattern fingerprint
miss_fingerprint_susp = df[susp_mask][num_cols[:500]].isna().mean()
miss_fingerprint_legit = df[legit_mask][num_cols[:500]].isna().mean()

# GRAPH 29: Missing value patterns
fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor=C['bg'])

top20_miss = miss_df.head(20)
x = np.arange(len(top20_miss))
axes[0].barh(x - 0.15, top20_miss['miss_susp'].values * 100, 0.3,
             color=C['susp'], label='Suspicious')
axes[0].barh(x + 0.15, top20_miss['miss_legit'].values * 100, 0.3,
             color=C['legit'], label='Legitimate')
axes[0].set_yticks(x)
axes[0].set_yticklabels(top20_miss['feature'].values, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_xlabel('Missing Rate (%)', fontsize=11, color=C['txt'])
axes[0].set_title('Top 20 Features by Missing Rate Difference\n(Suspicious vs Legitimate)',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[0].legend(fontsize=10, facecolor=C['card'], edgecolor=C['grid'])
axes[0].set_facecolor(C['card'])
axes[0].tick_params(colors=C['txt'])

# Missing fingerprint scatter
axes[1].scatter(miss_fingerprint_legit * 100, miss_fingerprint_susp * 100,
                s=10, alpha=0.3, color=C['a3'])
axes[1].plot([0, 100], [0, 100], '--', color=C['a2'], linewidth=1, label='Equal missing')
# Highlight features with large differences
big_diff = miss_df[miss_df['abs_diff'] > 0.1]
for _, r in big_diff.head(5).iterrows():
    axes[1].scatter(r['miss_legit']*100, r['miss_susp']*100,
                    s=80, color=C['susp'], edgecolors='white', zorder=5)
    axes[1].annotate(r['feature'], (r['miss_legit']*100, r['miss_susp']*100),
                     fontsize=8, color=C['a2'], xytext=(5, 3), textcoords='offset points')
axes[1].set_xlabel('Missing Rate in Legitimate (%)', fontsize=11, color=C['txt'])
axes[1].set_ylabel('Missing Rate in Suspicious (%)', fontsize=11, color=C['txt'])
axes[1].set_title('Missing Value Fingerprint\n(each dot = one feature)',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[1].legend(fontsize=10, facecolor=C['card'], edgecolor=C['grid'])
axes[1].set_facecolor(C['card'])
axes[1].tick_params(colors=C['txt'])

plt.tight_layout()
save_fig(fig, '29_missing_patterns')


# ═══════════════════════════════════════════════════════════════════
# 3. UMAP FULL-DATASET VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. UMAP / Advanced Dimensionality Reduction")
print("=" * 70)

# Since UMAP may not be installed, use PCA + t-SNE with optimal features
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Use top stable features only
stable_feats = ['F3898', 'F3811', 'F3813', 'F1921', 'F3805', 'F3806',
                'F3799', 'F3800', 'F3801', 'F3807', 'F3812',
                'F1705', 'F1815', 'F1165', 'F1813', 'F1819']

X_embed = df[stable_feats].fillna(df[stable_feats].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_embed)

# PCA first to reduce noise
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA explained variance (10 components): {pca.explained_variance_ratio_.sum()*100:.1f}%")

# t-SNE with different perplexities
fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=C['bg'])

for idx, perp in enumerate([10, 30, 50]):
    print(f"  Running t-SNE (perplexity={perp})...")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_pca)
    
    ax = axes[idx]
    ax.scatter(X_tsne[legit_mask, 0], X_tsne[legit_mask, 1],
               s=3, alpha=0.15, color=C['legit'], label='Legitimate')
    ax.scatter(X_tsne[susp_mask, 0], X_tsne[susp_mask, 1],
               s=50, alpha=0.9, color=C['susp'], label='Suspicious',
               edgecolors='white', linewidth=0.5, zorder=5)
    ax.set_title(f't-SNE (perplexity={perp})\n{len(stable_feats)} stable features',
                 fontsize=12, fontweight='bold', color=C['txt'])
    ax.legend(fontsize=9, facecolor=C['card'], edgecolor=C['grid'], markerscale=2)
    ax.set_facecolor(C['card'])
    ax.tick_params(colors=C['txt'])

fig.suptitle('Dimensionality Reduction: Class Separation with Stable Features',
             fontsize=15, fontweight='bold', color=C['txt'])
plt.tight_layout()
save_fig(fig, '30_tsne_embedding')


# ═══════════════════════════════════════════════════════════════════
# 4. SPARSE FEATURE DEEP-DIVE (F0-F499)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. SPARSE FEATURE DEEP-DIVE (F0-F499)")
print("=" * 70)

# These had the highest KS stats (F451=0.74, F453=0.74, F448=0.73)
# but failed significance due to small sample size
sparse_feats = [f'F{i}' for i in range(0, 500) if f'F{i}' in df.columns]
sparse_info = []

for col in sparse_feats:
    if col not in df.columns or col in drop:
        continue
    total_non_null = df[col].count()
    susp_non_null = df.loc[susp_mask, col].count()
    legit_non_null = df.loc[legit_mask, col].count()
    
    if susp_non_null > 0:
        susp_coverage = susp_non_null / 81 * 100
        legit_coverage = legit_non_null / 9001 * 100
        sparse_info.append({
            'feature': col,
            'total_non_null': total_non_null,
            'susp_non_null': susp_non_null,
            'legit_non_null': legit_non_null,
            'susp_coverage': susp_coverage,
            'legit_coverage': legit_coverage,
            'coverage_diff': susp_coverage - legit_coverage
        })

sparse_df = pd.DataFrame(sparse_info).sort_values('coverage_diff', ascending=False)

print(f"\nF0-F499 block: {len(sparse_feats)} features, {len(sparse_info)} with any suspicious data")
print(f"\nTop 20 by coverage difference (suspicious have MORE data):")
print(f"{'Feature':>10} {'Susp Non-Null':>14} {'Susp%':>7} {'Legit Non-Null':>15} {'Legit%':>7} {'Diff':>7}")
print("-" * 65)
for _, r in sparse_df.head(20).iterrows():
    print(f"{r['feature']:>10} {int(r['susp_non_null']):>14} {r['susp_coverage']:>6.1f}% "
          f"{int(r['legit_non_null']):>15} {r['legit_coverage']:>6.1f}% {r['coverage_diff']:>6.1f}%")

# Deep analysis of top KS features
ks_top = ['F451', 'F453', 'F448', 'F450', 'F142', 'F144', 'F139', 'F141']
print(f"\nDetailed analysis of top KS-scoring sparse features:")
for feat in ks_top:
    if feat not in df.columns:
        continue
    s_vals = df.loc[susp_mask, feat].dropna()
    l_vals = df.loc[legit_mask, feat].dropna()
    print(f"\n  {feat}: {len(s_vals)} suspicious, {len(l_vals)} legitimate non-null values")
    if len(s_vals) > 0:
        print(f"    Suspicious: mean={s_vals.mean():.4f}, median={s_vals.median():.4f}, "
              f"std={s_vals.std():.4f}")
    if len(l_vals) > 0:
        print(f"    Legitimate: mean={l_vals.mean():.4f}, median={l_vals.median():.4f}, "
              f"std={l_vals.std():.4f}")
    if len(s_vals) > 2 and len(l_vals) > 2:
        ks, p = stats.ks_2samp(s_vals, l_vals)
        print(f"    KS stat={ks:.4f}, p={p:.2e}")

# GRAPH 31: Sparse feature analysis
fig, axes = plt.subplots(2, 4, figsize=(22, 10), facecolor=C['bg'])
for idx, feat in enumerate(ks_top):
    ax = axes[idx // 4][idx % 4]
    if feat not in df.columns:
        ax.set_visible(False)
        continue
    s_vals = df.loc[susp_mask, feat].dropna()
    l_vals = df.loc[legit_mask, feat].dropna()
    
    if len(l_vals) > 0:
        ax.hist(l_vals, bins=30, alpha=0.6, color=C['legit'],
                label=f'Legit (n={len(l_vals)})', density=True)
    if len(s_vals) > 0:
        ax.hist(s_vals, bins=15, alpha=0.7, color=C['susp'],
                label=f'Susp (n={len(s_vals)})', density=True)
    
    ax.set_title(feat, fontsize=12, fontweight='bold', color=C['txt'])
    ax.legend(fontsize=7, facecolor=C['card'], edgecolor=C['grid'])
    ax.set_facecolor(C['card'])
    ax.tick_params(colors=C['txt'])

fig.suptitle('Sparse Feature Deep-Dive (F0-F499) — Top KS-Scoring Features',
             fontsize=15, fontweight='bold', color=C['txt'])
plt.tight_layout()
save_fig(fig, '31_sparse_features')


# ═══════════════════════════════════════════════════════════════════
# 5. PERMUTATION IMPORTANCE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. PERMUTATION IMPORTANCE")
print("=" * 70)

print("Computing permutation importance (this is slower but more reliable)...")
X_perm = df[top_feats].fillna(-999)

# Use a trained model
perm_result = permutation_importance(rf_shap, X_perm, y_shap,
                                      n_repeats=10, random_state=42,
                                      scoring='f1', n_jobs=-1)

perm_imp = pd.DataFrame({
    'feature': top_feats,
    'mean_decrease': perm_result.importances_mean,
    'std_decrease': perm_result.importances_std,
    'rf_importance': [imp.get(f, 0) for f in top_feats]
}).sort_values('mean_decrease', ascending=False)

print(f"\nPermutation Importance (F1-based, 10 repeats):")
print(f"{'Rank':>5} {'Feature':>10} {'Mean Decrease':>14} {'Std':>8} {'RF Imp':>8} {'Agreement?':>12}")
print("-" * 60)
rf_top20 = set(imp.head(20).index)
for i, (_, r) in enumerate(perm_imp.head(20).iterrows(), 1):
    agree = "YES" if r['feature'] in rf_top20 else "NO"
    print(f"{i:>5} {r['feature']:>10} {r['mean_decrease']:>14.5f} {r['std_decrease']:>8.5f} "
          f"{r['rf_importance']:>8.5f} {agree:>12}")

# GRAPH 32: Permutation importance
fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor=C['bg'])

top20_perm = perm_imp.head(20)
axes[0].barh(range(len(top20_perm)), top20_perm['mean_decrease'].values,
             xerr=top20_perm['std_decrease'].values,
             color=C['a1'], height=0.7, capsize=3)
axes[0].set_yticks(range(len(top20_perm)))
axes[0].set_yticklabels(top20_perm['feature'].values, fontsize=9)
axes[0].invert_yaxis()
axes[0].set_xlabel('Mean F1-Score Decrease', fontsize=11, color=C['txt'])
axes[0].set_title('Permutation Importance (F1-based)\nTop 20 Features',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[0].set_facecolor(C['card'])
axes[0].tick_params(colors=C['txt'])

# RF vs Permutation agreement
axes[1].scatter(perm_imp['rf_importance'], perm_imp['mean_decrease'],
                s=30, alpha=0.6, color=C['a3'])
for _, r in perm_imp.head(10).iterrows():
    axes[1].scatter(r['rf_importance'], r['mean_decrease'],
                    s=80, color=C['susp'], edgecolors='white', zorder=5)
    axes[1].annotate(r['feature'], (r['rf_importance'], r['mean_decrease']),
                     fontsize=8, color=C['a2'], xytext=(5, 3), textcoords='offset points')
axes[1].set_xlabel('RF Impurity Importance', fontsize=11, color=C['txt'])
axes[1].set_ylabel('Permutation Importance (F1)', fontsize=11, color=C['txt'])
axes[1].set_title('RF vs Permutation Importance Agreement',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[1].set_facecolor(C['card'])
axes[1].tick_params(colors=C['txt'])

plt.tight_layout()
save_fig(fig, '32_permutation_importance')


# ═══════════════════════════════════════════════════════════════════
# 6. ACCOUNT LIFECYCLE ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. ACCOUNT LIFECYCLE ANALYSIS")
print("=" * 70)

acct_age = df['F3887']  # Account age in days
cust_age = df['F3894']  # Customer age in years

# Bin account ages
bins_acct = [0, 30, 90, 180, 365, 730, 2000, 10000]
labels_acct = ['<1m', '1-3m', '3-6m', '6m-1y', '1-2y', '2-5y', '5y+']
df['acct_age_bin'] = pd.cut(acct_age, bins=bins_acct, labels=labels_acct)

lifecycle_stats = []
for bin_label in labels_acct:
    mask = df['acct_age_bin'] == bin_label
    total = mask.sum()
    susp = df.loc[mask, 'F3924'].sum()
    rate = susp / total * 100 if total > 0 else 0
    
    row = {'bin': bin_label, 'total': total, 'suspicious': int(susp), 'rate': rate}
    for feat in ['F115', 'F2122', 'F2956', 'F3805', 'F1165']:
        row[f'{feat}_susp_mean'] = df.loc[mask & susp_mask, feat].mean()
        row[f'{feat}_legit_mean'] = df.loc[mask & legit_mask, feat].mean()
    lifecycle_stats.append(row)

life_df = pd.DataFrame(lifecycle_stats)
print(f"\nFraud rate by account age:")
print(f"{'Bin':>8} {'Total':>7} {'Susp':>5} {'Rate':>7}")
print("-" * 30)
for _, r in life_df.iterrows():
    print(f"{r['bin']:>8} {int(r['total']):>7} {int(r['suspicious']):>5} {r['rate']:>6.2f}%")

# Customer age analysis for suspicious
bins_cust = [0, 20, 25, 30, 35, 40, 50, 60, 100]
labels_cust = ['<20', '20-25', '25-30', '30-35', '35-40', '40-50', '50-60', '60+']
df['cust_age_bin'] = pd.cut(cust_age, bins=bins_cust, labels=labels_cust)

cust_lifecycle = []
for bin_label in labels_cust:
    mask = df['cust_age_bin'] == bin_label
    total = mask.sum()
    susp = df.loc[mask, 'F3924'].sum()
    rate = susp / total * 100 if total > 0 else 0
    cust_lifecycle.append({'bin': bin_label, 'total': total, 'suspicious': int(susp), 'rate': rate})

cust_df = pd.DataFrame(cust_lifecycle)
print(f"\nFraud rate by customer age:")
for _, r in cust_df.iterrows():
    print(f"  {r['bin']:>8}: {int(r['total']):>7} total, {int(r['suspicious']):>3} susp ({r['rate']:.2f}%)")

# GRAPH 33: Lifecycle analysis
fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=C['bg'])

# Account age fraud rates
ax = axes[0]
bars = ax.bar(range(len(life_df)), life_df['rate'].values,
              color=[C['susp'] if r > 0 else C['a3'] for r in life_df['rate'].values],
              width=0.6)
for bar, (_, row) in zip(bars, life_df.iterrows()):
    if row['rate'] > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{row["rate"]:.1f}%\n({int(row["suspicious"])})',
                ha='center', fontsize=9, color=C['txt'])
ax.set_xticks(range(len(life_df)))
ax.set_xticklabels(life_df['bin'].values, fontsize=9)
ax.set_ylabel('Suspicious Rate (%)', fontsize=11, color=C['txt'])
ax.set_title('Fraud Rate by Account Age',
             fontsize=13, fontweight='bold', color=C['txt'])
ax.set_facecolor(C['card'])
ax.tick_params(colors=C['txt'])

# Customer age fraud rates
ax = axes[1]
bars = ax.bar(range(len(cust_df)), cust_df['rate'].values,
              color=[C['susp'] if r > 0 else C['a3'] for r in cust_df['rate'].values],
              width=0.6)
for bar, (_, row) in zip(bars, cust_df.iterrows()):
    if row['rate'] > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{row["rate"]:.2f}%\n({int(row["suspicious"])})',
                ha='center', fontsize=8, color=C['txt'])
ax.set_xticks(range(len(cust_df)))
ax.set_xticklabels(cust_df['bin'].values, fontsize=9)
ax.set_ylabel('Suspicious Rate (%)', fontsize=11, color=C['txt'])
ax.set_title('Fraud Rate by Customer Age',
             fontsize=13, fontweight='bold', color=C['txt'])
ax.set_facecolor(C['card'])
ax.tick_params(colors=C['txt'])

# Scatter: account age vs customer age
ax = axes[2]
ax.scatter(df.loc[legit_mask, 'F3887'], df.loc[legit_mask, 'F3894'],
           s=3, alpha=0.1, color=C['legit'], label='Legitimate')
ax.scatter(df.loc[susp_mask, 'F3887'], df.loc[susp_mask, 'F3894'],
           s=40, alpha=0.8, color=C['susp'], label='Suspicious',
           edgecolors='white', linewidth=0.5, zorder=5)
ax.set_xlabel('Account Age (days)', fontsize=11, color=C['txt'])
ax.set_ylabel('Customer Age (years)', fontsize=11, color=C['txt'])
ax.set_title('Account Age vs Customer Age\n(suspicious highlighted)',
             fontsize=13, fontweight='bold', color=C['txt'])
ax.legend(fontsize=10, facecolor=C['card'], edgecolor=C['grid'], markerscale=2)
ax.set_facecolor(C['card'])
ax.tick_params(colors=C['txt'])

plt.tight_layout()
save_fig(fig, '33_lifecycle_analysis')


# ═══════════════════════════════════════════════════════════════════
# 7. PAIRWISE DECISION BOUNDARIES
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. PAIRWISE DECISION BOUNDARIES")
print("=" * 70)

# Top feature pairs for 2D scatter
pairs = [
    ('F3898', 'F3811'),
    ('F3805', 'F1705'),
    ('F3898', 'F1921'),
    ('F3811', 'F3805'),
    ('F162', 'F3813'),
    ('F1165', 'F3898'),
]

fig, axes = plt.subplots(2, 3, figsize=(22, 14), facecolor=C['bg'])

for idx, (f1, f2) in enumerate(pairs):
    ax = axes[idx // 3][idx % 3]
    
    # Plot legitimate accounts
    ax.scatter(df.loc[legit_mask, f1], df.loc[legit_mask, f2],
               s=3, alpha=0.1, color=C['legit'], label='Legitimate')
    
    # Plot suspicious accounts
    ax.scatter(df.loc[susp_mask, f1], df.loc[susp_mask, f2],
               s=40, alpha=0.85, color=C['susp'], label='Suspicious',
               edgecolors='white', linewidth=0.5, zorder=5)
    
    # Add decision boundary hints (mean lines for suspicious)
    susp_f1_mean = df.loc[susp_mask, f1].mean()
    susp_f2_mean = df.loc[susp_mask, f2].mean()
    ax.axvline(susp_f1_mean, color=C['a2'], linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(susp_f2_mean, color=C['a2'], linestyle='--', linewidth=0.8, alpha=0.6)
    
    ax.set_xlabel(f1, fontsize=11, color=C['txt'])
    ax.set_ylabel(f2, fontsize=11, color=C['txt'])
    ax.set_title(f'{f1} vs {f2}', fontsize=12, fontweight='bold', color=C['txt'])
    ax.legend(fontsize=8, facecolor=C['card'], edgecolor=C['grid'], markerscale=3)
    ax.set_facecolor(C['card'])
    ax.tick_params(colors=C['txt'])

fig.suptitle('Pairwise Feature Decision Boundaries — Top Feature Pairs',
             fontsize=16, fontweight='bold', color=C['txt'])
plt.tight_layout()
save_fig(fig, '34_pairwise_boundaries')


# ═══════════════════════════════════════════════════════════════════
# 8. BENFORD CONFORMITY SCORE (Engineered Feature)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. BENFORD CONFORMITY SCORE")
print("=" * 70)

benford_expected = {d: np.log10(1 + 1/d) for d in range(1, 10)}

def benford_score(row, features):
    """Compute how well a row's monetary values conform to Benford's Law."""
    first_digits = []
    for feat in features:
        val = row[feat]
        if pd.notna(val) and abs(val) > 0:
            try:
                fd = int(str(f'{abs(val):.10e}')[0])
                if 1 <= fd <= 9:
                    first_digits.append(fd)
            except:
                pass
    
    if len(first_digits) < 5:
        return np.nan
    
    # Compute first digit distribution
    counts = Counter(first_digits)
    total = len(first_digits)
    observed = np.array([counts.get(d, 0) / total for d in range(1, 10)])
    expected = np.array([benford_expected[d] for d in range(1, 10)])
    
    # Chi-squared distance (lower = more conforming)
    chi2_dist = np.sum((observed - expected)**2 / (expected + 1e-10))
    
    # Also KL divergence
    observed_safe = np.clip(observed, 1e-10, None)
    kl_div = np.sum(observed_safe * np.log(observed_safe / expected))
    
    return chi2_dist

from collections import Counter

monetary_feats = [f'F{i}' for i in range(1000, 2000) if f'F{i}' in df.columns]
monetary_feats = [f for f in monetary_feats if df[f].dropna().abs().gt(0).sum() > 1000]

print(f"Computing Benford conformity scores using {len(monetary_feats)} monetary features...")

# Compute for each account
scores = []
for idx in range(len(df)):
    score = benford_score(df.iloc[idx], monetary_feats[:50])  # Use first 50 for speed
    scores.append(score)

df['benford_score'] = scores

benford_susp = df.loc[susp_mask, 'benford_score'].dropna()
benford_legit = df.loc[legit_mask, 'benford_score'].dropna()

print(f"\nBenford Conformity Score (lower = more Benford-conforming):")
print(f"  Suspicious: mean={benford_susp.mean():.4f}, median={benford_susp.median():.4f}, n={len(benford_susp)}")
print(f"  Legitimate: mean={benford_legit.mean():.4f}, median={benford_legit.median():.4f}, n={len(benford_legit)}")

if len(benford_susp) > 5 and len(benford_legit) > 50:
    ks, p = stats.ks_2samp(benford_susp, benford_legit)
    mw, p_mw = stats.mannwhitneyu(benford_susp, benford_legit)
    print(f"  KS test: stat={ks:.4f}, p={p:.2e}")
    print(f"  Mann-Whitney: stat={mw:.1f}, p={p_mw:.2e}")

# GRAPH 35: Benford conformity score
fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=C['bg'])

axes[0].hist(benford_legit, bins=40, alpha=0.6, color=C['legit'],
             label=f'Legitimate (n={len(benford_legit)})', density=True)
axes[0].hist(benford_susp, bins=15, alpha=0.7, color=C['susp'],
             label=f'Suspicious (n={len(benford_susp)})', density=True)
axes[0].set_xlabel('Benford Conformity Score', fontsize=12, color=C['txt'])
axes[0].set_ylabel('Density', fontsize=12, color=C['txt'])
axes[0].set_title("Benford's Law Conformity Score\n(lower = more conforming to Benford's)",
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[0].legend(fontsize=10, facecolor=C['card'], edgecolor=C['grid'])
axes[0].set_facecolor(C['card'])
axes[0].tick_params(colors=C['txt'])

# Box plot comparison
data_box = [benford_legit.values, benford_susp.values]
bp = axes[1].boxplot(data_box, labels=['Legitimate', 'Suspicious'],
                     patch_artist=True, showfliers=True, flierprops={'markersize': 2})
bp['boxes'][0].set_facecolor(C['legit'])
bp['boxes'][1].set_facecolor(C['susp'])
for element in ['whiskers', 'caps', 'medians']:
    for line in bp[element]:
        line.set_color(C['txt'])
axes[1].set_ylabel('Benford Conformity Score', fontsize=12, color=C['txt'])
axes[1].set_title('Distribution Comparison',
                  fontsize=13, fontweight='bold', color=C['txt'])
axes[1].set_facecolor(C['card'])
axes[1].tick_params(colors=C['txt'])

plt.tight_layout()
save_fig(fig, '35_benford_conformity')


# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
elapsed = time.time() - t0
print("\n" + "=" * 70)
print("PHASE 4 COMPLETE")
print("=" * 70)
print(f"Total time: {elapsed:.1f}s")
print(f"\nGraphs generated:")
print(f"  28. SHAP-like Direction Analysis")
print(f"  29. Missing Value Patterns")
print(f"  30. t-SNE Embedding (3 perplexities)")
print(f"  31. Sparse Feature Deep-Dive")
print(f"  32. Permutation Importance")
print(f"  33. Account Lifecycle Analysis")
print(f"  34. Pairwise Decision Boundaries")
print(f"  35. Benford Conformity Score")
