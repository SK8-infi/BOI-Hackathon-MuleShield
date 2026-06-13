"""
BOI Hackathon - Deep Investigation Phase 2
- Mutual Information (non-linear feature importance)
- Feature redundancy / near-duplicate detection
- Feature interaction analysis
- Rule-based pattern discovery
- Suspicious account clustering
- High-value feature combinations

Output: files/graphs/ (additional graphs) + console analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import time
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = 'files/graphs'
os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use('dark_background')
COLORS = {
    'legit': '#00D4AA', 'suspicious': '#FF4B6E', 'accent1': '#6C5CE7',
    'accent2': '#FDCB6E', 'accent3': '#74B9FF', 'accent4': '#A29BFE',
    'bg': '#0D1117', 'card': '#161B22', 'text': '#E6EDF3', 'grid': '#21262D',
}

def save_fig(fig, name, dpi=200):
    path = os.path.join(OUT_DIR, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=COLORS['bg'], edgecolor='none')
    plt.close(fig)
    print(f'  [OK] Saved: {path}')


# ── Load Data ────────────────────────────────────────────────────────
start = time.time()
print("=" * 60)
print("DEEP INVESTIGATION - PHASE 2")
print("=" * 60)
print("\nLoading dataset...")
df = pd.read_csv('DataSet.csv')
target = df['F3924']
print(f"Loaded in {time.time()-start:.1f}s\n")


# ==================================================================
# 1. MUTUAL INFORMATION (Non-Linear Feature Importance)
# ==================================================================
print("=" * 60)
print("1. MUTUAL INFORMATION ANALYSIS")
print("=" * 60)

# Select numeric features, exclude leakage
drop_cols = ['Unnamed: 0', 'F3924', 'F3912', 'F2230', 'F3888']
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in drop_cols]

# Remove constant/all-null
valid_cols = []
for c in numeric_cols:
    s = df[c].dropna()
    if len(s) > 10 and s.std() > 0:
        valid_cols.append(c)

print(f"Computing MI for {len(valid_cols)} valid numeric features...")
# Fill NaN with median for MI computation
X_mi = df[valid_cols].fillna(df[valid_cols].median())
mi_start = time.time()
mi_scores = mutual_info_classif(X_mi, target, random_state=42, n_neighbors=5)
mi_series = pd.Series(mi_scores, index=valid_cols).sort_values(ascending=False)
print(f"MI computed in {time.time()-mi_start:.1f}s")

# Top 30 by MI
top30_mi = mi_series.head(30)
print("\nTop 30 Features by Mutual Information:")
print("-" * 50)
for i, (feat, score) in enumerate(top30_mi.items(), 1):
    print(f"  {i:2d}. {feat:8s}  MI = {score:.4f}")

# Compare MI vs Correlation
correlations = {}
for col in valid_cols:
    correlations[col] = abs(df[col].corr(target))
corr_series = pd.Series(correlations).sort_values(ascending=False)

# GRAPH 15: MI vs Correlation comparison
print("\nGenerating MI vs Correlation comparison graph...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=COLORS['bg'])

# 15a: Top 20 by MI
top20_mi = mi_series.head(20)
axes[0].barh(top20_mi.index[::-1], top20_mi.values[::-1], color=COLORS['accent1'],
             edgecolor=COLORS['bg'], height=0.7)
for bar_container in axes[0].containers:
    for bar in bar_container:
        axes[0].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                     f'{bar.get_width():.4f}', va='center', fontsize=9, color=COLORS['text'])
axes[0].set_title('Top 20 by Mutual Information\n(captures non-linear relationships)',
                   fontsize=13, fontweight='bold', color=COLORS['text'])
axes[0].set_xlabel('MI Score', fontsize=11, color=COLORS['text'])
axes[0].set_facecolor(COLORS['card'])
axes[0].tick_params(colors=COLORS['text'])

# 15b: Scatter MI vs Correlation
mi_vals = [mi_series.get(f, 0) for f in valid_cols]
corr_vals = [corr_series.get(f, 0) for f in valid_cols]
axes[1].scatter(corr_vals, mi_vals, alpha=0.3, s=10, color=COLORS['accent3'])

# Highlight top MI features
for feat in top20_mi.index:
    axes[1].scatter(corr_series.get(feat, 0), mi_series.get(feat, 0), 
                    s=50, color=COLORS['suspicious'], zorder=5)
    if mi_series[feat] > 0.02:
        axes[1].annotate(feat, (corr_series.get(feat, 0), mi_series.get(feat, 0)),
                         fontsize=7, color=COLORS['accent2'], xytext=(5, 3),
                         textcoords='offset points')

axes[1].set_xlabel('|Correlation| with Target', fontsize=11, color=COLORS['text'])
axes[1].set_ylabel('Mutual Information with Target', fontsize=11, color=COLORS['text'])
axes[1].set_title('MI vs Correlation\n(red = top 20 MI features)', fontsize=13, fontweight='bold', color=COLORS['text'])
axes[1].set_facecolor(COLORS['card'])
axes[1].tick_params(colors=COLORS['text'])

plt.tight_layout()
save_fig(fig, '15_mutual_information')


# ==================================================================
# 2. FEATURE REDUNDANCY ANALYSIS
# ==================================================================
print("\n" + "=" * 60)
print("2. FEATURE REDUNDANCY ANALYSIS")
print("=" * 60)

# Take top 100 MI features for redundancy check
top100 = mi_series.head(100).index.tolist()
print(f"Computing pairwise correlations for top {len(top100)} MI features...")

corr_top100 = df[top100].corr()

# Find highly correlated pairs (|r| > 0.95)
high_corr_pairs = []
for i in range(len(top100)):
    for j in range(i+1, len(top100)):
        r = abs(corr_top100.iloc[i, j])
        if r > 0.95:
            high_corr_pairs.append((top100[i], top100[j], r))

high_corr_pairs.sort(key=lambda x: -x[2])
print(f"\nHighly correlated feature pairs (|r| > 0.95) in top 100 MI features:")
print("-" * 60)
for f1, f2, r in high_corr_pairs[:20]:
    mi1 = mi_series.get(f1, 0)
    mi2 = mi_series.get(f2, 0)
    keep = f1 if mi1 >= mi2 else f2
    drop = f2 if mi1 >= mi2 else f1
    print(f"  {f1:8s} <-> {f2:8s}  r={r:.4f}  | Keep {keep}, drop {drop}")

print(f"\nTotal near-duplicate pairs: {len(high_corr_pairs)}")
redundant_to_drop = set()
for f1, f2, r in high_corr_pairs:
    mi1 = mi_series.get(f1, 0)
    mi2 = mi_series.get(f2, 0)
    redundant_to_drop.add(f2 if mi1 >= mi2 else f1)
print(f"Features to drop (lower MI in each pair): {len(redundant_to_drop)}")

# GRAPH 16: Top 100 feature correlation clustered heatmap
print("\nGenerating clustered correlation heatmap for top 50 MI features...")
top50 = mi_series.head(50).index.tolist()
corr_top50 = df[top50].corr()

fig, ax = plt.subplots(figsize=(16, 14), facecolor=COLORS['bg'])
# Cluster using hierarchical
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
dist_matrix = 1 - corr_top50.abs()
np.fill_diagonal(dist_matrix.values, 0)

# Make sure it's symmetric and non-negative
dist_matrix = dist_matrix.clip(lower=0)
condensed = squareform(dist_matrix)
linkage_matrix = linkage(condensed, method='ward')
order = leaves_list(linkage_matrix)

reordered = corr_top50.iloc[order, order]
sns.heatmap(reordered, cmap='RdBu_r', center=0, ax=ax, 
            linewidths=0.1, linecolor=COLORS['bg'],
            xticklabels=True, yticklabels=True,
            cbar_kws={'shrink': 0.7, 'label': 'Correlation'})
ax.set_title('Clustered Correlation Heatmap - Top 50 MI Features\n(reveals feature groups)',
             fontsize=15, fontweight='bold', color=COLORS['text'])
ax.tick_params(colors=COLORS['text'], labelsize=7)
plt.tight_layout()
save_fig(fig, '16_clustered_correlations')


# ==================================================================
# 3. RULE-BASED PATTERN DISCOVERY
# ==================================================================
print("\n" + "=" * 60)
print("3. RULE-BASED PATTERN DISCOVERY")
print("=" * 60)

# Find simple threshold rules that have high precision
print("\nSearching for high-precision single-feature rules...")
print("-" * 80)

rules = []
# Use top 50 MI features
for feat in top50:
    s = df[feat].dropna()
    if len(s) < 100:
        continue
    
    # Try various percentile thresholds
    for pct in [90, 95, 99, 1, 5, 10]:
        threshold = s.quantile(pct / 100)
        if pct >= 50:
            flagged = df[df[feat] >= threshold]
        else:
            flagged = df[df[feat] <= threshold]
        
        if len(flagged) < 5 or len(flagged) > 1000:
            continue
        
        n_susp = flagged['F3924'].sum()
        precision = n_susp / len(flagged) if len(flagged) > 0 else 0
        recall = n_susp / 81 if 81 > 0 else 0
        
        if precision > 0.03 and recall > 0.05:  # At least 3% precision and 5% recall
            direction = '>=' if pct >= 50 else '<='
            rules.append({
                'feature': feat, 'direction': direction, 'threshold': threshold,
                'percentile': pct, 'flagged': len(flagged), 'caught': int(n_susp),
                'precision': precision, 'recall': recall,
                'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            })

rules_df = pd.DataFrame(rules).sort_values('f1', ascending=False).drop_duplicates(subset=['feature'], keep='first')
print(f"\nTop 25 Single-Feature Rules (by F1-score):")
print(f"{'Feature':>10} {'Rule':>20} {'Flagged':>8} {'Caught':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
print("-" * 80)
for _, row in rules_df.head(25).iterrows():
    rule_str = f"{row['direction']} {row['threshold']:.4f}"
    print(f"{row['feature']:>10} {rule_str:>20} {int(row['flagged']):>8} {int(row['caught']):>7} "
          f"{row['precision']:>7.2%} {row['recall']:>7.2%} {row['f1']:>7.4f}")


# Multi-feature rule combinations
print("\n\nSearching for high-precision COMBINATION rules...")
print("-" * 80)

combo_rules = []
top_rule_features = rules_df.head(10)['feature'].tolist()

for i in range(len(top_rule_features)):
    for j in range(i+1, len(top_rule_features)):
        f1, f2 = top_rule_features[i], top_rule_features[j]
        r1 = rules_df[rules_df['feature'] == f1].iloc[0]
        r2 = rules_df[rules_df['feature'] == f2].iloc[0]
        
        # Apply both rules
        if r1['direction'] == '>=':
            mask1 = df[f1] >= r1['threshold']
        else:
            mask1 = df[f1] <= r1['threshold']
        if r2['direction'] == '>=':
            mask2 = df[f2] >= r2['threshold']
        else:
            mask2 = df[f2] <= r2['threshold']
        
        # AND combination
        flagged_and = df[mask1 & mask2]
        if len(flagged_and) >= 3:
            n_susp = flagged_and['F3924'].sum()
            prec = n_susp / len(flagged_and)
            rec = n_susp / 81
            f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            combo_rules.append({
                'rule': f"{f1} {r1['direction']} {r1['threshold']:.3f} AND {f2} {r2['direction']} {r2['threshold']:.3f}",
                'flagged': len(flagged_and), 'caught': int(n_susp),
                'precision': prec, 'recall': rec, 'f1': f1_score
            })
        
        # OR combination
        flagged_or = df[mask1 | mask2]
        if len(flagged_or) >= 3:
            n_susp = flagged_or['F3924'].sum()
            prec = n_susp / len(flagged_or)
            rec = n_susp / 81
            f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if prec > 0.02:
                combo_rules.append({
                    'rule': f"{f1} {r1['direction']} {r1['threshold']:.3f} OR {f2} {r2['direction']} {r2['threshold']:.3f}",
                    'flagged': len(flagged_or), 'caught': int(n_susp),
                    'precision': prec, 'recall': rec, 'f1': f1_score
                })

combo_df = pd.DataFrame(combo_rules).sort_values('f1', ascending=False)
print(f"\nTop 15 Combination Rules (by F1-score):")
print(f"{'Rule':>70} {'Flagged':>8} {'Caught':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
print("-" * 110)
for _, row in combo_df.head(15).iterrows():
    print(f"{row['rule']:>70} {int(row['flagged']):>8} {int(row['caught']):>7} "
          f"{row['precision']:>7.2%} {row['recall']:>7.2%} {row['f1']:>7.4f}")


# ==================================================================
# 4. SUSPICIOUS ACCOUNT CLUSTERING
# ==================================================================
print("\n" + "=" * 60)
print("4. SUSPICIOUS ACCOUNT CLUSTERING (Are there sub-types?)")
print("=" * 60)

# Use top 20 MI features for clustering
top20_feats = mi_series.head(20).index.tolist()
df_susp = df[df['F3924'] == 1].copy()

# Fill missing with median from full dataset
X_cluster = df_susp[top20_feats].fillna(df[top20_feats].median())
X_cluster_norm = (X_cluster - X_cluster.mean()) / (X_cluster.std() + 1e-8)

# Try K=2,3,4 clusters
from sklearn.metrics import silhouette_score

print("\nClustering 81 suspicious accounts using top 20 MI features...")
for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster_norm)
    sil = silhouette_score(X_cluster_norm, labels)
    print(f"  K={k}: Silhouette={sil:.3f}")
    for c in range(k):
        mask = labels == c
        n = mask.sum()
        print(f"    Cluster {c}: {n} accounts")

# Use k=3 for detailed analysis
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
df_susp['cluster'] = km3.fit_predict(X_cluster_norm)

print("\nCluster profiles (k=3):")
print("-" * 80)
cat_feats = ['F3891', 'F3890', 'F3886']
for c in range(3):
    cluster = df_susp[df_susp['cluster'] == c]
    print(f"\n  CLUSTER {c} ({len(cluster)} accounts):")
    for cat in cat_feats:
        top = cluster[cat].value_counts().head(2)
        print(f"    {cat}: {dict(top)}")
    # Numeric summaries for key features
    for nf in ['F115', 'F2122', 'F2956', 'F3894', 'F3887']:
        if nf in cluster.columns:
            vals = cluster[nf].dropna()
            if len(vals) > 0:
                print(f"    {nf}: mean={vals.mean():.2f}, median={vals.median():.2f}")

# GRAPH 17: PCA + t-SNE visualization of suspicious clusters
print("\nGenerating cluster visualization...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=COLORS['bg'])

# Include both legit and suspicious for context
X_all = df[top20_feats].fillna(df[top20_feats].median())
X_all_norm = (X_all - X_all.mean()) / (X_all.std() + 1e-8)

# PCA
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(X_all_norm)

# Plot legit (small dots) and suspicious (colored by cluster)
legit_mask = df['F3924'] == 0
axes[0].scatter(pca_result[legit_mask, 0], pca_result[legit_mask, 1], 
                s=3, alpha=0.15, color=COLORS['legit'], label='Legitimate')

cluster_colors = [COLORS['suspicious'], COLORS['accent2'], COLORS['accent1']]
for c in range(3):
    c_mask = (df['F3924'] == 1) & (df.index.isin(df_susp[df_susp['cluster'] == c].index))
    axes[0].scatter(pca_result[c_mask, 0], pca_result[c_mask, 1],
                    s=80, alpha=0.9, color=cluster_colors[c], edgecolors='white',
                    linewidth=0.5, label=f'Suspicious C{c} (n={c_mask.sum()})', zorder=5)

axes[0].set_title(f'PCA Projection (top 20 MI features)\nExplained var: {pca.explained_variance_ratio_.sum():.1%}',
                  fontsize=13, fontweight='bold', color=COLORS['text'])
axes[0].set_xlabel('PC1', fontsize=11, color=COLORS['text'])
axes[0].set_ylabel('PC2', fontsize=11, color=COLORS['text'])
axes[0].legend(fontsize=9, facecolor=COLORS['card'], edgecolor=COLORS['grid'], markerscale=2)
axes[0].set_facecolor(COLORS['card'])
axes[0].tick_params(colors=COLORS['text'])

# t-SNE (sample legit for speed)
np.random.seed(42)
legit_sample_idx = np.random.choice(df[legit_mask].index, size=500, replace=False)
combined_idx = np.concatenate([legit_sample_idx, df_susp.index.values])
X_tsne_input = X_all_norm.loc[combined_idx]
labels_tsne = df.loc[combined_idx, 'F3924'].values

print("  Running t-SNE (500 legit sample + 81 suspicious)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(X_tsne_input)

n_legit_sample = len(legit_sample_idx)
axes[1].scatter(tsne_result[:n_legit_sample, 0], tsne_result[:n_legit_sample, 1],
                s=10, alpha=0.3, color=COLORS['legit'], label='Legitimate (sample)')

for c in range(3):
    c_idx_in_susp = df_susp[df_susp['cluster'] == c].index
    c_positions = [i for i, idx in enumerate(combined_idx) if idx in c_idx_in_susp]
    axes[1].scatter(tsne_result[c_positions, 0], tsne_result[c_positions, 1],
                    s=80, alpha=0.9, color=cluster_colors[c], edgecolors='white',
                    linewidth=0.5, label=f'Suspicious C{c}', zorder=5)

axes[1].set_title('t-SNE Projection (top 20 MI features)\n500 legit + 81 suspicious',
                  fontsize=13, fontweight='bold', color=COLORS['text'])
axes[1].set_xlabel('t-SNE 1', fontsize=11, color=COLORS['text'])
axes[1].set_ylabel('t-SNE 2', fontsize=11, color=COLORS['text'])
axes[1].legend(fontsize=9, facecolor=COLORS['card'], edgecolor=COLORS['grid'], markerscale=2)
axes[1].set_facecolor(COLORS['card'])
axes[1].tick_params(colors=COLORS['text'])

plt.tight_layout()
save_fig(fig, '17_cluster_visualization')


# ==================================================================
# 5. FEATURE INTERACTION ANALYSIS
# ==================================================================
print("\n" + "=" * 60)
print("5. FEATURE INTERACTION ANALYSIS")
print("=" * 60)

# Create interaction features and check if they beat individual MI
print("\nTesting feature interactions (ratio, product, diff)...")
interaction_results = []
top10_mi = mi_series.head(10).index.tolist()

for i in range(len(top10_mi)):
    for j in range(i+1, len(top10_mi)):
        f1, f2 = top10_mi[i], top10_mi[j]
        s1 = df[f1].fillna(df[f1].median())
        s2 = df[f2].fillna(df[f2].median())
        
        # Product
        prod = s1 * s2
        if prod.std() > 0:
            mi = mutual_info_classif(prod.values.reshape(-1, 1), target, random_state=42)[0]
            interaction_results.append({'interaction': f'{f1} * {f2}', 'mi': mi, 'type': 'product'})
        
        # Ratio (safe division)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = s1 / (s2 + 1e-8)
        ratio = ratio.clip(-1e6, 1e6).fillna(0)
        if ratio.std() > 0:
            mi = mutual_info_classif(ratio.values.reshape(-1, 1), target, random_state=42)[0]
            interaction_results.append({'interaction': f'{f1} / {f2}', 'mi': mi, 'type': 'ratio'})
        
        # Difference
        diff = s1 - s2
        if diff.std() > 0:
            mi = mutual_info_classif(diff.values.reshape(-1, 1), target, random_state=42)[0]
            interaction_results.append({'interaction': f'{f1} - {f2}', 'mi': mi, 'type': 'diff'})

inter_df = pd.DataFrame(interaction_results).sort_values('mi', ascending=False)
print(f"\nTop 20 Feature Interactions (by MI):")
print(f"{'Interaction':>35} {'MI':>8} {'Type':>8}  vs Best Individual")
print("-" * 75)

for _, row in inter_df.head(20).iterrows():
    parts = row['interaction'].split(' ')
    f1_name = parts[0]
    f2_name = parts[2]
    best_indiv = max(mi_series.get(f1_name, 0), mi_series.get(f2_name, 0))
    improvement = ((row['mi'] / best_indiv) - 1) * 100 if best_indiv > 0 else 0
    marker = " <<<" if improvement > 10 else ""
    print(f"{row['interaction']:>35} {row['mi']:>8.4f} {row['type']:>8}  (best indiv={best_indiv:.4f}, "
          f"{'+'if improvement>0 else ''}{improvement:.1f}%){marker}")


# GRAPH 18: Feature Interactions Bar Chart
print("\nGenerating feature interaction graph...")
fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS['bg'])

top15_inter = inter_df.head(15)
type_colors_map = {'product': COLORS['accent1'], 'ratio': COLORS['accent2'], 'diff': COLORS['accent3']}
bar_colors = [type_colors_map[t] for t in top15_inter['type']]

bars = ax.barh(range(len(top15_inter)), top15_inter['mi'].values, color=bar_colors,
               edgecolor=COLORS['bg'], height=0.7)
ax.set_yticks(range(len(top15_inter)))
ax.set_yticklabels(top15_inter['interaction'].values, fontsize=9)
ax.invert_yaxis()

for bar, (_, row) in zip(bars, top15_inter.iterrows()):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{row["mi"]:.4f}', va='center', fontsize=9, color=COLORS['text'])

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS['accent1'], label='Product (A * B)'),
                   Patch(facecolor=COLORS['accent2'], label='Ratio (A / B)'),
                   Patch(facecolor=COLORS['accent3'], label='Difference (A - B)')]
ax.legend(handles=legend_elements, fontsize=10, facecolor=COLORS['card'], edgecolor=COLORS['grid'])

ax.set_xlabel('Mutual Information', fontsize=12, color=COLORS['text'])
ax.set_title('Top 15 Feature Interactions (from top 10 MI features)',
             fontsize=14, fontweight='bold', color=COLORS['text'])
ax.set_facecolor(COLORS['card'])
ax.tick_params(colors=COLORS['text'])
plt.tight_layout()
save_fig(fig, '18_feature_interactions')


# ==================================================================
# 6. ENGINEERED FEATURES ANALYSIS
# ==================================================================
print("\n" + "=" * 60)
print("6. ENGINEERED FEATURES")
print("=" * 60)

# Create domain-knowledge features
eng_features = {}

# a) Transactions per day (activity intensity)
acct_age_days = df['F3887'].fillna(df['F3887'].median())
txn_count = df['F2956'].fillna(0)
eng_features['txn_per_day'] = txn_count / (acct_age_days.clip(lower=1))

# b) Missing values count per row
eng_features['missing_count'] = df.isnull().sum(axis=1)

# c) Risk score x Alert flag interaction
eng_features['risk_x_alert'] = df['F115'].fillna(0) * df['F670'].fillna(0)

# d) Digital ratio inverse (how non-digital)
eng_features['non_digital'] = 1 - df['F2122'].fillna(0)

# e) Age-occupation risk (young + student = risky)
le = LabelEncoder()
occ_encoded = le.fit_transform(df['F3891'].fillna('unknown'))
eng_features['young_factor'] = (100 - df['F3894'].fillna(35).clip(0, 100)) / 100

# f) Balance direction indicator
eng_features['balance_positive'] = (df['F3836'].fillna(0) > 0).astype(float)

print("Engineered features MI scores:")
print("-" * 50)
for name, values in eng_features.items():
    v = values.fillna(0).values.reshape(-1, 1)
    if np.std(v) > 0:
        mi = mutual_info_classif(v, target, random_state=42)[0]
        print(f"  {name:25s}  MI = {mi:.4f}")


# ==================================================================
# 7. COMPREHENSIVE FEATURE RANKING
# ==================================================================
print("\n" + "=" * 60)
print("7. COMPREHENSIVE FEATURE RANKING")
print("=" * 60)

# Merge MI and correlation into a combined rank
combined = pd.DataFrame({
    'mi': mi_series,
    'corr': corr_series.reindex(mi_series.index),
}).dropna()

combined['mi_rank'] = combined['mi'].rank(ascending=False)
combined['corr_rank'] = combined['corr'].rank(ascending=False)
combined['combined_rank'] = (combined['mi_rank'] + combined['corr_rank']) / 2
combined = combined.sort_values('combined_rank')

print(f"\nTop 30 Features by Combined Rank (MI + Correlation):")
print(f"{'Rank':>5} {'Feature':>10} {'MI':>8} {'|Corr|':>8} {'MI Rank':>8} {'Corr Rank':>10} {'Comb Rank':>10}")
print("-" * 70)
for i, (feat, row) in enumerate(combined.head(30).iterrows(), 1):
    print(f"{i:>5} {feat:>10} {row['mi']:>8.4f} {row['corr']:>8.4f} {row['mi_rank']:>8.0f} "
          f"{row['corr_rank']:>10.0f} {row['combined_rank']:>10.1f}")

# GRAPH 19: Combined Feature Ranking
print("\nGenerating combined ranking graph...")
top25_combined = combined.head(25)

fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['bg'])

y = range(len(top25_combined))
ax.barh(y, top25_combined['mi'] / top25_combined['mi'].max(), height=0.35, 
        color=COLORS['accent1'], alpha=0.9, label='MI (normalized)')
ax.barh([yi + 0.35 for yi in y], top25_combined['corr'] / top25_combined['corr'].max(), 
        height=0.35, color=COLORS['accent3'], alpha=0.9, label='|Correlation| (normalized)')

ax.set_yticks([yi + 0.175 for yi in y])
ax.set_yticklabels(top25_combined.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Normalized Score', fontsize=12, color=COLORS['text'])
ax.set_title('Top 25 Features by Combined Rank (MI + Correlation)',
             fontsize=15, fontweight='bold', color=COLORS['text'])
ax.legend(fontsize=11, facecolor=COLORS['card'], edgecolor=COLORS['grid'])
ax.set_facecolor(COLORS['card'])
ax.tick_params(colors=COLORS['text'])
plt.tight_layout()
save_fig(fig, '19_combined_ranking')


# ==================================================================
# SUMMARY
# ==================================================================
elapsed = time.time() - start
print("\n" + "=" * 60)
print("DEEP INVESTIGATION COMPLETE")
print("=" * 60)
print(f"Total time: {elapsed:.1f}s")
print(f"\nNew graphs generated:")
print(f"  15. Mutual Information Analysis")
print(f"  16. Clustered Correlation Heatmap")
print(f"  17. PCA + t-SNE Cluster Visualization")
print(f"  18. Feature Interactions")
print(f"  19. Combined Feature Ranking")
print(f"\nKey new findings to document:")
print(f"  - Top MI features (non-linear importance)")
print(f"  - {len(high_corr_pairs)} near-duplicate feature pairs found")
print(f"  - {len(redundant_to_drop)} features can be dropped for redundancy")
print(f"  - Best single-feature rules identified")
print(f"  - Best feature combinations identified")
print(f"  - 3 clusters found within suspicious accounts")
print(f"  - Feature interaction improvements quantified")
