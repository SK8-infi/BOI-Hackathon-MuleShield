"""
BOI Hackathon — Comprehensive Visualization Suite
Generates all key graphs for the mule account classification research.
Saves to: files/graphs/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────
OUT_DIR = 'files/graphs'
os.makedirs(OUT_DIR, exist_ok=True)

# Dark premium style
plt.style.use('dark_background')
COLORS = {
    'legit': '#00D4AA',     # teal
    'suspicious': '#FF4B6E', # coral red
    'accent1': '#6C5CE7',    # purple
    'accent2': '#FDCB6E',    # gold
    'accent3': '#74B9FF',    # sky blue
    'accent4': '#A29BFE',    # lavender
    'bg': '#0D1117',
    'card': '#161B22',
    'text': '#E6EDF3',
    'grid': '#21262D',
}
PALETTE_CAT = ['#00D4AA', '#FF4B6E', '#6C5CE7', '#FDCB6E', '#74B9FF', 
               '#A29BFE', '#FF7675', '#55EFC4', '#FD79A8', '#0984E3',
               '#E17055', '#00CEC9', '#636E72', '#D63031', '#2D3436',
               '#B2BEC3', '#DFE6E9']

def save_fig(fig, name, dpi=200):
    path = os.path.join(OUT_DIR, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=COLORS['bg'], edgecolor='none')
    plt.close(fig)
    print(f'  [OK] Saved: {path}')

# ── Load Data ───────────────────────────────────────────────────────
start = time.time()
print("Loading dataset...")
df = pd.read_csv('DataSet.csv')
df_legit = df[df['F3924'] == 0]
df_susp = df[df['F3924'] == 1]
print(f"Loaded in {time.time()-start:.1f}s\n")

# ====================================================================
# GRAPH 1: Class Imbalance
# ====================================================================
print("Graph 1: Class Imbalance...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLORS['bg'])

# Pie chart
counts = [len(df_legit), len(df_susp)]
labels = [f'Legitimate\n{counts[0]:,} (99.11%)', f'Suspicious\n{counts[1]:,} (0.89%)']
explode = (0, 0.15)
axes[0].pie(counts, labels=labels, explode=explode, colors=[COLORS['legit'], COLORS['suspicious']],
            autopct='', startangle=90, textprops={'color': COLORS['text'], 'fontsize': 12, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': COLORS['bg'], 'linewidth': 2})
axes[0].set_title('Target Distribution (F3924)', fontsize=16, fontweight='bold', color=COLORS['text'], pad=20)

# Bar chart with log scale
bars = axes[1].bar(['Legitimate (0)', 'Suspicious (1)'], counts,
                    color=[COLORS['legit'], COLORS['suspicious']], edgecolor=COLORS['bg'], linewidth=2,
                    width=0.5)
axes[1].set_yscale('log')
axes[1].set_ylabel('Count (log scale)', fontsize=12, color=COLORS['text'])
axes[1].set_title('111:1 Class Imbalance', fontsize=16, fontweight='bold', color=COLORS['text'])
for bar, count in zip(bars, counts):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height()*1.1,
                f'{count:,}', ha='center', va='bottom', fontsize=14, fontweight='bold', color=COLORS['text'])
axes[1].set_facecolor(COLORS['card'])
axes[1].tick_params(colors=COLORS['text'])
axes[1].spines['bottom'].set_color(COLORS['grid'])
axes[1].spines['left'].set_color(COLORS['grid'])

fig.suptitle('', y=1.02)
plt.tight_layout()
save_fig(fig, '01_class_imbalance')

# ====================================================================
# GRAPH 2: Missing Values Heatmap
# ====================================================================
print("Graph 2: Missing Values Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=COLORS['bg'])

# Column-level histogram
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_pct_nonzero = missing_pct[missing_pct > 0]
axes[0].hist(missing_pct_nonzero, bins=50, color=COLORS['accent1'], edgecolor=COLORS['bg'], alpha=0.9)
axes[0].axvline(x=50, color=COLORS['suspicious'], linestyle='--', linewidth=2, label='50% threshold')
axes[0].set_xlabel('Missing %', fontsize=12, color=COLORS['text'])
axes[0].set_ylabel('Number of Columns', fontsize=12, color=COLORS['text'])
axes[0].set_title('Column-Level Missing Values Distribution', fontsize=14, fontweight='bold', color=COLORS['text'])
axes[0].legend(fontsize=11, facecolor=COLORS['card'], edgecolor=COLORS['grid'])
axes[0].set_facecolor(COLORS['card'])
axes[0].tick_params(colors=COLORS['text'])

# Row-level histogram
row_missing = df.isnull().sum(axis=1)
axes[1].hist(row_missing, bins=50, color=COLORS['accent3'], edgecolor=COLORS['bg'], alpha=0.9)
axes[1].axvline(x=row_missing.mean(), color=COLORS['accent2'], linestyle='--', linewidth=2, 
                label=f'Mean: {row_missing.mean():.0f}')
axes[1].set_xlabel('Missing Features per Row', fontsize=12, color=COLORS['text'])
axes[1].set_ylabel('Number of Rows', fontsize=12, color=COLORS['text'])
axes[1].set_title('Row-Level Missing Values Distribution', fontsize=14, fontweight='bold', color=COLORS['text'])
axes[1].legend(fontsize=11, facecolor=COLORS['card'], edgecolor=COLORS['grid'])
axes[1].set_facecolor(COLORS['card'])
axes[1].tick_params(colors=COLORS['text'])

plt.tight_layout()
save_fig(fig, '02_missing_values')

# ====================================================================
# GRAPH 3: Feature Type Distribution
# ====================================================================
print("Graph 3: Feature Type Distribution...")
type_counts = {
    'Negative\nFloats': 882, 'Large\nFloats': 599, 'Binary\n{0,1}': 527,
    'Proportions\n(0-1)': 478, 'Small\nFloats': 463, 'Count-like\nIntegers': 405,
    'Constant\n(drop)': 359, 'All Null\n(drop)': 63, 'Low-card\nInteger': 26,
    'Categorical': 8, 'Binary\n(other)': 15
}

fig, ax = plt.subplots(figsize=(14, 7), facecolor=COLORS['bg'])
colors_bar = [COLORS['accent1'], COLORS['accent3'], COLORS['legit'], COLORS['accent2'],
              COLORS['accent4'], '#E17055', '#636E72', '#2D3436', '#00CEC9', 
              COLORS['suspicious'], '#B2BEC3']

bars = ax.barh(list(type_counts.keys()), list(type_counts.values()), color=colors_bar,
               edgecolor=COLORS['bg'], linewidth=1.5, height=0.7)
for bar, val in zip(bars, type_counts.values()):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            f'{val}', ha='left', va='center', fontsize=12, fontweight='bold', color=COLORS['text'])

ax.set_xlabel('Number of Features', fontsize=13, color=COLORS['text'])
ax.set_title('Feature Type Distribution (3,923 features)', fontsize=16, fontweight='bold', color=COLORS['text'])
ax.set_facecolor(COLORS['card'])
ax.tick_params(colors=COLORS['text'], labelsize=11)
ax.invert_yaxis()
ax.set_xlim(0, max(type_counts.values()) * 1.15)
plt.tight_layout()
save_fig(fig, '03_feature_types')

# ====================================================================
# GRAPH 4: Categorical Features vs Fraud Rate
# ====================================================================
print("Graph 4: Categorical Features vs Fraud Rate...")
fig, axes = plt.subplots(2, 3, figsize=(20, 14), facecolor=COLORS['bg'])

cat_configs = [
    ('F3891', 'Occupation', axes[0, 0]),
    ('F3890', 'Branch Area', axes[0, 1]),
    ('F3893', 'Customer Segment', axes[0, 2]),
    ('F3886', 'Account Type (Top 6)', axes[1, 0]),
    ('F3892', 'Gender', axes[1, 1]),
    ('F3889', 'Account Scheme', axes[1, 2]),
]

for feat, title, ax in cat_configs:
    grouped = df.groupby(feat)['F3924'].agg(['sum', 'count']).reset_index()
    grouped['rate'] = grouped['sum'] / grouped['count'] * 100
    grouped = grouped.sort_values('rate', ascending=True)
    if feat == 'F3886':
        grouped = grouped.nlargest(6, 'count').sort_values('rate', ascending=True)
    
    bars = ax.barh(grouped[feat].astype(str), grouped['rate'], 
                   color=COLORS['accent3'], edgecolor=COLORS['bg'], height=0.6)
    
    for bar, (_, row) in zip(bars, grouped.iterrows()):
        label = f"{row['rate']:.2f}% ({int(row['sum'])}/{int(row['count'])})"
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                label, ha='left', va='center', fontsize=10, color=COLORS['text'])
    
    ax.set_title(f'{title} ({feat})', fontsize=13, fontweight='bold', color=COLORS['text'])
    ax.set_xlabel('Suspicious Rate (%)', fontsize=11, color=COLORS['text'])
    ax.set_facecolor(COLORS['card'])
    ax.tick_params(colors=COLORS['text'])
    ax.set_xlim(0, grouped['rate'].max() * 1.6)

fig.suptitle('Fraud/Suspicious Rate by Categorical Features', fontsize=18, fontweight='bold', 
             color=COLORS['text'], y=1.01)
plt.tight_layout()
save_fig(fig, '04_categorical_fraud_rates')

# ====================================================================
# GRAPH 5: Key Features — Legit vs Suspicious Comparison
# ====================================================================
print("Graph 5: Key Features Comparison...")
key_features = ['F115', 'F670', 'F2082', 'F2122', 'F2678', 'F2956', 'F3836', 'F3894']
key_labels = ['F115\n(Risk Score)', 'F670\n(Alert Flag)', 'F2082\n(Rare Txn)', 'F2122\n(Digital)',
              'F2678\n(Amt Dev)', 'F2956\n(Txn Count)', 'F3836\n(Balance)', 'F3894\n(Age)']

fig, axes = plt.subplots(2, 4, figsize=(22, 10), facecolor=COLORS['bg'])
axes_flat = axes.flatten()

for i, (feat, label) in enumerate(zip(key_features, key_labels)):
    ax = axes_flat[i]
    data_l = df_legit[feat].dropna()
    data_s = df_susp[feat].dropna()
    
    # Clip outliers for visualization
    q99 = data_l.quantile(0.99)
    q01 = data_l.quantile(0.01)
    data_l_clip = data_l.clip(q01, q99)
    data_s_clip = data_s.clip(q01, q99)
    
    ax.hist(data_l_clip, bins=40, alpha=0.6, color=COLORS['legit'], label='Legit', density=True, edgecolor='none')
    ax.hist(data_s_clip, bins=20, alpha=0.7, color=COLORS['suspicious'], label='Suspicious', density=True, edgecolor='none')
    
    ax.axvline(data_l.mean(), color=COLORS['legit'], linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvline(data_s.mean(), color=COLORS['suspicious'], linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.set_title(label, fontsize=11, fontweight='bold', color=COLORS['text'])
    ax.set_facecolor(COLORS['card'])
    ax.tick_params(colors=COLORS['text'], labelsize=8)
    if i == 0:
        ax.legend(fontsize=9, facecolor=COLORS['card'], edgecolor=COLORS['grid'])

fig.suptitle('Key Feature Distributions: Legitimate vs Suspicious Accounts', fontsize=16,
             fontweight='bold', color=COLORS['text'], y=1.02)
plt.tight_layout()
save_fig(fig, '05_key_features_comparison')

# ====================================================================
# GRAPH 6: Top 20 Correlated Features with Target
# ====================================================================
print("Graph 6: Top Correlated Features...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['F3924', 'Unnamed: 0'])
target = df['F3924']

correlations = {}
for col in numeric_cols:
    s = df[col]
    if s.notna().sum() < 10 or s.std() == 0:
        continue
    correlations[col] = s.corr(target)

corr_s = pd.Series(correlations).dropna()
# Exclude F3912 (leakage)
corr_s = corr_s.drop('F3912', errors='ignore')
top20 = corr_s.abs().nlargest(20)

fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['bg'])
colors_corr = [COLORS['legit'] if correlations[f] > 0 else COLORS['suspicious'] for f in top20.index]
bars = ax.barh(top20.index[::-1], top20.values[::-1], color=colors_corr[::-1],
               edgecolor=COLORS['bg'], height=0.7)

for bar, feat in zip(bars, top20.index[::-1]):
    actual = correlations[feat]
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{actual:+.4f}', ha='left', va='center', fontsize=10, color=COLORS['text'])

ax.set_xlabel('Absolute Correlation with Target', fontsize=12, color=COLORS['text'])
ax.set_title('Top 20 Features Correlated with Target (F3912 excluded)', 
             fontsize=15, fontweight='bold', color=COLORS['text'])
ax.set_facecolor(COLORS['card'])
ax.tick_params(colors=COLORS['text'])

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS['legit'], label='Positive correlation'),
                   Patch(facecolor=COLORS['suspicious'], label='Negative correlation')]
ax.legend(handles=legend_elements, fontsize=11, facecolor=COLORS['card'], edgecolor=COLORS['grid'])
plt.tight_layout()
save_fig(fig, '06_top_correlations')

# ====================================================================
# GRAPH 7: Feature Block Composition
# ====================================================================
print("Graph 7: Feature Block Composition...")

# Categorize features
categories_map = {}
for col in df.columns:
    if col in ['Unnamed: 0', 'F3924'] or not col.startswith('F') or not col[1:].isdigit():
        continue
    s = df[col].dropna()
    if len(s) == 0:
        categories_map[col] = 'all_null'
    elif s.nunique() <= 1:
        categories_map[col] = 'constant'
    elif df[col].dtype == 'object':
        categories_map[col] = 'categorical'
    elif set(s.unique()) == {0, 1} or set(s.unique()) == {0.0, 1.0}:
        categories_map[col] = 'binary'
    elif s.min() < 0:
        categories_map[col] = 'negative'
    elif s.min() >= 0 and s.max() <= 1.0001:
        categories_map[col] = 'proportion'
    elif (s == s.astype(int)).all() and s.min() >= 0:
        categories_map[col] = 'count'
    elif s.max() - s.min() >= 100:
        categories_map[col] = 'large_float'
    else:
        categories_map[col] = 'small_float'

# Group by blocks of 500
block_data = {}
type_order = ['proportion', 'small_float', 'binary', 'large_float', 'count', 'negative', 'constant', 'all_null', 'categorical']
type_colors = {
    'proportion': '#FDCB6E', 'small_float': '#74B9FF', 'binary': '#00D4AA',
    'large_float': '#6C5CE7', 'count': '#E17055', 'negative': '#FF4B6E',
    'constant': '#636E72', 'all_null': '#2D3436', 'categorical': '#55EFC4'
}

block_size = 500
for col, cat in categories_map.items():
    num = int(col[1:])
    block = (num // block_size) * block_size
    key = f'F{block}-{block+block_size-1}'
    if key not in block_data:
        block_data[key] = {t: 0 for t in type_order}
    block_data[key][cat] = block_data[key].get(cat, 0) + 1

fig, ax = plt.subplots(figsize=(16, 8), facecolor=COLORS['bg'])
block_keys = sorted(block_data.keys(), key=lambda x: int(x.split('-')[0][1:]))
bottoms = np.zeros(len(block_keys))

for typ in type_order:
    values = [block_data[k].get(typ, 0) for k in block_keys]
    ax.bar(range(len(block_keys)), values, bottom=bottoms, label=typ.replace('_', ' ').title(),
           color=type_colors[typ], edgecolor=COLORS['bg'], linewidth=0.5, width=0.85)
    bottoms += values

ax.set_xticks(range(len(block_keys)))
ax.set_xticklabels(block_keys, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Number of Features', fontsize=12, color=COLORS['text'])
ax.set_title('Feature Type Composition by Block (500-feature ranges)', 
             fontsize=15, fontweight='bold', color=COLORS['text'])
ax.legend(loc='upper right', fontsize=9, facecolor=COLORS['card'], edgecolor=COLORS['grid'], ncol=3)
ax.set_facecolor(COLORS['card'])
ax.tick_params(colors=COLORS['text'])
plt.tight_layout()
save_fig(fig, '07_feature_blocks')

# ====================================================================
# GRAPH 8: Suspicious Account Profile (Spider/Radar)
# ====================================================================
print("Graph 8: Suspicious Account Profile...")
fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=COLORS['bg'])

# 8a: Occupation distribution
occ_data = df_susp['F3891'].value_counts()
axes[0].pie(occ_data.values, labels=occ_data.index, colors=PALETTE_CAT[:len(occ_data)],
            autopct='%1.1f%%', textprops={'color': COLORS['text'], 'fontsize': 10},
            wedgeprops={'edgecolor': COLORS['bg'], 'linewidth': 2}, startangle=90)
axes[0].set_title('Occupation of\nSuspicious Accounts', fontsize=14, fontweight='bold', color=COLORS['text'])

# 8b: Area distribution
area_data = df_susp['F3890'].value_counts()
area_labels = {'R': 'Rural', 'SU': 'Semi-Urban', 'M': 'Metro', 'U': 'Urban'}
axes[1].pie(area_data.values, labels=[area_labels.get(x, x) for x in area_data.index],
            colors=PALETTE_CAT[:len(area_data)],
            autopct='%1.1f%%', textprops={'color': COLORS['text'], 'fontsize': 11},
            wedgeprops={'edgecolor': COLORS['bg'], 'linewidth': 2}, startangle=90)
axes[1].set_title('Branch Area of\nSuspicious Accounts', fontsize=14, fontweight='bold', color=COLORS['text'])

# 8c: Account type
acct_data = df_susp['F3886'].value_counts()
axes[2].pie(acct_data.values, labels=acct_data.index, colors=PALETTE_CAT[:len(acct_data)],
            autopct='%1.1f%%', textprops={'color': COLORS['text'], 'fontsize': 11},
            wedgeprops={'edgecolor': COLORS['bg'], 'linewidth': 2}, startangle=90)
axes[2].set_title('Account Type of\nSuspicious Accounts', fontsize=14, fontweight='bold', color=COLORS['text'])

fig.suptitle('Profile of 81 Suspicious / Mule Accounts', fontsize=18, fontweight='bold',
             color=COLORS['text'], y=1.02)
plt.tight_layout()
save_fig(fig, '08_suspicious_profile')

# ====================================================================
# GRAPH 9: Data Leakage — F3912 vs Target
# ====================================================================
print("Graph 9: Data Leakage Analysis...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=COLORS['bg'])

# 9a: F3912 cross-tab heatmap
ct = pd.crosstab(df['F3912'], df['F3924'])
sns.heatmap(ct, annot=True, fmt='d', cmap='RdYlGn_r', ax=axes[0], linewidths=2,
            linecolor=COLORS['bg'], annot_kws={'fontsize': 18, 'fontweight': 'bold'},
            cbar_kws={'shrink': 0.8})
axes[0].set_title('F3912 vs F3924 (Target)\nDATA LEAKAGE!', fontsize=14, fontweight='bold', 
                  color=COLORS['suspicious'])
axes[0].set_xlabel('F3924 (Target)', fontsize=12, color=COLORS['text'])
axes[0].set_ylabel('F3912 (Fraud Flag)', fontsize=12, color=COLORS['text'])
axes[0].tick_params(colors=COLORS['text'])

# 9b: F2230 time period
period_data = df.groupby('F2230')['F3924'].agg(['sum', 'count']).reset_index()
period_data['rate'] = period_data['sum'] / period_data['count'] * 100
period_data = period_data.sort_values('count', ascending=False)

x = range(len(period_data))
bars1 = axes[1].bar(x, period_data['count'], color=COLORS['accent3'], alpha=0.7, label='Total accounts')
bars2 = axes[1].bar(x, period_data['sum'], color=COLORS['suspicious'], alpha=0.9, label='Suspicious accounts')

axes[1].set_xticks(x)
axes[1].set_xticklabels(period_data['F2230'], fontsize=12)
axes[1].set_ylabel('Count', fontsize=12, color=COLORS['text'])
axes[1].set_title('F2230 (Time Period) — Temporal Leakage\n100% of non-Oct25 = Suspicious', 
                  fontsize=14, fontweight='bold', color=COLORS['suspicious'])
axes[1].legend(fontsize=11, facecolor=COLORS['card'], edgecolor=COLORS['grid'])
axes[1].set_facecolor(COLORS['card'])
axes[1].tick_params(colors=COLORS['text'])
axes[1].set_yscale('log')

for bar, (_, row) in zip(bars1, period_data.iterrows()):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height()*1.1,
                f'{int(row["count"]):,}\n({row["rate"]:.0f}% susp)',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLORS['text'])

plt.tight_layout()
save_fig(fig, '09_data_leakage')

# ====================================================================
# GRAPH 10: Key Feature Box Plots (Legit vs Suspicious)
# ====================================================================
print("Graph 10: Key Feature Box Plots...")
box_features = ['F115', 'F321', 'F527', 'F531', 'F1692', 'F2122']
box_labels = ['F115 (Risk)', 'F321 (Velocity)', 'F527 (Peer)', 'F531 (Diversity)', 
              'F1692 (Activity Count)', 'F2122 (Digital Ratio)']

fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=COLORS['bg'])
axes_flat = axes.flatten()

for i, (feat, label) in enumerate(zip(box_features, box_labels)):
    ax = axes_flat[i]
    data = df[[feat, 'F3924']].dropna()
    
    bp = ax.boxplot([data[data['F3924']==0][feat], data[data['F3924']==1][feat]],
                    labels=['Legit', 'Suspicious'], patch_artist=True, 
                    widths=0.5, showfliers=False)
    
    bp['boxes'][0].set_facecolor(COLORS['legit'])
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(COLORS['suspicious'])
    bp['boxes'][1].set_alpha(0.6)
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color=COLORS['text'])
    
    ax.set_title(label, fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_facecolor(COLORS['card'])
    ax.tick_params(colors=COLORS['text'])

fig.suptitle('Key Features: Distribution Comparison (Box Plots, Outliers Hidden)', 
             fontsize=16, fontweight='bold', color=COLORS['text'], y=1.02)
plt.tight_layout()
save_fig(fig, '10_key_feature_boxplots')

# ====================================================================
# GRAPH 11: Correlation Heatmap of Key Features
# ====================================================================
print("Graph 11: Key Feature Correlation Heatmap...")
key_all = ['F115', 'F321', 'F527', 'F531', 'F670', 'F1692', 'F2082', 'F2122',
           'F2582', 'F2678', 'F2737', 'F2956', 'F3043', 'F3836', 'F3887', 'F3894', 'F3924']
corr_matrix = df[key_all].corr()

fig, ax = plt.subplots(figsize=(14, 12), facecolor=COLORS['bg'])
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, linewidths=0.5, linecolor=COLORS['bg'], 
            annot_kws={'fontsize': 8}, square=True,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
ax.set_title('Correlation Heatmap — Key Features + Target', fontsize=16, fontweight='bold', color=COLORS['text'])
ax.tick_params(colors=COLORS['text'], labelsize=10)
plt.tight_layout()
save_fig(fig, '11_correlation_heatmap')

# ====================================================================
# GRAPH 12: Account Age & Customer Age Distributions
# ====================================================================
print("Graph 12: Age Distributions...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=COLORS['bg'])

# Account age (F3887 in days → years)
for cls, color, label in [(0, COLORS['legit'], 'Legit'), (1, COLORS['suspicious'], 'Suspicious')]:
    data = df[df['F3924']==cls]['F3887'].dropna() / 365.25
    axes[0].hist(data, bins=40, alpha=0.6, color=color, label=f'{label} (n={len(data)})', 
                density=True, edgecolor='none')

axes[0].set_xlabel('Account Age (years)', fontsize=12, color=COLORS['text'])
axes[0].set_ylabel('Density', fontsize=12, color=COLORS['text'])
axes[0].set_title('Account Age Distribution (F3887)', fontsize=14, fontweight='bold', color=COLORS['text'])
axes[0].legend(fontsize=11, facecolor=COLORS['card'], edgecolor=COLORS['grid'])
axes[0].set_facecolor(COLORS['card'])
axes[0].tick_params(colors=COLORS['text'])

# Customer age (F3894)
for cls, color, label in [(0, COLORS['legit'], 'Legit'), (1, COLORS['suspicious'], 'Suspicious')]:
    data = df[df['F3924']==cls]['F3894'].dropna()
    axes[1].hist(data, bins=40, alpha=0.6, color=color, label=f'{label} (n={len(data)})', 
                density=True, edgecolor='none')

axes[1].set_xlabel('Customer Age (years)', fontsize=12, color=COLORS['text'])
axes[1].set_ylabel('Density', fontsize=12, color=COLORS['text'])
axes[1].set_title('Customer Age Distribution (F3894)', fontsize=14, fontweight='bold', color=COLORS['text'])
axes[1].legend(fontsize=11, facecolor=COLORS['card'], edgecolor=COLORS['grid'])
axes[1].set_facecolor(COLORS['card'])
axes[1].tick_params(colors=COLORS['text'])

plt.tight_layout()
save_fig(fig, '12_age_distributions')

# ====================================================================
# GRAPH 13: Feature Importance Preview (correlation-based)
# ====================================================================
print("Graph 13: Feature Importance by Block...")

# Compute average absolute correlation per block
block_corr = {}
for col, corr_val in correlations.items():
    if not col.startswith('F') or not col[1:].isdigit():
        continue
    num = int(col[1:])
    block = (num // 500) * 500
    key = f'F{block}-{block+499}'
    if key not in block_corr:
        block_corr[key] = []
    block_corr[key].append(abs(corr_val))

block_means = {k: np.mean(v) for k, v in block_corr.items()}
block_maxs = {k: np.max(v) for k, v in block_corr.items()}

fig, ax = plt.subplots(figsize=(14, 7), facecolor=COLORS['bg'])
keys = sorted(block_means.keys(), key=lambda x: int(x.split('-')[0][1:]))
means = [block_means[k] for k in keys]
maxs = [block_maxs[k] for k in keys]

x = range(len(keys))
ax.bar(x, maxs, color=COLORS['accent1'], alpha=0.4, label='Max |correlation|', width=0.8)
ax.bar(x, means, color=COLORS['accent1'], alpha=0.9, label='Mean |correlation|', width=0.8)

ax.set_xticks(x)
ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Absolute Correlation with Target', fontsize=12, color=COLORS['text'])
ax.set_title('Predictive Power by Feature Block (F3912 excluded from F3500–3999)', 
             fontsize=14, fontweight='bold', color=COLORS['text'])
ax.legend(fontsize=11, facecolor=COLORS['card'], edgecolor=COLORS['grid'])
ax.set_facecolor(COLORS['card'])
ax.tick_params(colors=COLORS['text'])
plt.tight_layout()
save_fig(fig, '13_block_importance')

# ====================================================================
# GRAPH 14: Missing Values Pattern — Suspicious vs Legit
# ====================================================================
print("Graph 14: Missing Values by Class...")
fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['bg'])

row_missing_legit = df_legit.isnull().sum(axis=1)
row_missing_susp = df_susp.isnull().sum(axis=1)

ax.hist(row_missing_legit, bins=40, alpha=0.6, color=COLORS['legit'], label=f'Legit (mean={row_missing_legit.mean():.0f})',
        density=True, edgecolor='none')
ax.hist(row_missing_susp, bins=15, alpha=0.7, color=COLORS['suspicious'], label=f'Suspicious (mean={row_missing_susp.mean():.0f})',
        density=True, edgecolor='none')
ax.axvline(row_missing_legit.mean(), color=COLORS['legit'], linestyle='--', linewidth=2)
ax.axvline(row_missing_susp.mean(), color=COLORS['suspicious'], linestyle='--', linewidth=2)

ax.set_xlabel('Number of Missing Features per Row', fontsize=12, color=COLORS['text'])
ax.set_ylabel('Density', fontsize=12, color=COLORS['text'])
ax.set_title('Missing Values Pattern: Legitimate vs Suspicious Accounts', 
             fontsize=15, fontweight='bold', color=COLORS['text'])
ax.legend(fontsize=12, facecolor=COLORS['card'], edgecolor=COLORS['grid'])
ax.set_facecolor(COLORS['card'])
ax.tick_params(colors=COLORS['text'])
plt.tight_layout()
save_fig(fig, '14_missing_by_class')

# ====================================================================
# Done
# ====================================================================
elapsed = time.time() - start
print(f'\n[DONE] All 14 graphs saved to {OUT_DIR}/')
print(f'Total time: {elapsed:.1f}s')
