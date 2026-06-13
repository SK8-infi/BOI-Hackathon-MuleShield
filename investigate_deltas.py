"""
Quick investigation: Do F2500-F2799 (temporal deltas/MoM changes) have 
discriminative power for mule detection?
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import mannwhitneyu

print("Loading dataset...")
df = pd.read_csv(r"c:\Github\BOI Hackathon\DataSet.csv")

# Exclude known leakage
drop_cols = ['Unnamed: 0', 'F3912', 'F2230']
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

target = df['F3924']

# Focus on the delta range: F2500-F2799
delta_cols = [f'F{i}' for i in range(2500, 2800) if f'F{i}' in df.columns]
print(f"\n=== F2500-F2799 TEMPORAL DELTA FEATURES ===")
print(f"Total features in range: {len(delta_cols)}")

# Check non-zero/non-constant
non_const = [c for c in delta_cols if df[c].nunique() > 1]
non_zero = [c for c in delta_cols if (df[c] != 0).any()]
print(f"Non-constant features: {len(non_const)}")
print(f"Features with any non-zero value: {len(non_zero)}")

# Sparsity analysis
sparsity = [(c, (df[c] == 0).mean()) for c in delta_cols]
very_sparse = [c for c, s in sparsity if s > 0.95]
moderate = [c for c, s in sparsity if 0.5 <= s <= 0.95]
dense = [c for c, s in sparsity if s < 0.5]
print(f"Very sparse (>95% zeros): {len(very_sparse)}")
print(f"Moderate (50-95% zeros): {len(moderate)}")
print(f"Dense (<50% zeros): {len(dense)}")

# Basic stats for mule vs legitimate
print(f"\n=== MULE vs LEGITIMATE COMPARISON ===")
mule_mask = target == 1
legit_mask = target == 0
print(f"Mules: {mule_mask.sum()}, Legit: {legit_mask.sum()}")

# Mann-Whitney U test on all non-constant delta features
print(f"\n--- Mann-Whitney U Tests (p < 0.05 = significant) ---")
sig_features = []
for col in non_const:
    mule_vals = df.loc[mule_mask, col]
    legit_vals = df.loc[legit_mask, col]
    # Only test if both groups have variation
    if mule_vals.std() > 0 or legit_vals.std() > 0:
        try:
            stat, p = mannwhitneyu(mule_vals, legit_vals, alternative='two-sided')
            if p < 0.05:
                sig_features.append((col, p, mule_vals.mean(), legit_vals.mean()))
        except:
            pass

sig_features.sort(key=lambda x: x[1])
print(f"\nSignificant features (p < 0.05): {len(sig_features)} out of {len(non_const)}")
print(f"\nTop 20 most significant:")
print(f"{'Feature':<10} {'p-value':<14} {'Mule Mean':<14} {'Legit Mean':<14} {'Diff':<10}")
print("-" * 62)
for feat, p, mm, lm in sig_features[:20]:
    print(f"{feat:<10} {p:<14.2e} {mm:<14.4f} {lm:<14.4f} {mm-lm:<10.4f}")

# Mutual Information for top delta features
print(f"\n=== MUTUAL INFORMATION (Top Delta Features) ===")
if len(non_const) > 0:
    # Use only non-constant features
    delta_data = df[non_const].fillna(0)
    mi_scores = mutual_info_classif(delta_data, target, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({'feature': non_const, 'MI': mi_scores}).sort_values('MI', ascending=False)
    
    print(f"\nTop 20 by Mutual Information:")
    print(f"{'Feature':<10} {'MI Score':<12}")
    print("-" * 22)
    for _, row in mi_df.head(20).iterrows():
        print(f"{row['feature']:<10} {row['MI']:<12.4f}")
    
    print(f"\n# Features with MI > 0.01: {(mi_df['MI'] > 0.01).sum()}")
    print(f"# Features with MI > 0.001: {(mi_df['MI'] > 0.001).sum()}")
    print(f"# Features with MI = 0: {(mi_df['MI'] == 0).sum()}")

# Check: Are mule deltas different?
print(f"\n=== MULE BEHAVIOR PATTERN IN DELTAS ===")
if len(sig_features) > 0:
    top_delta_feats = [f[0] for f in sig_features[:10]]
    print(f"\nFor top 10 significant delta features:")
    for feat in top_delta_feats:
        mule_vals = df.loc[mule_mask, feat]
        legit_vals = df.loc[legit_mask, feat]
        print(f"\n  {feat}:")
        print(f"    Mule  - mean: {mule_vals.mean():.4f}, median: {mule_vals.median():.4f}, "
              f"std: {mule_vals.std():.4f}, zeros: {(mule_vals==0).mean()*100:.1f}%")
        print(f"    Legit - mean: {legit_vals.mean():.4f}, median: {legit_vals.median():.4f}, "
              f"std: {legit_vals.std():.4f}, zeros: {(legit_vals==0).mean()*100:.1f}%")

# Compare with our existing FULL-60 features
print(f"\n=== ARE THESE IN OUR CURRENT MODEL? ===")
# Our current features
current_base = ['F3898', 'F1819', 'F3799', 'F1165', 'F1813', 'F3806', 'F162', 'F3800']
current_ps = ['F115', 'F321', 'F527', 'F531', 'F670', 'F1692', 'F2082', 'F2087', 
              'F2095', 'F2156', 'F2178', 'F2224', 'F2287', 'F2288', 'F2423', 'F2463']
all_current = current_base + current_ps

delta_in_model = [f for f in all_current if f in delta_cols]
print(f"Delta features already in model: {len(delta_in_model)}")
if delta_in_model:
    print(f"  {delta_in_model}")

# Check if any top delta features are NOT in our model
if len(sig_features) > 0:
    top_deltas_not_in_model = [f[0] for f in sig_features[:20] if f[0] not in all_current]
    print(f"\nTop significant delta features NOT in our model: {len(top_deltas_not_in_model)}")
    for f in top_deltas_not_in_model[:10]:
        print(f"  {f}")

print("\n=== DONE ===")
