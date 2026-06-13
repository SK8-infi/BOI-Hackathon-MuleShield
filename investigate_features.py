import pandas as pd
import numpy as np
import time

start = time.time()
print("Loading dataset...")
df = pd.read_csv('DataSet.csv')
print(f"Loaded in {time.time()-start:.1f}s\n")

# ============================================================
# PHASE 1: Categorize ALL columns by their statistical shape
# ============================================================
print('=' * 80)
print('PHASE 1: FEATURE TYPE CLASSIFICATION')
print('=' * 80)

categories = {
    'constant': [],           # Only 1 unique value
    'binary_01': [],          # Exactly {0, 1}
    'binary_other': [],       # 2 unique values, not {0,1}
    'low_card_int': [],       # 3-10 unique integer values
    'count_like': [],         # Non-negative integers, >10 unique
    'proportion_01': [],      # Continuous float in [0, 1]
    'small_float': [],        # Float, small range (< 100)
    'large_float': [],        # Float, large range (> 100)
    'negative_float': [],     # Contains negative values
    'categorical': [],        # Object type
    'all_null': [],           # 100% missing
    'other': []
}

for col in df.columns:
    if col in ['Unnamed: 0', 'F3924']:
        continue
    
    s = df[col].dropna()
    
    if len(s) == 0:
        categories['all_null'].append(col)
        continue
    
    if s.nunique() <= 1:
        categories['constant'].append(col)
        continue
    
    if df[col].dtype == 'object':
        categories['categorical'].append(col)
        continue
    
    unique_vals = set(s.unique())
    
    if unique_vals == {0, 1} or unique_vals == {0.0, 1.0}:
        categories['binary_01'].append(col)
        continue
    
    if s.nunique() == 2:
        categories['binary_other'].append(col)
        continue
    
    if s.min() < 0:
        categories['negative_float'].append(col)
        continue
    
    # Check if all values are integers
    is_integer = (s == s.astype(int)).all() if s.dtype == 'float64' else True
    
    if is_integer and s.nunique() <= 10:
        categories['low_card_int'].append(col)
        continue
    
    if is_integer and s.min() >= 0:
        categories['count_like'].append(col)
        continue
    
    if s.min() >= 0 and s.max() <= 1.0001:
        categories['proportion_01'].append(col)
        continue
    
    if s.max() - s.min() < 100:
        categories['small_float'].append(col)
        continue
    
    if s.max() - s.min() >= 100:
        categories['large_float'].append(col)
        continue
    
    categories['other'].append(col)

for cat, cols in categories.items():
    print(f'\n{cat}: {len(cols)} features')
    if len(cols) <= 20:
        print(f'  {cols}')
    else:
        print(f'  First 10: {cols[:10]}')
        print(f'  Last 10: {cols[-10:]}')

# ============================================================
# PHASE 2: Analyze the KNOWN categorical features in detail
# ============================================================
print('\n\n' + '=' * 80)
print('PHASE 2: KNOWN CATEGORICAL FEATURES')
print('=' * 80)

known_cats = {
    'F2230': 'Possible: Data/reporting period',
    'F3886': 'Account Type (Savings, Current, MSME, etc.)',
    'F3888': 'Account Opening Date (date format)',
    'F3889': 'Account Scheme (G365D=General 365 Day, L=Loan periods)',
    'F3890': 'Area Code (R=Rural, M=Metro, SU=Semi-Urban, U=Urban)',
    'F3891': 'Occupation',
    'F3892': 'Gender',
    'F3893': 'Customer Segment (RETAIL/CORPORATE)',
}

for f, desc in known_cats.items():
    print(f'\n{f}: {desc}')
    print(f'  Values: {df[f].unique().tolist()}')

# ============================================================
# PHASE 3: Analyze binary features - what do they represent?
# ============================================================
print('\n\n' + '=' * 80)
print('PHASE 3: BINARY (0/1) FEATURES ANALYSIS')
print('=' * 80)

binary_cols = categories['binary_01']
print(f'\nTotal binary features: {len(binary_cols)}')

# Group binary features by their "1" percentage
binary_stats = []
for col in binary_cols:
    pct_one = df[col].mean() * 100
    missing_pct = df[col].isnull().sum() / len(df) * 100
    # Check if it correlates with any categorical feature
    binary_stats.append({
        'feature': col,
        'pct_one': pct_one,
        'missing_pct': missing_pct,
        'corr_target': df[col].corr(df['F3924'])
    })

binary_df = pd.DataFrame(binary_stats).sort_values('pct_one')
print('\nBinary features by % of 1s:')
print(f'  Near-zero (<1% ones): {len(binary_df[binary_df["pct_one"] < 1])}')
print(f'  Rare (1-10% ones): {len(binary_df[(binary_df["pct_one"] >= 1) & (binary_df["pct_one"] < 10)])}')
print(f'  Moderate (10-50% ones): {len(binary_df[(binary_df["pct_one"] >= 10) & (binary_df["pct_one"] < 50)])}')
print(f'  Common (50-90% ones): {len(binary_df[(binary_df["pct_one"] >= 50) & (binary_df["pct_one"] < 90)])}')
print(f'  Very common (>90% ones): {len(binary_df[binary_df["pct_one"] >= 90])}')

# Binary features most correlated with target
print('\nBinary features most correlated with target:')
binary_df_sorted = binary_df.reindex(binary_df['corr_target'].abs().sort_values(ascending=False).index)
for _, row in binary_df_sorted.head(15).iterrows():
    print(f'  {row["feature"]}: corr={row["corr_target"]:.4f}, %ones={row["pct_one"]:.2f}%, missing={row["missing_pct"]:.1f}%')

# ============================================================
# PHASE 4: Detect one-hot encoded groups
# ============================================================
print('\n\n' + '=' * 80)
print('PHASE 4: ONE-HOT ENCODED GROUPS DETECTION')
print('=' * 80)

# Check if consecutive binary columns sum to 1 (one-hot groups)
binary_nums = sorted([int(c[1:]) for c in binary_cols])
print(f'\nChecking consecutive binary feature groups...')

# Sample some ranges of consecutive binary features
groups_found = []
i = 0
while i < len(binary_nums):
    # Find consecutive run
    run = [binary_nums[i]]
    j = i + 1
    while j < len(binary_nums) and binary_nums[j] == binary_nums[j-1] + 1:
        run.append(binary_nums[j])
        j += 1
    
    if len(run) >= 3:
        run_cols = [f'F{n}' for n in run]
        # Check if they sum to ~1 per row
        row_sums = df[run_cols].sum(axis=1)
        sum_stats = row_sums.dropna()
        if len(sum_stats) > 0 and abs(sum_stats.mean() - 1.0) < 0.05:
            groups_found.append({
                'start': run[0], 'end': run[-1], 'size': len(run),
                'cols': run_cols, 'mean_sum': sum_stats.mean()
            })
    i = j

print(f'Found {len(groups_found)} potential one-hot encoded groups:')
for g in groups_found[:20]:
    print(f'  F{g["start"]}-F{g["end"]} ({g["size"]} cols, row sum mean={g["mean_sum"]:.3f})')

# ============================================================
# PHASE 5: Analyze proportion/ratio features (0-1 range)
# ============================================================
print('\n\n' + '=' * 80)
print('PHASE 5: PROPORTION/RATIO FEATURES (0-1 range)')
print('=' * 80)

prop_cols = categories['proportion_01']
print(f'\nTotal proportion features: {len(prop_cols)}')
print('\nSample statistics:')
for col in prop_cols[:15]:
    s = df[col].dropna()
    print(f'  {col}: mean={s.mean():.4f}, std={s.std():.4f}, unique={s.nunique()}, missing={df[col].isnull().sum()/len(df)*100:.1f}%')

# ============================================================
# PHASE 6: Analyze count-like features
# ============================================================
print('\n\n' + '=' * 80)
print('PHASE 6: COUNT-LIKE FEATURES (non-negative integers)')
print('=' * 80)

count_cols = categories['count_like']
print(f'\nTotal count-like features: {len(count_cols)}')
print('\nSample statistics:')
for col in count_cols[:15]:
    s = df[col].dropna()
    print(f'  {col}: min={s.min():.0f}, max={s.max():.0f}, mean={s.mean():.2f}, std={s.std():.2f}, unique={s.nunique()}, missing={df[col].isnull().sum()/len(df)*100:.1f}%')

# ============================================================
# PHASE 7: Analyze large float features (monetary amounts?)
# ============================================================
print('\n\n' + '=' * 80)
print('PHASE 7: LARGE FLOAT FEATURES (possible monetary amounts)')
print('=' * 80)

large_cols = categories['large_float']
print(f'\nTotal large float features: {len(large_cols)}')
print('\nSample statistics:')
for col in large_cols[:20]:
    s = df[col].dropna()
    print(f'  {col}: min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}, median={s.median():.2f}, unique={s.nunique()}, missing={df[col].isnull().sum()/len(df)*100:.1f}%')

# ============================================================
# PHASE 8: Analyze negative float features
# ============================================================
print('\n\n' + '=' * 80)
print('PHASE 8: NEGATIVE FLOAT FEATURES (possible deltas/changes)')
print('=' * 80)

neg_cols = categories['negative_float']
print(f'\nTotal features with negative values: {len(neg_cols)}')
print('\nSample statistics:')
for col in neg_cols[:20]:
    s = df[col].dropna()
    pct_neg = (s < 0).sum() / len(s) * 100
    print(f'  {col}: min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.4f}, %negative={pct_neg:.1f}%, unique={s.nunique()}, missing={df[col].isnull().sum()/len(df)*100:.1f}%')

# ============================================================
# PHASE 9: Feature number ranges and patterns
# ============================================================
print('\n\n' + '=' * 80)
print('PHASE 9: FEATURE NUMBER BLOCK ANALYSIS')
print('=' * 80)

# Analyze features in blocks of 100 to see if there are thematic groupings
blocks = {}
for col in df.columns:
    if col.startswith('F') and col[1:].isdigit():
        num = int(col[1:])
        block = (num // 100) * 100
        block_key = f'F{block}-F{block+99}'
        if block_key not in blocks:
            blocks[block_key] = {'cols': [], 'types': {}}
        blocks[block_key]['cols'].append(col)
        
        # Determine type
        for cat, cols_list in categories.items():
            if col in cols_list:
                blocks[block_key]['types'][cat] = blocks[block_key]['types'].get(cat, 0) + 1
                break

print('\nFeature blocks (100-feature ranges) - dominant types:')
for block_key in sorted(blocks.keys(), key=lambda x: int(x.split('-')[0][1:])):
    info = blocks[block_key]
    total = len(info['cols'])
    dominant = sorted(info['types'].items(), key=lambda x: x[1], reverse=True)
    dom_str = ', '.join([f'{t}:{c}' for t, c in dominant[:3]])
    print(f'  {block_key} ({total} cols): {dom_str}')

# ============================================================
# PHASE 10: Interpret the key features from problem statement
# ============================================================
print('\n\n' + '=' * 80)
print('PHASE 10: INTERPRETING KEY FEATURES FROM PROBLEM STATEMENT')
print('=' * 80)

key_interpretations = {
    'F115': {'range': '[0,1]', 'unique': 97, 'type': 'proportion'},
    'F321': {'range': '[0,10.6]', 'unique': 308, 'type': 'ratio/score'},
    'F527': {'range': '[0,48.42]', 'unique': 301, 'type': 'ratio/multiplier'},
    'F531': {'range': '[0,20.96]', 'unique': 500, 'type': 'ratio/score'},
    'F670': {'range': '{0,1}', 'unique': 2, 'type': 'binary flag'},
    'F1692': {'range': '[0,14]', 'unique': 12, 'type': 'count (low)'},
    'F2082': {'range': '[0,1]', 'unique': 571, 'type': 'proportion'},
    'F2122': {'range': '[0,1]', 'unique': 1235, 'type': 'proportion'},
    'F2582': {'range': '[-0.88,18.89]', 'unique': 290, 'type': 'change/delta'},
    'F2678': {'range': '[-0.91,1.6M]', 'unique': 435, 'type': 'monetary amount'},
    'F2737': {'range': '[-0.94,1707]', 'unique': 379, 'type': 'monetary/ratio'},
    'F2956': {'range': '[0,11548]', 'unique': 718, 'type': 'count/amount'},
    'F3043': {'range': '[0,21819]', 'unique': 713, 'type': 'count/amount'},
    'F3836': {'range': '[-20B,16B]', 'unique': 8898, 'type': 'large monetary'},
    'F3887': {'range': '[0,1510]', 'unique': 472, 'type': 'count'},
    'F3894': {'range': '[-2,94]', 'unique': 95, 'type': 'age/tenure'},
}

for f, info in key_interpretations.items():
    s = df[f].dropna()
    print(f'\n{f}:')
    print(f'  Range: {info["range"]}, Unique: {info["unique"]}, Likely type: {info["type"]}')
    if df[f].dtype in ['int64', 'float64']:
        print(f'  Percentiles: 25%={s.quantile(0.25):.4f}, 50%={s.quantile(0.5):.4f}, 75%={s.quantile(0.75):.4f}, 95%={s.quantile(0.95):.4f}')
        # Check if the feature correlates with F3894 (likely age) or F3887 (likely count)
        if f not in ['F3894', 'F3887']:
            corr_age = df[f].corr(df['F3894'])
            corr_count = df[f].corr(df['F3887'])
            print(f'  Corr with F3894(age?): {corr_age:.4f}, Corr with F3887(count?): {corr_count:.4f}')

# ============================================================
# PHASE 11: Check F3888 (date) to understand account age
# ============================================================
print('\n\n' + '=' * 80)
print('PHASE 11: DATE ANALYSIS (F3888)')
print('=' * 80)

dates = pd.to_datetime(df['F3888'], format='mixed', dayfirst=False, errors='coerce')
print(f'Date range: {dates.min()} to {dates.max()}')
print(f'Null dates: {dates.isnull().sum()}')

# Calculate account age in years from date
ref_date = pd.Timestamp('2025-10-01')  # Likely reference point given F2230
account_age_years = (ref_date - dates).dt.days / 365.25
print(f'\nAccount age (years from Oct 2025):')
print(f'  Min: {account_age_years.min():.1f}')
print(f'  Max: {account_age_years.max():.1f}')
print(f'  Mean: {account_age_years.mean():.1f}')
print(f'  Median: {account_age_years.median():.1f}')

# Does F3894 correlate with account age?
corr_age = account_age_years.corr(df['F3894'])
print(f'\nCorrelation of account_age with F3894: {corr_age:.4f}')
# Also check F3887
corr_txn = account_age_years.corr(df['F3887'])
print(f'Correlation of account_age with F3887: {corr_txn:.4f}')

# ============================================================
# PHASE 12: Feature correlation clusters
# ============================================================
print('\n\n' + '=' * 80)
print('PHASE 12: FEATURE CORRELATION CLUSTERS (among key features)')
print('=' * 80)

key_feat_list = ['F115', 'F321', 'F527', 'F531', 'F670', 'F1692', 'F2082', 'F2122',
                'F2582', 'F2678', 'F2737', 'F2956', 'F3043', 'F3836', 'F3887', 'F3894']

corr_matrix = df[key_feat_list].corr()
print('\nHighly correlated pairs among key features (|corr| > 0.3):')
for i in range(len(key_feat_list)):
    for j in range(i+1, len(key_feat_list)):
        c = corr_matrix.iloc[i, j]
        if abs(c) > 0.3:
            print(f'  {key_feat_list[i]} <-> {key_feat_list[j]}: {c:.4f}')

# ============================================================
# PHASE 13: Investigate F3912 (high correlation with target)
# ============================================================
print('\n\n' + '=' * 80)
print('PHASE 13: F3912 INVESTIGATION (0.97 corr with target)')
print('=' * 80)
s = df['F3912']
print(f'dtype: {s.dtype}')
print(f'unique values: {s.nunique()}')
print(f'value counts:\n{s.value_counts()}')
print(f'\nCross-tab with target:')
print(pd.crosstab(df['F3912'], df['F3924']))

# Also check neighbors of F3912
for f in ['F3908', 'F3909', 'F3910', 'F3911', 'F3912', 'F3913', 'F3914']:
    s2 = df[f]
    print(f'\n{f}: dtype={s2.dtype}, unique={s2.nunique()}, corr_target={s2.corr(df["F3924"]):.4f}')
    if s2.nunique() <= 10:
        print(f'  values: {s2.value_counts().to_dict()}')

elapsed = time.time() - start
print(f'\n\nTotal analysis time: {elapsed:.1f}s')
