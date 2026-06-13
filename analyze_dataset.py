import pandas as pd
import numpy as np
import time

start = time.time()
print("Loading dataset...")
df = pd.read_csv('DataSet.csv')
print(f"Loaded in {time.time()-start:.1f}s")

# ============================================================
# 1. DATASET OVERVIEW
# ============================================================
print('\n' + '=' * 60)
print('1. DATASET OVERVIEW')
print('=' * 60)
print(f'Shape: {df.shape[0]} rows x {df.shape[1]} columns')
print(f'Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB')

# ============================================================
# 2. TARGET VARIABLE (F3924) - CLASS IMBALANCE
# ============================================================
print('\n' + '=' * 60)
print('2. TARGET VARIABLE (F3924) - CLASS IMBALANCE')
print('=' * 60)
legit = (df['F3924'] == 0).sum()
suspicious = (df['F3924'] == 1).sum()
print(f'Legitimate accounts (0): {legit} ({legit/len(df)*100:.2f}%)')
print(f'Suspicious/Mule accounts (1): {suspicious} ({suspicious/len(df)*100:.2f}%)')
print(f'Imbalance ratio: {legit / suspicious:.1f}:1')

# ============================================================
# 3. DATA TYPES
# ============================================================
print('\n' + '=' * 60)
print('3. DATA TYPES')
print('=' * 60)
print(df.dtypes.value_counts())

# ============================================================
# 4. MISSING VALUES ANALYSIS
# ============================================================
print('\n' + '=' * 60)
print('4. MISSING VALUES ANALYSIS')
print('=' * 60)
missing_pct = (df.isnull().sum() / len(df)) * 100

bands = [
    (0, 0, '0% (No missing)'),
    (0.01, 10, '0-10%'),
    (10, 25, '10-25%'),
    (25, 50, '25-50%'),
    (50, 75, '50-75%'),
    (75, 99.99, '75-100%'),
    (100, 100, '100% (All missing)')
]
for low, high, label in bands:
    if low == 0 and high == 0:
        count = (missing_pct == 0).sum()
    elif low == 100:
        count = (missing_pct == 100).sum()
    else:
        count = ((missing_pct > low) & (missing_pct <= high)).sum()
    print(f'  {label}: {count} columns')

cols_100_missing = missing_pct[missing_pct == 100].index.tolist()
print(f'\nTotal columns with 100% missing: {len(cols_100_missing)}')

# Row-level missing
row_missing = df.isnull().sum(axis=1)
print(f'\nRow-level missing values:')
print(f'  Min per row: {row_missing.min()}')
print(f'  Max per row: {row_missing.max()}')
print(f'  Mean per row: {row_missing.mean():.1f}')
print(f'  Median per row: {row_missing.median():.1f}')

# ============================================================
# 5. CONSTANT/NEAR-CONSTANT COLUMNS
# ============================================================
print('\n' + '=' * 60)
print('5. CONSTANT/NEAR-CONSTANT COLUMNS')
print('=' * 60)
nunique = df.nunique()
const_cols = nunique[nunique <= 1].index.tolist()
print(f'Columns with <=1 unique value: {len(const_cols)}')
binary_cols = nunique[nunique == 2].index.tolist()
print(f'Columns with exactly 2 unique values (binary): {len(binary_cols)}')

# ============================================================
# 6. KEY FEATURES ANALYSIS (from problem statement)
# ============================================================
print('\n' + '=' * 60)
print('6. KEY FEATURES DETAILED ANALYSIS')
print('=' * 60)
key_features = ['F115', 'F321', 'F527', 'F531', 'F670', 'F1692', 'F2082', 'F2122',
                'F2582', 'F2678', 'F2737', 'F2956', 'F3043', 'F3836', 'F3887', 'F3894']

df_legit = df[df['F3924'] == 0]
df_susp = df[df['F3924'] == 1]

print(f'{"Feature":<10} {"Type":<8} {"NonNull%":<10} {"Unique":<8} {"CorrTarget":<12} {"MeanLegit":<14} {"MeanSusp":<14}')
print('-' * 80)
for f in key_features:
    if f in df.columns and df[f].dtype in ['int64', 'float64']:
        nn_pct = f'{df[f].notna().sum()/len(df)*100:.1f}%'
        uniq = df[f].nunique()
        corr = df[f].corr(df['F3924'])
        mean_0 = df_legit[f].mean()
        mean_1 = df_susp[f].mean()
        print(f'{f:<10} {str(df[f].dtype):<8} {nn_pct:<10} {uniq:<8} {corr:<12.4f} {mean_0:<14.4f} {mean_1:<14.4f}')

# ============================================================
# 7. CATEGORICAL FEATURES
# ============================================================
print('\n' + '=' * 60)
print('7. CATEGORICAL FEATURES')
print('=' * 60)
cat_features_map = {
    'F2230': 'Time Period',
    'F3886': 'Account Type',
    'F3888': 'Date',
    'F3889': 'Account Scheme',
    'F3890': 'Area Code',
    'F3891': 'Occupation',
    'F3892': 'Gender',
    'F3893': 'Customer Segment'
}

for f, desc in cat_features_map.items():
    if f in df.columns and df[f].dtype == 'object':
        print(f'\n--- {f} ({desc}) ---')
        print(f'  Unique values: {df[f].nunique()}')
        if df[f].nunique() <= 20:
            print('  Value counts:')
            for val, cnt in df[f].value_counts().items():
                susp_cnt = df_susp[df_susp[f] == val].shape[0]
                susp_pct = susp_cnt / cnt * 100 if cnt > 0 else 0
                print(f'    {val}: {cnt} (suspicious: {susp_cnt}, {susp_pct:.2f}%)')

# ============================================================
# 8. TOP CORRELATED FEATURES WITH TARGET
# (Efficient: compute correlation with target only, not full matrix)
# ============================================================
print('\n' + '=' * 60)
print('8. TOP 30 FEATURES MOST CORRELATED WITH TARGET (F3924)')
print('=' * 60)
print("Computing correlations with target only (not full matrix)...")

numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['F3924', 'Unnamed: 0'])
target = df['F3924']

# Compute correlation with target for each numeric column efficiently
correlations = {}
for col in numeric_cols:
    series = df[col]
    # Skip if all NaN or constant
    if series.notna().sum() < 10 or series.std() == 0:
        continue
    correlations[col] = series.corr(target)

corr_series = pd.Series(correlations).dropna().abs().sort_values(ascending=False)
print(f'\nFeatures with computable correlation: {len(corr_series)}')
print(f'\nTop 30:')
print(f'{"Feature":<10} {"Correlation":<14} {"Direction":<10}')
print('-' * 35)
for feat in corr_series.head(30).index:
    actual = correlations[feat]
    direction = "+" if actual > 0 else "-"
    print(f'{feat:<10} {abs(actual):<14.4f} {direction:<10}')

# ============================================================
# 9. FEATURE SPARSITY ANALYSIS
# ============================================================
print('\n' + '=' * 60)
print('9. FEATURE SPARSITY (zero-dominance)')
print('=' * 60)
# For numeric columns with data, check how many are mostly zero
zero_dominant = 0
for col in numeric_cols:
    if df[col].notna().sum() > 0:
        zero_pct = (df[col] == 0).sum() / df[col].notna().sum() * 100
        if zero_pct > 90:
            zero_dominant += 1
print(f'Columns where >90% of non-null values are zero: {zero_dominant}')

# ============================================================
# 10. SUSPICIOUS ACCOUNTS PROFILE
# ============================================================
print('\n' + '=' * 60)
print('10. SUSPICIOUS ACCOUNTS (F3924=1) PROFILE')
print('=' * 60)
print(f'Total suspicious accounts: {len(df_susp)}')
for f in ['F3886', 'F3889', 'F3890', 'F3891', 'F3892', 'F3893']:
    if f in df.columns:
        print(f'\n  {f} ({cat_features_map.get(f, "")}) distribution in suspicious:')
        vc = df_susp[f].value_counts()
        for val, cnt in vc.items():
            print(f'    {val}: {cnt} ({cnt/len(df_susp)*100:.1f}%)')

# ============================================================
# 11. USABLE FEATURES SUMMARY
# ============================================================
print('\n' + '=' * 60)
print('11. USABLE FEATURES SUMMARY')
print('=' * 60)
# Features with less than 50% missing and more than 1 unique value
usable = []
for col in df.columns:
    if col in ['Unnamed: 0', 'F3924']:
        continue
    miss_pct = df[col].isnull().sum() / len(df) * 100
    uniq = df[col].nunique()
    if miss_pct < 50 and uniq > 1:
        usable.append(col)
print(f'Features with <50% missing and >1 unique value: {len(usable)}')

usable_numeric = [c for c in usable if df[c].dtype in ['int64', 'float64']]
usable_cat = [c for c in usable if df[c].dtype == 'object']
print(f'  Numeric: {len(usable_numeric)}')
print(f'  Categorical: {len(usable_cat)}')

elapsed = time.time() - start
print(f'\n\nTotal analysis time: {elapsed:.1f}s')
