"""
Predict Suspicious Mule Accounts
=================================
Usage: python predict_blend.py <input_csv> [output_csv]

The input CSV must have the same columns as DataSet.csv.
"""
import pandas as pd
import numpy as np
import joblib, sys, os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def clean_col(s):
    if s.dtype == object:
        s = s.astype(str).str.strip('[]').str.strip()
    return pd.to_numeric(s, errors='coerce')

def build_features(df, artifact):
    OUR_RAW = ['F3898', 'F1819', 'F3799', 'F1165', 'F1813', 'F3806', 'F162', 'F3800']
    STABLE_8 = ['F3811', 'F3806', 'F3799', 'F3805', 'F3813', 'F3801', 'F3898', 'F3807']
    PS_FEATURES = ['F115', 'F321', 'F527', 'F531', 'F670', 'F1692', 'F2082', 'F2122',
                   'F2582', 'F2678', 'F2737', 'F2956', 'F3043', 'F3836', 'F3887', 'F3894']

    features = pd.DataFrame(index=df.index)
    for f in OUR_RAW:
        features[f] = clean_col(df[f]).values.astype(np.float64)

    f162 = clean_col(df['F162']).values.astype(np.float64)
    f3898 = clean_col(df['F3898']).values.astype(np.float64)
    features['F162_div_F3898'] = np.where(f3898 != 0, f162 / (f3898 + 1e-10), 0)
    stable_clean = pd.DataFrame({c: clean_col(df[c]) for c in STABLE_8}, index=df.index)
    features['max_value_top8'] = stable_clean.max(axis=1).values.astype(np.float64)
    f3805 = clean_col(df['F3805']).values.astype(np.float64)
    features['F3898_div_F3805'] = np.where(f3805 != 0, f3898 / (f3805 + 1e-10), 0)
    f0_f499 = [f'F{i}' for i in range(500) if f'F{i}' in df.columns]
    features['missing_count_F0_F500'] = df[f0_f499].isnull().sum(axis=1).values.astype(np.float64)
    f3811 = clean_col(df['F3811']).values.astype(np.float64)
    features['F3811_div_F3805'] = np.where(f3805 != 0, f3811 / (f3805 + 1e-10), 0)
    features['F3898_div_F3811'] = np.where(f3811 != 0, f3898 / (f3811 + 1e-10), 0)
    for col in ['F162_div_F3898', 'F3898_div_F3805', 'F3811_div_F3805', 'F3898_div_F3811']:
        features[col] = np.clip(features[col], -1e6, 1e6)
    for f in PS_FEATURES:
        features[f] = clean_col(df[f]).values.astype(np.float64)
    features = features.fillna(0)

    X_enhanced = features.values.astype(np.float64)

    # FULL-60
    full = features.copy()
    top5 = ['F3898', 'F1165', 'F1819', 'F1813', 'F162']
    for i in range(len(top5)):
        for j in range(i+1, len(top5)):
            a, b = top5[i], top5[j]
            full[f'{a}_x_{b}'] = features[a] * features[b]
            full[f'{a}_div_{b}'] = np.where(features[b] != 0, features[a] / (features[b] + 1e-10), 0)
    for f in OUR_RAW:
        full[f'{f}_rank'] = features[f].rank(pct=True)
    X_for_km = artifact['scaler'].transform(features.values)
    full['cluster_id'] = artifact['kmeans'].predict(X_for_km)
    full['low_value_count'] = 0
    for f in STABLE_8:
        col = clean_col(df[f])
        full['low_value_count'] += (col < col.median()).astype(int).values
    for c in full.columns:
        if full[c].dtype in [np.float64, np.float32]:
            full[c] = np.clip(full[c], -1e6, 1e6)
    full = full.fillna(0)
    X_full = full.values.astype(np.float64)

    return X_enhanced, X_full

def predict(input_csv, output_csv=None):
    print(f"Loading model...")
    artifact = joblib.load('model/final_blend_model.joblib')

    print(f"Loading data: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"  Rows: {len(df)}")

    X_enhanced, X_full = build_features(df, artifact)

    proba1 = artifact['model1'].predict_proba(X_enhanced)[:, 1]
    proba2 = artifact['model2'].predict_proba(X_full)[:, 1]

    w1, w2 = artifact['weight1'], artifact['weight2']
    blend_proba = w1 * proba1 + w2 * proba2

    threshold = artifact['threshold']
    predictions = (blend_proba >= threshold).astype(int)

    result = pd.DataFrame({
        'row_index': range(len(df)),
        'suspicious_probability': blend_proba.round(4),
        'prediction': predictions,
        'risk_level': pd.cut(blend_proba, bins=[0, 0.2, 0.5, 0.8, 1.0],
                             labels=['Low', 'Medium', 'High', 'Critical'])
    })

    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_predictions.csv')
    result.to_csv(output_csv, index=False)
    
    n_flagged = predictions.sum()
    print(f"\n  Results:")
    print(f"    Flagged as suspicious: {n_flagged} / {len(df)}")
    print(f"    Saved to: {output_csv}")
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict_blend.py <input.csv> [output.csv]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    predict(sys.argv[1], out)
