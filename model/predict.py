"""
BOI Hackathon - Mule Account Prediction Script
===============================================
Load a trained model and predict on new data.

Usage:
    python model/predict.py                        # Score DataSet.csv
    python model/predict.py new_data.csv           # Score custom file
    python model/predict.py new_data.csv --output results.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
import joblib
import json


def load_model(model_dir='model'):
    """Load saved model and feature engineer."""
    model_path = os.path.join(model_dir, 'xgboost_model.joblib')
    fe_path = os.path.join(model_dir, 'feature_engineer.joblib')
    meta_path = os.path.join(model_dir, 'model_metadata.json')

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        print("  Run 'python model/pipeline.py' first to train the model.")
        sys.exit(1)

    model = joblib.load(model_path)
    fe = joblib.load(fe_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    print(f"[OK] Loaded model: {metadata['model_type']}")
    print(f"     Features: {metadata['n_features']}")
    print(f"     CV F1: {metadata['cv_metrics']['f1_mean']}")
    print(f"     Optimal threshold: {metadata['optimal_threshold']}")

    return model, fe, metadata


def predict(model, fe, df, threshold=None, metadata=None):
    """Generate predictions for a dataframe."""
    if threshold is None:
        threshold = metadata.get('optimal_threshold', 0.5) if metadata else 0.5

    # Engineer features
    X_df = fe.transform(df)
    X = X_df.values

    # Predict probabilities
    probs = model.predict_proba(X)[:, 1]

    # Create output
    results = pd.DataFrame({
        'row_index': range(len(df)),
        'risk_score': np.round(probs, 6),
        'prediction': (probs >= threshold).astype(int),
        'risk_tier': pd.cut(probs, bins=[0, 0.1, 0.2, 0.5, 0.8, 1.0],
                            labels=['Minimal', 'Low', 'Medium', 'High', 'Very High'],
                            include_lowest=True),
    })

    # Add identifiers if available
    if 'F3888' in df.columns:
        results['account_open_date'] = df['F3888'].values
    if 'F3894' in df.columns:
        results['customer_age'] = df['F3894'].values
    if 'F3887' in df.columns:
        results['account_age_days'] = df['F3887'].values
    if 'F3924' in df.columns:
        results['true_label'] = df['F3924'].values

    return results.sort_values('risk_score', ascending=False)


def main():
    parser = argparse.ArgumentParser(description='Mule Account Risk Scoring')
    parser.add_argument('input', nargs='?', default='DataSet.csv',
                        help='Input CSV file (default: DataSet.csv)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV file (default: predictions_<input>.csv)')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                        help='Classification threshold (default: from model metadata)')
    parser.add_argument('--model-dir', '-m', default='model',
                        help='Model directory (default: model)')
    args = parser.parse_args()

    # Load model
    model, fe, metadata = load_model(args.model_dir)

    # Load data
    print(f"\nLoading data: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(df)} accounts loaded")

    # Predict
    threshold = args.threshold or metadata.get('optimal_threshold', 0.5)
    print(f"\nScoring with threshold={threshold:.4f}...")
    results = predict(model, fe, df, threshold, metadata)

    # Summary
    flagged = results[results['prediction'] == 1]
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"  Total accounts:  {len(results):,}")
    print(f"  Flagged (susp):  {len(flagged):,} ({len(flagged)/len(results)*100:.2f}%)")
    print(f"  Threshold used:  {threshold:.4f}")

    # Risk tier breakdown
    print(f"\n  Risk Tiers:")
    for tier in ['Very High', 'High', 'Medium', 'Low', 'Minimal']:
        count = (results['risk_tier'] == tier).sum()
        print(f"    {tier:>10}: {count:>6,}")

    # If true labels available, compute metrics
    if 'true_label' in results.columns:
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        y_true = results['true_label'].values
        y_pred = results['prediction'].values
        y_prob = results['risk_score'].values
        print(f"\n  Evaluation (vs true labels):")
        print(f"    F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"    Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"    Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"    AUC:       {roc_auc_score(y_true, y_prob):.4f}")

    # Save
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f'predictions_{base}.csv'

    results.to_csv(args.output, index=False)
    print(f"\n  [OK] Results saved to: {args.output}")

    # Show top flagged
    print(f"\n  Top 10 Highest Risk Accounts:")
    print(f"  {'Row':>6} {'Score':>8} {'Tier':>10} {'CustAge':>8} {'AcctDays':>8}")
    print(f"  {'-'*45}")
    for _, row in results.head(10).iterrows():
        age = row.get('customer_age', 'N/A')
        days = row.get('account_age_days', 'N/A')
        print(f"  {int(row['row_index']):>6} {row['risk_score']:>8.4f} {str(row['risk_tier']):>10} "
              f"{str(age):>8} {str(days):>8}")


if __name__ == '__main__':
    main()
