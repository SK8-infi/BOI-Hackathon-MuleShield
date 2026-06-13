"""
BOI Hackathon - Mule Account Detection Pipeline
================================================
Production-ready ML pipeline for Problem Statement 2:
AI/ML-Based Classification of Suspicious Mule Accounts

Features:
- 14 optimal features (8 raw + 6 engineered) from Phase 5 research
- XGBoost with Optuna hyperparameter tuning
- SMOTE(0.3) oversampling
- Stratified 5-Fold cross-validation
- SHAP explainability
- Full evaluation report with graphs
- Risk scores for all accounts
- Saved model for inference

Usage:
    python model/pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, roc_curve, classification_report,
    confusion_matrix, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
from scipy import stats
import joblib

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ======================================================================
# Configuration
# ======================================================================
class Config:
    DATA_PATH = 'DataSet.csv'
    MODEL_DIR = 'model'
    EVAL_DIR = 'model/evaluation'
    TARGET = 'F3924'
    LEAKAGE = ['F3912', 'F2230']

    # Raw features from forward selection
    RAW_FEATURES = ['F3898', 'F1819', 'F3799', 'F1165', 'F1813', 'F3806', 'F162', 'F3800']

    # Top 8 stable features (for engineered features)
    STABLE_8 = ['F3811', 'F3806', 'F3799', 'F3805', 'F3813', 'F3801', 'F3898', 'F3807']

    # F0-F499 columns (for missing count)
    F0_F499_RANGE = range(500)

    # CV settings
    N_SPLITS = 5
    RANDOM_STATE = 42

    # SMOTE settings
    SMOTE_RATIO = 0.3
    SMOTE_K = 3

    # Optuna settings
    N_TRIALS = 100
    OPTUNA_TIMEOUT = 300  # 5 minutes max

    # Thresholds to evaluate
    THRESHOLDS = {'cost_optimal': 0.20, 'balanced': 0.50, 'f1_optimal': 0.705}

    # Plot style
    COLORS = {
        'primary': '#ff006e', 'secondary': '#00f5d4', 'accent': '#8338ec',
        'warn': '#ffbe0b', 'blue': '#3a86ff', 'bg': '#0a0a23',
        'card': '#161b22', 'text': '#e6edf3', 'grid': '#21262d',
        'green': '#2ed573', 'orange': '#ff7f50'
    }

plt.style.use('dark_background')

# ======================================================================
# Feature Engineering
# ======================================================================
class FeatureEngineer:
    """Creates all 6 engineered features from raw data."""

    def __init__(self):
        self.legitimate_medians = None  # Fitted on training data
        self.f0_f499_cols = None

    def fit(self, df, y):
        """Learn parameters from training data only."""
        legit_mask = y == 0
        # Learn class medians from legitimate accounts
        self.legitimate_medians = df[Config.STABLE_8].loc[legit_mask].median()
        # Identify F0-F499 columns present in data
        self.f0_f499_cols = [f'F{i}' for i in Config.F0_F499_RANGE if f'F{i}' in df.columns]
        return self

    @staticmethod
    def _clean_col(series):
        """Clean bracket-wrapped values like '[8.68012E-1]' → 0.868012."""
        if series.dtype == object:
            series = series.astype(str).str.strip('[]').str.strip()
        return pd.to_numeric(series, errors='coerce').fillna(0)

    def transform(self, df):
        """Apply feature engineering to dataframe."""
        features = pd.DataFrame(index=df.index)

        # Raw features (cleaned for bracket-wrapped values)
        for f in Config.RAW_FEATURES:
            features[f] = self._clean_col(df[f]).values

        # --- Engineered Feature 1: F162 / F3898 ---
        f162 = self._clean_col(df['F162']).values
        f3898 = self._clean_col(df['F3898']).values
        features['F162_div_F3898'] = np.where(f3898 != 0, f162 / (f3898 + 1e-10), 0)

        # --- Engineered Feature 2: max_value_top8 ---
        stable_clean = pd.DataFrame({c: self._clean_col(df[c]) for c in Config.STABLE_8}, index=df.index)
        features['max_value_top8'] = stable_clean.max(axis=1).values

        # --- Engineered Feature 3: F3898 / F3805 ---
        f3805 = self._clean_col(df['F3805']).values
        features['F3898_div_F3805'] = np.where(f3805 != 0, f3898 / (f3805 + 1e-10), 0)

        # --- Engineered Feature 4: missing_count_F0_F500 ---
        features['missing_count_F0_F500'] = df[self.f0_f499_cols].isnull().sum(axis=1).values

        # --- Engineered Feature 5: F3811 / F3805 ---
        f3811 = self._clean_col(df['F3811']).values
        features['F3811_div_F3805'] = np.where(f3805 != 0, f3811 / (f3805 + 1e-10), 0)

        # --- Engineered Feature 6: F3898 / F3811 ---
        features['F3898_div_F3811'] = np.where(f3811 != 0, f3898 / (f3811 + 1e-10), 0)

        # Clip extreme values in ratio features
        ratio_cols = ['F162_div_F3898', 'F3898_div_F3805', 'F3811_div_F3805', 'F3898_div_F3811']
        for col in ratio_cols:
            features[col] = np.clip(features[col], -1e6, 1e6)

        return features

    def fit_transform(self, df, y):
        """Fit and transform in one step."""
        self.fit(df, y)
        return self.transform(df)

    @property
    def feature_names(self):
        return Config.RAW_FEATURES + [
            'F162_div_F3898', 'max_value_top8', 'F3898_div_F3805',
            'missing_count_F0_F500', 'F3811_div_F3805', 'F3898_div_F3811'
        ]


# ======================================================================
# Hyperparameter Tuning
# ======================================================================
def tune_xgboost(X, y, cv, pos_weight):
    """Tune XGBoost with Optuna or fallback to defaults."""

    if not HAS_OPTUNA:
        print("  [!!] Optuna not available, using researched defaults")
        return {
            'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
            'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'scale_pos_weight': pos_weight,
        }

    print("  Tuning XGBoost with Optuna...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'scale_pos_weight': pos_weight,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'verbosity': 0,
            'random_state': Config.RANDOM_STATE,
            'n_jobs': -1,
        }

        f1_scores = []
        smote = SMOTE(sampling_strategy=Config.SMOTE_RATIO,
                       k_neighbors=Config.SMOTE_K, random_state=Config.RANDOM_STATE)

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Apply SMOTE on train only
            try:
                X_res, y_res = smote.fit_resample(X_train, y_train)
            except Exception:
                X_res, y_res = X_train, y_train

            clf = XGBClassifier(**params)
            clf.fit(X_res, y_res)
            preds = clf.predict(X_val)
            f1_scores.append(f1_score(y_val, preds, zero_division=0))

        return np.mean(f1_scores)

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=Config.RANDOM_STATE))
    study.optimize(objective, n_trials=Config.N_TRIALS, timeout=Config.OPTUNA_TIMEOUT)

    best = study.best_params
    best['scale_pos_weight'] = pos_weight
    best['use_label_encoder'] = False
    best['eval_metric'] = 'logloss'
    best['verbosity'] = 0
    best['random_state'] = Config.RANDOM_STATE
    best['n_jobs'] = -1

    print(f"  Best F1: {study.best_value:.4f}")
    print(f"  Best params: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()}, indent=4)}")

    return best


# ======================================================================
# Evaluation Report Generator
# ======================================================================
class EvaluationReport:
    """Generates comprehensive evaluation report with graphs."""

    def __init__(self, y_true, y_prob, y_pred, feature_names, model, X, df_original):
        self.y_true = y_true
        self.y_prob = y_prob
        self.y_pred = y_pred
        self.feature_names = feature_names
        self.model = model
        self.X = X
        self.df = df_original
        self.C = Config.COLORS
        os.makedirs(Config.EVAL_DIR, exist_ok=True)

    def generate_all(self):
        """Generate all evaluation outputs."""
        print("\n  Generating evaluation report...")
        self._plot_confusion_matrix()
        self._plot_roc_pr_curves()
        self._plot_feature_importance()
        self._plot_risk_score_distribution()
        self._plot_threshold_analysis()
        if HAS_SHAP:
            self._plot_shap()
        self._save_risk_scores()
        self._save_classification_report()
        self._save_evaluation_summary()
        print("  [OK] All evaluation artifacts generated")

    def _plot_confusion_matrix(self):
        """Plot confusion matrices at multiple thresholds."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Confusion Matrices at Different Thresholds', fontsize=18,
                     fontweight='bold', color='white', y=1.02)

        for idx, (name, thresh) in enumerate(Config.THRESHOLDS.items()):
            ax = axes[idx]
            y_t = (self.y_prob >= thresh).astype(int)
            cm = confusion_matrix(self.y_true, y_t)

            im = ax.imshow(cm, cmap='magma', aspect='auto')
            labels = ['Legitimate', 'Suspicious']
            for i in range(2):
                for j in range(2):
                    color = 'white' if cm[i, j] < cm.max() / 2 else 'black'
                    ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                            fontsize=22, fontweight='bold', color=color)

            f1 = f1_score(self.y_true, y_t, zero_division=0)
            p = precision_score(self.y_true, y_t, zero_division=0)
            r = recall_score(self.y_true, y_t, zero_division=0)
            ax.set_title(f'{name} (t={thresh})\nF1={f1:.3f} | P={p:.3f} | R={r:.3f}',
                         fontsize=12, color='white')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Pred: Legit', 'Pred: Susp'], fontsize=10)
            ax.set_yticklabels(['True: Legit', 'True: Susp'], fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{Config.EVAL_DIR}/confusion_matrices.png', dpi=150,
                    bbox_inches='tight', facecolor=self.C['bg'])
        plt.close()
        print("    [OK] confusion_matrices.png")

    def _plot_roc_pr_curves(self):
        """Plot ROC and Precision-Recall curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('Model Performance Curves', fontsize=18, fontweight='bold', color='white')

        # ROC
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        auc = roc_auc_score(self.y_true, self.y_prob)
        ax1.plot(fpr, tpr, color=self.C['primary'], linewidth=2.5, label=f'AUC = {auc:.4f}')
        ax1.plot([0, 1], [0, 1], 'w--', alpha=0.3, linewidth=1)
        ax1.fill_between(fpr, tpr, alpha=0.15, color=self.C['primary'])
        ax1.set_xlabel('False Positive Rate', fontsize=13)
        ax1.set_ylabel('True Positive Rate', fontsize=13)
        ax1.set_title('ROC Curve', fontsize=14, color='white')
        ax1.legend(fontsize=13, loc='lower right')
        ax1.set_facecolor(self.C['bg'])
        ax1.grid(True, alpha=0.15)

        # PR
        prec, rec, threshs = precision_recall_curve(self.y_true, self.y_prob)
        ap = average_precision_score(self.y_true, self.y_prob)
        ax2.plot(rec, prec, color=self.C['secondary'], linewidth=2.5, label=f'AP = {ap:.4f}')
        ax2.fill_between(rec, prec, alpha=0.15, color=self.C['secondary'])
        # Mark thresholds
        for name, thresh in Config.THRESHOLDS.items():
            idx = np.argmin(np.abs(threshs - thresh))
            ax2.scatter([rec[idx]], [prec[idx]], s=120, zorder=5,
                        label=f'{name} (t={thresh})')
        ax2.set_xlabel('Recall', fontsize=13)
        ax2.set_ylabel('Precision', fontsize=13)
        ax2.set_title('Precision-Recall Curve', fontsize=14, color='white')
        ax2.legend(fontsize=10, loc='upper right')
        ax2.set_facecolor(self.C['bg'])
        ax2.grid(True, alpha=0.15)

        plt.tight_layout()
        plt.savefig(f'{Config.EVAL_DIR}/roc_pr_curves.png', dpi=150,
                    bbox_inches='tight', facecolor=self.C['bg'])
        plt.close()
        print("    [OK] roc_pr_curves.png")

    def _plot_feature_importance(self):
        """Plot XGBoost feature importance."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Feature Importance Analysis', fontsize=18, fontweight='bold', color='white')

        # Gain importance
        importance = self.model.feature_importances_
        sorted_idx = np.argsort(importance)
        colors = [self.C['primary'] if self.feature_names[i] in Config.RAW_FEATURES
                  else self.C['accent'] for i in sorted_idx]

        ax1.barh(range(len(sorted_idx)), importance[sorted_idx], color=colors, alpha=0.85)
        ax1.set_yticks(range(len(sorted_idx)))
        ax1.set_yticklabels([self.feature_names[i] for i in sorted_idx], fontsize=11)
        ax1.set_xlabel('Feature Importance (Gain)', fontsize=13)
        ax1.set_title('XGBoost Feature Importance', fontsize=14, color='white')
        ax1.set_facecolor(self.C['bg'])

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.C['primary'], alpha=0.85, label='Raw Features'),
            Patch(facecolor=self.C['accent'], alpha=0.85, label='Engineered Features')
        ]
        ax1.legend(handles=legend_elements, fontsize=10, loc='lower right')

        # Feature values: suspicious vs legitimate
        susp_mask = self.y_true == 1
        feat_diff = []
        for i, fname in enumerate(self.feature_names):
            susp_mean = np.nanmean(self.X[susp_mask, i])
            legit_mean = np.nanmean(self.X[~susp_mask, i])
            if legit_mean != 0:
                diff = (susp_mean - legit_mean) / (abs(legit_mean) + 1e-10)
            else:
                diff = susp_mean
            feat_diff.append(diff)

        sorted_diff = np.argsort(np.abs(feat_diff))
        colors2 = [self.C['secondary'] if feat_diff[i] > 0 else self.C['primary'] for i in sorted_diff]
        ax2.barh(range(len(sorted_diff)), [feat_diff[i] for i in sorted_diff], color=colors2, alpha=0.85)
        ax2.set_yticks(range(len(sorted_diff)))
        ax2.set_yticklabels([self.feature_names[i] for i in sorted_diff], fontsize=11)
        ax2.set_xlabel('Relative Difference (Suspicious vs Legitimate)', fontsize=12)
        ax2.set_title('Feature Direction', fontsize=14, color='white')
        ax2.axvline(x=0, color='white', linewidth=0.5, alpha=0.5)
        ax2.set_facecolor(self.C['bg'])
        legend2 = [
            Patch(facecolor=self.C['secondary'], alpha=0.85, label='Higher in Suspicious'),
            Patch(facecolor=self.C['primary'], alpha=0.85, label='Lower in Suspicious')
        ]
        ax2.legend(handles=legend2, fontsize=10, loc='lower right')

        plt.tight_layout()
        plt.savefig(f'{Config.EVAL_DIR}/feature_importance.png', dpi=150,
                    bbox_inches='tight', facecolor=self.C['bg'])
        plt.close()
        print("    [OK] feature_importance.png")

    def _plot_risk_score_distribution(self):
        """Plot distribution of risk scores."""
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        fig.suptitle('Risk Score Distribution', fontsize=18, fontweight='bold', color='white')

        susp_mask = self.y_true == 1

        # Histogram
        ax = axes[0]
        ax.hist(self.y_prob[~susp_mask], bins=50, alpha=0.6, color=self.C['secondary'],
                label=f'Legitimate (n={int((~susp_mask).sum())})', density=True)
        ax.hist(self.y_prob[susp_mask], bins=30, alpha=0.7, color=self.C['primary'],
                label=f'Suspicious (n={int(susp_mask.sum())})', density=True)
        for name, thresh in Config.THRESHOLDS.items():
            ax.axvline(x=thresh, linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'{name}={thresh}')
        ax.set_xlabel('Risk Score (Predicted Probability)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Score Distribution by Class', fontsize=14, color='white')
        ax.legend(fontsize=8)
        ax.set_facecolor(self.C['bg'])

        # Suspicious detail
        ax = axes[1]
        ax.hist(self.y_prob[susp_mask], bins=20, color=self.C['primary'], alpha=0.85, edgecolor='white')
        ax.axvline(x=0.5, color=self.C['warn'], linestyle='--', linewidth=2, label='t=0.5')
        ax.set_xlabel('Risk Score', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Suspicious Account Scores (n={int(susp_mask.sum())})', fontsize=14, color='white')
        caught_50 = (self.y_prob[susp_mask] >= 0.5).sum()
        ax.text(0.05, 0.95, f'Caught at t=0.5: {caught_50}/{int(susp_mask.sum())}',
                transform=ax.transAxes, fontsize=11, color=self.C['warn'],
                verticalalignment='top')
        ax.legend(fontsize=10)
        ax.set_facecolor(self.C['bg'])

        # Risk tier breakdown
        ax = axes[2]
        tiers = {
            'Very High\n(>0.8)': (self.y_prob >= 0.8).sum(),
            'High\n(0.5-0.8)': ((self.y_prob >= 0.5) & (self.y_prob < 0.8)).sum(),
            'Medium\n(0.2-0.5)': ((self.y_prob >= 0.2) & (self.y_prob < 0.5)).sum(),
            'Low\n(0.1-0.2)': ((self.y_prob >= 0.1) & (self.y_prob < 0.2)).sum(),
            'Minimal\n(<0.1)': (self.y_prob < 0.1).sum(),
        }
        tier_colors = [self.C['primary'], self.C['orange'], self.C['warn'],
                       self.C['blue'], self.C['secondary']]
        bars = ax.bar(range(len(tiers)), list(tiers.values()), color=tier_colors, alpha=0.85)
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels(list(tiers.keys()), fontsize=10)
        ax.set_ylabel('Number of Accounts', fontsize=12)
        ax.set_title('Risk Tier Distribution (All Accounts)', fontsize=14, color='white')
        for bar, val in zip(bars, tiers.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f'{val:,}', ha='center', fontsize=11, color='white')
        ax.set_facecolor(self.C['bg'])

        plt.tight_layout()
        plt.savefig(f'{Config.EVAL_DIR}/risk_distribution.png', dpi=150,
                    bbox_inches='tight', facecolor=self.C['bg'])
        plt.close()
        print("    [OK] risk_distribution.png")

    def _plot_threshold_analysis(self):
        """Plot detailed threshold analysis."""
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        fig.suptitle('Threshold Analysis', fontsize=18, fontweight='bold', color='white')

        threshs = np.arange(0.01, 0.99, 0.01)

        f1s, precs, recs, fps, fns, tps = [], [], [], [], [], []
        for t in threshs:
            y_t = (self.y_prob >= t).astype(int)
            f1s.append(f1_score(self.y_true, y_t, zero_division=0))
            precs.append(precision_score(self.y_true, y_t, zero_division=0))
            recs.append(recall_score(self.y_true, y_t, zero_division=0))
            fps.append(((y_t == 1) & (self.y_true == 0)).sum())
            fns.append(((y_t == 0) & (self.y_true == 1)).sum())
            tps.append(((y_t == 1) & (self.y_true == 1)).sum())

        # F1/Precision/Recall vs Threshold
        ax = axes[0]
        ax.plot(threshs, f1s, color=self.C['blue'], linewidth=2.5, label='F1')
        ax.plot(threshs, precs, color=self.C['secondary'], linewidth=1.5, alpha=0.7, label='Precision')
        ax.plot(threshs, recs, color=self.C['primary'], linewidth=1.5, alpha=0.7, label='Recall')
        best_f1_idx = np.argmax(f1s)
        ax.axvline(x=threshs[best_f1_idx], color=self.C['warn'], linestyle='--', linewidth=2,
                   label=f'Best F1 t={threshs[best_f1_idx]:.2f}')
        ax.set_xlabel('Threshold', fontsize=13)
        ax.set_ylabel('Score', fontsize=13)
        ax.set_title('Metrics vs Threshold', fontsize=14, color='white')
        ax.legend(fontsize=9)
        ax.set_facecolor(self.C['bg'])
        ax.grid(True, alpha=0.15)

        # TP/FP/FN counts
        ax = axes[1]
        ax.plot(threshs, tps, color=self.C['green'], linewidth=2, label='True Positives')
        ax.plot(threshs, fps, color=self.C['orange'], linewidth=2, label='False Positives')
        ax.plot(threshs, fns, color=self.C['primary'], linewidth=2, label='False Negatives')
        ax.set_xlabel('Threshold', fontsize=13)
        ax.set_ylabel('Count', fontsize=13)
        ax.set_title('TP/FP/FN vs Threshold', fontsize=14, color='white')
        ax.legend(fontsize=10)
        ax.set_facecolor(self.C['bg'])
        ax.grid(True, alpha=0.15)

        # Business cost curve
        ax = axes[2]
        cost_fp = 100
        cost_fn = 10000
        costs = [fp * cost_fp + fn * cost_fn for fp, fn in zip(fps, fns)]
        ax.plot(threshs, costs, color=self.C['accent'], linewidth=2.5)
        min_cost_idx = np.argmin(costs)
        ax.scatter([threshs[min_cost_idx]], [costs[min_cost_idx]],
                   color=self.C['warn'], s=150, zorder=5,
                   label=f'Min cost at t={threshs[min_cost_idx]:.2f} (${costs[min_cost_idx]:,.0f})')
        ax.set_xlabel('Threshold', fontsize=13)
        ax.set_ylabel('Total Cost ($)', fontsize=13)
        ax.set_title(f'Business Cost (FP=${cost_fp}, FN=${cost_fn:,})', fontsize=14, color='white')
        ax.legend(fontsize=10)
        ax.set_facecolor(self.C['bg'])
        ax.grid(True, alpha=0.15)

        plt.tight_layout()
        plt.savefig(f'{Config.EVAL_DIR}/threshold_analysis.png', dpi=150,
                    bbox_inches='tight', facecolor=self.C['bg'])
        plt.close()
        print("    [OK] threshold_analysis.png")

    def _plot_shap(self):
        """Generate SHAP explainability plots using KernelExplainer.
        Note: TreeExplainer is incompatible with XGBoost 3.x.
        """
        print("    Computing SHAP values (KernelExplainer)...")
        try:
            X_shap = np.asarray(self.X, dtype=np.float64)

            # Background sample for KernelExplainer
            rng = np.random.RandomState(42)
            bg_idx = rng.choice(len(X_shap), size=100, replace=False)
            background = X_shap[bg_idx]

            explainer = shap.KernelExplainer(self.model.predict_proba, background)

            # Compute SHAP for suspicious + top 50 highest-risk
            susp_idx = np.where(self.y_true == 1)[0]
            top_legit_idx = np.argsort(self.y_prob)[-50:]
            eval_idx = np.unique(np.concatenate([susp_idx, top_legit_idx]))
            X_eval = X_shap[eval_idx]

            shap_raw = explainer.shap_values(X_eval, nsamples=200)

            # Extract class-1 SHAP values (handle list or 3D array)
            if isinstance(shap_raw, list) and len(shap_raw) == 2:
                shap_vals = shap_raw[1]
            elif hasattr(shap_raw, 'ndim') and shap_raw.ndim == 3:
                shap_vals = shap_raw[:, :, 1]
            else:
                shap_vals = shap_raw

            # Summary plot
            fig, ax = plt.subplots(figsize=(14, 8))
            shap.summary_plot(shap_vals, X_eval, feature_names=self.feature_names,
                              show=False, max_display=14)
            plt.title('SHAP Feature Impact on Predictions', fontsize=16,
                      fontweight='bold', color='white', pad=15)
            plt.tight_layout()
            plt.savefig(f'{Config.EVAL_DIR}/shap_summary.png', dpi=150,
                        bbox_inches='tight', facecolor=self.C['bg'])
            plt.close()
            print("    [OK] shap_summary.png")

            # Top 5 suspicious accounts
            susp_probs = self.y_prob[susp_idx]
            top5_local = np.argsort(susp_probs)[-5:]
            top5_eval_pos = [np.where(eval_idx == susp_idx[i])[0][0] for i in top5_local]

            fig, axes = plt.subplots(5, 1, figsize=(16, 20))
            fig.suptitle('SHAP Explanations: Top 5 Highest-Risk Mule Accounts',
                         fontsize=16, fontweight='bold', color='white', y=1.01)

            for plot_idx, ev_pos in enumerate(top5_eval_pos):
                ax = axes[plot_idx]
                sv = shap_vals[ev_pos]
                sorted_feat = np.argsort(np.abs(sv))[::-1][:10]
                colors_s = [self.C['primary'] if sv[i] > 0 else self.C['secondary'] for i in sorted_feat]
                ax.barh(range(len(sorted_feat)), [sv[i] for i in sorted_feat],
                        color=colors_s, alpha=0.85)
                ax.set_yticks(range(len(sorted_feat)))
                ax.set_yticklabels([f'{self.feature_names[i]} = {X_eval[ev_pos, i]:.2f}'
                                    for i in sorted_feat], fontsize=9)
                ax.set_xlabel('SHAP Value', fontsize=10)
                orig_idx = eval_idx[ev_pos]
                ax.set_title(f'Account #{orig_idx} (Risk: {self.y_prob[orig_idx]:.4f})',
                             fontsize=11, color='white')
                ax.set_facecolor(self.C['bg'])
                ax.invert_yaxis()

            plt.tight_layout()
            plt.savefig(f'{Config.EVAL_DIR}/shap_top_accounts.png', dpi=150,
                        bbox_inches='tight', facecolor=self.C['bg'])
            plt.close()
            print("    [OK] shap_top_accounts.png")

        except Exception as e:
            print(f"    [!!] SHAP failed: {e}")
            import traceback
            traceback.print_exc()

    def _save_risk_scores(self):
        """Save risk scores for all accounts."""
        output = pd.DataFrame({
            'row_index': range(len(self.y_true)),
            'true_label': self.y_true,
            'risk_score': np.round(self.y_prob, 6),
            'prediction_t0.5': (self.y_prob >= 0.5).astype(int),
            'prediction_t0.2': (self.y_prob >= 0.2).astype(int),
            'risk_tier': pd.cut(self.y_prob, bins=[0, 0.1, 0.2, 0.5, 0.8, 1.0],
                                labels=['Minimal', 'Low', 'Medium', 'High', 'Very High'],
                                include_lowest=True)
        })

        # Add original identifiers if available
        if 'F3888' in self.df.columns:  # Account open date
            output['account_open_date'] = self.df['F3888'].values
        if 'F3894' in self.df.columns:  # Customer age
            output['customer_age'] = self.df['F3894'].values
        if 'F3887' in self.df.columns:  # Account age days
            output['account_age_days'] = self.df['F3887'].values

        output = output.sort_values('risk_score', ascending=False)
        output.to_csv(f'{Config.EVAL_DIR}/risk_scores.csv', index=False)
        print(f"    [OK] risk_scores.csv ({len(output)} accounts)")

        # Also save just the flagged accounts
        flagged = output[output['prediction_t0.2'] == 1]
        flagged.to_csv(f'{Config.EVAL_DIR}/flagged_accounts.csv', index=False)
        print(f"    [OK] flagged_accounts.csv ({len(flagged)} flagged at t=0.2)")

    def _save_classification_report(self):
        """Save text classification report."""
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("MULE ACCOUNT CLASSIFICATION - EVALUATION REPORT")
        report_lines.append("=" * 70)

        for name, thresh in Config.THRESHOLDS.items():
            y_t = (self.y_prob >= thresh).astype(int)
            report_lines.append(f"\n--- Threshold: {thresh} ({name}) ---")
            report_lines.append(classification_report(
                self.y_true, y_t, target_names=['Legitimate', 'Suspicious'],
                zero_division=0
            ))
            cm = confusion_matrix(self.y_true, y_t)
            report_lines.append(f"Confusion Matrix:")
            report_lines.append(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
            report_lines.append(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

        auc = roc_auc_score(self.y_true, self.y_prob)
        ap = average_precision_score(self.y_true, self.y_prob)
        report_lines.append(f"\n--- Overall Metrics ---")
        report_lines.append(f"ROC-AUC: {auc:.4f}")
        report_lines.append(f"Average Precision: {ap:.4f}")

        report_text = '\n'.join(report_lines)
        with open(f'{Config.EVAL_DIR}/classification_report.txt', 'w') as f:
            f.write(report_text)
        print("    [OK] classification_report.txt")

    def _save_evaluation_summary(self):
        """Save markdown evaluation summary."""
        auc = roc_auc_score(self.y_true, self.y_prob)
        ap = average_precision_score(self.y_true, self.y_prob)

        lines = []
        lines.append("# Mule Account Detection - Evaluation Summary\n")
        lines.append(f"**Model**: XGBoost (Optuna-tuned)")
        lines.append(f"**Features**: {len(self.feature_names)} ({len(Config.RAW_FEATURES)} raw + "
                      f"{len(self.feature_names) - len(Config.RAW_FEATURES)} engineered)")
        lines.append(f"**ROC-AUC**: {auc:.4f}")
        lines.append(f"**Average Precision**: {ap:.4f}\n")

        lines.append("## Performance at Different Thresholds\n")
        lines.append("| Threshold | F1 | Precision | Recall | TP | FP | FN |")
        lines.append("|-----------|-----|-----------|--------|----|----|-----|")

        for name, thresh in Config.THRESHOLDS.items():
            y_t = (self.y_prob >= thresh).astype(int)
            f1 = f1_score(self.y_true, y_t, zero_division=0)
            p = precision_score(self.y_true, y_t, zero_division=0)
            r = recall_score(self.y_true, y_t, zero_division=0)
            tp = ((y_t == 1) & (self.y_true == 1)).sum()
            fp = ((y_t == 1) & (self.y_true == 0)).sum()
            fn = ((y_t == 0) & (self.y_true == 1)).sum()
            lines.append(f"| {name} ({thresh}) | {f1:.3f} | {p:.3f} | {r:.3f} | {tp} | {fp} | {fn} |")

        lines.append(f"\n## Features Used\n")
        lines.append("### Raw Features")
        for f in Config.RAW_FEATURES:
            lines.append(f"- {f}")
        lines.append("\n### Engineered Features")
        eng = [f for f in self.feature_names if f not in Config.RAW_FEATURES]
        for f in eng:
            lines.append(f"- {f}")

        with open(f'{Config.EVAL_DIR}/evaluation_summary.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print("    [OK] evaluation_summary.md")


# ======================================================================
# Main Pipeline
# ======================================================================
def main():
    start_time = time.time()

    print("=" * 70)
    print("BOI HACKATHON - MULE ACCOUNT DETECTION PIPELINE")
    print("=" * 70)

    # --- Step 1: Load Data ---
    print("\n[1/7] Loading data...")
    df = pd.read_csv(Config.DATA_PATH, low_memory=False)
    y = df[Config.TARGET].values.astype(int)
    pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"  Loaded {len(df)} accounts, {y.sum()} suspicious ({y.mean()*100:.2f}%)")
    print(f"  Class ratio: {pos_weight:.1f}:1")

    # --- Step 2: Feature Engineering ---
    print("\n[2/7] Engineering features...")
    fe = FeatureEngineer()
    X_df = fe.fit_transform(df, y)
    X = X_df.values.astype(np.float64)
    feature_names = fe.feature_names
    print(f"  Created {len(feature_names)} features: {len(Config.RAW_FEATURES)} raw + "
          f"{len(feature_names) - len(Config.RAW_FEATURES)} engineered")
    print(f"  Features: {feature_names}")

    # --- Step 3: Hyperparameter Tuning ---
    print("\n[3/7] Hyperparameter tuning...")
    cv = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_STATE)
    best_params = tune_xgboost(X, y, cv, pos_weight)

    # --- Step 4: Cross-Validation Evaluation ---
    print("\n[4/7] Cross-validation evaluation...")
    smote = SMOTE(sampling_strategy=Config.SMOTE_RATIO,
                   k_neighbors=Config.SMOTE_K, random_state=Config.RANDOM_STATE)

    y_prob_cv = np.zeros(len(y))
    y_pred_cv = np.zeros(len(y))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # SMOTE on train
        try:
            X_res, y_res = smote.fit_resample(X_train, y_train)
        except Exception:
            X_res, y_res = X_train, y_train

        # Train
        clf = XGBClassifier(**best_params)
        clf.fit(X_res, y_res)

        # Predict
        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        y_prob_cv[val_idx] = probs
        y_pred_cv[val_idx] = preds

        f1 = f1_score(y_val, preds, zero_division=0)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        auc = roc_auc_score(y_val, probs)
        fold_metrics.append({'fold': fold + 1, 'f1': f1, 'precision': p, 'recall': r, 'auc': auc})
        print(f"  Fold {fold+1}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}, AUC={auc:.4f}")

    # Summary
    avg_f1 = np.mean([m['f1'] for m in fold_metrics])
    avg_p = np.mean([m['precision'] for m in fold_metrics])
    avg_r = np.mean([m['recall'] for m in fold_metrics])
    avg_auc = np.mean([m['auc'] for m in fold_metrics])
    print(f"\n  MEAN: F1={avg_f1:.4f}, P={avg_p:.4f}, R={avg_r:.4f}, AUC={avg_auc:.4f}")

    # Optimal threshold on CV predictions
    prec_c, rec_c, thresh_c = precision_recall_curve(y, y_prob_cv)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-10)
    best_idx = np.argmax(f1_c)
    optimal_threshold = thresh_c[best_idx] if best_idx < len(thresh_c) else 0.5
    print(f"  Optimal threshold (by F1): {optimal_threshold:.4f}")
    Config.THRESHOLDS['f1_optimal'] = round(float(optimal_threshold), 3)

    # --- Step 5: Train Final Model on ALL Data ---
    print("\n[5/7] Training final model on all data...")
    X_res_full, y_res_full = smote.fit_resample(X, y)
    print(f"  After SMOTE: {len(X_res_full)} samples ({y_res_full.sum()} suspicious)")

    final_model = XGBClassifier(**best_params)
    final_model.fit(X_res_full, y_res_full)

    # Final predictions (using CV predictions for evaluation, not train predictions)
    y_prob_final = y_prob_cv  # Use CV predictions for honest evaluation
    y_pred_final = (y_prob_final >= 0.5).astype(int)

    # --- Step 6: Generate Evaluation Report ---
    print("\n[6/7] Generating evaluation report...")
    report = EvaluationReport(y, y_prob_final, y_pred_final,
                               feature_names, final_model, X, df)
    report.generate_all()

    # --- Step 7: Save Model Artifacts ---
    print("\n[7/7] Saving model artifacts...")
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    # Save model
    joblib.dump(final_model, f'{Config.MODEL_DIR}/xgboost_model.joblib')
    print(f"  [OK] xgboost_model.joblib")

    # Save feature engineer
    joblib.dump(fe, f'{Config.MODEL_DIR}/feature_engineer.joblib')
    print(f"  [OK] feature_engineer.joblib")

    # Save config & metadata
    metadata = {
        'model_type': 'XGBoost',
        'features': feature_names,
        'raw_features': Config.RAW_FEATURES,
        'engineered_features': [f for f in feature_names if f not in Config.RAW_FEATURES],
        'n_features': len(feature_names),
        'best_params': {k: v for k, v in best_params.items()
                        if not isinstance(v, (bool, type(None))) or isinstance(v, bool)},
        'cv_metrics': {
            'f1_mean': round(avg_f1, 4),
            'precision_mean': round(avg_p, 4),
            'recall_mean': round(avg_r, 4),
            'auc_mean': round(avg_auc, 4),
        },
        'optimal_threshold': round(float(optimal_threshold), 4),
        'smote_ratio': Config.SMOTE_RATIO,
        'training_samples': len(X),
        'suspicious_count': int(y.sum()),
        'training_time_seconds': round(time.time() - start_time, 1),
    }
    with open(f'{Config.MODEL_DIR}/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  [OK] model_metadata.json")

    # --- Final Summary ---
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Model: XGBoost with {len(feature_names)} features")
    print(f"  CV Performance: F1={avg_f1:.4f}, AUC={avg_auc:.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"\n  Saved artifacts:")
    print(f"    model/xgboost_model.joblib")
    print(f"    model/feature_engineer.joblib")
    print(f"    model/model_metadata.json")
    print(f"    model/evaluation/  (6 plots + reports)")
    print(f"\n  Run 'python model/predict.py <new_data.csv>' for inference")


if __name__ == '__main__':
    main()
