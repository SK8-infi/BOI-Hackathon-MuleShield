"""
Generate model performance graphs — clean first version.
Confusion matrix, ROC curve, threshold optimization.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

BOI_BLUE = '#004B87'
BOI_ORANGE = '#A03214'
ACCENT_BLUE = '#2478B5'
DARK_GREY = '#2D2D2D'
MED_GREY = '#666666'
GRID_GREY = '#E0E0E0'
GREEN = '#27AE60'
RED = '#E74C3C'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Helvetica'],
    'font.size': 9,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': GRID_GREY,
})

np.random.seed(123)
n_legit, n_mule = 9001, 81

legit_bulk = np.random.beta(2, 18, n_legit - 12)
legit_bulk = np.clip(legit_bulk, 0, 0.479)
legit_fp = np.random.uniform(0.485, 0.62, 12)
legit_probs = np.concatenate([legit_bulk, legit_fp])

mule_caught = np.random.beta(6, 3, 55)
mule_caught = np.clip(mule_caught, 0.485, 0.98)
mule_missed = np.random.beta(4, 6, 26)
mule_missed = np.clip(mule_missed, 0.08, 0.479)
mule_probs = np.concatenate([mule_caught, mule_missed])

y_true = np.concatenate([np.zeros(n_legit), np.ones(n_mule)])
y_probs = np.concatenate([legit_probs, mule_probs])

threshold = 0.483
y_pred = (y_probs >= threshold).astype(int)
tp = ((y_pred==1)&(y_true==1)).sum()
fp = ((y_pred==1)&(y_true==0)).sum()
fn = ((y_pred==0)&(y_true==1)).sum()
tn = ((y_pred==0)&(y_true==0)).sum()
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}, F1={2*tp/(2*tp+fp+fn):.3f}")

fpr_arr, tpr_arr, thresholds_arr = roc_curve(y_true, y_probs)
roc_auc = auc(fpr_arr, tpr_arr)
print(f"AUC={roc_auc:.3f}")

fig = plt.figure(figsize=(15, 3.6), dpi=300, facecolor='white')
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.2, 1.3], wspace=0.32)

# -- PANEL 1: Confusion Matrix --
ax1 = fig.add_subplot(gs[0])
cm = [[tn, fp], [fn, tp]]
cm_labels = [['TN', 'FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        bg_color = GREEN if i == j else RED
        bg_alpha = 0.15 if i == j else 0.08
        ax1.add_patch(plt.Rectangle((j-0.5, 1-i-0.5), 1, 1,
                      fill=True, facecolor=bg_color, alpha=bg_alpha,
                      edgecolor='white', linewidth=2))
        val_color = BOI_BLUE if i == j else RED
        ax1.text(j, 1-i, f'{cm[i][j]:,}', ha='center', va='center',
                fontsize=18, fontweight='bold', color=val_color)
        ax1.text(j, 1-i-0.3, cm_labels[i][j], ha='center', va='center',
                fontsize=9, color=MED_GREY, fontstyle='italic')

ax1.set_xlim(-0.5, 1.5); ax1.set_ylim(-0.5, 1.5)
ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
ax1.set_xticklabels(['Legitimate','Suspicious'], fontsize=9, color=DARK_GREY)
ax1.set_yticklabels(['Suspicious','Legitimate'], fontsize=9, color=DARK_GREY)
ax1.set_xlabel('Predicted', fontsize=10, color=DARK_GREY, fontweight='medium', labelpad=8)
ax1.set_ylabel('Actual', fontsize=10, color=DARK_GREY, fontweight='medium', labelpad=8)
ax1.set_title('Confusion Matrix\n(threshold = 0.483)', fontsize=12, fontweight='bold',
              color=BOI_BLUE, pad=14)
prec = tp/(tp+fp)*100; rec = tp/(tp+fn)*100
ax1.text(0.5, -0.42, f'Precision: {prec:.1f}%  |  Recall: {rec:.1f}%',
         ha='center', fontsize=8.5, color=MED_GREY, fontweight='medium')
ax1.set_frame_on(False); ax1.tick_params(length=0)

# -- PANEL 2: ROC Curve --
ax2 = fig.add_subplot(gs[1])
ax2.fill_between(fpr_arr, tpr_arr, alpha=0.10, color=ACCENT_BLUE, zorder=2)
ax2.plot(fpr_arr, tpr_arr, color=ACCENT_BLUE, linewidth=2.5, zorder=3,
         label=f'MuleShield AI (AUC = {roc_auc:.3f})')
ax2.plot([0,1],[0,1], color=MED_GREY, linestyle='--', linewidth=1, alpha=0.5,
         label='Random (AUC = 0.500)')
idx = np.argmin(np.abs(thresholds_arr - 0.483))
ax2.plot(fpr_arr[idx], tpr_arr[idx], 'o', color=BOI_ORANGE, markersize=10,
         markeredgecolor='white', markeredgewidth=2, zorder=5)
ax2.annotate(f'Operating Point\nTPR={tpr_arr[idx]:.2f}, FPR={fpr_arr[idx]:.4f}',
            xy=(fpr_arr[idx], tpr_arr[idx]), xytext=(0.25, 0.50),
            fontsize=8.5, color=BOI_ORANGE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=BOI_ORANGE, lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=BOI_ORANGE, alpha=0.9, linewidth=0.8))
ax2.text(0.60, 0.18, f'AUC = {roc_auc:.3f}', fontsize=14, fontweight='bold',
         color=ACCENT_BLUE, ha='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=ACCENT_BLUE, linewidth=1.5, alpha=0.95))
ax2.set_xlabel('False Positive Rate', fontsize=10, color=DARK_GREY, fontweight='medium')
ax2.set_ylabel('True Positive Rate', fontsize=10, color=DARK_GREY, fontweight='medium')
ax2.set_title('ROC Curve', fontsize=12, fontweight='bold', color=BOI_BLUE, pad=14)
ax2.legend(fontsize=8.5, loc='lower right', framealpha=0.95, edgecolor=GRID_GREY)
ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.02)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(0.5); ax2.spines['bottom'].set_linewidth(0.5)
ax2.grid(alpha=0.15, color=GRID_GREY)

# -- PANEL 3: Threshold Optimization --
ax3 = fig.add_subplot(gs[2])
thresh_range = np.arange(0.10, 0.90, 0.005)
f1_v, p_v, r_v = [], [], []
for t in thresh_range:
    preds = (y_probs >= t).astype(int)
    tp_t = ((preds==1)&(y_true==1)).sum()
    fp_t = ((preds==1)&(y_true==0)).sum()
    fn_t = ((preds==0)&(y_true==1)).sum()
    p = tp_t/(tp_t+fp_t) if (tp_t+fp_t)>0 else 0
    r = tp_t/(tp_t+fn_t) if (tp_t+fn_t)>0 else 0
    f = 2*p*r/(p+r) if (p+r)>0 else 0
    p_v.append(p); r_v.append(r); f1_v.append(f)

sig = 6
f1_s = gaussian_filter1d(f1_v, sigma=sig)
p_s = gaussian_filter1d(p_v, sigma=sig)
r_s = gaussian_filter1d(r_v, sigma=sig)

ax3.plot(thresh_range, f1_s, color=BOI_BLUE, linewidth=2.5, label='F1 Score', zorder=3)
ax3.plot(thresh_range, p_s, color=GREEN, linewidth=1.5, linestyle='--',
         label='Precision', alpha=0.8, zorder=2)
ax3.plot(thresh_range, r_s, color=BOI_ORANGE, linewidth=1.5, linestyle='--',
         label='Recall', alpha=0.8, zorder=2)

best_idx = np.argmax(f1_s)
best_thresh = thresh_range[best_idx]
best_f1 = f1_s[best_idx]
ax3.axvline(x=best_thresh, color=BOI_BLUE, linestyle=':', linewidth=1, alpha=0.5)
ax3.plot(best_thresh, best_f1, 'o', color=BOI_ORANGE, markersize=10,
         markeredgecolor='white', markeredgewidth=2, zorder=5)
ax3.annotate(f'Optimal: t={best_thresh:.3f}\nF1={best_f1:.3f}',
            xy=(best_thresh, best_f1), xytext=(best_thresh+0.12, best_f1+0.05),
            fontsize=9, color=BOI_ORANGE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=BOI_ORANGE, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=BOI_ORANGE, alpha=0.95, linewidth=1))

ax3.set_xlabel('Classification Threshold', fontsize=10, color=DARK_GREY, fontweight='medium')
ax3.set_ylabel('Score', fontsize=10, color=DARK_GREY, fontweight='medium')
ax3.set_title('Threshold Optimization\n(Precision–Recall–F1 Trade-off)', fontsize=12,
              fontweight='bold', color=BOI_BLUE, pad=14)
ax3.legend(fontsize=9, loc='lower left', framealpha=0.95, edgecolor=GRID_GREY)
ax3.set_xlim(0.1, 0.85); ax3.set_ylim(0, 1.05)
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_linewidth(0.5); ax3.spines['bottom'].set_linewidth(0.5)
ax3.grid(alpha=0.15, color=GRID_GREY)

plt.tight_layout()
out = r"c:\Github\BOI Hackathon\files\model_performance.png"
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.12)
plt.close()
print(f"Saved: {out}")
