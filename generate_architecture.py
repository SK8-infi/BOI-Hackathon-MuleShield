"""
MuleShield AI — System Architecture V6.
Final polish: feedback loop fully inside canvas, clean label.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import warnings
warnings.filterwarnings('ignore')

# === Colors ===
C_INPUT    = {'bg': '#E8D5F5', 'border': '#7D3C98', 'text': '#5B2C6F'}
C_PIPELINE = {'bg': '#D4EFDF', 'border': '#1E8449', 'text': '#145A32'}
C_ENGINE   = {'bg': '#D6EAF8', 'border': '#2471A3', 'text': '#1A5276'}
C_GENAI    = {'bg': '#FDEBD0', 'border': '#CA6F1E', 'text': '#935116'}
C_DASH     = {'bg': '#FADBD8', 'border': '#C0392B', 'text': '#922B21'}
C_LOOP     = {'bg': '#D5F5E3', 'border': '#27AE60', 'text': '#1E8449'}
C_FED      = {'bg': '#E8DAEF', 'border': '#8E44AD', 'text': '#6C3483'}

BOI_BLUE = '#004B87'
DARK     = '#1C1C1C'
MED_GREY = '#555555'
WHITE    = '#FFFFFF'
ARROW_COL= '#333333'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Helvetica'],
    'figure.facecolor': WHITE,
})

# Wider canvas to fit feedback loop label
fig, ax = plt.subplots(1, 1, figsize=(19, 14.5), dpi=300, facecolor=WHITE)
ax.set_xlim(0, 19)
ax.set_ylim(0, 14.5)
ax.axis('off')
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)


def section(x, y, w, h, title, color, zorder=1):
    bg = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.08",
                        facecolor=color['bg'], edgecolor=color['border'],
                        linewidth=2.0, alpha=0.85, zorder=zorder)
    ax.add_patch(bg)
    hdr = FancyBboxPatch((x + 0.02, y + h - 0.42), w - 0.04, 0.38,
                         boxstyle="round,pad=0.05",
                         facecolor=color['border'], edgecolor='none',
                         alpha=0.85, zorder=zorder + 1)
    ax.add_patch(hdr)
    ax.text(x + w/2, y + h - 0.23, title, ha='center', va='center',
            fontsize=10, fontweight='bold', color=WHITE, zorder=zorder + 2)


def subbox(x, y, w, h, title, lines, color, title_fs=8.5, line_fs=7, zorder=3):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.05",
                       facecolor=WHITE, edgecolor=color['border'],
                       linewidth=1.3, alpha=0.95, zorder=zorder)
    ax.add_patch(b)
    ax.text(x + w/2, y + h - 0.22, title, ha='center', va='center',
            fontsize=title_fs, fontweight='bold', color=color['text'],
            zorder=zorder + 1)
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 0.48 - i*0.22, line,
                ha='center', va='center', fontsize=line_fs,
                color=MED_GREY, zorder=zorder + 1)


def arrow(x1, y1, x2, y2, color=ARROW_COL, lw=1.5, zorder=5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                               shrinkA=2, shrinkB=2),
                zorder=zorder)

def wide_arrow(x1, y1, x2, y2, color=ARROW_COL, lw=2.5, zorder=5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                               mutation_scale=18, shrinkA=3, shrinkB=3),
                zorder=zorder)


# ══════════════════════════════════════
# TITLE
# ══════════════════════════════════════
ax.text(9.5, 14.1, 'MuleShield AI — System Architecture',
        ha='center', fontsize=22, fontweight='bold', color=BOI_BLUE, zorder=10)
ax.text(9.5, 13.7, 'End-to-End Mule Account Detection Platform  ·  PSB Hackathon 2026',
        ha='center', fontsize=10.5, color=MED_GREY, zorder=10)
ax.plot([3, 16], [13.48, 13.48], color='#F2C83C', linewidth=3, alpha=0.6, zorder=5)


# ══════════════════════════════════════
# ROW 1: DATA INPUT + PIPELINE + FEDERATED + LEGEND
# ══════════════════════════════════════
section(0.3, 10.9, 3.8, 2.3, 'DATA INPUT', C_INPUT)

subbox(0.5, 11.85, 3.4, 0.95, 'Bank CBS / Core Banking',
       ['Real-time transaction feeds', '9,082 accounts · 3,925 features'], C_INPUT)
subbox(0.5, 11.05, 3.4, 0.72, 'External Sources',
       ['I4C Registry · KYC/CKYC'], C_INPUT)

section(4.5, 10.9, 5.6, 2.3, 'DATA PIPELINE (TEE Enclave)', C_PIPELINE)

subbox(4.7, 11.8, 2.5, 1.0, 'Feature Store',
       ['60 engineered features', '3,925 → 60 via MI + FFS'], C_PIPELINE, 8)
subbox(7.45, 11.8, 2.45, 1.0, 'Data Validation',
       ['Bracket cleanup', 'Type coercion · NaN handling'], C_PIPELINE, 8)
subbox(4.7, 11.05, 5.2, 0.65, 'Feature Groups: BASE(8) + RATIO(6) + DOMAIN(16) + INTERACTION(30)',
       [], C_PIPELINE, 7)

section(10.5, 10.9, 4.2, 2.3, 'FEDERATED LEARNING', C_FED)

subbox(10.7, 11.75, 3.8, 1.05, 'Aggregation Server (TEE)',
       ['FedAvg / FedProx', 'Differential Privacy (ε=1.0)', 'Secure aggregation'], C_FED, 8)
subbox(10.7, 11.05, 3.8, 0.62, 'Privacy: Gradients only — no raw data shared',
       [], C_FED, 7)

# --- LEGEND ---
section(15.1, 10.9, 3.2, 2.3, 'LEGEND', {'bg': '#F5F5F5', 'border': '#888888', 'text': '#333333'})

legend_data = [
    (ARROW_COL,        '-',  'Data / signal flow'),
    (C_LOOP['border'], '--', 'Feedback loop'),
]
legend_sq = [
    (C_INPUT['border'],    'Input'),
    (C_PIPELINE['border'], 'Pipeline'),
    (C_ENGINE['border'],   'ML Engine'),
    (C_GENAI['border'],    'GenAI'),
    (C_DASH['border'],     'Dashboard'),
]

for i, (col, ls, desc) in enumerate(legend_data):
    yy = 12.65 - i * 0.30
    ax.plot([15.35, 15.95], [yy, yy], color=col, linewidth=2.5,
            linestyle=ls, zorder=6)
    ax.annotate('', xy=(16.0, yy), xytext=(15.85, yy),
                arrowprops=dict(arrowstyle='-|>', color=col, lw=2), zorder=7)
    ax.text(16.2, yy, desc, fontsize=7, color=DARK, va='center', zorder=6)

for i, (col, desc) in enumerate(legend_sq):
    yy = 12.0 - i * 0.24
    sq = Rectangle((15.38, yy - 0.08), 0.16, 0.16,
                    facecolor=col, edgecolor='none', zorder=6)
    ax.add_patch(sq)
    ax.text(15.7, yy, desc, fontsize=7, color=DARK, va='center', zorder=6)

# Arrows between top sections
wide_arrow(4.1, 12.05, 4.5, 12.05, ARROW_COL)
wide_arrow(10.1, 12.05, 10.5, 12.05, C_FED['border'])


# ══════════════════════════════════════
# ROW 2: ML DETECTION ENGINE
# ══════════════════════════════════════
section(0.3, 8.0, 17.6, 2.55, 'ML DETECTION ENGINE', C_ENGINE)

subbox(0.5, 8.45, 3.2, 1.65, 'Model Carousel',
       ['Model A: XGBoost Blend (Active)', 'Model B: Retrained (Shadow)',
        'Model C: Fed-Tuned (Candidate)', 'Auto-promotion when F1↑ > 2%'], C_ENGINE, 9)

subbox(3.95, 8.45, 3.1, 1.65, 'Feature Engineering',
       ['3,925 → 60 features', 'MI ranking + correlation filter',
        'Ratio · domain · interaction', 'Missing count as signal'], C_ENGINE, 9)

subbox(7.3, 8.45, 3.2, 1.65, 'Training Pipeline',
       ['Optuna (100 trials)', 'Stratified 5-fold CV',
        'SMOTE(k=3, ratio=0.3)', 'Scale_pos_weight = 111'], C_ENGINE, 9)

subbox(10.75, 8.45, 3.2, 1.65, 'Real-Time Scoring',
       ['XGBoost Ensemble', 'Threshold t = 0.483',
        'F1 = 0.743 · AUC = 0.982', 'Precision 82% · Recall 68%'], C_ENGINE, 9)

subbox(14.2, 8.45, 3.45, 1.65, 'Threshold Optimization',
       ['Exhaustive sweep 0.01–0.99', 'Cost-aware objective',
        'Tiered: t=0.2 (alert) / 0.48 (flag)', 'FP rate: 0.13%'], C_ENGINE, 9)

arrow(3.7, 9.28, 3.95, 9.28, C_ENGINE['border'])
arrow(7.05, 9.28, 7.3, 9.28, C_ENGINE['border'])
arrow(10.5, 9.28, 10.75, 9.28, C_ENGINE['border'])
arrow(13.95, 9.28, 14.2, 9.28, C_ENGINE['border'])

wide_arrow(7.5, 10.9, 7.5, 10.55, ARROW_COL)
wide_arrow(12.6, 10.9, 12.6, 10.55, C_FED['border'])


# ══════════════════════════════════════
# ROW 3: GenAI INTELLIGENCE
# ══════════════════════════════════════
section(0.3, 5.0, 17.6, 2.65, 'GenAI INTELLIGENCE LAYER', C_GENAI)

subbox(0.5, 5.45, 3.3, 1.75, 'GenAI Explainability',
       ['SHAP → Natural Language', 'Per-feature attributions',
        '"Zero activity for 6+ months —',
        '3.2× more common in mules"'], C_GENAI, 9)

subbox(4.05, 5.45, 3.3, 1.75, 'Investigation Copilot',
       ['Multi-Agent LLM system', 'Auto-aggregation of evidence',
        'Similar case matching (89%)',
        'Action recommendations'], C_GENAI, 9)

subbox(7.6, 5.45, 3.3, 1.75, 'SAR Generator',
       ['RAG + LLM pipeline', 'FIU-IND / RBI compliant drafts',
        'Every claim traceable to data',
        'Cuts report time 90%'], C_GENAI, 9)

subbox(11.15, 5.45, 3.2, 1.75, 'RAG Knowledge Base',
       ['RBI AML guidelines', 'FIU-IND STR templates',
        'Historical confirmed cases',
        'SHAP feature attributions'], C_GENAI, 9)

subbox(14.6, 5.45, 3.1, 1.75, 'Fairness & Audit',
       ['SHAP audit trail', 'Bias detection (PSI)',
        'Human-in-the-loop',
        'FREE-AI framework'], C_GENAI, 9)

wide_arrow(9.5, 8.0, 9.5, 7.65, ARROW_COL)


# ══════════════════════════════════════
# ROW 4: ANALYST DASHBOARD + SELF-IMPROVING
# ══════════════════════════════════════
section(0.3, 1.8, 10.0, 2.85, 'ANALYST DASHBOARD', C_DASH)

subbox(0.5, 2.3, 2.2, 1.88, 'Risk Heatmap',
       ['Geographic view', 'Temporal patterns',
        'Cluster visualization', 'Drill-down'], C_DASH, 8.5)

subbox(2.9, 2.3, 2.2, 1.88, 'Alert Queue',
       ['Priority-sorted', 'Risk score ranked',
        'Batch actions', 'SLA tracking'], C_DASH, 8.5)

subbox(5.3, 2.3, 2.45, 1.88, 'Account Deep Dive',
       ['SHAP waterfall plot', 'Transaction timeline',
        'Peer comparison', 'GenAI narrative'], C_DASH, 8.5)

subbox(7.95, 2.3, 2.15, 1.88, 'Feedback Interface',
       ['✓ True Positive', '✗ False Positive',
        '? Needs More Info', '→ feeds retrain loop'], C_DASH, 8.5)

section(10.7, 1.8, 7.2, 2.85, 'SELF-IMPROVING ENGINE', C_LOOP)

subbox(10.9, 2.85, 3.3, 1.32, 'Continuous Learning',
       ['Analyst feedback → labels', 'Label propagation to peers',
        'Auto-retrain trigger'], C_LOOP, 8.5)

subbox(14.45, 2.85, 3.25, 1.32, 'Drift & Promotion',
       ['PSI + KL-divergence', 'Concept drift detection',
        'Shadow → prod if F1↑ > 2%'], C_LOOP, 8.5)

subbox(10.9, 2.0, 6.8, 0.75,
       'I4C Suspect Registry sync · Model Carousel auto-promotion · Quarterly audit',
       [], C_LOOP, 7.5)

wide_arrow(5.3, 5.0, 5.3, 4.65, ARROW_COL)

# Dashboard → Self-Improving
wide_arrow(10.3, 3.25, 10.7, 3.25, '#D4AC0D', lw=2.5)

# --- FEEDBACK LOOP (U-shaped dashed path) ---
loop_x = 17.95  # keep inside content area
loop_col = '#1E8449'  # strong green
loop_lw = 2.8

# Small background strip for the loop path area
loop_bg = FancyBboxPatch((17.5, 3.0), 1.1, 7.5,
                         boxstyle="round,pad=0.05",
                         facecolor='#F0FFF0', edgecolor='none',
                         alpha=0.4, zorder=0)
ax.add_patch(loop_bg)

# 1. Bottom horizontal: Self-Improving right → loop column
ax.plot([17.7, loop_x], [3.5, 3.5],
        color=loop_col, linewidth=loop_lw, linestyle='--', alpha=0.85, zorder=5)

# 2. Vertical: bottom → top  
ax.plot([loop_x, loop_x], [3.5, 10.0],
        color=loop_col, linewidth=loop_lw, linestyle='--', alpha=0.85, zorder=5)

# 3. Top horizontal with arrowhead → into ML Engine
ax.annotate('', xy=(17.65, 10.0), xytext=(loop_x, 10.0),
            arrowprops=dict(arrowstyle='-|>', color=loop_col,
                           lw=loop_lw, linestyle='dashed',
                           mutation_scale=18), zorder=6)

# Label alongside the vertical line
ax.text(18.35, 6.75, 'Feedback', fontsize=8.5, color='#145A32',
        fontweight='bold', ha='center', va='center',
        rotation=90, zorder=6)
ax.text(18.55, 6.75, 'Loop', fontsize=8.5, color='#145A32',
        fontweight='bold', ha='center', va='center',
        rotation=90, zorder=6)


# ══════════════════════════════════════
# BOTTOM BANNER: Key Metrics
# ══════════════════════════════════════
banner = FancyBboxPatch((0.3, 0.3), 17.6, 1.10,
                        boxstyle="round,pad=0.08",
                        facecolor='#F8F9FA', edgecolor='#DEE2E6',
                        linewidth=1.5, zorder=2)
ax.add_patch(banner)

metrics = [
    ('F1 Score', '0.743'),
    ('AUC-ROC', '0.982'),
    ('Precision', '82.1%'),
    ('Recall', '67.9%'),
    ('FP Rate', '0.13%'),
    ('Features', '3,925→60'),
    ('Accounts', '9,082'),
    ('Mules Found', '55/81'),
]

for i, (label, value) in enumerate(metrics):
    mx = 1.5 + i * 2.15
    ax.text(mx, 1.03, value, ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=BOI_BLUE, zorder=5)
    ax.text(mx, 0.65, label, ha='center', va='center',
            fontsize=7, color=MED_GREY, zorder=5)

for i in range(len(metrics) - 1):
    mx = 1.5 + i * 2.15 + 1.075
    ax.plot([mx, mx], [0.55, 1.20], color='#DEE2E6', linewidth=1, zorder=4)


# ══════════════════════════════════════
# SAVE
# ══════════════════════════════════════
out = r"c:\Github\BOI Hackathon\files\system_architecture.png"
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=WHITE, pad_inches=0.15)
plt.close()
print(f"Saved: {out}")
