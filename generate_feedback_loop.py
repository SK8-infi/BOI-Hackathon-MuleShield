"""
MuleShield AI — Self-Improving Feedback Loop Diagram (Clean version)
Just headings, no detail bullets. Clean cycle layout.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── COLORS ──
BLUE_DARK   = '#004B87'
BLUE_MED    = '#2471A3'
BLUE_LIGHT  = '#D6EAF8'
GREEN_DARK  = '#1E8449'
GREEN_MED   = '#27AE60'
GREEN_LIGHT = '#D5F5E3'
ORANGE_DARK = '#A03214'
ORANGE_MED  = '#CA6F1E'
ORANGE_LIGHT= '#FDEBD0'
PURPLE_DARK = '#6C3483'
PURPLE_MED  = '#8E44AD'
PURPLE_LIGHT= '#E8D5F5'
RED_DARK    = '#922B21'
RED_MED     = '#C0392B'
RED_LIGHT   = '#FADBD8'
TEAL_DARK   = '#0E6655'
TEAL_MED    = '#148F77'
TEAL_LIGHT  = '#D1F2EB'
GOLD        = '#F2C83C'
WHITE       = '#FFFFFF'
DARK        = '#1C1C1C'
MED_GREY    = '#555555'
LIGHT_GREY  = '#F8F9FA'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Helvetica'],
    'figure.facecolor': WHITE,
})

fig, ax = plt.subplots(1, 1, figsize=(14, 9), dpi=300, facecolor=WHITE)
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)


def draw_block(x, y, w, h, title, subtitle, bg_color, border_color, text_color, zorder=3):
    """Draw a clean block with title and optional one-line subtitle."""
    # Shadow
    shadow = FancyBboxPatch((x + 0.05, y - 0.05), w, h,
                            boxstyle="round,pad=0.10",
                            facecolor='#00000012', edgecolor='none',
                            zorder=zorder - 1)
    ax.add_patch(shadow)
    # Main box
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.10",
                         facecolor=bg_color, edgecolor=border_color,
                         linewidth=2.2, zorder=zorder)
    ax.add_patch(box)
    # Title
    if subtitle:
        ax.text(x + w/2, y + h/2 + 0.15, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color=text_color, zorder=zorder + 2)
        ax.text(x + w/2, y + h/2 - 0.18, subtitle, ha='center', va='center',
                fontsize=7.5, color=MED_GREY, zorder=zorder + 2)
    else:
        ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color=text_color, zorder=zorder + 2)


def flow_arrow(x1, y1, x2, y2, color='#333333', lw=2.5, connectionstyle=None, zorder=6):
    """Draw a flow arrow."""
    props = dict(arrowstyle='-|>', color=color, lw=lw,
                 mutation_scale=18, shrinkA=6, shrinkB=6)
    if connectionstyle:
        props['connectionstyle'] = connectionstyle
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=props, zorder=zorder)


def step_number(x, y, num, color, zorder=8):
    """Draw a numbered step circle."""
    c = Circle((x, y), 0.24, facecolor=color, edgecolor=WHITE,
               linewidth=2.5, zorder=zorder)
    ax.add_patch(c)
    ax.text(x, y, str(num), ha='center', va='center',
            fontsize=11, fontweight='bold', color=WHITE, zorder=zorder + 1)


# ═══════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════
ax.text(7, 8.55, 'Self-Improving Feedback Loop', ha='center',
        fontsize=20, fontweight='bold', color=BLUE_DARK, zorder=10)
ax.text(7, 8.15, 'Continuous Learning Pipeline  —  MuleShield AI',
        ha='center', fontsize=10, color=MED_GREY, zorder=10)
ax.plot([3, 11], [8.0, 8.0], color=GOLD, linewidth=3, alpha=0.6, zorder=5)


# ═══════════════════════════════════════════
# BACKGROUND — subtle cycle ring
# ═══════════════════════════════════════════
theta = np.linspace(0, 2 * np.pi, 100)
r = 2.6
cx_bg, cy_bg = 7, 4.3
ax.plot(cx_bg + r * np.cos(theta), cy_bg + r * np.sin(theta),
        color='#E8E8E8', linewidth=35, alpha=0.20, zorder=0, solid_capstyle='round')


# ═══════════════════════════════════════════
# 6 STEPS — clockwise cycle
# ═══════════════════════════════════════════

# Step 1: ML Model (Top center-right)
bw, bh = 3.2, 0.9
draw_block(5.8, 6.8, bw, bh, 'ML Model Scoring',
           'XGBoost Ensemble  |  t = 0.483',
           BLUE_LIGHT, BLUE_MED, '#1A5276')
step_number(5.6, 7.55, 1, BLUE_MED)

# Step 2: Analyst Dashboard (Right)
draw_block(10.0, 4.6, bw, bh, 'Analyst Dashboard',
           'Review  |  Investigate  |  Decide',
           ORANGE_LIGHT, ORANGE_MED, '#935116')
step_number(9.8, 5.35, 2, ORANGE_MED)

# Step 3: Feedback Collection (Bottom right)
draw_block(9.2, 2.2, bw + 0.3, bh, 'Feedback Collection',
           'TP  |  FP  |  Needs More Info',
           GREEN_LIGHT, GREEN_MED, GREEN_DARK)
step_number(9.0, 2.95, 3, GREEN_MED)

# Step 4: Label Propagation (Bottom left)
draw_block(1.8, 2.2, bw + 0.3, bh, 'Label Propagation',
           'Drift monitoring  |  PSI + KL-div',
           PURPLE_LIGHT, PURPLE_MED, PURPLE_DARK)
step_number(1.6, 2.95, 4, PURPLE_MED)

# Step 5: Auto-Retrain (Left)
draw_block(0.8, 4.6, bw, bh, 'Auto-Retrain',
           'Optuna  |  5-fold CV  |  SMOTE',
           RED_LIGHT, RED_MED, RED_DARK)
step_number(0.6, 5.35, 5, RED_MED)

# Step 6: Model Carousel (Top left)
draw_block(1.8, 6.8, bw, bh, 'Model Carousel',
           'Shadow test  |  Promote if F1 > 2%',
           TEAL_LIGHT, TEAL_MED, TEAL_DARK)
step_number(1.6, 7.55, 6, TEAL_MED)


# ═══════════════════════════════════════════
# FLOW ARROWS (clockwise)
# ═══════════════════════════════════════════

# 1 -> 2: Top center -> Right
flow_arrow(9.0, 7.0, 10.0, 5.5, BLUE_MED,
           connectionstyle='arc3,rad=-0.15')

# 2 -> 3: Right -> Bottom right
flow_arrow(11.6, 4.6, 11.2, 3.1, ORANGE_MED,
           connectionstyle='arc3,rad=-0.1')

# 3 -> 4: Bottom right -> Bottom left
flow_arrow(9.2, 2.65, 5.3, 2.65, GREEN_MED)

# 4 -> 5: Bottom left -> Left
flow_arrow(2.2, 3.1, 2.0, 4.6, PURPLE_MED,
           connectionstyle='arc3,rad=-0.1')

# 5 -> 6: Left -> Top left
flow_arrow(2.4, 5.5, 3.0, 6.8, RED_MED,
           connectionstyle='arc3,rad=-0.15')

# 6 -> 1: Top left -> Top center
flow_arrow(5.0, 7.25, 5.8, 7.25, TEAL_MED)


# ═══════════════════════════════════════════
# FLOW LABELS — horizontal, clean
# ═══════════════════════════════════════════
arrow_labels = [
    (9.8, 6.3,  'Flagged accounts',    BLUE_DARK),
    (12.0, 3.8, 'Review & decide',     ORANGE_DARK),
    (7.2, 2.25, 'TP / FP labels',      GREEN_DARK),
    (1.5, 3.8,  'New training data',   PURPLE_DARK),
    (2.1, 6.3,  'Retrained model',     RED_DARK),
    (5.4, 7.85, 'Promoted',            TEAL_DARK),
]

for lx, ly, txt, col in arrow_labels:
    ax.text(lx, ly, txt, ha='center', va='center', fontsize=7,
            fontweight='bold', color=col, fontstyle='italic',
            zorder=8,
            bbox=dict(boxstyle='round,pad=0.12', facecolor=WHITE,
                     edgecolor='none', alpha=0.85))


# ═══════════════════════════════════════════
# CENTER — key message
# ═══════════════════════════════════════════
center_bg = FancyBboxPatch((5.0, 3.7), 4.0, 1.2,
                           boxstyle="round,pad=0.10",
                           facecolor='#FFF9E6', edgecolor=GOLD,
                           linewidth=2, alpha=0.9, zorder=7)
ax.add_patch(center_bg)

ax.text(7, 4.55, 'Continuous Improvement', ha='center', va='center',
        fontsize=12, fontweight='bold', color=BLUE_DARK, zorder=8)
ax.text(7, 4.15, 'Auto-adapts to evolving fraud patterns', ha='center',
        fontsize=8, color=MED_GREY, zorder=8)
ax.text(7, 3.90, 'without manual intervention', ha='center',
        fontsize=8, color=MED_GREY, zorder=8)


# ═══════════════════════════════════════════
# BOTTOM BANNER
# ═══════════════════════════════════════════
banner = FancyBboxPatch((0.3, 0.15), 13.4, 0.65,
                        boxstyle="round,pad=0.06",
                        facecolor=LIGHT_GREY, edgecolor='#DEE2E6',
                        linewidth=1.2, zorder=2)
ax.add_patch(banner)

stats = [
    ('Promotion Gate', 'F1 > 2%'),
    ('Drift Detection', 'PSI + KL-div'),
    ('Label Propagation', '5 similar accts'),
    ('Review Speedup', '90% faster'),
]
for i, (label, value) in enumerate(stats):
    sx = 2.0 + i * 3.2
    ax.text(sx, 0.58, value, ha='center', va='center',
            fontsize=9, fontweight='bold', color=BLUE_DARK, zorder=5)
    ax.text(sx, 0.30, label, ha='center', va='center',
            fontsize=7, color=MED_GREY, zorder=5)

for i in range(len(stats) - 1):
    sx = 2.0 + i * 3.2 + 1.6
    ax.plot([sx, sx], [0.22, 0.68], color='#DEE2E6', linewidth=1, zorder=4)


# ═══════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════
out = r"c:\Github\BOI Hackathon\files\feedback_loop.png"
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=WHITE, pad_inches=0.12)
plt.close()
print(f"Saved: {out}")
