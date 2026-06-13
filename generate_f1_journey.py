"""
Generate F1 improvement journey graph for IDEA_PROPOSAL.tex
Shows the progression across all phases with a polished bar+line chart.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

# === BOI Color Palette ===
BOI_BLUE    = '#004B87'
BOI_ORANGE  = '#A03214'
ACCENT_BLUE = '#2478B5'
DARK_GREY   = '#2D2D2D'
MED_GREY    = '#666666'
LIGHT_GREY  = '#F7F7F7'
GRID_GREY   = '#E0E0E0'
GREEN       = '#27AE60'
GOLD        = '#F2C83C'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Helvetica'],
    'font.size': 9,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': GRID_GREY,
})

# === Data from our research ===
phases = [
    'Phase 5\nBaseline',
    'Phase 5\n+SMOTE',
    'Phase 5\n+Optuna',
    'Phase 6\nPS Features',
    'Phase 7\nNeural Nets',
    'Phase 8\nAdvanced',
    'Phase 9\nFinal Blend',
]

f1_scores = [0.604, 0.621, 0.636, 0.684, 0.704, 0.704, 0.743]
improvements = [0] + [f1_scores[i] - f1_scores[i-1] for i in range(1, len(f1_scores))]

# === Figure ===
fig, ax = plt.subplots(figsize=(14, 4.5), dpi=300, facecolor='white')

x = np.arange(len(phases))
bar_width = 0.55

# Gradient-like bars (color intensity based on F1)
colors = []
for i, f1 in enumerate(f1_scores):
    if i == len(f1_scores) - 1:  # Final
        colors.append(BOI_ORANGE)
    else:
        # Interpolate from light blue to deep blue
        t = (f1 - min(f1_scores)) / (max(f1_scores) - min(f1_scores))
        r = int(36 + (0 - 36) * t)
        g = int(120 + (75 - 120) * t)
        b = int(181 + (135 - 181) * t)
        colors.append(f'#{r:02x}{g:02x}{b:02x}')

# Bars
bars = ax.bar(x, f1_scores, bar_width, color=colors, edgecolor='white',
              linewidth=1.5, zorder=3, alpha=0.92)

# Line overlay
line = ax.plot(x, f1_scores, color=BOI_BLUE, linewidth=2, marker='o',
               markersize=7, markerfacecolor='white', markeredgecolor=BOI_BLUE,
               markeredgewidth=2, zorder=5)

# F1 value labels on top of bars
for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
    weight = 'bold' if i == len(f1_scores) - 1 else 'semibold'
    fontsize = 12 if i == len(f1_scores) - 1 else 10
    color = BOI_ORANGE if i == len(f1_scores) - 1 else BOI_BLUE

    ax.text(bar.get_x() + bar.get_width()/2., f1 + 0.008,
            f'{f1:.3f}', ha='center', va='bottom', fontweight=weight,
            fontsize=fontsize, color=color,
            path_effects=[pe.withStroke(linewidth=3, foreground='white')])

# Improvement arrows/labels between bars
for i in range(1, len(f1_scores)):
    delta = improvements[i]
    if delta > 0:
        ax.annotate('', xy=(x[i], f1_scores[i] - 0.005),
                    xytext=(x[i-1], f1_scores[i-1] - 0.005),
                    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2,
                                   connectionstyle='arc3,rad=-0.15'),
                    zorder=4)
        # Delta label
        mid_x = (x[i] + x[i-1]) / 2
        mid_y = (f1_scores[i] + f1_scores[i-1]) / 2 - 0.025
        ax.text(mid_x, mid_y, f'+{delta:.3f}', ha='center', va='center',
                fontsize=7.5, color=GREEN, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         edgecolor=GREEN, alpha=0.85, linewidth=0.5))

# Reference line — starting point
ax.axhline(y=f1_scores[0], color=MED_GREY, linestyle=':', linewidth=0.8, alpha=0.5, zorder=1)
ax.text(len(phases) - 0.3, f1_scores[0] - 0.01, 'baseline', fontsize=7,
        color=MED_GREY, ha='right', fontstyle='italic', alpha=0.7)

# Total improvement annotation
total_improvement = f1_scores[-1] - f1_scores[0]
ax.annotate(f'Total: +{total_improvement:.3f}\n({total_improvement/f1_scores[0]*100:.1f}% improvement)',
            xy=(len(phases)-1, f1_scores[-1] + 0.008),
            xytext=(len(phases)-1.2, f1_scores[-1] + 0.06),
            fontsize=9, fontweight='bold', color=BOI_ORANGE,
            ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor=BOI_ORANGE, linewidth=1.2, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color=BOI_ORANGE, lw=1.5))

# Styling
ax.set_xticks(x)
ax.set_xticklabels(phases, fontsize=8.5, color=DARK_GREY, linespacing=1.3)
ax.set_ylabel('F1 Score', fontsize=10, color=DARK_GREY, fontweight='medium')
ax.set_title('F1 Score Progression Across Model Development Phases',
             fontsize=13, fontweight='bold', color=BOI_BLUE, pad=15)

ax.set_ylim(0.55, f1_scores[-1] + 0.09)
ax.set_xlim(-0.5, len(phases) - 0.3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.tick_params(left=True, bottom=False, labelsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f'{v:.2f}'))
ax.grid(axis='y', alpha=0.25, color=GRID_GREY, linestyle='-', zorder=0)

# Background bands for phases
phase_groups = [
    (0, 2, 'Phase 5', '#E8F0FE'),    # light blue
    (3, 3, 'Phase 6', '#FFF3E0'),    # light orange
    (4, 5, 'Phase 7–8', '#F3E5F5'),  # light purple
    (6, 6, 'Phase 9', '#E8F5E9'),    # light green
]
for start, end, label, bg_color in phase_groups:
    ax.axvspan(start - 0.4, end + 0.4, color=bg_color, alpha=0.35, zorder=0)

plt.tight_layout()
output_path = r"c:\Github\BOI Hackathon\files\f1_journey.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.12)
plt.close()
print(f"Saved: {output_path}")
print("Done!")
