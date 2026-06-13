"""
Generate publication-quality EDA graphs for IDEA_PROPOSAL.tex
Polished matplotlib version with better typography and layout.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# === Professional Color Palette ===
BOI_BLUE    = '#004B87'
BOI_ORANGE  = '#A03214'
ACCENT_BLUE = '#2478B5'
LIGHT_BLUE  = '#D6EAF8'
LIGHT_ORANGE= '#FADBD8'
GOLD        = '#F2C83C'
DARK_GREY   = '#2D2D2D'
MED_GREY    = '#666666'
LIGHT_GREY  = '#F7F7F7'
GRID_GREY   = '#E0E0E0'
GREEN       = '#27AE60'
RED         = '#E74C3C'

# === Load Data ===
print("Loading dataset...")
df = pd.read_csv(r"c:\Github\BOI Hackathon\DataSet.csv")
target = df['F3924']
drop_cols = ['Unnamed: 0', 'F3912', 'F2230']
df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])

# === Font Setup ===
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Helvetica'],
    'font.size': 9,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': GRID_GREY,
    'axes.grid': False,
    'xtick.color': MED_GREY,
    'ytick.color': MED_GREY,
})

# ================================================
# FIGURE
# ================================================
fig = plt.figure(figsize=(15, 5.2), dpi=300, facecolor='white')
gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.3, 1.5], wspace=0.32)

# ──────────────────────────────────────────────
# PANEL 1: Class Imbalance — Donut Chart
# ──────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])

counts = [(target == 0).sum(), (target == 1).sum()]
colors = [ACCENT_BLUE, BOI_ORANGE]

wedges, texts, autotexts = ax1.pie(
    counts,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.78,
    explode=(0, 0.08),
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2.5),
    textprops=dict(fontsize=9, color=DARK_GREY)
)

# Remove default labels
for t in texts:
    t.set_text('')

# Style percentage text
autotexts[0].set_fontweight('bold')
autotexts[0].set_fontsize(13)
autotexts[0].set_color('white')
autotexts[1].set_fontweight('bold')
autotexts[1].set_fontsize(9)
autotexts[1].set_color(BOI_ORANGE)

# Center text
ax1.text(0, 0.08, '111 : 1', ha='center', va='center',
         fontsize=16, fontweight='bold', color=BOI_BLUE)
ax1.text(0, -0.12, 'imbalance', ha='center', va='center',
         fontsize=9, color=MED_GREY)

# Legend below
legend_patches = [
    mpatches.Patch(facecolor=ACCENT_BLUE, edgecolor='white', label=f'Legitimate — {counts[0]:,}'),
    mpatches.Patch(facecolor=BOI_ORANGE, edgecolor='white', label=f'Suspicious — {counts[1]:,}'),
]
ax1.legend(handles=legend_patches, loc='lower center', fontsize=8,
           frameon=True, framealpha=0.95, edgecolor=GRID_GREY,
           bbox_to_anchor=(0.5, -0.08), ncol=1, handlelength=1.2)

ax1.set_title('Class Distribution', fontsize=12, fontweight='bold',
              color=BOI_BLUE, pad=14)

# ──────────────────────────────────────────────
# PANEL 2: Mule "Low-Everything" Signature
# ──────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])

top_features = ['F3898', 'F1819', 'F3799', 'F1165', 'F1813', 'F3806', 'F162', 'F3800']

ratios = []
for f in top_features:
    legit_mean = df_clean.loc[target == 0, f].mean()
    mule_mean = df_clean.loc[target == 1, f].mean()
    ratios.append(mule_mean / legit_mean if legit_mean > 0 else 0)

y = np.arange(len(top_features))

# Background shading for alternating rows
for i in range(len(top_features)):
    if i % 2 == 0:
        ax2.axhspan(i - 0.4, i + 0.4, color=LIGHT_GREY, alpha=0.5, zorder=0)

# Bars
bar_colors = [BOI_ORANGE if r < 0.5 else ACCENT_BLUE for r in ratios]
bars = ax2.barh(y, ratios, color=bar_colors, height=0.55, edgecolor='white',
                linewidth=0.8, zorder=3, alpha=0.9)

# Add ratio labels
for i, (bar, ratio) in enumerate(zip(bars, ratios)):
    w = bar.get_width()
    label_x = max(w + 0.03, 0.08)
    color = RED if ratio < 0.1 else (MED_GREY if ratio < 0.5 else GREEN)
    ax2.text(label_x, bar.get_y() + bar.get_height()/2,
             f'{ratio:.3f}×', ha='left', va='center', fontsize=8.5,
             fontweight='bold', color=color, fontfamily='monospace')

# Baseline reference
ax2.axvline(x=1.0, color=ACCENT_BLUE, linestyle='--', linewidth=1, alpha=0.4, zorder=1)
ax2.text(1.03, len(top_features) - 0.2, 'Legit\nbaseline', fontsize=7,
         color=ACCENT_BLUE, va='top', alpha=0.7, fontstyle='italic')

ax2.set_yticks(y)
ax2.set_yticklabels(top_features, fontsize=9.5, fontfamily='monospace',
                    fontweight='medium', color=DARK_GREY)
ax2.set_xlabel('Mule Mean ÷ Legitimate Mean', fontsize=9, color=MED_GREY)
ax2.set_title('Mule "Low-Everything" Signature\n(Top 8 Features by Mutual Information)',
              fontsize=12, fontweight='bold', color=BOI_BLUE, pad=14)
ax2.set_xlim(0, max(ratios) * 1.35)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(0.5)
ax2.spines['bottom'].set_linewidth(0.5)
ax2.tick_params(left=False)
ax2.grid(axis='x', alpha=0.2, color=GRID_GREY, linestyle='-', zorder=0)

# ──────────────────────────────────────────────
# PANEL 3: Feature Activity by Block
# ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])

block_ranges = [
    ('F1–299\nProportions',    1,    299),
    ('F300–599\nRatios',       300,  599),
    ('F600–1499\nFlags+Amts',  600,  1499),
    ('F1500–2199\nCounts',     1500, 2199),
    ('F2200–2499\nScores',     2200, 2499),
    ('F2500–2799\nDeltas',     2500, 2799),
    ('F2800–3099\nRaw Counts', 2800, 3099),
    ('F3100–3799\nGrowth',     3100, 3799),
    ('F3800–3924\nProfile',    3800, 3924),
]

block_names = []
mule_nz_vals = []
legit_nz_vals = []

for name, start, end in block_ranges:
    cols = [f'F{i}' for i in range(start, end + 1) if f'F{i}' in df_clean.columns]
    if cols:
        block_names.append(name)
        mule_nz_vals.append((df_clean.loc[target == 1, cols] != 0).mean().mean() * 100)
        legit_nz_vals.append((df_clean.loc[target == 0, cols] != 0).mean().mean() * 100)

y = np.arange(len(block_names))
h = 0.32

# Background shading
for i in range(len(block_names)):
    if i % 2 == 0:
        ax3.axhspan(i - 0.45, i + 0.45, color=LIGHT_GREY, alpha=0.5, zorder=0)

bars_l = ax3.barh(y + h/2, legit_nz_vals, h, label='Legitimate',
                  color=ACCENT_BLUE, alpha=0.85, edgecolor='white', linewidth=0.8, zorder=3)
bars_m = ax3.barh(y - h/2, mule_nz_vals, h, label='Suspicious',
                  color=BOI_ORANGE, alpha=0.85, edgecolor='white', linewidth=0.8, zorder=3)

# Difference annotations for key blocks
for i in range(len(block_names)):
    diff = legit_nz_vals[i] - mule_nz_vals[i]
    if abs(diff) > 5:
        max_val = max(legit_nz_vals[i], mule_nz_vals[i])
        ax3.text(min(max_val + 1.5, 98), y[i],
                 f'Δ{diff:+.0f}%', fontsize=6.5, color=MED_GREY,
                 va='center', fontstyle='italic')

ax3.set_xlabel('% Non-Zero Values', fontsize=9, color=MED_GREY)
ax3.set_title('Feature Activity by Block\n(Mule vs Legitimate)',
              fontsize=12, fontweight='bold', color=BOI_BLUE, pad=14)
ax3.set_yticks(y)
ax3.set_yticklabels(block_names, fontsize=7.5)
ax3.set_xlim(0, 108)
ax3.legend(fontsize=8.5, loc='lower right', framealpha=0.95, edgecolor=GRID_GREY,
           handlelength=1.2)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_linewidth(0.5)
ax3.spines['bottom'].set_linewidth(0.5)
ax3.tick_params(left=False, labelsize=8)
ax3.grid(axis='x', alpha=0.2, color=GRID_GREY, linestyle='-', zorder=0)

# ──────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────
plt.tight_layout()
output_path = r"c:\Github\BOI Hackathon\files\eda_summary.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.12)
plt.close()
print(f"\nSaved: {output_path}")
print("Done!")
