"""
Supplementary Figure S2: Tangent-guided Gap Bridging Algorithm
Publication-quality figure for Advanced Healthcare Materials
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from PIL import Image
from pathlib import Path

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output\paper_panels_depth')

# ============================================================
# Panel A: Algorithm flowchart (clean, journal-style)
# ============================================================
fig_a, ax_a = plt.subplots(figsize=(7, 1.8))
ax_a.set_xlim(0, 10)
ax_a.set_ylim(0, 2.0)
ax_a.axis('off')

box_color = '#f5f5f5'
edge_color = '#333333'
text_color = '#1a1a1a'
sub_color = '#666666'

steps = [
    {"x": 0.15, "title": "Step 1", "name": "Endpoint\nDetection",
     "desc": "neighbor = 1"},
    {"x": 2.15, "title": "Step 2", "name": "Tangent\nEstimation",
     "desc": "trace 10-20 px"},
    {"x": 4.15, "title": "Step 3", "name": "Candidate\nMatching",
     "desc": "dist, angle, overlap"},
    {"x": 6.15, "title": "Step 4", "name": "Bézier\nInterpolation",
     "desc": "cubic curve"},
    {"x": 8.15, "title": "Step 5", "name": "Greedy\nSelection",
     "desc": "best-first, no reuse"},
]

bw, bh = 1.7, 1.6
for s in steps:
    box = FancyBboxPatch((s["x"], 0.2), bw, bh,
                          boxstyle="round,pad=0.08",
                          facecolor=box_color, edgecolor=edge_color,
                          linewidth=1.0)
    ax_a.add_patch(box)
    cx = s["x"] + bw / 2
    ax_a.text(cx, 1.55, s["title"], fontsize=6.5, ha='center', va='center',
              color='#888888', fontweight='bold')
    ax_a.text(cx, 1.1, s["name"], fontsize=8, ha='center', va='center',
              color=text_color, fontweight='bold', linespacing=1.3)
    ax_a.text(cx, 0.5, s["desc"], fontsize=6.5, ha='center', va='center',
              color=sub_color, fontstyle='italic')

# Arrows between steps
for i in range(4):
    x_start = steps[i]["x"] + bw + 0.02
    x_end = steps[i + 1]["x"] - 0.02
    ax_a.annotate('', xy=(x_end, 1.0), xytext=(x_start, 1.0),
                  arrowprops=dict(arrowstyle='->', color='#888888',
                                  lw=1.2, mutation_scale=12))

plt.tight_layout(pad=0.1)
fig_a.savefig(OUT / 'figS2_a_flowchart.png', dpi=400, bbox_inches='tight',
              facecolor='white', edgecolor='none')
plt.close()
print('Saved figS2_a_flowchart.png')

# ============================================================
# Panel B: Schematic (before → after bridging)
# ============================================================
fig_b, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(7, 3.0))

for ax, title in [(ax_b1, 'Before bridging'), (ax_b2, 'After bridging')]:
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold', color='#333333', pad=8)

# --- Before: two broken skeleton segments with tangent arrows ---
# Segment 1: entering from top-left
seg1_x = [0.0, 0.8, 1.5, 2.0]
seg1_y = [4.0, 3.3, 2.8, 2.4]
ax_b1.plot(seg1_x, seg1_y, color='#cc3333', linewidth=4, solid_capstyle='round')

# Segment 2: exiting to bottom-right
seg2_x = [3.0, 3.5, 4.2, 5.0]
seg2_y = [1.8, 1.4, 0.9, 0.2]
ax_b1.plot(seg2_x, seg2_y, color='#cc3333', linewidth=4, solid_capstyle='round')

# Endpoint markers
ep1 = (2.0, 2.4)
ep2 = (3.0, 1.8)
ax_b1.plot(*ep1, 'o', color='#f1c40f', markersize=10, markeredgecolor='#333333',
           markeredgewidth=1.2, zorder=5)
ax_b1.plot(*ep2, 'o', color='#f1c40f', markersize=10, markeredgecolor='#333333',
           markeredgewidth=1.2, zorder=5)

# Tangent arrows (outward from endpoints)
t1_dx, t1_dy = 0.7, -0.55
t2_dx, t2_dy = -0.7, 0.55
ax_b1.annotate('', xy=(ep1[0] + t1_dx, ep1[1] + t1_dy), xytext=ep1,
               arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2.0,
                               mutation_scale=15))
ax_b1.annotate('', xy=(ep2[0] + t2_dx, ep2[1] + t2_dy), xytext=ep2,
               arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2.0,
                               mutation_scale=15))

# Labels
ax_b1.text(ep1[0] + t1_dx + 0.15, ep1[1] + t1_dy - 0.1, r'$\mathbf{t}_1$',
           fontsize=11, color='#2980b9', fontweight='bold')
ax_b1.text(ep2[0] + t2_dx - 0.35, ep2[1] + t2_dy + 0.1, r'$\mathbf{t}_2$',
           fontsize=11, color='#2980b9', fontweight='bold')
ax_b1.text(ep1[0] - 0.4, ep1[1] + 0.25, 'EP₁', fontsize=8, color='#333333')
ax_b1.text(ep2[0] + 0.15, ep2[1] - 0.35, 'EP₂', fontsize=8, color='#333333')

# Gap indication
mid_x = (ep1[0] + ep2[0]) / 2
mid_y = (ep1[1] + ep2[1]) / 2
ax_b1.plot([ep1[0], ep2[0]], [ep1[1], ep2[1]], '--', color='#aaaaaa', linewidth=1)
ax_b1.text(mid_x - 0.15, mid_y + 0.35, 'gap', fontsize=8, color='#999999',
           fontstyle='italic', ha='center')

# Distance annotation
ax_b1.annotate('', xy=(ep2[0], ep2[1] - 0.6), xytext=(ep1[0], ep1[1] - 0.6),
               arrowprops=dict(arrowstyle='<->', color='#888888', lw=0.8))
ax_b1.text(mid_x, mid_y - 0.75, 'd < 80 px', fontsize=6.5, color='#888888',
           ha='center', fontstyle='italic')

# Angle arcs (simplified)
ax_b1.text(ep1[0] + 0.4, ep1[1] - 0.05, 'θ₁', fontsize=7, color='#2980b9')
ax_b1.text(ep2[0] - 0.55, ep2[1] + 0.15, 'θ₂', fontsize=7, color='#2980b9')

# --- After: connected with Bézier ---
ax_b2.plot(seg1_x, seg1_y, color='#cc3333', linewidth=4, solid_capstyle='round')
ax_b2.plot(seg2_x, seg2_y, color='#cc3333', linewidth=4, solid_capstyle='round')

# Bézier bridge
t_vals = np.linspace(0, 1, 50)
p1 = np.array([2.0, 2.4])
p2 = np.array([3.0, 1.8])
c1 = p1 + np.array([t1_dx, t1_dy]) * 1.2
c2 = p2 + np.array([t2_dx, t2_dy]) * 1.2
bridge_x = (1 - t_vals)**3 * p1[0] + 3*(1 - t_vals)**2 * t_vals * c1[0] + \
           3*(1 - t_vals) * t_vals**2 * c2[0] + t_vals**3 * p2[0]
bridge_y = (1 - t_vals)**3 * p1[1] + 3*(1 - t_vals)**2 * t_vals * c1[1] + \
           3*(1 - t_vals) * t_vals**2 * c2[1] + t_vals**3 * p2[1]
ax_b2.plot(bridge_x, bridge_y, color='#27ae60', linewidth=4,
           solid_capstyle='round', linestyle='-')

# Control point lines (thin, dashed)
ax_b2.plot([p1[0], c1[0]], [p1[1], c1[1]], '--', color='#aaaaaa', linewidth=0.8)
ax_b2.plot([p2[0], c2[0]], [p2[1], c2[1]], '--', color='#aaaaaa', linewidth=0.8)
ax_b2.plot(c1[0], c1[1], 's', color='#aaaaaa', markersize=4)
ax_b2.plot(c2[0], c2[1], 's', color='#aaaaaa', markersize=4)

# Labels
ax_b2.text(c1[0] + 0.1, c1[1] - 0.25, 'C₁', fontsize=7, color='#888888')
ax_b2.text(c2[0] - 0.35, c2[1] + 0.15, 'C₂', fontsize=7, color='#888888')

# Bridge label
bm_x = bridge_x[len(bridge_x)//2]
bm_y = bridge_y[len(bridge_y)//2]
ax_b2.text(bm_x + 0.3, bm_y + 0.3, 'Bézier\nbridge', fontsize=8,
           color='#27ae60', fontweight='bold', ha='center', linespacing=1.3)

# Endpoint dots
ax_b2.plot(*ep1, 'o', color='#f1c40f', markersize=10, markeredgecolor='#333333',
           markeredgewidth=1.2, zorder=5)
ax_b2.plot(*ep2, 'o', color='#f1c40f', markersize=10, markeredgecolor='#333333',
           markeredgewidth=1.2, zorder=5)

# Formula annotation below
formula = r'$P(t) = (1\!-\!t)^3 P_1 + 3(1\!-\!t)^2 t\, C_1 + 3(1\!-\!t)\, t^2 C_2 + t^3 P_2$'
ax_b2.text(2.5, -0.3, formula, fontsize=8, ha='center', color='#555555')

plt.tight_layout(pad=0.3)
fig_b.savefig(OUT / 'figS2_b_scheme.png', dpi=400, bbox_inches='tight',
              facecolor='white', edgecolor='none')
plt.close()
print('Saved figS2_b_scheme.png')

# ============================================================
# Panel C: Real bridging examples (before / after)
# ============================================================
# Reuse the zoom crops from create_zoom_3stage_v3.py
# Load existing before/after pairs
fig_c, axes = plt.subplots(2, 3, figsize=(7, 4.5))

titles_top = ['Example 1', 'Example 2', 'Example 3']
for col in range(3):
    before = np.array(Image.open(OUT / f'n_zoom_{col+1}_before.png'))
    after = np.array(Image.open(OUT / f'n_zoom_{col+1}_after.png'))

    axes[0, col].imshow(before)
    axes[0, col].set_title(titles_top[col], fontsize=9, fontweight='bold', color='#333333')
    axes[0, col].axis('off')
    if col == 0:
        axes[0, col].set_ylabel('Before', fontsize=10, fontweight='bold', color='#888888')

    axes[1, col].imshow(after)
    axes[1, col].axis('off')
    if col == 0:
        axes[1, col].set_ylabel('After', fontsize=10, fontweight='bold', color='#888888')

    # Add thin borders
    for row in range(2):
        for spine in axes[row, col].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.8)

plt.tight_layout(h_pad=0.3, w_pad=0.3)
fig_c.savefig(OUT / 'figS2_c_examples.png', dpi=400, bbox_inches='tight',
              facecolor='white', edgecolor='none')
plt.close()
print('Saved figS2_c_examples.png')

# ============================================================
# Combined Figure S2 (A + B + C stacked)
# ============================================================
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(7, 9.5))
gs = GridSpec(3, 1, figure=fig, height_ratios=[1.5, 2.8, 3.8], hspace=0.25)

# Panel A
ax_a = fig.add_subplot(gs[0])
img_a = np.array(Image.open(OUT / 'figS2_a_flowchart.png'))
ax_a.imshow(img_a)
ax_a.axis('off')
ax_a.text(-0.02, 1.05, 'A', transform=ax_a.transAxes, fontsize=14,
          fontweight='bold', va='top')

# Panel B
ax_b = fig.add_subplot(gs[1])
img_b = np.array(Image.open(OUT / 'figS2_b_scheme.png'))
ax_b.imshow(img_b)
ax_b.axis('off')
ax_b.text(-0.02, 1.05, 'B', transform=ax_b.transAxes, fontsize=14,
          fontweight='bold', va='top')

# Panel C
ax_c = fig.add_subplot(gs[2])
img_c = np.array(Image.open(OUT / 'figS2_c_examples.png'))
ax_c.imshow(img_c)
ax_c.axis('off')
ax_c.text(-0.02, 1.05, 'C', transform=ax_c.transAxes, fontsize=14,
          fontweight='bold', va='top')

plt.savefig(OUT / 'figS2_combined.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('Saved figS2_combined.png')

print('\nAll Figure S2 panels saved.')
