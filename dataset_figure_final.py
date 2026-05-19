import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

OUT = Path('analysis_output')
DATA = Path('[1] Data/Original color stack data')

# Load
s1_bf_gt = np.array(Image.open(DATA/'Sample 1_BF and GT(512x512 two images).png'))
s1_layers_img = np.array(Image.open(DATA/'Sample 1_GT and 8 layer images(256x256 nine images).png'))
s2_bf_gt = np.array(Image.open(DATA/'Sample 2_BF and GT(512x512 two images).png'))
s2_layers_img = np.array(Image.open(DATA/'Sample 2_GT and 8 layer images(256x256 nine images).png'))

s1_bf, s1_gt = s1_bf_gt[:512], s1_bf_gt[512:]
s2_bf, s2_gt = s2_bf_gt[:512], s2_bf_gt[512:]
s1_gt_sm = s1_layers_img[:256]
s1_layers = [s1_layers_img[256*i:256*(i+1)] for i in range(1, 9)]
s2_gt_sm = s2_layers_img[:256]
s2_layers = [s2_layers_img[256*i:256*(i+1)] for i in range(1, 9)]

# Layer grouping: L1-3=Top(Blue), L4-5=Middle(Yellow), L6-8=Bottom(Red)
group_names = ['Top (Blue)\nL1-L3, ~0-30 um', 'Middle (Yellow)\nL4-L5, ~30-50 um', 'Bottom (Red)\nL6-L8, ~50-80 um']
group_colors_border = ['#4488FF', '#FFD700', '#FF4444']
group_indices = [[0,1,2], [3,4], [5,6,7]]

def merge_layers(layers, indices):
    """Max-project selected layers into one image."""
    merged = np.zeros_like(layers[0], dtype=np.float64)
    for idx in indices:
        merged = np.maximum(merged, layers[idx].astype(np.float64))
    return np.clip(merged, 0, 255).astype(np.uint8)

# ===== MAIN COMPOSITE FIGURE =====
fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
gs_main = gridspec.GridSpec(5, 1, figure=fig, hspace=0.45,
                            height_ratios=[1.0, 0.65, 0.55, 0.7, 0.7])

# --- Panel a: BF → Multi-color GT (Sample 1) ---
gs_a = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[0], wspace=0.15,
                                         width_ratios=[1, 0.15, 1])
ax_bf = fig.add_subplot(gs_a[0])
ax_bf.imshow(s1_bf)
ax_bf.set_title('Brightfield (BF)', fontsize=9, fontweight='bold')
ax_bf.axis('off')

# Arrow
ax_arr = fig.add_subplot(gs_a[1])
ax_arr.axis('off')
ax_arr.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'),
                xycoords='axes fraction')

ax_gt = fig.add_subplot(gs_a[2])
ax_gt.imshow(s1_gt)
ax_gt.set_title('Depth-encoded\nFluorescence (GT)', fontsize=9, fontweight='bold')
ax_gt.axis('off')

ax_bf.text(-0.08, 1.05, 'a', transform=ax_bf.transAxes, fontsize=14, fontweight='bold', va='top')

# --- Panel b: 8 individual layers with grouping brackets ---
gs_b = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=gs_main[1], wspace=0.08)

for i in range(8):
    ax = fig.add_subplot(gs_b[i])
    ax.imshow(cv2.resize(s1_layers[i], (256, 256), interpolation=cv2.INTER_LANCZOS4))
    ax.set_title(f'L{i+1}', fontsize=7, fontweight='bold')
    ax.axis('off')
    # Color border by group
    for gi, indices in enumerate(group_indices):
        if i in indices:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(group_colors_border[gi])
                spine.set_linewidth(2.5)
            break

fig.axes[2].text(-0.15, 1.05, 'b', transform=fig.axes[2].transAxes, fontsize=14, fontweight='bold', va='top')

# Add group labels below
# Get axes positions after layout
fig.canvas.draw()

# --- Panel c: 3 merged groups + GT comparison ---
gs_c = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_main[2], wspace=0.1,
                                         width_ratios=[1, 1, 1, 0.15, 1])
merged_groups = [merge_layers(s1_layers, gi) for gi in group_indices]

for ci, (mg, gn, gc) in enumerate(zip(merged_groups, group_names, group_colors_border)):
    ax = fig.add_subplot(gs_c[ci])
    ax.imshow(cv2.resize(mg, (256, 256), interpolation=cv2.INTER_LANCZOS4))
    ax.set_title(gn, fontsize=7, fontweight='bold', color=gc)
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(gc)
        spine.set_linewidth(2)

# Arrow
ax_arr2 = fig.add_subplot(gs_c[3])
ax_arr2.axis('off')
ax_arr2.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                 arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                 xycoords='axes fraction')
ax_arr2.text(0.5, 0.7, 'Merge', fontsize=6, ha='center', va='bottom', transform=ax_arr2.transAxes)

ax_gt2 = fig.add_subplot(gs_c[4])
ax_gt2.imshow(cv2.resize(s1_gt, (256, 256), interpolation=cv2.INTER_LANCZOS4))
ax_gt2.set_title('Multi-color\nProjection', fontsize=7, fontweight='bold')
ax_gt2.axis('off')

fig.axes[len(fig.axes)-5].text(-0.15, 1.05, 'c', transform=fig.axes[len(fig.axes)-5].transAxes,
                                fontsize=14, fontweight='bold', va='top')

# --- Panel d: Hue histogram with 3-group annotation ---
gs_d = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[3], wspace=0.35)

ax_hue = fig.add_subplot(gs_d[0])
hsv = cv2.cvtColor(s1_gt, cv2.COLOR_RGB2HSV)
h_ch, s_ch = hsv[:,:,0], hsv[:,:,1]
gray = cv2.cvtColor(s1_gt, cv2.COLOR_RGB2GRAY)
fg = (gray > 15) & (s_ch > 20)
hue_vals = h_ch[fg]
ax_hue.hist(hue_vals, bins=180, range=(0,180), color='#555555', alpha=0.8, edgecolor='none')

# Color bands with group labels
bands = [
    (0, 12, '#FF4444', 0.2), (155, 180, '#FF4444', 0.2),
    (13, 55, '#FFD700', 0.2),
    (85, 140, '#4488FF', 0.2),
]
for h0, h1, color, alpha in bands:
    ax_hue.axvspan(h0, h1, alpha=alpha, color=color)

ymax = ax_hue.get_ylim()[1]
ax_hue.annotate('Bottom\n(L6-L8)', xy=(6, ymax*0.75), fontsize=7, fontweight='bold', color='#CC0000', ha='center')
ax_hue.annotate('Middle\n(L4-L5)', xy=(34, ymax*0.75), fontsize=7, fontweight='bold', color='#AA8800', ha='center')
ax_hue.annotate('Top\n(L1-L3)', xy=(112, ymax*0.75), fontsize=7, fontweight='bold', color='#2266CC', ha='center')

ax_hue.set_xlabel('Hue (OpenCV 0-179)', fontsize=9)
ax_hue.set_ylabel('Pixel Count', fontsize=9)
ax_hue.set_title('Hue Distribution\n(GT Projection)', fontsize=9, fontweight='bold')
ax_hue.spines['top'].set_visible(False)
ax_hue.spines['right'].set_visible(False)
ax_hue.tick_params(labelsize=8)

ax_hue.text(-0.15, 1.05, 'd', transform=ax_hue.transAxes, fontsize=14, fontweight='bold', va='top')

# Per-layer intensity with grouping
ax_int = fig.add_subplot(gs_d[1])
mean_ints = []
bar_colors = []
group_color_map = {0:'#4488FF', 1:'#4488FF', 2:'#4488FF',
                   3:'#FFD700', 4:'#FFD700',
                   5:'#FF4444', 6:'#FF4444', 7:'#FF4444'}

for i, layer in enumerate(s1_layers):
    g = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
    mean_val = np.mean(g[g > 10]) if np.sum(g > 10) > 0 else 0
    mean_ints.append(mean_val)
    bar_colors.append(group_color_map[i])

bars = ax_int.barh(range(8), mean_ints, color=bar_colors, edgecolor='black', linewidth=0.5, height=0.7)
ax_int.set_yticks(range(8))
ax_int.set_yticklabels([f'L{i+1}' for i in range(8)], fontsize=8)
ax_int.set_xlabel('Mean Intensity (a.u.)', fontsize=9)
ax_int.set_title('Per-layer Fluorescence\nIntensity', fontsize=9, fontweight='bold')
ax_int.invert_yaxis()
ax_int.spines['top'].set_visible(False)
ax_int.spines['right'].set_visible(False)
ax_int.tick_params(labelsize=8)

# Add depth scale
ax_int2 = ax_int.twinx()
ax_int2.set_ylim(ax_int.get_ylim())
ax_int2.set_yticks(range(8))
ax_int2.set_yticklabels([f'{d:.0f} um' for d in np.linspace(0, 80, 8)], fontsize=7, color='gray')
ax_int2.spines['top'].set_visible(False)

# Group brackets on left
bracket_groups = [(0, 2, 'Top', '#4488FF'), (3, 4, 'Mid', '#FFD700'), (5, 7, 'Bot', '#FF4444')]
for y0, y1, label, color in bracket_groups:
    ax_int.annotate('', xy=(-0.05, y0-0.3), xytext=(-0.05, y1+0.3),
                    xycoords=('axes fraction', 'data'), textcoords=('axes fraction', 'data'),
                    arrowprops=dict(arrowstyle='-', color=color, lw=2))

# --- Panel e: Depth map ---
gs_e = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[4], wspace=0.25)

ax_dm = fig.add_subplot(gs_e[0])
h_img, w_img = s1_layers[0].shape[:2]
depth_map = np.zeros((h_img, w_img))
cov_count = np.zeros((h_img, w_img))
for i, layer in enumerate(s1_layers):
    g = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
    mask = g > 15
    depth_map[mask] += (i + 1)
    cov_count[mask] += 1
avg_d = np.where(cov_count > 0, depth_map / cov_count, np.nan)
im_d = ax_dm.imshow(avg_d, cmap='coolwarm', vmin=1, vmax=8)
ax_dm.set_title('Mean Depth Map\n(per-pixel average layer)', fontsize=9, fontweight='bold')
ax_dm.axis('off')
cb = plt.colorbar(im_d, ax=ax_dm, fraction=0.046, pad=0.04, shrink=0.8)
cb.set_label('Layer (1=Top, 8=Bottom)', fontsize=7)
cb.ax.tick_params(labelsize=7)

ax_dm.text(-0.1, 1.05, 'e', transform=ax_dm.transAxes, fontsize=14, fontweight='bold', va='top')

# Sample 2 depth map
ax_dm2 = fig.add_subplot(gs_e[1])
h2, w2 = s2_layers[0].shape[:2]
dm2 = np.zeros((h2, w2))
cc2 = np.zeros((h2, w2))
for i, layer in enumerate(s2_layers):
    g = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
    mask = g > 15
    dm2[mask] += (i + 1)
    cc2[mask] += 1
ad2 = np.where(cc2 > 0, dm2 / cc2, np.nan)
im_d2 = ax_dm2.imshow(ad2, cmap='coolwarm', vmin=1, vmax=8)
ax_dm2.set_title('Mean Depth Map\n(Sample 2)', fontsize=9, fontweight='bold')
ax_dm2.axis('off')
cb2 = plt.colorbar(im_d2, ax=ax_dm2, fraction=0.046, pad=0.04, shrink=0.8)
cb2.set_label('Layer (1=Top, 8=Bottom)', fontsize=7)
cb2.ax.tick_params(labelsize=7)

plt.savefig(OUT/'dataset_figure_final.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Final dataset figure saved!')
