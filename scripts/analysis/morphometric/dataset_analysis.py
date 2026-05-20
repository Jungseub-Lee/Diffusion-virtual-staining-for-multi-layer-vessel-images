import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

OUT = Path('analysis_output')
DATA = Path('[1] Data/Original color stack data')

# Load images
s1_bf_gt = np.array(Image.open(DATA/'Sample 1_BF and GT(512x512 two images).png'))
s1_layers = np.array(Image.open(DATA/'Sample 1_GT and 8 layer images(256x256 nine images).png'))
s2_bf_gt = np.array(Image.open(DATA/'Sample 2_BF and GT(512x512 two images).png'))
s2_layers = np.array(Image.open(DATA/'Sample 2_GT and 8 layer images(256x256 nine images).png'))

# Split images
s1_bf = s1_bf_gt[:512, :, :]
s1_gt = s1_bf_gt[512:, :, :]
s2_bf = s2_bf_gt[:512, :, :]
s2_gt = s2_bf_gt[512:, :, :]

# Split 8 layers
s1_gt_small = s1_layers[:256, :, :]
s1_8layers = [s1_layers[256*i:256*(i+1), :, :] for i in range(1, 9)]
s2_gt_small = s2_layers[:256, :, :]
s2_8layers = [s2_layers[256*i:256*(i+1), :, :] for i in range(1, 9)]

# Save individual layers
for i, layer in enumerate(s1_8layers):
    Image.fromarray(layer).save(OUT/f's1_layer_{i+1}.png')
for i, layer in enumerate(s2_8layers):
    Image.fromarray(layer).save(OUT/f's2_layer_{i+1}.png')

# ========== COMPOSITE FIGURE ==========
fig = plt.figure(figsize=(10, 14))
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3,
                       height_ratios=[1.2, 0.8, 0.8, 0.8])

# --- Row 0: BF + GT ---
ax_bf = fig.add_subplot(gs[0, 0])
ax_bf.imshow(s1_bf)
ax_bf.set_title('Brightfield', fontsize=10, fontweight='bold')
ax_bf.axis('off')

ax_gt = fig.add_subplot(gs[0, 1])
ax_gt.imshow(s1_gt)
ax_gt.set_title('Multi-color\nProjection (GT)', fontsize=10, fontweight='bold')
ax_gt.axis('off')

# BF with zoom insets for in-focus vs defocus
ax_bf2 = fig.add_subplot(gs[0, 2:])
ax_bf2.imshow(s1_bf)
ax_bf2.set_title('Brightfield with Depth Cues', fontsize=10, fontweight='bold')
ax_bf2.axis('off')

# Add zoom inset boxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
# In-focus region (sharp vessel)
rect1 = plt.Rectangle((150, 150), 120, 120, linewidth=2, edgecolor='lime', facecolor='none')
ax_bf2.add_patch(rect1)
ax_bf2.text(210, 145, 'In-focus', color='lime', fontsize=7, ha='center', fontweight='bold')

# Out-of-focus region (blurred vessel)
rect2 = plt.Rectangle((300, 50), 120, 120, linewidth=2, edgecolor='red', facecolor='none')
ax_bf2.add_patch(rect2)
ax_bf2.text(360, 45, 'Defocused', color='red', fontsize=7, ha='center', fontweight='bold')

# --- Row 1: 4 representative layers ---
selected = [0, 2, 5, 7]
layer_names = ['Layer 1\n(Top, ~0 um)', 'Layer 3\n(~23 um)', 'Layer 6\n(~57 um)', 'Layer 8\n(Bottom, ~80 um)']
for ci, (li, ln) in enumerate(zip(selected, layer_names)):
    ax_l = fig.add_subplot(gs[1, ci])
    ax_l.imshow(cv2.resize(s1_8layers[li], (512, 512), interpolation=cv2.INTER_LANCZOS4))
    ax_l.set_title(ln, fontsize=8, fontweight='bold')
    ax_l.axis('off')

# --- Row 2: Hue histogram + Per-layer intensity ---
ax_hue = fig.add_subplot(gs[2, :2])
hsv = cv2.cvtColor(s1_gt, cv2.COLOR_RGB2HSV)
h_ch, s_ch = hsv[:,:,0], hsv[:,:,1]
gray = cv2.cvtColor(s1_gt, cv2.COLOR_RGB2GRAY)
fg = (gray > 15) & (s_ch > 20)
hue_vals = h_ch[fg]
ax_hue.hist(hue_vals, bins=180, range=(0,180), color='gray', alpha=0.7, edgecolor='none')
for h0, h1, color in [(0,12,'#FF4444'), (155,180,'#FF4444'), (13,55,'#FFD700'), (85,140,'#4488FF')]:
    ax_hue.axvspan(h0, h1, alpha=0.15, color=color)
ax_hue.text(6, ax_hue.get_ylim()[1]*0.01, 'Bottom\n(Red)', ha='center', va='bottom', fontsize=7, fontweight='bold', color='#CC0000')
ax_hue.text(34, ax_hue.get_ylim()[1]*0.01, 'Middle\n(Yellow)', ha='center', va='bottom', fontsize=7, fontweight='bold', color='#AA8800')
ax_hue.text(112, ax_hue.get_ylim()[1]*0.01, 'Top\n(Blue)', ha='center', va='bottom', fontsize=7, fontweight='bold', color='#2266CC')
ax_hue.set_xlabel('Hue (OpenCV 0-179)', fontsize=9)
ax_hue.set_ylabel('Pixel Count', fontsize=9)
ax_hue.set_title('Hue Distribution of GT Projection', fontsize=10, fontweight='bold')
ax_hue.spines['top'].set_visible(False)
ax_hue.spines['right'].set_visible(False)

# Per-layer intensity
ax_int = fig.add_subplot(gs[2, 2:])
mean_ints = []
layer_colors = []
for i, layer in enumerate(s1_8layers):
    g = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
    mean_val = np.mean(g[g > 10]) if np.sum(g > 10) > 0 else 0
    mean_ints.append(mean_val)
    fg_l = g > 10
    if np.sum(fg_l) > 0:
        layer_colors.append(np.clip(np.mean(layer[fg_l], axis=0) / 255.0, 0, 1))
    else:
        layer_colors.append([0.5, 0.5, 0.5])

ax_int.barh(range(8), mean_ints, color=layer_colors, edgecolor='black', linewidth=0.5, height=0.7)
ax_int.set_yticks(range(8))
ax_int.set_yticklabels([f'L{i+1}' for i in range(8)], fontsize=8)
ax_int.set_xlabel('Mean Fluorescence Intensity (a.u.)', fontsize=9)
ax_int.set_title('Per-layer Intensity Profile', fontsize=10, fontweight='bold')
ax_int.invert_yaxis()
ax_int.spines['top'].set_visible(False)
ax_int.spines['right'].set_visible(False)
# Depth axis
ax_int2 = ax_int.twinx()
ax_int2.set_ylim(ax_int.get_ylim())
ax_int2.set_yticks(range(8))
ax_int2.set_yticklabels([f'{d:.0f} um' for d in np.linspace(0, 80, 8)], fontsize=7, color='gray')
ax_int2.spines['top'].set_visible(False)

# --- Row 3: Depth map + Overlap ---
ax_dm = fig.add_subplot(gs[3, :2])
h_img, w_img = s1_8layers[0].shape[:2]
depth_map = np.zeros((h_img, w_img))
cov_count = np.zeros((h_img, w_img))
for i, layer in enumerate(s1_8layers):
    g = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
    mask = g > 15
    depth_map[mask] += (i + 1)
    cov_count[mask] += 1
avg_d = np.where(cov_count > 0, depth_map / cov_count, np.nan)
im_d = ax_dm.imshow(avg_d, cmap='coolwarm', vmin=1, vmax=8)
ax_dm.set_title('Mean Depth Map', fontsize=10, fontweight='bold')
ax_dm.axis('off')
cb = plt.colorbar(im_d, ax=ax_dm, fraction=0.046, pad=0.04)
cb.set_label('Layer (1=Top, 8=Bottom)', fontsize=7)

ax_ov = fig.add_subplot(gs[3, 2:])
overlap_v = cov_count.copy()
overlap_v[overlap_v == 0] = np.nan
im_ov = ax_ov.imshow(overlap_v, cmap='YlOrRd', vmin=1, vmax=6)
ax_ov.set_title('Layer Overlap Count', fontsize=10, fontweight='bold')
ax_ov.axis('off')
cb2 = plt.colorbar(im_ov, ax=ax_ov, fraction=0.046, pad=0.04)
cb2.set_label('# overlapping layers', fontsize=7)

# Panel labels
panels = [(ax_bf, 'a'), (fig.axes[3], 'b'), (ax_hue, 'c'), (ax_dm, 'd')]
for ax_p, label in panels:
    ax_p.text(-0.12, 1.08, label, transform=ax_p.transAxes, fontsize=16, fontweight='bold', va='top')

plt.savefig(OUT/'dataset_composite_figure.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Composite figure saved!')

# ========== Hue histogram for both samples ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, gt, name in [(axes[0], s1_gt, 'Sample 1'), (axes[1], s2_gt, 'Sample 2')]:
    hsv_s = cv2.cvtColor(gt, cv2.COLOR_RGB2HSV)
    h_s, s_s = hsv_s[:,:,0], hsv_s[:,:,1]
    gray_s = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
    fg_s = (gray_s > 15) & (s_s > 20)
    hue_s = h_s[fg_s]
    ax.hist(hue_s, bins=180, range=(0,180), color='gray', alpha=0.7, edgecolor='none')
    for h0, h1, color in [(0,12,'#FF4444'), (155,180,'#FF4444'), (13,55,'#FFD700'), (85,140,'#4488FF')]:
        ax.axvspan(h0, h1, alpha=0.15, color=color)
    ax.set_xlabel('Hue (OpenCV 0-179)', fontsize=11)
    ax.set_ylabel('Pixel Count', fontsize=11)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.suptitle('Hue Distribution of Multi-color Depth-encoded Fluorescence', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT/'dataset_hue_histogram.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ========== Layer intensity for both samples ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, layers, name in [(axes[0], s1_8layers, 'Sample 1'), (axes[1], s2_8layers, 'Sample 2')]:
    mi = []
    lc = []
    for i, layer in enumerate(layers):
        g = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
        mi.append(np.mean(g[g > 10]) if np.sum(g > 10) > 0 else 0)
        fg_l = g > 10
        if np.sum(fg_l) > 0:
            lc.append(np.clip(np.mean(layer[fg_l], axis=0) / 255.0, 0, 1))
        else:
            lc.append([0.5, 0.5, 0.5])
    ax.barh(range(8), mi, color=lc, edgecolor='black', linewidth=0.5, height=0.7)
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'Layer {i+1}' for i in range(8)], fontsize=9)
    ax.set_xlabel('Mean Intensity (a.u.)', fontsize=11)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.suptitle('Per-layer Mean Fluorescence Intensity vs Depth', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT/'dataset_layer_intensity.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ========== Depth overlap for both samples ==========
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for row, (layers, name) in enumerate([(s1_8layers, 'Sample 1'), (s2_8layers, 'Sample 2')]):
    h_i, w_i = layers[0].shape[:2]
    dm = np.zeros((h_i, w_i))
    cc = np.zeros((h_i, w_i))
    for i, layer in enumerate(layers):
        g = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
        mask = g > 15
        dm[mask] += (i + 1)
        cc[mask] += 1
    ad = np.where(cc > 0, dm / cc, np.nan)
    ov = cc.copy(); ov[ov == 0] = np.nan

    im1 = axes[row, 0].imshow(ad, cmap='coolwarm', vmin=1, vmax=8)
    axes[row, 0].set_title(f'{name} - Mean Depth', fontsize=11, fontweight='bold')
    axes[row, 0].axis('off')
    plt.colorbar(im1, ax=axes[row, 0], fraction=0.046, pad=0.04)

    im2 = axes[row, 1].imshow(ov, cmap='YlOrRd', vmin=1, vmax=6)
    axes[row, 1].set_title(f'{name} - Overlap Count', fontsize=11, fontweight='bold')
    axes[row, 1].axis('off')
    plt.colorbar(im2, ax=axes[row, 1], fraction=0.046, pad=0.04)

plt.suptitle('Depth Distribution and Multi-layer Overlap Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT/'dataset_depth_overlap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print('All figures saved successfully!')
