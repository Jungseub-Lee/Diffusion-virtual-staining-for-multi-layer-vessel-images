import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

OUT = Path('analysis_output/panels')
OUT.mkdir(exist_ok=True)
DATA = Path('[1] Data/Original color stack data')

# Load
s1_bf_gt = np.array(Image.open(DATA/'Sample 1_BF and GT(512x512 two images).png'))
s1_layers_img = np.array(Image.open(DATA/'Sample 1_GT and 8 layer images(256x256 nine images).png'))
s2_bf_gt = np.array(Image.open(DATA/'Sample 2_BF and GT(512x512 two images).png'))
s2_layers_img = np.array(Image.open(DATA/'Sample 2_GT and 8 layer images(256x256 nine images).png'))

s1_bf, s1_gt = s1_bf_gt[:512], s1_bf_gt[512:]
s2_bf, s2_gt = s2_bf_gt[:512], s2_bf_gt[512:]
s1_layers = [s1_layers_img[256*i:256*(i+1)] for i in range(1, 9)]
s2_layers = [s2_layers_img[256*i:256*(i+1)] for i in range(1, 9)]

# Save individual images (high res, no titles - titles go in PPT)
for name, img in [('panel_a_bf', s1_bf), ('panel_a_gt', s1_gt),
                   ('panel_a_bf_s2', s2_bf), ('panel_a_gt_s2', s2_gt)]:
    Image.fromarray(img).save(OUT/f'{name}.png', dpi=(300,300))

for i, layer in enumerate(s1_layers):
    img_up = cv2.resize(layer, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    Image.fromarray(img_up).save(OUT/f'panel_b_L{i+1}.png', dpi=(300,300))
for i, layer in enumerate(s2_layers):
    img_up = cv2.resize(layer, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    Image.fromarray(img_up).save(OUT/f'panel_b_s2_L{i+1}.png', dpi=(300,300))

# Panel c: merged groups
def merge_layers(layers, indices):
    merged = np.zeros_like(layers[0], dtype=np.float64)
    for idx in indices:
        merged = np.maximum(merged, layers[idx].astype(np.float64))
    return np.clip(merged, 0, 255).astype(np.uint8)

groups = [[0,1,2], [3,4], [5,6,7]]
group_names_short = ['top', 'middle', 'bottom']
for gi, (indices, gn) in enumerate(zip(groups, group_names_short)):
    mg = merge_layers(s1_layers, indices)
    mg_up = cv2.resize(mg, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    Image.fromarray(mg_up).save(OUT/f'panel_c_{gn}.png', dpi=(300,300))

# ===== Panel d: Hue histogram (FIXED - shift hue so red is continuous) =====
SHIFT = 30  # shift hue by 30 so red wraps into continuous range

fig, ax = plt.subplots(figsize=(5, 3.5))

hsv = cv2.cvtColor(s1_gt, cv2.COLOR_RGB2HSV)
h_ch, s_ch = hsv[:,:,0], hsv[:,:,1]
gray = cv2.cvtColor(s1_gt, cv2.COLOR_RGB2GRAY)
fg = (gray > 15) & (s_ch > 20)
hue_raw = h_ch[fg].astype(np.int32)
hue_shifted = (hue_raw + SHIFT) % 180

ax.hist(hue_shifted, bins=180, range=(0,180), color='#666666', alpha=0.8, edgecolor='none')

# Shifted band positions:
# Red (0-12 + 155-179) → shifted: (30-42) + (5-29) = 5-42
# Yellow (13-55) → shifted: 43-85
# Blue (85-140) → shifted: 115-170
bands = [
    (5, 42, '#FF4444', 0.2, 'Bottom (Red)\nL6-L8'),
    (43, 85, '#FFD700', 0.2, 'Middle (Yellow)\nL4-L5'),
    (115, 170, '#4488FF', 0.2, 'Top (Blue)\nL1-L3'),
]
for h0, h1, color, alpha, label in bands:
    ax.axvspan(h0, h1, alpha=alpha, color=color)
    ymax = ax.get_ylim()[1]
    ax.text((h0+h1)/2, ymax*0.88, label, ha='center', va='top', fontsize=8,
            fontweight='bold', color=color.replace('FF4444','CC0000').replace('FFD700','AA8800').replace('4488FF','2266CC'))

# Create readable x-axis: show original hue values
tick_positions = [0, 30, 60, 90, 120, 150, 180]
tick_labels = [str((t - SHIFT) % 180) for t in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=9)
ax.set_xlabel('Hue (shifted for visualization)', fontsize=10)
ax.set_ylabel('Pixel Count', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(OUT/'panel_d_hue.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ===== Panel d: Per-layer intensity =====
fig, ax = plt.subplots(figsize=(4.5, 3.5))

group_color_map = {0:'#4488FF', 1:'#4488FF', 2:'#4488FF',
                   3:'#FFD700', 4:'#FFD700',
                   5:'#FF4444', 6:'#FF4444', 7:'#FF4444'}

mean_ints = []
bar_colors = []
for i, layer in enumerate(s1_layers):
    g = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
    mean_val = np.mean(g[g > 10]) if np.sum(g > 10) > 0 else 0
    mean_ints.append(mean_val)
    bar_colors.append(group_color_map[i])

y_pos = np.arange(8)
bars = ax.barh(y_pos, mean_ints, color=bar_colors, edgecolor='black', linewidth=0.5, height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels([f'L{i+1}' for i in range(8)], fontsize=9)
ax.set_xlabel('Mean Fluorescence Intensity (a.u.)', fontsize=10)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=9)

# Depth axis on right
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(y_pos)
depth_vals = np.linspace(0, 80, 8)
ax2.set_yticklabels([f'{d:.0f} \u03bcm' for d in depth_vals], fontsize=8, color='#888888')
ax2.spines['top'].set_visible(False)
ax2.tick_params(labelsize=8)

# Group brackets with text
bracket_info = [(0, 2, 'Top\n(Blue)', '#4488FF'), (3, 4, 'Mid\n(Yel)', '#FFD700'), (5, 7, 'Bot\n(Red)', '#FF4444')]
for y0, y1, label, color in bracket_info:
    mid = (y0 + y1) / 2
    ax.annotate(label, xy=(-0.22, mid), xycoords=('axes fraction', 'data'),
                fontsize=6, fontweight='bold', color=color, ha='center', va='center')

plt.tight_layout()
plt.savefig(OUT/'panel_d_intensity.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ===== Panel e: Depth maps =====
for sample_name, layers in [('s1', s1_layers), ('s2', s2_layers)]:
    fig, ax = plt.subplots(figsize=(4, 3.5))
    h_img, w_img = layers[0].shape[:2]
    depth_map = np.zeros((h_img, w_img))
    cov_count = np.zeros((h_img, w_img))
    for i, layer in enumerate(layers):
        g = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
        mask = g > 15
        depth_map[mask] += (i + 1)
        cov_count[mask] += 1
    avg_d = np.where(cov_count > 0, depth_map / cov_count, np.nan)
    im = ax.imshow(avg_d, cmap='coolwarm', vmin=1, vmax=8)
    ax.axis('off')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cb.set_label('Layer (1=Top, 8=Bottom)', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(OUT/f'panel_e_depth_{sample_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

print('All individual panels saved to analysis_output/panels/')
import os
for f in sorted(os.listdir(OUT)):
    print(f'  {f}')
