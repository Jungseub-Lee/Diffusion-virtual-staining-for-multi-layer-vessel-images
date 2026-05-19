"""
Figure 1E: Compact pipeline scheme
Single-plane BF → LBBDM → Depth-encoded FL → Layer separation → Per-layer quantification
Clean, publication-quality scheme for Advanced Healthcare Materials
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
from PIL import Image

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output\paper_panels_depth')
DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')

# Load sample images for thumbnails
s1 = np.array(Image.open(DATA / 'Original color stack data' / 'Sample 1_BF and GT(512x512 two images).png'))
bf_raw = s1[:512]
gt_raw = s1[512:]

# Crop to a nice representative region (top 80%, center)
cy, cx = 200, 256
crop = 90
bf_crop = bf_raw[cy-crop:cy+crop, cx-crop:cx+crop]
gt_crop = gt_raw[cy-crop:cy+crop, cx-crop:cx+crop]

# Create R and B channel views
import cv2
gt_blur = cv2.GaussianBlur(gt_raw, (7, 7), 0)
R = gt_blur[:,:,0].astype(float)
B = gt_blur[:,:,2].astype(float)

# Bottom layer (R>B) visualization
r_view = gt_raw.copy().astype(float)
r_mask = R > B
r_view[~r_mask] = r_view[~r_mask] * 0.15
r_crop = r_view[cy-crop:cy+crop, cx-crop:cx+crop].astype(np.uint8)

# Top layer (B>R) visualization
b_view = gt_raw.copy().astype(float)
b_mask = B > R
b_view[~b_mask] = b_view[~b_mask] * 0.15
b_crop = b_view[cy-crop:cy+crop, cx-crop:cx+crop].astype(np.uint8)

# Load skeleton panels
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage

h, w = gt_raw.shape[:2]
by = int(h * 0.8)
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dk3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Quick skeleton for thumbnail
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
diff = B - R

r_m = remove_small_objects(cv2.morphologyEx((R>30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)
r_sk = skeletonize(r_m) & (diff < 0)

b_seed = B>50; b_low = B>20
b_ll, nl = ndimage.label(b_low)
b_hyst = np.zeros_like(b_low, dtype=bool)
for i in range(1, nl+1):
    reg = b_ll == i
    if np.any(b_seed[reg]): b_hyst |= reg
b_hyst = remove_small_objects(cv2.morphologyEx(b_hyst.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2)>0, min_size=50)
b_sk = skeletonize(b_hyst) & (diff > 0)

# Combined skel view
skel_vis = np.zeros((h, w, 3), dtype=np.uint8)
skel_vis[cv2.dilate(r_sk.astype(np.uint8), dk3) > 0] = [230, 70, 70]
skel_vis[cv2.dilate(b_sk.astype(np.uint8), dk3) > 0] = [70, 140, 240]
skel_crop = skel_vis[cy-crop:cy+crop, cx-crop:cx+crop]

# ==================== BUILD FIGURE ====================
fig = plt.figure(figsize=(12, 2.2))

# Colors
C_BF = '#7f8c8d'
C_MODEL = '#2c3e50'
C_MULTI = '#8e44ad'
C_BOT = '#c0392b'
C_TOP = '#2980b9'
C_QUANT = '#27ae60'
C_ARROW = '#555555'
C_BG = '#f8f9fa'

# Layout: 5 stages horizontally
# [BF] → [LBBDM] → [Multi-color FL] → [Layer Sep] → [Quantification]
stages_x = [0.02, 0.22, 0.42, 0.62, 0.82]
stage_w = 0.16
img_y = 0.28
img_h = 0.55
label_y = 0.88

# Helper: add image thumbnail with border
def add_thumb(ax_pos, img, border_color, label_above='', label_below=''):
    ax = fig.add_axes(ax_pos)
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2)
    return ax

# Helper: add arrow between stages
def add_arrow(x1, x2, y, text='', color=C_ARROW):
    fig.text((x1 + x2) / 2, y, '\u2192', fontsize=18, ha='center', va='center',
             color=color, fontweight='bold', transform=fig.transFigure)
    if text:
        fig.text((x1 + x2) / 2, y - 0.13, text, fontsize=6, ha='center', va='center',
                 color=color, fontstyle='italic', transform=fig.transFigure)

# Stage 1: BF input
tw = 0.12; th = 0.5
add_thumb([stages_x[0] + 0.02, img_y, tw, th], bf_crop, C_BF)
fig.text(stages_x[0] + 0.02 + tw/2, label_y, 'Single-plane\nBrightfield', fontsize=7,
         ha='center', va='center', fontweight='bold', color=C_BF)

# Arrow 1
add_arrow(stages_x[0] + tw + 0.025, stages_x[1] + 0.02, img_y + th/2, 'Virtual\nStaining')

# Stage 2: Model (text box, no image)
box_x = stages_x[1] + 0.01
box_y = img_y + 0.05
box_w = tw + 0.02
box_h = th - 0.1
ax_model = fig.add_axes([box_x, box_y, box_w, box_h])
ax_model.set_xlim(0, 1); ax_model.set_ylim(0, 1)
ax_model.set_xticks([]); ax_model.set_yticks([])
ax_model.set_facecolor('#ecf0f1')
for spine in ax_model.spines.values():
    spine.set_edgecolor(C_MODEL); spine.set_linewidth(2)
ax_model.text(0.5, 0.65, 'LBBDM', fontsize=9, ha='center', va='center',
              fontweight='bold', color=C_MODEL)
ax_model.text(0.5, 0.35, 'Latent\nBrownian\nBridge', fontsize=5.5, ha='center', va='center',
              color='#555555')

fig.text(box_x + box_w/2, label_y, 'Generative\nModel', fontsize=7,
         ha='center', va='center', fontweight='bold', color=C_MODEL)

# Arrow 2
add_arrow(stages_x[1] + tw + 0.045, stages_x[2] + 0.02, img_y + th/2)

# Stage 3: Multi-color FL output
add_thumb([stages_x[2] + 0.02, img_y, tw, th], gt_crop, C_MULTI)
fig.text(stages_x[2] + 0.02 + tw/2, label_y, 'Depth-encoded\nFluorescence', fontsize=7,
         ha='center', va='center', fontweight='bold', color=C_MULTI)

# Arrow 3
add_arrow(stages_x[2] + tw + 0.025, stages_x[3] + 0.01, img_y + th/2, 'RGB Channel\nSeparation')

# Stage 4: Layer separation (two small images stacked)
sub_h = th * 0.47
sub_gap = th * 0.06
add_thumb([stages_x[3] + 0.02, img_y + sub_h + sub_gap, tw, sub_h], b_crop, C_TOP)
add_thumb([stages_x[3] + 0.02, img_y, tw, sub_h], r_crop, C_BOT)

# Small labels
fig.text(stages_x[3] + 0.02 + tw + 0.005, img_y + sub_h + sub_gap + sub_h/2,
         'Top\n(B>R)', fontsize=5, ha='left', va='center', color=C_TOP, fontweight='bold')
fig.text(stages_x[3] + 0.02 + tw + 0.005, img_y + sub_h/2,
         'Bot\n(R>B)', fontsize=5, ha='left', va='center', color=C_BOT, fontweight='bold')

fig.text(stages_x[3] + 0.02 + tw/2, label_y, 'Depth-layer\nSeparation', fontsize=7,
         ha='center', va='center', fontweight='bold', color='#555555')

# Arrow 4
add_arrow(stages_x[3] + tw + 0.055, stages_x[4] + 0.01, img_y + th/2, 'Skeletonize\n& Quantify')

# Stage 5: Quantification (skeleton + metrics)
add_thumb([stages_x[4] + 0.02, img_y, tw, th], skel_crop, C_QUANT)
fig.text(stages_x[4] + 0.02 + tw/2, label_y, 'Per-layer\nMorphometrics', fontsize=7,
         ha='center', va='center', fontweight='bold', color=C_QUANT)

# Metrics labels below skeleton
metrics_text = 'Area | Length\nJunctions | Endpoints'
fig.text(stages_x[4] + 0.02 + tw/2, img_y - 0.08, metrics_text, fontsize=5,
         ha='center', va='center', color=C_QUANT, fontstyle='italic')

# Panel label
fig.text(0.005, 0.95, 'E', fontsize=14, fontweight='bold', va='top', ha='left')

plt.savefig(OUT / 'fig1e_pipeline.png', dpi=400, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print('Saved fig1e_pipeline.png')
print(f'Size: {(OUT / "fig1e_pipeline.png").stat().st_size / 1024:.0f} KB')
