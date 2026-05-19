"""
B skel: B > R인 영역만 남기기 (B dominant = 순수 파란 혈관)
R skel: R > B인 영역만 남기기 (R dominant = 순수 빨간 혈관)
crossing에서는 두 채널이 비슷해지므로 자연스럽게 gap 발생
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')
DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')

s1 = np.array(Image.open(DATA / 'Original color stack data' / 'Sample 1_BF and GT(512x512 two images).png'))
gt_raw = s1[512:]
gt_blur = cv2.GaussianBlur(gt_raw, (7, 7), 0)
h, w = gt_raw.shape[:2]

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dk3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

R = gt_blur[:,:,0].astype(np.float32)
G = gt_blur[:,:,1].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)

# ===== Skeletons =====
# R skeleton (thresh=30)
r_mask = R > 30
r_mask = cv2.morphologyEx(r_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
r_mask = remove_small_objects(r_mask, min_size=50)
r_skel = skeletonize(r_mask)

# B skeleton (hysteresis seed=50, low=20)
b_seed = B > 50
b_low = B > 20
b_low_l, n_l = ndimage.label(b_low)
b_hyst = np.zeros_like(b_low, dtype=bool)
for i in range(1, n_l + 1):
    region = b_low_l == i
    if np.any(b_seed[region]):
        b_hyst |= region
b_hyst = cv2.morphologyEx(b_hyst.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0
b_hyst = remove_small_objects(b_hyst, min_size=50)
b_skel = skeletonize(b_hyst)

# Base skel (참고)
vessel_all = remove_small_objects(
    cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0,
    min_size=100)
base_skel = skeletonize(vessel_all)

# ===== B-R difference map =====
diff_BR = B - R  # positive = B dominant, negative = R dominant

# ===== Figure 1: B-R difference heatmap + dominant filtering =====
fig1, axes1 = plt.subplots(2, 4, figsize=(36, 16))
fig1.suptitle('B-R Dominance: B skel keeps B>R, R skel keeps R>B', fontsize=16, fontweight='bold')

# B-R difference heatmap (전체)
im0 = axes1[0,0].imshow(diff_BR, cmap='coolwarm', vmin=-100, vmax=100)
axes1[0,0].set_title('B - R difference\n(blue=B dominant, red=R dominant)', fontsize=12)
plt.colorbar(im0, ax=axes1[0,0], shrink=0.7)

# GT reference
axes1[0,1].imshow(gt_blur)
axes1[0,1].set_title('GT (blurred)', fontsize=12)

# B>R mask
b_dominant = diff_BR > 0
axes1[0,2].imshow(b_dominant, cmap='Blues')
axes1[0,2].set_title('B > R region (B dominant)', fontsize=12)

# R>B mask
r_dominant = diff_BR < 0
axes1[0,3].imshow(r_dominant, cmap='Reds')
axes1[0,3].set_title('R > B region (R dominant)', fontsize=12)

# ===== Margin 변형: B > R + margin =====
margins = [0, 5, 10, 20]

fig2, axes2 = plt.subplots(len(margins), 4, figsize=(36, 8*len(margins)))
fig2.suptitle('Dominance filter with margin: B skel where B>R+margin, R skel where R>B+margin',
              fontsize=16, fontweight='bold')

for row, margin in enumerate(margins):
    b_dom = diff_BR > margin   # B > R + margin
    r_dom = diff_BR < -margin  # R > B + margin

    b_filtered = b_skel & b_dom
    r_filtered = r_skel & r_dom

    b_gap = b_skel & ~b_filtered
    r_gap = r_skel & ~r_filtered

    # Panel 1: B filtered skel on B channel
    b_vis = np.zeros((h, w, 3), dtype=np.uint8)
    b_bg = np.clip(B * 0.4, 0, 255).astype(np.uint8)
    b_vis[:,:,2] = b_bg
    b_vis[cv2.dilate(b_skel.astype(np.uint8), dk3) > 0] = [60, 60, 120]
    b_vis[cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0] = [50, 220, 255]
    axes2[row, 0].imshow(b_vis)
    b_kept = int(np.sum(b_filtered))
    b_total = int(np.sum(b_skel))
    axes2[row, 0].set_title(f'B skel (B>R+{margin})\nkept={b_kept}/{b_total} ({100*b_kept/max(b_total,1):.0f}%)',
                            fontsize=11)

    # Panel 2: R filtered skel on R channel
    r_vis = np.zeros((h, w, 3), dtype=np.uint8)
    r_bg = np.clip(R * 0.4, 0, 255).astype(np.uint8)
    r_vis[:,:,0] = r_bg
    r_vis[cv2.dilate(r_skel.astype(np.uint8), dk3) > 0] = [120, 60, 60]
    r_vis[cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0] = [255, 230, 50]
    axes2[row, 1].imshow(r_vis)
    r_kept = int(np.sum(r_filtered))
    r_total = int(np.sum(r_skel))
    axes2[row, 1].set_title(f'R skel (R>B+{margin})\nkept={r_kept}/{r_total} ({100*r_kept/max(r_total,1):.0f}%)',
                            fontsize=11)

    # Panel 3: Combined on GT
    on_gt = gt_raw.copy().astype(float) * 0.3
    on_gt[cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0] = [255, 80, 80]
    on_gt[cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0] = [80, 150, 255]
    both = (cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0) & \
           (cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0)
    on_gt[both] = [200, 50, 200]
    axes2[row, 2].imshow(on_gt.astype(np.uint8))
    axes2[row, 2].set_title(f'R(red)+B(blue) on GT\nmargin={margin}', fontsize=11)

    # Panel 4: Gap analysis
    gap_vis = np.zeros((h, w, 3), dtype=np.uint8)
    gap_vis[cv2.dilate(r_filtered.astype(np.uint8), dk3) > 0] = [200, 60, 60]
    gap_vis[cv2.dilate(b_filtered.astype(np.uint8), dk3) > 0] = [60, 120, 200]
    gap_vis[cv2.dilate(r_gap.astype(np.uint8), dk4) > 0] = [255, 200, 100]
    gap_vis[cv2.dilate(b_gap.astype(np.uint8), dk4) > 0] = [100, 255, 200]
    axes2[row, 3].imshow(gap_vis)
    axes2[row, 3].set_title(f'Gaps: R(yellow)={int(np.sum(r_gap))}px B(cyan)={int(np.sum(b_gap))}px',
                            fontsize=11)

for ax in axes2.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'rgb_dominant_filter_margins.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved rgb_dominant_filter_margins.png')

# ===== Figure 3: B-R on skeleton heatmap + histogram =====
fig3, axes3 = plt.subplots(2, 3, figsize=(27, 16))
fig3.suptitle('B-R value on skeleton pixels', fontsize=16, fontweight='bold')

# B-R value on B skeleton
br_on_bskel = np.full((h, w), np.nan, dtype=np.float32)
br_on_bskel[b_skel] = diff_BR[b_skel]
br_vis = np.zeros((h, w), dtype=np.float32)
br_vis[b_skel] = diff_BR[b_skel]
br_vis_d = cv2.dilate(br_vis, dk4)
# mask out non-skel
mask_d = cv2.dilate(b_skel.astype(np.uint8), dk4) > 0
br_vis_d[~mask_d] = np.nan
im1 = axes3[0,0].imshow(br_vis_d, cmap='coolwarm', vmin=-80, vmax=80)
axes3[0,0].set_title('(B-R) on B skeleton\nblue=R dominant, red=B dominant', fontsize=12)
plt.colorbar(im1, ax=axes3[0,0], shrink=0.7)

# B-R value on R skeleton
rr_on_rskel = np.zeros((h, w), dtype=np.float32)
rr_on_rskel[r_skel] = diff_BR[r_skel]
rr_vis_d = cv2.dilate(rr_on_rskel, dk4)
mask_r = cv2.dilate(r_skel.astype(np.uint8), dk4) > 0
rr_vis_d[~mask_r] = np.nan
im2 = axes3[0,1].imshow(rr_vis_d, cmap='coolwarm', vmin=-80, vmax=80)
axes3[0,1].set_title('(B-R) on R skeleton\nblue=R dominant, red=B dominant', fontsize=12)
plt.colorbar(im2, ax=axes3[0,1], shrink=0.7)

# GT ref
axes3[0,2].imshow(gt_blur)
axes3[0,2].set_title('GT (blurred)', fontsize=12)

# Histogram: B-R at B skel
br_vals_b = diff_BR[b_skel]
axes3[1,0].hist(br_vals_b, bins=60, color='steelblue', edgecolor='black', alpha=0.8)
axes3[1,0].axvline(0, color='black', linestyle='-', linewidth=2, label='B=R')
axes3[1,0].axvline(10, color='orange', linestyle='--', linewidth=2, label='margin=10')
axes3[1,0].set_title('(B-R) distribution on B skel pixels', fontsize=12)
axes3[1,0].set_xlabel('B - R value')
axes3[1,0].legend()

# Histogram: B-R at R skel (should be mostly negative)
br_vals_r = diff_BR[r_skel]
axes3[1,1].hist(br_vals_r, bins=60, color='indianred', edgecolor='black', alpha=0.8)
axes3[1,1].axvline(0, color='black', linestyle='-', linewidth=2, label='B=R')
axes3[1,1].axvline(-10, color='orange', linestyle='--', linewidth=2, label='margin=-10')
axes3[1,1].set_title('(B-R) distribution on R skel pixels', fontsize=12)
axes3[1,1].set_xlabel('B - R value')
axes3[1,1].legend()

# Base ref
base_vis = gt_raw.copy().astype(float) * 0.3
base_vis[cv2.dilate(base_skel.astype(np.uint8), dk4) > 0] = [0, 255, 0]
axes3[1,2].imshow(base_vis.astype(np.uint8))
axes3[1,2].set_title('Base skel (reference)', fontsize=12)

for ax in axes3.flat:
    ax.axis('off') if not ax.has_data() else None
plt.tight_layout()
plt.savefig(OUT / 'rgb_dominant_histogram.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved rgb_dominant_histogram.png')
print('Done!')
