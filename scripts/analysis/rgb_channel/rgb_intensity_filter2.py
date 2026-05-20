"""
B skel intensity 필터: blur된 원본의 B채널 value 사용
R skel도 동일하게 blur된 원본의 R채널 value 사용
gt_blur (7,7) 기준 — 추가 blur 없이 직접 사용
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

# blur된 원본에서 직접 채널 추출
R = gt_blur[:,:,0].astype(np.float32)
G = gt_blur[:,:,1].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)

# ===== R skeleton (thresh=30, post-blur) =====
r_mask = R > 30
r_mask = cv2.morphologyEx(r_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
r_mask = remove_small_objects(r_mask, min_size=50)
r_skel = skeletonize(r_mask)

# ===== B skeleton (hysteresis seed=50, low=20) =====
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

# ===== 핵심: blur된 원본의 채널값을 직접 skel pixel에서 읽기 =====
# 추가 blur 없이 gt_blur의 R, B 채널 그대로 사용

intensity_thresholds = [20, 30, 40, 50, 60, 70]

fig, axes = plt.subplots(len(intensity_thresholds), 4, figsize=(32, 8*len(intensity_thresholds)))
fig.suptitle('Skeleton filtered by gt_blur channel value (no extra blur)',
             fontsize=16, fontweight='bold')

for row, int_thresh in enumerate(intensity_thresholds):
    # B skel: gt_blur의 B값이 int_thresh 이상인 skel pixel만 남기기
    b_filtered = b_skel & (B >= int_thresh)
    r_filtered = r_skel & (R >= int_thresh)

    # Panel 1: B skel on B channel heatmap
    b_vis = np.zeros((h, w, 3), dtype=np.uint8)
    # B channel 배경
    b_bg = (B / 255.0 * 0.5 * 255).astype(np.uint8)
    b_vis[:,:,2] = b_bg
    b_vis[:,:,1] = (b_bg * 0.3).astype(np.uint8)
    # full skel dim
    b_skel_d = cv2.dilate(b_skel.astype(np.uint8), dk3) > 0
    b_vis[b_skel_d] = [80, 80, 140]
    # filtered bright
    b_filt_d = cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0
    b_vis[b_filt_d] = [50, 220, 255]
    axes[row, 0].imshow(b_vis)
    b_kept = int(np.sum(b_filtered))
    b_total = int(np.sum(b_skel))
    axes[row, 0].set_title(f'B skel: B≥{int_thresh}\nkept={b_kept}/{b_total} ({100*b_kept/max(b_total,1):.0f}%)',
                           fontsize=11)

    # Panel 2: R skel on R channel heatmap
    r_vis = np.zeros((h, w, 3), dtype=np.uint8)
    r_bg = (R / 255.0 * 0.5 * 255).astype(np.uint8)
    r_vis[:,:,0] = r_bg
    r_vis[:,:,1] = (r_bg * 0.3).astype(np.uint8)
    r_skel_d = cv2.dilate(r_skel.astype(np.uint8), dk3) > 0
    r_vis[r_skel_d] = [140, 80, 80]
    r_filt_d = cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0
    r_vis[r_filt_d] = [255, 230, 50]
    axes[row, 1].imshow(r_vis)
    r_kept = int(np.sum(r_filtered))
    r_total = int(np.sum(r_skel))
    axes[row, 1].set_title(f'R skel: R≥{int_thresh}\nkept={r_kept}/{r_total} ({100*r_kept/max(r_total,1):.0f}%)',
                           fontsize=11)

    # Panel 3: Filtered R+B on GT
    on_gt = gt_raw.copy().astype(float) * 0.3
    on_gt[cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0] = [255, 80, 80]
    on_gt[cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0] = [80, 150, 255]
    both = (cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0) & \
           (cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0)
    on_gt[both] = [200, 50, 200]
    axes[row, 2].imshow(on_gt.astype(np.uint8))
    axes[row, 2].set_title(f'Filtered R(red)+B(blue) on GT\nchannel≥{int_thresh}', fontsize=11)

    # Panel 4: Gap 위치
    r_gap = r_skel & ~r_filtered
    b_gap = b_skel & ~b_filtered
    gap_vis = np.zeros((h, w, 3), dtype=np.uint8)
    gap_vis[cv2.dilate(r_filtered.astype(np.uint8), dk3) > 0] = [200, 60, 60]
    gap_vis[cv2.dilate(b_filtered.astype(np.uint8), dk3) > 0] = [60, 120, 200]
    gap_vis[cv2.dilate(r_gap.astype(np.uint8), dk4) > 0] = [255, 200, 100]
    gap_vis[cv2.dilate(b_gap.astype(np.uint8), dk4) > 0] = [100, 255, 200]
    axes[row, 3].imshow(gap_vis)
    axes[row, 3].set_title(f'Kept(R/B) + Gap(yellow/cyan)\nR-gap={int(np.sum(r_gap))}px B-gap={int(np.sum(b_gap))}px',
                           fontsize=11)

for ax in axes.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'rgb_intensity_filter_direct.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved rgb_intensity_filter_direct.png')

# ===== Figure 2: B채널 value 분포 on skeleton =====
fig2, axes2 = plt.subplots(2, 3, figsize=(27, 16))
fig2.suptitle('Channel value distribution on skeleton pixels (from gt_blur)', fontsize=16, fontweight='bold')

# B value heatmap on B skeleton
b_val_on_skel = np.zeros((h, w), dtype=np.float32)
b_val_on_skel[b_skel] = B[b_skel]
b_val_vis = cv2.dilate(b_val_on_skel, dk4)
axes2[0, 0].imshow(b_val_vis, cmap='coolwarm', vmin=0, vmax=150)
axes2[0, 0].set_title('B channel value on B skel\n(blue=low, red=high)', fontsize=12)
plt.colorbar(axes2[0, 0].images[0], ax=axes2[0, 0], shrink=0.7)

# R value heatmap on R skeleton
r_val_on_skel = np.zeros((h, w), dtype=np.float32)
r_val_on_skel[r_skel] = R[r_skel]
r_val_vis = cv2.dilate(r_val_on_skel, dk4)
axes2[0, 1].imshow(r_val_vis, cmap='coolwarm', vmin=0, vmax=150)
axes2[0, 1].set_title('R channel value on R skel\n(blue=low, red=high)', fontsize=12)
plt.colorbar(axes2[0, 1].images[0], ax=axes2[0, 1], shrink=0.7)

# Histogram: B values at B skel pixels
b_vals = B[b_skel]
axes2[0, 2].hist(b_vals, bins=50, color='steelblue', edgecolor='black', alpha=0.8)
axes2[0, 2].axvline(30, color='red', linestyle='--', linewidth=2, label='thresh=30')
axes2[0, 2].axvline(50, color='orange', linestyle='--', linewidth=2, label='thresh=50')
axes2[0, 2].set_title('B channel value distribution\n(on B skel pixels)', fontsize=12)
axes2[0, 2].set_xlabel('B channel value')
axes2[0, 2].set_ylabel('pixel count')
axes2[0, 2].legend()

# Histogram: R values at R skel pixels
r_vals = R[r_skel]
axes2[1, 0].hist(r_vals, bins=50, color='indianred', edgecolor='black', alpha=0.8)
axes2[1, 0].axvline(30, color='red', linestyle='--', linewidth=2, label='thresh=30')
axes2[1, 0].axvline(50, color='orange', linestyle='--', linewidth=2, label='thresh=50')
axes2[1, 0].set_title('R channel value distribution\n(on R skel pixels)', fontsize=12)
axes2[1, 0].set_xlabel('R channel value')
axes2[1, 0].set_ylabel('pixel count')
axes2[1, 0].legend()

# GT reference
axes2[1, 1].imshow(gt_blur)
axes2[1, 1].set_title('GT (blurred)', fontsize=12)

# Base skel reference
base_vis = gt_raw.copy().astype(float) * 0.3
base_vis[cv2.dilate(base_skel.astype(np.uint8), dk4) > 0] = [0, 255, 0]
axes2[1, 2].imshow(base_vis.astype(np.uint8))
axes2[1, 2].set_title('Base skel (reference)', fontsize=12)

for ax in axes2.flat:
    if not ax.has_data():
        ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'rgb_skel_value_distribution.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved rgb_skel_value_distribution.png')
print('Done!')
