"""
Mask threshold sweep: R threshold and B extension threshold variations.
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

DATA = Path(r'[1] Data')
OUT = Path(r'analysis_output/debug_pipeline')
OUT.mkdir(exist_ok=True)

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
dk2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

sid = '18-19-512'
gt = np.array(Image.open(DATA / 'Multi-color' / 'LSGAN' / f'{sid}_real_B.png'))
gt_blur = cv2.GaussianBlur(gt, (7,7), 0)
h, w = gt.shape[:2]; by = int(h*0.8)
R = gt_blur[:,:,0].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)

# ============================================================
# R mask sweep: threshold = 30(current), 35, 40, 45, 50
# ============================================================
r_thresholds = [30, 35, 40, 45, 50]

fig, axes = plt.subplots(len(r_thresholds), 3, figsize=(14, len(r_thresholds)*3.5))
fig.suptitle(f'{sid} — R Mask Threshold Sweep', fontsize=14, fontweight='bold')

for row, thr in enumerate(r_thresholds):
    r_mask = remove_small_objects(
        cv2.morphologyEx((R>thr).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)
    r_skel = skeletonize(r_mask)
    r_labels, r_n = ndimage.label(r_skel)

    # Binary mask
    axes[row,0].imshow(r_mask[:by], cmap='gray')
    axes[row,0].set_title(f'R>{thr} mask | {np.sum(r_mask[:by])}px', fontsize=10)

    # Skel on black
    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(r_skel[:by].astype(np.uint8), dk2)>0] = [230,70,70]
    axes[row,1].imshow(vis)
    axes[row,1].set_title(f'R skel | {r_n} comp, {np.sum(r_skel[:by])}px', fontsize=10)

    # Skel on GT
    vis = gt[:by].copy().astype(float) * 0.4
    vis[cv2.dilate(r_skel[:by].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))>0] = [230,70,70]
    axes[row,2].imshow(vis.astype(np.uint8))
    axes[row,2].set_title(f'R skel on GT (thr={thr})', fontsize=10)

    current = ' ← current' if thr == 30 else ''
    print(f'R>{thr}: mask={np.sum(r_mask[:by])}px skel={np.sum(r_skel[:by])}px comp={r_n}{current}')

for ax in axes.flat: ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / f'mask_sweep_R_{sid}.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved mask_sweep_R_{sid}.png\n')

# ============================================================
# B mask sweep: extension = 20(current), 30, 40, 50 (seed always 50)
# ============================================================
b_extensions = [20, 30, 40, 50]

fig, axes = plt.subplots(len(b_extensions), 3, figsize=(14, len(b_extensions)*3.5))
fig.suptitle(f'{sid} — B Mask Extension Threshold Sweep (seed=50)', fontsize=14, fontweight='bold')

for row, ext in enumerate(b_extensions):
    b_seed = B > 50
    b_low = B > ext
    b_lab, b_n_lab = ndimage.label(b_low)
    b_mask = np.zeros_like(b_low, dtype=bool)
    for i in range(1, b_n_lab+1):
        if np.any(b_seed[b_lab==i]): b_mask |= (b_lab==i)
    b_mask = remove_small_objects(
        cv2.morphologyEx(b_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2)>0, min_size=50)
    b_skel = skeletonize(b_mask)
    b_labels, b_n = ndimage.label(b_skel)

    # Binary mask
    axes[row,0].imshow(b_mask[:by], cmap='gray')
    axes[row,0].set_title(f'B ext>{ext} mask | {np.sum(b_mask[:by])}px', fontsize=10)

    # Skel on black
    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(b_skel[:by].astype(np.uint8), dk2)>0] = [70,140,240]
    axes[row,1].imshow(vis)
    axes[row,1].set_title(f'B skel | {b_n} comp, {np.sum(b_skel[:by])}px', fontsize=10)

    # Skel on GT
    vis = gt[:by].copy().astype(float) * 0.4
    vis[cv2.dilate(b_skel[:by].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))>0] = [70,140,240]
    axes[row,2].imshow(vis.astype(np.uint8))
    axes[row,2].set_title(f'B skel on GT (ext={ext})', fontsize=10)

    current = ' ← current' if ext == 20 else ''
    print(f'B ext>{ext}: mask={np.sum(b_mask[:by])}px skel={np.sum(b_skel[:by])}px comp={b_n}{current}')

for ax in axes.flat: ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / f'mask_sweep_B_{sid}.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved mask_sweep_B_{sid}.png\n')

# ============================================================
# Overlap comparison: R mask ∩ B mask at different thresholds
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
fig.suptitle(f'{sid} — R∩B Overlap at Different Thresholds', fontsize=13, fontweight='bold')

# Top row: R=30 fixed, B extension varies
r_mask_30 = remove_small_objects(
    cv2.morphologyEx((R>30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)
for col, ext in enumerate(b_extensions):
    b_seed = B > 50; b_low = B > ext
    b_lab, b_n_lab = ndimage.label(b_low)
    b_m = np.zeros_like(b_low, dtype=bool)
    for i in range(1, b_n_lab+1):
        if np.any(b_seed[b_lab==i]): b_m |= (b_lab==i)
    b_m = remove_small_objects(
        cv2.morphologyEx(b_m.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2)>0, min_size=50)

    overlap = r_mask_30[:by] & b_m[:by]
    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[r_mask_30[:by] & ~b_m[:by]] = [200, 60, 60]   # R only
    vis[b_m[:by] & ~r_mask_30[:by]] = [60, 120, 200]   # B only
    vis[overlap] = [200, 100, 200]                       # overlap
    axes[0,col].imshow(vis)
    ovl_pct = np.sum(overlap) / max(np.sum(r_mask_30[:by]),1) * 100
    axes[0,col].set_title(f'R>30 + B ext>{ext}\noverlap={np.sum(overlap)}px ({ovl_pct:.0f}% of R)', fontsize=9)

# Bottom row: B ext=30 fixed, R threshold varies
b_seed = B > 50; b_low = B > 30
b_lab, b_n_lab = ndimage.label(b_low)
b_mask_30 = np.zeros_like(b_low, dtype=bool)
for i in range(1, b_n_lab+1):
    if np.any(b_seed[b_lab==i]): b_mask_30 |= (b_lab==i)
b_mask_30 = remove_small_objects(
    cv2.morphologyEx(b_mask_30.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2)>0, min_size=50)

for col, thr in enumerate([30, 35, 40, 45]):
    r_m = remove_small_objects(
        cv2.morphologyEx((R>thr).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)

    overlap = r_m[:by] & b_mask_30[:by]
    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[r_m[:by] & ~b_mask_30[:by]] = [200, 60, 60]
    vis[b_mask_30[:by] & ~r_m[:by]] = [60, 120, 200]
    vis[overlap] = [200, 100, 200]
    axes[1,col].imshow(vis)
    ovl_pct = np.sum(overlap) / max(np.sum(r_m[:by]),1) * 100
    axes[1,col].set_title(f'R>{thr} + B ext>30\noverlap={np.sum(overlap)}px ({ovl_pct:.0f}% of R)', fontsize=9)

for ax in axes.flat: ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / f'mask_overlap_{sid}.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved mask_overlap_{sid}.png')
print('Done!')
