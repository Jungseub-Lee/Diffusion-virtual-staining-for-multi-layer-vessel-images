"""
R: thresh=30 (post-blur) 확정
B: 구멍 메우기 전략 비교
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
k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
dk3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

R = gt_blur[:,:,0].astype(np.float32)
G = gt_blur[:,:,1].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)

# Base skel (참고)
vessel_all = remove_small_objects(
    cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0,
    min_size=100)
base_skel = skeletonize(vessel_all)

def skel_features(skel):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    endpoints = np.argwhere(nb == 1)
    junctions = np.argwhere(nb >= 3)
    return endpoints, junctions

# ===== R: thresh=30 post-blur 확정 =====
r_mask = R > 30
r_mask = cv2.morphologyEx(r_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
r_mask = remove_small_objects(r_mask, min_size=50)
r_skel = skeletonize(r_mask)

# ===== B channel: 4가지 전략 비교 =====

# Strategy 1: Raw B > 30 (현재 방식, 구멍 있음)
b_raw30 = B > 30
b_raw30 = cv2.morphologyEx(b_raw30.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
b_raw30 = remove_small_objects(b_raw30, min_size=50)

# Strategy 2: B ratio = B/(R+G+B) > threshold
total = R + G + B + 1e-6
b_ratio = B / total
b_ratio_mask = b_ratio > 0.30  # B가 전체의 30% 이상
b_ratio_mask = cv2.morphologyEx(b_ratio_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
b_ratio_mask = remove_small_objects(b_ratio_mask, min_size=50)

# Strategy 3: Raw B > 30 + bigger morphological closing
b_close = B > 30
b_close = cv2.morphologyEx(b_close.astype(np.uint8)*255, cv2.MORPH_CLOSE, k7, iterations=3) > 0
b_close = remove_small_objects(b_close, min_size=50)

# Strategy 4: Hysteresis — seed: B>50, expand: B>20
b_seed = B > 50
b_low = B > 20
b_seed_labeled, n_seed = ndimage.label(b_seed)
b_low_labeled, n_low = ndimage.label(b_low)
# hysteresis: low region이 seed와 연결되면 keep
b_hysteresis = np.zeros_like(b_low, dtype=bool)
for i in range(1, n_low + 1):
    region = b_low_labeled == i
    if np.any(b_seed[region]):
        b_hysteresis |= region
b_hysteresis = cv2.morphologyEx(b_hysteresis.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0
b_hysteresis = remove_small_objects(b_hysteresis, min_size=50)

# Strategy 5: Ratio + Hysteresis 결합
b_ratio_seed = b_ratio > 0.35
b_ratio_low = b_ratio > 0.20
b_ratio_seed_l, n_rs = ndimage.label(b_ratio_seed)
b_ratio_low_l, n_rl = ndimage.label(b_ratio_low)
b_ratio_hyst = np.zeros_like(b_ratio_low, dtype=bool)
for i in range(1, n_rl + 1):
    region = b_ratio_low_l == i
    if np.any(b_ratio_seed[region]):
        b_ratio_hyst |= region
b_ratio_hyst = cv2.morphologyEx(b_ratio_hyst.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0
b_ratio_hyst = remove_small_objects(b_ratio_hyst, min_size=50)

strategies = [
    ('Raw B>30\n(current)', b_raw30),
    ('B ratio>0.30', b_ratio_mask),
    ('Raw B>30\n+ close k7×3', b_close),
    ('Hysteresis\nseed=50, low=20', b_hysteresis),
    ('Ratio Hyst\nseed=0.35, low=0.20', b_ratio_hyst),
]

# ===== Figure: 5가지 전략 mask + skeleton 비교 =====
fig, axes = plt.subplots(5, 4, figsize=(32, 36))
fig.suptitle('B Channel: Hole-filling strategies comparison', fontsize=16, fontweight='bold')

for row, (name, mask) in enumerate(strategies):
    skel = skeletonize(mask)
    eps, jns = skel_features(skel)

    # Mask
    axes[row, 0].imshow(mask, cmap='gray')
    axes[row, 0].set_title(f'{name}\nmask px={np.sum(mask)}', fontsize=11)

    # Mask zoomed at crossing
    # Find a crossing region (center-ish area)
    cy, cx = h//2, w//2
    crop = 80
    axes[row, 1].imshow(mask[cy-crop:cy+crop, cx-crop:cx+crop], cmap='gray')
    axes[row, 1].set_title(f'Zoom crossing region', fontsize=11)

    # Skeleton on mask
    skel_vis = np.zeros((h, w, 3), dtype=np.uint8)
    skel_vis[mask] = [30, 40, 80]
    skel_vis[cv2.dilate(skel.astype(np.uint8), dk3) > 0] = [100, 150, 255]
    axes[row, 2].imshow(skel_vis)
    axes[row, 2].set_title(f'B skel: EP={len(eps)} JN={len(jns)}', fontsize=11)

    # On GT with R skel
    on_gt = gt_raw.copy().astype(float) * 0.3
    on_gt[cv2.dilate(r_skel.astype(np.uint8), dk4) > 0] = [255, 80, 80]
    on_gt[cv2.dilate(skel.astype(np.uint8), dk4) > 0] = [80, 130, 255]
    both = (cv2.dilate(r_skel.astype(np.uint8), dk4) > 0) & \
           (cv2.dilate(skel.astype(np.uint8), dk4) > 0)
    on_gt[both] = [200, 50, 200]
    axes[row, 3].imshow(on_gt.astype(np.uint8))
    axes[row, 3].set_title(f'R(red)+B(blue) on GT', fontsize=11)

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig(OUT / 'rgb_B_hole_strategies.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved rgb_B_hole_strategies.png')

# ===== Figure 2: B ratio heatmap =====
fig2, axes2 = plt.subplots(1, 4, figsize=(28, 6))
fig2.suptitle('Channel Analysis', fontsize=14, fontweight='bold')

axes2[0].imshow(R, cmap='Reds', vmin=0, vmax=255)
axes2[0].set_title('R channel')
axes2[1].imshow(B, cmap='Blues', vmin=0, vmax=255)
axes2[1].set_title('B channel')
axes2[2].imshow(b_ratio, cmap='Blues', vmin=0, vmax=0.6)
axes2[2].set_title('B/(R+G+B) ratio')

# B channel value at crossing — show where holes are
hole_region = (~b_raw30) & (gray > 10)  # vessel area but B not captured
hole_vis = np.zeros((h, w, 3), dtype=np.uint8)
hole_vis[b_raw30] = [60, 90, 160]
hole_vis[hole_region] = [255, 100, 50]  # holes in orange
axes2[3].imshow(hole_vis)
axes2[3].set_title('B mask(blue) + holes(orange)')

for ax in axes2.flat: ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'rgb_B_ratio_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved rgb_B_ratio_analysis.png')

# Print summary
print("\n=== B Strategy Metrics ===")
for name, mask in strategies:
    skel = skeletonize(mask)
    eps, jns = skel_features(skel)
    print(f"{name.replace(chr(10),' '):30s} | EP={len(eps):3d}  JN={len(jns):3d}  L={int(np.sum(skel)):5d}")

r_eps, r_jns = skel_features(r_skel)
base_eps, base_jns = skel_features(base_skel)
print(f"\nR skel (thresh=30):            | EP={len(r_eps):3d}  JN={len(r_jns):3d}  L={int(np.sum(r_skel)):5d}")
print(f"Base skel (ref):               | EP={len(base_eps):3d}  JN={len(base_jns):3d}  L={int(np.sum(base_skel)):5d}")
print('\nDone!')
