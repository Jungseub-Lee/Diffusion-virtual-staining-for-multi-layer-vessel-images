"""
R/(R+B) ratio로 dominance 판정 — threshold sweep
B skel에서 B가 진짜 dominant한 곳만 남기기
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

# R/(R+B) ratio — R과 B만 비교
rb_sum = R + B + 1e-6
r_rb_ratio = R / rb_sum  # 0.5 = 동일, >0.5 = R dominant, <0.5 = B dominant
b_rb_ratio = B / rb_sum

# ===== Skeletons (full) =====
r_mask = R > 30
r_mask = cv2.morphologyEx(r_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
r_mask = remove_small_objects(r_mask, min_size=50)
r_skel_full = skeletonize(r_mask)

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
b_skel_full = skeletonize(b_hyst)

vessel_mask = gray > 10
vessel_mask = cv2.morphologyEx(vessel_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
vessel_mask = remove_small_objects(vessel_mask, min_size=100)
base_skel = skeletonize(vessel_mask)

def skel_features(skel):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    return np.argwhere(nb == 1), np.argwhere(nb >= 3)

# ===== Figure 0: R/(R+B) ratio heatmap 먼저 확인 =====
fig0, axes0 = plt.subplots(1, 4, figsize=(36, 8))
fig0.suptitle('R/(R+B) ratio analysis', fontsize=16, fontweight='bold')

im0 = axes0[0].imshow(r_rb_ratio, cmap='coolwarm', vmin=0.3, vmax=0.7)
axes0[0].set_title('R/(R+B) ratio\n>0.5=R dom, <0.5=B dom', fontsize=12)
plt.colorbar(im0, ax=axes0[0], shrink=0.7)

axes0[1].imshow(gt_blur)
axes0[1].set_title('GT (blurred)', fontsize=12)

# Histogram of R/(R+B) on vessel area only
r_ratio_vessel = r_rb_ratio[vessel_mask]
axes0[2].hist(r_ratio_vessel, bins=80, color='purple', edgecolor='black', alpha=0.8)
axes0[2].axvline(0.5, color='black', linewidth=2, label='0.5 (equal)')
axes0[2].axvline(0.55, color='red', linewidth=1.5, linestyle='--', label='0.55')
axes0[2].axvline(0.45, color='blue', linewidth=1.5, linestyle='--', label='0.45')
axes0[2].axvline(0.40, color='cyan', linewidth=1.5, linestyle='--', label='0.40')
axes0[2].set_title('R/(R+B) distribution\n(vessel pixels only)', fontsize=12)
axes0[2].set_xlabel('R/(R+B)')
axes0[2].legend(fontsize=9)

# On skeleton pixels
r_ratio_on_rskel = r_rb_ratio[r_skel_full]
r_ratio_on_bskel = r_rb_ratio[b_skel_full]
axes0[3].hist(r_ratio_on_rskel, bins=50, color='red', alpha=0.5, label=f'R skel (n={len(r_ratio_on_rskel)})')
axes0[3].hist(r_ratio_on_bskel, bins=50, color='blue', alpha=0.5, label=f'B skel (n={len(r_ratio_on_bskel)})')
axes0[3].axvline(0.5, color='black', linewidth=2)
axes0[3].set_title('R/(R+B) on skeleton pixels', fontsize=12)
axes0[3].set_xlabel('R/(R+B)')
axes0[3].legend()

for ax in axes0:
    if not ax.has_data():
        ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'skel_rb_ratio_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved skel_rb_ratio_analysis.png')

# ===== Figure 1: Ratio threshold sweep =====
# R skel: keep where R/(R+B) >= thresh (R dominant)
# B skel: keep where B/(R+B) >= thresh, 즉 R/(R+B) <= 1-thresh
r_thresholds = [0.50, 0.55, 0.60, 0.65]
b_thresholds = [0.50, 0.45, 0.40, 0.35]  # B/(R+B) >= 이 값

fig1, axes1 = plt.subplots(len(r_thresholds), 5, figsize=(40, 7*len(r_thresholds)))
fig1.suptitle('R/(R+B) ratio threshold sweep for skeleton filtering', fontsize=16, fontweight='bold')

for row, (rt, bt) in enumerate(zip(r_thresholds, b_thresholds)):
    # R filtered: R/(R+B) >= rt
    r_filt = r_skel_full & (r_rb_ratio >= rt)
    # 작은 fragment 제거
    rl, nr = ndimage.label(r_filt)
    for i in range(1, nr+1):
        if np.sum(rl == i) < 8:
            r_filt[rl == i] = False

    # B filtered: B/(R+B) >= bt, 즉 R/(R+B) <= 1-bt
    b_filt = b_skel_full & (b_rb_ratio >= bt)
    bl, nb = ndimage.label(b_filt)
    for i in range(1, nb+1):
        if np.sum(bl == i) < 8:
            b_filt[bl == i] = False

    r_ep, r_jn = skel_features(r_filt)
    b_ep, b_jn = skel_features(b_filt)

    # R filtered on GT
    vis = gt_raw.copy().astype(float) * 0.25
    vis[cv2.dilate(r_filt.astype(np.uint8), dk4) > 0] = [255, 80, 80]
    axes1[row, 0].imshow(vis.astype(np.uint8))
    axes1[row, 0].set_title(f'R skel: R/(R+B)≥{rt}\nL={int(np.sum(r_filt))} EP={len(r_ep)} JN={len(r_jn)}', fontsize=11)

    # B filtered on GT
    vis = gt_raw.copy().astype(float) * 0.25
    vis[cv2.dilate(b_filt.astype(np.uint8), dk4) > 0] = [80, 150, 255]
    axes1[row, 1].imshow(vis.astype(np.uint8))
    axes1[row, 1].set_title(f'B skel: B/(R+B)≥{bt}\nL={int(np.sum(b_filt))} EP={len(b_ep)} JN={len(b_jn)}', fontsize=11)

    # Combined
    vis = gt_raw.copy().astype(float) * 0.25
    vis[cv2.dilate(r_filt.astype(np.uint8), dk4) > 0] = [255, 80, 80]
    vis[cv2.dilate(b_filt.astype(np.uint8), dk4) > 0] = [80, 150, 255]
    both = (cv2.dilate(r_filt.astype(np.uint8), dk4) > 0) & (cv2.dilate(b_filt.astype(np.uint8), dk4) > 0)
    vis[both] = [200, 50, 200]
    axes1[row, 2].imshow(vis.astype(np.uint8))
    axes1[row, 2].set_title(f'R(red)+B(blue) combined\nR≥{rt} / B≥{bt}', fontsize=11)

    # Skeleton only view
    skel_vis = np.zeros((h, w, 3), dtype=np.uint8)
    skel_vis[cv2.dilate(r_filt.astype(np.uint8), dk3) > 0] = [220, 60, 60]
    skel_vis[cv2.dilate(b_filt.astype(np.uint8), dk3) > 0] = [60, 130, 220]
    axes1[row, 3].imshow(skel_vis)
    axes1[row, 3].set_title('Skeleton only', fontsize=11)

    # Gap analysis
    r_gap = r_skel_full & ~r_filt
    b_gap = b_skel_full & ~b_filt
    gap_vis = np.zeros((h, w, 3), dtype=np.uint8)
    gap_vis[cv2.dilate(r_filt.astype(np.uint8), dk3) > 0] = [200, 60, 60]
    gap_vis[cv2.dilate(b_filt.astype(np.uint8), dk3) > 0] = [60, 120, 200]
    gap_vis[cv2.dilate(r_gap.astype(np.uint8), dk3) > 0] = [255, 200, 100]
    gap_vis[cv2.dilate(b_gap.astype(np.uint8), dk3) > 0] = [100, 255, 200]
    axes1[row, 4].imshow(gap_vis)
    axes1[row, 4].set_title(f'Gaps: R(yellow)={int(np.sum(r_gap))} B(cyan)={int(np.sum(b_gap))}', fontsize=11)

for ax in axes1.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'skel_rb_ratio_sweep.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved skel_rb_ratio_sweep.png')

# ===== Print summary =====
base_ep, base_jn = skel_features(base_skel)
print(f"\nBase skel: L={int(np.sum(base_skel))} EP={len(base_ep)} JN={len(base_jn)}")
print(f"R full: L={int(np.sum(r_skel_full))}")
print(f"B full: L={int(np.sum(b_skel_full))}")
print()
for rt, bt in zip(r_thresholds, b_thresholds):
    rf = r_skel_full & (r_rb_ratio >= rt)
    bf = b_skel_full & (b_rb_ratio >= bt)
    rl, nr = ndimage.label(rf)
    for i in range(1, nr+1):
        if np.sum(rl == i) < 8: rf[rl == i] = False
    bl, nb = ndimage.label(bf)
    for i in range(1, nb+1):
        if np.sum(bl == i) < 8: bf[bl == i] = False
    re, rj = skel_features(rf)
    be, bj = skel_features(bf)
    print(f"R≥{rt}/B≥{bt}: R(L={int(np.sum(rf)):4d} EP={len(re):2d} JN={len(rj):3d}) "
          f"B(L={int(np.sum(bf)):4d} EP={len(be):2d} JN={len(bj):3d})")

print('\nDone!')
