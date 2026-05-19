"""
3가지 접근 동시 비교 (Sample 1 only):
1. R-B difference (margin=0, 즉 B>R / R>B) — threshold 낮게
2. B/(R+G+B) ratio 기반 필터링
3. Curvature + 서로 다른 색상이 근처에 있는 경우만
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

total = R + G + B + 1e-6
b_ratio = B / total
r_ratio = R / total
diff_BR = B - R

# ===== Skeletons =====
r_mask = R > 30
r_mask = cv2.morphologyEx(r_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
r_mask = remove_small_objects(r_mask, min_size=50)
r_skel = skeletonize(r_mask)

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

# Base skel
vessel_mask = gray > 10
vessel_mask = cv2.morphologyEx(vessel_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
vessel_mask = remove_small_objects(vessel_mask, min_size=100)
base_skel = skeletonize(vessel_mask)

def skel_features(skel):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    return np.argwhere(nb == 1), np.argwhere(nb >= 3)

# ================================================================
# Approach 1: R-B difference (margin=0)
# ================================================================
# B skel에서 B>=R인 곳만, R skel에서 R>=B인 곳만
b_filt_diff = b_skel & (diff_BR >= 0)
r_filt_diff = r_skel & (diff_BR <= 0)

# ================================================================
# Approach 2: B/(R+G+B) ratio
# ================================================================
ratio_thresholds = [0.30, 0.35, 0.40]

# ================================================================
# Approach 3: Curvature (서로 다른 색상 근처만)
# ================================================================
# contour curvature
vessel_u8 = vessel_mask.astype(np.uint8) * 255
contours_cv, _ = cv2.findContours(vessel_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
big_contours = [c for c in contours_cv if len(c) >= 50]

def compute_curvature(contour, window=12):
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 2 * window + 1:
        return np.zeros(n)
    curvature = np.zeros(n)
    for i in range(n):
        i_prev = (i - window) % n
        i_next = (i + window) % n
        t1 = pts[i] - pts[i_prev]
        t2 = pts[i_next] - pts[i]
        len1, len2 = np.linalg.norm(t1), np.linalg.norm(t2)
        if len1 < 1e-6 or len2 < 1e-6:
            continue
        t1, t2 = t1/len1, t2/len2
        cross = t1[0]*t2[1] - t1[1]*t2[0]
        dot = np.clip(t1[0]*t2[0] + t1[1]*t2[1], -1, 1)
        curvature[i] = np.abs(np.arctan2(cross, dot))
    return curvature

# "서로 다른 색상이 근처에 있는 영역" = B dominant와 R dominant가 가까운 곳
b_dom_region = diff_BR > 10
r_dom_region = diff_BR < -10
# 각각 dilate해서 겹치는 곳 = 두 색상이 인접
b_dom_dilated = cv2.dilate(b_dom_region.astype(np.uint8)*255,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
r_dom_dilated = cv2.dilate(r_dom_region.astype(np.uint8)*255,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
mixed_zone = b_dom_dilated & r_dom_dilated

# curvature map
curv_map = np.zeros((h, w), dtype=np.float32)
for c in big_contours:
    curv = compute_curvature(c, window=12)
    pts = c.reshape(-1, 2)
    for (x, y), cv_val in zip(pts, curv):
        if 0 <= y < h and 0 <= x < w:
            curv_map[y, x] = max(curv_map[y, x], cv_val)

# high curvature + mixed zone만
curv_in_mixed = (curv_map >= 0.4) & mixed_zone

# ================================================================
# Figure 1: 3 approaches 비교
# ================================================================
fig, axes = plt.subplots(3, 4, figsize=(36, 24))
fig.suptitle('3 Approaches Comparison', fontsize=18, fontweight='bold')

# --- Row 1: R-B difference (margin=0) ---
axes[0,0].set_ylabel('Approach 1:\nR-B diff (margin=0)', fontsize=14, fontweight='bold')

# B skel filtered
b_vis1 = np.zeros((h, w, 3), dtype=np.uint8)
b_bg = np.clip(B * 0.3, 0, 255).astype(np.uint8)
b_vis1[:,:,2] = b_bg
b_vis1[cv2.dilate(b_skel.astype(np.uint8), dk3) > 0] = [60, 60, 100]
b_vis1[cv2.dilate(b_filt_diff.astype(np.uint8), dk4) > 0] = [50, 220, 255]
b_kept = int(np.sum(b_filt_diff))
axes[0,0].imshow(b_vis1)
axes[0,0].set_title(f'B skel (B≥R): {b_kept}/{int(np.sum(b_skel))} px', fontsize=12)

# R skel filtered
r_vis1 = np.zeros((h, w, 3), dtype=np.uint8)
r_bg = np.clip(R * 0.3, 0, 255).astype(np.uint8)
r_vis1[:,:,0] = r_bg
r_vis1[cv2.dilate(r_skel.astype(np.uint8), dk3) > 0] = [100, 60, 60]
r_vis1[cv2.dilate(r_filt_diff.astype(np.uint8), dk4) > 0] = [255, 230, 50]
r_kept = int(np.sum(r_filt_diff))
axes[0,1].imshow(r_vis1)
axes[0,1].set_title(f'R skel (R≥B): {r_kept}/{int(np.sum(r_skel))} px', fontsize=12)

# Combined on GT
on_gt1 = gt_raw.copy().astype(float) * 0.3
on_gt1[cv2.dilate(r_filt_diff.astype(np.uint8), dk4) > 0] = [255, 80, 80]
on_gt1[cv2.dilate(b_filt_diff.astype(np.uint8), dk4) > 0] = [80, 150, 255]
axes[0,2].imshow(on_gt1.astype(np.uint8))
axes[0,2].set_title('R(red) + B(blue) on GT', fontsize=12)

# Gap
r_gap1 = r_skel & ~r_filt_diff
b_gap1 = b_skel & ~b_filt_diff
gap1 = np.zeros((h, w, 3), dtype=np.uint8)
gap1[cv2.dilate(r_filt_diff.astype(np.uint8), dk3) > 0] = [200, 60, 60]
gap1[cv2.dilate(b_filt_diff.astype(np.uint8), dk3) > 0] = [60, 120, 200]
gap1[cv2.dilate(r_gap1.astype(np.uint8), dk4) > 0] = [255, 200, 100]
gap1[cv2.dilate(b_gap1.astype(np.uint8), dk4) > 0] = [100, 255, 200]
axes[0,3].imshow(gap1)
axes[0,3].set_title(f'Gaps: R={int(np.sum(r_gap1))}px B={int(np.sum(b_gap1))}px', fontsize=12)

# --- Row 2: B/(R+G+B) ratio ---
axes[1,0].set_ylabel('Approach 2:\nB/(R+G+B) ratio', fontsize=14, fontweight='bold')

# B ratio heatmap
im_ratio = axes[1,0].imshow(b_ratio, cmap='Blues', vmin=0, vmax=0.6)
axes[1,0].set_title('B/(R+G+B) ratio heatmap', fontsize=12)
plt.colorbar(im_ratio, ax=axes[1,0], shrink=0.7)

# R ratio heatmap
im_rratio = axes[1,1].imshow(r_ratio, cmap='Reds', vmin=0, vmax=0.6)
axes[1,1].set_title('R/(R+G+B) ratio heatmap', fontsize=12)
plt.colorbar(im_rratio, ax=axes[1,1], shrink=0.7)

# ratio 기반 필터: B ratio > 0.35, R ratio > 0.35
b_filt_ratio = b_skel & (b_ratio >= 0.35)
r_filt_ratio = r_skel & (r_ratio >= 0.35)
on_gt2 = gt_raw.copy().astype(float) * 0.3
on_gt2[cv2.dilate(r_filt_ratio.astype(np.uint8), dk4) > 0] = [255, 80, 80]
on_gt2[cv2.dilate(b_filt_ratio.astype(np.uint8), dk4) > 0] = [80, 150, 255]
axes[1,2].imshow(on_gt2.astype(np.uint8))
axes[1,2].set_title(f'Ratio≥0.35: B={int(np.sum(b_filt_ratio))}px R={int(np.sum(r_filt_ratio))}px', fontsize=12)

r_gap2 = r_skel & ~r_filt_ratio
b_gap2 = b_skel & ~b_filt_ratio
gap2 = np.zeros((h, w, 3), dtype=np.uint8)
gap2[cv2.dilate(r_filt_ratio.astype(np.uint8), dk3) > 0] = [200, 60, 60]
gap2[cv2.dilate(b_filt_ratio.astype(np.uint8), dk3) > 0] = [60, 120, 200]
gap2[cv2.dilate(r_gap2.astype(np.uint8), dk4) > 0] = [255, 200, 100]
gap2[cv2.dilate(b_gap2.astype(np.uint8), dk4) > 0] = [100, 255, 200]
axes[1,3].imshow(gap2)
axes[1,3].set_title(f'Gaps: R={int(np.sum(r_gap2))}px B={int(np.sum(b_gap2))}px', fontsize=12)

# --- Row 3: Curvature (mixed zone only) ---
axes[2,0].set_ylabel('Approach 3:\nCurvature\n(mixed zone)', fontsize=14, fontweight='bold')

# Mixed zone
mix_vis = np.zeros((h, w, 3), dtype=np.uint8)
mix_vis[b_dom_region] = [40, 40, 120]
mix_vis[r_dom_region] = [120, 40, 40]
mix_vis[mixed_zone & ~b_dom_region & ~r_dom_region] = [80, 40, 80]
axes[2,0].imshow(mix_vis)
axes[2,0].set_title('B dom(blue) R dom(red)\nmixed zone(purple boundary)', fontsize=11)

# Curvature in mixed zone only
curv_mixed_vis = np.zeros((h, w), dtype=np.float32)
curv_mixed_vis[mixed_zone] = curv_map[mixed_zone]
curv_d = cv2.dilate(curv_mixed_vis, dk3)
curv_d[curv_d == 0] = np.nan
axes[2,1].imshow(curv_d, cmap='hot', vmin=0, vmax=1.0)
axes[2,1].set_title('Curvature in mixed zone only\n(bright=high)', fontsize=12)

# High curvature points in mixed zone on GT
curv_on_gt = gt_raw.copy().astype(float) * 0.3
curv_pts = np.argwhere(curv_in_mixed)
for (y, x) in curv_pts:
    cv2.circle(curv_on_gt, (x, y), 3, (255, 255, 0), -1)
# contours
for c in big_contours:
    pts = c.reshape(-1, 2)
    for (x, y) in pts:
        if 0 <= y < h and 0 <= x < w and mixed_zone[y, x]:
            curv_on_gt[y, x] = [0, 200, 0]
axes[2,2].imshow(curv_on_gt.astype(np.uint8))
axes[2,2].set_title(f'High curv in mixed zone\n{len(curv_pts)} points (yellow)', fontsize=12)

# Cluster the mixed-zone high curvature points
hc_mask = np.zeros((h, w), dtype=np.uint8)
hc_mask[curv_in_mixed] = 255
hc_dilated = cv2.dilate(hc_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
labels, n_cl = ndimage.label(hc_dilated)
cross_vis = gt_raw.copy().astype(float) * 0.4
for i in range(1, n_cl + 1):
    cl_pts = np.argwhere(labels == i)
    if len(cl_pts) < 2:
        continue
    cy, cx = cl_pts.mean(axis=0).astype(int)
    rad = int(np.max(np.linalg.norm(cl_pts - [cy, cx], axis=1))) + 5
    cv2.circle(cross_vis, (cx, cy), rad, (0, 255, 255), 2)
axes[2,3].imshow(cross_vis.astype(np.uint8))
axes[2,3].set_title(f'Crossing candidates (mixed+curv)\n{n_cl} clusters', fontsize=12)

for ax in axes.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'rgb_3approaches_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved rgb_3approaches_comparison.png')

# ================================================================
# Figure 2: Ratio approach 상세 — threshold sweep
# ================================================================
fig2, axes2 = plt.subplots(3, 4, figsize=(36, 24))
fig2.suptitle('Approach 2 Detail: B/(R+G+B) ratio threshold sweep', fontsize=16, fontweight='bold')

for row, rt in enumerate([0.30, 0.35, 0.40]):
    b_f = b_skel & (b_ratio >= rt)
    r_f = r_skel & (r_ratio >= rt)

    # B skel on ratio heatmap
    b_hm = np.stack([np.zeros_like(B), np.zeros_like(B), np.clip(b_ratio * 400, 0, 255)], axis=-1).astype(np.uint8)
    b_hm[cv2.dilate(b_skel.astype(np.uint8), dk3) > 0] = [60, 60, 100]
    b_hm[cv2.dilate(b_f.astype(np.uint8), dk4) > 0] = [50, 220, 255]
    axes2[row, 0].imshow(b_hm)
    axes2[row, 0].set_title(f'B skel on B-ratio (≥{rt})\nkept={int(np.sum(b_f))}/{int(np.sum(b_skel))}', fontsize=11)

    # R skel on ratio heatmap
    r_hm = np.stack([np.clip(r_ratio * 400, 0, 255), np.zeros_like(R), np.zeros_like(R)], axis=-1).astype(np.uint8)
    r_hm[cv2.dilate(r_skel.astype(np.uint8), dk3) > 0] = [100, 60, 60]
    r_hm[cv2.dilate(r_f.astype(np.uint8), dk4) > 0] = [255, 230, 50]
    axes2[row, 1].imshow(r_hm)
    axes2[row, 1].set_title(f'R skel on R-ratio (≥{rt})\nkept={int(np.sum(r_f))}/{int(np.sum(r_skel))}', fontsize=11)

    # Combined on GT
    on_gt = gt_raw.copy().astype(float) * 0.3
    on_gt[cv2.dilate(r_f.astype(np.uint8), dk4) > 0] = [255, 80, 80]
    on_gt[cv2.dilate(b_f.astype(np.uint8), dk4) > 0] = [80, 150, 255]
    both = (cv2.dilate(r_f.astype(np.uint8), dk4) > 0) & (cv2.dilate(b_f.astype(np.uint8), dk4) > 0)
    on_gt[both] = [200, 50, 200]
    axes2[row, 2].imshow(on_gt.astype(np.uint8))
    axes2[row, 2].set_title(f'Combined on GT (ratio≥{rt})', fontsize=12)

    # Gap
    rg = r_skel & ~r_f
    bg = b_skel & ~b_f
    gv = np.zeros((h, w, 3), dtype=np.uint8)
    gv[cv2.dilate(r_f.astype(np.uint8), dk3) > 0] = [200, 60, 60]
    gv[cv2.dilate(b_f.astype(np.uint8), dk3) > 0] = [60, 120, 200]
    gv[cv2.dilate(rg.astype(np.uint8), dk4) > 0] = [255, 200, 100]
    gv[cv2.dilate(bg.astype(np.uint8), dk4) > 0] = [100, 255, 200]
    axes2[row, 3].imshow(gv)
    axes2[row, 3].set_title(f'Gaps: R={int(np.sum(rg))} B={int(np.sum(bg))}', fontsize=12)

for ax in axes2.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'rgb_ratio_sweep.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved rgb_ratio_sweep.png')

# Print metrics
print('\n=== Approach 1: R-B diff (margin=0) ===')
b_ep, b_jn = skel_features(b_filt_diff)
r_ep, r_jn = skel_features(r_filt_diff)
print(f'B filt: {int(np.sum(b_filt_diff))}px EP={len(b_ep)} JN={len(b_jn)}')
print(f'R filt: {int(np.sum(r_filt_diff))}px EP={len(r_ep)} JN={len(r_jn)}')

print('\n=== Approach 2: Ratio ===')
for rt in [0.30, 0.35, 0.40]:
    bf = b_skel & (b_ratio >= rt)
    rf = r_skel & (r_ratio >= rt)
    be, bj = skel_features(bf)
    re, rj = skel_features(rf)
    print(f'ratio≥{rt}: B={int(np.sum(bf))}px(EP={len(be)} JN={len(bj)})  R={int(np.sum(rf))}px(EP={len(re)} JN={len(rj)})')

print('\n=== Reference ===')
base_ep, base_jn = skel_features(base_skel)
print(f'Base skel: {int(np.sum(base_skel))}px EP={len(base_ep)} JN={len(base_jn)}')
print(f'B skel full: {int(np.sum(b_skel))}px')
print(f'R skel full: {int(np.sum(r_skel))}px')

print('\nDone!')
