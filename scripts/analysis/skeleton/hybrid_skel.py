"""
Hybrid skeleton:
- 비-crossing 구간: base skel + R/B ratio coloring
- Crossing 구간: R/B separate skel (각각 crossing 관통)
- 하단 20%: base skel (green, single layer)

Crossing 검출: curvature(mixed zone) + R/B dominance transition
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
dk5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

R = gt_blur[:,:,0].astype(np.float32)
G = gt_blur[:,:,1].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
total = R + G + B + 1e-6
b_ratio = B / total
r_ratio = R / total
diff_BR = B - R

boundary_y = int(h * 0.8)

# ===== All skeletons =====
vessel_mask = gray > 10
vessel_mask = cv2.morphologyEx(vessel_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
vessel_mask = remove_small_objects(vessel_mask, min_size=100)
base_skel = skeletonize(vessel_mask)

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

# ===== Step 1: Crossing zone 검출 =====
# Method: R dominant와 B dominant 영역이 겹치는 곳 = crossing zone
b_dom = diff_BR > 5
r_dom = diff_BR < -5
# dilate해서 인접 영역 찾기
b_dom_d = cv2.dilate(b_dom.astype(np.uint8)*255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))) > 0
r_dom_d = cv2.dilate(r_dom.astype(np.uint8)*255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))) > 0
crossing_zone_raw = b_dom_d & r_dom_d & vessel_mask

# crossing zone을 약간 dilate해서 주변 포함
crossing_zone = cv2.dilate(crossing_zone_raw.astype(np.uint8)*255,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0

# 하단 20%에서는 crossing zone 제거
crossing_zone[boundary_y:, :] = False

non_crossing_zone = ~crossing_zone

print(f"Crossing zone: {np.sum(crossing_zone)} pixels ({100*np.sum(crossing_zone)/(h*w):.1f}% of image)")

# ===== Step 2: Hybrid skeleton 구성 =====
# 하단 20%: base skel (green)
hybrid_base = np.zeros((h, w), dtype=bool)  # base skel (하단 + 비-crossing 상단)
hybrid_r = np.zeros((h, w), dtype=bool)     # R skel (crossing 구간)
hybrid_b = np.zeros((h, w), dtype=bool)     # B skel (crossing 구간)
hybrid_r_colored = np.zeros((h, w), dtype=bool)  # base skel R-colored (비-crossing 상단)
hybrid_b_colored = np.zeros((h, w), dtype=bool)  # base skel B-colored (비-crossing 상단)

# 하단: base skel
hybrid_base[boundary_y:, :] = base_skel[boundary_y:, :]

# 상단 비-crossing: base skel + ratio coloring
upper_non_cross = base_skel & non_crossing_zone
upper_non_cross[boundary_y:, :] = False
hybrid_r_colored = upper_non_cross & (r_ratio > b_ratio)
hybrid_b_colored = upper_non_cross & (b_ratio >= r_ratio)

# 상단 crossing: R/B separate skel
hybrid_r = r_skel & crossing_zone
hybrid_r[boundary_y:, :] = False
hybrid_b = b_skel & crossing_zone
hybrid_b[boundary_y:, :] = False

# ===== Step 3: 연결 분석 =====
# crossing zone 경계에서 base skel과 R/B skel이 만나는지
# crossing zone edge
crossing_edge = cv2.dilate(crossing_zone.astype(np.uint8), k3) - crossing_zone.astype(np.uint8)
crossing_edge = crossing_edge > 0

# base skel이 crossing edge에 닿는 점
base_at_edge = base_skel & crossing_edge
# R/B skel이 crossing zone 안에서 edge에 닿는 점
r_near_edge = cv2.dilate(hybrid_r.astype(np.uint8), k5) > 0
b_near_edge = cv2.dilate(hybrid_b.astype(np.uint8), k5) > 0
r_connects = base_at_edge & r_near_edge & (r_ratio > b_ratio)
b_connects = base_at_edge & b_near_edge & (b_ratio >= r_ratio)

print(f"Base skel at crossing edge: {np.sum(base_at_edge)} px")
print(f"R connection points: {np.sum(r_connects)} px")
print(f"B connection points: {np.sum(b_connects)} px")

# ===== Figure 1: Hybrid skeleton 전체 =====
fig, axes = plt.subplots(2, 4, figsize=(36, 16))
fig.suptitle('Hybrid Skeleton: base(non-crossing) + R/B(crossing)', fontsize=16, fontweight='bold')

# Crossing zone on GT
cz_vis = gt_raw.copy().astype(float) * 0.5
cz_vis[crossing_zone] = cz_vis[crossing_zone] * 0.5 + np.array([100, 0, 100]) * 0.5
cz_vis[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[0,0].imshow(cz_vis.astype(np.uint8))
axes[0,0].set_title('Crossing zones (purple) on GT', fontsize=12)

# Hybrid skeleton
hyb_vis = gt_raw.copy().astype(float) * 0.25
# 하단 base (green)
hyb_vis[cv2.dilate(hybrid_base.astype(np.uint8), dk4) > 0] = [0, 220, 0]
# 상단 non-crossing R colored
hyb_vis[cv2.dilate(hybrid_r_colored.astype(np.uint8), dk4) > 0] = [255, 80, 80]
# 상단 non-crossing B colored
hyb_vis[cv2.dilate(hybrid_b_colored.astype(np.uint8), dk4) > 0] = [80, 150, 255]
# Crossing R skel
hyb_vis[cv2.dilate(hybrid_r.astype(np.uint8), dk5) > 0] = [255, 120, 50]
# Crossing B skel
hyb_vis[cv2.dilate(hybrid_b.astype(np.uint8), dk5) > 0] = [50, 180, 255]
hyb_vis[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[0,1].imshow(hyb_vis.astype(np.uint8))
axes[0,1].set_title('Hybrid skeleton\ngreen=base, red/blue=colored, bright=crossing R/B', fontsize=11)

# Base skel reference
base_ref = gt_raw.copy().astype(float) * 0.25
base_ref[cv2.dilate(base_skel.astype(np.uint8), dk4) > 0] = [0, 255, 0]
base_ref[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[0,2].imshow(base_ref.astype(np.uint8))
axes[0,2].set_title('Base skeleton (reference)', fontsize=12)

# R/B separate skel reference (상단 전체)
rb_ref = gt_raw.copy().astype(float) * 0.25
rb_ref[cv2.dilate(r_skel.astype(np.uint8), dk4) > 0] = [255, 80, 80]
rb_ref[cv2.dilate(b_skel.astype(np.uint8), dk4) > 0] = [80, 150, 255]
rb_ref[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[0,3].imshow(rb_ref.astype(np.uint8))
axes[0,3].set_title('R/B separate skel (reference)', fontsize=12)

# --- Row 2: Zoom into crossing regions ---
# crossing zone의 connected components → 각각 zoom
crossing_labels, n_cross = ndimage.label(crossing_zone)
# 큰 crossing만
cross_sizes = [(i, np.sum(crossing_labels == i)) for i in range(1, n_cross + 1)]
cross_sizes.sort(key=lambda x: -x[1])
top_crosses = cross_sizes[:4]

for col, (ci, csize) in enumerate(top_crosses):
    pts = np.argwhere(crossing_labels == ci)
    cy, cx = pts.mean(axis=0).astype(int)
    crop = 55
    y0, y1 = max(0, cy-crop), min(h, cy+crop)
    x0, x1 = max(0, cx-crop), min(w, cx+crop)

    zoom = gt_raw[y0:y1, x0:x1].copy().astype(float) * 0.25

    # crossing zone 영역 표시 (반투명)
    cz_crop = crossing_zone[y0:y1, x0:x1]
    zoom[cz_crop] = zoom[cz_crop] * 0.5 + np.array([60, 0, 60]) * 0.5

    # base skel (non-crossing) — colored
    rc = hybrid_r_colored[y0:y1, x0:x1]
    bc = hybrid_b_colored[y0:y1, x0:x1]
    zoom[cv2.dilate(rc.astype(np.uint8), dk4) > 0] = [255, 80, 80]
    zoom[cv2.dilate(bc.astype(np.uint8), dk4) > 0] = [80, 150, 255]

    # R/B crossing skel — 밝게
    rr = hybrid_r[y0:y1, x0:x1]
    bb = hybrid_b[y0:y1, x0:x1]
    zoom[cv2.dilate(rr.astype(np.uint8), dk5) > 0] = [255, 150, 50]
    zoom[cv2.dilate(bb.astype(np.uint8), dk5) > 0] = [50, 200, 255]

    # connection points
    rc_pts = r_connects[y0:y1, x0:x1]
    bc_pts = b_connects[y0:y1, x0:x1]
    for (ry, rx) in np.argwhere(rc_pts):
        cv2.circle(zoom, (rx, ry), 4, (255, 255, 0), -1)
    for (by, bx) in np.argwhere(bc_pts):
        cv2.circle(zoom, (bx, by), 4, (0, 255, 255), -1)

    axes[1, col].imshow(zoom.astype(np.uint8))
    axes[1, col].set_title(f'Crossing #{col+1} (size={csize}px)\n'
                           f'purple=zone, bright=R/B skel\n'
                           f'yellow/cyan dots=connection pts', fontsize=10)

for ax in axes.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'hybrid_skeleton.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved hybrid_skeleton.png')

# ===== Figure 2: 3개 skeleton 나란히 비교 (skeleton only, no GT) =====
fig2, axes2 = plt.subplots(1, 3, figsize=(30, 9))
fig2.suptitle('Skeleton Comparison (skeleton only)', fontsize=16, fontweight='bold')

# Base skel
s1_vis = np.zeros((h, w, 3), dtype=np.uint8)
s1_vis[cv2.dilate(base_skel.astype(np.uint8), dk4) > 0] = [0, 200, 0]
axes2[0].imshow(s1_vis)
axes2[0].set_title('Base skeleton (single layer)', fontsize=13)

# R/B separate
s2_vis = np.zeros((h, w, 3), dtype=np.uint8)
s2_vis[cv2.dilate(r_skel.astype(np.uint8), dk4) > 0] = [220, 60, 60]
s2_vis[cv2.dilate(b_skel.astype(np.uint8), dk4) > 0] = [60, 130, 220]
both = (cv2.dilate(r_skel.astype(np.uint8), dk4) > 0) & (cv2.dilate(b_skel.astype(np.uint8), dk4) > 0)
s2_vis[both] = [180, 50, 180]
axes2[1].imshow(s2_vis)
axes2[1].set_title('R/B separate skeletons', fontsize=13)

# Hybrid
s3_vis = np.zeros((h, w, 3), dtype=np.uint8)
s3_vis[cv2.dilate(hybrid_base.astype(np.uint8), dk4) > 0] = [0, 200, 0]
s3_vis[cv2.dilate(hybrid_r_colored.astype(np.uint8), dk4) > 0] = [220, 60, 60]
s3_vis[cv2.dilate(hybrid_b_colored.astype(np.uint8), dk4) > 0] = [60, 130, 220]
s3_vis[cv2.dilate(hybrid_r.astype(np.uint8), dk5) > 0] = [255, 140, 40]
s3_vis[cv2.dilate(hybrid_b.astype(np.uint8), dk5) > 0] = [40, 190, 255]
s3_vis[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes2[2].imshow(s3_vis)
axes2[2].set_title('Hybrid: base(green) + colored(R/B)\n+ crossing R/B(bright)', fontsize=12)

for ax in axes2:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'hybrid_skeleton_compare.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved hybrid_skeleton_compare.png')

# Metrics
def skel_features(skel):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    return np.argwhere(nb == 1), np.argwhere(nb >= 3)

print("\n=== Metrics (top 80% only) ===")
top_mask = np.ones((h, w), dtype=bool)
top_mask[boundary_y:, :] = False

for name, skel in [('Base skel', base_skel), ('R skel', r_skel), ('B skel', b_skel)]:
    s = skel & top_mask
    ep, jn = skel_features(s)
    print(f"{name:15s}: L={int(np.sum(s)):5d} EP={len(ep):3d} JN={len(jn):3d}")

# Hybrid totals
hyb_all = hybrid_r_colored | hybrid_b_colored | hybrid_r | hybrid_b
ep_h, jn_h = skel_features(hyb_all)
print(f"{'Hybrid (top)':15s}: L={int(np.sum(hyb_all)):5d} EP={len(ep_h):3d} JN={len(jn_h):3d}")

print(f"\nCrossing zones: {n_cross} regions")
for i, (ci, cs) in enumerate(top_crosses):
    print(f"  #{i+1}: {cs} px")

print('\nDone!')
