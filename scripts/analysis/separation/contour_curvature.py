"""
Vessel contour curvature analysis:
- Binary mask → contour 추출
- Contour 따라 local curvature 계산
- Curvature spike = crossing 후보
- R/B dominance와 결합
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
from scipy.signal import savgol_filter
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
diff_BR = B - R

# ===== Binary vessel mask =====
vessel_mask = gray > 10
vessel_mask = cv2.morphologyEx(vessel_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
vessel_mask = remove_small_objects(vessel_mask, min_size=100)
vessel_u8 = vessel_mask.astype(np.uint8) * 255

# ===== Contour 추출 =====
contours, hierarchy = cv2.findContours(vessel_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 큰 contour만 (작은건 노이즈)
min_contour_len = 50
big_contours = [c for c in contours if len(c) >= min_contour_len]
print(f"Total contours: {len(contours)}, big contours (>={min_contour_len}): {len(big_contours)}")

# ===== Curvature 계산 함수 =====
def compute_curvature(contour, window=15):
    """
    Contour 따라 local curvature 계산
    window: tangent vector 계산 시 앞뒤로 몇 pixel 볼지
    Returns: curvature array (same length as contour)
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 2 * window + 1:
        return np.zeros(n)

    curvature = np.zeros(n)
    for i in range(n):
        # 앞/뒤 window만큼의 tangent vector
        i_prev = (i - window) % n
        i_next = (i + window) % n

        # tangent vectors
        t1 = pts[i] - pts[i_prev]
        t2 = pts[i_next] - pts[i]

        # normalize
        len1 = np.linalg.norm(t1)
        len2 = np.linalg.norm(t2)
        if len1 < 1e-6 or len2 < 1e-6:
            continue
        t1 = t1 / len1
        t2 = t2 / len2

        # angle between tangents (curvature)
        cross = t1[0] * t2[1] - t1[1] * t2[0]
        dot = np.clip(t1[0]*t2[0] + t1[1]*t2[1], -1, 1)
        angle = np.abs(np.arctan2(cross, dot))
        curvature[i] = angle

    return curvature

# ===== 모든 contour에 대해 curvature 계산 =====
all_curvatures = []
all_points = []
for c in big_contours:
    curv = compute_curvature(c, window=12)
    all_curvatures.append(curv)
    all_points.append(c.reshape(-1, 2))

# ===== Figure 1: Contour + curvature heatmap =====
fig1, axes1 = plt.subplots(2, 3, figsize=(30, 18))
fig1.suptitle('Vessel Contour Curvature Analysis', fontsize=16, fontweight='bold')

# Panel 1: All contours on GT
contour_vis = gt_raw.copy().astype(float) * 0.4
cv2.drawContours(contour_vis.astype(np.uint8), big_contours, -1, (0, 255, 0), 1)
contour_img = contour_vis.astype(np.uint8).copy()
cv2.drawContours(contour_img, big_contours, -1, (0, 255, 0), 1)
axes1[0,0].imshow(contour_img)
axes1[0,0].set_title(f'Vessel contours ({len(big_contours)} contours)', fontsize=13)

# Panel 2: Curvature as color along contour
curv_vis = np.zeros((h, w), dtype=np.float32)
for pts, curv in zip(all_points, all_curvatures):
    for (x, y), c in zip(pts, curv):
        if 0 <= y < h and 0 <= x < w:
            curv_vis[y, x] = max(curv_vis[y, x], c)

# dilate for visibility
curv_vis_d = cv2.dilate(curv_vis, dk3)
curv_vis_d[curv_vis_d == 0] = np.nan

im1 = axes1[0,1].imshow(curv_vis_d, cmap='hot', vmin=0, vmax=1.0)
axes1[0,1].set_title('Curvature along contours\n(bright=high curvature)', fontsize=13)
plt.colorbar(im1, ax=axes1[0,1], shrink=0.7, label='curvature (rad)')

# Panel 3: High curvature points on GT
# threshold for "spike"
curv_thresholds = [0.3, 0.5, 0.7]
high_curv_vis = gt_raw.copy().astype(float) * 0.3
colors = [(255, 255, 0), (255, 150, 0), (255, 0, 0)]  # yellow, orange, red
for thresh, color in zip(curv_thresholds, colors):
    for pts, curv in zip(all_points, all_curvatures):
        high_idx = curv >= thresh
        high_pts = pts[high_idx]
        for (x, y) in high_pts:
            if 0 <= y < h and 0 <= x < w:
                cv2.circle(high_curv_vis, (x, y), 3, color, -1)

axes1[0,2].imshow(high_curv_vis.astype(np.uint8))
axes1[0,2].set_title('High curvature points\nyellow≥0.3  orange≥0.5  red≥0.7', fontsize=12)

# Panel 4: B-R dominance map reference
im4 = axes1[1,0].imshow(diff_BR, cmap='coolwarm', vmin=-100, vmax=100)
axes1[1,0].set_title('B-R difference\n(blue=R dom, red=B dom)', fontsize=13)
plt.colorbar(im4, ax=axes1[1,0], shrink=0.7)

# Panel 5: High curvature + B-R dominance overlay
combo_vis = np.zeros((h, w, 3), dtype=np.uint8)
# B dominant region
combo_vis[diff_BR > 0] = [30, 30, 80]
# R dominant region
combo_vis[diff_BR < 0] = [80, 30, 30]
# Contours
for pts in all_points:
    for (x, y) in pts:
        if 0 <= y < h and 0 <= x < w:
            combo_vis[y, x] = [100, 100, 100]

# High curvature points colored by dominance
for pts, curv in zip(all_points, all_curvatures):
    high_idx = curv >= 0.4
    high_pts = pts[high_idx]
    for (x, y) in high_pts:
        if 0 <= y < h and 0 <= x < w:
            if diff_BR[y, x] > 0:
                cv2.circle(combo_vis, (x, y), 4, (100, 180, 255), -1)  # B dominant → blue dot
            else:
                cv2.circle(combo_vis, (x, y), 4, (255, 100, 100), -1)  # R dominant → red dot

axes1[1,1].imshow(combo_vis)
axes1[1,1].set_title('High curvature (≥0.4) + dominance\nblue=B dom, red=R dom', fontsize=12)

# Panel 6: Curvature spike clustering → crossing candidates
# High curvature points가 밀집된 곳 = crossing
high_curv_mask = np.zeros((h, w), dtype=np.uint8)
for pts, curv in zip(all_points, all_curvatures):
    high_idx = curv >= 0.4
    for (x, y) in pts[high_idx]:
        if 0 <= y < h and 0 <= x < w:
            high_curv_mask[y, x] = 255

# dilate + find clusters
high_curv_dilated = cv2.dilate(high_curv_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
cluster_labels, n_clusters = ndimage.label(high_curv_dilated)

crossing_vis = gt_raw.copy().astype(float) * 0.4
# Draw each cluster with bounding circle
for i in range(1, n_clusters + 1):
    cluster_pts = np.argwhere(cluster_labels == i)  # (y, x)
    if len(cluster_pts) < 3:
        continue
    cy, cx = cluster_pts.mean(axis=0).astype(int)
    radius = int(np.max(np.linalg.norm(cluster_pts - [cy, cx], axis=1))) + 5
    cv2.circle(crossing_vis, (cx, cy), radius, (0, 255, 255), 2)
    cv2.putText(crossing_vis, f'C{i}', (cx-10, cy-radius-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# Also draw contours
cv2.drawContours(crossing_vis.astype(np.uint8), big_contours, -1, (0, 200, 0), 1)
crossing_img = crossing_vis.astype(np.uint8)
axes1[1,2].imshow(crossing_img)
axes1[1,2].set_title(f'Crossing candidates: {n_clusters} clusters\n(high curvature clusters)', fontsize=12)

for ax in axes1.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'contour_curvature_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved contour_curvature_analysis.png')

# ===== Figure 2: Zoom into crossing regions =====
# 주요 crossing 영역 몇 개를 zoom
# cluster centroid 기준으로 crop
fig2_rows = min(n_clusters, 6)
if fig2_rows > 0:
    fig2, axes2 = plt.subplots(fig2_rows, 4, figsize=(32, 7*fig2_rows))
    if fig2_rows == 1:
        axes2 = axes2[np.newaxis, :]
    fig2.suptitle('Zoom into crossing candidates', fontsize=16, fontweight='bold')

    for row in range(fig2_rows):
        cluster_pts = np.argwhere(cluster_labels == row + 1)
        if len(cluster_pts) < 3:
            continue
        cy, cx = cluster_pts.mean(axis=0).astype(int)
        crop = 60
        y0, y1 = max(0, cy-crop), min(h, cy+crop)
        x0, x1 = max(0, cx-crop), min(w, cx+crop)

        # GT crop
        axes2[row, 0].imshow(gt_raw[y0:y1, x0:x1])
        axes2[row, 0].set_title(f'C{row+1}: GT crop', fontsize=11)

        # Curvature crop
        curv_crop = curv_vis_d[y0:y1, x0:x1].copy()
        axes2[row, 1].imshow(curv_crop, cmap='hot', vmin=0, vmax=1.0)
        axes2[row, 1].set_title(f'Curvature', fontsize=11)

        # Contour + high curvature points
        contour_crop = gt_raw[y0:y1, x0:x1].copy().astype(float) * 0.4
        # draw contours in crop
        for pts, curv in zip(all_points, all_curvatures):
            for (x, y), c in zip(pts, curv):
                yy, xx = y - y0, x - x0
                if 0 <= yy < (y1-y0) and 0 <= xx < (x1-x0):
                    if c >= 0.4:
                        cv2.circle(contour_crop, (xx, yy), 3, (255, 0, 0), -1)
                    else:
                        contour_crop[yy, xx] = [0, 200, 0]
        axes2[row, 2].imshow(contour_crop.astype(np.uint8))
        axes2[row, 2].set_title(f'Contour: green=normal, red=high curv', fontsize=11)

        # B-R dominance crop + high curvature
        br_crop = diff_BR[y0:y1, x0:x1]
        dom_vis = np.zeros((y1-y0, x1-x0, 3), dtype=np.uint8)
        dom_vis[br_crop > 0] = [50, 80, 180]
        dom_vis[br_crop < 0] = [180, 50, 50]
        dom_vis[~vessel_mask[y0:y1, x0:x1]] = [0, 0, 0]
        for pts, curv in zip(all_points, all_curvatures):
            for (x, y), c in zip(pts, curv):
                yy, xx = y - y0, x - x0
                if 0 <= yy < (y1-y0) and 0 <= xx < (x1-x0) and c >= 0.4:
                    cv2.circle(dom_vis, (xx, yy), 3, (255, 255, 0), -1)
        axes2[row, 3].imshow(dom_vis)
        axes2[row, 3].set_title(f'B-R dominance + high curv(yellow)', fontsize=11)

    for ax in axes2.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT / 'contour_curvature_zoom.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved contour_curvature_zoom.png ({fig2_rows} crossings)')
else:
    print('No crossing clusters found')

print('Done!')
