"""
Final quantification v3:
- JN: neighbor>=3 pixel들을 clustering해서 인접한 것은 하나의 junction으로
- EP: 동일하게 clustering
- top 80% baseline
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
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
diff_BR = B - R

boundary_y = int(h * 0.8)
top_mask = np.zeros((h, w), dtype=bool)
top_mask[:boundary_y, :] = True

# ===== Skeletons =====
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

vessel_all = remove_small_objects(
    cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0,
    min_size=100)
base_skel = skeletonize(vessel_all)
vessel_mask_loose = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

# Dominance filter
r_filtered = r_skel_full & (diff_BR < 0)
b_filtered = b_skel_full & (diff_BR > 0)

# ===== Bridging =====
def get_endpoints_with_tangent(skel, trace_len=20):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    endpoints = np.argwhere(nb == 1)
    results = []
    for ep_pos in endpoints:
        y0, x0 = ep_pos
        path = [(y0, x0)]
        visited = {(y0, x0)}
        cy, cx = y0, x0
        for _ in range(trace_len):
            found = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0: continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and (ny, nx) not in visited:
                        path.append((ny, nx))
                        visited.add((ny, nx))
                        cy, cx = ny, nx
                        found = True
                        break
                if found: break
            if not found: break
        if len(path) >= 5:
            n_avg = min(len(path), 10)
            tangent = np.array(path[0], dtype=float) - np.array(path[n_avg-1], dtype=float)
            norm = np.linalg.norm(tangent)
            if norm > 0: tangent /= norm
            results.append({'pos': (y0, x0), 'tangent': tangent, 'path': path})
    return results

def _bridge_pts(ep1, ep2):
    p1 = np.array(ep1['pos'], dtype=float)
    p2 = np.array(ep2['pos'], dtype=float)
    t1, t2 = ep1['tangent'], ep2['tangent']
    dist = np.linalg.norm(p2 - p1)
    ctrl1 = p1 + t1 * dist * 0.4
    ctrl2 = p2 + t2 * dist * 0.4
    n_pts = max(int(dist * 1.5), 10)
    pts = []
    for t in np.linspace(0, 1, n_pts):
        p = (1-t)**3*p1 + 3*(1-t)**2*t*ctrl1 + 3*(1-t)*t**2*ctrl2 + t**3*p2
        pts.append((int(round(p[0])), int(round(p[1]))))
    return pts

def do_bridging(skel, vessel_mask, max_dist=80, max_angle_deg=60):
    eps = get_endpoints_with_tangent(skel, trace_len=20)
    candidates = []
    for i in range(len(eps)):
        for j in range(i+1, len(eps)):
            p1 = np.array(eps[i]['pos'], dtype=float)
            p2 = np.array(eps[j]['pos'], dtype=float)
            dist = np.linalg.norm(p2 - p1)
            if dist < 5 or dist > max_dist: continue
            dir_12 = (p2 - p1) / dist
            cos1 = np.dot(eps[i]['tangent'], dir_12)
            cos2 = np.dot(eps[j]['tangent'], -dir_12)
            thresh = np.cos(np.radians(max_angle_deg))
            if cos1 < thresh or cos2 < thresh: continue
            pts = _bridge_pts(eps[i], eps[j])
            on_v = sum(1 for (y, x) in pts if 0 <= y < h and 0 <= x < w and vessel_mask[y, x])
            vr = on_v / max(len(pts), 1)
            if vr < 0.7: continue
            score = (cos1 + cos2) / 2 - dist / max_dist * 0.2 + vr * 0.2
            candidates.append((i, j, score, dist, vr))
    candidates.sort(key=lambda x: -x[2])
    used = set()
    matches = []
    for i, j, sc, d, vr in candidates:
        if i not in used and j not in used:
            matches.append((i, j, sc, d, vr))
            used.add(i); used.add(j)
    bridged = skel.copy()
    for i, j, *_ in matches:
        pts = _bridge_pts(eps[i], eps[j])
        for (y, x) in pts:
            if 0 <= y < h and 0 <= x < w:
                bridged[y, x] = True
    return bridged, len(matches)

r_bridged_raw, r_nb = do_bridging(r_filtered, vessel_mask_loose)
b_bridged_raw, b_nb = do_bridging(b_filtered, vessel_mask_loose)

# Re-skeletonize after bridging to ensure 1-pixel width
r_bridged = skeletonize(r_bridged_raw)
b_bridged = skeletonize(b_bridged_raw)

# ===== 핵심: Clustered EP/JN 계산 =====
def get_clustered_ep_jn(skel, region_mask):
    """
    EP/JN pixel을 추출한 뒤 인접한 것들을 하나의 point로 clustering.
    각 cluster의 centroid를 하나의 EP/JN으로 카운트.
    """
    s = skel & region_mask
    su = s.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su

    length = int(np.sum(s))

    # EP pixels
    ep_mask = (nb == 1)
    ep_labels, n_ep = ndimage.label(ep_mask)
    ep_centroids = []
    for i in range(1, n_ep + 1):
        pts = np.argwhere(ep_labels == i)
        cy, cx = pts.mean(axis=0)
        ep_centroids.append((int(round(cy)), int(round(cx))))

    # JN pixels — neighbor >= 3 → clustering → verify by removing and counting fragments
    jn_mask = (nb >= 3)
    # dilate to merge adjacent JN pixels into one cluster
    jn_dilated = cv2.dilate(jn_mask.astype(np.uint8), k3)
    jn_labels, n_jn_raw = ndimage.label(jn_dilated)
    jn_centroids = []
    for i in range(1, n_jn_raw + 1):
        cluster_mask = (jn_labels == i) & jn_mask
        pts = np.argwhere(cluster_mask)
        if len(pts) == 0:
            pts = np.argwhere(jn_labels == i)
        cy, cx = pts.mean(axis=0)
        jy, jx = int(round(cy)), int(round(cx))

        # Verify: remove junction cluster from skeleton, count remaining
        # fragments in a local window
        cluster_all = (jn_labels == i)
        # local window around cluster
        ys, xs = np.where(cluster_all)
        pad = 3
        ly0 = max(0, ys.min() - pad)
        ly1 = min(s.shape[0], ys.max() + pad + 1)
        lx0 = max(0, xs.min() - pad)
        lx1 = min(s.shape[1], xs.max() + pad + 1)

        local_skel = s[ly0:ly1, lx0:lx1].copy()
        local_cluster = cluster_all[ly0:ly1, lx0:lx1]
        # Remove junction pixels
        local_skel[local_cluster] = False
        # Count remaining connected components = number of branches
        _, n_branches = ndimage.label(local_skel)
        if n_branches >= 3:
            jn_centroids.append((jy, jx))

    # EP 중 baseline(boundary_y) 근처에 있는 것은 제외 — 잘린 경계에서 생긴 EP
    boundary_margin = 5
    ep_centroids = [(y, x) for (y, x) in ep_centroids
                    if y < boundary_y - boundary_margin]

    return ep_centroids, jn_centroids, length

# Single GT
single_ep, single_jn, single_L = get_clustered_ep_jn(base_skel, top_mask)
# R bridged
r_ep, r_jn, r_L = get_clustered_ep_jn(r_bridged, top_mask)
# B bridged
b_ep, b_jn, b_L = get_clustered_ep_jn(b_bridged, top_mask)

# ===== EP 분류: real vs connected =====
r_skel_dilated = cv2.dilate(r_bridged.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
b_skel_dilated = cv2.dilate(b_bridged.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0

def classify_ep(ep_list, other_skel_dilated):
    real, conn = [], []
    for (y, x) in ep_list:
        if other_skel_dilated[y, x]:
            conn.append((y, x))
        else:
            real.append((y, x))
    return real, conn

r_real_ep, r_conn_ep = classify_ep(r_ep, b_skel_dilated)
b_real_ep, b_conn_ep = classify_ep(b_ep, r_skel_dilated)

# ===== Print =====
print("=" * 65)
print(f"  QUANTIFICATION (top 80%, y < {boundary_y}) - Clustered EP/JN")
print("=" * 65)
print(f"\n{'':28s} {'L':>6s} {'EP':>5s} {'JN':>5s}  EP detail")
print("-" * 70)
print(f"{'Single GT (base)':28s} {single_L:6d} {len(single_ep):5d} {len(single_jn):5d}")
print(f"{'R (R>B, bridged)':28s} {r_L:6d} {len(r_ep):5d} {len(r_jn):5d}  "
      f"real={len(r_real_ep)} + conn={len(r_conn_ep)}")
print(f"{'B (B>R, bridged)':28s} {b_L:6d} {len(b_ep):5d} {len(b_jn):5d}  "
      f"real={len(b_real_ep)} + conn={len(b_conn_ep)}")
print(f"{'R + B sum':28s} {r_L+b_L:6d} {len(r_ep)+len(b_ep):5d} {len(r_jn)+len(b_jn):5d}  "
      f"real={len(r_real_ep)+len(b_real_ep)} + conn={len(r_conn_ep)+len(b_conn_ep)}")

# ===== Figure =====
fig, axes = plt.subplots(2, 3, figsize=(30, 18))
fig.suptitle(f'Skeleton Quantification — Clustered EP/JN (top 80%, y < {boundary_y})',
             fontsize=18, fontweight='bold')

# Single GT
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(base_skel.astype(np.uint8), dk4) > 0] = [0, 200, 0]
for (y, x) in single_ep:
    cv2.circle(vis, (x, y), 7, (255, 255, 0), 2)
for (y, x) in single_jn:
    cv2.circle(vis, (x, y), 7, (255, 0, 255), 2)
vis[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[0,0].imshow(vis.astype(np.uint8))
axes[0,0].set_title(f'Single GT (base)\nL={single_L}  EP={len(single_ep)}  JN={len(single_jn)}',
                    fontsize=13)

# R bridged
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0] = [255, 80, 80]
for (y, x) in r_real_ep:
    cv2.circle(vis, (x, y), 7, (255, 255, 0), 2)
for (y, x) in r_conn_ep:
    cv2.circle(vis, (x, y), 7, (0, 255, 255), 2)
for (y, x) in r_jn:
    cv2.circle(vis, (x, y), 6, (255, 0, 255), 2)
vis[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[0,1].imshow(vis.astype(np.uint8))
axes[0,1].set_title(f'R skel (R>B, bridged +{r_nb})\nL={r_L}  EP={len(r_ep)}(real={len(r_real_ep)} conn={len(r_conn_ep)})  JN={len(r_jn)}',
                    fontsize=10)

# B bridged
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0] = [80, 150, 255]
for (y, x) in b_real_ep:
    cv2.circle(vis, (x, y), 7, (255, 255, 0), 2)
for (y, x) in b_conn_ep:
    cv2.circle(vis, (x, y), 7, (0, 255, 255), 2)
for (y, x) in b_jn:
    cv2.circle(vis, (x, y), 6, (255, 0, 255), 2)
vis[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[0,2].imshow(vis.astype(np.uint8))
axes[0,2].set_title(f'B skel (B>R, bridged +{b_nb})\nL={b_L}  EP={len(b_ep)}(real={len(b_real_ep)} conn={len(b_conn_ep)})  JN={len(b_jn)}',
                    fontsize=10)

# R+B combined
vis = np.zeros((h, w, 3), dtype=np.uint8)
vis[cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0] = [220, 60, 60]
vis[cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0] = [60, 130, 220]
for (y, x) in r_real_ep + b_real_ep:
    cv2.circle(vis, (x, y), 7, (255, 255, 0), 2)
for (y, x) in r_conn_ep + b_conn_ep:
    cv2.circle(vis, (x, y), 7, (0, 255, 255), 2)
for (y, x) in r_jn + b_jn:
    cv2.circle(vis, (x, y), 5, (255, 0, 255), 2)
vis[boundary_y-1:boundary_y+1, :] = [128, 128, 0]
axes[1,0].imshow(vis)
axes[1,0].set_title(f'R+B combined\nyellow=real EP  cyan=connected EP  magenta=JN', fontsize=11)

# Bar chart
labels = ['Single\nGT', 'R', 'B', 'R+B\nsum', 'R+B\n(real EP)']
l_vals = [single_L, r_L, b_L, r_L+b_L, r_L+b_L]
ep_vals = [len(single_ep), len(r_ep), len(b_ep),
           len(r_ep)+len(b_ep), len(r_real_ep)+len(b_real_ep)]
jn_vals = [len(single_jn), len(r_jn), len(b_jn),
           len(r_jn)+len(b_jn), len(r_jn)+len(b_jn)]
colors_b = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']

x_pos = np.arange(len(labels))
w_bar = 0.25
axes[1,1].bar(x_pos - w_bar, [v/100 for v in l_vals], w_bar, label='L/100', color=colors_b, alpha=0.8)
axes[1,1].bar(x_pos, ep_vals, w_bar, label='EP', color=colors_b, alpha=0.5,
              edgecolor='yellow', linewidth=2)
axes[1,1].bar(x_pos + w_bar, jn_vals, w_bar, label='JN', color=colors_b, alpha=0.3,
              edgecolor='magenta', linewidth=2)
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels(labels, fontsize=10)
axes[1,1].legend(fontsize=10)
axes[1,1].set_title('Metrics (clustered EP/JN)', fontsize=13)
for i, (l, e, j) in enumerate(zip(l_vals, ep_vals, jn_vals)):
    axes[1,1].text(i - w_bar, l/100 + 0.3, str(l), ha='center', fontsize=8, fontweight='bold', color='white')
    axes[1,1].text(i, e + 0.3, str(e), ha='center', fontsize=8, fontweight='bold', color='white')
    axes[1,1].text(i + w_bar, j + 0.3, str(j), ha='center', fontsize=8, fontweight='bold', color='white')

# Summary
axes[1,2].axis('off')
txt = (f"{'='*55}\n"
       f"  QUANTIFICATION (top 80%, y < {boundary_y})\n"
       f"  Clustered EP/JN (adjacent pixels → 1 point)\n"
       f"{'='*55}\n\n"
       f"{'':26s} {'L':>6s} {'EP':>4s} {'JN':>4s}\n"
       f"{'-'*50}\n"
       f"{'Single GT (base)':26s} {single_L:6d} {len(single_ep):4d} {len(single_jn):4d}\n\n"
       f"{'R (R>B, bridged)':26s} {r_L:6d} {len(r_ep):4d} {len(r_jn):4d}\n"
       f"  EP: real={len(r_real_ep)} + conn={len(r_conn_ep)}\n\n"
       f"{'B (B>R, bridged)':26s} {b_L:6d} {len(b_ep):4d} {len(b_jn):4d}\n"
       f"  EP: real={len(b_real_ep)} + conn={len(b_conn_ep)}\n\n"
       f"{'-'*50}\n"
       f"{'R + B sum':26s} {r_L+b_L:6d} {len(r_ep)+len(b_ep):4d} {len(r_jn)+len(b_jn):4d}\n"
       f"{'R + B real EP only':26s} {'':6s} {len(r_real_ep)+len(b_real_ep):4d}\n\n"
       f"  yellow circle = real EP\n"
       f"  cyan circle = connected EP\n"
       f"    (합치면 연결, 카운팅 제외)\n"
       f"  magenta circle = Junction")
axes[1,2].text(0.02, 0.95, txt, fontsize=11, transform=axes[1,2].transAxes,
               fontfamily='monospace', color='white', va='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.9))

for ax in axes.flat:
    ax.set_facecolor('black')
plt.tight_layout()
plt.savefig(OUT / 'skel_final_quantify_v3.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('\nSaved skel_final_quantify_v3.png')
print('Done!')
