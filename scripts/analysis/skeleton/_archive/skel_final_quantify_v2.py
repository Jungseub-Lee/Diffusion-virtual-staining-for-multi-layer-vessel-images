"""
Final quantification v2:
- R, B 개별 정량 (EP, JN 시각화)
- EP 분류: 진짜 EP vs 겹쳐서 연결되는 EP (vessel 내부에 있는 EP)
- 전부 top 80% (y < boundary_y) 기준
- Single GT도 동일 baseline 적용
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
    for i, j, sc, d, vr in matches:
        pts = _bridge_pts(eps[i], eps[j])
        for (y, x) in pts:
            if 0 <= y < h and 0 <= x < w:
                bridged[y, x] = True
    return bridged, len(matches)

# Bridge
r_bridged, r_n_bridges = do_bridging(r_filtered, vessel_mask_loose)
b_bridged, b_n_bridges = do_bridging(b_filtered, vessel_mask_loose)

# ===== EP/JN 추출 (top 80% only) =====
def get_ep_jn(skel, region_mask):
    s = skel & region_mask
    su = s.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    ep = np.argwhere(nb == 1)
    jn = np.argwhere(nb >= 3)
    length = int(np.sum(s))
    return ep, jn, length

# Single GT
single_ep, single_jn, single_L = get_ep_jn(base_skel, top_mask)
# R bridged
r_ep, r_jn, r_L = get_ep_jn(r_bridged, top_mask)
# B bridged
b_ep, b_jn, b_L = get_ep_jn(b_bridged, top_mask)

# ===== EP 분류: 진짜 EP vs 연결 EP =====
# "연결 EP" = 상대 skel 또는 vessel 내부에 있어서 합치면 연결되는 EP
# 판단: EP가 상대 skel의 dilated 영역 안에 있으면 → 연결 EP
# 또는 EP가 vessel mask 내부에 있고 상대 skel과 가까우면 → 연결 EP

r_skel_dilated = cv2.dilate(r_bridged.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
b_skel_dilated = cv2.dilate(b_bridged.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0

def classify_endpoints(ep_list, other_skel_dilated, vessel_mask):
    """
    EP를 분류:
    - real_ep: 진짜 endpoint (혈관 끝)
    - connected_ep: 상대 skel 근처에 있어서 합치면 연결되는 EP
    """
    real_eps = []
    connected_eps = []
    for (y, x) in ep_list:
        if other_skel_dilated[y, x]:
            connected_eps.append((y, x))
        elif vessel_mask[y, x]:
            # vessel 내부에 있지만 상대 skel 근처는 아닌 경우
            # → crossing 구간에서 끊긴 EP (bridge 대상이었지만 연결 안 된 것)
            # 여전히 vessel 내부이므로 connected로 분류
            # 좀 더 엄격하게: vessel 내부 + 근처에 상대 skel이 있으면
            # dilate를 더 크게 해서 확인
            other_big = cv2.dilate(other_skel_dilated.astype(np.uint8),
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
            if other_big[y, x]:
                connected_eps.append((y, x))
            else:
                real_eps.append((y, x))
        else:
            real_eps.append((y, x))
    return real_eps, connected_eps

r_real_ep, r_conn_ep = classify_endpoints(r_ep, b_skel_dilated, vessel_all)
b_real_ep, b_conn_ep = classify_endpoints(b_ep, r_skel_dilated, vessel_all)

print("=" * 60)
print(f"  QUANTIFICATION (top 80%, y < {boundary_y})")
print("=" * 60)
print(f"\n{'':28s} {'L':>6s} {'EP':>5s} {'JN':>5s}  EP detail")
print("-" * 70)
print(f"{'Single GT (base skel)':28s} {single_L:6d} {len(single_ep):5d} {len(single_jn):5d}")
print(f"{'R filtered+bridged':28s} {r_L:6d} {len(r_ep):5d} {len(r_jn):5d}  "
      f"real={len(r_real_ep)} + connected={len(r_conn_ep)}")
print(f"{'B filtered+bridged':28s} {b_L:6d} {len(b_ep):5d} {len(b_jn):5d}  "
      f"real={len(b_real_ep)} + connected={len(b_conn_ep)}")
print(f"{'R + B sum':28s} {r_L+b_L:6d} {len(r_ep)+len(b_ep):5d} {len(r_jn)+len(b_jn):5d}  "
      f"real={len(r_real_ep)+len(b_real_ep)} + connected={len(r_conn_ep)+len(b_conn_ep)}")
print(f"\n  → R+B 실효 EP (real only): {len(r_real_ep)+len(b_real_ep)}")
print(f"  → Connected EP (합치면 연결): {len(r_conn_ep)+len(b_conn_ep)}")

# ===== Figure 1: EP/JN 시각화 =====
fig, axes = plt.subplots(2, 3, figsize=(30, 18))
fig.suptitle(f'Skeleton EP/JN Visualization (top 80%, y < {boundary_y})', fontsize=18, fontweight='bold')

# Single GT
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(base_skel.astype(np.uint8), dk4) > 0] = [0, 200, 0]
for (y, x) in single_ep:
    cv2.circle(vis, (x, y), 6, (255, 255, 0), -1)  # EP yellow
for (y, x) in single_jn:
    cv2.circle(vis, (x, y), 5, (255, 0, 255), -1)   # JN magenta
vis[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[0,0].imshow(vis.astype(np.uint8))
axes[0,0].set_title(f'Single GT (base skel)\nL={single_L}  EP={len(single_ep)}(yellow)  JN={len(single_jn)}(magenta)',
                    fontsize=12)

# R bridged with EP/JN
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0] = [255, 80, 80]
for (y, x) in r_real_ep:
    cv2.circle(vis, (x, y), 6, (255, 255, 0), -1)  # real EP yellow
for (y, x) in r_conn_ep:
    cv2.circle(vis, (x, y), 6, (0, 255, 255), -1)   # connected EP cyan
for (y, x) in r_jn:
    cv2.circle(vis, (x, y), 4, (255, 0, 255), -1)
vis[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[0,1].imshow(vis.astype(np.uint8))
axes[0,1].set_title(f'R skel (R>B, bridged)\nL={r_L}  EP={len(r_ep)}(real={len(r_real_ep)}+conn={len(r_conn_ep)})  JN={len(r_jn)}',
                    fontsize=10)

# B bridged with EP/JN
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0] = [80, 150, 255]
for (y, x) in b_real_ep:
    cv2.circle(vis, (x, y), 6, (255, 255, 0), -1)
for (y, x) in b_conn_ep:
    cv2.circle(vis, (x, y), 6, (0, 255, 255), -1)
for (y, x) in b_jn:
    cv2.circle(vis, (x, y), 4, (255, 0, 255), -1)
vis[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[0,2].imshow(vis.astype(np.uint8))
axes[0,2].set_title(f'B skel (B>R, bridged)\nL={b_L}  EP={len(b_ep)}(real={len(b_real_ep)}+conn={len(b_conn_ep)})  JN={len(b_jn)}',
                    fontsize=10)

# R+B combined (skeleton only)
vis = np.zeros((h, w, 3), dtype=np.uint8)
vis[cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0] = [220, 60, 60]
vis[cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0] = [60, 130, 220]
# Real EP (yellow), Connected EP (cyan)
for (y, x) in r_real_ep + b_real_ep:
    cv2.circle(vis, (x, y), 6, (255, 255, 0), -1)
for (y, x) in r_conn_ep + b_conn_ep:
    cv2.circle(vis, (x, y), 6, (0, 255, 255), -1)
vis[boundary_y-1:boundary_y+1, :] = [128, 128, 0]
axes[1,0].imshow(vis)
axes[1,0].set_title(f'R+B combined\nyellow=real EP({len(r_real_ep)+len(b_real_ep)})  '
                    f'cyan=connected EP({len(r_conn_ep)+len(b_conn_ep)})', fontsize=11)

# Bar chart comparison
labels = ['Single\nGT', 'R\nbridged', 'B\nbridged', 'R+B\nsum', 'R+B\n(real EP)']
l_vals = [single_L, r_L, b_L, r_L+b_L, r_L+b_L]
ep_vals = [len(single_ep), len(r_ep), len(b_ep),
           len(r_ep)+len(b_ep), len(r_real_ep)+len(b_real_ep)]
jn_vals = [len(single_jn), len(r_jn), len(b_jn),
           len(r_jn)+len(b_jn), len(r_jn)+len(b_jn)]
colors_b = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']

x = np.arange(len(labels))
width = 0.25

ax_bar = axes[1,1]
bars1 = ax_bar.bar(x - width, [v/100 for v in l_vals], width, label='Length/100', color=colors_b, alpha=0.7)
bars2 = ax_bar.bar(x, ep_vals, width, label='EP', color=colors_b, alpha=0.5, edgecolor='yellow', linewidth=2)
bars3 = ax_bar.bar(x + width, jn_vals, width, label='JN', color=colors_b, alpha=0.3, edgecolor='magenta', linewidth=2)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, fontsize=9)
ax_bar.legend(fontsize=10)
ax_bar.set_title('Metrics comparison (top 80%)', fontsize=13)

# Text values on bars
for i, (l, e, j) in enumerate(zip(l_vals, ep_vals, jn_vals)):
    ax_bar.text(i - width, l/100 + 0.5, str(l), ha='center', fontsize=8, color='white')
    ax_bar.text(i, e + 0.5, str(e), ha='center', fontsize=8, color='white')
    ax_bar.text(i + width, j + 0.5, str(j), ha='center', fontsize=8, color='white')

# Summary table
axes[1,2].axis('off')
txt = (f"{'='*55}\n"
       f"  QUANTIFICATION (top 80%, y < {boundary_y})\n"
       f"{'='*55}\n\n"
       f"{'':26s} {'L':>6s} {'EP':>5s} {'JN':>5s}\n"
       f"{'-'*55}\n"
       f"{'Single GT (base)':26s} {single_L:6d} {len(single_ep):5d} {len(single_jn):5d}\n\n"
       f"{'R (R>B, bridged)':26s} {r_L:6d} {len(r_ep):5d} {len(r_jn):5d}\n"
       f"  real EP={len(r_real_ep)}, connected EP={len(r_conn_ep)}\n\n"
       f"{'B (B>R, bridged)':26s} {b_L:6d} {len(b_ep):5d} {len(b_jn):5d}\n"
       f"  real EP={len(b_real_ep)}, connected EP={len(b_conn_ep)}\n\n"
       f"{'-'*55}\n"
       f"{'R + B sum':26s} {r_L+b_L:6d} {len(r_ep)+len(b_ep):5d} {len(r_jn)+len(b_jn):5d}\n"
       f"{'R + B (real EP only)':26s} {'':6s} {len(r_real_ep)+len(b_real_ep):5d}\n\n"
       f"  * connected EP = 상대 skel 근처\n"
       f"    (합치면 연결, 카운팅 제외)\n"
       f"  * yellow = real EP\n"
       f"  * cyan = connected EP")
axes[1,2].text(0.02, 0.95, txt, fontsize=11, transform=axes[1,2].transAxes,
               fontfamily='monospace', color='white', va='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.9))

for ax in axes.flat:
    ax.axis('off') if not ax.has_data() else None
    ax.set_facecolor('black')

plt.tight_layout()
plt.savefig(OUT / 'skel_final_quantify_v2.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('\nSaved skel_final_quantify_v2.png')

# ===== Figure 2: EP classification zoom =====
# connected EP들 zoom
all_conn = [(y, x, 'R') for (y, x) in r_conn_ep] + [(y, x, 'B') for (y, x) in b_conn_ep]
n_show = min(8, len(all_conn))

if n_show > 0:
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig2, axes2 = plt.subplots(rows, cols, figsize=(28, 7*rows))
    if rows == 1: axes2 = axes2[np.newaxis, :]
    fig2.suptitle('Connected EP detail (cyan = EP that connects when R+B merged)', fontsize=14, fontweight='bold')

    for idx in range(n_show):
        ax = axes2.flat[idx]
        cy, cx, cn = all_conn[idx]
        crop = 40
        y0 = max(0, cy-crop); y1 = min(h, cy+crop)
        x0 = max(0, cx-crop); x1 = min(w, cx+crop)

        z = gt_raw[y0:y1, x0:x1].copy().astype(float) * 0.3
        z[cv2.dilate(r_bridged[y0:y1, x0:x1].astype(np.uint8), dk4) > 0] = [255, 80, 80]
        z[cv2.dilate(b_bridged[y0:y1, x0:x1].astype(np.uint8), dk4) > 0] = [80, 150, 255]
        cv2.circle(z, (cx-x0, cy-y0), 6, (0, 255, 255), 2)
        ax.imshow(z.astype(np.uint8))
        ax.set_title(f'{cn} conn EP ({cy},{cx})', fontsize=10)
        ax.axis('off')

    for idx in range(n_show, rows*cols):
        axes2.flat[idx].axis('off')
    plt.tight_layout()
    plt.savefig(OUT / 'skel_connected_ep_detail.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved skel_connected_ep_detail.png')

print('Done!')
