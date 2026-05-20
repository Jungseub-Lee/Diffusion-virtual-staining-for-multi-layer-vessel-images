"""
Gap bridging v2:
- R skel: R>30 mask (기존)
- B skel: hysteresis(seed=50, low=20) → B dominant (B>R) 영역만 남기기
- R skel도: R dominant (R>B) 영역만 남기기
- bridge: vessel mask 위만 유효
- 파라미터 완화: angle 60°, dist 80, trace 20
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
diff_BR = B - R

# Vessel mask (전체)
vessel_mask = gray > 10
vessel_mask = cv2.morphologyEx(vessel_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
vessel_mask = remove_small_objects(vessel_mask, min_size=100)
# 약간 dilate해서 bridge가 약간 벗어나도 허용
vessel_mask_loose = cv2.dilate(vessel_mask.astype(np.uint8)*255, k5) > 0

# ===== R skeleton: R channel mask → skel → R>B 필터 =====
r_mask_raw = R > 30
r_mask_raw = cv2.morphologyEx(r_mask_raw.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
r_mask_raw = remove_small_objects(r_mask_raw, min_size=50)
r_skel_full = skeletonize(r_mask_raw)
# R dominant 영역만 (R > B)
r_skel = r_skel_full & (diff_BR <= 0)
# 너무 작은 fragment 제거
r_skel_labeled, n_r = ndimage.label(r_skel)
for i in range(1, n_r + 1):
    if np.sum(r_skel_labeled == i) < 10:
        r_skel[r_skel_labeled == i] = False

# ===== B skeleton: hysteresis → skel → B>R 필터 =====
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
# B dominant 영역만 (B > R)
b_skel = b_skel_full & (diff_BR >= 0)
# 작은 fragment 제거
b_skel_labeled, n_b = ndimage.label(b_skel)
for i in range(1, n_b + 1):
    if np.sum(b_skel_labeled == i) < 10:
        b_skel[b_skel_labeled == i] = False

# Base skel (참고)
base_skel = skeletonize(vessel_mask)

def skel_features(skel):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    return np.argwhere(nb == 1), np.argwhere(nb >= 3)

print("=== Before bridging ===")
r_ep, r_jn = skel_features(r_skel)
b_ep, b_jn = skel_features(b_skel)
rf_ep, rf_jn = skel_features(r_skel_full)
bf_ep, bf_jn = skel_features(b_skel_full)
print(f"R full skel: L={int(np.sum(r_skel_full))} EP={len(rf_ep)} JN={len(rf_jn)}")
print(f"R filtered (R>B): L={int(np.sum(r_skel))} EP={len(r_ep)} JN={len(r_jn)}")
print(f"B full skel: L={int(np.sum(b_skel_full))} EP={len(bf_ep)} JN={len(bf_jn)}")
print(f"B filtered (B>R): L={int(np.sum(b_skel))} EP={len(b_ep)} JN={len(b_jn)}")

# ===== Gap bridging =====
def get_endpoints_with_tangent(skel, trace_len=20):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    endpoints = np.argwhere(nb == 1)

    results = []
    for ep in endpoints:
        y0, x0 = ep
        path = [(y0, x0)]
        visited = {(y0, x0)}
        cy, cx = y0, x0

        for _ in range(trace_len):
            found = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and (ny, nx) not in visited:
                        path.append((ny, nx))
                        visited.add((ny, nx))
                        cy, cx = ny, nx
                        found = True
                        break
                if found:
                    break
            if not found:
                break

        if len(path) >= 5:
            # tangent: 마지막 몇 개 점으로 평균 방향 (더 안정적)
            p_end = np.array(path[0], dtype=float)
            # 여러 점의 평균 방향
            n_avg = min(len(path), 10)
            p_inner = np.array(path[n_avg-1], dtype=float)
            tangent = p_end - p_inner
            norm = np.linalg.norm(tangent)
            if norm > 0:
                tangent = tangent / norm
            results.append({
                'pos': (y0, x0),
                'tangent': tangent,
                'path': path
            })

    return results

def match_endpoints(eps_list, vessel_mask, max_dist=80, max_angle_deg=60):
    candidates = []
    n = len(eps_list)

    for i in range(n):
        for j in range(i+1, n):
            p1 = np.array(eps_list[i]['pos'], dtype=float)
            p2 = np.array(eps_list[j]['pos'], dtype=float)
            t1 = eps_list[i]['tangent']
            t2 = eps_list[j]['tangent']

            dist = np.linalg.norm(p2 - p1)
            if dist < 5 or dist > max_dist:
                continue

            dir_12 = (p2 - p1) / dist
            dir_21 = -dir_12

            cos1 = np.dot(t1, dir_12)
            cos2 = np.dot(t2, dir_21)

            angle_thresh = np.cos(np.radians(max_angle_deg))
            if cos1 < angle_thresh or cos2 < angle_thresh:
                continue

            # Bridge path가 vessel mask 위를 지나가는지 확인
            bridge_pts = _bridge_points(eps_list[i], eps_list[j])
            on_vessel = sum(1 for (y, x) in bridge_pts
                          if 0 <= y < h and 0 <= x < w and vessel_mask[y, x])
            vessel_ratio = on_vessel / max(len(bridge_pts), 1)
            if vessel_ratio < 0.7:  # 70% 이상 vessel 위에 있어야 함
                continue

            score = (cos1 + cos2) / 2 - dist / max_dist * 0.2 + vessel_ratio * 0.2
            candidates.append((i, j, score, dist, vessel_ratio))

    candidates.sort(key=lambda x: -x[2])
    used = set()
    matches = []
    for i, j, score, dist, vr in candidates:
        if i not in used and j not in used:
            matches.append((i, j, score, dist, vr))
            used.add(i)
            used.add(j)
    return matches

def _bridge_points(ep1, ep2):
    p1 = np.array(ep1['pos'], dtype=float)
    p2 = np.array(ep2['pos'], dtype=float)
    t1 = ep1['tangent']
    t2 = ep2['tangent']
    dist = np.linalg.norm(p2 - p1)

    ctrl1 = p1 + t1 * dist * 0.4
    ctrl2 = p2 + t2 * dist * 0.4

    n_pts = max(int(dist * 1.5), 10)
    pts = []
    for t in np.linspace(0, 1, n_pts):
        p = (1-t)**3 * p1 + 3*(1-t)**2*t * ctrl1 + 3*(1-t)*t**2 * ctrl2 + t**3 * p2
        y, x = int(round(p[0])), int(round(p[1]))
        pts.append((y, x))
    return pts

# R bridging
print("\n=== R gap bridging ===")
r_eps = get_endpoints_with_tangent(r_skel, trace_len=20)
print(f"R endpoints: {len(r_eps)}")
r_matches = match_endpoints(r_eps, vessel_mask_loose, max_dist=80, max_angle_deg=60)
print(f"R matched pairs: {len(r_matches)}")

r_bridged = r_skel.copy()
r_bridges = []
for i, j, score, dist, vr in r_matches:
    pts = _bridge_points(r_eps[i], r_eps[j])
    r_bridges.append((pts, score, dist, vr))
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w:
            r_bridged[y, x] = True
    print(f"  R bridge: dist={dist:.0f} score={score:.2f} vessel%={vr:.0%}")

# B bridging
print("\n=== B gap bridging ===")
b_eps = get_endpoints_with_tangent(b_skel, trace_len=20)
print(f"B endpoints: {len(b_eps)}")
b_matches = match_endpoints(b_eps, vessel_mask_loose, max_dist=80, max_angle_deg=60)
print(f"B matched pairs: {len(b_matches)}")

b_bridged = b_skel.copy()
b_bridges = []
for i, j, score, dist, vr in b_matches:
    pts = _bridge_points(b_eps[i], b_eps[j])
    b_bridges.append((pts, score, dist, vr))
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w:
            b_bridged[y, x] = True
    print(f"  B bridge: dist={dist:.0f} score={score:.2f} vessel%={vr:.0%}")

# Metrics after
r_ep_a, r_jn_a = skel_features(r_bridged)
b_ep_a, b_jn_a = skel_features(b_bridged)
base_ep, base_jn = skel_features(base_skel)

print(f"\n=== After bridging ===")
print(f"R skel: EP {len(r_ep)}→{len(r_ep_a)}, JN {len(r_jn)}→{len(r_jn_a)}, L {int(np.sum(r_skel))}→{int(np.sum(r_bridged))}")
print(f"B skel: EP {len(b_ep)}→{len(b_ep_a)}, JN {len(b_jn)}→{len(b_jn_a)}, L {int(np.sum(b_skel))}→{int(np.sum(b_bridged))}")
print(f"Base:   EP={len(base_ep)} JN={len(base_jn)} L={int(np.sum(base_skel))}")

# ===== Figure 1: Full skel vs Filtered skel 비교 =====
fig0, axes0 = plt.subplots(2, 3, figsize=(30, 18))
fig0.suptitle('Skeleton: Full vs Dominance-filtered (R>B / B>R)', fontsize=16, fontweight='bold')

# R full
r_full_vis = gt_raw.copy().astype(float) * 0.25
r_full_vis[cv2.dilate(r_skel_full.astype(np.uint8), dk4) > 0] = [255, 80, 80]
axes0[0,0].imshow(r_full_vis.astype(np.uint8))
axes0[0,0].set_title(f'R skel FULL (R>30)\nEP={len(rf_ep)} JN={len(rf_jn)} L={int(np.sum(r_skel_full))}', fontsize=12)

# R filtered
r_filt_vis = gt_raw.copy().astype(float) * 0.25
r_filt_vis[cv2.dilate(r_skel.astype(np.uint8), dk4) > 0] = [255, 80, 80]
axes0[0,1].imshow(r_filt_vis.astype(np.uint8))
axes0[0,1].set_title(f'R skel filtered (R>B)\nEP={len(r_ep)} JN={len(r_jn)} L={int(np.sum(r_skel))}', fontsize=12)

# R-B diff heatmap
im_diff = axes0[0,2].imshow(diff_BR, cmap='coolwarm', vmin=-80, vmax=80)
axes0[0,2].set_title('B-R difference\nred=B dom, blue=R dom', fontsize=12)
plt.colorbar(im_diff, ax=axes0[0,2], shrink=0.7)

# B full
b_full_vis = gt_raw.copy().astype(float) * 0.25
b_full_vis[cv2.dilate(b_skel_full.astype(np.uint8), dk4) > 0] = [80, 150, 255]
axes0[1,0].imshow(b_full_vis.astype(np.uint8))
axes0[1,0].set_title(f'B skel FULL (hyst 50/20)\nEP={len(bf_ep)} JN={len(bf_jn)} L={int(np.sum(b_skel_full))}', fontsize=12)

# B filtered
b_filt_vis = gt_raw.copy().astype(float) * 0.25
b_filt_vis[cv2.dilate(b_skel.astype(np.uint8), dk4) > 0] = [80, 150, 255]
axes0[1,1].imshow(b_filt_vis.astype(np.uint8))
axes0[1,1].set_title(f'B skel filtered (B>R)\nEP={len(b_ep)} JN={len(b_jn)} L={int(np.sum(b_skel))}', fontsize=12)

# GT
axes0[1,2].imshow(gt_blur)
axes0[1,2].set_title('GT (blurred)', fontsize=12)

for ax in axes0.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'skel_dominance_filter.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved skel_dominance_filter.png')

# ===== Figure 2: Before/After bridging =====
fig, axes = plt.subplots(2, 4, figsize=(36, 16))
fig.suptitle('Gap Bridging v2: Dominance-filtered + vessel mask constraint', fontsize=16, fontweight='bold')

# R before
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_skel.astype(np.uint8), dk4) > 0] = [255, 80, 80]
for ep in r_eps:
    y, x = ep['pos']
    cv2.circle(vis, (x, y), 4, (255, 255, 0), -1)
    ty, tx = ep['tangent']
    cv2.arrowedLine(vis, (x, y), (int(x+tx*20), int(y+ty*20)), (255, 255, 0), 1, tipLength=0.3)
axes[0,0].imshow(vis.astype(np.uint8))
axes[0,0].set_title(f'R filtered BEFORE\nEP={len(r_ep)}', fontsize=11)

# R after
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0] = [255, 80, 80]
for pts, sc, d, vr in r_bridges:
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w:
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
axes[0,1].imshow(vis.astype(np.uint8))
axes[0,1].set_title(f'R AFTER bridging\nEP={len(r_ep_a)} ({len(r_matches)} bridges)', fontsize=11)

# B before
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(b_skel.astype(np.uint8), dk4) > 0] = [80, 150, 255]
for ep in b_eps:
    y, x = ep['pos']
    cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
    ty, tx = ep['tangent']
    cv2.arrowedLine(vis, (x, y), (int(x+tx*20), int(y+ty*20)), (0, 255, 255), 1, tipLength=0.3)
axes[0,2].imshow(vis.astype(np.uint8))
axes[0,2].set_title(f'B filtered BEFORE\nEP={len(b_ep)}', fontsize=11)

# B after
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0] = [80, 150, 255]
for pts, sc, d, vr in b_bridges:
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w:
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
axes[0,3].imshow(vis.astype(np.uint8))
axes[0,3].set_title(f'B AFTER bridging\nEP={len(b_ep_a)} ({len(b_matches)} bridges)', fontsize=11)

# Row 2: Combined + zoom
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_skel.astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis[cv2.dilate(b_skel.astype(np.uint8), dk4) > 0] = [80, 150, 255]
axes[1,0].imshow(vis.astype(np.uint8))
axes[1,0].set_title('Combined BEFORE', fontsize=12)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis[cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0] = [80, 150, 255]
for pts, *_ in r_bridges:
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w:
            cv2.circle(vis, (x, y), 2, (255, 200, 50), -1)
for pts, *_ in b_bridges:
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w:
            cv2.circle(vis, (x, y), 2, (50, 255, 200), -1)
axes[1,1].imshow(vis.astype(np.uint8))
axes[1,1].set_title('Combined AFTER', fontsize=12)

# Base skel reference
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(base_skel.astype(np.uint8), dk4) > 0] = [0, 255, 0]
axes[1,2].imshow(vis.astype(np.uint8))
axes[1,2].set_title(f'Base skel ref\nEP={len(base_ep)} JN={len(base_jn)} L={int(np.sum(base_skel))}', fontsize=11)

# Metrics table
axes[1,3].axis('off')
txt = (f"=== Metrics ===\n\n"
       f"R skel (R>B filtered):\n"
       f"  Before: EP={len(r_ep):3d} JN={len(r_jn):3d} L={int(np.sum(r_skel)):5d}\n"
       f"  After:  EP={len(r_ep_a):3d} JN={len(r_jn_a):3d} L={int(np.sum(r_bridged)):5d}\n"
       f"  Bridges: {len(r_matches)}\n\n"
       f"B skel (B>R filtered):\n"
       f"  Before: EP={len(b_ep):3d} JN={len(b_jn):3d} L={int(np.sum(b_skel)):5d}\n"
       f"  After:  EP={len(b_ep_a):3d} JN={len(b_jn_a):3d} L={int(np.sum(b_bridged)):5d}\n"
       f"  Bridges: {len(b_matches)}\n\n"
       f"Base skel:\n"
       f"  EP={len(base_ep):3d} JN={len(base_jn):3d} L={int(np.sum(base_skel)):5d}")
axes[1,3].text(0.05, 0.9, txt, fontsize=12, transform=axes[1,3].transAxes,
               fontfamily='monospace', color='white', va='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

for ax in axes.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'skel_gap_bridge_v2.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved skel_gap_bridge_v2.png')

# ===== Figure 3: 개별 bridge zoom =====
all_info = []
for i, j, sc, d, vr in r_matches:
    all_info.append(('R', r_eps[i], r_eps[j], sc, d, vr, r_skel))
for i, j, sc, d, vr in b_matches:
    all_info.append(('B', b_eps[i], b_eps[j], sc, d, vr, b_skel))
all_info.sort(key=lambda x: -x[3])

n_show = min(8, len(all_info))
if n_show > 0:
    cols = min(4, n_show)
    rows = (n_show + cols - 1) // cols
    fig3, axes3 = plt.subplots(rows, cols, figsize=(8*cols, 7*rows))
    if rows == 1 and cols == 1:
        axes3 = np.array([[axes3]])
    elif rows == 1:
        axes3 = axes3[np.newaxis, :]
    elif cols == 1:
        axes3 = axes3[:, np.newaxis]
    fig3.suptitle('Individual bridges (zoom)', fontsize=16, fontweight='bold')

    for idx in range(n_show):
        ax = axes3.flat[idx]
        cn, ep1, ep2, sc, d, vr, skel = all_info[idx]
        p1 = np.array(ep1['pos'])
        p2 = np.array(ep2['pos'])
        center = ((p1 + p2) / 2).astype(int)
        crop = max(50, int(d * 0.8))
        y0 = max(0, center[0]-crop); y1 = min(h, center[0]+crop)
        x0 = max(0, center[1]-crop); x1 = min(w, center[1]+crop)

        z = gt_raw[y0:y1, x0:x1].copy().astype(float) * 0.3
        c = [255, 80, 80] if cn == 'R' else [80, 150, 255]
        z[cv2.dilate(skel[y0:y1, x0:x1].astype(np.uint8), dk4) > 0] = c

        for ep in [ep1, ep2]:
            ey, ex = ep['pos']
            if y0 <= ey < y1 and x0 <= ex < x1:
                cv2.circle(z, (ex-x0, ey-y0), 5, (255, 255, 0), -1)
                ty, tx = ep['tangent']
                cv2.arrowedLine(z, (ex-x0, ey-y0),
                               (int(ex-x0+tx*25), int(ey-y0+ty*25)),
                               (255, 255, 0), 2, tipLength=0.3)

        bridge = _bridge_points(ep1, ep2)
        for (py, px) in bridge:
            if y0 <= py < y1 and x0 <= px < x1:
                cv2.circle(z, (px-x0, py-y0), 2, (0, 255, 0), -1)

        ax.imshow(z.astype(np.uint8))
        ax.set_title(f'{cn} #{idx+1} sc={sc:.2f} d={d:.0f} v={vr:.0%}', fontsize=10)
        ax.axis('off')

    for idx in range(n_show, rows*cols):
        axes3.flat[idx].axis('off')
    plt.tight_layout()
    plt.savefig(OUT / 'skel_gap_bridge_v2_detail.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved skel_gap_bridge_v2_detail.png')

print('Done!')
