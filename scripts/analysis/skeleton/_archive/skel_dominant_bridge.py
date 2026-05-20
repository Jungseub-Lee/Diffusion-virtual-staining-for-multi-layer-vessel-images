"""
rgb_dominant_filter_margins.png 기준 (margin=0):
- R skel full → R>B인 영역만 남기기 (diff_BR < 0)
- B skel full → B>R인 영역만 남기기 (diff_BR > 0)
- 각각 gap bridging 적용
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

vessel_mask = gray > 10
vessel_mask = cv2.morphologyEx(vessel_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
vessel_mask = remove_small_objects(vessel_mask, min_size=100)
vessel_mask_loose = cv2.dilate(vessel_mask.astype(np.uint8)*255, k5) > 0

# ===== Full skeletons =====
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

base_skel = skeletonize(vessel_mask)

# ===== Dominance filter (margin=0): B>R / R>B =====
r_skel = r_skel_full & (diff_BR <= 0)  # R > B
b_skel = b_skel_full & (diff_BR >= 0)  # B > R

# 작은 fragment 제거
for skel_ref in [r_skel, b_skel]:
    labels, n = ndimage.label(skel_ref)
    for i in range(1, n + 1):
        if np.sum(labels == i) < 8:
            skel_ref[labels == i] = False

def skel_features(skel):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    return np.argwhere(nb == 1), np.argwhere(nb >= 3)

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
            p_end = np.array(path[0], dtype=float)
            p_inner = np.array(path[n_avg-1], dtype=float)
            tangent = p_end - p_inner
            norm = np.linalg.norm(tangent)
            if norm > 0: tangent = tangent / norm
            results.append({'pos': (y0, x0), 'tangent': tangent, 'path': path})
    return results

def bridge_points(ep1, ep2):
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
        y, x = int(round(p[0])), int(round(p[1]))
        pts.append((y, x))
    return pts

def match_and_bridge(skel, vessel_mask, max_dist=80, max_angle_deg=60, trace_len=20):
    eps = get_endpoints_with_tangent(skel, trace_len)
    candidates = []
    n = len(eps)
    for i in range(n):
        for j in range(i+1, n):
            p1 = np.array(eps[i]['pos'], dtype=float)
            p2 = np.array(eps[j]['pos'], dtype=float)
            dist = np.linalg.norm(p2 - p1)
            if dist < 5 or dist > max_dist: continue
            dir_12 = (p2 - p1) / dist
            cos1 = np.dot(eps[i]['tangent'], dir_12)
            cos2 = np.dot(eps[j]['tangent'], -dir_12)
            thresh = np.cos(np.radians(max_angle_deg))
            if cos1 < thresh or cos2 < thresh: continue
            pts = bridge_points(eps[i], eps[j])
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
    bridges = []
    for i, j, sc, d, vr in matches:
        pts = bridge_points(eps[i], eps[j])
        bridges.append((pts, sc, d, vr, eps[i], eps[j]))
        for (y, x) in pts:
            if 0 <= y < h and 0 <= x < w:
                bridged[y, x] = True

    return bridged, bridges, eps, matches

# Run bridging
print("=== R skeleton (R>B, margin=0) ===")
r_ep_b, r_jn_b = skel_features(r_skel)
print(f"Before: L={int(np.sum(r_skel))} EP={len(r_ep_b)} JN={len(r_jn_b)}")
r_bridged, r_bridges, r_eps, r_matches = match_and_bridge(r_skel, vessel_mask_loose)
r_ep_a, r_jn_a = skel_features(r_bridged)
print(f"After:  L={int(np.sum(r_bridged))} EP={len(r_ep_a)} JN={len(r_jn_a)} Bridges={len(r_bridges)}")

print("\n=== B skeleton (B>R, margin=0) ===")
b_ep_b, b_jn_b = skel_features(b_skel)
print(f"Before: L={int(np.sum(b_skel))} EP={len(b_ep_b)} JN={len(b_jn_b)}")
b_bridged, b_bridges, b_eps, b_matches = match_and_bridge(b_skel, vessel_mask_loose)
b_ep_a, b_jn_a = skel_features(b_bridged)
print(f"After:  L={int(np.sum(b_bridged))} EP={len(b_ep_a)} JN={len(b_jn_a)} Bridges={len(b_bridges)}")

base_ep, base_jn = skel_features(base_skel)
print(f"\nBase:   L={int(np.sum(base_skel))} EP={len(base_ep)} JN={len(base_jn)}")

# ===== Figure 1: Before / After / Combined =====
fig, axes = plt.subplots(3, 4, figsize=(36, 24))
fig.suptitle('Dominance filter (margin=0) + Gap Bridging', fontsize=18, fontweight='bold')

# Row 0: R
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_skel.astype(np.uint8), dk4) > 0] = [255, 80, 80]
for ep in r_eps:
    y, x = ep['pos']
    cv2.circle(vis, (x, y), 4, (255, 255, 0), -1)
    ty, tx = ep['tangent']
    cv2.arrowedLine(vis, (x, y), (int(x+tx*20), int(y+ty*20)), (255, 255, 0), 1, tipLength=0.3)
axes[0,0].imshow(vis.astype(np.uint8))
axes[0,0].set_title(f'R (R>B) BEFORE\nL={int(np.sum(r_skel))} EP={len(r_ep_b)}', fontsize=12)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0] = [255, 80, 80]
for pts, sc, d, vr, ep1, ep2 in r_bridges:
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w:
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
axes[0,1].imshow(vis.astype(np.uint8))
axes[0,1].set_title(f'R AFTER bridging\nL={int(np.sum(r_bridged))} EP={len(r_ep_a)} (+{len(r_bridges)} bridges)', fontsize=11)

# Row 0: B
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(b_skel.astype(np.uint8), dk4) > 0] = [80, 150, 255]
for ep in b_eps:
    y, x = ep['pos']
    cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
    ty, tx = ep['tangent']
    cv2.arrowedLine(vis, (x, y), (int(x+tx*20), int(y+ty*20)), (0, 255, 255), 1, tipLength=0.3)
axes[0,2].imshow(vis.astype(np.uint8))
axes[0,2].set_title(f'B (B>R) BEFORE\nL={int(np.sum(b_skel))} EP={len(b_ep_b)}', fontsize=12)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0] = [80, 150, 255]
for pts, sc, d, vr, ep1, ep2 in b_bridges:
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w:
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
axes[0,3].imshow(vis.astype(np.uint8))
axes[0,3].set_title(f'B AFTER bridging\nL={int(np.sum(b_bridged))} EP={len(b_ep_a)} (+{len(b_bridges)} bridges)', fontsize=11)

# Row 1: Combined before/after + base ref
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

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(base_skel.astype(np.uint8), dk4) > 0] = [0, 255, 0]
axes[1,2].imshow(vis.astype(np.uint8))
axes[1,2].set_title(f'Base skel (ref)\nL={int(np.sum(base_skel))} EP={len(base_ep)} JN={len(base_jn)}', fontsize=11)

# Skeleton only
s_vis = np.zeros((h, w, 3), dtype=np.uint8)
s_vis[cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0] = [220, 60, 60]
s_vis[cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0] = [60, 130, 220]
both = (cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0) & (cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0)
s_vis[both] = [180, 50, 180]
axes[1,3].imshow(s_vis)
axes[1,3].set_title('R+B bridged skeleton only', fontsize=12)

# Row 2: Zoom into bridge locations
all_bridges = [(b, 'R') for b in r_bridges] + [(b, 'B') for b in b_bridges]
all_bridges.sort(key=lambda x: -x[0][1])  # score순

for col in range(4):
    if col < len(all_bridges):
        (pts, sc, d, vr, ep1, ep2), cn = all_bridges[col]
        p1, p2 = np.array(ep1['pos']), np.array(ep2['pos'])
        center = ((p1 + p2) / 2).astype(int)
        crop = max(50, int(d * 0.8))
        y0 = max(0, center[0]-crop); y1 = min(h, center[0]+crop)
        x0 = max(0, center[1]-crop); x1 = min(w, center[1]+crop)

        skel = r_skel if cn == 'R' else b_skel
        bridged = r_bridged if cn == 'R' else b_bridged
        c = [255, 80, 80] if cn == 'R' else [80, 150, 255]

        z = gt_raw[y0:y1, x0:x1].copy().astype(float) * 0.3
        z[cv2.dilate(skel[y0:y1, x0:x1].astype(np.uint8), dk4) > 0] = c
        # bridge
        for (py, px) in pts:
            if y0 <= py < y1 and x0 <= px < x1:
                cv2.circle(z, (px-x0, py-y0), 3, (0, 255, 0), -1)
        # endpoints + tangent
        for ep in [ep1, ep2]:
            ey, ex = ep['pos']
            if y0 <= ey < y1 and x0 <= ex < x1:
                cv2.circle(z, (ex-x0, ey-y0), 5, (255, 255, 0), -1)
                ty, tx = ep['tangent']
                cv2.arrowedLine(z, (ex-x0, ey-y0),
                               (int(ex-x0+tx*25), int(ey-y0+ty*25)),
                               (255, 255, 0), 2, tipLength=0.3)

        axes[2, col].imshow(z.astype(np.uint8))
        axes[2, col].set_title(f'{cn} bridge sc={sc:.2f} d={d:.0f}', fontsize=10)
    else:
        axes[2, col].axis('off')

for ax in axes.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'skel_dominant_bridge.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('\nSaved skel_dominant_bridge.png')

# ===== Print bridge details =====
print("\n=== R bridges ===")
for idx, (pts, sc, d, vr, ep1, ep2) in enumerate(r_bridges):
    print(f"  #{idx+1}: ({ep1['pos']})→({ep2['pos']}) dist={d:.0f} score={sc:.2f} vessel={vr:.0%}")
print(f"\n=== B bridges ===")
for idx, (pts, sc, d, vr, ep1, ep2) in enumerate(b_bridges):
    print(f"  #{idx+1}: ({ep1['pos']})→({ep2['pos']}) dist={d:.0f} score={sc:.2f} vessel={vr:.0%}")

print('\nDone!')
