"""
rgb_dominant_filter.py의 skeleton 생성 로직 그대로 사용 + gap bridging
margin=0 기준: B skel에서 B>R, R skel에서 R>B
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

# ===== 원본 rgb_dominant_filter.py와 동일한 skeleton 생성 =====
# R skeleton (thresh=30)
r_mask = R > 30
r_mask = cv2.morphologyEx(r_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
r_mask = remove_small_objects(r_mask, min_size=50)
r_skel = skeletonize(r_mask)

# B skeleton (hysteresis seed=50, low=20)
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

# Base skel (참고)
vessel_all = remove_small_objects(
    cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0,
    min_size=100)
base_skel = skeletonize(vessel_all)

# Vessel mask for bridge validation
vessel_mask = vessel_all
vessel_mask_loose = cv2.dilate(vessel_mask.astype(np.uint8)*255, k5) > 0

# ===== Dominance filter (margin=0) — 원본과 동일 =====
b_dom = diff_BR > 0   # B > R
r_dom = diff_BR < 0   # R > B

b_filtered = b_skel & b_dom
r_filtered = r_skel & r_dom

def skel_features(skel):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    return np.argwhere(nb == 1), np.argwhere(nb >= 3)

print("=== Full skeletons ===")
r_ep_f, r_jn_f = skel_features(r_skel)
b_ep_f, b_jn_f = skel_features(b_skel)
print(f"R skel full: L={int(np.sum(r_skel))} EP={len(r_ep_f)} JN={len(r_jn_f)}")
print(f"B skel full: L={int(np.sum(b_skel))} EP={len(b_ep_f)} JN={len(b_jn_f)}")

print("\n=== After dominance filter (margin=0) ===")
r_ep_d, r_jn_d = skel_features(r_filtered)
b_ep_d, b_jn_d = skel_features(b_filtered)
print(f"R filtered (R>B): L={int(np.sum(r_filtered))} EP={len(r_ep_d)} JN={len(r_jn_d)}")
print(f"B filtered (B>R): L={int(np.sum(b_filtered))} EP={len(b_ep_d)} JN={len(b_jn_d)}")

# ===== Gap bridging functions =====
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

def match_and_bridge(skel, vessel_mask, max_dist=80, max_angle_deg=60):
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
    bridges = []
    for i, j, sc, d, vr in matches:
        pts = _bridge_pts(eps[i], eps[j])
        bridges.append((pts, sc, d, vr, eps[i], eps[j]))
        for (y, x) in pts:
            if 0 <= y < h and 0 <= x < w:
                bridged[y, x] = True
    return bridged, bridges, eps

# ===== Bridge R and B filtered skeletons =====
print("\n=== R bridging ===")
r_bridged, r_bridges, r_eps = match_and_bridge(r_filtered, vessel_mask_loose)
r_ep_br, r_jn_br = skel_features(r_bridged)
print(f"R bridged: L={int(np.sum(r_bridged))} EP={len(r_ep_br)} JN={len(r_jn_br)} Bridges={len(r_bridges)}")

print("\n=== B bridging ===")
b_bridged, b_bridges, b_eps = match_and_bridge(b_filtered, vessel_mask_loose)
b_ep_br, b_jn_br = skel_features(b_bridged)
print(f"B bridged: L={int(np.sum(b_bridged))} EP={len(b_ep_br)} JN={len(b_jn_br)} Bridges={len(b_bridges)}")

base_ep, base_jn = skel_features(base_skel)
print(f"\nBase: L={int(np.sum(base_skel))} EP={len(base_ep)} JN={len(base_jn)}")

# ===== Figure 1: 전체 과정 =====
fig, axes = plt.subplots(3, 4, figsize=(36, 24))
fig.suptitle('Dominance (margin=0) + Gap Bridging — using original skel code', fontsize=18, fontweight='bold')

# Row 0: Full skel → Filtered
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_skel.astype(np.uint8), dk4) > 0] = [255, 80, 80]
axes[0,0].imshow(vis.astype(np.uint8))
axes[0,0].set_title(f'R skel full (R>30)\nL={int(np.sum(r_skel))} EP={len(r_ep_f)} JN={len(r_jn_f)}', fontsize=11)

vis = gt_raw.copy().astype(float) * 0.25
# dim: full skel, bright: filtered
vis[cv2.dilate(r_skel.astype(np.uint8), dk3) > 0] = [100, 40, 40]
vis[cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0] = [255, 80, 80]
axes[0,1].imshow(vis.astype(np.uint8))
axes[0,1].set_title(f'R filtered (R>B)\nbright={int(np.sum(r_filtered))}/{int(np.sum(r_skel))} ({100*np.sum(r_filtered)/max(np.sum(r_skel),1):.0f}%)', fontsize=11)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(b_skel.astype(np.uint8), dk4) > 0] = [80, 150, 255]
axes[0,2].imshow(vis.astype(np.uint8))
axes[0,2].set_title(f'B skel full (hyst 50/20)\nL={int(np.sum(b_skel))} EP={len(b_ep_f)} JN={len(b_jn_f)}', fontsize=11)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(b_skel.astype(np.uint8), dk3) > 0] = [40, 60, 100]
vis[cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0] = [80, 150, 255]
axes[0,3].imshow(vis.astype(np.uint8))
axes[0,3].set_title(f'B filtered (B>R)\nbright={int(np.sum(b_filtered))}/{int(np.sum(b_skel))} ({100*np.sum(b_filtered)/max(np.sum(b_skel),1):.0f}%)', fontsize=11)

# Row 1: Filtered + endpoints → Bridged
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0] = [255, 80, 80]
for ep in r_eps:
    y, x = ep['pos']
    cv2.circle(vis, (x, y), 4, (255, 255, 0), -1)
    ty, tx = ep['tangent']
    cv2.arrowedLine(vis, (x, y), (int(x+tx*18), int(y+ty*18)), (255, 255, 0), 1, tipLength=0.3)
axes[1,0].imshow(vis.astype(np.uint8))
axes[1,0].set_title(f'R filtered + endpoints\nEP={len(r_ep_d)}', fontsize=11)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0] = [255, 80, 80]
for pts, sc, d, vr, ep1, ep2 in r_bridges:
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w:
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
axes[1,1].imshow(vis.astype(np.uint8))
axes[1,1].set_title(f'R bridged\nL={int(np.sum(r_bridged))} EP={len(r_ep_br)} (+{len(r_bridges)} bridges)', fontsize=11)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0] = [80, 150, 255]
for ep in b_eps:
    y, x = ep['pos']
    cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
    ty, tx = ep['tangent']
    cv2.arrowedLine(vis, (x, y), (int(x+tx*18), int(y+ty*18)), (0, 255, 255), 1, tipLength=0.3)
axes[1,2].imshow(vis.astype(np.uint8))
axes[1,2].set_title(f'B filtered + endpoints\nEP={len(b_ep_d)}', fontsize=11)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0] = [80, 150, 255]
for pts, sc, d, vr, ep1, ep2 in b_bridges:
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w:
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
axes[1,3].imshow(vis.astype(np.uint8))
axes[1,3].set_title(f'B bridged\nL={int(np.sum(b_bridged))} EP={len(b_ep_br)} (+{len(b_bridges)} bridges)', fontsize=11)

# Row 2: Combined before/after + base + metrics
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis[cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0] = [80, 150, 255]
axes[2,0].imshow(vis.astype(np.uint8))
axes[2,0].set_title('Combined BEFORE bridge', fontsize=12)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis[cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0] = [80, 150, 255]
for pts, *_ in r_bridges:
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w: cv2.circle(vis, (x, y), 2, (255, 200, 50), -1)
for pts, *_ in b_bridges:
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w: cv2.circle(vis, (x, y), 2, (50, 255, 200), -1)
axes[2,1].imshow(vis.astype(np.uint8))
axes[2,1].set_title('Combined AFTER bridge', fontsize=12)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(base_skel.astype(np.uint8), dk4) > 0] = [0, 255, 0]
axes[2,2].imshow(vis.astype(np.uint8))
axes[2,2].set_title(f'Base skel (ref)\nL={int(np.sum(base_skel))} EP={len(base_ep)} JN={len(base_jn)}', fontsize=11)

axes[2,3].axis('off')
txt = (f"=== Summary ===\n\n"
       f"R skel full:      L={int(np.sum(r_skel)):5d}\n"
       f"R filtered (R>B): L={int(np.sum(r_filtered)):5d} EP={len(r_ep_d):3d}\n"
       f"R bridged:        L={int(np.sum(r_bridged)):5d} EP={len(r_ep_br):3d} (+{len(r_bridges)} bridges)\n\n"
       f"B skel full:      L={int(np.sum(b_skel)):5d}\n"
       f"B filtered (B>R): L={int(np.sum(b_filtered)):5d} EP={len(b_ep_d):3d}\n"
       f"B bridged:        L={int(np.sum(b_bridged)):5d} EP={len(b_ep_br):3d} (+{len(b_bridges)} bridges)\n\n"
       f"Base skel:        L={int(np.sum(base_skel)):5d} EP={len(base_ep):3d} JN={len(base_jn):3d}")
axes[2,3].text(0.05, 0.9, txt, fontsize=13, transform=axes[2,3].transAxes,
               fontfamily='monospace', color='white', va='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

for ax in axes.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'skel_dominant_bridge_v3.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('\nSaved skel_dominant_bridge_v3.png')

# ===== Figure 2: Bridge zoom =====
all_br = [(b, 'R') for b in r_bridges] + [(b, 'B') for b in b_bridges]
all_br.sort(key=lambda x: -x[0][1])
n_show = min(8, len(all_br))

if n_show > 0:
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig2, axes2 = plt.subplots(rows, cols, figsize=(32, 7*rows))
    if rows == 1: axes2 = axes2[np.newaxis, :]
    fig2.suptitle('Bridge details (zoom)', fontsize=16, fontweight='bold')

    for idx in range(n_show):
        ax = axes2.flat[idx]
        (pts, sc, d, vr, ep1, ep2), cn = all_br[idx]
        p1, p2 = np.array(ep1['pos']), np.array(ep2['pos'])
        center = ((p1 + p2) / 2).astype(int)
        crop = max(50, int(d * 0.8))
        y0 = max(0, center[0]-crop); y1 = min(h, center[0]+crop)
        x0 = max(0, center[1]-crop); x1 = min(w, center[1]+crop)

        skel = r_filtered if cn == 'R' else b_filtered
        c = [255, 80, 80] if cn == 'R' else [80, 150, 255]
        z = gt_raw[y0:y1, x0:x1].copy().astype(float) * 0.3
        z[cv2.dilate(skel[y0:y1, x0:x1].astype(np.uint8), dk4) > 0] = c
        for (py, px) in pts:
            if y0 <= py < y1 and x0 <= px < x1:
                cv2.circle(z, (px-x0, py-y0), 3, (0, 255, 0), -1)
        for ep in [ep1, ep2]:
            ey, ex = ep['pos']
            if y0 <= ey < y1 and x0 <= ex < x1:
                cv2.circle(z, (ex-x0, ey-y0), 5, (255, 255, 0), -1)
                ty, tx = ep['tangent']
                cv2.arrowedLine(z, (ex-x0, ey-y0),
                               (int(ex-x0+tx*25), int(ey-y0+ty*25)),
                               (255, 255, 0), 2, tipLength=0.3)
        ax.imshow(z.astype(np.uint8))
        ax.set_title(f'{cn} sc={sc:.2f} d={d:.0f} v={vr:.0%}', fontsize=10)
        ax.axis('off')

    for idx in range(n_show, rows*cols):
        axes2.flat[idx].axis('off')
    plt.tight_layout()
    plt.savefig(OUT / 'skel_dominant_bridge_v3_detail.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved skel_dominant_bridge_v3_detail.png')

print('Done!')
