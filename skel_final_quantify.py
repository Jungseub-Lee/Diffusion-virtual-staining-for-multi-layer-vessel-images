"""
Final: R/B 개별 bridge → cross-color bridge (R↔B) → 정량화
하단 20% 제외하고 비교: single GT vs multi GT (R+B)
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

boundary_y = int(h * 0.8)
top_mask = np.zeros((h, w), dtype=bool)
top_mask[:boundary_y, :] = True

# ===== Skeletons (원본 rgb_dominant_filter.py 동일) =====
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

vessel_all = remove_small_objects(
    cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0,
    min_size=100)
base_skel = skeletonize(vessel_all)
vessel_mask_loose = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

# Dominance filter (margin=0)
r_filtered = r_skel & (diff_BR < 0)
b_filtered = b_skel & (diff_BR > 0)

# ===== Bridge functions =====
def skel_features(skel, mask=None):
    s = skel & mask if mask is not None else skel
    su = s.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    ep = np.argwhere(nb == 1)
    jn = np.argwhere(nb >= 3)
    length = int(np.sum(s))
    return ep, jn, length

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
    bridges = []
    for i, j, sc, d, vr in matches:
        pts = _bridge_pts(eps[i], eps[j])
        bridges.append((pts, sc, d, vr, eps[i], eps[j]))
        for (y, x) in pts:
            if 0 <= y < h and 0 <= x < w:
                bridged[y, x] = True
    return bridged, bridges, eps

def do_cross_bridge(skel_a, skel_b, vessel_mask, max_dist=80, max_angle_deg=60):
    """Cross-color bridging: skel_a의 endpoint → skel_b의 endpoint 연결"""
    eps_a = get_endpoints_with_tangent(skel_a, trace_len=20)
    eps_b = get_endpoints_with_tangent(skel_b, trace_len=20)
    candidates = []
    for i, ea in enumerate(eps_a):
        for j, eb in enumerate(eps_b):
            p1 = np.array(ea['pos'], dtype=float)
            p2 = np.array(eb['pos'], dtype=float)
            dist = np.linalg.norm(p2 - p1)
            if dist < 3 or dist > max_dist: continue
            dir_12 = (p2 - p1) / dist
            cos1 = np.dot(ea['tangent'], dir_12)
            cos2 = np.dot(eb['tangent'], -dir_12)
            thresh = np.cos(np.radians(max_angle_deg))
            if cos1 < thresh or cos2 < thresh: continue
            pts = _bridge_pts(ea, eb)
            on_v = sum(1 for (y, x) in pts if 0 <= y < h and 0 <= x < w and vessel_mask[y, x])
            vr = on_v / max(len(pts), 1)
            if vr < 0.7: continue
            score = (cos1 + cos2) / 2 - dist / max_dist * 0.2 + vr * 0.2
            candidates.append((i, j, score, dist, vr))
    candidates.sort(key=lambda x: -x[2])
    used_a, used_b = set(), set()
    matches = []
    for i, j, sc, d, vr in candidates:
        if i not in used_a and j not in used_b:
            matches.append((i, j, sc, d, vr))
            used_a.add(i); used_b.add(j)
    bridges = []
    for i, j, sc, d, vr in matches:
        pts = _bridge_pts(eps_a[i], eps_b[j])
        bridges.append((pts, sc, d, vr, eps_a[i], eps_b[j]))
    return bridges

# ===== Step 1: 개별 bridge =====
print("=== Step 1: Individual bridging ===")
r_bridged, r_bridges, r_eps = do_bridging(r_filtered, vessel_mask_loose)
b_bridged, b_bridges, b_eps = do_bridging(b_filtered, vessel_mask_loose)
print(f"R: {len(r_bridges)} bridges")
print(f"B: {len(b_bridges)} bridges")

# ===== Step 2: Cross-color bridge (R↔B) =====
print("\n=== Step 2: Cross-color bridging (R↔B) ===")
cross_bridges = do_cross_bridge(r_bridged, b_bridged, vessel_mask_loose, max_dist=60, max_angle_deg=50)
print(f"Cross bridges: {len(cross_bridges)}")

# Cross bridge를 양쪽 skel에 추가 (연결 지점)
r_final = r_bridged.copy()
b_final = b_bridged.copy()
cross_skel = np.zeros((h, w), dtype=bool)  # cross bridge만 별도 저장
for pts, sc, d, vr, ep1, ep2 in cross_bridges:
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w:
            cross_skel[y, x] = True
    print(f"  Cross: ({ep1['pos']})↔({ep2['pos']}) dist={d:.0f} score={sc:.2f}")

# Combined: R + B + cross
combined_skel = r_final | b_final | cross_skel

# ===== Step 3: 정량화 (top 80% only) =====
print("\n" + "="*60)
print("=== QUANTIFICATION (top 80% only, y < {}) ===".format(boundary_y))
print("="*60)

# Single GT = base skel
single_ep, single_jn, single_L = skel_features(base_skel, top_mask)

# Multi GT: R bridged
r_ep, r_jn, r_L = skel_features(r_final, top_mask)
# Multi GT: B bridged
b_ep, b_jn, b_L = skel_features(b_final, top_mask)
# Multi GT: combined (R + B + cross)
c_ep, c_jn, c_L = skel_features(combined_skel, top_mask)

# R full / B full (before filter)
rf_ep, rf_jn, rf_L = skel_features(r_skel, top_mask)
bf_ep, bf_jn, bf_L = skel_features(b_skel, top_mask)

print(f"\n{'':30s} {'Length':>8s} {'EP':>6s} {'JN':>6s}")
print("-" * 55)
print(f"{'Single GT (base skel)':30s} {single_L:8d} {len(single_ep):6d} {len(single_jn):6d}")
print(f"{'':30s}")
print(f"{'R skel full':30s} {rf_L:8d} {len(rf_ep):6d} {len(rf_jn):6d}")
print(f"{'R filtered+bridged':30s} {r_L:8d} {len(r_ep):6d} {len(r_jn):6d}")
print(f"{'B skel full':30s} {bf_L:8d} {len(bf_ep):6d} {len(bf_jn):6d}")
print(f"{'B filtered+bridged':30s} {b_L:8d} {len(b_ep):6d} {len(b_jn):6d}")
print(f"{'':30s}")
print(f"{'Multi GT (R+B+cross)':30s} {c_L:8d} {len(c_ep):6d} {len(c_jn):6d}")
print(f"{'Cross bridges':30s} {'':8s} {len(cross_bridges):6d}")

# ===== Figure 1: Full pipeline =====
fig, axes = plt.subplots(3, 4, figsize=(36, 24))
fig.suptitle('Final Pipeline: Dominance filter → Bridge → Cross-bridge → Quantify (top 80%)',
             fontsize=16, fontweight='bold')

# Row 0: R pipeline
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0] = [255, 80, 80]
axes[0,0].imshow(vis.astype(np.uint8))
axes[0,0].set_title(f'R filtered (R>B)', fontsize=12)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0] = [255, 80, 80]
for pts, *_ in r_bridges:
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w: cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
axes[0,1].imshow(vis.astype(np.uint8))
axes[0,1].set_title(f'R bridged (+{len(r_bridges)})', fontsize=12)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0] = [80, 150, 255]
axes[0,2].imshow(vis.astype(np.uint8))
axes[0,2].set_title(f'B filtered (B>R)', fontsize=12)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0] = [80, 150, 255]
for pts, *_ in b_bridges:
    for (y, x) in pts:
        if 0 <= y < h and 0 <= x < w: cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
axes[0,3].imshow(vis.astype(np.uint8))
axes[0,3].set_title(f'B bridged (+{len(b_bridges)})', fontsize=12)

# Row 1: Combined + cross bridge
vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis[cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0] = [80, 150, 255]
axes[1,0].imshow(vis.astype(np.uint8))
axes[1,0].set_title('R+B before cross-bridge', fontsize=12)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(r_final.astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis[cv2.dilate(b_final.astype(np.uint8), dk4) > 0] = [80, 150, 255]
vis[cv2.dilate(cross_skel.astype(np.uint8), dk4) > 0] = [0, 255, 0]
axes[1,1].imshow(vis.astype(np.uint8))
axes[1,1].set_title(f'R+B + cross-bridge (green)\n{len(cross_bridges)} cross bridges', fontsize=11)

vis = gt_raw.copy().astype(float) * 0.25
vis[cv2.dilate(base_skel.astype(np.uint8), dk4) > 0] = [0, 255, 0]
vis[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[1,2].imshow(vis.astype(np.uint8))
axes[1,2].set_title('Single GT (base skel)\nyellow=boundary', fontsize=12)

# Skeleton only comparison
s_vis = np.zeros((h, w, 3), dtype=np.uint8)
s_vis[cv2.dilate(r_final.astype(np.uint8), dk4) > 0] = [220, 60, 60]
s_vis[cv2.dilate(b_final.astype(np.uint8), dk4) > 0] = [60, 130, 220]
s_vis[cv2.dilate(cross_skel.astype(np.uint8), dk4) > 0] = [0, 220, 0]
s_vis[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[1,3].imshow(s_vis)
axes[1,3].set_title('Multi GT skeleton only\nR/B/cross(green)', fontsize=12)

# Row 2: Quantification comparison
# Bar chart
labels = ['Single GT\n(base)', 'R filtered\n+bridged', 'B filtered\n+bridged', 'Multi GT\n(R+B+cross)']
lengths = [single_L, r_L, b_L, c_L]
eps_vals = [len(single_ep), len(r_ep), len(b_ep), len(c_ep)]
jns_vals = [len(single_jn), len(r_jn), len(b_jn), len(c_jn)]
colors_bar = ['green', 'red', 'blue', 'purple']

x_pos = np.arange(len(labels))
axes[2,0].bar(x_pos, lengths, color=colors_bar, alpha=0.7, edgecolor='white')
axes[2,0].set_xticks(x_pos)
axes[2,0].set_xticklabels(labels, fontsize=9)
axes[2,0].set_title('Length (top 80%)', fontsize=13)
axes[2,0].set_ylabel('pixels')
for i, v in enumerate(lengths):
    axes[2,0].text(i, v + 20, str(v), ha='center', fontsize=10, fontweight='bold', color='white')

axes[2,1].bar(x_pos, eps_vals, color=colors_bar, alpha=0.7, edgecolor='white')
axes[2,1].set_xticks(x_pos)
axes[2,1].set_xticklabels(labels, fontsize=9)
axes[2,1].set_title('Endpoints (top 80%)', fontsize=13)
axes[2,1].set_ylabel('count')
for i, v in enumerate(eps_vals):
    axes[2,1].text(i, v + 0.5, str(v), ha='center', fontsize=10, fontweight='bold', color='white')

axes[2,2].bar(x_pos, jns_vals, color=colors_bar, alpha=0.7, edgecolor='white')
axes[2,2].set_xticks(x_pos)
axes[2,2].set_xticklabels(labels, fontsize=9)
axes[2,2].set_title('Junctions (top 80%)', fontsize=13)
axes[2,2].set_ylabel('count')
for i, v in enumerate(jns_vals):
    axes[2,2].text(i, v + 1, str(v), ha='center', fontsize=10, fontweight='bold', color='white')

# Summary table
axes[2,3].axis('off')
txt = (f"{'='*45}\n"
       f"  QUANTIFICATION (top 80%, y<{boundary_y})\n"
       f"{'='*45}\n\n"
       f"{'':24s} {'L':>6s} {'EP':>5s} {'JN':>5s}\n"
       f"{'-'*45}\n"
       f"{'Single GT (base)':24s} {single_L:6d} {len(single_ep):5d} {len(single_jn):5d}\n\n"
       f"{'R filtered+bridged':24s} {r_L:6d} {len(r_ep):5d} {len(r_jn):5d}\n"
       f"{'B filtered+bridged':24s} {b_L:6d} {len(b_ep):5d} {len(b_jn):5d}\n"
       f"{'Multi GT (R+B+cross)':24s} {c_L:6d} {len(c_ep):5d} {len(c_jn):5d}\n\n"
       f"{'Cross bridges':24s} {'':6s} {len(cross_bridges):5d}\n"
       f"{'-'*45}\n"
       f"R+B total length:        {r_L+b_L:6d}\n"
       f"Single GT / Multi ratio: {single_L/(c_L+0.01):.2f}x")
axes[2,3].text(0.02, 0.95, txt, fontsize=12, transform=axes[2,3].transAxes,
               fontfamily='monospace', color='white', va='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.9))

for ax in axes.flat:
    if ax.has_data() and not hasattr(ax, '_adjustable'):
        pass  # keep
    ax.set_facecolor('black') if not ax.has_data() else None

plt.tight_layout()
plt.savefig(OUT / 'skel_final_quantify.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('\nSaved skel_final_quantify.png')

# ===== Figure 2: Cross-bridge zoom =====
if cross_bridges:
    n_show = min(6, len(cross_bridges))
    cols = min(3, n_show)
    rows = (n_show + cols - 1) // cols
    fig2, axes2 = plt.subplots(rows, cols, figsize=(10*cols, 8*rows))
    if rows == 1 and cols == 1: axes2 = np.array([[axes2]])
    elif rows == 1: axes2 = axes2[np.newaxis, :]
    elif cols == 1: axes2 = axes2[:, np.newaxis]
    fig2.suptitle('Cross-color bridges (R↔B)', fontsize=16, fontweight='bold')

    for idx in range(n_show):
        ax = axes2.flat[idx]
        pts, sc, d, vr, ep1, ep2 = cross_bridges[idx]
        p1, p2 = np.array(ep1['pos']), np.array(ep2['pos'])
        center = ((p1 + p2) / 2).astype(int)
        crop = max(50, int(d * 0.8))
        y0 = max(0, center[0]-crop); y1 = min(h, center[0]+crop)
        x0 = max(0, center[1]-crop); x1 = min(w, center[1]+crop)

        z = gt_raw[y0:y1, x0:x1].copy().astype(float) * 0.3
        z[cv2.dilate(r_bridged[y0:y1, x0:x1].astype(np.uint8), dk4) > 0] = [255, 80, 80]
        z[cv2.dilate(b_bridged[y0:y1, x0:x1].astype(np.uint8), dk4) > 0] = [80, 150, 255]
        for (py, px) in pts:
            if y0 <= py < y1 and x0 <= px < x1:
                cv2.circle(z, (px-x0, py-y0), 3, (0, 255, 0), -1)
        for ep, c in [(ep1, (255,255,0)), (ep2, (0,255,255))]:
            ey, ex = ep['pos']
            if y0 <= ey < y1 and x0 <= ex < x1:
                cv2.circle(z, (ex-x0, ey-y0), 5, c, -1)
                ty, tx = ep['tangent']
                cv2.arrowedLine(z, (ex-x0, ey-y0),
                               (int(ex-x0+tx*25), int(ey-y0+ty*25)), c, 2, tipLength=0.3)
        ax.imshow(z.astype(np.uint8))
        ax.set_title(f'R↔B sc={sc:.2f} d={d:.0f}', fontsize=11)
        ax.axis('off')

    for idx in range(n_show, rows*cols):
        axes2.flat[idx].axis('off')
    plt.tight_layout()
    plt.savefig(OUT / 'skel_cross_bridge_detail.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved skel_cross_bridge_detail.png')

print('\nDone!')
