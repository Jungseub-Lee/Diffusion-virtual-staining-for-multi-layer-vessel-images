"""
R/B skeleton gap bridging:
- 끊긴 endpoint 찾기
- endpoint의 tangent 방향 계산
- 방향 + 거리로 matching endpoint 찾기
- 자연스럽게 연결
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
from scipy.interpolate import CubicSpline
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

vessel_mask = gray > 10
vessel_mask = cv2.morphologyEx(vessel_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
vessel_mask = remove_small_objects(vessel_mask, min_size=100)

# ===== Endpoint + tangent 추출 =====
def get_endpoints_with_tangent(skel, trace_len=15):
    """
    skeleton에서 endpoint 찾고, 각 endpoint에서 skeleton을 따라가며
    tangent direction 계산
    """
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    endpoints = np.argwhere(nb == 1)  # (y, x)

    results = []
    for ep in endpoints:
        y0, x0 = ep
        # skeleton을 따라 trace_len만큼 추적
        path = [(y0, x0)]
        visited = {(y0, x0)}
        cy, cx = y0, x0

        for _ in range(trace_len):
            found_next = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and (ny, nx) not in visited:
                        path.append((ny, nx))
                        visited.add((ny, nx))
                        cy, cx = ny, nx
                        found_next = True
                        break
                if found_next:
                    break
            if not found_next:
                break

        if len(path) >= 3:
            # tangent = endpoint에서 멀어지는 방향 (endpoint → 안쪽)의 반대
            # 즉 혈관이 뻗어나가는 방향
            p_end = np.array(path[0], dtype=float)  # endpoint
            p_inner = np.array(path[-1], dtype=float)  # 안쪽 점
            tangent = p_end - p_inner  # endpoint에서 바깥으로 향하는 방향
            norm = np.linalg.norm(tangent)
            if norm > 0:
                tangent = tangent / norm
            results.append({
                'pos': (y0, x0),
                'tangent': tangent,  # (dy, dx) 방향, 바깥으로
                'path': path
            })

    return results

def match_endpoints(eps_list, max_dist=60, max_angle_deg=45):
    """
    endpoint 쌍 매칭:
    - 거리: max_dist 이내
    - 각도: ep1의 tangent 연장이 ep2 방향과 일치 (양쪽 모두)
    Returns: list of (ep1_idx, ep2_idx, score)
    """
    matches = []
    n = len(eps_list)
    used = set()

    # 모든 쌍 score 계산
    candidates = []
    for i in range(n):
        for j in range(i+1, n):
            p1 = np.array(eps_list[i]['pos'], dtype=float)
            p2 = np.array(eps_list[j]['pos'], dtype=float)
            t1 = eps_list[i]['tangent']
            t2 = eps_list[j]['tangent']

            dist = np.linalg.norm(p2 - p1)
            if dist < 3 or dist > max_dist:
                continue

            # ep1에서 ep2 방향
            dir_12 = (p2 - p1) / dist
            # ep2에서 ep1 방향
            dir_21 = -dir_12

            # ep1의 tangent이 ep2를 가리키는지
            cos1 = np.dot(t1, dir_12)
            # ep2의 tangent이 ep1을 가리키는지
            cos2 = np.dot(t2, dir_21)

            # 양쪽 모두 가리켜야 함
            angle_thresh = np.cos(np.radians(max_angle_deg))
            if cos1 < angle_thresh or cos2 < angle_thresh:
                continue

            # Score: 높을수록 좋음 (angle 일치 + 거리 가까움)
            score = (cos1 + cos2) / 2 - dist / max_dist * 0.3
            candidates.append((i, j, score, dist))

    # greedy matching: score 높은 순
    candidates.sort(key=lambda x: -x[2])
    for i, j, score, dist in candidates:
        if i not in used and j not in used:
            matches.append((i, j, score, dist))
            used.add(i)
            used.add(j)

    return matches

def bridge_endpoints(ep1, ep2, skel):
    """두 endpoint를 부드러운 곡선으로 연결"""
    p1 = np.array(ep1['pos'], dtype=float)
    p2 = np.array(ep2['pos'], dtype=float)
    t1 = ep1['tangent']
    t2 = ep2['tangent']
    dist = np.linalg.norm(p2 - p1)

    # cubic bezier-like: control points
    ctrl1 = p1 + t1 * dist * 0.4
    ctrl2 = p2 + t2 * dist * 0.4

    # interpolate
    n_pts = max(int(dist * 1.5), 10)
    bridge_pts = []
    for t in np.linspace(0, 1, n_pts):
        # cubic bezier
        p = (1-t)**3 * p1 + 3*(1-t)**2*t * ctrl1 + 3*(1-t)*t**2 * ctrl2 + t**3 * p2
        y, x = int(round(p[0])), int(round(p[1]))
        if 0 <= y < skel.shape[0] and 0 <= x < skel.shape[1]:
            bridge_pts.append((y, x))

    return bridge_pts

# ===== R skeleton gap bridging =====
print("=== R skeleton ===")
r_eps = get_endpoints_with_tangent(r_skel, trace_len=15)
print(f"Endpoints: {len(r_eps)}")
r_matches = match_endpoints(r_eps, max_dist=50, max_angle_deg=40)
print(f"Matched pairs: {len(r_matches)}")

r_bridged = r_skel.copy()
r_bridges = []
for i, j, score, dist in r_matches:
    pts = bridge_endpoints(r_eps[i], r_eps[j], r_skel)
    r_bridges.append(pts)
    for (y, x) in pts:
        r_bridged[y, x] = True

print("=== B skeleton ===")
b_eps = get_endpoints_with_tangent(b_skel, trace_len=15)
print(f"Endpoints: {len(b_eps)}")
b_matches = match_endpoints(b_eps, max_dist=50, max_angle_deg=40)
print(f"Matched pairs: {len(b_matches)}")

b_bridged = b_skel.copy()
b_bridges = []
for i, j, score, dist in b_matches:
    pts = bridge_endpoints(b_eps[i], b_eps[j], b_skel)
    b_bridges.append(pts)
    for (y, x) in pts:
        b_bridged[y, x] = True

# ===== Metrics =====
def skel_features(skel):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    return np.argwhere(nb == 1), np.argwhere(nb >= 3)

r_ep_before, r_jn_before = skel_features(r_skel)
r_ep_after, r_jn_after = skel_features(r_bridged)
b_ep_before, b_jn_before = skel_features(b_skel)
b_ep_after, b_jn_after = skel_features(b_bridged)

print(f"\nR skel: EP {len(r_ep_before)}→{len(r_ep_after)}, JN {len(r_jn_before)}→{len(r_jn_after)}")
print(f"B skel: EP {len(b_ep_before)}→{len(b_ep_after)}, JN {len(b_jn_before)}→{len(b_jn_after)}")

# ===== Figure 1: Before/After 비교 =====
fig, axes = plt.subplots(2, 4, figsize=(36, 16))
fig.suptitle('Gap Bridging: Endpoint matching by tangent direction', fontsize=16, fontweight='bold')

# R before
r_vis_b = gt_raw.copy().astype(float) * 0.25
r_vis_b[cv2.dilate(r_skel.astype(np.uint8), dk4) > 0] = [255, 80, 80]
# endpoints
for ep in r_eps:
    y, x = ep['pos']
    cv2.circle(r_vis_b, (x, y), 5, (255, 255, 0), -1)
    # tangent arrow
    ty, tx = ep['tangent']
    cv2.arrowedLine(r_vis_b, (x, y), (int(x+tx*20), int(y+ty*20)), (255, 255, 0), 1, tipLength=0.3)
axes[0,0].imshow(r_vis_b.astype(np.uint8))
axes[0,0].set_title(f'R skel BEFORE: EP={len(r_ep_before)}\nyellow=endpoints+tangent', fontsize=11)

# R after
r_vis_a = gt_raw.copy().astype(float) * 0.25
r_vis_a[cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0] = [255, 80, 80]
# bridges highlighted
for pts in r_bridges:
    for (y, x) in pts:
        cv2.circle(r_vis_a, (x, y), 3, (255, 255, 0), -1)
axes[0,1].imshow(r_vis_a.astype(np.uint8))
axes[0,1].set_title(f'R skel AFTER: EP={len(r_ep_after)}\nyellow=bridged segments', fontsize=11)

# B before
b_vis_b = gt_raw.copy().astype(float) * 0.25
b_vis_b[cv2.dilate(b_skel.astype(np.uint8), dk4) > 0] = [80, 150, 255]
for ep in b_eps:
    y, x = ep['pos']
    cv2.circle(b_vis_b, (x, y), 5, (0, 255, 255), -1)
    ty, tx = ep['tangent']
    cv2.arrowedLine(b_vis_b, (x, y), (int(x+tx*20), int(y+ty*20)), (0, 255, 255), 1, tipLength=0.3)
axes[0,2].imshow(b_vis_b.astype(np.uint8))
axes[0,2].set_title(f'B skel BEFORE: EP={len(b_ep_before)}\ncyan=endpoints+tangent', fontsize=11)

# B after
b_vis_a = gt_raw.copy().astype(float) * 0.25
b_vis_a[cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0] = [80, 150, 255]
for pts in b_bridges:
    for (y, x) in pts:
        cv2.circle(b_vis_a, (x, y), 3, (0, 255, 255), -1)
axes[0,3].imshow(b_vis_a.astype(np.uint8))
axes[0,3].set_title(f'B skel AFTER: EP={len(b_ep_after)}\ncyan=bridged segments', fontsize=11)

# --- Row 2: Combined + zoom ---
# Combined before
comb_b = gt_raw.copy().astype(float) * 0.25
comb_b[cv2.dilate(r_skel.astype(np.uint8), dk4) > 0] = [255, 80, 80]
comb_b[cv2.dilate(b_skel.astype(np.uint8), dk4) > 0] = [80, 150, 255]
axes[1,0].imshow(comb_b.astype(np.uint8))
axes[1,0].set_title('Combined BEFORE', fontsize=12)

# Combined after
comb_a = gt_raw.copy().astype(float) * 0.25
comb_a[cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0] = [255, 80, 80]
comb_a[cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0] = [80, 150, 255]
# bridges
for pts in r_bridges:
    for (y, x) in pts:
        cv2.circle(comb_a, (x, y), 2, (255, 200, 50), -1)
for pts in b_bridges:
    for (y, x) in pts:
        cv2.circle(comb_a, (x, y), 2, (50, 255, 200), -1)
axes[1,1].imshow(comb_a.astype(np.uint8))
axes[1,1].set_title('Combined AFTER\nyellow=R bridge, cyan=B bridge', fontsize=11)

# Zoom into a crossing area
# Find area with most bridges
if r_bridges or b_bridges:
    all_bridge_pts = []
    for pts in r_bridges + b_bridges:
        all_bridge_pts.extend(pts)
    if all_bridge_pts:
        bp_arr = np.array(all_bridge_pts)
        # cluster center
        cy, cx = int(np.median(bp_arr[:,0])), int(np.median(bp_arr[:,1]))
    else:
        cy, cx = h//2, w//2
else:
    cy, cx = h//2, w//2

crop = 80
y0, y1 = max(0, cy-crop), min(h, cy+crop)
x0, x1 = max(0, cx-crop), min(w, cx+crop)

# Zoom before
zoom_b = gt_raw[y0:y1, x0:x1].copy().astype(float) * 0.3
zoom_b[cv2.dilate(r_skel[y0:y1, x0:x1].astype(np.uint8), dk4) > 0] = [255, 80, 80]
zoom_b[cv2.dilate(b_skel[y0:y1, x0:x1].astype(np.uint8), dk4) > 0] = [80, 150, 255]
# endpoints in zoom
for ep in r_eps:
    ey, ex = ep['pos']
    if y0 <= ey < y1 and x0 <= ex < x1:
        cv2.circle(zoom_b, (ex-x0, ey-y0), 5, (255, 255, 0), -1)
        ty, tx = ep['tangent']
        cv2.arrowedLine(zoom_b, (ex-x0, ey-y0),
                       (int(ex-x0+tx*25), int(ey-y0+ty*25)), (255, 255, 0), 2, tipLength=0.3)
for ep in b_eps:
    ey, ex = ep['pos']
    if y0 <= ey < y1 and x0 <= ex < x1:
        cv2.circle(zoom_b, (ex-x0, ey-y0), 5, (0, 255, 255), -1)
        ty, tx = ep['tangent']
        cv2.arrowedLine(zoom_b, (ex-x0, ey-y0),
                       (int(ex-x0+tx*25), int(ey-y0+ty*25)), (0, 255, 255), 2, tipLength=0.3)
axes[1,2].imshow(zoom_b.astype(np.uint8))
axes[1,2].set_title('Zoom BEFORE: endpoints+tangents', fontsize=11)

# Zoom after
zoom_a = gt_raw[y0:y1, x0:x1].copy().astype(float) * 0.3
zoom_a[cv2.dilate(r_bridged[y0:y1, x0:x1].astype(np.uint8), dk4) > 0] = [255, 80, 80]
zoom_a[cv2.dilate(b_bridged[y0:y1, x0:x1].astype(np.uint8), dk4) > 0] = [80, 150, 255]
for pts in r_bridges:
    for (py, px) in pts:
        if y0 <= py < y1 and x0 <= px < x1:
            cv2.circle(zoom_a, (px-x0, py-y0), 3, (255, 200, 50), -1)
for pts in b_bridges:
    for (py, px) in pts:
        if y0 <= py < y1 and x0 <= px < x1:
            cv2.circle(zoom_a, (px-x0, py-y0), 3, (50, 255, 200), -1)
axes[1,3].imshow(zoom_a.astype(np.uint8))
axes[1,3].set_title('Zoom AFTER: bridged', fontsize=11)

for ax in axes.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'skel_gap_bridge.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('\nSaved skel_gap_bridge.png')

# ===== Figure 2: Match 상세 — 각 bridge 개별 표시 =====
n_show = min(8, len(r_matches) + len(b_matches))
if n_show > 0:
    fig2, axes2 = plt.subplots(2, 4, figsize=(32, 14))
    fig2.suptitle('Individual bridge connections', fontsize=16, fontweight='bold')

    all_matches_info = []
    for i, j, score, dist in r_matches:
        all_matches_info.append(('R', r_eps[i], r_eps[j], score, dist, r_skel))
    for i, j, score, dist in b_matches:
        all_matches_info.append(('B', b_eps[i], b_eps[j], score, dist, b_skel))
    all_matches_info.sort(key=lambda x: -x[3])  # score순

    for idx in range(min(8, len(all_matches_info))):
        ax = axes2.flat[idx]
        color_name, ep1, ep2, score, dist, skel = all_matches_info[idx]

        p1 = np.array(ep1['pos'])
        p2 = np.array(ep2['pos'])
        center = ((p1 + p2) / 2).astype(int)
        crop = max(40, int(dist))
        y0c = max(0, center[0]-crop)
        y1c = min(h, center[0]+crop)
        x0c = max(0, center[1]-crop)
        x1c = min(w, center[1]+crop)

        zoom = gt_raw[y0c:y1c, x0c:x1c].copy().astype(float) * 0.3
        skel_crop = skel[y0c:y1c, x0c:x1c]
        c = [255, 80, 80] if color_name == 'R' else [80, 150, 255]
        zoom[cv2.dilate(skel_crop.astype(np.uint8), dk4) > 0] = c

        # endpoints
        for ep in [ep1, ep2]:
            ey, ex = ep['pos']
            if y0c <= ey < y1c and x0c <= ex < x1c:
                cv2.circle(zoom, (ex-x0c, ey-y0c), 5, (255, 255, 0), -1)
                ty, tx = ep['tangent']
                cv2.arrowedLine(zoom, (ex-x0c, ey-y0c),
                               (int(ex-x0c+tx*20), int(ey-y0c+ty*20)),
                               (255, 255, 0), 2, tipLength=0.3)

        # bridge
        bridge_pts = bridge_endpoints(ep1, ep2, skel)
        for (py, px) in bridge_pts:
            if y0c <= py < y1c and x0c <= px < x1c:
                cv2.circle(zoom, (px-x0c, py-y0c), 2, (0, 255, 0), -1)

        ax.imshow(zoom.astype(np.uint8))
        ax.set_title(f'{color_name} bridge #{idx+1}\nscore={score:.2f} dist={dist:.0f}px', fontsize=10)
        ax.axis('off')

    for idx in range(len(all_matches_info), 8):
        axes2.flat[idx].axis('off')

    plt.tight_layout()
    plt.savefig(OUT / 'skel_gap_bridge_detail.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved skel_gap_bridge_detail.png')

print('Done!')
