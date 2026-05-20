"""
Generate improved bridging zoom panels + algorithm diagram for paper figure.
Uses same bridge logic as skel_dominant_bridge_v3.py with arrows + Bezier paths.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')
DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')
pd_dir = OUT / 'paper_panels_depth'

s1 = np.array(Image.open(DATA / 'Original color stack data' / 'Sample 1_BF and GT(512x512 two images).png'))
gt_raw = s1[512:]
gt_blur = cv2.GaussianBlur(gt_raw, (7, 7), 0)
h, w = gt_raw.shape[:2]
by = int(h * 0.8)

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dk3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

R = gt_blur[:,:,0].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
diff_BR = B - R

# R skeleton
r_mask = cv2.morphologyEx((R > 30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
r_mask = remove_small_objects(r_mask, min_size=50)
r_skel = skeletonize(r_mask)

# B skeleton (hysteresis)
b_seed = B > 50; b_low = B > 20
b_lab, b_n = ndimage.label(b_low)
b_hyst = np.zeros_like(b_low, dtype=bool)
for i in range(1, b_n + 1):
    if np.any(b_seed[b_lab == i]): b_hyst |= (b_lab == i)
b_hyst = cv2.morphologyEx(b_hyst.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0
b_hyst = remove_small_objects(b_hyst, min_size=50)
b_skel = skeletonize(b_hyst)

# Dominance filter
r_filtered = r_skel & (diff_BR < 0)
b_filtered = b_skel & (diff_BR > 0)

# Vessel mask
vessel_all = remove_small_objects(
    cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0,
    min_size=100)
vessel_mask = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

# ===== Bridge functions (same as v3) =====
def get_endpoints_with_tangent(skel, trace_len=20):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    endpoints = np.argwhere(nb == 1)
    results = []
    for ep in endpoints:
        y0, x0 = ep
        path = [(y0, x0)]; visited = {(y0, x0)}
        cy, cx = y0, x0
        for _ in range(trace_len):
            found = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0: continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and (ny, nx) not in visited:
                        path.append((ny, nx)); visited.add((ny, nx))
                        cy, cx = ny, nx; found = True; break
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
    used = set(); matches = []
    for i, j, sc, d, vr in candidates:
        if i not in used and j not in used:
            matches.append((i, j, sc, d, vr)); used.add(i); used.add(j)
    bridged = skel.copy(); bridges = []
    for i, j, sc, d, vr in matches:
        pts = _bridge_pts(eps[i], eps[j])
        bridges.append((pts, sc, d, vr, eps[i], eps[j]))
        for (y, x) in pts:
            if 0 <= y < h and 0 <= x < w: bridged[y, x] = True
    return bridged, bridges, eps

# Run bridging
r_bridged, r_bridges, r_eps = match_and_bridge(r_filtered, vessel_mask)
b_bridged, b_bridges, b_eps = match_and_bridge(b_filtered, vessel_mask)
print(f"R bridges: {len(r_bridges)}, B bridges: {len(b_bridges)}")

# ===== Generate zoom panels with arrows =====
all_bridges_info = [(br, 'R') for br in r_bridges] + [(br, 'B') for br in b_bridges]
# Filter to analysis zone
valid = [(br, c) for br, c in all_bridges_info
         if (br[4]['pos'][0] + br[5]['pos'][0]) / 2 < by]
# Sort by score (best first)
valid.sort(key=lambda x: -x[0][1])
print(f"Valid bridges: {len(valid)}")

zoom_pad = 55
n_zoom = min(6, len(valid))

for idx in range(n_zoom):
    (pts, sc, dist, vr, ep1, ep2), color = valid[idx]
    cy = int((ep1['pos'][0] + ep2['pos'][0]) / 2)
    cx = int((ep1['pos'][1] + ep2['pos'][1]) / 2)
    y0, y1 = max(0, cy - zoom_pad), min(by, cy + zoom_pad)
    x0, x1 = max(0, cx - zoom_pad), min(w, cx + zoom_pad)

    # Build visualization
    vis = gt_raw[y0:y1, x0:x1].copy().astype(float) * 0.25

    if color == 'R':
        skel_color = [255, 80, 80]
        ep_color = (255, 255, 0)  # yellow
        vis[cv2.dilate(r_filtered[y0:y1, x0:x1].astype(np.uint8), dk4) > 0] = skel_color
    else:
        skel_color = [80, 150, 255]
        ep_color = (0, 255, 255)  # cyan
        vis[cv2.dilate(b_filtered[y0:y1, x0:x1].astype(np.uint8), dk4) > 0] = skel_color

    vis = vis.astype(np.uint8)

    # Draw bridge path (green)
    for (py, px) in pts:
        ly, lx = py - y0, px - x0
        if 0 <= ly < vis.shape[0] and 0 <= lx < vis.shape[1]:
            cv2.circle(vis, (lx, ly), 2, (0, 255, 0), -1)

    # Draw endpoints with tangent arrows
    arrow_len = 22
    for ep, c in [(ep1, ep_color), (ep2, ep_color)]:
        ey, ex = ep['pos']
        if y0 <= ey < y1 and x0 <= ex < x1:
            ly, lx = ey - y0, ex - x0
            cv2.circle(vis, (lx, ly), 5, c, -1)
            ty, tx = ep['tangent']
            ax_end = (int(lx + tx * arrow_len), int(ly + ty * arrow_len))
            cv2.arrowedLine(vis, (lx, ly), ax_end, c, 2, tipLength=0.3)

    # Add score/dist text
    cv2.putText(vis, f's={sc:.2f} d={dist:.0f}', (3, vis.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(vis, f'{color} bridge', (3, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 100, 100) if color == 'R' else (100, 160, 255), 1)

    fig_i, ax_i = plt.subplots(figsize=(3, 3))
    ax_i.imshow(vis); ax_i.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(pd_dir / f'zoom_bridge_{idx+1}.png', dpi=300, bbox_inches='tight',
                pad_inches=0.02, facecolor='black')
    plt.close()

print(f"Saved {n_zoom} zoom bridge panels")

# ===== Full overview: before/after bridging =====
fig_ov, axes_ov = plt.subplots(1, 3, figsize=(18, 5))

# Before bridging
vis1 = gt_raw[:by].copy().astype(float) * 0.2
vis1[cv2.dilate(r_filtered[:by].astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis1[cv2.dilate(b_filtered[:by].astype(np.uint8), dk4) > 0] = [80, 150, 255]
# Mark all endpoints
for ep in r_eps:
    y, x = ep['pos']
    if y < by:
        cv2.circle(vis1.astype(np.uint8), (x, y), 3, (255, 255, 0), -1)
        ty, tx = ep['tangent']
        cv2.arrowedLine(vis1.astype(np.uint8), (x, y),
                       (int(x+tx*12), int(y+ty*12)), (255, 255, 0), 1, tipLength=0.3)
vis1 = vis1.astype(np.uint8)
for ep in r_eps:
    y, x = ep['pos']
    if y < by:
        cv2.circle(vis1, (x, y), 3, (255, 255, 0), -1)
        ty, tx = ep['tangent']
        cv2.arrowedLine(vis1, (x, y), (int(x+tx*12), int(y+ty*12)), (255, 255, 0), 1, tipLength=0.3)
for ep in b_eps:
    y, x = ep['pos']
    if y < by:
        cv2.circle(vis1, (x, y), 3, (0, 255, 255), -1)
        ty, tx = ep['tangent']
        cv2.arrowedLine(vis1, (x, y), (int(x+tx*12), int(y+ty*12)), (0, 255, 255), 1, tipLength=0.3)
axes_ov[0].imshow(vis1)
axes_ov[0].set_title('Before Bridging\n(endpoints + tangent arrows)', fontsize=12, fontweight='bold')

# After bridging
vis2 = gt_raw[:by].copy().astype(float) * 0.2
vis2[cv2.dilate(r_bridged[:by].astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis2[cv2.dilate(b_bridged[:by].astype(np.uint8), dk4) > 0] = [80, 150, 255]
vis2 = vis2.astype(np.uint8)
for pts, sc, d, vr, ep1, ep2 in r_bridges:
    for (y, x) in pts:
        if 0 <= y < by and 0 <= x < w:
            cv2.circle(vis2, (x, y), 2, (255, 200, 50), -1)
for pts, sc, d, vr, ep1, ep2 in b_bridges:
    for (y, x) in pts:
        if 0 <= y < by and 0 <= x < w:
            cv2.circle(vis2, (x, y), 2, (50, 255, 200), -1)
axes_ov[1].imshow(vis2)
axes_ov[1].set_title(f'After Bridging\nR: +{len(r_bridges)} bridges, B: +{len(b_bridges)} bridges',
                     fontsize=12, fontweight='bold')

# Combined final
vis3 = gt_raw[:by].copy().astype(float) * 0.2
vis3[cv2.dilate(r_bridged[:by].astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis3[cv2.dilate(b_bridged[:by].astype(np.uint8), dk4) > 0] = [80, 150, 255]
vis3 = vis3.astype(np.uint8)
axes_ov[2].imshow(vis3)
axes_ov[2].set_title('Final R+B Skeleton', fontsize=12, fontweight='bold')

for ax in axes_ov: ax.axis('off')
plt.tight_layout()
plt.savefig(pd_dir / 'bridge_overview.png', dpi=300, bbox_inches='tight', facecolor='black')
plt.close()
print("Saved bridge_overview.png")

# ===== Algorithm diagram =====
fig_alg, ax_alg = plt.subplots(figsize=(8, 5))
ax_alg.set_xlim(0, 10); ax_alg.set_ylim(0, 6)
ax_alg.set_aspect('equal')
ax_alg.axis('off')
ax_alg.set_facecolor('#1a1a2e')
fig_alg.patch.set_facecolor('#1a1a2e')

# Title
ax_alg.text(5, 5.6, 'Tangent-guided Gap Bridging Algorithm', ha='center',
            fontsize=14, fontweight='bold', color='white')

# Step boxes
steps = [
    (0.3, 4.2, 'Step 1', 'Endpoint Detection',
     'Find skeleton endpoints\n(neighbor count = 1)', '#e74c3c'),
    (3.6, 4.2, 'Step 2', 'Tangent Estimation',
     'Trace back 10-20px\nalong skeleton branch\n→ compute direction', '#f39c12'),
    (6.9, 4.2, 'Step 3', 'Candidate Matching',
     'For each EP pair:\n• distance < 80px\n• angular alignment > cos(60°)\n• vessel mask overlap > 70%', '#2ecc71'),
    (1.8, 1.2, 'Step 4', 'Bézier Interpolation',
     'P(t) = (1-t)³P₁ + 3(1-t)²tC₁\n        + 3(1-t)t²C₂ + t³P₂\nC₁ = P₁ + 0.4d·t₁\nC₂ = P₂ + 0.4d·t₂', '#3498db'),
    (5.8, 1.2, 'Step 5', 'Score & Select',
     'score = (cos₁+cos₂)/2\n         - d/d_max·0.2\n         + vessel_ratio·0.2\nGreedy: best-first, no reuse', '#9b59b6'),
]

for x, y, step, title, desc, color in steps:
    box_w, box_h = 2.8, 2.2
    rect = plt.Rectangle((x, y), box_w, box_h, fill=True,
                          facecolor=color + '22', edgecolor=color, linewidth=2, zorder=2)
    ax_alg.add_patch(rect)
    ax_alg.text(x + 0.1, y + box_h - 0.15, step, fontsize=7, color=color,
                fontweight='bold', va='top')
    ax_alg.text(x + 0.1, y + box_h - 0.4, title, fontsize=9, color='white',
                fontweight='bold', va='top')
    ax_alg.text(x + 0.15, y + box_h - 0.7, desc, fontsize=6.5, color='#cccccc',
                va='top', fontfamily='monospace')

# Arrows between steps
arrow_props = dict(arrowstyle='->', color='#888888', lw=1.5)
ax_alg.annotate('', xy=(3.6, 5.3), xytext=(3.1, 5.3), arrowprops=arrow_props)
ax_alg.annotate('', xy=(6.9, 5.3), xytext=(6.4, 5.3), arrowprops=arrow_props)
ax_alg.annotate('', xy=(5, 4.2), xytext=(5, 3.6), arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5))
ax_alg.annotate('', xy=(5.8, 2.3), xytext=(4.6, 2.3), arrowprops=arrow_props)

# Mini diagram: showing tangent + bridge concept
# Draw two skeleton stubs with gap
stub_y = 0.3
ax_alg.plot([0.5, 1.8], [stub_y, stub_y + 0.3], '-', color='#ff5555', lw=3, zorder=3)
ax_alg.plot([2.8, 4.0], [stub_y + 0.5, stub_y + 0.2], '-', color='#ff5555', lw=3, zorder=3)
# Tangent arrows
ax_alg.annotate('', xy=(2.2, stub_y + 0.38), xytext=(1.8, stub_y + 0.3),
                arrowprops=dict(arrowstyle='->', color='#ffff00', lw=2))
ax_alg.annotate('', xy=(2.4, stub_y + 0.42), xytext=(2.8, stub_y + 0.5),
                arrowprops=dict(arrowstyle='->', color='#ffff00', lw=2))
# Bridge curve (green dashed)
bx = np.linspace(1.8, 2.8, 20)
b_curve = stub_y + 0.3 + 0.15 * np.sin(np.linspace(0, np.pi, 20))
ax_alg.plot(bx, b_curve, '--', color='#00ff88', lw=2, zorder=3)
ax_alg.text(2.3, stub_y - 0.15, 'Bézier bridge', fontsize=7, color='#00ff88', ha='center')
ax_alg.text(1.7, stub_y + 0.5, 't₁', fontsize=8, color='#ffff00', fontweight='bold')
ax_alg.text(2.9, stub_y + 0.65, 't₂', fontsize=8, color='#ffff00', fontweight='bold')

plt.tight_layout(pad=0.5)
plt.savefig(pd_dir / 'bridge_algorithm.png', dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("Saved bridge_algorithm.png")

print("\nAll bridge panels done!")
