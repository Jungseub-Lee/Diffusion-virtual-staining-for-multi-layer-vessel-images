"""
Create 3-stage zoom panels for (n): Before → Bridging → After
- Before: broken R or B skel (no bridge)
- Bridging: skeleton + bridge paths with R/B distinct colors
- After: clean connected skel, R+B with transparency for overlap
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output\paper_panels_depth')
DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')

s1 = np.array(Image.open(DATA / 'Original color stack data' / 'Sample 1_BF and GT(512x512 two images).png'))
gt_raw = s1[512:]
gt_blur = cv2.GaussianBlur(gt_raw, (7, 7), 0)
h, w = gt_raw.shape[:2]

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

R = gt_blur[:,:,0].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
diff = B - R

# Vessel mask
vessel_all = remove_small_objects(
    cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=100)
vessel_dilated = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

# R skel
r_mask = remove_small_objects(
    cv2.morphologyEx((R > 30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=50)
r_skel = skeletonize(r_mask)
r_filtered = r_skel & (diff < 0)

# B skel (hysteresis)
b_seed = B > 50; b_low = B > 20
b_ll, nl = ndimage.label(b_low)
b_hyst = np.zeros_like(b_low, dtype=bool)
for i in range(1, nl+1):
    reg = b_ll == i
    if np.any(b_seed[reg]): b_hyst |= reg
b_hyst = remove_small_objects(
    cv2.morphologyEx(b_hyst.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0, min_size=50)
b_skel = skeletonize(b_hyst)
b_filtered = b_skel & (diff > 0)

# Bridging function (returns bridge info)
def get_endpoints_with_tangent(skel, trace_len=20):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    endpoints = np.argwhere(nb == 1)
    results = []
    for ep in endpoints:
        y0, x0 = ep
        path = [(y0, x0)]; visited = {(y0, x0)}; cy, cx = y0, x0
        for _ in range(trace_len):
            found = False
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    if dy==0 and dx==0: continue
                    ny, nx = cy+dy, cx+dx
                    if 0<=ny<h and 0<=nx<w and skel[ny,nx] and (ny,nx) not in visited:
                        path.append((ny,nx)); visited.add((ny,nx)); cy,cx = ny,nx; found = True; break
                if found: break
            if not found: break
        if len(path) >= 5:
            n = min(len(path), 10)
            t = np.array(path[0], dtype=float) - np.array(path[n-1], dtype=float)
            norm = np.linalg.norm(t)
            if norm > 0: t /= norm
            results.append({'pos': (y0,x0), 'tangent': t, 'path': path})
    return results

def bezier_bridge(ep1, ep2):
    p1 = np.array(ep1['pos'], dtype=float); p2 = np.array(ep2['pos'], dtype=float)
    t1, t2 = ep1['tangent'], ep2['tangent']
    d = np.linalg.norm(p2-p1)
    c1 = p1 + t1*d*0.4; c2 = p2 + t2*d*0.4
    n = max(int(d*1.5), 10)
    pts = []
    for t in np.linspace(0,1,n):
        p = (1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2
        pts.append((int(round(p[0])), int(round(p[1]))))
    return pts

def do_bridging_with_info(skel, vessel_mask, max_dist=80, max_angle=60):
    eps = get_endpoints_with_tangent(skel, 20)
    cos_thresh = np.cos(np.radians(max_angle))
    candidates = []
    for i in range(len(eps)):
        for j in range(i+1, len(eps)):
            p1 = np.array(eps[i]['pos'], dtype=float); p2 = np.array(eps[j]['pos'], dtype=float)
            d = np.linalg.norm(p2-p1)
            if d < 5 or d > max_dist: continue
            dir12 = (p2-p1)/d
            c1 = np.dot(eps[i]['tangent'], dir12); c2 = np.dot(eps[j]['tangent'], -dir12)
            if c1 < cos_thresh or c2 < cos_thresh: continue
            pts = bezier_bridge(eps[i], eps[j])
            on_v = sum(1 for (y,x) in pts if 0<=y<h and 0<=x<w and vessel_mask[y,x])
            vr = on_v/max(len(pts),1)
            if vr < 0.7: continue
            sc = (c1+c2)/2 - d/max_dist*0.2 + vr*0.2
            candidates.append({'i':i,'j':j,'score':sc,'ep1':eps[i],'ep2':eps[j]})
    candidates.sort(key=lambda x: -x['score'])
    used = set(); matches = []
    for c in candidates:
        if c['i'] not in used and c['j'] not in used:
            matches.append(c); used.add(c['i']); used.add(c['j'])
    bridged = skel.copy(); bridge_paths = []
    for m in matches:
        pts = bezier_bridge(m['ep1'], m['ep2'])
        bridge_paths.append({'pts': pts, 'ep1': m['ep1'], 'ep2': m['ep2']})
        for (y,x) in pts:
            if 0<=y<h and 0<=x<w: bridged[y,x] = True
    bridged = skeletonize(bridged)
    return bridged, bridge_paths

r_bridged, r_bridges = do_bridging_with_info(r_filtered, vessel_dilated)
b_bridged, b_bridges = do_bridging_with_info(b_filtered, vessel_dilated)

# Find best bridge crops (longest bridges)
all_bridges = []
for bi in r_bridges:
    pts = bi['pts']
    ys = [p[0] for p in pts]; xs = [p[1] for p in pts]
    all_bridges.append({'ch':'R', 'pts':pts, 'ep1':bi['ep1'], 'ep2':bi['ep2'],
                       'cy':np.mean(ys), 'cx':np.mean(xs), 'length':len(pts)})
for bi in b_bridges:
    pts = bi['pts']
    ys = [p[0] for p in pts]; xs = [p[1] for p in pts]
    all_bridges.append({'ch':'B', 'pts':pts, 'ep1':bi['ep1'], 'ep2':bi['ep2'],
                       'cy':np.mean(ys), 'cx':np.mean(xs), 'length':len(pts)})

# Sort by length (longer bridges are more illustrative)
all_bridges.sort(key=lambda x: -x['length'])

# Take top 3 R and top 3 B
r_top = [b for b in all_bridges if b['ch'] == 'R'][:3]
b_top = [b for b in all_bridges if b['ch'] == 'B'][:3]
selected = r_top + b_top

BG_ALPHA = 0.5  # brighter background
crop_size = 80

for idx, br in enumerate(selected):
    cy, cx = int(br['cy']), int(br['cx'])
    y0 = max(0, cy - crop_size); y1 = min(h, cy + crop_size)
    x0 = max(0, cx - crop_size); x1 = min(w, cx + crop_size)
    ch = br['ch']

    # --- BEFORE: broken skeleton, no bridge ---
    before = gt_raw[y0:y1, x0:x1].copy().astype(float) * BG_ALPHA
    if ch == 'R':
        sk_local = r_filtered[y0:y1, x0:x1]
        skel_color = [255, 80, 80]
    else:
        sk_local = b_filtered[y0:y1, x0:x1]
        skel_color = [80, 150, 255]
    sk_dil = cv2.dilate(sk_local.astype(np.uint8), dk4) > 0
    before[sk_dil] = skel_color
    Image.fromarray(before.astype(np.uint8)).save(OUT / f'n3_zoom_{idx+1}_before.png')

    # --- BRIDGING: skeleton + bridge path with distinct color ---
    bridging = gt_raw[y0:y1, x0:x1].copy().astype(float) * BG_ALPHA
    bridging[sk_dil] = skel_color

    # Draw bridge path in distinct color
    if ch == 'R':
        bridge_color = [255, 180, 50]  # orange for R bridges
    else:
        bridge_color = [50, 255, 180]  # teal for B bridges

    for (py, px) in br['pts']:
        ly, lx = py - y0, px - x0
        if 0 <= ly < (y1-y0) and 0 <= lx < (x1-x0):
            # Draw thicker bridge
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = ly+dy, lx+dx
                    if 0 <= ny < (y1-y0) and 0 <= nx < (x1-x0):
                        bridging[ny, nx] = bridge_color

    # Draw tangent arrows
    ep1 = br['ep1']; ep2 = br['ep2']
    for ep, arrow_len in [(ep1, 12), (ep2, 12)]:
        ey, ex = ep['pos'][0] - y0, ep['pos'][1] - x0
        ty, tx = ep['tangent']
        ay, ax = int(ey + ty*arrow_len), int(ex + tx*arrow_len)
        if 0 <= ey < (y1-y0) and 0 <= ex < (x1-x0):
            cv2.arrowedLine(bridging, (ex, ey), (ax, ay), bridge_color, 2, tipLength=0.3)

    Image.fromarray(bridging.astype(np.uint8)).save(OUT / f'n3_zoom_{idx+1}_bridging.png')

    # --- AFTER: clean connected skeleton with transparency overlap ---
    after = gt_raw[y0:y1, x0:x1].copy().astype(float) * BG_ALPHA
    # Draw R skeleton semi-transparent
    r_local = cv2.dilate(r_bridged[y0:y1, x0:x1].astype(np.uint8), dk4) > 0
    b_local = cv2.dilate(b_bridged[y0:y1, x0:x1].astype(np.uint8), dk4) > 0

    # R layer (semi-transparent)
    r_only = r_local & ~b_local
    both = r_local & b_local
    b_only = b_local & ~r_local

    after[r_only] = [255, 80, 80]
    after[b_only] = [80, 150, 255]
    # Overlap: blend R and B
    after[both] = np.array(after[both]) * 0.0 + np.array([180, 100, 220])  # purple blend

    Image.fromarray(after.astype(np.uint8)).save(OUT / f'n3_zoom_{idx+1}_after.png')

    print(f'  Saved zoom {idx+1} ({ch}, len={br["length"]}): '
          f'crop ({y0},{x0})-({y1},{x1})')

# Also update the main m_ panels: before/bridging/after full images
# BEFORE: R+B filtered skeletons (no bridges)
m_before = gt_raw.copy().astype(float) * BG_ALPHA
m_before[cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0] = [255, 80, 80]
m_before[cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0] = [80, 150, 255]
Image.fromarray(m_before.astype(np.uint8)).save(OUT / 'm_before.png')

# BRIDGING: filtered skeletons + bridge paths in R-orange / B-teal
m_bridging = gt_raw.copy().astype(float) * BG_ALPHA
m_bridging[cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0] = [255, 80, 80]
m_bridging[cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0] = [80, 150, 255]

# Draw R bridges in orange
for bi in r_bridges:
    for (py, px) in bi['pts']:
        if 0<=py<h and 0<=px<w:
            for dy in range(-1,2):
                for dx in range(-1,2):
                    ny, nx = py+dy, px+dx
                    if 0<=ny<h and 0<=nx<w:
                        m_bridging[ny,nx] = [255, 180, 50]
    # Tangent arrows
    for ep in [bi['ep1'], bi['ep2']]:
        ey, ex = ep['pos']
        ty, tx = ep['tangent']
        ay, ax = int(ey+ty*15), int(ex+tx*15)
        cv2.arrowedLine(m_bridging, (ex,ey), (ax,ay), (255,180,50), 2, tipLength=0.3)

# Draw B bridges in teal
for bi in b_bridges:
    for (py, px) in bi['pts']:
        if 0<=py<h and 0<=px<w:
            for dy in range(-1,2):
                for dx in range(-1,2):
                    ny, nx = py+dy, px+dx
                    if 0<=ny<h and 0<=nx<w:
                        m_bridging[ny,nx] = [50, 255, 180]
    for ep in [bi['ep1'], bi['ep2']]:
        ey, ex = ep['pos']
        ty, tx = ep['tangent']
        ay, ax = int(ey+ty*15), int(ex+tx*15)
        cv2.arrowedLine(m_bridging, (ex,ey), (ax,ay), (50,255,180), 2, tipLength=0.3)

Image.fromarray(m_bridging.astype(np.uint8)).save(OUT / 'm_bridging.png')

# AFTER: clean bridged skeletons with transparency
m_after = gt_raw.copy().astype(float) * BG_ALPHA
r_dil = cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0
b_dil = cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0
r_only = r_dil & ~b_dil
b_only = b_dil & ~r_dil
both = r_dil & b_dil
m_after[r_only] = [255, 80, 80]
m_after[b_only] = [80, 150, 255]
m_after[both] = [180, 100, 220]  # purple for overlap
Image.fromarray(m_after.astype(np.uint8)).save(OUT / 'm_after.png')

print(f'\nR bridges: {len(r_bridges)}, B bridges: {len(b_bridges)}')
print(f'Selected {len(selected)} zoom regions')
print('Done!')
