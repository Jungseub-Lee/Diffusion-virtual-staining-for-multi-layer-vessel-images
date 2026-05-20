"""
Create 3-stage zoom panels for (n): Before → Bridging → After
- R+B 동시 표현, 겹침은 투명 overlap
- Bridging에서 crop 내 모든 bridge를 표현
"""
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

vessel_all = remove_small_objects(
    cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=100)
vessel_dilated = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

r_mask = remove_small_objects(
    cv2.morphologyEx((R > 30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=50)
r_skel = skeletonize(r_mask)
r_filtered = r_skel & (diff < 0)

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

BG_ALPHA = 0.5
R_COLOR = np.array([255, 80, 80], dtype=float)
B_COLOR = np.array([80, 150, 255], dtype=float)
R_BRIDGE_COLOR = [255, 180, 50]   # orange
B_BRIDGE_COLOR = [50, 255, 180]   # teal
OVERLAP_ALPHA = 0.5  # blend ratio for overlap

def draw_rb_overlap(bg, r_dil, b_dil):
    """Draw R and B skeletons with semi-transparent overlap."""
    vis = bg.copy()
    r_only = r_dil & ~b_dil
    b_only = b_dil & ~r_dil
    both = r_dil & b_dil
    vis[r_only] = R_COLOR
    vis[b_only] = B_COLOR
    # Overlap: blend R and B colors
    vis[both] = (R_COLOR * OVERLAP_ALPHA + B_COLOR * (1 - OVERLAP_ALPHA)).astype(np.uint8)
    return vis

def bridges_in_crop(bridges, y0, y1, x0, x1, margin=10):
    """Find all bridges that have at least one endpoint within the crop region."""
    result = []
    for bi in bridges:
        for ep in [bi['ep1'], bi['ep2']]:
            ey, ex = ep['pos']
            if y0-margin <= ey < y1+margin and x0-margin <= ex < x1+margin:
                result.append(bi)
                break
    return result

def draw_bridges_on_vis(vis, bridges, y0, x0, ch_color):
    """Draw all bridge paths and tangent arrows on a cropped visualization."""
    lh, lw = vis.shape[:2]
    for bi in bridges:
        # Draw bridge path
        for (py, px) in bi['pts']:
            ly, lx = py - y0, px - x0
            if 0 <= ly < lh and 0 <= lx < lw:
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = ly+dy, lx+dx
                        if 0 <= ny < lh and 0 <= nx < lw:
                            vis[ny, nx] = ch_color
        # Draw tangent arrows
        for ep in [bi['ep1'], bi['ep2']]:
            ey, ex = ep['pos'][0] - y0, ep['pos'][1] - x0
            ty, tx = ep['tangent']
            ay, ax = int(ey + ty*12), int(ex + tx*12)
            if 0 <= ey < lh and 0 <= ex < lw:
                cv2.arrowedLine(vis, (ex, ey), (ax, ay), ch_color, 2, tipLength=0.3)
    return vis

# Collect ALL bridges with center info
all_bridges = []
for bi in r_bridges:
    pts = bi['pts']
    ys = [p[0] for p in pts]; xs = [p[1] for p in pts]
    all_bridges.append({'ch':'R', 'bi':bi, 'cy':np.mean(ys), 'cx':np.mean(xs), 'length':len(pts)})
for bi in b_bridges:
    pts = bi['pts']
    ys = [p[0] for p in pts]; xs = [p[1] for p in pts]
    all_bridges.append({'ch':'B', 'bi':bi, 'cy':np.mean(ys), 'cx':np.mean(xs), 'length':len(pts)})

# Sort by length, pick top 3 diverse regions
all_bridges.sort(key=lambda x: -x['length'])

# Pick 3 well-separated crop regions
crop_size = 80
selected_crops = []
for br in all_bridges:
    cy, cx = int(br['cy']), int(br['cx'])
    # Check not too close to existing selections
    too_close = False
    for sc in selected_crops:
        if abs(cy - sc['cy']) < crop_size and abs(cx - sc['cx']) < crop_size:
            too_close = True
            break
    if not too_close:
        selected_crops.append({'cy': cy, 'cx': cx})
        if len(selected_crops) >= 3:
            break

# Generate 3-stage crops
for idx, sc in enumerate(selected_crops):
    cy, cx = sc['cy'], sc['cx']
    y0 = max(0, cy - crop_size); y1 = min(h, cy + crop_size)
    x0 = max(0, cx - crop_size); x1 = min(w, cx + crop_size)

    bg = gt_raw[y0:y1, x0:x1].copy().astype(float) * BG_ALPHA

    # Local skeleton masks
    r_filt_local = cv2.dilate(r_filtered[y0:y1, x0:x1].astype(np.uint8), dk4) > 0
    b_filt_local = cv2.dilate(b_filtered[y0:y1, x0:x1].astype(np.uint8), dk4) > 0
    r_br_local = cv2.dilate(r_bridged[y0:y1, x0:x1].astype(np.uint8), dk4) > 0
    b_br_local = cv2.dilate(b_bridged[y0:y1, x0:x1].astype(np.uint8), dk4) > 0

    # --- BEFORE: R+B filtered skeletons (no bridges), with overlap ---
    before = draw_rb_overlap(bg.copy(), r_filt_local, b_filt_local)
    Image.fromarray(before.astype(np.uint8)).save(OUT / f'n3_zoom_{idx+1}_before.png')

    # --- BRIDGING: R+B filtered + ALL bridges in this crop ---
    bridging = draw_rb_overlap(bg.copy(), r_filt_local, b_filt_local)
    # Find and draw all R bridges in crop
    r_local_bridges = bridges_in_crop(r_bridges, y0, y1, x0, x1)
    b_local_bridges = bridges_in_crop(b_bridges, y0, y1, x0, x1)
    bridging = draw_bridges_on_vis(bridging, r_local_bridges, y0, x0, R_BRIDGE_COLOR)
    bridging = draw_bridges_on_vis(bridging, b_local_bridges, y0, x0, B_BRIDGE_COLOR)
    Image.fromarray(bridging.astype(np.uint8)).save(OUT / f'n3_zoom_{idx+1}_bridging.png')

    # --- AFTER: clean bridged R+B with overlap ---
    after = draw_rb_overlap(bg.copy(), r_br_local, b_br_local)
    Image.fromarray(after.astype(np.uint8)).save(OUT / f'n3_zoom_{idx+1}_after.png')

    n_rb = len(r_local_bridges)
    n_bb = len(b_local_bridges)
    print(f'  Zoom {idx+1}: crop ({y0},{x0})-({y1},{x1})  '
          f'R bridges={n_rb}, B bridges={n_bb}')

print(f'\nTotal R bridges: {len(r_bridges)}, B bridges: {len(b_bridges)}')
print('Done!')
