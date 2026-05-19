"""Detect connection points: R EP + B EP nearby + yellow midpoint."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage

DATA = Path(r'[1] Data')
OUT = Path(r'analysis_output/debug_pipeline')
OUT.mkdir(exist_ok=True)
multi_dir = DATA / 'Multi-color'
sids = ['1-19-716', '9-15-512', '16-18-716', '18-19-512', '1-16-512']
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
dk2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

def get_jn_ep(skel, h, by):
    roi = np.zeros_like(skel, dtype=bool); roi[:by] = True
    sc = skel & roi; su = sc.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    jm = nb >= 3; jlab, n_raw = ndimage.label(jm)
    jn_pts = []
    for i in range(1, n_raw+1):
        cluster = jlab == i; py, px = np.where(cluster)
        pad = 5
        ymin = max(0, py.min()-pad); ymax = min(h, py.max()+pad+1)
        xmin = max(0, px.min()-pad); xmax = min(skel.shape[1], px.max()+pad+1)
        local = sc[ymin:ymax, xmin:xmax].copy()
        cdil = cv2.dilate(cluster[ymin:ymax, xmin:xmax].astype(np.uint8), k3) > 0
        local[cdil] = False
        _, nbr = ndimage.label(local)
        if nbr >= 3:
            jn_pts.append((int(py.mean()), int(px.mean())))
    ep_pts = [(y, x) for y, x in np.argwhere((nb == 1) & roi) if y < by - 3]
    return jn_pts, ep_pts

def get_eps_tangent(skel, h, w, tl=20):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    eps = np.argwhere(nb == 1); res = []
    for ep in eps:
        y0, x0 = ep; path = [(y0,x0)]; vis = {(y0,x0)}; cy, cx = y0, x0
        for _ in range(tl):
            f = False
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    if dy==0 and dx==0: continue
                    ny, nx = cy+dy, cx+dx
                    if 0<=ny<h and 0<=nx<w and skel[ny,nx] and (ny,nx) not in vis:
                        path.append((ny,nx)); vis.add((ny,nx)); cy,cx=ny,nx; f=True; break
                if f: break
            if not f: break
        if len(path) >= 5:
            n2 = min(len(path), 10)
            t = np.array(path[0], dtype=float) - np.array(path[n2-1], dtype=float)
            nm = np.linalg.norm(t)
            if nm > 0: t /= nm
            res.append({'pos': (y0,x0), 'tangent': t})
    return res

def bezier(e1, e2):
    p1 = np.array(e1['pos'], dtype=float); p2 = np.array(e2['pos'], dtype=float)
    d = np.linalg.norm(p2-p1); t1, t2 = e1['tangent'], e2['tangent']
    c1 = p1 + t1*d*0.4; c2 = p2 + t2*d*0.4
    n = max(int(d*1.5), 10)
    pts = []
    for t in np.linspace(0, 1, n):
        p = (1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2
        pts.append((int(round(p[0])), int(round(p[1]))))
    return pts

def do_bridge(skel, vm, h, w):
    eps = get_eps_tangent(skel, h, w)
    ct = np.cos(np.radians(60)); cands = []
    for i in range(len(eps)):
        for j in range(i+1, len(eps)):
            p1 = np.array(eps[i]['pos'], dtype=float)
            p2 = np.array(eps[j]['pos'], dtype=float)
            d = np.linalg.norm(p2-p1)
            if d < 5 or d > 80: continue
            d12 = (p2-p1)/d
            c1 = np.dot(eps[i]['tangent'], d12)
            c2 = np.dot(eps[j]['tangent'], -d12)
            if c1 < ct or c2 < ct: continue
            pts = bezier(eps[i], eps[j])
            ov = sum(1 for (y,x) in pts if 0<=y<h and 0<=x<w and vm[y,x])
            vr = ov / max(len(pts), 1)
            if vr < 0.7: continue
            sc = (c1+c2)/2 - d/80*0.2 + vr*0.2
            cands.append((i, j, sc, eps[i], eps[j], pts))
    cands.sort(key=lambda x: -x[2])
    used = set(); bridges = []; bridged = skel.copy()
    for i, j, sc, e1, e2, pts in cands:
        if i not in used and j not in used:
            bridges.append({'ep1': e1, 'ep2': e2, 'pts': pts})
            for (y, x) in pts:
                if 0<=y<h and 0<=x<w: bridged[y, x] = True
            used.add(i); used.add(j)
    bridged = skeletonize(bridged)
    return bridged, bridges

def find_connection_points(r_ep, b_ep, yellow_mask, max_dist=15):
    """Find R-B EP pairs that are close and have yellow midpoint."""
    connections = []
    used_r = set(); used_b = set()
    # Build all pairs sorted by distance
    pairs = []
    for ri, (ry, rx) in enumerate(r_ep):
        for bi, (by2, bx) in enumerate(b_ep):
            d = np.sqrt((ry-by2)**2 + (rx-bx)**2)
            if d < max_dist:
                my, mx = int((ry+by2)/2), int((rx+bx)/2)
                pairs.append((d, ri, bi, ry, rx, by2, bx, my, mx))
    pairs.sort(key=lambda x: x[0])

    for d, ri, bi, ry, rx, by2, bx, my, mx in pairs:
        if ri in used_r or bi in used_b:
            continue
        # Check if midpoint is in yellow region
        h_img, w_img = yellow_mask.shape
        if 0 <= my < h_img and 0 <= mx < w_img:
            # Check yellow in a small neighborhood around midpoint
            pad = 3
            y0 = max(0, my-pad); y1 = min(h_img, my+pad+1)
            x0 = max(0, mx-pad); x1 = min(w_img, mx+pad+1)
            yellow_ratio = np.mean(yellow_mask[y0:y1, x0:x1])

            if yellow_ratio > 0.2:  # at least 20% yellow in neighborhood
                connections.append({
                    'r_ep': (ry, rx), 'b_ep': (by2, bx),
                    'mid': (my, mx), 'dist': d, 'yellow_ratio': yellow_ratio
                })
                used_r.add(ri); used_b.add(bi)
    return connections

BG_A = 0.6; WV = 0.35

def make_gt_bg(crop):
    v = crop.copy().astype(float) * BG_A
    return v * (1-WV) + 255*WV

for sid in sids:
    gt_path = None
    for m in ['LSGAN', 'Pix2pix', 'WGANGP']:
        p = multi_dir / m / f'{sid}_real_B.png'
        if p.exists():
            gt_path = p; break
    if gt_path is None:
        continue

    gt = np.array(Image.open(gt_path))
    gt_blur = cv2.GaussianBlur(gt, (7, 7), 0)
    h, w = gt.shape[:2]; by = int(h * 0.8)
    R = gt_blur[:,:,0].astype(np.float32)
    B = gt_blur[:,:,2].astype(np.float32)
    gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
    diff = B - R

    # Yellow mask from HSV
    hsv = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2HSV)
    hue = hsv[:,:,0].astype(float)
    sat = hsv[:,:,1].astype(float)
    yellow_mask = (hue >= 15) & (hue <= 50) & (sat > 40)

    # Build skels
    r_mask = remove_small_objects(
        cv2.morphologyEx((R > 30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=50)
    r_skel = skeletonize(r_mask); r_filt = r_skel & (diff < 0)

    b_seed = B > 50; b_low = B > 20
    b_lab, b_n = ndimage.label(b_low)
    b_mask_img = np.zeros_like(b_low, dtype=bool)
    for i in range(1, b_n+1):
        if np.any(b_seed[b_lab == i]):
            b_mask_img |= (b_lab == i)
    b_mask_img = remove_small_objects(
        cv2.morphologyEx(b_mask_img.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0, min_size=50)
    b_skel = skeletonize(b_mask_img); b_filt = b_skel & (diff > 0)

    vessel_all = remove_small_objects(
        cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=100)
    vm = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

    # Bridge
    r_bridged, _ = do_bridge(r_filt, vm, h, w)
    b_bridged, _ = do_bridge(b_filt, vm, h, w)

    # Get EP after bridging
    r_jn, r_ep = get_jn_ep(r_bridged, h, by)
    b_jn, b_ep = get_jn_ep(b_bridged, h, by)

    # Find connection points
    connections = find_connection_points(r_ep, b_ep, yellow_mask, max_dist=15)

    # Classify EP
    conn_r_set = {c['r_ep'] for c in connections}
    conn_b_set = {c['b_ep'] for c in connections}
    r_real = [p for p in r_ep if p not in conn_r_set]
    r_conn = [p for p in r_ep if p in conn_r_set]
    b_real = [p for p in b_ep if p not in conn_b_set]
    b_conn = [p for p in b_ep if p in conn_b_set]

    print(f'\n===== {sid} =====')
    print(f'R EP: {len(r_ep)} (real={len(r_real)}, conn={len(r_conn)})')
    print(f'B EP: {len(b_ep)} (real={len(b_real)}, conn={len(b_conn)})')
    print(f'Connection points: {len(connections)}')
    for c in connections:
        print(f'  R({c["r_ep"][0]},{c["r_ep"][1]}) <-> B({c["b_ep"][0]},{c["b_ep"][1]}) '
              f'dist={c["dist"]:.1f} yellow={c["yellow_ratio"]:.2f}')

    # === Figure: 1x3 ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{sid} — Connection Points (R-B EP pairs in Yellow zone)', fontsize=14, fontweight='bold')

    # Col 0: Yellow mask + EP positions
    vis = np.zeros((by, w, 3), dtype=np.uint8)
    vis[yellow_mask[:by]] = [180, 180, 50]
    # Show R and B skel thin
    vis[cv2.dilate(r_bridged[:by].astype(np.uint8), dk2) > 0] = [230, 70, 70]
    vis[cv2.dilate(b_bridged[:by].astype(np.uint8), dk2) > 0] = [70, 140, 240]
    for y, x in r_ep:
        cv2.circle(vis, (x, y), 4, (255, 255, 0), -1)
    for y, x in b_ep:
        cv2.circle(vis, (x, y), 4, (255, 255, 0), -1)
    # Draw connection lines
    for c in connections:
        ry, rx = c['r_ep']; by2, bx = c['b_ep']
        if ry < by and by2 < by:
            cv2.line(vis, (rx, ry), (bx, by2), (0, 255, 255), 2)
            cv2.circle(vis, (c['mid'][1], c['mid'][0]), 5, (0, 255, 255), -1)
    axes[0].imshow(vis)
    axes[0].set_title(f'Yellow zone + R/B skel + EP\nConnections: {len(connections)}', fontsize=11)

    # Col 1: R+B on GT with connection points
    vis = make_gt_bg(gt[:by])
    r_d = cv2.dilate(r_bridged[:by].astype(np.uint8), dk4) > 0
    b_d = cv2.dilate(b_bridged[:by].astype(np.uint8), dk4) > 0
    vis[r_d & ~b_d] = [230, 70, 70]
    vis[b_d & ~r_d] = [70, 140, 240]
    vis[r_d & b_d] = [180, 100, 220]
    vis = vis.astype(np.uint8)
    # JN
    for y, x in r_jn + b_jn:
        cv2.circle(vis, (x, y), 5, (255, 0, 255), 2)
    # Real EP
    for y, x in r_real + b_real:
        cv2.circle(vis, (x, y), 5, (255, 255, 0), 2)
    # Connection EP (cyan)
    for y, x in r_conn + b_conn:
        cv2.circle(vis, (x, y), 5, (0, 255, 255), 2)
    # Connection lines
    for c in connections:
        ry, rx = c['r_ep']; by2, bx = c['b_ep']
        if ry < by and by2 < by:
            cv2.line(vis, (rx, ry), (bx, by2), (0, 255, 255), 1)
    axes[1].imshow(vis)
    tj = len(r_jn) + len(b_jn)
    te_real = len(r_real) + len(b_real)
    axes[1].set_title(f'R+B on GT\nJN={tj} EP(real)={te_real} Conn={len(connections)}', fontsize=11)

    # Col 2: GT for reference
    axes[2].imshow(gt[:by])
    axes[2].set_title('GT (reference)', fontsize=11)

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT / f'connection_{sid}.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved connection_{sid}.png')

print('\nDone!')
