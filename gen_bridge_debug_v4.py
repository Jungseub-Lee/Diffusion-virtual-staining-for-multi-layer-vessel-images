"""Bridge debug v4: EP-EP with crossing check + EP-Line bridge + tri-junction."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage
from scipy.spatial.distance import cdist

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
        cdil_m = cv2.dilate(cluster[ymin:ymax, xmin:xmax].astype(np.uint8), k3) > 0
        local[cdil_m] = False
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
            res.append({'pos': (y0,x0), 'tangent': t, 'path': path})
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

def line_pts(p1, p2):
    d = max(abs(p2[0]-p1[0]), abs(p2[1]-p1[1]), 1)
    pts = []
    for i in range(d+1):
        t = i / d
        y = int(round(p1[0]*(1-t) + p2[0]*t))
        x = int(round(p1[1]*(1-t) + p2[1]*t))
        pts.append((y, x))
    return pts

def count_skel_crossings(pts, skel, h, w, exclude_endpoints=5):
    """Count how many times a bridge path crosses existing skeleton.
    Exclude first/last N pixels (they're near the endpoints themselves)."""
    crossings = 0
    was_on = False
    for idx, (y, x) in enumerate(pts):
        if idx < exclude_endpoints or idx > len(pts) - exclude_endpoints:
            continue
        if 0 <= y < h and 0 <= x < w and skel[y, x]:
            if not was_on:
                crossings += 1
                was_on = True
        else:
            was_on = False
    return crossings

def find_ep_to_line_bridges(skel, eps, vm, h, w, dom_mask, ch_mask,
                            search_radius=25, dom_thresh=0.3):
    """For each EP, find the nearest skel pixel within radius (not on own branch).
    Connect with Bezier curve: starts along EP tangent, arrives at target.
    Distance-first (no tangent constraint for target selection).
    Returns list of bridges with type='line'."""
    bridges = []
    connected_eps = set()

    # Precompute all skel pixel locations for fast search
    skel_ys, skel_xs = np.where(skel)
    if len(skel_ys) == 0:
        return bridges, connected_eps
    skel_coords = np.column_stack([skel_ys, skel_xs])

    for idx, ep in enumerate(eps):
        y0, x0 = ep['pos']
        ty, tx = ep['tangent']

        # Get EP's own branch pixels (first 8 pixels along its path) to exclude
        own_branch = set()
        for py, px in ep.get('path', [])[:8]:
            own_branch.add((py, px))

        # Find ALL candidate skel pixels within radius, sorted by distance
        candidates = []
        for sy, sx in skel_coords:
            if (sy, sx) in own_branch:
                continue
            d = np.sqrt((sy - y0)**2 + (sx - x0)**2)
            if d < 3 or d > search_radius:
                continue
            candidates.append((d, sy, sx))
        candidates.sort(key=lambda x: x[0])  # nearest first

        # Try each candidate until one passes all checks
        matched = False
        for cand_dist, tgt_y, tgt_x in candidates:
            # Build Bezier curve: EP tangent → target point
            p1 = np.array([y0, x0], dtype=float)
            p2 = np.array([tgt_y, tgt_x], dtype=float)
            d = np.linalg.norm(p2 - p1)
            c1 = p1 + np.array([ty, tx]) * d * 0.5
            c2 = p2
            n_pts = max(int(d * 1.5), 8)
            path = []
            for t in np.linspace(0, 1, n_pts):
                p = (1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2
                path.append((int(round(p[0])), int(round(p[1]))))

            # Check 1: vessel mask (70%)
            vm_count = sum(1 for (y, x) in path if 0<=y<h and 0<=x<w and vm[y, x])
            if vm_count / max(len(path), 1) < 0.7:
                continue

            # Check 2: channel mask (70%) — strict
            if ch_mask is not None:
                ch_count = sum(1 for (y, x) in path if 0<=y<h and 0<=x<w and ch_mask[y, x])
                if ch_count / max(len(path), 1) < 0.7:
                    continue

            # Check 3: no consecutive 3+ pixels off channel mask
            if ch_mask is not None:
                off_count = 0
                too_far = False
                for (py, px) in path:
                    if 0<=py<h and 0<=px<w and not ch_mask[py, px]:
                        off_count += 1
                        if off_count >= 3:
                            too_far = True; break
                    else:
                        off_count = 0
                if too_far:
                    continue

            # Check 4: dominance
            if dom_mask is not None:
                dom_count = sum(1 for (y, x) in path if 0<=y<h and 0<=x<w and dom_mask[y, x])
                if dom_count / max(len(path), 1) < dom_thresh:
                    continue

            # All checks passed — use this candidate
            bridges.append({
                'type': 'line',
                'ep_idx': idx,
                'ep': ep,
                'hit': (tgt_y, tgt_x),
                'pts': path,
                'length': len(path),
                'dist': cand_dist
            })
            connected_eps.add(idx)
            matched = True
            break  # take the nearest valid one

    return bridges, connected_eps

def do_bridge_v4(skel, vm, h, w, dom_mask=None, ch_mask=None, dom_thresh=0.3,
                 max_dist=100, max_angle=70):
    """V4 bridging: EP-Line (strict) + EP-EP (crossing check) + Tri-junction.
    ch_mask: the channel's own binary mask (R mask or B mask) for stricter checks."""
    eps = get_eps_tangent(skel, h, w)
    ct = np.cos(np.radians(max_angle))
    all_bridges = []

    # === Phase 1: EP-to-Line bridges (strict: must stay on channel mask) ===
    line_bridges, line_connected = find_ep_to_line_bridges(
        skel, eps, vm, h, w, dom_mask, ch_mask, search_radius=25, dom_thresh=dom_thresh)

    # Apply line bridges first
    bridged = skel.copy()
    for br in line_bridges:
        for (y, x) in br['pts']:
            if 0<=y<h and 0<=x<w: bridged[y, x] = True
        all_bridges.append(br)

    # Re-skeletonize after line bridges, then get new EPs for pair matching
    bridged = skeletonize(bridged)
    eps2 = get_eps_tangent(bridged, h, w)

    # === Phase 2: EP-EP pair bridges (with crossing check) ===
    cands = []
    for i in range(len(eps2)):
        for j in range(i+1, len(eps2)):
            p1 = np.array(eps2[i]['pos'], dtype=float)
            p2 = np.array(eps2[j]['pos'], dtype=float)
            d = np.linalg.norm(p2-p1)
            if d < 3 or d > max_dist: continue
            d12 = (p2-p1)/d
            c1 = np.dot(eps2[i]['tangent'], d12)
            c2 = np.dot(eps2[j]['tangent'], -d12)
            if c1 < ct or c2 < ct: continue
            pts = bezier(eps2[i], eps2[j])
            ov = sum(1 for (y,x) in pts if 0<=y<h and 0<=x<w and vm[y,x])
            vr = ov / max(len(pts), 1)
            if vr < 0.7: continue
            # Channel mask check: at least 50% on own channel mask
            if ch_mask is not None:
                ch_ov = sum(1 for (y,x) in pts if 0<=y<h and 0<=x<w and ch_mask[y,x])
                cr = ch_ov / max(len(pts), 1)
                if cr < 0.5: continue
            # Dominance check
            if dom_mask is not None:
                dom_ov = sum(1 for (y,x) in pts if 0<=y<h and 0<=x<w and dom_mask[y,x])
                dr = dom_ov / max(len(pts), 1)
                if dr < dom_thresh: continue
            # Crossing check: reject if bridge crosses existing skel >= 2 times
            crossings = count_skel_crossings(pts, bridged, h, w)
            if crossings >= 2: continue
            sc = (c1+c2)/2 - d/max_dist*0.2 + vr*0.2
            cands.append((i, j, sc, eps2[i], eps2[j], pts))

    cands.sort(key=lambda x: -x[2])
    used = set()
    for i, j, sc, e1, e2, pts in cands:
        if i not in used and j not in used:
            # Double-check crossing on current bridged state
            crossings = count_skel_crossings(pts, bridged, h, w)
            if crossings >= 2: continue
            all_bridges.append({'type': 'pair', 'ep1': e1, 'ep2': e2, 'pts': pts})
            for (y, x) in pts:
                if 0<=y<h and 0<=x<w: bridged[y, x] = True
            used.add(i); used.add(j)

    # Re-skeletonize
    bridged = skeletonize(bridged)

    # === Phase 3: Tri-junction recovery ===
    eps3 = get_eps_tangent(bridged, h, w)
    if len(eps3) >= 3:
        positions = np.array([e['pos'] for e in eps3])
        dist_mat = cdist(positions, positions)
        visited = set()
        for idx in range(len(eps3)):
            if idx in visited: continue
            nearby = [j for j in range(len(eps3))
                      if j != idx and j not in visited and dist_mat[idx, j] < 20]
            if len(nearby) >= 2:
                group = [idx] + nearby[:2]
                group_eps = [eps3[g] for g in group]
                cy = int(np.mean([e['pos'][0] for e in group_eps]))
                cx = int(np.mean([e['pos'][1] for e in group_eps]))
                # Centroid must be on vessel mask AND channel mask
                on_vm = 0 <= cy < h and 0 <= cx < w and vm[cy, cx]
                on_ch = ch_mask is None or (0 <= cy < h and 0 <= cx < w and ch_mask[cy, cx])
                if on_vm and on_ch:
                    for g in group: visited.add(g)
                    for e in group_eps:
                        pts = line_pts(e['pos'], (cy, cx))
                        for (y, x) in pts:
                            if 0<=y<h and 0<=x<w: bridged[y, x] = True
                    all_bridges.append({'type': 'tri', 'centroid': (cy, cx), 'eps': group_eps})

    bridged = skeletonize(bridged)
    return bridged, all_bridges

BG_A = 0.6; WV = 0.35

def make_gt_bg(crop):
    v = crop.copy().astype(float) * BG_A
    return v * (1-WV) + 255*WV

def draw_all_bridges(vis, bridges, pair_color, line_color, tri_color, by):
    for br in bridges:
        if br['type'] == 'pair':
            for py, px in br['pts']:
                if 0<=py<by and 0<=px<vis.shape[1]:
                    for dy in range(-1,2):
                        for dx in range(-1,2):
                            ny, nx = py+dy, px+dx
                            if 0<=ny<by and 0<=nx<vis.shape[1]:
                                vis[ny, nx] = pair_color
        elif br['type'] == 'line':
            for py, px in br['pts']:
                if 0<=py<by and 0<=px<vis.shape[1]:
                    for dy in range(-1,2):
                        for dx in range(-1,2):
                            ny, nx = py+dy, px+dx
                            if 0<=ny<by and 0<=nx<vis.shape[1]:
                                vis[ny, nx] = line_color
            # Mark hit point
            hy, hx = br['hit']
            if 0<=hy<by and 0<=hx<vis.shape[1]:
                cv2.circle(vis, (hx, hy), 3, (255,255,255), -1)
        elif br['type'] == 'tri':
            cy, cx = br['centroid']
            for e in br['eps']:
                pts = line_pts(e['pos'], (cy, cx))
                for py, px in pts:
                    if 0<=py<by and 0<=px<vis.shape[1]:
                        for dy in range(-1,2):
                            for dx in range(-1,2):
                                ny, nx = py+dy, px+dx
                                if 0<=ny<by and 0<=nx<vis.shape[1]:
                                    vis[ny, nx] = tri_color
            if 0<=cy<by and 0<=cx<vis.shape[1]:
                cv2.circle(vis, (cx, cy), 4, (0,255,0), -1)

for sid in sids:
    gt_path = None
    for m in ['LSGAN', 'Pix2pix', 'WGANGP']:
        p = multi_dir / m / f'{sid}_real_B.png'
        if p.exists(): gt_path = p; break
    if gt_path is None: continue

    gt = np.array(Image.open(gt_path))
    gt_blur = cv2.GaussianBlur(gt, (7,7), 0)
    h, w = gt.shape[:2]; by = int(h*0.8)
    R = gt_blur[:,:,0].astype(np.float32)
    B = gt_blur[:,:,2].astype(np.float32)
    gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY); diff = B - R

    r_mask = remove_small_objects(
        cv2.morphologyEx((R>30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)
    r_skel = skeletonize(r_mask); r_filt = r_skel & (diff < 0)

    b_seed = B>50; b_low = B>20
    b_lab, b_n = ndimage.label(b_low)
    b_mask_img = np.zeros_like(b_low, dtype=bool)
    for i in range(1, b_n+1):
        if np.any(b_seed[b_lab==i]): b_mask_img |= (b_lab==i)
    b_mask_img = remove_small_objects(
        cv2.morphologyEx(b_mask_img.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2)>0, min_size=50)
    b_skel = skeletonize(b_mask_img); b_filt = b_skel & (diff > 0)

    vessel_all = remove_small_objects(
        cv2.morphologyEx((gray>10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=100)
    vm = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

    r_dom = diff < 0; b_dom = diff > 0
    r_bridged, r_br = do_bridge_v4(r_filt, vm, h, w, dom_mask=r_dom, ch_mask=r_mask)
    b_bridged, b_br = do_bridge_v4(b_filt, vm, h, w, dom_mask=b_dom, ch_mask=b_mask_img)

    r_jn_pre, r_ep_pre = get_jn_ep(r_filt, h, by)
    b_jn_pre, b_ep_pre = get_jn_ep(b_filt, h, by)
    r_jn_post, r_ep_post = get_jn_ep(r_bridged, h, by)
    b_jn_post, b_ep_post = get_jn_ep(b_bridged, h, by)

    r_line = sum(1 for b in r_br if b['type']=='line')
    r_pair = sum(1 for b in r_br if b['type']=='pair')
    r_tri = sum(1 for b in r_br if b['type']=='tri')
    b_line = sum(1 for b in b_br if b['type']=='line')
    b_pair = sum(1 for b in b_br if b['type']=='pair')
    b_tri = sum(1 for b in b_br if b['type']=='tri')

    print(f'{sid}:')
    print(f'  R pre: JN={len(r_jn_pre)} EP={len(r_ep_pre)} | post: JN={len(r_jn_post)} EP={len(r_ep_post)} | line={r_line} pair={r_pair} tri={r_tri}')
    print(f'  B pre: JN={len(b_jn_pre)} EP={len(b_ep_pre)} | post: JN={len(b_jn_post)} EP={len(b_ep_post)} | line={b_line} pair={b_pair} tri={b_tri}')

    # === Figure: 4 rows x 3 cols ===
    fig, axes = plt.subplots(4, 3, figsize=(15, 17))
    fig.suptitle(f'{sid} — V4: EP-Line + EP-EP (crossing check) + Tri', fontsize=13, fontweight='bold')

    # Row 0: GT + channels
    axes[0,0].imshow(gt[:by]); axes[0,0].set_title('GT', fontsize=11)
    r_vis = np.zeros((by,w,3), dtype=np.uint8)
    r_vis[:,:,0] = gt_blur[:by,:,0]; r_vis[:,:,1] = (gt_blur[:by,:,0]*0.3).astype(np.uint8)
    axes[0,1].imshow(r_vis); axes[0,1].set_title('R channel', fontsize=11)
    b_vis = np.zeros((by,w,3), dtype=np.uint8)
    b_vis[:,:,2] = gt_blur[:by,:,2]; b_vis[:,:,1] = (gt_blur[:by,:,2]*0.3).astype(np.uint8)
    axes[0,2].imshow(b_vis); axes[0,2].set_title('B channel', fontsize=11)

    # Row 1: Before
    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(r_filt[:by].astype(np.uint8), dk2)>0] = [230,70,70]
    for y,x in r_jn_pre: cv2.circle(vis,(x,y),4,(255,0,255),-1)
    for y,x in r_ep_pre: cv2.circle(vis,(x,y),3,(255,255,0),-1)
    axes[1,0].imshow(vis); axes[1,0].set_title(f'Before: R JN={len(r_jn_pre)} EP={len(r_ep_pre)}', fontsize=10)

    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(b_filt[:by].astype(np.uint8), dk2)>0] = [70,140,240]
    for y,x in b_jn_pre: cv2.circle(vis,(x,y),4,(255,0,255),-1)
    for y,x in b_ep_pre: cv2.circle(vis,(x,y),3,(255,255,0),-1)
    axes[1,1].imshow(vis); axes[1,1].set_title(f'Before: B JN={len(b_jn_pre)} EP={len(b_ep_pre)}', fontsize=10)

    vis = make_gt_bg(gt[:by])
    r_d = cv2.dilate(r_filt[:by].astype(np.uint8), dk4)>0
    b_d = cv2.dilate(b_filt[:by].astype(np.uint8), dk4)>0
    vis[r_d & ~b_d] = [230,70,70]; vis[b_d & ~r_d] = [70,140,240]; vis[r_d & b_d] = [180,100,220]
    vis = vis.astype(np.uint8)
    for y,x in r_jn_pre+b_jn_pre: cv2.circle(vis,(x,y),5,(255,0,255),2)
    for y,x in r_ep_pre+b_ep_pre: cv2.circle(vis,(x,y),5,(255,255,0),2)
    axes[1,2].imshow(vis)
    axes[1,2].set_title(f'Before: R+B  JN={len(r_jn_pre)+len(b_jn_pre)} EP={len(r_ep_pre)+len(b_ep_pre)}', fontsize=10)

    # Row 2: Bridging visualization
    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(r_filt[:by].astype(np.uint8), dk2)>0] = [230,70,70]
    draw_all_bridges(vis, r_br, [255,180,50], [255,100,255], [0,255,100], by)
    axes[2,0].imshow(vis)
    axes[2,0].set_title(f'R: line={r_line} pair={r_pair} tri={r_tri}', fontsize=10)

    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(b_filt[:by].astype(np.uint8), dk2)>0] = [70,140,240]
    draw_all_bridges(vis, b_br, [50,255,180], [255,100,255], [0,255,100], by)
    axes[2,1].imshow(vis)
    axes[2,1].set_title(f'B: line={b_line} pair={b_pair} tri={b_tri}', fontsize=10)

    vis = make_gt_bg(gt[:by])
    vis[r_d & ~b_d] = [230,70,70]; vis[b_d & ~r_d] = [70,140,240]; vis[r_d & b_d] = [180,100,220]
    vis = vis.astype(np.uint8)
    draw_all_bridges(vis, r_br, [255,180,50], [255,100,255], [0,255,100], by)
    draw_all_bridges(vis, b_br, [50,255,180], [255,100,255], [0,255,100], by)
    axes[2,2].imshow(vis)
    axes[2,2].set_title('Bridging on GT (orange=pair, pink=line, green=tri)', fontsize=9)

    # Row 3: After
    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(r_bridged[:by].astype(np.uint8), dk2)>0] = [230,70,70]
    for y,x in r_jn_post: cv2.circle(vis,(x,y),4,(255,0,255),-1)
    for y,x in r_ep_post: cv2.circle(vis,(x,y),3,(255,255,0),-1)
    axes[3,0].imshow(vis); axes[3,0].set_title(f'After: R JN={len(r_jn_post)} EP={len(r_ep_post)}', fontsize=10)

    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(b_bridged[:by].astype(np.uint8), dk2)>0] = [70,140,240]
    for y,x in b_jn_post: cv2.circle(vis,(x,y),4,(255,0,255),-1)
    for y,x in b_ep_post: cv2.circle(vis,(x,y),3,(255,255,0),-1)
    axes[3,1].imshow(vis); axes[3,1].set_title(f'After: B JN={len(b_jn_post)} EP={len(b_ep_post)}', fontsize=10)

    vis = make_gt_bg(gt[:by])
    r_d2 = cv2.dilate(r_bridged[:by].astype(np.uint8), dk4)>0
    b_d2 = cv2.dilate(b_bridged[:by].astype(np.uint8), dk4)>0
    vis[r_d2 & ~b_d2] = [230,70,70]; vis[b_d2 & ~r_d2] = [70,140,240]; vis[r_d2 & b_d2] = [180,100,220]
    vis = vis.astype(np.uint8)
    for y,x in r_jn_post+b_jn_post: cv2.circle(vis,(x,y),5,(255,0,255),2)
    for y,x in r_ep_post+b_ep_post: cv2.circle(vis,(x,y),5,(255,255,0),2)
    axes[3,2].imshow(vis)
    axes[3,2].set_title(f'After: R+B  JN={len(r_jn_post)+len(b_jn_post)} EP={len(r_ep_post)+len(b_ep_post)}', fontsize=10)

    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT / f'bridge_v4_{sid}.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved bridge_v4_{sid}.png')

print('Done!')
