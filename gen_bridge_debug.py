"""Bridge before/after debug for 5 samples."""
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

def do_bridge(skel, vm, h, w, dom_mask=None, dom_thresh=0.5):
    """Bridge with optional dominance mask constraint.
    dom_mask: boolean mask where this channel is dominant (e.g. diff<0 for R)
    dom_thresh: fraction of bridge path that must be in dominance region (0.5 = 50%)
    """
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
            # Dominance check: bridge must stay in its channel's territory
            if dom_mask is not None:
                dom_ov = sum(1 for (y,x) in pts if 0<=y<h and 0<=x<w and dom_mask[y,x])
                dr = dom_ov / max(len(pts), 1)
                if dr < dom_thresh: continue
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

BG_A = 0.6; WV = 0.35

def make_gt_bg(crop):
    v = crop.copy().astype(float) * BG_A
    return v * (1-WV) + 255*WV

def draw_bridges(vis, bridges, color, by):
    for br in bridges:
        for py, px in br['pts']:
            if 0 <= py < by and 0 <= px < vis.shape[1]:
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = py+dy, px+dx
                        if 0 <= ny < by and 0 <= nx < vis.shape[1]:
                            vis[ny, nx] = color

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

    r_mask = remove_small_objects(
        cv2.morphologyEx((R > 30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=50)
    r_skel = skeletonize(r_mask)
    r_filt = r_skel & (diff < 0)

    b_seed = B > 50; b_low = B > 20
    b_lab, b_n = ndimage.label(b_low)
    b_mask = np.zeros_like(b_low, dtype=bool)
    for i in range(1, b_n+1):
        if np.any(b_seed[b_lab == i]):
            b_mask |= (b_lab == i)
    b_mask = remove_small_objects(
        cv2.morphologyEx(b_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0, min_size=50)
    b_skel = skeletonize(b_mask)
    b_filt = b_skel & (diff > 0)

    vessel_all = remove_small_objects(
        cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=100)
    vm = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

    # Dominance masks
    r_dom = diff < 0   # R > B
    b_dom = diff > 0   # B > R

    # Bridge with dominance constraint (50% of path in channel territory)
    r_bridged, r_br = do_bridge(r_filt, vm, h, w, dom_mask=r_dom, dom_thresh=0.5)
    b_bridged, b_br = do_bridge(b_filt, vm, h, w, dom_mask=b_dom, dom_thresh=0.5)

    r_jn_pre, r_ep_pre = get_jn_ep(r_filt, h, by)
    b_jn_pre, b_ep_pre = get_jn_ep(b_filt, h, by)
    r_jn_post, r_ep_post = get_jn_ep(r_bridged, h, by)
    b_jn_post, b_ep_post = get_jn_ep(b_bridged, h, by)

    print(f'{sid}:')
    print(f'  R pre: JN={len(r_jn_pre)} EP={len(r_ep_pre)} | post: JN={len(r_jn_post)} EP={len(r_ep_post)} | br={len(r_br)}')
    print(f'  B pre: JN={len(b_jn_pre)} EP={len(b_ep_pre)} | post: JN={len(b_jn_post)} EP={len(b_ep_post)} | br={len(b_br)}')

    # Figure: 2 rows x 3 cols
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(f'{sid} — Before vs After Bridging', fontsize=14, fontweight='bold')

    stages = [
        ('Before', r_filt, b_filt, r_jn_pre, b_jn_pre, r_ep_pre, b_ep_pre, [], []),
        ('After', r_bridged, b_bridged, r_jn_post, b_jn_post, r_ep_post, b_ep_post, r_br, b_br),
    ]

    for row, (label, r_sk, b_sk, rjn, bjn, rep, bep, rbr, bbr) in enumerate(stages):
        # Col 0: R skel + JN + EP
        vis = np.zeros((by, w, 3), dtype=np.uint8)
        vis[cv2.dilate(r_sk[:by].astype(np.uint8), dk2) > 0] = [230, 70, 70]
        draw_bridges(vis, rbr, [255, 180, 50], by)
        for y, x in rjn:
            cv2.circle(vis, (x, y), 4, (255, 0, 255), -1)
        for y, x in rep:
            cv2.circle(vis, (x, y), 3, (255, 255, 0), -1)
        axes[row, 0].imshow(vis)
        t = f'{label}: R JN={len(rjn)} EP={len(rep)}'
        if rbr: t += f' br={len(rbr)}'
        axes[row, 0].set_title(t, fontsize=10)

        # Col 1: B skel + JN + EP
        vis = np.zeros((by, w, 3), dtype=np.uint8)
        vis[cv2.dilate(b_sk[:by].astype(np.uint8), dk2) > 0] = [70, 140, 240]
        draw_bridges(vis, bbr, [50, 255, 180], by)
        for y, x in bjn:
            cv2.circle(vis, (x, y), 4, (255, 0, 255), -1)
        for y, x in bep:
            cv2.circle(vis, (x, y), 3, (255, 255, 0), -1)
        axes[row, 1].imshow(vis)
        t = f'{label}: B JN={len(bjn)} EP={len(bep)}'
        if bbr: t += f' br={len(bbr)}'
        axes[row, 1].set_title(t, fontsize=10)

        # Col 2: R+B on GT + JN + EP
        vis = make_gt_bg(gt[:by])
        r_d = cv2.dilate(r_sk[:by].astype(np.uint8), dk4) > 0
        b_d = cv2.dilate(b_sk[:by].astype(np.uint8), dk4) > 0
        vis[r_d & ~b_d] = [230, 70, 70]
        vis[b_d & ~r_d] = [70, 140, 240]
        vis[r_d & b_d] = [180, 100, 220]
        vis = vis.astype(np.uint8)
        all_jn = rjn + bjn
        all_ep = rep + bep
        for y, x in all_jn:
            cv2.circle(vis, (x, y), 5, (255, 0, 255), 2)
        for y, x in all_ep:
            cv2.circle(vis, (x, y), 5, (255, 255, 0), 2)
        axes[row, 2].imshow(vis)
        axes[row, 2].set_title(f'{label}: R+B on GT  JN={len(all_jn)} EP={len(all_ep)}', fontsize=10)

    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT / f'bridge_detail_{sid}.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved bridge_detail_{sid}.png')

print('Done!')
