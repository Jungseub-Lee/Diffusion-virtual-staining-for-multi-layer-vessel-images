"""
New pipeline v2:
- R mask: R > 45
- B mask: B/(R+B+1) > 0.55 & B > 15 (M2 method)
- Skel: no dominance cut
- Bridge: EP-EP with channel mask check
- Depth tag: crossing region labeling
- Run on all 5 samples
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

DATA = Path(r'[1] Data')
OUT = Path(r'analysis_output/debug_pipeline')
OUT.mkdir(exist_ok=True)

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
dk2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

sids = ['1-19-716', '9-15-512', '16-18-716', '18-19-512', '1-16-512']

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
    return [(int(round(((1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2)[0])),
             int(round(((1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2)[1])))
            for t in np.linspace(0, 1, n)]

def simple_bridge(skel, ch_mask, vm, h, w, max_dist=60, max_angle=65):
    eps = get_eps_tangent(skel, h, w)
    ct = np.cos(np.radians(max_angle))
    cands = []
    for i in range(len(eps)):
        for j in range(i+1, len(eps)):
            p1 = np.array(eps[i]['pos'], dtype=float)
            p2 = np.array(eps[j]['pos'], dtype=float)
            d = np.linalg.norm(p2-p1)
            if d < 3 or d > max_dist: continue
            d12 = (p2-p1)/d
            c1v = np.dot(eps[i]['tangent'], d12)
            c2v = np.dot(eps[j]['tangent'], -d12)
            if c1v < ct or c2v < ct: continue
            pts = bezier(eps[i], eps[j])
            ch_count = sum(1 for (y,x) in pts if 0<=y<h and 0<=x<w and ch_mask[y,x])
            if ch_count / max(len(pts),1) < 0.7: continue
            vm_count = sum(1 for (y,x) in pts if 0<=y<h and 0<=x<w and vm[y,x])
            if vm_count / max(len(pts),1) < 0.7: continue
            sc = (c1v+c2v)/2 - d/max_dist*0.2
            cands.append((i, j, sc, eps[i], eps[j], pts))
    cands.sort(key=lambda x: -x[2])
    used = set(); bridged = skel.copy(); n_bridges = 0
    for i, j, sc, e1, e2, pts in cands:
        if i not in used and j not in used:
            for (y,x) in pts:
                if 0<=y<h and 0<=x<w: bridged[y,x] = True
            used.add(i); used.add(j); n_bridges += 1
    return skeletonize(bridged), n_bridges

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
        if nbr >= 3: jn_pts.append((int(py.mean()), int(px.mean())))
    ep_pts = [(y,x) for y,x in np.argwhere((nb==1) & roi) if y < by-3]
    return jn_pts, ep_pts

BG_A = 0.5; WV = 0.4

for sid in sids:
    gt_path = None
    for m in ['LSGAN','Pix2pix','WGANGP']:
        p = DATA / 'Multi-color' / m / f'{sid}_real_B.png'
        if p.exists(): gt_path = p; break
    if gt_path is None: continue

    gt = np.array(Image.open(gt_path))
    gt_blur = cv2.GaussianBlur(gt, (7,7), 0)
    h, w = gt.shape[:2]; by = int(h*0.8)
    R = gt_blur[:,:,0].astype(np.float32)
    B = gt_blur[:,:,2].astype(np.float32)
    gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
    diff = B - R

    # === NEW MASKS ===
    r_mask = remove_small_objects(
        cv2.morphologyEx((R>45).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)
    ratio = B / (R + B + 1)
    b_mask = remove_small_objects(
        cv2.morphologyEx(((ratio > 0.55) & (B > 15)).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)

    vessel_all = remove_small_objects(
        cv2.morphologyEx((gray>10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=100)
    vm = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

    # === SKEL (no dominance cut) ===
    r_skel = skeletonize(r_mask)
    b_skel = skeletonize(b_mask)

    # === BRIDGE ===
    r_bridged, r_nbr = simple_bridge(r_skel, r_mask, vm, h, w)
    b_bridged, b_nbr = simple_bridge(b_skel, b_mask, vm, h, w)

    # === DEPTH TAG ===
    crossing = r_mask & b_mask
    r_dominant = r_bridged & (diff <= 0)
    r_under_b = r_bridged & (diff > 0)
    b_dominant = b_bridged & (diff >= 0)
    b_under_r = b_bridged & (diff < 0)

    # === METRICS ===
    r_jn, r_ep = get_jn_ep(r_bridged, h, by)
    b_jn, b_ep = get_jn_ep(b_bridged, h, by)
    r_comp = ndimage.label(r_bridged)[1]
    b_comp = ndimage.label(b_bridged)[1]
    ovl = np.sum(r_mask[:by] & b_mask[:by])
    ovl_pct = ovl / max(np.sum(r_mask[:by]),1) * 100

    print(f'\n{sid}:')
    print(f'  R mask={np.sum(r_mask[:by])}px | B mask={np.sum(b_mask[:by])}px | overlap={ovl}px ({ovl_pct:.0f}% of R)')
    print(f'  R: {r_comp} comp, JN={len(r_jn)}, EP={len(r_ep)}, L={np.sum(r_bridged[:by])}, bridges={r_nbr}')
    print(f'  B: {b_comp} comp, JN={len(b_jn)}, EP={len(b_ep)}, L={np.sum(b_bridged[:by])}, bridges={b_nbr}')
    print(f'  R depth: dominant={np.sum(r_dominant[:by])}px under_B={np.sum(r_under_b[:by])}px')
    print(f'  B depth: dominant={np.sum(b_dominant[:by])}px under_R={np.sum(b_under_r[:by])}px')

    # === FIGURE: 4 rows x 3 cols ===
    fig, axes = plt.subplots(4, 3, figsize=(15, 17))
    fig.suptitle(f'{sid} — New Pipeline v2\nR>45 | B ratio>0.55 | No dom cut | Bridge | Depth tag',
                 fontsize=12, fontweight='bold')

    # Row 0: GT, R mask, B mask
    axes[0,0].imshow(gt[:by]); axes[0,0].set_title('GT', fontsize=11)
    axes[0,1].imshow(r_mask[:by], cmap='gray'); axes[0,1].set_title(f'R mask (R>45) | {np.sum(r_mask[:by])}px', fontsize=10)
    axes[0,2].imshow(b_mask[:by], cmap='gray'); axes[0,2].set_title(f'B mask (ratio>0.55) | {np.sum(b_mask[:by])}px', fontsize=10)

    # Row 1: R+B overlap, R skel, B skel
    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[r_mask[:by] & ~b_mask[:by]] = [200,60,60]
    vis[b_mask[:by] & ~r_mask[:by]] = [60,120,200]
    vis[r_mask[:by] & b_mask[:by]] = [200,100,200]
    axes[1,0].imshow(vis); axes[1,0].set_title(f'R(red)+B(blue) overlap(purple)={ovl_pct:.0f}%', fontsize=9)

    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(r_bridged[:by].astype(np.uint8), dk2)>0] = [230,70,70]
    axes[1,1].imshow(vis); axes[1,1].set_title(f'R skel bridged | {r_comp} comp, br={r_nbr}', fontsize=10)

    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(b_bridged[:by].astype(np.uint8), dk2)>0] = [70,140,240]
    axes[1,2].imshow(vis); axes[1,2].set_title(f'B skel bridged | {b_comp} comp, br={b_nbr}', fontsize=10)

    # Row 2: Depth-tagged skels
    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(r_dominant[:by].astype(np.uint8), dk2)>0] = [255,80,80]
    vis[cv2.dilate(r_under_b[:by].astype(np.uint8), dk2)>0] = [120,40,40]
    axes[2,0].imshow(vis); axes[2,0].set_title(f'R depth: bright=dominant dark=underB', fontsize=9)

    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(b_dominant[:by].astype(np.uint8), dk2)>0] = [80,150,255]
    vis[cv2.dilate(b_under_r[:by].astype(np.uint8), dk2)>0] = [40,60,120]
    axes[2,1].imshow(vis); axes[2,1].set_title(f'B depth: bright=dominant dark=underR', fontsize=9)

    # Combined depth
    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(r_dominant[:by].astype(np.uint8), dk2)>0] = [255,80,80]
    vis[cv2.dilate(r_under_b[:by].astype(np.uint8), dk2)>0] = [120,40,40]
    vis[cv2.dilate(b_dominant[:by].astype(np.uint8), dk2)>0] = [80,150,255]
    vis[cv2.dilate(b_under_r[:by].astype(np.uint8), dk2)>0] = [40,60,120]
    axes[2,2].imshow(vis); axes[2,2].set_title('Combined depth-tagged', fontsize=10)

    # Row 3: On GT with EP/JN markers
    vis = gt[:by].copy().astype(float) * BG_A
    vis = vis * (1-WV) + 255*WV
    r_d4 = cv2.dilate(r_bridged[:by].astype(np.uint8), dk4)>0
    b_d4 = cv2.dilate(b_bridged[:by].astype(np.uint8), dk4)>0
    vis[r_d4 & ~b_d4] = [230,70,70]
    vis[b_d4 & ~r_d4] = [70,140,240]
    vis[r_d4 & b_d4] = [180,100,220]
    vis = vis.astype(np.uint8)
    for y,x in r_jn+b_jn: cv2.circle(vis,(x,y),5,(255,0,255),2)
    for y,x in r_ep+b_ep: cv2.circle(vis,(x,y),5,(255,255,0),2)
    axes[3,0].imshow(vis)
    axes[3,0].set_title(f'R+B on GT | JN={len(r_jn)+len(b_jn)} EP={len(r_ep)+len(b_ep)}', fontsize=10)

    # R only on GT
    vis = gt[:by].copy().astype(float) * BG_A
    vis = vis * (1-WV) + 255*WV
    vis[r_d4] = [230,70,70]; vis = vis.astype(np.uint8)
    for y,x in r_jn: cv2.circle(vis,(x,y),5,(255,0,255),2)
    for y,x in r_ep: cv2.circle(vis,(x,y),5,(255,255,0),2)
    axes[3,1].imshow(vis); axes[3,1].set_title(f'R on GT | JN={len(r_jn)} EP={len(r_ep)}', fontsize=10)

    # B only on GT
    vis = gt[:by].copy().astype(float) * BG_A
    vis = vis * (1-WV) + 255*WV
    vis[b_d4] = [70,140,240]; vis = vis.astype(np.uint8)
    for y,x in b_jn: cv2.circle(vis,(x,y),5,(255,0,255),2)
    for y,x in b_ep: cv2.circle(vis,(x,y),5,(255,255,0),2)
    axes[3,2].imshow(vis); axes[3,2].set_title(f'B on GT | JN={len(b_jn)} EP={len(b_ep)}', fontsize=10)

    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT / f'newpipe_v2_{sid}.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved newpipe_v2_{sid}.png')

print('\nDone!')
