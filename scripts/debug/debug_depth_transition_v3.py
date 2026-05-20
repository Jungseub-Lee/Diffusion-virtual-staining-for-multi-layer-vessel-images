"""
Depth transition v3: Full 4-step visualization for each threshold (30/60/90%).
Same style as v1 depth_transition but with raw yellow (no dilation).
One figure per sample, 3 threshold columns x 4 step rows.
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

sids = ['18-19-512', '1-19-716', '1-16-512']
thresholds = [0.3, 0.6, 0.9]

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

def bezier_pts(e1_pos, e1_tan, e2_pos, e2_tan):
    p1 = np.array(e1_pos, dtype=float); p2 = np.array(e2_pos, dtype=float)
    d = np.linalg.norm(p2-p1)
    c1 = p1 + np.array(e1_tan)*d*0.4; c2 = p2 + np.array(e2_tan)*d*0.4
    n = max(int(d*1.5), 10)
    return [(int(round(((1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2)[0])),
             int(round(((1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2)[1])))
            for t in np.linspace(0, 1, n)]

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
    R = gt_blur[:,:,0].astype(np.float32); B = gt_blur[:,:,2].astype(np.float32)
    gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2HSV)
    hue = hsv[:,:,0].astype(float); sat = hsv[:,:,1].astype(float)
    diff = B - R

    r_mask = remove_small_objects(
        cv2.morphologyEx((R>45).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)
    ratio = B / (R + B + 1)
    b_mask = remove_small_objects(
        cv2.morphologyEx(((ratio > 0.55) & (B > 15)).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)

    vessel_all = remove_small_objects(
        cv2.morphologyEx((gray>10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=100)
    vm = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

    r_skel = skeletonize(r_mask); b_skel = skeletonize(b_mask)

    # Yellow zone (raw, no dilation)
    yellow_hue = (hue >= 15) & (hue <= 50) & (sat > 30) & (gray > 15)
    yellow_diff = (np.abs(diff) < 15) & (R > 20) & (B > 20)
    yellow_zone = yellow_hue | yellow_diff

    r_eps = [e for e in get_eps_tangent(r_skel, h, w) if e['pos'][0] < by-3]
    b_eps = [e for e in get_eps_tangent(b_skel, h, w) if e['pos'][0] < by-3]

    # Find all pairs
    pairs = []
    for ri, rep in enumerate(r_eps):
        ry, rx = rep['pos']
        for bi, bep in enumerate(b_eps):
            by2, bx = bep['pos']
            d = np.sqrt((ry-by2)**2 + (rx-bx)**2)
            if 2 < d < 25:
                path = bezier_pts(rep['pos'], rep['tangent'], bep['pos'], bep['tangent'])
                y_count = sum(1 for (py,px) in path if 0<=py<h and 0<=px<w and yellow_zone[py,px])
                y_ratio = y_count / max(len(path), 1)
                vm_count = sum(1 for (py,px) in path if 0<=py<h and 0<=px<w and vm[py,px])
                vm_ratio = vm_count / max(len(path), 1)
                pairs.append({
                    'ri': ri, 'bi': bi, 'r_ep': rep, 'b_ep': bep,
                    'dist': d, 'path': path,
                    'y_ratio': y_ratio, 'vm_ratio': vm_ratio,
                    'mid': (int((ry+by2)/2), int((rx+bx)/2))
                })

    print(f"\n{'='*60}")
    print(f"{sid}: {len(r_eps)} R EPs, {len(b_eps)} B EPs, {len(pairs)} pairs")

    # === Figure: 4 rows x 3 cols (thresholds) ===
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f'{sid} — Depth Transition: Yellow Threshold 30% / 60% / 90% (raw, no dilation)',
                 fontsize=13, fontweight='bold')

    for col, thr in enumerate(thresholds):
        passed = [p for p in pairs if p['y_ratio'] >= thr and p['vm_ratio'] >= 0.6]
        failed = [p for p in pairs if not (p['y_ratio'] >= thr and p['vm_ratio'] >= 0.6)]
        # Greedy
        passed.sort(key=lambda x: x['dist'])
        used_r = set(); used_b = set(); selected = []
        for p in passed:
            if p['ri'] not in used_r and p['bi'] not in used_b:
                selected.append(p); used_r.add(p['ri']); used_b.add(p['bi'])

        n_pass = len(passed); n_sel = len(selected)
        print(f"  thr={thr:.0%}: pass={n_pass} selected={n_sel}")

        # Row 0: All pair candidates
        vis = gt[:by].copy().astype(float) * 0.5
        vis[cv2.dilate(r_skel[:by].astype(np.uint8), dk2)>0] = [230,70,70]
        vis[cv2.dilate(b_skel[:by].astype(np.uint8), dk2)>0] = [70,140,240]
        vis = vis.astype(np.uint8)
        for p in pairs:
            ry, rx = p['r_ep']['pos']; by2, bx = p['b_ep']['pos']
            is_pass = p['y_ratio'] >= thr and p['vm_ratio'] >= 0.6
            color = (0,255,0) if is_pass else (255,80,80)
            cv2.line(vis, (rx,ry), (bx,by2), color, 1, cv2.LINE_AA)
            cv2.circle(vis, (rx,ry), 3, (255,100,100), -1)
            cv2.circle(vis, (bx,by2), 3, (100,150,255), -1)
        axes[0,col].imshow(vis)
        axes[0,col].set_title(f'thr≥{thr:.0%} | Step 1: pairs ({len(pairs)})\ngreen=pass({n_pass}) red=fail({len(failed)})', fontsize=9)

        # Row 1: Yellow zone + midpoints
        vis = np.zeros((by,w,3), dtype=np.uint8)
        vis[yellow_hue[:by]] = [220,220,50]
        vis[yellow_diff[:by] & ~yellow_hue[:by]] = [160,160,50]
        for p in pairs:
            my, mx = p['mid']
            if my < by:
                is_on_y = p['y_ratio'] >= thr
                color = (0,255,0) if is_on_y else (255,0,0)
                cv2.circle(vis, (mx,my), 5, color, 2)
                cv2.putText(vis, f"{p['y_ratio']:.0%}", (mx+6, my+3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255,255,255), 1)
        axes[1,col].imshow(vis)
        axes[1,col].set_title(f'Step 2: Yellow zone + path ratios', fontsize=9)

        # Row 2: Pass/Fail paths
        vis = gt[:by].copy().astype(float) * 0.4
        vis[cv2.dilate(r_skel[:by].astype(np.uint8), dk2)>0] = [230,70,70]
        vis[cv2.dilate(b_skel[:by].astype(np.uint8), dk2)>0] = [70,140,240]
        vis = vis.astype(np.uint8)
        for p in failed:
            for py, px in p.get('path', []):
                if 0<=py<by and 0<=px<w: vis[py,px] = [150,50,50]
        for p in passed:
            for py, px in p.get('path', []):
                if 0<=py<by and 0<=px<w:
                    for dy in range(-1,2):
                        for dx in range(-1,2):
                            ny, nx = py+dy, px+dx
                            if 0<=ny<by and 0<=nx<w: vis[ny,nx] = [50,255,100]
        axes[2,col].imshow(vis)
        axes[2,col].set_title(f'Step 3: PASS({n_pass}) green | FAIL({len(failed)}) dark', fontsize=9)

        # Row 3: Final selected on GT
        vis = gt[:by].copy().astype(float) * BG_A
        vis = (vis * (1-WV) + 255*WV)
        vis[cv2.dilate(r_skel[:by].astype(np.uint8), dk4)>0] = [230,70,70]
        vis[cv2.dilate(b_skel[:by].astype(np.uint8), dk4)>0] = [70,140,240]
        vis = vis.astype(np.uint8)
        for p in selected:
            for py, px in p['path']:
                if 0<=py<by and 0<=px<w:
                    for dy in range(-1,2):
                        for dx in range(-1,2):
                            ny, nx = py+dy, px+dx
                            if 0<=ny<by and 0<=nx<w: vis[ny,nx] = [255,220,50]
            my, mx = p['mid']
            if my < by: cv2.circle(vis, (mx,my), 6, (0,255,255), 2)
        axes[3,col].imshow(vis)
        axes[3,col].set_title(f'Step 4: Selected({n_sel}) yellow + cyan=conn.pt', fontsize=9)

    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT / f'depth_trans_v3_{sid}.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved depth_trans_v3_{sid}.png')

print('\nDone!')
