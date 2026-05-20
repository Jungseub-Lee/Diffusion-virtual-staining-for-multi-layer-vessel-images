"""
Depth transition bridge: R EP ↔ B EP through Yellow zone.
Step-by-step visualization for verification.
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
    c1 = p1 + np.array(e1_tan)*d*0.4
    c2 = p2 + np.array(e2_tan)*d*0.4
    n = max(int(d*1.5), 10)
    pts = []
    for t in np.linspace(0, 1, n):
        p = (1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2
        pts.append((int(round(p[0])), int(round(p[1]))))
    return pts

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
    hsv = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2HSV)
    hue = hsv[:,:,0].astype(float)
    sat = hsv[:,:,1].astype(float)
    diff = B - R

    # Masks (new pipeline v2)
    r_mask = remove_small_objects(
        cv2.morphologyEx((R>45).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)
    ratio = B / (R + B + 1)
    b_mask = remove_small_objects(
        cv2.morphologyEx(((ratio > 0.55) & (B > 15)).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)

    # Skels (no dominance cut)
    r_skel = skeletonize(r_mask)
    b_skel = skeletonize(b_mask)

    # Yellow zone: multiple criteria
    # Hue-based: hue 15-50, sat > 30
    yellow_hue = (hue >= 15) & (hue <= 50) & (sat > 30) & (gray > 15)
    # Also: R and B are similar (transition zone)
    yellow_ratio = (np.abs(diff) < 20) & (R > 20) & (B > 20)
    # Combined yellow zone
    yellow_zone = yellow_hue | yellow_ratio
    yellow_dilated = cv2.dilate(yellow_zone.astype(np.uint8)*255, k5) > 0

    # Get EPs
    r_eps = get_eps_tangent(r_skel, h, w)
    b_eps = get_eps_tangent(b_skel, h, w)
    # Filter to analysis zone
    r_eps = [e for e in r_eps if e['pos'][0] < by - 3]
    b_eps = [e for e in b_eps if e['pos'][0] < by - 3]

    print(f"\n{'='*60}")
    print(f"{sid}: R EPs={len(r_eps)}, B EPs={len(b_eps)}")

    # ============================================================
    # Step 1: Find all R-B EP pairs within 20px
    # ============================================================
    max_pair_dist = 25
    pairs = []
    for ri, rep in enumerate(r_eps):
        ry, rx = rep['pos']
        for bi, bep in enumerate(b_eps):
            by2, bx = bep['pos']
            d = np.sqrt((ry-by2)**2 + (rx-bx)**2)
            if d < max_pair_dist and d > 2:
                mid_y, mid_x = int((ry+by2)/2), int((rx+bx)/2)
                pairs.append({
                    'ri': ri, 'bi': bi,
                    'r_ep': rep, 'b_ep': bep,
                    'dist': d,
                    'mid': (mid_y, mid_x)
                })

    print(f"Step 1: {len(pairs)} R-B EP pairs within {max_pair_dist}px")
    for p in pairs:
        ry, rx = p['r_ep']['pos']; by2, bx = p['b_ep']['pos']
        print(f"  R({ry},{rx}) ↔ B({by2},{bx}) dist={p['dist']:.1f} mid=({p['mid'][0]},{p['mid'][1]})")

    # ============================================================
    # Step 2: Check yellow zone at midpoint and along path
    # ============================================================
    for p in pairs:
        my, mx = p['mid']
        # Check midpoint
        mid_on_yellow = bool(yellow_dilated[my, mx]) if 0<=my<h and 0<=mx<w else False
        # Check path: build bezier between R EP and B EP
        path = bezier_pts(p['r_ep']['pos'], p['r_ep']['tangent'],
                         p['b_ep']['pos'], p['b_ep']['tangent'])
        # Count yellow zone pixels along path
        yellow_count = sum(1 for (py,px) in path if 0<=py<h and 0<=px<w and yellow_dilated[py,px])
        yellow_ratio_path = yellow_count / max(len(path), 1)
        # Count vessel mask pixels
        vessel_all = remove_small_objects(
            cv2.morphologyEx((gray>10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=100)
        vm = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0
        vm_count = sum(1 for (py,px) in path if 0<=py<h and 0<=px<w and vm[py,px])
        vm_ratio_path = vm_count / max(len(path), 1)

        p['mid_on_yellow'] = mid_on_yellow
        p['yellow_ratio'] = yellow_ratio_path
        p['vm_ratio'] = vm_ratio_path
        p['path'] = path
        # Pass criteria: midpoint on yellow OR path >= 30% yellow, AND vessel mask >= 60%
        p['pass'] = (mid_on_yellow or yellow_ratio_path >= 0.3) and vm_ratio_path >= 0.6
        p['reason'] = f"mid_Y={mid_on_yellow} path_Y={yellow_ratio_path:.0%} vm={vm_ratio_path:.0%}"

        status = "PASS" if p['pass'] else "FAIL"
        ry, rx = p['r_ep']['pos']; by2, bx = p['b_ep']['pos']
        print(f"  {status}: R({ry},{rx})↔B({by2},{bx}) | {p['reason']}")

    passed = [p for p in pairs if p['pass']]
    failed = [p for p in pairs if not p['pass']]
    print(f"\nStep 2: PASS={len(passed)} FAIL={len(failed)}")

    # ============================================================
    # Step 3: Greedy selection (no EP reuse)
    # ============================================================
    passed.sort(key=lambda x: x['dist'])  # prefer shorter connections
    used_r = set(); used_b = set()
    selected = []
    for p in passed:
        if p['ri'] not in used_r and p['bi'] not in used_b:
            selected.append(p)
            used_r.add(p['ri']); used_b.add(p['bi'])

    print(f"Step 3: Selected {len(selected)} depth transition bridges")
    for p in selected:
        ry, rx = p['r_ep']['pos']; by2, bx = p['b_ep']['pos']
        print(f"  R({ry},{rx}) ↔ B({by2},{bx}) dist={p['dist']:.1f}")

    # ============================================================
    # VISUALIZATION: 4 panels
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{sid} — Depth Transition Bridge (R↔B through Yellow)',
                 fontsize=13, fontweight='bold')

    # Panel 1: GT + all pair candidates (lines between EPs)
    vis = gt[:by].copy().astype(float) * 0.5
    # Draw R skel
    vis[cv2.dilate(r_skel[:by].astype(np.uint8), dk2)>0] = [230,70,70]
    vis[cv2.dilate(b_skel[:by].astype(np.uint8), dk2)>0] = [70,140,240]
    vis = vis.astype(np.uint8)
    # Draw all pair candidates as dashed lines
    for p in pairs:
        ry, rx = p['r_ep']['pos']; by2, bx = p['b_ep']['pos']
        color = (0,255,0) if p['pass'] else (255,100,100)
        cv2.line(vis, (rx,ry), (bx,by2), color, 1, cv2.LINE_AA)
        cv2.circle(vis, (rx,ry), 4, (255,100,100), -1)  # R EP = red dot
        cv2.circle(vis, (bx,by2), 4, (100,150,255), -1)  # B EP = blue dot
    axes[0,0].imshow(vis)
    axes[0,0].set_title(f'Step 1: All pairs ({len(pairs)}) | green=pass, red=fail', fontsize=10)

    # Panel 2: Yellow zone mask
    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[yellow_hue[:by]] = [200, 200, 50]    # hue-based yellow
    vis[yellow_ratio[:by] & ~yellow_hue[:by]] = [150, 150, 50]  # ratio-based addition
    # Mark midpoints
    for p in pairs:
        my, mx = p['mid']
        if my < by:
            color = (0,255,0) if p['mid_on_yellow'] else (255,0,0)
            cv2.circle(vis, (mx,my), 5, color, 2)
    axes[0,1].imshow(vis)
    axes[0,1].set_title(f'Step 2: Yellow zone | midpoints (green=on yellow, red=off)', fontsize=9)

    # Panel 3: PASS vs FAIL bridges on GT
    vis = gt[:by].copy().astype(float) * 0.4
    vis[cv2.dilate(r_skel[:by].astype(np.uint8), dk2)>0] = [230,70,70]
    vis[cv2.dilate(b_skel[:by].astype(np.uint8), dk2)>0] = [70,140,240]
    vis = vis.astype(np.uint8)
    # Draw FAIL paths (thin red)
    for p in failed:
        for py, px in p.get('path', []):
            if 0<=py<by and 0<=px<w: vis[py,px] = [150,50,50]
    # Draw PASS paths (thick green)
    for p in passed:
        for py, px in p.get('path', []):
            if 0<=py<by and 0<=px<w:
                for dy in range(-1,2):
                    for dx in range(-1,2):
                        ny, nx = py+dy, px+dx
                        if 0<=ny<by and 0<=nx<w: vis[ny,nx] = [50,255,100]
    axes[0,2].imshow(vis)
    axes[0,2].set_title(f'Step 3: PASS({len(passed)}) green | FAIL({len(failed)}) dark red', fontsize=10)

    # Panel 4: Final selected bridges on GT
    vis = gt[:by].copy().astype(float) * 0.5
    vis = (vis * 0.6 + 255*0.4)
    vis[cv2.dilate(r_skel[:by].astype(np.uint8), dk4)>0] = [230,70,70]
    vis[cv2.dilate(b_skel[:by].astype(np.uint8), dk4)>0] = [70,140,240]
    vis = vis.astype(np.uint8)
    # Draw selected bridges (yellow = depth transition)
    for p in selected:
        for py, px in p['path']:
            if 0<=py<by and 0<=px<w:
                for dy in range(-1,2):
                    for dx in range(-1,2):
                        ny, nx = py+dy, px+dx
                        if 0<=ny<by and 0<=nx<w: vis[ny,nx] = [255,220,50]
        # Mark connection point at midpoint
        my, mx = p['mid']
        if my < by: cv2.circle(vis, (mx,my), 6, (0,255,255), 2)  # cyan = connection point
    axes[1,0].imshow(vis)
    axes[1,0].set_title(f'Step 4: Selected ({len(selected)}) yellow bridge, cyan=connection pt', fontsize=9)

    # Panel 5: Final combined skel (R+B+transitions) on black
    combined = r_skel[:by] | b_skel[:by]
    # Add bridge paths
    for p in selected:
        for py, px in p['path']:
            if 0<=py<by and 0<=px<w: combined[py,px] = True
    combined = skeletonize(combined[:by])
    vis = np.zeros((by,w,3), dtype=np.uint8)
    # Color by depth
    r_d = cv2.dilate(r_skel[:by].astype(np.uint8), dk2) > 0
    b_d = cv2.dilate(b_skel[:by].astype(np.uint8), dk2) > 0
    comb_d = cv2.dilate(combined.astype(np.uint8), dk2) > 0
    # Base: combined in gray
    vis[comb_d] = [150,150,150]
    # Overlay R and B colors
    vis[r_d] = [230,70,70]
    vis[b_d] = [70,140,240]
    vis[r_d & b_d] = [180,100,220]
    # Transition bridges in yellow
    for p in selected:
        for py, px in p['path']:
            if 0<=py<by and 0<=px<w:
                vis[py,px] = [255,220,50]
    axes[1,1].imshow(vis)
    axes[1,1].set_title(f'Combined skel (R+B+transition)', fontsize=10)

    # Panel 6: Summary text
    axes[1,2].axis('off')
    txt = (f"{sid} Summary\n\n"
           f"R EPs: {len(r_eps)}\n"
           f"B EPs: {len(b_eps)}\n"
           f"Pairs found (<{max_pair_dist}px): {len(pairs)}\n"
           f"Yellow zone check:\n"
           f"  PASS: {len(passed)}\n"
           f"  FAIL: {len(failed)}\n"
           f"Selected (greedy): {len(selected)}\n\n"
           f"Criteria:\n"
           f"  - R EP ↔ B EP dist < {max_pair_dist}px\n"
           f"  - Midpoint on yellow zone\n"
           f"    OR path ≥ 30% yellow\n"
           f"  - Path ≥ 60% on vessel mask\n"
           f"  - Greedy: shortest first,\n"
           f"    no EP reuse")
    axes[1,2].text(0.05, 0.95, txt, fontsize=10, fontfamily='monospace',
                   va='top', transform=axes[1,2].transAxes)

    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT / f'depth_transition_{sid}.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved depth_transition_{sid}.png')

print('\nDone!')
