"""
Detailed bridge debug: show masks, EP candidates, bridge logic step by step.
Single sample (18-19-512) for deep inspection.
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
multi_dir = DATA / 'Multi-color'

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
dk2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

sid = '18-19-512'
gt_path = multi_dir / 'LSGAN' / f'{sid}_real_B.png'
gt = np.array(Image.open(gt_path))
gt_blur = cv2.GaussianBlur(gt, (7,7), 0)
h, w = gt.shape[:2]; by = int(h*0.8)
R = gt_blur[:,:,0].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
diff = B - R

# Build masks
r_mask = remove_small_objects(
    cv2.morphologyEx((R>30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)
b_seed = B>50; b_low = B>20
b_lab, b_n = ndimage.label(b_low)
b_mask = np.zeros_like(b_low, dtype=bool)
for i in range(1, b_n+1):
    if np.any(b_seed[b_lab==i]): b_mask |= (b_lab==i)
b_mask = remove_small_objects(
    cv2.morphologyEx(b_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2)>0, min_size=50)

vessel_all = remove_small_objects(
    cv2.morphologyEx((gray>10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=100)
vm = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

r_skel = skeletonize(r_mask); r_filt = r_skel & (diff < 0)
b_skel = skeletonize(b_mask); b_filt = b_skel & (diff > 0)

# Get EPs with tangent
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

# For both R and B, analyze EP-Line bridge candidates
for ch_name, skel_filt, ch_mask_img, dom_mask, skel_color in [
    ('R', r_filt, r_mask, diff < 0, [230,70,70]),
    ('B', b_filt, b_mask, diff > 0, [70,140,240])]:

    eps = get_eps_tangent(skel_filt, h, w)
    skel_coords = np.argwhere(skel_filt)

    print(f"\n{'='*60}")
    print(f"{ch_name} channel: {len(eps)} endpoints, skel={np.sum(skel_filt)} px")
    print(f"Mask condition: {'R>30, morphClose(k3,2), removeSmall(50)' if ch_name=='R' else 'B hyst(50/20), morphClose(k5,2), removeSmall(50)'}")
    print(f"Dominance: {'diff<0 (R>B)' if ch_name=='R' else 'diff>0 (B>R)'}")
    print(f"{'='*60}")

    # For each EP, find nearest skel pixel and check bridge conditions
    bridge_candidates = []
    for idx, ep in enumerate(eps):
        y0, x0 = ep['pos']
        if y0 >= by: continue  # skip baseline region
        ty, tx = ep['tangent']
        own_branch = set(ep.get('path', [])[:8])

        # Find nearest skel pixel
        best_dist = 26; best_target = None
        for sy, sx in skel_coords:
            if (int(sy), int(sx)) in own_branch: continue
            d = np.sqrt((sy-y0)**2 + (sx-x0)**2)
            if d < 3 or d > 25: continue
            if d < best_dist:
                best_dist = float(d)
                best_target = (int(sy), int(sx))

        if best_target is None:
            status = "NO_TARGET"
            reason = "No skel pixel within 25px radius"
            bridge_candidates.append({
                'ep': ep, 'idx': idx, 'target': None,
                'status': status, 'reason': reason, 'dist': None
            })
            continue

        # Build bezier path
        p1 = np.array([y0, x0], dtype=float)
        p2 = np.array(best_target, dtype=float)
        d = np.linalg.norm(p2-p1)
        c1 = p1 + np.array([ty, tx]) * d * 0.5
        c2 = p2
        n_pts = max(int(d*1.5), 8)
        path = []
        for t in np.linspace(0, 1, n_pts):
            p = (1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2
            path.append((int(round(p[0])), int(round(p[1]))))

        # Check conditions
        vm_count = sum(1 for (y,x) in path if 0<=y<h and 0<=x<w and vm[y,x])
        vm_ratio = vm_count / max(len(path), 1)
        ch_count = sum(1 for (y,x) in path if 0<=y<h and 0<=x<w and ch_mask_img[y,x])
        ch_ratio = ch_count / max(len(path), 1)
        dom_count = sum(1 for (y,x) in path if 0<=y<h and 0<=x<w and dom_mask[y,x])
        dom_ratio = dom_count / max(len(path), 1)

        # Consecutive off-channel check
        off_count = 0; max_off = 0
        for (py, px) in path:
            if 0<=py<h and 0<=px<w and not ch_mask_img[py, px]:
                off_count += 1; max_off = max(max_off, off_count)
            else:
                off_count = 0

        # Determine status
        if vm_ratio < 0.7:
            status = "FAIL_VM"; reason = f"vessel mask {vm_ratio:.0%} < 70%"
        elif ch_ratio < 0.7:
            status = "FAIL_CH"; reason = f"channel mask {ch_ratio:.0%} < 70%"
        elif max_off >= 3:
            status = "FAIL_CONSEC"; reason = f"consecutive {max_off}px off channel mask"
        elif dom_ratio < 0.3:
            status = "FAIL_DOM"; reason = f"dominance {dom_ratio:.0%} < 30%"
        else:
            status = "PASS"; reason = f"vm={vm_ratio:.0%} ch={ch_ratio:.0%} dom={dom_ratio:.0%}"

        bridge_candidates.append({
            'ep': ep, 'idx': idx, 'target': best_target,
            'status': status, 'reason': reason, 'dist': best_dist,
            'vm_ratio': vm_ratio, 'ch_ratio': ch_ratio, 'dom_ratio': dom_ratio,
            'max_off': max_off, 'path': path
        })

    # Print summary
    pass_count = sum(1 for c in bridge_candidates if c['status'] == 'PASS')
    fail_count = sum(1 for c in bridge_candidates if c['status'].startswith('FAIL'))
    no_target = sum(1 for c in bridge_candidates if c['status'] == 'NO_TARGET')
    print(f"\nEP candidates in analysis zone: {len(bridge_candidates)}")
    print(f"  PASS: {pass_count} | FAIL: {fail_count} | NO_TARGET: {no_target}")

    fail_reasons = {}
    for c in bridge_candidates:
        if c['status'].startswith('FAIL'):
            key = c['status']
            fail_reasons[key] = fail_reasons.get(key, 0) + 1
    for k, v in sorted(fail_reasons.items()):
        print(f"  {k}: {v}")

    print(f"\nDetailed EP list:")
    for c in bridge_candidates:
        ep = c['ep']
        y0, x0 = ep['pos']
        t_str = f"({ep['tangent'][0]:.2f},{ep['tangent'][1]:.2f})"
        if c['target']:
            print(f"  EP{c['idx']:>3d} ({y0:>3d},{x0:>3d}) tan={t_str} → target({c['target'][0]:>3d},{c['target'][1]:>3d}) "
                  f"d={c['dist']:.1f} {c['status']:>12s} | {c['reason']}")
        else:
            print(f"  EP{c['idx']:>3d} ({y0:>3d},{x0:>3d}) tan={t_str} → {c['status']:>12s} | {c['reason']}")

    # === Visualization: 2x2 ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'{sid} — {ch_name} Channel Bridge Debug\n'
                 f'Mask: {"R>30 + morphClose(k3,2)" if ch_name=="R" else "B hyst(50/20) + morphClose(k5,2)"} | '
                 f'Dom: {"R>B (diff<0)" if ch_name=="R" else "B>R (diff>0)"} | '
                 f'Bridge: vm>=70%, ch>=70%, consec<3, dom>=30%',
                 fontsize=11, fontweight='bold')

    # Panel 1: GT + mask overlay
    vis = gt[:by].copy().astype(float)
    mask_overlay = np.zeros((by, w, 3), dtype=float)
    mask_overlay[ch_mask_img[:by]] = skel_color
    vis = vis * 0.6 + mask_overlay * 0.4
    axes[0,0].imshow(vis.astype(np.uint8))
    axes[0,0].set_title(f'{ch_name} mask (colored) on GT | {np.sum(ch_mask_img[:by])} px', fontsize=10)

    # Panel 2: Skel + all EP candidates (color-coded by status)
    vis = np.zeros((by, w, 3), dtype=np.uint8)
    vis[cv2.dilate(skel_filt[:by].astype(np.uint8), dk2)>0] = skel_color
    # Show channel mask boundary faintly
    ch_edge = cv2.Canny(ch_mask_img[:by].astype(np.uint8)*255, 100, 200)
    vis[ch_edge > 0] = [60, 60, 60]

    for c in bridge_candidates:
        y0, x0 = c['ep']['pos']
        if y0 >= by: continue
        if c['status'] == 'PASS':
            color = (0, 255, 0)  # green = will bridge
        elif c['status'] == 'NO_TARGET':
            color = (100, 100, 100)  # gray = no target
        else:
            color = (255, 50, 50)  # red = failed condition
        cv2.circle(vis, (x0, y0), 4, color, -1)
        # Draw tangent arrow
        ty, tx = c['ep']['tangent']
        ax_y, ax_x = int(y0 + ty*10), int(x0 + tx*10)
        cv2.arrowedLine(vis, (x0, y0), (ax_x, ax_y), color, 1, tipLength=0.3)

    axes[0,1].imshow(vis)
    axes[0,1].set_title(f'EPs: green=PASS({pass_count}) red=FAIL({fail_count}) gray=NO_TARGET({no_target})', fontsize=9)

    # Panel 3: PASS bridges with bezier paths
    vis = np.zeros((by, w, 3), dtype=np.uint8)
    vis[cv2.dilate(skel_filt[:by].astype(np.uint8), dk2)>0] = skel_color
    # Show mask faintly
    vis[ch_mask_img[:by] & ~(cv2.dilate(skel_filt[:by].astype(np.uint8), dk2)>0)] = [30, 30, 30]

    for c in bridge_candidates:
        if c['status'] != 'PASS' or c.get('path') is None: continue
        for py, px in c['path']:
            if 0<=py<by and 0<=px<w:
                for dy in range(-1,2):
                    for dx in range(-1,2):
                        ny, nx = py+dy, px+dx
                        if 0<=ny<by and 0<=nx<w:
                            vis[ny, nx] = [255, 180, 50]  # orange bridge
        # Mark EP
        y0, x0 = c['ep']['pos']
        if y0 < by: cv2.circle(vis, (x0, y0), 4, (0, 255, 0), -1)
        # Mark target
        if c['target']:
            ty, tx = c['target']
            if ty < by: cv2.circle(vis, (tx, ty), 3, (255, 255, 255), -1)

    axes[1,0].imshow(vis)
    axes[1,0].set_title(f'PASS bridges: {pass_count} (orange=path, green=EP, white=target)', fontsize=9)

    # Panel 4: FAIL bridges with paths (show what was rejected)
    vis = np.zeros((by, w, 3), dtype=np.uint8)
    vis[cv2.dilate(skel_filt[:by].astype(np.uint8), dk2)>0] = skel_color
    vis[ch_mask_img[:by] & ~(cv2.dilate(skel_filt[:by].astype(np.uint8), dk2)>0)] = [30, 30, 30]

    for c in bridge_candidates:
        if not c['status'].startswith('FAIL') or c.get('path') is None: continue
        path_color = {
            'FAIL_VM': [255, 0, 0],
            'FAIL_CH': [255, 100, 0],
            'FAIL_CONSEC': [255, 200, 0],
            'FAIL_DOM': [200, 0, 200],
        }.get(c['status'], [128, 128, 128])
        for py, px in c['path']:
            if 0<=py<by and 0<=px<w:
                vis[py, px] = path_color
        y0, x0 = c['ep']['pos']
        if y0 < by: cv2.circle(vis, (x0, y0), 3, (255, 50, 50), -1)

    axes[1,1].imshow(vis)
    axes[1,1].set_title(f'FAIL bridges: red=VM, orange=CH, yellow=CONSEC, purple=DOM', fontsize=9)

    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT / f'bridge_logic_{sid}_{ch_name}.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved bridge_logic_{sid}_{ch_name}.png')

print('\nDone!')
