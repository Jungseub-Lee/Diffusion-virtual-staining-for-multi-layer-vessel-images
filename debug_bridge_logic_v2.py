"""
Bridge debug v2:
- Binary masks shown as white
- EP classification: true EP vs bridge-needed EP
- Loose mask for bridge path checking
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

# === Tight masks (for skel) ===
r_mask_tight = remove_small_objects(
    cv2.morphologyEx((R>30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)
b_seed = B>50; b_low = B>20
b_lab, b_n = ndimage.label(b_low)
b_mask_tight = np.zeros_like(b_low, dtype=bool)
for i in range(1, b_n+1):
    if np.any(b_seed[b_lab==i]): b_mask_tight |= (b_lab==i)
b_mask_tight = remove_small_objects(
    cv2.morphologyEx(b_mask_tight.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2)>0, min_size=50)

# === Loose masks (for bridge path checking) ===
r_mask_loose = cv2.morphologyEx((R>15).astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=3) > 0
b_mask_loose = cv2.morphologyEx((B>10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=3) > 0

# === Vessel mask ===
vessel_all = remove_small_objects(
    cv2.morphologyEx((gray>10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=100)
vm = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

# === Skeletons ===
r_skel = skeletonize(r_mask_tight); r_filt = r_skel & (diff < 0)
b_skel = skeletonize(b_mask_tight); b_filt = b_skel & (diff > 0)

# === EP tangent ===
def get_eps_tangent(skel, h, w, tl=20):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    eps = np.argwhere(nb == 1); res = []
    for ep in eps:
        y0, x0 = ep; path_l = [(y0,x0)]; vis = {(y0,x0)}; cy, cx = y0, x0
        for _ in range(tl):
            f = False
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    if dy==0 and dx==0: continue
                    ny, nx = cy+dy, cx+dx
                    if 0<=ny<h and 0<=nx<w and skel[ny,nx] and (ny,nx) not in vis:
                        path_l.append((ny,nx)); vis.add((ny,nx)); cy,cx=ny,nx; f=True; break
                if f: break
            if not f: break
        if len(path_l) >= 5:
            n2 = min(len(path_l), 10)
            t = np.array(path_l[0], dtype=float) - np.array(path_l[n2-1], dtype=float)
            nm = np.linalg.norm(t)
            if nm > 0: t /= nm
            res.append({'pos': (y0,x0), 'tangent': t, 'path': path_l})
    return res

# === EP Classification ===
def classify_eps(eps, skel, tight_mask, dom_mask, by):
    """
    Classify EPs into:
    - 'true_ep': real vessel endpoint (sprout tip)
    - 'bridge_needed': should be connected (broken by crossing/mask edge)

    Criteria for bridge_needed:
    1. Near mask edge (tight_mask boundary within 5px)
    2. Near dominance boundary (dom_mask boundary within 5px)
    3. Short fragment (branch length < 15px)
    4. Near opposite channel crossing region
    """
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)

    # Precompute mask edge
    mask_edge = cv2.dilate(tight_mask.astype(np.uint8), k3) - tight_mask.astype(np.uint8)
    mask_edge_dil = cv2.dilate(mask_edge, k5) > 0  # 5px around mask edge

    # Dominance edge
    dom_edge = cv2.dilate(dom_mask.astype(np.uint8), k3) - dom_mask.astype(np.uint8)
    dom_edge_dil = cv2.dilate(dom_edge.astype(np.uint8), k5) > 0

    results = []
    for ep in eps:
        y0, x0 = ep['pos']
        if y0 >= by: continue

        branch_len = len(ep.get('path', []))
        near_mask_edge = bool(mask_edge_dil[y0, x0]) if 0<=y0<h and 0<=x0<w else False
        near_dom_edge = bool(dom_edge_dil[y0, x0]) if 0<=y0<h and 0<=x0<w else False
        is_short = branch_len < 15

        # Dominance strength at EP position
        dom_val = dom_mask[y0, x0] if 0<=y0<h and 0<=x0<w else False

        # Bridge needed if:
        # - near mask edge OR near dominance edge OR short fragment
        # AND the EP tangent points toward a region where mask continues
        reasons = []
        if near_mask_edge: reasons.append('mask_edge')
        if near_dom_edge: reasons.append('dom_edge')
        if is_short: reasons.append('short_frag')

        if len(reasons) > 0:
            ep_type = 'bridge_needed'
        else:
            ep_type = 'true_ep'

        results.append({
            'ep': ep,
            'type': ep_type,
            'reasons': reasons,
            'branch_len': branch_len,
            'near_mask_edge': near_mask_edge,
            'near_dom_edge': near_dom_edge,
            'is_short': is_short,
            'dom_at_ep': bool(dom_val),
        })

    return results

# === Run for both channels ===
for ch_name, skel_filt, tight_mask, loose_mask, dom_mask_raw, skel_color, mask_label in [
    ('R', r_filt, r_mask_tight, r_mask_loose, diff < 0, [230,70,70], 'R>30 + morphClose(k3,2)'),
    ('B', b_filt, b_mask_tight, b_mask_loose, diff > 0, [70,140,240], 'B hyst(50/20) + morphClose(k5,2)')]:

    eps = get_eps_tangent(skel_filt, h, w)
    classified = classify_eps(eps, skel_filt, tight_mask, dom_mask_raw, by)

    true_eps = [c for c in classified if c['type'] == 'true_ep']
    bridge_eps = [c for c in classified if c['type'] == 'bridge_needed']

    print(f"\n{'='*60}")
    print(f"{ch_name} channel: {len(eps)} total EPs in analysis zone → "
          f"true_ep={len(true_eps)} bridge_needed={len(bridge_eps)}")
    print(f"{'='*60}")

    for c in classified:
        ep = c['ep']; y0, x0 = ep['pos']
        r_str = ','.join(c['reasons']) if c['reasons'] else '-'
        print(f"  ({y0:>3d},{x0:>3d}) type={c['type']:>14s} branch={c['branch_len']:>3d}px "
              f"mask_edge={c['near_mask_edge']} dom_edge={c['near_dom_edge']} short={c['is_short']} "
              f"reasons=[{r_str}]")

    # === Figure: 3x2 ===
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle(f'{sid} — {ch_name} EP Classification\n'
                 f'Tight mask: {mask_label} | Loose mask: {"R>15" if ch_name=="R" else "B>10"} + morphClose(k5,3)',
                 fontsize=11, fontweight='bold')

    # Row 0: GT | Tight mask (white binary)
    axes[0,0].imshow(gt[:by]); axes[0,0].set_title('GT (multi-color)', fontsize=10)
    axes[0,1].imshow(tight_mask[:by], cmap='gray')
    axes[0,1].set_title(f'{ch_name} tight mask (binary) | {mask_label}', fontsize=9)

    # Row 1: Loose mask (white binary) | Dominance map
    axes[1,0].imshow(loose_mask[:by], cmap='gray')
    axes[1,0].set_title(f'{ch_name} loose mask (for bridge path)', fontsize=9)

    # Dominance: red=R>B, blue=B>R
    dom_vis = np.zeros((by,w,3), dtype=np.uint8)
    dom_vis[diff[:by] < 0] = [180, 60, 60]
    dom_vis[diff[:by] > 0] = [60, 100, 180]
    axes[1,1].imshow(dom_vis); axes[1,1].set_title('Dominance: red=R>B, blue=B>R', fontsize=9)

    # Row 2: Skel + classified EPs
    vis = np.zeros((by,w,3), dtype=np.uint8)
    # Show tight mask faintly
    vis[tight_mask[:by]] = [40, 40, 40]
    # Skel
    vis[cv2.dilate(skel_filt[:by].astype(np.uint8), dk2)>0] = skel_color

    # Draw EPs
    for c in classified:
        y0, x0 = c['ep']['pos']
        if y0 >= by: continue
        if c['type'] == 'true_ep':
            color = (0, 255, 0)  # green = true endpoint
            r = 5
        else:
            color = (255, 165, 0)  # orange = bridge needed
            r = 5
        cv2.circle(vis, (x0, y0), r, color, 2)
        # Tangent arrow
        ty, tx = c['ep']['tangent']
        ax_y, ax_x = int(y0 + ty*12), int(x0 + tx*12)
        cv2.arrowedLine(vis, (x0, y0), (ax_x, ax_y), color, 1, tipLength=0.3)

    axes[2,0].imshow(vis)
    axes[2,0].set_title(f'{ch_name} skel + EPs: green=true({len(true_eps)}) orange=bridge({len(bridge_eps)})', fontsize=9)

    # Same but on GT background
    vis2 = gt[:by].copy().astype(float) * 0.4
    vis2[cv2.dilate(skel_filt[:by].astype(np.uint8), dk4)>0] = skel_color
    for c in classified:
        y0, x0 = c['ep']['pos']
        if y0 >= by: continue
        color = (0, 255, 0) if c['type'] == 'true_ep' else (255, 165, 0)
        cv2.circle(vis2, (x0, y0), 6, color, 2)
    axes[2,1].imshow(vis2.astype(np.uint8))
    axes[2,1].set_title(f'{ch_name} skel on GT + classified EPs', fontsize=9)

    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT / f'ep_classify_{sid}_{ch_name}.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved ep_classify_{sid}_{ch_name}.png')

# === Also show tight vs loose mask comparison ===
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f'{sid} — Tight vs Loose Mask Comparison', fontsize=13, fontweight='bold')

axes[0,0].imshow(gt[:by]); axes[0,0].set_title('GT', fontsize=10)
axes[0,1].imshow(r_mask_tight[:by], cmap='gray'); axes[0,1].set_title('R tight (R>30)', fontsize=10)
axes[0,2].imshow(r_mask_loose[:by], cmap='gray'); axes[0,2].set_title('R loose (R>15) — for bridge path', fontsize=10)
axes[1,0].imshow(gt[:by]); axes[1,0].set_title('GT', fontsize=10)
axes[1,1].imshow(b_mask_tight[:by], cmap='gray'); axes[1,1].set_title('B tight (hyst 50/20)', fontsize=10)
axes[1,2].imshow(b_mask_loose[:by], cmap='gray'); axes[1,2].set_title('B loose (B>10) — for bridge path', fontsize=10)

for ax in axes.flat: ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / f'mask_compare_{sid}.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'\nSaved mask_compare_{sid}.png')
print('Done!')
