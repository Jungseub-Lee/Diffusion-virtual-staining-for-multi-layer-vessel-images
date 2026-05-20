"""
Gradient-Contour-based Vessel Separation at Crossings (Sample 1 only).

Core idea:
- 혈관 crossing에서 hue gradient의 급격한 ridge = 겹친 혈관의 내부 외곽선
- 이 내부 ridge를 외곽 contour와 각도/곡률 연속성으로 자연스럽게 연결
- 연결된 경로로 겹친 혈관을 분리
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import label, gaussian_filter
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import warnings
warnings.filterwarnings('ignore')

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')
DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')
OUT.mkdir(exist_ok=True)

# ===== Load Sample 1 =====
s1 = np.array(Image.open(DATA / 'Original color stack data' / 'Sample 1_BF and GT(512x512 two images).png'))
gt = s1[512:]
h_img, w_img = gt.shape[:2]

hsv = cv2.cvtColor(gt, cv2.COLOR_RGB2HSV)
hue = hsv[:,:,0].astype(np.float32)
sat = hsv[:,:,1].astype(np.float32)
gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)

# ===== Step 1: Vessel mask =====
vessel_mask = (gray > 15) & (sat > 20)
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
vessel_mask = cv2.morphologyEx(vessel_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
vessel_mask = remove_small_objects(vessel_mask, min_size=100)

# ===== Step 2: Hue gradient (circular-aware) =====
hue_rad = hue * np.pi / 90.0
hue_cos = np.cos(hue_rad) * vessel_mask
hue_sin = np.sin(hue_rad) * vessel_mask

gc_x = cv2.Sobel(hue_cos, cv2.CV_64F, 1, 0, ksize=3)
gc_y = cv2.Sobel(hue_cos, cv2.CV_64F, 0, 1, ksize=3)
gs_x = cv2.Sobel(hue_sin, cv2.CV_64F, 1, 0, ksize=3)
gs_y = cv2.Sobel(hue_sin, cv2.CV_64F, 0, 1, ksize=3)

grad_mag = np.sqrt(gc_x**2 + gc_y**2 + gs_x**2 + gs_y**2) * vessel_mask
p97 = np.percentile(grad_mag[vessel_mask], 97)
grad_norm = np.clip(grad_mag / (p97 + 1e-6), 0, 1)
grad_smooth = gaussian_filter(grad_norm, sigma=1.5) * vessel_mask

# ===== Step 3: Ridge lines (internal boundaries) =====
ridge_thresh = 0.25
high_grad = (grad_smooth > ridge_thresh) & vessel_mask
high_grad = cv2.morphologyEx(high_grad.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3) > 0
high_grad = remove_small_objects(high_grad, min_size=15)
ridge_skel = skeletonize(high_grad)

# ===== Step 4: Outer contour =====
contour_img = np.zeros((h_img, w_img), dtype=np.uint8)
contours_list = cv2.findContours(vessel_mask.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
cv2.drawContours(contour_img, contours_list, -1, 255, 1)
outer_contour = contour_img > 0

# ===== Step 5: Interior ridges (not at boundary) =====
vessel_eroded = cv2.erode(vessel_mask.astype(np.uint8)*255, k3, iterations=3) > 0
interior_ridges = ridge_skel & vessel_eroded

# ===== Step 6: Ridge endpoint/junction detection =====
def skel_features(skel):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    endpoints = np.argwhere(nb == 1)
    junctions = np.argwhere(nb >= 3)
    return endpoints, junctions

def tangent_at(skel, pt, radius=10):
    """PCA-based tangent direction at a skeleton point."""
    y, x = pt
    y0, y1 = max(0, y-radius), min(skel.shape[0], y+radius+1)
    x0, x1 = max(0, x-radius), min(skel.shape[1], x+radius+1)
    pts = np.argwhere(skel[y0:y1, x0:x1]) + np.array([y0, x0])
    if len(pts) < 3:
        return None
    centered = pts - np.mean(pts, axis=0)
    cov = np.cov(centered.T)
    if cov.shape != (2,2):
        return None
    evals, evecs = np.linalg.eigh(cov)
    tangent = evecs[:, np.argmax(evals)]  # (dy, dx)
    # Orient: tangent should point outward from skeleton
    fwd = np.array([y, x]) + tangent * radius
    bwd = np.array([y, x]) - tangent * radius
    fwd_pts = np.sum(np.linalg.norm(pts - fwd, axis=1) < radius * 0.8)
    bwd_pts = np.sum(np.linalg.norm(pts - bwd, axis=1) < radius * 0.8)
    if fwd_pts > bwd_pts:
        tangent = -tangent
    return tangent

def trace_branch(skel, start, max_len=150):
    """Trace a skeleton branch from endpoint, return ordered points."""
    h, w = skel.shape
    visited = set()
    pts = [tuple(start)]
    visited.add(pts[0])
    cur = start
    for _ in range(max_len):
        y, x = cur
        nbs = []
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                if dy==0 and dx==0: continue
                ny, nx = y+dy, x+dx
                if 0<=ny<h and 0<=nx<w and skel[ny,nx] and (ny,nx) not in visited:
                    nbs.append((ny,nx))
        if not nbs:
            break
        nxt = nbs[0]
        visited.add(nxt)
        pts.append(nxt)
        cur = nxt
    return np.array(pts)

# ===== Step 7: Match ridge endpoints to contour with angle continuity =====
ridge_eps, ridge_jns = skel_features(ridge_skel)
contour_pts = np.argwhere(outer_contour)
contour_tree = cKDTree(contour_pts) if len(contour_pts) > 0 else None

connections = []
if contour_tree and len(ridge_eps) > 0:
    for ep in ridge_eps:
        dist, idx = contour_tree.query(ep, k=1)
        if dist > 25:
            continue
        cp = contour_pts[idx]

        # Tangent at ridge endpoint
        t_ridge = tangent_at(ridge_skel, ep, radius=12)
        # Tangent at contour point
        t_contour = tangent_at(outer_contour, cp, radius=12)

        # Direction from ep to cp
        direction = (cp - ep).astype(float)
        dir_len = np.linalg.norm(direction)
        if dir_len < 1:
            continue
        dir_unit = direction / dir_len

        # Alignment: how well does ridge tangent point toward contour?
        align = abs(np.dot(dir_unit, t_ridge / (np.linalg.norm(t_ridge)+1e-10))) if t_ridge is not None else 0.5

        # Angle continuity between ridge and contour tangents
        if t_ridge is not None and t_contour is not None:
            cos_a = np.clip(abs(np.dot(t_ridge, t_contour)) / (np.linalg.norm(t_ridge)*np.linalg.norm(t_contour)+1e-10), 0, 1)
            angle_score = cos_a
        else:
            angle_score = 0.5

        connections.append({
            'ridge_ep': ep, 'contour_pt': cp,
            'distance': dist, 'alignment': align, 'angle_score': angle_score,
            't_ridge': t_ridge, 't_contour': t_contour,
        })

# Filter: keep only good connections
good_conns = [c for c in connections if c['distance'] < 20 and c['alignment'] > 0.3]

# ===== Step 8: Build Bezier separation curves =====
def bezier_curve(p0, p1, t0, t1, n_pts=50):
    """Cubic Bezier from p0 to p1 with tangent directions t0, t1."""
    d = np.linalg.norm(p1 - p0)
    scale = d * 0.4  # control point distance
    c0 = p0 + t0 * scale if t0 is not None else p0 + (p1 - p0) * 0.33
    c1 = p1 - t1 * scale if t1 is not None else p1 - (p1 - p0) * 0.33
    t_vals = np.linspace(0, 1, n_pts)
    curve = (
        (1-t_vals[:,None])**3 * p0 +
        3*(1-t_vals[:,None])**2 * t_vals[:,None] * c0 +
        3*(1-t_vals[:,None]) * t_vals[:,None]**2 * c1 +
        t_vals[:,None]**3 * p1
    )
    return curve.astype(int)

separation_mask = np.zeros((h_img, w_img), dtype=bool)
# Add ridge skeleton as separator
separation_mask |= ridge_skel

# Add Bezier connections for good matches
for conn in good_conns:
    ep = conn['ridge_ep'].astype(float)
    cp = conn['contour_pt'].astype(float)
    t_r = conn['t_ridge']
    t_c = conn['t_contour']
    curve_pts = bezier_curve(ep, cp, t_r, t_c, n_pts=60)
    for pt in curve_pts:
        yy, xx = int(np.clip(pt[0], 0, h_img-1)), int(np.clip(pt[1], 0, w_img-1))
        separation_mask[yy, xx] = True

# Dilate separation line (thin but reliable)
sep_dilated = cv2.dilate(separation_mask.astype(np.uint8)*255,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))) > 0

# ===== Step 9: Split vessel using separation line =====
split_mask = vessel_mask & ~sep_dilated
split_labels, n_regions = label(split_mask)

r_mask = np.zeros((h_img, w_img), dtype=bool)
y_mask = np.zeros((h_img, w_img), dtype=bool)
b_mask = np.zeros((h_img, w_img), dtype=bool)

for rid in range(1, n_regions + 1):
    region = split_labels == rid
    if np.sum(region) < 30:
        continue
    rh = hue[region & (sat > 20)]
    if len(rh) == 0:
        continue
    rc = np.sum((rh <= 15) | (rh >= 150))
    yc = np.sum((rh >= 16) & (rh <= 55))
    bc = np.sum((rh >= 80) & (rh <= 145))
    if rc >= yc and rc >= bc:
        r_mask |= region
    elif bc >= yc:
        b_mask |= region
    else:
        y_mask |= region

r_mask = remove_small_objects(r_mask, min_size=50)
y_mask = remove_small_objects(y_mask, min_size=50)
b_mask = remove_small_objects(b_mask, min_size=50)

# ===== VISUALIZATION (Sample 1 only) =====
dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
ridge_dk = cv2.dilate(ridge_skel.astype(np.uint8), dk) > 0
contour_dk = cv2.dilate(outer_contour.astype(np.uint8), dk) > 0
sep_dk_vis = cv2.dilate(separation_mask.astype(np.uint8), dk) > 0

# --- Main overview figure ---
fig, axes = plt.subplots(3, 4, figsize=(28, 18))
fig.suptitle('Sample 1: Gradient-Contour Vessel Separation', fontsize=16, fontweight='bold')

# Row 1: Analysis
axes[0,0].imshow(gt); axes[0,0].set_title('Original GT')
im1 = axes[0,1].imshow(grad_smooth, cmap='hot', vmin=0, vmax=0.7)
plt.colorbar(im1, ax=axes[0,1], fraction=0.046, shrink=0.8)
axes[0,1].set_title('Hue Gradient\n(sharp=internal boundary)')

# Ridge + contour overlay
ov1 = gt.copy().astype(float) / 255
ov_rgba = np.zeros((*gt.shape[:2], 4))
ov_rgba[contour_dk] = [0, 1, 1, 0.6]
ov_rgba[ridge_dk] = [1, 0, 1, 0.8]
axes[0,2].imshow(ov1); axes[0,2].imshow(ov_rgba)
axes[0,2].set_title('Outer Contour (cyan)\n+ Ridge (magenta)')

# Connections on GT
axes[0,3].imshow(gt.astype(float)/255)
for conn in connections:
    ep, cp = conn['ridge_ep'], conn['contour_pt']
    good = conn['distance'] < 20 and conn['alignment'] > 0.3
    color = 'lime' if good else 'orangered'
    lw = 2.5 if good else 1
    axes[0,3].plot([ep[1], cp[1]], [ep[0], cp[0]], '-', color=color, lw=lw, alpha=0.8)
    axes[0,3].plot(ep[1], ep[0], 'o', color='magenta', ms=4, zorder=5)
    axes[0,3].plot(cp[1], cp[0], 's', color='cyan', ms=4, zorder=5)
    if good and conn['t_ridge'] is not None:
        t = conn['t_ridge'] * 12
        axes[0,3].annotate('', xy=(ep[1]+t[1], ep[0]+t[0]), xytext=(ep[1], ep[0]),
                          arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))
axes[0,3].set_title(f'Connections ({len(good_conns)} good)\ngreen=accepted, arrows=tangent')

# Row 2: Separation process
sep_vis = np.zeros((*gt.shape[:2], 3))
sep_vis[vessel_mask] = [0.3, 0.3, 0.3]
sep_vis[sep_dk_vis] = [1, 0, 0]
axes[1,0].imshow(sep_vis)
axes[1,0].set_title('Separation Lines on Vessel\n(ridge + Bezier connections)')

# Split regions
cmap_tab = plt.cm.tab20(np.linspace(0, 1, 20))
reg_vis = np.zeros((*gt.shape[:2], 3))
for i in range(1, min(n_regions+1, 100)):
    reg_vis[split_labels == i] = cmap_tab[i % 20][:3]
axes[1,1].imshow(reg_vis)
axes[1,1].set_title(f'Split Regions ({n_regions})')

# Depth masks
mask_vis = np.zeros_like(gt)
mask_vis[r_mask] = [230, 70, 70]
mask_vis[y_mask] = [240, 220, 50]
mask_vis[b_mask] = [70, 130, 240]
axes[1,2].imshow(mask_vis)
axes[1,2].set_title('Separated Depth Masks (R/Y/B)')

# Overlay on GT
ov2 = gt.copy().astype(float) * 0.4
ov2[r_mask] = np.clip(ov2[r_mask]*0.3 + np.array([230,70,70])*0.7, 0, 255)
ov2[y_mask] = np.clip(ov2[y_mask]*0.3 + np.array([240,220,50])*0.7, 0, 255)
ov2[b_mask] = np.clip(ov2[b_mask]*0.3 + np.array([70,130,240])*0.7, 0, 255)
axes[1,3].imshow(ov2.astype(np.uint8))
axes[1,3].set_title('Masks on GT')

# Row 3: Skeletons + comparison with old
sk_r = skeletonize(r_mask) if np.any(r_mask) else np.zeros_like(r_mask)
sk_y = skeletonize(y_mask) if np.any(y_mask) else np.zeros_like(y_mask)
sk_b = skeletonize(b_mask) if np.any(b_mask) else np.zeros_like(b_mask)
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

skel_gt = gt.copy().astype(float) * 0.5
skel_gt[cv2.dilate(sk_r.astype(np.uint8), dk4) > 0] = [230, 70, 70]
skel_gt[cv2.dilate(sk_y.astype(np.uint8), dk4) > 0] = [240, 220, 50]
skel_gt[cv2.dilate(sk_b.astype(np.uint8), dk4) > 0] = [70, 130, 240]
axes[2,0].imshow(skel_gt.astype(np.uint8))
axes[2,0].set_title('New: Skeletons on GT')

# Old method
fg_old = (gray > 15) & (sat > 30)
r_old = ((hue<=12)|(hue>=155)) & fg_old
y_old = ((hue>=13)&(hue<=55)) & fg_old
b_old = ((hue>=85)&(hue<=140)) & fg_old
kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
r_old = cv2.morphologyEx(r_old.astype(np.uint8)*255, cv2.MORPH_CLOSE, kk, iterations=2) > 0
y_old = cv2.morphologyEx(y_old.astype(np.uint8)*255, cv2.MORPH_CLOSE, kk, iterations=2) > 0
b_old = cv2.morphologyEx(b_old.astype(np.uint8)*255, cv2.MORPH_CLOSE, kk, iterations=2) > 0
sk_r_o = skeletonize(r_old>0); sk_y_o = skeletonize(y_old>0); sk_b_o = skeletonize(b_old>0)

old_gt = gt.copy().astype(float) * 0.5
old_gt[cv2.dilate(sk_r_o.astype(np.uint8), dk4) > 0] = [230, 70, 70]
old_gt[cv2.dilate(sk_y_o.astype(np.uint8), dk4) > 0] = [240, 220, 50]
old_gt[cv2.dilate(sk_b_o.astype(np.uint8), dk4) > 0] = [70, 130, 240]
axes[2,1].imshow(old_gt.astype(np.uint8))
axes[2,1].set_title('Old: Skeletons on GT')

# Metrics
crop_y = int(h_img * 0.8)
vm = np.ones((h_img, w_img), dtype=bool); vm[crop_y:] = False
def metrics(skels):
    j, l, e = 0, 0, 0
    for sk in skels:
        eps, jns = skel_features(sk)
        j += np.sum(jns[:,0] < crop_y) if len(jns) > 0 else 0
        l += np.sum(sk & vm)
        e += np.sum(eps[:,0] < crop_y) if len(eps) > 0 else 0
    return j, l, e

j_n, l_n, e_n = metrics([sk_r, sk_y, sk_b])
j_o, l_o, e_o = metrics([sk_r_o, sk_y_o, sk_b_o])

axes[2,2].axis('off')
axes[2,2].text(0.1, 0.7, f'Gradient-Contour:\n  J={j_n}  L={l_n}  E={e_n}', fontsize=14,
               transform=axes[2,2].transAxes, fontfamily='monospace', color='lime')
axes[2,2].text(0.1, 0.3, f'Old (hue-only):\n  J={j_o}  L={l_o}  E={e_o}', fontsize=14,
               transform=axes[2,2].transAxes, fontfamily='monospace', color='orange')
axes[2,2].set_title('Metrics Comparison')
axes[2,2].set_facecolor('black')

axes[2,3].axis('off')

for ax_row in axes:
    for ax in ax_row:
        ax.axis('off')

plt.tight_layout()
plt.savefig(OUT / 'gradient_contour_separation_sample1.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved gradient_contour_separation_sample1.png')

# --- Crossing zoom panels ---
cross_labels, n_cross = label(interior_ridges)
sizes = [(i, np.sum(cross_labels==i)) for i in range(1, n_cross+1) if np.sum(cross_labels==i) > 5]
sizes.sort(key=lambda x: -x[1])
n_zoom = min(4, len(sizes))

if n_zoom > 0:
    fig2, axes2 = plt.subplots(n_zoom, 6, figsize=(36, 6*n_zoom))
    if n_zoom == 1: axes2 = axes2[np.newaxis, :]
    fig2.suptitle('Sample 1: Crossing Zoom — Ridge→Contour→Separation', fontsize=14, fontweight='bold')

    for zi in range(n_zoom):
        idx = sizes[zi][0]
        ys_c, xs_c = np.where(cross_labels == idx)
        cy, cx = int(np.mean(ys_c)), int(np.mean(xs_c))
        pad = 80
        y0, y1 = max(0, cy-pad), min(h_img, cy+pad)
        x0, x1 = max(0, cx-pad), min(w_img, cx+pad)
        sl = (slice(y0,y1), slice(x0,x1))

        axes2[zi,0].imshow(gt[sl]); axes2[zi,0].set_title(f'Crossing {zi+1}: GT')
        axes2[zi,1].imshow(grad_smooth[sl], cmap='hot', vmin=0, vmax=0.7)
        axes2[zi,1].set_title('Hue Gradient')

        # Ridge + contour
        rv = np.zeros((y1-y0, x1-x0, 3))
        rv[vessel_mask[sl]] = [0.3, 0.3, 0.3]
        lr = cv2.dilate(ridge_skel[sl].astype(np.uint8), dk) > 0
        lc = cv2.dilate(outer_contour[sl].astype(np.uint8), dk) > 0
        rv[lr] = [1, 0, 1]; rv[lc] = [0, 1, 1]
        axes2[zi,2].imshow(rv); axes2[zi,2].set_title('Ridge(mag) + Contour(cyan)')

        # Connections in this region
        axes2[zi,3].imshow(gt[sl].astype(float)/255)
        for conn in connections:
            ep, cp = conn['ridge_ep'], conn['contour_pt']
            if y0 <= ep[0] < y1 and x0 <= ep[1] < x1:
                good = conn['distance'] < 20 and conn['alignment'] > 0.3
                color = 'lime' if good else 'orangered'
                axes2[zi,3].plot([ep[1]-x0, cp[1]-x0], [ep[0]-y0, cp[0]-y0], '-', color=color, lw=2)
                axes2[zi,3].plot(ep[1]-x0, ep[0]-y0, 'o', color='magenta', ms=5)
                # Bezier curve visualization
                if good and conn['t_ridge'] is not None:
                    bz = bezier_curve(ep.astype(float), cp.astype(float),
                                      conn['t_ridge'], conn['t_contour'], n_pts=30)
                    axes2[zi,3].plot(bz[:,1]-x0, bz[:,0]-y0, '--', color='yellow', lw=1.5)
        axes2[zi,3].set_title('Connections + Bezier')

        # New result
        lm = np.zeros((y1-y0, x1-x0, 3))
        lm[r_mask[sl]] = [0.9,0.3,0.3]; lm[y_mask[sl]] = [0.9,0.9,0.2]; lm[b_mask[sl]] = [0.3,0.5,0.9]
        axes2[zi,4].imshow(lm); axes2[zi,4].set_title('New: Separated')

        # Old result
        lo = np.zeros((y1-y0, x1-x0, 3))
        lo[r_old[sl]] = [0.9,0.3,0.3]; lo[y_old[sl]] = [0.9,0.9,0.2]; lo[b_old[sl]] = [0.3,0.5,0.9]
        axes2[zi,5].imshow(lo); axes2[zi,5].set_title('Old: Hue-only')

        for ax in axes2[zi]: ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUT / 'gradient_contour_crossings_zoom.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved gradient_contour_crossings_zoom.png')

print(f'\nMetrics: New J={j_n} L={l_n} E={e_n} | Old J={j_o} L={l_o} E={e_o}')
print('Done!')
