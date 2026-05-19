"""
Watershed-based depth separation.
Logic:
1. Compute hue gradient within vessel mask → internal boundaries
2. Use gradient as barrier for watershed segmentation
3. Each sub-region gets consistent depth assignment
4. Merge same-depth adjacent regions → clean per-depth masks
5. Crossings naturally separate because gradient barrier splits them
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import label, gaussian_filter, binary_dilation, binary_erosion
import warnings
warnings.filterwarnings('ignore')

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')
DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')

# Load Sample 1 GT
s1 = np.array(Image.open(DATA / 'Original color stack data' / 'Sample 1_BF and GT(512x512 two images).png'))
gt = s1[512:]

# Also test on actual dataset samples
gt_dir = DATA / 'Multi-color' / 'Pix2pix'

def watershed_depth_separation(img, debug_name=None):
    """
    Separate overlapping vessels by depth using watershed on hue gradient.
    Returns: r_mask, y_mask, b_mask (clean, crossing-aware)
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv[:,:,0].astype(np.float32)
    sat = hsv[:,:,1].astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Step 1: Vessel mask
    vessel = (gray > 15) & (sat > 20)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    vessel = cv2.morphologyEx(vessel.astype(np.uint8)*255, cv2.MORPH_CLOSE, k, iterations=2) > 0
    vessel = remove_small_objects(vessel, min_size=100)

    # Step 2: Hue gradient (circular-aware)
    hue_rad = hue * np.pi / 90.0
    hue_cos = np.cos(hue_rad) * vessel
    hue_sin = np.sin(hue_rad) * vessel

    # Use Sobel for gradient
    gc_x = cv2.Sobel(hue_cos, cv2.CV_64F, 1, 0, ksize=3)
    gc_y = cv2.Sobel(hue_cos, cv2.CV_64F, 0, 1, ksize=3)
    gs_x = cv2.Sobel(hue_sin, cv2.CV_64F, 1, 0, ksize=3)
    gs_y = cv2.Sobel(hue_sin, cv2.CV_64F, 0, 1, ksize=3)

    grad_mag = np.sqrt(gc_x**2 + gc_y**2 + gs_x**2 + gs_y**2)
    grad_mag = grad_mag * vessel

    # Normalize
    if np.max(grad_mag) > 0:
        grad_norm = grad_mag / np.percentile(grad_mag[vessel], 97)
        grad_norm = np.clip(grad_norm, 0, 1)
    else:
        grad_norm = grad_mag

    # Smooth slightly
    grad_smooth = gaussian_filter(grad_norm, sigma=1.5) * vessel

    # Step 3: Create markers for watershed
    # Seeds = regions with LOW gradient (homogeneous hue interiors)
    # Barriers = regions with HIGH gradient (depth boundaries)
    barrier_thresh = 0.25
    barrier = grad_smooth > barrier_thresh
    barrier = barrier & vessel

    # Erode vessel mask with barrier removed → seed regions
    seed_mask = vessel & ~barrier
    seed_mask = cv2.morphologyEx(seed_mask.astype(np.uint8)*255, cv2.MORPH_OPEN, k, iterations=1) > 0
    seed_mask = remove_small_objects(seed_mask, min_size=30)

    # Label seed regions
    seed_labels, n_seeds = label(seed_mask)

    # Step 4: Watershed using gradient as elevation map
    elevation = (grad_smooth * 255).astype(np.uint8)
    # Invert: low gradient = low elevation (basins), high gradient = high elevation (ridges)
    ws_labels = watershed(elevation, markers=seed_labels, mask=vessel)

    # Step 5: Assign each watershed region a depth based on dominant hue
    region_depth = {}
    for region_id in range(1, n_seeds + 1):
        region_pixels = ws_labels == region_id
        if np.sum(region_pixels) < 10:
            continue

        # Get hues in this region
        region_hues = hue[region_pixels & (sat > 20)]
        if len(region_hues) == 0:
            region_depth[region_id] = 'unknown'
            continue

        r_count = np.sum((region_hues <= 15) | (region_hues >= 150))
        y_count = np.sum((region_hues >= 16) & (region_hues <= 55))
        b_count = np.sum((region_hues >= 80) & (region_hues <= 145))

        total = r_count + y_count + b_count
        if total == 0:
            region_depth[region_id] = 'unknown'
        elif r_count >= y_count and r_count >= b_count:
            region_depth[region_id] = 'red'
        elif b_count >= y_count:
            region_depth[region_id] = 'blue'
        else:
            region_depth[region_id] = 'yellow'

    # Step 6: Build per-depth masks
    r_mask = np.zeros((h, w), dtype=bool)
    y_mask = np.zeros((h, w), dtype=bool)
    b_mask = np.zeros((h, w), dtype=bool)

    for region_id, depth in region_depth.items():
        region = ws_labels == region_id
        if depth == 'red':
            r_mask |= region
        elif depth == 'yellow':
            y_mask |= region
        elif depth == 'blue':
            b_mask |= region

    # Clean up
    r_mask = remove_small_objects(r_mask, min_size=50)
    y_mask = remove_small_objects(y_mask, min_size=50)
    b_mask = remove_small_objects(b_mask, min_size=50)

    debug = {
        'vessel': vessel,
        'grad_smooth': grad_smooth,
        'barrier': barrier,
        'seed_labels': seed_labels,
        'ws_labels': ws_labels,
        'n_seeds': n_seeds,
        'region_depth': region_depth,
    }

    return r_mask, y_mask, b_mask, debug

# ===== Test on Sample 1 =====
print("Processing Sample 1...")
r, y, b, dbg = watershed_depth_separation(gt, "sample1")

# Visualize
fig, axes = plt.subplots(3, 4, figsize=(28, 18))
fig.suptitle('Watershed Depth Separation (from projection only)', fontsize=16, fontweight='bold')

# Row 1: Process
axes[0,0].imshow(gt); axes[0,0].set_title('Original GT', fontsize=12)

im1 = axes[0,1].imshow(dbg['grad_smooth'], cmap='hot', vmin=0, vmax=0.6)
plt.colorbar(im1, ax=axes[0,1], fraction=0.046, shrink=0.8)
axes[0,1].set_title('Hue Gradient\n(high = depth boundary)', fontsize=12)

# Barrier visualization
barrier_vis = np.zeros((*gt.shape[:2], 3), dtype=np.uint8)
barrier_vis[dbg['vessel'] & ~dbg['barrier']] = [80, 80, 80]
barrier_vis[dbg['barrier']] = [255, 255, 0]
axes[0,2].imshow(barrier_vis)
axes[0,2].set_title(f'Barrier (yellow) on vessel\nSplits crossing zones', fontsize=12)

# Seed regions
seed_vis = np.zeros((*gt.shape[:2], 3), dtype=np.uint8)
cmap = plt.cm.tab20(np.linspace(0, 1, min(20, dbg['n_seeds'])))
for i in range(1, min(dbg['n_seeds']+1, 200)):
    color = (np.array(cmap[i % 20][:3]) * 255).astype(np.uint8)
    seed_vis[dbg['seed_labels'] == i] = color
axes[0,3].imshow(seed_vis)
axes[0,3].set_title(f'Seed Regions ({dbg["n_seeds"]})\n(homogeneous hue zones)', fontsize=12)

# Row 2: Results
# Watershed regions colored by depth
ws_depth_vis = np.zeros((*gt.shape[:2], 3), dtype=np.uint8)
depth_colors = {'red': (230, 70, 70), 'yellow': (240, 220, 50), 'blue': (70, 130, 240), 'unknown': (100, 100, 100)}
for rid, depth in dbg['region_depth'].items():
    ws_depth_vis[dbg['ws_labels'] == rid] = depth_colors.get(depth, (100, 100, 100))
axes[1,0].imshow(ws_depth_vis)
axes[1,0].set_title('Watershed Regions\n(colored by depth)', fontsize=12)

# Final masks
mask_vis = np.zeros_like(gt)
mask_vis[r] = [230, 70, 70]
mask_vis[y] = [240, 220, 50]
mask_vis[b] = [70, 130, 240]
axes[1,1].imshow(mask_vis)
rb_overlap = r & b
axes[1,1].set_title(f'Final Depth Masks\nR∩B overlap: {np.sum(rb_overlap)}px', fontsize=12)

# Overlay on GT
overlay = gt.copy().astype(np.float64) * 0.4
overlay[r] = np.clip(overlay[r] * 0.5 + np.array([230, 70, 70]) * 0.5, 0, 255)
overlay[y] = np.clip(overlay[y] * 0.5 + np.array([240, 220, 50]) * 0.5, 0, 255)
overlay[b] = np.clip(overlay[b] * 0.5 + np.array([70, 130, 240]) * 0.5, 0, 255)
axes[1,2].imshow(overlay.astype(np.uint8))
axes[1,2].set_title('Depth Masks on GT', fontsize=12)

# Skeleton comparison
dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
sk_r, sk_y, sk_b = skeletonize(r), skeletonize(y), skeletonize(b)
skel_vis = np.zeros_like(gt)
skel_vis[cv2.dilate(sk_r.astype(np.uint8), dk) > 0] = [230, 70, 70]
skel_vis[cv2.dilate(sk_y.astype(np.uint8), dk) > 0] = [240, 220, 50]
skel_vis[cv2.dilate(sk_b.astype(np.uint8), dk) > 0] = [70, 130, 240]
axes[1,3].imshow(skel_vis)
axes[1,3].set_title('Depth-separated Skeletons', fontsize=12)

# Row 3: Comparison with old method and single
# Old exclusive hue method
hsv_old = cv2.cvtColor(gt, cv2.COLOR_RGB2HSV)
h_old, s_old = hsv_old[:,:,0], hsv_old[:,:,1]
gray_old = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
fg_old = (gray_old > 15) & (s_old > 30)
r_old = ((h_old <= 12) | (h_old >= 155)) & fg_old
y_old = ((h_old >= 13) & (h_old <= 55)) & fg_old
b_old = ((h_old >= 85) & (h_old <= 140)) & fg_old
kk2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
r_old = cv2.morphologyEx(r_old.astype(np.uint8)*255, cv2.MORPH_CLOSE, kk2, iterations=2) > 0
y_old = cv2.morphologyEx(y_old.astype(np.uint8)*255, cv2.MORPH_CLOSE, kk2, iterations=2) > 0
b_old = cv2.morphologyEx(b_old.astype(np.uint8)*255, cv2.MORPH_CLOSE, kk2, iterations=2) > 0

old_mask = np.zeros_like(gt)
old_mask[r_old] = [230, 70, 70]; old_mask[y_old] = [240, 220, 50]; old_mask[b_old] = [70, 130, 240]
axes[2,0].imshow(old_mask)
axes[2,0].set_title('Old Method (exclusive hue)', fontsize=12)

sk_r_old, sk_y_old, sk_b_old = skeletonize(r_old>0), skeletonize(y_old>0), skeletonize(b_old>0)
old_skel_vis = np.zeros_like(gt)
old_skel_vis[cv2.dilate(sk_r_old.astype(np.uint8), dk) > 0] = [230, 70, 70]
old_skel_vis[cv2.dilate(sk_y_old.astype(np.uint8), dk) > 0] = [240, 220, 50]
old_skel_vis[cv2.dilate(sk_b_old.astype(np.uint8), dk) > 0] = [70, 130, 240]
axes[2,1].imshow(old_skel_vis)
axes[2,1].set_title('Old Skeletons', fontsize=12)

# Single color
vessel_single = gray_old > 15
vessel_single = cv2.morphologyEx(vessel_single.astype(np.uint8)*255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=2) > 0
vessel_single = remove_small_objects(vessel_single > 0, min_size=100)
sk_single = skeletonize(vessel_single)
single_vis = np.zeros_like(gt)
single_vis[cv2.dilate(sk_single.astype(np.uint8), dk) > 0] = [0, 200, 0]
axes[2,2].imshow(single_vis)

def find_features(skel):
    su = skel.astype(np.uint8)
    ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ks) * su
    return np.argwhere(nb == 1), np.argwhere(nb >= 3)

crop_y = int(gt.shape[0] * 0.8)
vm = np.ones(gt.shape[:2], dtype=bool); vm[crop_y:] = False

# Count metrics
def count_metrics(skels, vm):
    j = sum(np.sum(find_features(sk)[1][:, 0] < crop_y) if len(find_features(sk)[1]) > 0 else 0 for sk in skels)
    l = sum(np.sum(sk & vm) for sk in skels)
    e = sum(np.sum(find_features(sk)[0][:, 0] < crop_y) if len(find_features(sk)[0]) > 0 else 0 for sk in skels)
    return j, l, e

j_ws, l_ws, e_ws = count_metrics([sk_r, sk_y, sk_b], vm)
j_old, l_old, e_old = count_metrics([sk_r_old, sk_y_old, sk_b_old], vm)

ep_s, jn_s = find_features(sk_single)
j_single = np.sum(jn_s[:, 0] < crop_y) if len(jn_s) > 0 else 0
l_single = np.sum(sk_single & vm)
e_single = np.sum(ep_s[:, 0] < crop_y) if len(ep_s) > 0 else 0

axes[2,2].set_title(f'Single-color\nJ={j_single}, L={l_single}, E={e_single}', fontsize=12)

# Overlay all skeletons on GT
ov_ws = gt.copy().astype(np.float64) * 0.5
ov_ws[cv2.dilate(sk_r.astype(np.uint8), dk) > 0] = [230, 70, 70]
ov_ws[cv2.dilate(sk_y.astype(np.uint8), dk) > 0] = [240, 220, 50]
ov_ws[cv2.dilate(sk_b.astype(np.uint8), dk) > 0] = [70, 130, 240]
axes[2,3].imshow(ov_ws.astype(np.uint8))
axes[2,3].set_title(f'Watershed Skel on GT\nJ={j_ws}, L={l_ws}, E={e_ws}', fontsize=12)

print(f'\nMetrics comparison (above baseline):')
print(f'  Single-color:  J={j_single:4d}, L={l_single:5d}, E={e_single:3d}')
print(f'  Old (excl hue): J={j_old:4d}, L={l_old:5d}, E={e_old:3d}')
print(f'  Watershed:     J={j_ws:4d}, L={l_ws:5d}, E={e_ws:3d}')

for ax_row in axes:
    for ax in ax_row:
        ax.axhline(crop_y, color='yellow', lw=1, ls='--', alpha=0.5)
        ax.axis('off')

plt.tight_layout()
plt.savefig(OUT / 'watershed_depth_sample1.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('\nSaved watershed_depth_sample1.png')

# ===== Test on dataset samples =====
test_samples = ['18-19-512', '1-16-512', '16-18-716']
fig2, axes2 = plt.subplots(len(test_samples), 5, figsize=(30, 6*len(test_samples)))
fig2.suptitle('Watershed vs Old vs Single on Dataset Samples', fontsize=14, fontweight='bold')

for ri, sid in enumerate(test_samples):
    img = np.array(Image.open(gt_dir / f'{sid}_real_B.png'))
    h_img, w_img = img.shape[:2]
    cy = int(h_img * 0.8)
    vm_s = np.ones((h_img, w_img), dtype=bool); vm_s[cy:] = False

    r_ws, y_ws, b_ws, _ = watershed_depth_separation(img)
    sk_ws = [skeletonize(m) for m in [r_ws, y_ws, b_ws]]

    # Old method
    hsv_s = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_s, s_s = hsv_s[:,:,0], hsv_s[:,:,1]
    g_s = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fg_s = (g_s > 15) & (s_s > 30)
    masks_old = [((h_s<=12)|(h_s>=155))&fg_s, ((h_s>=13)&(h_s<=55))&fg_s, ((h_s>=85)&(h_s<=140))&fg_s]
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    masks_old = [cv2.morphologyEx(m.astype(np.uint8)*255, cv2.MORPH_CLOSE, kk, iterations=2) > 0 for m in masks_old]
    sk_old_s = [skeletonize(m > 0) for m in masks_old]

    # Single
    v_s = remove_small_objects((g_s > 15), min_size=100)
    v_s = cv2.morphologyEx(v_s.astype(np.uint8)*255, kk, iterations=2) > 0
    sk_ss = skeletonize(v_s > 0)

    j_ws_s, l_ws_s, _ = count_metrics(sk_ws, vm_s)
    j_old_s, l_old_s, _ = count_metrics(sk_old_s, vm_s)
    ep_ss, jn_ss = find_features(sk_ss)
    j_ss = np.sum(jn_ss[:,0] < cy) if len(jn_ss) > 0 else 0
    l_ss = np.sum(sk_ss & vm_s)

    # Vis
    axes2[ri,0].imshow(img); axes2[ri,0].set_title(f'{sid}\nOriginal GT', fontsize=10)

    single_v = np.zeros_like(img)
    single_v[cv2.dilate(sk_ss.astype(np.uint8), dk) > 0] = [0, 200, 0]
    axes2[ri,1].imshow(single_v); axes2[ri,1].set_title(f'Single J={j_ss} L={l_ss}', fontsize=10)

    old_v = np.zeros_like(img)
    for si, sk in enumerate(sk_old_s):
        old_v[cv2.dilate(sk.astype(np.uint8), dk) > 0] = [(230,70,70),(240,220,50),(70,130,240)][si]
    axes2[ri,2].imshow(old_v); axes2[ri,2].set_title(f'Old Multi J={j_old_s} L={l_old_s}', fontsize=10)

    ws_v = np.zeros_like(img)
    for si, sk in enumerate(sk_ws):
        ws_v[cv2.dilate(sk.astype(np.uint8), dk) > 0] = [(230,70,70),(240,220,50),(70,130,240)][si]
    axes2[ri,3].imshow(ws_v); axes2[ri,3].set_title(f'Watershed J={j_ws_s} L={l_ws_s}', fontsize=10)

    # Watershed masks
    ws_mask = np.zeros_like(img)
    ws_mask[r_ws] = [180, 50, 50]; ws_mask[y_ws] = [180, 180, 50]; ws_mask[b_ws] = [50, 100, 200]
    axes2[ri,4].imshow(ws_mask); axes2[ri,4].set_title('Watershed Masks', fontsize=10)

    print(f'{sid}: Single J={j_ss} L={l_ss} | Old J={j_old_s} L={l_old_s} | WS J={j_ws_s} L={l_ws_s}')

    for ax in axes2[ri]:
        ax.axhline(cy, color='yellow', lw=1, ls='--', alpha=0.5)
        ax.axis('off')

plt.tight_layout()
plt.savefig(OUT / 'watershed_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('\nSaved watershed_comparison.png')
