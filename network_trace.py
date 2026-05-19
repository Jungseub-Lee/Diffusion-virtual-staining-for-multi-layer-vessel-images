"""
Network-level vessel tracing from a single projected multi-color image.
No 8-layer stack needed - works purely from the 2D projection.

Approach:
1. Overall vessel mask → skeletonize → network graph
2. Split skeleton into branches (segments between junctions)
3. Each branch gets average hue → depth assignment
4. At junctions, match branches by color continuity
5. Separate crossing vessels from connected vessels
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import label
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')
OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')

def load(p): return np.array(Image.open(p).convert('RGB'))

def get_vessel_mask_color(img, min_area=100):
    """Get vessel mask from multi-color image."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = (gray > 15) & (hsv[:,:,1] > 20)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    u = cv2.morphologyEx(mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k, iterations=2)
    u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k, iterations=1)
    return remove_small_objects(u > 0, min_size=min_area)

def find_features(skel):
    su = skel.astype(np.uint8)
    ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ks) * su
    endpoints = nb == 1
    junctions = nb >= 3
    return endpoints, junctions, nb

def extract_branches(skel, endpoints, junctions):
    """
    Extract individual branches from skeleton.
    A branch = path of pixels between two junction/endpoint nodes.
    """
    h, w = skel.shape
    node_mask = endpoints | junctions
    branch_mask = skel & ~node_mask  # skeleton pixels that are not nodes

    # Label connected components of branch_mask
    labeled, n_branches = label(branch_mask)

    branches = []
    for i in range(1, n_branches + 1):
        pixels = np.argwhere(labeled == i)
        if len(pixels) < 3:  # skip tiny fragments
            continue

        # Find which nodes this branch connects to
        branch_binary = (labeled == i).astype(np.uint8)
        k = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
        dilated = cv2.dilate(branch_binary, k, iterations=1) > 0

        connected_nodes = []
        node_pixels = np.argwhere(node_mask)
        for ny, nx in node_pixels:
            if dilated[ny, nx]:
                connected_nodes.append((ny, nx))

        branches.append({
            'id': i,
            'pixels': pixels,
            'length': len(pixels),
            'connected_nodes': connected_nodes,
        })

    return branches

def assign_branch_depth(branch_pixels, hue_img, sat_img):
    """Assign depth layer to a branch based on average hue of its pixels."""
    hues = []
    for y, x in branch_pixels:
        if sat_img[y, x] > 20:
            hues.append(hue_img[y, x])

    if len(hues) == 0:
        return 'unknown', 0

    h_arr = np.array(hues)

    # Count pixels in each range
    r_count = np.sum((h_arr <= 12) | (h_arr >= 155))
    y_count = np.sum((h_arr >= 13) & (h_arr <= 55))
    b_count = np.sum((h_arr >= 85) & (h_arr <= 140))

    total = r_count + y_count + b_count
    if total == 0:
        return 'unknown', 0

    # Dominant color
    max_count = max(r_count, y_count, b_count)
    confidence = max_count / total

    if r_count == max_count:
        return 'red', confidence
    elif b_count == max_count:
        return 'blue', confidence
    else:
        return 'yellow', confidence

def classify_junction(junction_pt, branches, branch_depths):
    """
    Classify a junction as 'real' or 'crossing' based on connected branch colors.

    Real junction: branches of similar depth meet (R-R, B-B, R-Y, Y-B)
    Crossing: R and B branches meet without Y intermediary
    """
    jy, jx = junction_pt

    connected_branch_ids = []
    for b in branches:
        for ny, nx in b['connected_nodes']:
            if abs(ny - jy) <= 2 and abs(nx - jx) <= 2:
                connected_branch_ids.append(b['id'])
                break

    if len(connected_branch_ids) < 2:
        return 'real', connected_branch_ids

    # Get depths of connected branches
    colors = set()
    for bid in connected_branch_ids:
        if bid in branch_depths:
            colors.add(branch_depths[bid][0])

    colors.discard('unknown')

    # Crossing: R and B meet without Y
    if 'red' in colors and 'blue' in colors and 'yellow' not in colors:
        return 'crossing', connected_branch_ids

    return 'real', connected_branch_ids

# ===== Main analysis =====
# Use Sample 1 GT (projected image only!)
s1_bf_gt = np.array(Image.open(
    Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data\Original color stack data') /
    'Sample 1_BF and GT(512x512 two images).png'))
gt_img = s1_bf_gt[512:]

hsv = cv2.cvtColor(gt_img, cv2.COLOR_RGB2HSV)
hue_img = hsv[:,:,0]
sat_img = hsv[:,:,1]

# Step 1: Overall vessel mask and skeleton
vessel_mask = get_vessel_mask_color(gt_img)
skel = skeletonize(vessel_mask)
endpoints, junctions, nb_map = find_features(skel)

# Remove small spurs (pruning)
# Iteratively remove endpoints of very short branches
skel_pruned = skel.copy()
for _ in range(5):  # 5 iterations of pruning
    ep, jn, nb = find_features(skel_pruned)
    ep_coords = np.argwhere(ep)
    for ey, ex in ep_coords:
        # Trace from endpoint - if branch is < 10px, remove it
        path = [(ey, ex)]
        visited = set()
        visited.add((ey, ex))
        cy, cx = ey, ex
        while True:
            found_next = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < skel_pruned.shape[0] and 0 <= nx < skel_pruned.shape[1]:
                        if skel_pruned[ny, nx] and (ny, nx) not in visited:
                            # Check if this is a junction
                            if jn[ny, nx]:
                                found_next = False  # reached junction, stop
                                break
                            path.append((ny, nx))
                            visited.add((ny, nx))
                            cy, cx = ny, nx
                            found_next = True
                            break
                if not found_next or jn[cy, cx]:
                    break
            if not found_next:
                break

        if len(path) < 8:  # Short spur - remove
            for py, px in path:
                skel_pruned[py, px] = False

# Recompute features after pruning
endpoints_p, junctions_p, nb_p = find_features(skel_pruned)

# Step 2: Extract branches
branches = extract_branches(skel_pruned, endpoints_p, junctions_p)

# Step 3: Assign depth to each branch
branch_depths = {}
for b in branches:
    depth, conf = assign_branch_depth(b['pixels'], hue_img, sat_img)
    branch_depths[b['id']] = (depth, conf)
    b['depth'] = depth
    b['confidence'] = conf

# Step 4: Classify junctions
junction_coords = np.argwhere(junctions_p)
junction_classes = {}
for jy, jx in junction_coords:
    cls, connected = classify_junction((jy, jx), branches, branch_depths)
    junction_classes[(jy, jx)] = (cls, connected)

n_real = sum(1 for v in junction_classes.values() if v[0] == 'real')
n_crossing = sum(1 for v in junction_classes.values() if v[0] == 'crossing')

print(f'Total junctions: {len(junction_coords)}')
print(f'  Real junctions: {n_real}')
print(f'  Crossings (R↔B without Y): {n_crossing}')
print(f'Total branches: {len(branches)}')

depth_counts = defaultdict(int)
for bid, (d, c) in branch_depths.items():
    depth_counts[d] += 1
print(f'Branch depth distribution: {dict(depth_counts)}')

# ===== Visualization =====
depth_colors = {'red': (230, 70, 70), 'yellow': (240, 220, 50), 'blue': (70, 130, 240), 'unknown': (150, 150, 150)}

fig, axes = plt.subplots(2, 4, figsize=(28, 12))
fig.suptitle('Network-level Vessel Tracing (from projection only)', fontsize=16, fontweight='bold')

# 1. Original
axes[0,0].imshow(gt_img)
axes[0,0].set_title('Original GT (projected)', fontsize=12)

# 2. Overall skeleton (pruned)
skel_vis = np.zeros_like(gt_img)
skel_vis[cv2.dilate(skel_pruned.astype(np.uint8),
         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))) > 0] = (200, 200, 200)
axes[0,1].imshow(skel_vis)
ep_c = np.argwhere(endpoints_p)
jn_c = np.argwhere(junctions_p)
axes[0,1].plot(ep_c[:,1], ep_c[:,0], 'go', ms=4, label=f'Endpoints ({len(ep_c)})')
axes[0,1].plot(jn_c[:,1], jn_c[:,0], 'co', ms=4, label=f'Junctions ({len(jn_c)})')
axes[0,1].legend(fontsize=8)
axes[0,1].set_title(f'Pruned Skeleton\nJ={len(jn_c)}, E={len(ep_c)}', fontsize=12)

# 3. Branch-level depth coloring
branch_vis = np.zeros_like(gt_img)
dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
for b in branches:
    color = depth_colors.get(b['depth'], (150, 150, 150))
    mask = np.zeros(skel_pruned.shape, dtype=np.uint8)
    for py, px in b['pixels']:
        mask[py, px] = 1
    mask_thick = cv2.dilate(mask, dk) > 0
    branch_vis[mask_thick] = color

axes[0,2].imshow(branch_vis)
axes[0,2].set_title(f'Branch Depth Assignment\nR={depth_counts["red"]}, Y={depth_counts["yellow"]}, B={depth_counts["blue"]}', fontsize=12)

# 4. Branch depth on original
overlay = gt_img.copy().astype(np.float64)
overlay = overlay * 0.4  # dim original
for b in branches:
    color = depth_colors.get(b['depth'], (150, 150, 150))
    mask = np.zeros(skel_pruned.shape, dtype=np.uint8)
    for py, px in b['pixels']:
        mask[py, px] = 1
    mask_thick = cv2.dilate(mask, dk) > 0
    overlay[mask_thick] = color
axes[0,3].imshow(overlay.astype(np.uint8))
axes[0,3].set_title('Branch Depth on GT', fontsize=12)

# Row 2: Junction classification
# 5. All junctions colored by type
axes[1,0].imshow(branch_vis)
for (jy, jx), (cls, _) in junction_classes.items():
    if cls == 'real':
        axes[1,0].plot(jx, jy, 'o', color='cyan', ms=8, mec='white', mew=1.5)
    else:
        axes[1,0].plot(jx, jy, 'X', color='magenta', ms=10, mec='white', mew=1.5)
axes[1,0].set_title(f'Junction Classification\nCyan=Real({n_real}), Magenta=Crossing({n_crossing})', fontsize=11)

# 6. Crossing junctions zoomed
axes[1,1].imshow(gt_img)
for (jy, jx), (cls, _) in junction_classes.items():
    if cls == 'crossing':
        axes[1,1].plot(jx, jy, 'X', color='magenta', ms=12, mec='yellow', mew=2)
        # Draw circle around crossing
        circle = plt.Circle((jx, jy), 25, fill=False, color='magenta', lw=2, ls='--')
        axes[1,1].add_patch(circle)
axes[1,1].set_title(f'Crossing Points on GT\n({n_crossing} crossings detected)', fontsize=12)

# 7. Single-color skeleton for comparison
single_skel = skeletonize(vessel_mask)
single_vis = np.zeros_like(gt_img)
single_vis[cv2.dilate(single_skel.astype(np.uint8), dk) > 0] = (0, 200, 0)
_, single_jn, _ = find_features(single_skel)
single_jn_c = np.argwhere(single_jn)
axes[1,2].imshow(single_vis)
for jy, jx in single_jn_c:
    axes[1,2].plot(jx, jy, 'o', color='cyan', ms=4)
axes[1,2].set_title(f'Single-color Skeleton\nJ={len(single_jn_c)} (all counted as real)', fontsize=12)

# 8. Comparison summary
# Show crossing locations on single skeleton
axes[1,3].imshow(single_vis)
for (jy, jx), (cls, _) in junction_classes.items():
    if cls == 'crossing':
        # Find nearest junction in single skeleton
        if len(single_jn_c) > 0:
            dists = np.sqrt((single_jn_c[:,0] - jy)**2 + (single_jn_c[:,1] - jx)**2)
            nearest = np.argmin(dists)
            if dists[nearest] < 15:
                ny, nx = single_jn_c[nearest]
                axes[1,3].plot(nx, ny, 'X', color='magenta', ms=12, mec='yellow', mew=2)
                circle = plt.Circle((nx, ny), 20, fill=False, color='magenta', lw=2, ls='--')
                axes[1,3].add_patch(circle)

axes[1,3].set_title(f'Single Skel: Crossings Highlighted\n(These are false junctions in single-color)', fontsize=11)

for ax_row in axes:
    for ax in ax_row:
        ax.axis('off')

plt.tight_layout()
plt.savefig(OUT / 'network_trace_sample1.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# ===== Zoomed crossing comparison =====
crossing_pts = [(jy, jx) for (jy, jx), (cls, _) in junction_classes.items() if cls == 'crossing']
n_zoom = min(4, len(crossing_pts))

if n_zoom > 0:
    fig2, axes2 = plt.subplots(n_zoom, 4, figsize=(20, 5*n_zoom))
    if n_zoom == 1:
        axes2 = axes2[np.newaxis, :]
    fig2.suptitle('Zoomed Crossing Regions: Single vs Depth-separated', fontsize=14, fontweight='bold')

    pad = 50
    h, w = gt_img.shape[:2]
    for zi in range(n_zoom):
        cy, cx = crossing_pts[zi]
        y0, y1 = max(0, cy-pad), min(h, cy+pad)
        x0, x1 = max(0, cx-pad), min(w, cx+pad)

        axes2[zi,0].imshow(gt_img[y0:y1, x0:x1])
        axes2[zi,0].set_title('GT', fontsize=10)

        axes2[zi,1].imshow(branch_vis[y0:y1, x0:x1])
        axes2[zi,1].set_title('Depth branches', fontsize=10)

        axes2[zi,2].imshow(single_vis[y0:y1, x0:x1])
        # Mark false junction
        for jy2, jx2 in single_jn_c:
            if y0 <= jy2 < y1 and x0 <= jx2 < x1:
                axes2[zi,2].plot(jx2-x0, jy2-y0, 'o', color='red', ms=8, mec='white', mew=1.5)
        axes2[zi,2].set_title('Single skel (red=junction)', fontsize=10)

        axes2[zi,3].imshow(overlay.astype(np.uint8)[y0:y1, x0:x1])
        axes2[zi,3].plot(cx-x0, cy-y0, 'X', color='magenta', ms=15, mec='white', mew=2)
        axes2[zi,3].set_title('Crossing point', fontsize=10)

        for ax in axes2[zi]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUT / 'network_crossing_zoom_sample1.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

print(f'\nSummary:')
print(f'  Single-color junctions: {len(single_jn_c)}')
print(f'  Multi-color real junctions: {n_real}')
print(f'  Crossings identified: {n_crossing}')
print(f'  → {n_crossing} false junctions removed by depth-aware analysis')
print('\nDone! Check network_trace_sample1.png')
