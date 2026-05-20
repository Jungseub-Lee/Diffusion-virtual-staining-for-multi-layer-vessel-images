"""
Smart separation v2: Focus on the REAL problem.

The issue is NOT hole-filling. It's that R and B masks physically overlap/touch
in crossing zones, causing skeleton distortion.

New approach:
1. Detect crossing zones: where R and B are spatially close WITHOUT Y in between
2. In crossing zones, thin/erode the boundary so each layer's mask stays clean
3. Detect transition zones: where R→Y→B gradual change exists
4. In transition zones, ensure smooth connection

Key insight: At a crossing, R and B directly neighbor each other.
At a transition, R→Y→B with yellow as buffer.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import binary_fill_holes, label, binary_dilation
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')
OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')

def load(p): return np.array(Image.open(p).convert('RGB'))

def get_raw_layers(img, min_area=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s = hsv[:,:,0], hsv[:,:,1]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fg = (gray > 15) & (s > 30)
    r = ((h <= 12) | (h >= 155)) & fg
    y = ((h >= 13) & (h <= 55)) & fg
    b = ((h >= 85) & (h <= 140)) & fg
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    def clean(mask):
        u = cv2.morphologyEx(mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k, iterations=2)
        u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k, iterations=1)
        return remove_small_objects(u > 0, min_size=min_area)
    return clean(r), clean(y), clean(b)

def separate_layers_smart_v2(img, min_area=50):
    """
    Smart separation:
    1. Get raw R, Y, B masks
    2. Detect R-B contact zones (crossing indicator)
    3. At contact zones, use hue gradient to determine which layer "owns" disputed pixels
    4. Ensure each layer's mask is clean for skeletonization
    """
    r_raw, y_raw, b_raw = get_raw_layers(img, min_area)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv[:,:,0].astype(np.float32)
    overall = gray > 15

    # Step 1: Find R-B contact zone (where R and B are within ~5px of each other)
    k_contact = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    r_dilated = cv2.dilate(r_raw.astype(np.uint8), k_contact) > 0
    b_dilated = cv2.dilate(b_raw.astype(np.uint8), k_contact) > 0

    # Contact zone: where dilated R and dilated B overlap
    rb_contact = r_dilated & b_dilated

    # Check if Y is present in contact zone → transition vs crossing
    y_in_contact = rb_contact & y_raw
    crossing_zone = rb_contact & ~binary_dilation(y_raw, k_contact)

    # Step 2: At crossing zones, clean up each mask
    # Remove small bridges between R and B at crossings
    # Strategy: at crossing points, erode R and B slightly to separate them
    k_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    r_clean = r_raw.copy()
    b_clean = b_raw.copy()

    # At crossing zones, erode R and B boundaries to prevent merge
    crossing_r = crossing_zone & r_raw
    crossing_b = crossing_zone & b_raw
    if np.sum(crossing_r) > 0:
        r_clean[crossing_r] = False  # Remove R pixels in crossing zone
        # Re-add R pixels that are far from B
        r_core = cv2.erode(r_raw.astype(np.uint8), k_thin, iterations=1) > 0
        r_clean = r_clean | (r_core & ~b_raw)  # keep core R pixels not in B

    if np.sum(crossing_b) > 0:
        b_clean[crossing_b] = False
        b_core = cv2.erode(b_raw.astype(np.uint8), k_thin, iterations=1) > 0
        b_clean = b_clean | (b_core & ~r_raw)

    # Step 3: Fill holes for transitions (same vessel going R→Y→B)
    # For R: fill holes that contain Y
    def fill_transition_holes(mask, transition_mask, crossing_mask):
        labeled_arr, n = label(mask)
        filled = mask.copy()
        for i in range(1, n + 1):
            comp = labeled_arr == i
            comp_filled = binary_fill_holes(comp)
            hole = comp_filled & ~comp & overall
            if np.sum(hole) < 5:
                continue
            trans_px = np.sum(hole & transition_mask)
            cross_px = np.sum(hole & crossing_mask)
            if trans_px >= cross_px:
                filled |= hole
        return filled

    r_final = fill_transition_holes(r_clean, y_raw, b_raw)
    b_final = fill_transition_holes(b_clean, y_raw, r_raw)

    # Y: fill all holes
    y_labeled, y_n = label(y_raw)
    y_final = y_raw.copy()
    for i in range(1, y_n + 1):
        comp = y_labeled == i
        y_final |= binary_fill_holes(comp) & ~comp & overall

    # Clean up small artifacts
    r_final = remove_small_objects(r_final, min_size=min_area)
    b_final = remove_small_objects(b_final, min_size=min_area)
    y_final = remove_small_objects(y_final, min_size=min_area)

    debug = {
        'rb_contact': rb_contact,
        'crossing_zone': crossing_zone,
        'y_in_contact': y_in_contact,
        'crossing_r_px': int(np.sum(crossing_r)),
        'crossing_b_px': int(np.sum(crossing_b)),
    }

    return r_final, y_final, b_final, debug

def find_features(skeleton):
    su = skeleton.astype(np.uint8)
    ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ks) * su
    return np.argwhere(nb == 1), np.argwhere(nb >= 3)

def dilate_skel(s, t=3):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
    return cv2.dilate(s.astype(np.uint8)*255, k, iterations=1) > 0

# ===== Test =====
gt_dir = BASE / 'Multi-color' / 'Pix2pix'
samples = ['18-19-512', '1-16-512', '9-15-512', '16-18-716', '1-19-716']
layer_colors = [(230, 70, 70), (240, 220, 50), (70, 130, 240)]

for sid in samples:
    img = load(gt_dir / f'{sid}_real_B.png')
    h, w = img.shape[:2]
    crop_y = int(h * 0.8)
    vm = np.ones((h, w), dtype=bool); vm[crop_y:, :] = False

    # Old: raw layers without smart separation
    r_old, y_old, b_old = get_raw_layers(img)
    # Fill all holes (old method)
    for mask_ref in [r_old, y_old, b_old]:
        pass  # raw layers, no hole fill for cleaner comparison

    sk_old = [skeletonize(m) for m in [r_old, y_old, b_old]]

    # New: smart separation
    r_new, y_new, b_new, debug = separate_layers_smart_v2(img)
    sk_new = [skeletonize(m) for m in [r_new, y_new, b_new]]

    # Metrics
    old_j = sum(np.sum(find_features(sk)[1][:, 0] < crop_y) if len(find_features(sk)[1]) > 0 else 0 for sk in sk_old)
    new_j = sum(np.sum(find_features(sk)[1][:, 0] < crop_y) if len(find_features(sk)[1]) > 0 else 0 for sk in sk_new)
    old_l = sum(np.sum(sk & vm) for sk in sk_old)
    new_l = sum(np.sum(sk & vm) for sk in sk_new)

    print(f'\n{sid}:')
    print(f'  Raw (no fill):   J={old_j}, L={old_l}')
    print(f'  Smart fill:      J={new_j}, L={new_l}')
    print(f'  Crossing zone: R={debug["crossing_r_px"]}px, B={debug["crossing_b_px"]}px')

    # Visualization
    fig, axes = plt.subplots(2, 5, figsize=(30, 10))
    fig.suptitle(f'{sid}: Raw vs Smart Separation', fontsize=14, fontweight='bold')

    # Row 0: Raw
    axes[0, 0].imshow(img); axes[0, 0].set_title('Original GT', fontsize=10)

    mask_old = np.zeros_like(img)
    mask_old[r_old] = [180, 50, 50]; mask_old[y_old] = [180, 180, 50]; mask_old[b_old] = [50, 100, 200]
    axes[0, 1].imshow(mask_old); axes[0, 1].set_title('Raw Masks', fontsize=10)

    vis_old = np.zeros_like(img)
    for si, sk in enumerate(sk_old):
        vis_old[dilate_skel(sk, 4)] = layer_colors[si]
    axes[0, 2].imshow(vis_old); axes[0, 2].set_title(f'Raw Skeleton\nJ={old_j}, L={old_l}', fontsize=10)

    # Show on original
    ov_old = img.copy()
    for si, sk in enumerate(sk_old):
        ov_old[dilate_skel(sk, 3)] = layer_colors[si]
    axes[0, 3].imshow(ov_old); axes[0, 3].set_title('Raw Skeleton on GT', fontsize=10)

    # Debug: crossing zones
    dbg = np.zeros_like(img)
    dbg[debug['rb_contact']] = [100, 100, 100]
    dbg[debug['crossing_zone']] = [255, 0, 255]
    dbg[debug['y_in_contact']] = [0, 255, 0]
    axes[0, 4].imshow(dbg)
    axes[0, 4].set_title(f'Contact Analysis\nMagenta=crossing, Green=transition', fontsize=9)

    # Row 1: Smart
    axes[1, 0].imshow(img); axes[1, 0].set_title('Original GT', fontsize=10)

    mask_new = np.zeros_like(img)
    mask_new[r_new] = [180, 50, 50]; mask_new[y_new] = [180, 180, 50]; mask_new[b_new] = [50, 100, 200]
    rb_ov = r_new & b_new
    mask_new[rb_ov] = [255, 0, 255]
    axes[1, 1].imshow(mask_new); axes[1, 1].set_title(f'Smart Masks (magenta=R∩B: {np.sum(rb_ov)}px)', fontsize=10)

    vis_new = np.zeros_like(img)
    for si, sk in enumerate(sk_new):
        vis_new[dilate_skel(sk, 4)] = layer_colors[si]
    axes[1, 2].imshow(vis_new); axes[1, 2].set_title(f'Smart Skeleton\nJ={new_j}, L={new_l}', fontsize=10)

    ov_new = img.copy()
    for si, sk in enumerate(sk_new):
        ov_new[dilate_skel(sk, 3)] = layer_colors[si]
    axes[1, 3].imshow(ov_new); axes[1, 3].set_title('Smart Skeleton on GT', fontsize=10)

    # Difference map
    diff = np.zeros_like(img)
    for si in range(3):
        only_old = sk_old[si] & ~sk_new[si]
        only_new = sk_new[si] & ~sk_old[si]
        diff[dilate_skel(only_old, 4)] = [255, 100, 100]  # Red = removed
        diff[dilate_skel(only_new, 4)] = [100, 255, 100]  # Green = added
    axes[1, 4].imshow(diff); axes[1, 4].set_title('Difference\nRed=removed, Green=added', fontsize=10)

    for ax_row in axes:
        for ax in ax_row:
            ax.axhline(crop_y, color='yellow', lw=1, ls='--')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUT / f'smart_v2_{sid}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

print('\nDone! Check analysis_output/smart_v2_*.png')
