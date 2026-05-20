"""
Smart depth-layer separation: distinguishes natural transitions (R→Y→B) from crossings (R over B).
Key idea: Yellow (middle layer) acts as a transition indicator.
  - If a hole in R mask contains Y pixels → natural depth transition → fill
  - If a hole in R mask contains B pixels → crossing → DON'T fill
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import binary_fill_holes, label
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')
OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')

def load(p): return np.array(Image.open(p).convert('RGB'))

def get_raw_layers(img, min_area=50):
    """Get raw exclusive hue-based masks without any hole filling."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s = hsv[:,:,0], hsv[:,:,1]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fg = (gray > 15) & (s > 30)

    r_mask = ((h <= 12) | (h >= 155)) & fg
    y_mask = ((h >= 13) & (h <= 55)) & fg
    b_mask = ((h >= 85) & (h <= 140)) & fg

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    def clean(mask):
        u = cv2.morphologyEx(mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k, iterations=2)
        u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k, iterations=1)
        return remove_small_objects(u > 0, min_size=min_area)

    return clean(r_mask), clean(y_mask), clean(b_mask)

def smart_fill_holes(mask, transition_mask, crossing_mask, overall_vessel):
    """
    Fill holes in `mask` only if the hole contains mostly transition pixels.
    Don't fill if hole contains mostly crossing pixels.

    For R mask: transition=Y, crossing=B
    For B mask: transition=Y, crossing=R
    For Y mask: fill all holes (Y is the universal transition layer)
    """
    labeled, n = label(mask)
    filled = mask.copy()
    fill_info = []

    for i in range(1, n + 1):
        comp = labeled == i
        comp_filled = binary_fill_holes(comp)
        hole = comp_filled & ~comp & overall_vessel

        if np.sum(hole) < 5:  # skip tiny holes
            continue

        trans_count = np.sum(hole & transition_mask)
        cross_count = np.sum(hole & crossing_mask)
        total = trans_count + cross_count

        if total == 0:
            # Hole has no other layer pixels → likely just noise, fill it
            filled |= hole
            fill_info.append(('noise', np.sum(hole)))
        elif trans_count > cross_count * 0.5:
            # Mostly transition pixels → natural depth transition → fill
            filled |= hole
            fill_info.append(('transition', np.sum(hole), trans_count, cross_count))
        else:
            # Mostly crossing pixels → different vessel → DON'T fill
            fill_info.append(('crossing', np.sum(hole), trans_count, cross_count))

    return filled, fill_info

def separate_layers_smart(img, min_area=50):
    """Smart layer separation that preserves crossings."""
    r_raw, y_raw, b_raw = get_raw_layers(img, min_area)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    overall = gray > 15

    # Smart fill: R holes with Y=transition, B=crossing
    r_filled, r_info = smart_fill_holes(r_raw, y_raw, b_raw, overall)
    # Smart fill: B holes with Y=transition, R=crossing
    b_filled, b_info = smart_fill_holes(b_raw, y_raw, r_raw, overall)
    # Y: fill all holes (universal transition layer)
    y_filled = y_raw.copy()
    y_labeled, y_n = label(y_raw)
    for i in range(1, y_n + 1):
        comp = y_labeled == i
        comp_filled = binary_fill_holes(comp)
        y_filled |= comp_filled & ~comp & overall

    return r_filled, y_filled, b_filled, r_info, b_info

def old_separate_layers_hf(img, min_area=50):
    """Old method: fill ALL holes (for comparison)."""
    r_raw, y_raw, b_raw = get_raw_layers(img, min_area)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    overall = gray > 15

    def fill_all(mask):
        labeled, n = label(mask)
        filled = mask.copy()
        for i in range(1, n + 1):
            comp = labeled == i
            filled |= binary_fill_holes(comp) & ~comp & overall
        return filled

    return fill_all(r_raw), fill_all(y_raw), fill_all(b_raw)

def find_features(skeleton):
    su = skeleton.astype(np.uint8)
    ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ks) * su
    return np.argwhere(nb == 1), np.argwhere(nb >= 3)

def dilate_skel(s, t=3):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
    return cv2.dilate(s.astype(np.uint8)*255, k, iterations=1) > 0

# ===== Test on representative samples =====
gt_multi_dir = BASE / 'Multi-color' / 'Pix2pix'
samples = ['18-19-512', '1-16-512', '9-15-512', '16-18-716', '1-19-716']
layer_colors = [(230, 70, 70), (240, 220, 50), (70, 130, 240)]

for sid in samples:
    img = load(gt_multi_dir / f'{sid}_real_B.png')
    h, w = img.shape[:2]
    crop_y = int(h * 0.8)

    # Old method
    r_old, y_old, b_old = old_separate_layers_hf(img)
    sk_old = [skeletonize(m) for m in [r_old, y_old, b_old]]

    # New smart method
    r_new, y_new, b_new, r_info, b_info = separate_layers_smart(img)
    sk_new = [skeletonize(m) for m in [r_new, y_new, b_new]]

    # Count metrics (above baseline only)
    vm = np.ones((h, w), dtype=bool)
    vm[crop_y:, :] = False

    old_j = sum(len(find_features(sk)[1][find_features(sk)[1][:,0] < crop_y]) for sk in sk_old)
    new_j = sum(len(find_features(sk)[1][find_features(sk)[1][:,0] < crop_y]) for sk in sk_new)
    old_l = sum(np.sum(sk & vm) for sk in sk_old)
    new_l = sum(np.sum(sk & vm) for sk in sk_new)

    print(f'\n{sid}:')
    print(f'  Old (fill all):  J={old_j}, L={old_l}')
    print(f'  New (smart fill): J={new_j}, L={new_l}')
    print(f'  R holes: {len(r_info)} analyzed - {sum(1 for x in r_info if x[0]=="crossing")} crossings preserved')
    print(f'  B holes: {len(b_info)} analyzed - {sum(1 for x in b_info if x[0]=="crossing")} crossings preserved')

    # Visualize comparison
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(f'{sid}: Old (fill all) vs New (smart fill)', fontsize=14, fontweight='bold')

    # Row 0: Old method
    axes[0, 0].imshow(img); axes[0, 0].set_title('Original GT')

    vis_old = np.zeros_like(img)
    for si, sk in enumerate(sk_old):
        vis_old[dilate_skel(sk, 4)] = layer_colors[si]
    axes[0, 1].imshow(vis_old); axes[0, 1].set_title(f'Old Skeleton (fill all)\nJ={old_j}, L={old_l}')

    # Show old masks with overlap highlighted
    mask_vis_old = np.zeros_like(img)
    mask_vis_old[r_old] = [180, 50, 50]
    mask_vis_old[y_old] = [180, 180, 50]
    mask_vis_old[b_old] = [50, 100, 200]
    # Highlight R-B overlap
    rb_overlap_old = r_old & b_old
    mask_vis_old[rb_overlap_old] = [255, 0, 255]  # magenta for overlap
    axes[0, 2].imshow(mask_vis_old)
    axes[0, 2].set_title(f'Old Masks (magenta=R∩B overlap: {np.sum(rb_overlap_old)}px)')

    # Old skeleton on original
    vis_overlay_old = img.copy()
    for si, sk in enumerate(sk_old):
        vis_overlay_old[dilate_skel(sk, 3)] = layer_colors[si]
    axes[0, 3].imshow(vis_overlay_old); axes[0, 3].set_title('Old Skeleton on GT')

    # Row 1: New method
    axes[1, 0].imshow(img); axes[1, 0].set_title('Original GT')

    vis_new = np.zeros_like(img)
    for si, sk in enumerate(sk_new):
        vis_new[dilate_skel(sk, 4)] = layer_colors[si]
    axes[1, 1].imshow(vis_new); axes[1, 1].set_title(f'New Skeleton (smart fill)\nJ={new_j}, L={new_l}')

    # Show new masks
    mask_vis_new = np.zeros_like(img)
    mask_vis_new[r_new] = [180, 50, 50]
    mask_vis_new[y_new] = [180, 180, 50]
    mask_vis_new[b_new] = [50, 100, 200]
    rb_overlap_new = r_new & b_new
    mask_vis_new[rb_overlap_new] = [255, 0, 255]
    axes[1, 2].imshow(mask_vis_new)
    axes[1, 2].set_title(f'New Masks (magenta=R∩B overlap: {np.sum(rb_overlap_new)}px)')

    # New skeleton on original
    vis_overlay_new = img.copy()
    for si, sk in enumerate(sk_new):
        vis_overlay_new[dilate_skel(sk, 3)] = layer_colors[si]
    axes[1, 3].imshow(vis_overlay_new); axes[1, 3].set_title('New Skeleton on GT')

    for ax_row in axes:
        for ax in ax_row:
            ax.axhline(crop_y, color='yellow', lw=1, ls='--')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUT / f'smart_separation_{sid}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

print('\nDone! Check analysis_output/smart_separation_*.png')
