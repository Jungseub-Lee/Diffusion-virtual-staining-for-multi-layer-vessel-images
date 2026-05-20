"""
Visualization: thresholded masks + thick skeleton overlay on original GT images.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from skimage.morphology import skeletonize, remove_small_objects

data_dir = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data\Multi-color\Pix2pix")
output_dir = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output")
output_dir.mkdir(exist_ok=True)

sample_ids = ["1-0-512", "1-5-512", "1-10-512", "1-15-512"]


def separate_layers_hsv(img_rgb, min_area=50):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    vessel_mask = gray > 15
    sat_mask = s > 30
    fg_mask = vessel_mask & sat_mask

    bottom_hue = ((h <= 12) | (h >= 155)) & fg_mask
    middle_hue = ((h >= 13) & (h <= 55)) & fg_mask
    top_hue = ((h >= 85) & (h <= 140)) & fg_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def clean_mask(mask):
        m = mask.astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        mb = m > 0
        mb = remove_small_objects(mb, min_size=min_area)
        return mb

    return clean_mask(bottom_hue), clean_mask(middle_hue), clean_mask(top_hue)


def dilate_skeleton(skeleton, thickness=3):
    """Dilate skeleton for thicker visualization."""
    skel_uint8 = skeleton.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    return cv2.dilate(skel_uint8, kernel, iterations=1) > 0


for sid in sample_ids:
    img_path = data_dir / f"{sid}_real_B.png"
    img = np.array(Image.open(img_path).convert("RGB"))

    bottom, middle, top = separate_layers_hsv(img)

    skel_bottom = skeletonize(bottom)
    skel_middle = skeletonize(middle)
    skel_top = skeletonize(top)

    # Thick versions for overlay
    thick_bottom = dilate_skeleton(skel_bottom, thickness=5)
    thick_middle = dilate_skeleton(skel_middle, thickness=5)
    thick_top = dilate_skeleton(skel_top, thickness=5)

    # --- Figure: 3 rows x 4 cols ---
    # Row 0: Original | Bottom mask | Middle mask | Top mask
    # Row 1: Original dimmed + each mask overlay (semi-transparent colored)
    # Row 2: Original dimmed + thick skeleton overlay per layer | All skeletons combined on original
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f"Sample: {sid}", fontsize=16, fontweight="bold")

    # ---- Row 0: Raw masks ----
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original GT")

    # Bottom mask on black
    vis_bottom = np.zeros_like(img)
    vis_bottom[bottom] = img[bottom]
    axes[0, 1].imshow(vis_bottom)
    axes[0, 1].set_title("Bottom Layer Mask\n(Red/Magenta, H≤12 or H≥155)")

    # Middle mask on black
    vis_middle = np.zeros_like(img)
    vis_middle[middle] = img[middle]
    axes[0, 2].imshow(vis_middle)
    axes[0, 2].set_title("Middle Layer Mask\n(Yellow/Orange, H:13-55)")

    # Top mask on black
    vis_top = np.zeros_like(img)
    vis_top[top] = img[top]
    axes[0, 3].imshow(vis_top)
    axes[0, 3].set_title("Top Layer Mask\n(Cyan/Blue, H:85-140)")

    # ---- Row 1: Semi-transparent colored mask on dimmed original ----
    dim = (img * 0.3).astype(np.uint8)

    # Bottom overlay (red)
    ov_b = dim.copy()
    ov_b[bottom] = np.clip(img[bottom] * 0.5 + np.array([180, 30, 30]) * 0.5, 0, 255).astype(np.uint8)
    axes[1, 0].imshow(ov_b)
    axes[1, 0].set_title("Bottom on Original\n(Red overlay)")

    # Middle overlay (yellow)
    ov_m = dim.copy()
    ov_m[middle] = np.clip(img[middle] * 0.5 + np.array([220, 220, 30]) * 0.5, 0, 255).astype(np.uint8)
    axes[1, 1].imshow(ov_m)
    axes[1, 1].set_title("Middle on Original\n(Yellow overlay)")

    # Top overlay (blue)
    ov_t = dim.copy()
    ov_t[top] = np.clip(img[top] * 0.5 + np.array([30, 100, 255]) * 0.5, 0, 255).astype(np.uint8)
    axes[1, 2].imshow(ov_t)
    axes[1, 2].set_title("Top on Original\n(Blue overlay)")

    # All masks combined overlay
    ov_all = dim.copy()
    ov_all[bottom] = np.clip(ov_all[bottom].astype(float) + np.array([180, 30, 30]) * 0.6, 0, 255).astype(np.uint8)
    ov_all[middle] = np.clip(ov_all[middle].astype(float) + np.array([220, 220, 30]) * 0.6, 0, 255).astype(np.uint8)
    ov_all[top] = np.clip(ov_all[top].astype(float) + np.array([30, 100, 255]) * 0.6, 0, 255).astype(np.uint8)
    axes[1, 3].imshow(ov_all)
    axes[1, 3].set_title("All Layers Combined\n(R/Y/B overlay)")

    # ---- Row 2: Thick skeleton overlays on dimmed original ----
    # Bottom skeleton
    sk_b = dim.copy()
    sk_b[thick_bottom] = [255, 80, 80]
    axes[2, 0].imshow(sk_b)
    axes[2, 0].set_title("Bottom Skeleton\n(Red, thick)")

    # Middle skeleton
    sk_m = dim.copy()
    sk_m[thick_middle] = [255, 255, 60]
    axes[2, 1].imshow(sk_m)
    axes[2, 1].set_title("Middle Skeleton\n(Yellow, thick)")

    # Top skeleton
    sk_t = dim.copy()
    sk_t[thick_top] = [80, 140, 255]
    axes[2, 2].imshow(sk_t)
    axes[2, 2].set_title("Top Skeleton\n(Blue, thick)")

    # All skeletons on original
    sk_all = dim.copy()
    # Draw in order: bottom first, then middle, then top (top on top)
    sk_all[thick_bottom] = [255, 80, 80]
    sk_all[thick_middle] = [255, 255, 60]
    sk_all[thick_top] = [80, 140, 255]
    axes[2, 3].imshow(sk_all)
    axes[2, 3].set_title("All Skeletons\n(Depth-colored overlay)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / f"overlay_{sid}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: overlay_{sid}.png")

print("Done.")
