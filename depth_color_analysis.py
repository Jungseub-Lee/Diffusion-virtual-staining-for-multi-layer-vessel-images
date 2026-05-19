"""
Depth-aware color separation and skeletonization analysis
for multi-color depth-encoded fluorescence images.

Goal: Separate GT fluorescence images into depth layers by color,
then skeletonize each layer to quantify 3D vascular morphology.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label

# ============================================================
# 1. Load sample GT images
# ============================================================
data_dir = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data\Multi-color\Pix2pix")
output_dir = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output")
output_dir.mkdir(exist_ok=True)

# Pick a few representative samples
sample_ids = ["1-0-512", "1-5-512", "1-10-512", "1-15-512"]

# ============================================================
# 2. Analyze color distribution in HSV space
# ============================================================
def analyze_color_distribution(img_rgb):
    """Analyze the HSV color distribution of vessel pixels (non-black)."""
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Mask out background (black pixels)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    vessel_mask = gray > 15  # threshold to exclude background

    h_values = img_hsv[:, :, 0][vessel_mask]  # Hue channel (0-179 in OpenCV)
    s_values = img_hsv[:, :, 1][vessel_mask]  # Saturation
    v_values = img_hsv[:, :, 2][vessel_mask]  # Value

    return h_values, s_values, v_values, vessel_mask


# ============================================================
# 3. Color-based layer separation
# ============================================================
def separate_layers_hsv(img_rgb, min_area=50):
    """
    Separate image into depth layers based on HSV color.

    Depth color code (ZstackDepthColorCode):
    - Bottom (red/magenta): H ~ 0-10 or 160-179 (red range in OpenCV)
    - Middle (yellow/green): H ~ 15-45
    - Top (cyan/blue): H ~ 90-135

    Returns masks for bottom, middle, top layers.
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]

    # Background mask
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    vessel_mask = gray > 15
    sat_mask = s > 30  # exclude very low saturation (white/gray)
    fg_mask = vessel_mask & sat_mask

    # --- Layer separation by hue ---
    # Bottom layer (Red/Magenta): Hue 0-10 or 160-179
    bottom_hue = ((h <= 12) | (h >= 155)) & fg_mask

    # Middle layer (Yellow/Orange/Green): Hue 13-55
    middle_hue = ((h >= 13) & (h <= 55)) & fg_mask

    # Top layer (Cyan/Blue): Hue 85-140
    top_hue = ((h >= 85) & (h <= 140)) & fg_mask

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def clean_mask(mask):
        mask_uint8 = mask.astype(np.uint8) * 255
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
        # Remove small objects
        mask_bool = mask_uint8 > 0
        mask_bool = remove_small_objects(mask_bool, min_size=min_area)
        return mask_bool

    bottom_mask = clean_mask(bottom_hue)
    middle_mask = clean_mask(middle_hue)
    top_mask = clean_mask(top_hue)

    return bottom_mask, middle_mask, top_mask


def skeletonize_layer(mask):
    """Skeletonize a binary mask."""
    return skeletonize(mask)


def compute_skeleton_metrics(skeleton):
    """Compute basic morphometric features from a skeleton."""
    # Total length (pixel count of skeleton)
    total_length = np.sum(skeleton)

    if total_length == 0:
        return {"length": 0, "junctions": 0, "endpoints": 0}

    # Find junctions and endpoints using 3x3 neighborhood
    skel_uint8 = skeleton.astype(np.uint8)

    # Count neighbors for each skeleton pixel
    kernel_sum = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_uint8, -1, kernel_sum)
    neighbor_count = neighbor_count * skel_uint8  # only skeleton pixels

    # Endpoints: 1 neighbor, Junctions: 3+ neighbors
    endpoints = np.sum(neighbor_count == 1)
    junctions = np.sum(neighbor_count >= 3)

    return {
        "length": int(total_length),
        "junctions": int(junctions),
        "endpoints": int(endpoints)
    }


# ============================================================
# 4. Run analysis on sample images
# ============================================================
fig_main, axes_main = plt.subplots(len(sample_ids), 6, figsize=(24, 4 * len(sample_ids)))

all_results = {}

for i, sid in enumerate(sample_ids):
    img_path = data_dir / f"{sid}_real_B.png"
    img = np.array(Image.open(img_path).convert("RGB"))

    # Separate layers
    bottom, middle, top = separate_layers_hsv(img)

    # Skeletonize each layer
    skel_bottom = skeletonize_layer(bottom)
    skel_middle = skeletonize_layer(middle)
    skel_top = skeletonize_layer(top)

    # Compute metrics
    metrics_bottom = compute_skeleton_metrics(skel_bottom)
    metrics_middle = compute_skeleton_metrics(skel_middle)
    metrics_top = compute_skeleton_metrics(skel_top)

    # Combined skeleton (single-color equivalent)
    combined_mask = bottom | middle | top
    skel_combined = skeletonize_layer(combined_mask)
    metrics_combined = compute_skeleton_metrics(skel_combined)

    all_results[sid] = {
        "bottom": metrics_bottom,
        "middle": metrics_middle,
        "top": metrics_top,
        "combined": metrics_combined
    }

    # --- Visualization ---
    # Col 0: Original
    axes_main[i, 0].imshow(img)
    axes_main[i, 0].set_title(f"{sid}\nOriginal GT")
    axes_main[i, 0].axis("off")

    # Col 1: Bottom layer (red)
    overlay_bottom = np.zeros_like(img)
    overlay_bottom[bottom] = [255, 50, 50]
    overlay_bottom[skel_bottom] = [255, 255, 255]
    axes_main[i, 1].imshow(overlay_bottom)
    axes_main[i, 1].set_title(f"Bottom (Red)\nL={metrics_bottom['length']}, J={metrics_bottom['junctions']}, E={metrics_bottom['endpoints']}")
    axes_main[i, 1].axis("off")

    # Col 2: Middle layer (yellow)
    overlay_middle = np.zeros_like(img)
    overlay_middle[middle] = [255, 255, 50]
    overlay_middle[skel_middle] = [255, 255, 255]
    axes_main[i, 2].imshow(overlay_middle)
    axes_main[i, 2].set_title(f"Middle (Yellow)\nL={metrics_middle['length']}, J={metrics_middle['junctions']}, E={metrics_middle['endpoints']}")
    axes_main[i, 2].axis("off")

    # Col 3: Top layer (blue)
    overlay_top = np.zeros_like(img)
    overlay_top[top] = [50, 100, 255]
    overlay_top[skel_top] = [255, 255, 255]
    axes_main[i, 3].imshow(overlay_top)
    axes_main[i, 3].set_title(f"Top (Blue)\nL={metrics_top['length']}, J={metrics_top['junctions']}, E={metrics_top['endpoints']}")
    axes_main[i, 3].axis("off")

    # Col 4: Combined skeleton overlay (depth-aware)
    overlay_skel = np.zeros_like(img)
    overlay_skel[skel_bottom] = [255, 50, 50]
    overlay_skel[skel_middle] = [255, 255, 50]
    overlay_skel[skel_top] = [50, 100, 255]
    axes_main[i, 4].imshow(overlay_skel)
    axes_main[i, 4].set_title(f"Depth-Aware Skeleton\n(Layer-separated)")
    axes_main[i, 4].axis("off")

    # Col 5: Single combined skeleton (conventional)
    overlay_combined = np.zeros_like(img)
    overlay_combined[skel_combined] = [0, 255, 0]
    axes_main[i, 5].imshow(overlay_combined)
    axes_main[i, 5].set_title(f"Combined Skeleton\nL={metrics_combined['length']}, J={metrics_combined['junctions']}, E={metrics_combined['endpoints']}")
    axes_main[i, 5].axis("off")

plt.tight_layout()
plt.savefig(output_dir / "depth_layer_separation_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 5. Print summary
# ============================================================
print("\n" + "=" * 80)
print("DEPTH-AWARE MORPHOMETRIC ANALYSIS SUMMARY")
print("=" * 80)

for sid, res in all_results.items():
    print(f"\n--- {sid} ---")
    layer_total_length = res['bottom']['length'] + res['middle']['length'] + res['top']['length']
    layer_total_junctions = res['bottom']['junctions'] + res['middle']['junctions'] + res['top']['junctions']
    layer_total_endpoints = res['bottom']['endpoints'] + res['middle']['endpoints'] + res['top']['endpoints']

    print(f"  Bottom (Red):    Length={res['bottom']['length']:5d}, Junctions={res['bottom']['junctions']:3d}, Endpoints={res['bottom']['endpoints']:3d}")
    print(f"  Middle (Yellow): Length={res['middle']['length']:5d}, Junctions={res['middle']['junctions']:3d}, Endpoints={res['middle']['endpoints']:3d}")
    print(f"  Top (Blue):      Length={res['top']['length']:5d}, Junctions={res['top']['junctions']:3d}, Endpoints={res['top']['endpoints']:3d}")
    print(f"  ----- Layer-separated total -----")
    print(f"  Sum of layers:   Length={layer_total_length:5d}, Junctions={layer_total_junctions:3d}, Endpoints={layer_total_endpoints:3d}")
    print(f"  Combined (flat): Length={res['combined']['length']:5d}, Junctions={res['combined']['junctions']:3d}, Endpoints={res['combined']['endpoints']:3d}")

    if res['combined']['length'] > 0:
        length_ratio = layer_total_length / res['combined']['length']
        print(f"  Length ratio (layered/combined): {length_ratio:.2f}x")
    if res['combined']['junctions'] > 0:
        junc_ratio = layer_total_junctions / res['combined']['junctions']
        print(f"  Junction ratio (layered/combined): {junc_ratio:.2f}x")

# ============================================================
# 6. HSV Hue histogram for one sample (diagnostic)
# ============================================================
fig_hist, axes_hist = plt.subplots(1, len(sample_ids), figsize=(5 * len(sample_ids), 4))
for i, sid in enumerate(sample_ids):
    img_path = data_dir / f"{sid}_real_B.png"
    img = np.array(Image.open(img_path).convert("RGB"))
    h_vals, s_vals, v_vals, vmask = analyze_color_distribution(img)

    axes_hist[i].hist(h_vals, bins=180, range=(0, 180), color='gray', alpha=0.7)
    axes_hist[i].axvspan(0, 12, alpha=0.2, color='red', label='Bottom')
    axes_hist[i].axvspan(155, 180, alpha=0.2, color='red')
    axes_hist[i].axvspan(13, 55, alpha=0.2, color='yellow', label='Middle')
    axes_hist[i].axvspan(85, 140, alpha=0.2, color='blue', label='Top')
    axes_hist[i].set_title(f"{sid} Hue Distribution")
    axes_hist[i].set_xlabel("Hue (OpenCV 0-179)")
    axes_hist[i].set_ylabel("Pixel Count")
    axes_hist[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / "hue_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nResults saved to: {output_dir}")
print("  - depth_layer_separation_analysis.png")
print("  - hue_distribution.png")
