"""
Compare single-color vs multi-color depth-separated morphometric analysis.
- Exclude baseline (ECM interface) horizontal region
- Multi-color: separate by depth layers, skeletonize per layer
- Single-color: skeletonize as one flat projection
- Compare: junctions, endpoints, length
"""

import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from skimage.morphology import skeletonize, remove_small_objects
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11

multi_dir = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data\Multi-color\Pix2pix")
single_dir = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data\Single-color\Pix2pix")
output_dir = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output")
output_dir.mkdir(exist_ok=True)

# Use all available common samples
multi_samples = {f.stem.replace("_real_B","") for f in multi_dir.glob("*_real_B.png")}
single_samples = {f.stem.replace("_real_B","") for f in single_dir.glob("*_real_B.png")}
common_samples = sorted(multi_samples & single_samples)
print(f"Common samples: {len(common_samples)}")


def detect_baseline_row(img_rgb, threshold_ratio=0.4):
    """
    Detect the baseline (ECM interface) as the region where vessel pixels
    span most of the image width — typically a horizontal band at bottom.
    Returns the y-coordinate above which to analyze (crop out baseline).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    vessel = gray > 15
    h, w = vessel.shape

    # For each row, compute the fraction of vessel pixels
    row_density = np.sum(vessel, axis=1) / w

    # Find baseline: rows near bottom with high density (>threshold_ratio)
    # Scan from bottom upward
    baseline_top = h  # default: no cropping
    for y in range(h - 1, -1, -1):
        if row_density[y] < threshold_ratio:
            baseline_top = y
            break

    # Add a small margin above the baseline
    baseline_top = max(0, baseline_top - 5)
    return baseline_top


def separate_layers_hsv(img_rgb, min_area=50):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fg = (gray > 15) & (s > 30)

    bottom = ((h <= 12) | (h >= 155)) & fg
    middle = ((h >= 13) & (h <= 55)) & fg
    top = ((h >= 85) & (h <= 140)) & fg

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def clean(m):
        u = m.astype(np.uint8) * 255
        u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k, iterations=2)
        u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k, iterations=1)
        return remove_small_objects(u > 0, min_size=min_area)

    return clean(bottom), clean(middle), clean(top)


def get_vessel_mask(img_rgb, min_area=50):
    """Simple vessel mask for single-color (green channel dominant)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    vessel = gray > 15
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    u = vessel.astype(np.uint8) * 255
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k, iterations=2)
    u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k, iterations=1)
    return remove_small_objects(u > 0, min_size=min_area)


def compute_metrics(skeleton):
    total_length = np.sum(skeleton)
    if total_length == 0:
        return {"length": 0, "junctions": 0, "endpoints": 0}

    skel_u8 = skeleton.astype(np.uint8)
    kernel_sum = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel_u8, -1, kernel_sum) * skel_u8

    return {
        "length": int(total_length),
        "junctions": int(np.sum(neighbors >= 3)),
        "endpoints": int(np.sum(neighbors == 1))
    }


# ============================================================
# Run comparison on all common samples
# ============================================================
results = []

for sid in common_samples:
    # Load multi-color GT
    multi_img = np.array(Image.open(multi_dir / f"{sid}_real_B.png").convert("RGB"))
    single_img = np.array(Image.open(single_dir / f"{sid}_real_B.png").convert("RGB"))

    # Detect and crop baseline
    baseline_y = detect_baseline_row(multi_img, threshold_ratio=0.35)
    multi_crop = multi_img[:baseline_y, :]
    single_crop = single_img[:baseline_y, :]

    if baseline_y < 30:  # skip if almost entire image is baseline
        continue

    # --- Multi-color: depth-separated ---
    bottom, middle, top = separate_layers_hsv(multi_crop)
    skel_b = skeletonize(bottom)
    skel_m = skeletonize(middle)
    skel_t = skeletonize(top)

    m_bottom = compute_metrics(skel_b)
    m_middle = compute_metrics(skel_m)
    m_top = compute_metrics(skel_t)

    # Sum of layer-separated metrics
    multi_length = m_bottom["length"] + m_middle["length"] + m_top["length"]
    multi_junctions = m_bottom["junctions"] + m_middle["junctions"] + m_top["junctions"]
    multi_endpoints = m_bottom["endpoints"] + m_middle["endpoints"] + m_top["endpoints"]

    # Multi-color combined (flat projection - same as single would be)
    multi_combined_mask = bottom | middle | top
    skel_multi_combined = skeletonize(multi_combined_mask)
    m_multi_combined = compute_metrics(skel_multi_combined)

    # --- Single-color: flat ---
    single_mask = get_vessel_mask(single_crop)
    skel_single = skeletonize(single_mask)
    m_single = compute_metrics(skel_single)

    results.append({
        "sample": sid,
        "baseline_y": baseline_y,
        # Single-color
        "single_length": m_single["length"],
        "single_junctions": m_single["junctions"],
        "single_endpoints": m_single["endpoints"],
        # Multi-color combined (flat)
        "multi_flat_length": m_multi_combined["length"],
        "multi_flat_junctions": m_multi_combined["junctions"],
        "multi_flat_endpoints": m_multi_combined["endpoints"],
        # Multi-color depth-separated
        "multi_sep_length": multi_length,
        "multi_sep_junctions": multi_junctions,
        "multi_sep_endpoints": multi_endpoints,
        # Per-layer
        "bottom_junctions": m_bottom["junctions"],
        "middle_junctions": m_middle["junctions"],
        "top_junctions": m_top["junctions"],
        "bottom_length": m_bottom["length"],
        "middle_length": m_middle["length"],
        "top_length": m_top["length"],
    })

print(f"\nProcessed {len(results)} samples")

# ============================================================
# Summary statistics
# ============================================================
single_j = [r["single_junctions"] for r in results]
multi_flat_j = [r["multi_flat_junctions"] for r in results]
multi_sep_j = [r["multi_sep_junctions"] for r in results]

single_e = [r["single_endpoints"] for r in results]
multi_flat_e = [r["multi_flat_endpoints"] for r in results]
multi_sep_e = [r["multi_sep_endpoints"] for r in results]

single_l = [r["single_length"] for r in results]
multi_flat_l = [r["multi_flat_length"] for r in results]
multi_sep_l = [r["multi_sep_length"] for r in results]

print("\n" + "=" * 80)
print("AGGREGATE COMPARISON (mean ± std)")
print("=" * 80)
print(f"{'Metric':<25} {'Single-color':<22} {'Multi-flat':<22} {'Multi-separated':<22}")
print("-" * 91)
print(f"{'Junctions':<25} {np.mean(single_j):>7.1f} ± {np.std(single_j):>6.1f}   {np.mean(multi_flat_j):>7.1f} ± {np.std(multi_flat_j):>6.1f}   {np.mean(multi_sep_j):>7.1f} ± {np.std(multi_sep_j):>6.1f}")
print(f"{'Endpoints':<25} {np.mean(single_e):>7.1f} ± {np.std(single_e):>6.1f}   {np.mean(multi_flat_e):>7.1f} ± {np.std(multi_flat_e):>6.1f}   {np.mean(multi_sep_e):>7.1f} ± {np.std(multi_sep_e):>6.1f}")
print(f"{'Length':<25} {np.mean(single_l):>7.1f} ± {np.std(single_l):>6.1f}   {np.mean(multi_flat_l):>7.1f} ± {np.std(multi_flat_l):>6.1f}   {np.mean(multi_sep_l):>7.1f} ± {np.std(multi_sep_l):>6.1f}")

# Junction reduction ratio
flat_to_sep_ratio = [r["multi_sep_junctions"] / r["multi_flat_junctions"]
                     if r["multi_flat_junctions"] > 0 else np.nan for r in results]
flat_to_sep_ratio = [x for x in flat_to_sep_ratio if not np.isnan(x)]
print(f"\nJunction ratio (depth-separated / flat): {np.mean(flat_to_sep_ratio):.3f} ± {np.std(flat_to_sep_ratio):.3f}")
print(f"  → Depth separation removes ~{(1-np.mean(flat_to_sep_ratio))*100:.1f}% of junctions (false overlaps)")

# ============================================================
# Visualization: scatter + bar plots
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# --- Row 0: Scatter plots (single vs multi-sep) ---
# Junctions
ax = axes[0, 0]
ax.scatter(single_j, multi_sep_j, alpha=0.5, s=20, c='#4488ff', edgecolors='none')
maxv = max(max(single_j), max(multi_sep_j)) * 1.1
ax.plot([0, maxv], [0, maxv], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel("Single-color Junctions")
ax.set_ylabel("Multi-color Depth-separated Junctions")
ax.set_title("Junctions: Single vs Depth-separated")
ax.legend()

# Endpoints
ax = axes[0, 1]
ax.scatter(single_e, multi_sep_e, alpha=0.5, s=20, c='#ff6644', edgecolors='none')
maxv = max(max(single_e), max(multi_sep_e)) * 1.1
ax.plot([0, maxv], [0, maxv], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel("Single-color Endpoints")
ax.set_ylabel("Multi-color Depth-separated Endpoints")
ax.set_title("Endpoints: Single vs Depth-separated")
ax.legend()

# Length
ax = axes[0, 2]
ax.scatter(single_l, multi_sep_l, alpha=0.5, s=20, c='#44bb44', edgecolors='none')
maxv = max(max(single_l), max(multi_sep_l)) * 1.1
ax.plot([0, maxv], [0, maxv], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel("Single-color Length")
ax.set_ylabel("Multi-color Depth-separated Length")
ax.set_title("Length: Single vs Depth-separated")
ax.legend()

# --- Row 1: Junction comparison ---
# Bar: mean junctions across 3 approaches
ax = axes[1, 0]
means = [np.mean(single_j), np.mean(multi_flat_j), np.mean(multi_sep_j)]
stds = [np.std(single_j), np.std(multi_flat_j), np.std(multi_sep_j)]
bars = ax.bar(["Single-color\n(flat)", "Multi-color\n(flat/combined)", "Multi-color\n(depth-separated)"],
              means, yerr=stds, capsize=5,
              color=["#66bb66", "#ff8844", "#4488ff"], alpha=0.8)
ax.set_ylabel("Mean Junctions per Sample")
ax.set_title("Junction Count Comparison")
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{m:.0f}", ha='center', va='bottom', fontsize=10)

# Histogram of junction reduction ratio
ax = axes[1, 1]
ax.hist(flat_to_sep_ratio, bins=20, color='#4488ff', alpha=0.7, edgecolor='white')
ax.axvline(np.mean(flat_to_sep_ratio), color='red', linestyle='--',
           label=f'Mean: {np.mean(flat_to_sep_ratio):.2f}')
ax.set_xlabel("Junction Ratio (depth-separated / flat)")
ax.set_ylabel("Count")
ax.set_title(f"False Junction Removal\n({(1-np.mean(flat_to_sep_ratio))*100:.0f}% reduction on average)")
ax.legend()

# Endpoints comparison bar
ax = axes[1, 2]
means_e = [np.mean(single_e), np.mean(multi_flat_e), np.mean(multi_sep_e)]
stds_e = [np.std(single_e), np.std(multi_flat_e), np.std(multi_sep_e)]
bars = ax.bar(["Single-color\n(flat)", "Multi-color\n(flat/combined)", "Multi-color\n(depth-separated)"],
              means_e, yerr=stds_e, capsize=5,
              color=["#66bb66", "#ff8844", "#4488ff"], alpha=0.8)
ax.set_ylabel("Mean Endpoints per Sample")
ax.set_title("Endpoint Count Comparison")
for bar, m in zip(bars, means_e):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{m:.0f}", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "single_vs_multi_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# Visual comparison: 4 example samples side by side
# ============================================================
vis_samples = ["1-0-512", "1-5-512", "1-10-512", "1-15-512"]
fig, axes = plt.subplots(len(vis_samples), 5, figsize=(25, 5 * len(vis_samples)))

for i, sid in enumerate(vis_samples):
    multi_img = np.array(Image.open(multi_dir / f"{sid}_real_B.png").convert("RGB"))
    single_img = np.array(Image.open(single_dir / f"{sid}_real_B.png").convert("RGB"))
    baseline_y = detect_baseline_row(multi_img, threshold_ratio=0.35)
    multi_crop = multi_img[:baseline_y, :]
    single_crop = single_img[:baseline_y, :]

    bottom, middle, top = separate_layers_hsv(multi_crop)
    skel_b = skeletonize(bottom)
    skel_m = skeletonize(middle)
    skel_t = skeletonize(top)

    single_mask = get_vessel_mask(single_crop)
    skel_single = skeletonize(single_mask)

    multi_combined = bottom | middle | top
    skel_multi_flat = skeletonize(multi_combined)

    def dilate(s, t=4):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
        return cv2.dilate(s.astype(np.uint8)*255, k, iterations=1) > 0

    dim_m = (multi_crop * 0.25).astype(np.uint8)
    dim_s = (single_crop * 0.25).astype(np.uint8)

    # Col 0: Multi-color original
    axes[i, 0].imshow(multi_crop)
    axes[i, 0].set_title(f"{sid}\nMulti-color GT (baseline removed)")

    # Col 1: Single-color original
    axes[i, 1].imshow(single_crop)
    axes[i, 1].set_title(f"Single-color GT")

    # Col 2: Single-color skeleton
    sk_s = dim_s.copy()
    thick_single = dilate(skel_single, 5)
    sk_s[thick_single] = [0, 255, 0]
    m_s = compute_metrics(skel_single)
    axes[i, 2].imshow(sk_s)
    axes[i, 2].set_title(f"Single-color Skeleton\nJ={m_s['junctions']}, E={m_s['endpoints']}, L={m_s['length']}")

    # Col 3: Multi-color flat skeleton
    sk_mf = dim_m.copy()
    thick_flat = dilate(skel_multi_flat, 5)
    sk_mf[thick_flat] = [0, 255, 0]
    m_mf = compute_metrics(skel_multi_flat)
    axes[i, 3].imshow(sk_mf)
    axes[i, 3].set_title(f"Multi-color Flat Skeleton\nJ={m_mf['junctions']}, E={m_mf['endpoints']}, L={m_mf['length']}")

    # Col 4: Multi-color depth-separated skeleton
    sk_ms = dim_m.copy()
    sk_ms[dilate(skel_b, 5)] = [255, 80, 80]
    sk_ms[dilate(skel_m, 5)] = [255, 255, 60]
    sk_ms[dilate(skel_t, 5)] = [80, 140, 255]
    m_b = compute_metrics(skel_b)
    m_mi = compute_metrics(skel_m)
    m_t = compute_metrics(skel_t)
    total_j = m_b['junctions'] + m_mi['junctions'] + m_t['junctions']
    total_e = m_b['endpoints'] + m_mi['endpoints'] + m_t['endpoints']
    total_l = m_b['length'] + m_mi['length'] + m_t['length']
    axes[i, 4].imshow(sk_ms)
    axes[i, 4].set_title(f"Depth-separated Skeleton\nJ={total_j}, E={total_e}, L={total_l}")

    for j in range(5):
        axes[i, j].axis("off")

plt.tight_layout()
plt.savefig(output_dir / "visual_comparison_single_vs_multi.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved:")
print("  - single_vs_multi_comparison.png (statistics)")
print("  - visual_comparison_single_vs_multi.png (example images)")
