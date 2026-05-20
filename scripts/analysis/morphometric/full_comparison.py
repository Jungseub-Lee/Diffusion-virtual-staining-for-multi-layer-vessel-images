"""
Full comparison: Single-color flat vs Multi-color depth-separated
Applied to GT, LBBDM, PBBDM.
With corrected endpoint counting (exclude layer-boundary cuts).
"""

import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from skimage.morphology import skeletonize, remove_small_objects
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data")
output_dir = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output")
output_dir.mkdir(exist_ok=True)


# ============================================================
# Core functions
# ============================================================
def detect_baseline_row(img_rgb, threshold_ratio=0.35):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    vessel = gray > 15
    h, w = vessel.shape
    row_density = np.sum(vessel, axis=1) / w
    baseline_top = h
    for y in range(h - 1, -1, -1):
        if row_density[y] < threshold_ratio:
            baseline_top = y
            break
    return max(0, baseline_top - 5)


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


def count_true_endpoints(skel, own_mask, other_masks, radius=10):
    """
    Count only 'true' endpoints — exclude endpoints that are near
    vessel pixels in adjacent layers (layer-boundary cuts).

    skel: skeleton of this layer
    own_mask: binary mask of this layer
    other_masks: list of binary masks from other layers
    radius: search radius for nearby vessel pixels in other layers
    """
    skel_u8 = skel.astype(np.uint8)
    kernel_sum = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel_u8, -1, kernel_sum) * skel_u8

    # Find endpoint coordinates
    ep_coords = np.argwhere(neighbors == 1)  # (y, x)

    if len(ep_coords) == 0:
        return 0

    # Create dilated union of other layer masks
    other_union = np.zeros_like(own_mask, dtype=bool)
    for om in other_masks:
        other_union |= om

    # Dilate other_union by radius to create search zone
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    other_dilated = cv2.dilate(other_union.astype(np.uint8) * 255, k, iterations=1) > 0

    # True endpoint = endpoint NOT near other layer vessels
    true_count = 0
    for y, x in ep_coords:
        if not other_dilated[y, x]:
            true_count += 1

    return true_count


def analyze_sample_multi(img_rgb, baseline_y):
    """Analyze multi-color image with depth separation."""
    crop = img_rgb[:baseline_y, :]
    bottom, middle, top = separate_layers_hsv(crop)
    masks = [bottom, middle, top]

    skel_b = skeletonize(bottom)
    skel_m = skeletonize(middle)
    skel_t = skeletonize(top)
    skels = [skel_b, skel_m, skel_t]

    # Basic metrics per layer
    metrics = [compute_metrics(s) for s in skels]
    total_length = sum(m["length"] for m in metrics)
    total_junctions = sum(m["junctions"] for m in metrics)

    # Corrected endpoints: exclude layer-boundary cuts
    true_ep_b = count_true_endpoints(skel_b, bottom, [middle, top], radius=10)
    true_ep_m = count_true_endpoints(skel_m, middle, [bottom, top], radius=10)
    true_ep_t = count_true_endpoints(skel_t, top, [bottom, middle], radius=10)
    total_endpoints_corrected = true_ep_b + true_ep_m + true_ep_t
    total_endpoints_raw = sum(m["endpoints"] for m in metrics)

    return {
        "length": total_length,
        "junctions": total_junctions,
        "endpoints_raw": total_endpoints_raw,
        "endpoints_corrected": total_endpoints_corrected,
    }


def analyze_sample_single(img_rgb, baseline_y):
    """Analyze single-color image (flat)."""
    crop = img_rgb[:baseline_y, :]
    mask = get_vessel_mask(crop)
    skel = skeletonize(mask)
    m = compute_metrics(skel)
    return {
        "length": m["length"],
        "junctions": m["junctions"],
        "endpoints": m["endpoints"],
    }


# ============================================================
# Find common samples across GT, LBBDM, PBBDM
# ============================================================
gt_multi_dir = BASE / "Multi-color" / "Pix2pix"
gt_single_dir = BASE / "Single-color" / "Pix2pix"

gt_multi_samples = {f.stem.replace("_real_B", "") for f in gt_multi_dir.glob("*_real_B.png")}
gt_single_samples = {f.stem.replace("_real_B", "") for f in gt_single_dir.glob("*_real_B.png")}

lbbdm_multi_samples = {d.name for d in (BASE / "Multi-color" / "LBBDM").iterdir() if d.is_dir()}
lbbdm_single_samples = {d.name for d in (BASE / "Single-color" / "LBBDM").iterdir() if d.is_dir()}

pbbdm_multi_samples = {d.name for d in (BASE / "Multi-color" / "PBBDM").iterdir() if d.is_dir()}
pbbdm_single_samples = {d.name for d in (BASE / "Single-color" / "PBBDM").iterdir() if d.is_dir()}

common = sorted(
    gt_multi_samples & gt_single_samples &
    lbbdm_multi_samples & lbbdm_single_samples &
    pbbdm_multi_samples & pbbdm_single_samples
)
print(f"Common samples across all sources: {len(common)}")


# ============================================================
# Process all samples
# ============================================================
def load_img(path):
    return np.array(Image.open(path).convert("RGB"))


results = {
    "GT": {"single": [], "multi": []},
    "LBBDM": {"single": [], "multi": []},
    "PBBDM": {"single": [], "multi": []},
}

for i, sid in enumerate(common):
    if (i + 1) % 50 == 0:
        print(f"  Processing {i+1}/{len(common)}...")

    # Use GT multi-color to detect baseline (same field of view)
    gt_multi_img = load_img(gt_multi_dir / f"{sid}_real_B.png")
    baseline_y = detect_baseline_row(gt_multi_img)
    if baseline_y < 30:
        continue

    # --- GT ---
    gt_single_img = load_img(gt_single_dir / f"{sid}_real_B.png")
    results["GT"]["single"].append(analyze_sample_single(gt_single_img, baseline_y))
    results["GT"]["multi"].append(analyze_sample_multi(gt_multi_img, baseline_y))

    # --- LBBDM (use output_0) ---
    lbbdm_multi_img = load_img(BASE / "Multi-color" / "LBBDM" / sid / "output_0.png")
    lbbdm_single_img = load_img(BASE / "Single-color" / "LBBDM" / sid / "output_0.png")
    results["LBBDM"]["single"].append(analyze_sample_single(lbbdm_single_img, baseline_y))
    results["LBBDM"]["multi"].append(analyze_sample_multi(lbbdm_multi_img, baseline_y))

    # --- PBBDM (use output_0) ---
    pbbdm_multi_img = load_img(BASE / "Multi-color" / "PBBDM" / sid / "output_0.png")
    pbbdm_single_img = load_img(BASE / "Single-color" / "PBBDM" / sid / "output_0.png")
    results["PBBDM"]["single"].append(analyze_sample_single(pbbdm_single_img, baseline_y))
    results["PBBDM"]["multi"].append(analyze_sample_multi(pbbdm_multi_img, baseline_y))

n = len(results["GT"]["single"])
print(f"\nProcessed {n} samples")


# ============================================================
# Print summary table
# ============================================================
print("\n" + "=" * 100)
print("COMPARISON: Single-color (flat) vs Multi-color (depth-separated)")
print("Endpoints use corrected counting (layer-boundary cuts excluded)")
print("=" * 100)

for source in ["GT", "LBBDM", "PBBDM"]:
    s = results[source]["single"]
    m = results[source]["multi"]

    s_j = [x["junctions"] for x in s]
    m_j = [x["junctions"] for x in m]
    s_e = [x["endpoints"] for x in s]
    m_e_raw = [x["endpoints_raw"] for x in m]
    m_e_corr = [x["endpoints_corrected"] for x in m]
    s_l = [x["length"] for x in s]
    m_l = [x["length"] for x in m]

    # Paired t-test
    t_j, p_j = stats.ttest_rel(s_j, m_j)
    t_e, p_e = stats.ttest_rel(s_e, m_e_corr)
    t_l, p_l = stats.ttest_rel(s_l, m_l)

    print(f"\n--- {source} ---")
    print(f"{'Metric':<22} {'Single (flat)':<22} {'Multi (depth-sep)':<22} {'p-value':<12}")
    print("-" * 78)
    print(f"{'Junctions':<22} {np.mean(s_j):>7.1f} ± {np.std(s_j):>6.1f}   {np.mean(m_j):>7.1f} ± {np.std(m_j):>6.1f}   {p_j:.2e}")
    print(f"{'Endpoints (raw)':<22} {np.mean(s_e):>7.1f} ± {np.std(s_e):>6.1f}   {np.mean(m_e_raw):>7.1f} ± {np.std(m_e_raw):>6.1f}   -")
    print(f"{'Endpoints (corrected)':<22} {np.mean(s_e):>7.1f} ± {np.std(s_e):>6.1f}   {np.mean(m_e_corr):>7.1f} ± {np.std(m_e_corr):>6.1f}   {p_e:.2e}")
    print(f"{'Length':<22} {np.mean(s_l):>7.1f} ± {np.std(s_l):>6.1f}   {np.mean(m_l):>7.1f} ± {np.std(m_l):>6.1f}   {p_l:.2e}")


# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
metrics_to_plot = ["Junctions", "Endpoints (corrected)", "Length"]
colors_single = "#66bb66"
colors_multi = "#4488ff"

for row, source in enumerate(["GT", "LBBDM", "PBBDM"]):
    s = results[source]["single"]
    m = results[source]["multi"]

    data_pairs = [
        ([x["junctions"] for x in s], [x["junctions"] for x in m]),
        ([x["endpoints"] for x in s], [x["endpoints_corrected"] for x in m]),
        ([x["length"] for x in s], [x["length"] for x in m]),
    ]

    for col, (s_data, m_data) in enumerate(data_pairs):
        ax = axes[row, col]

        # Scatter plot
        ax.scatter(s_data, m_data, alpha=0.4, s=15, c='#4488ff', edgecolors='none')
        maxv = max(max(s_data) if s_data else 1, max(m_data) if m_data else 1) * 1.1
        ax.plot([0, maxv], [0, maxv], 'k--', alpha=0.3, linewidth=1)

        # Linear regression
        if len(s_data) > 2:
            slope, intercept, r, p, se = stats.linregress(s_data, m_data)
            x_fit = np.linspace(0, maxv, 100)
            ax.plot(x_fit, slope * x_fit + intercept, 'r-', alpha=0.6, linewidth=1.5,
                    label=f'R²={r**2:.3f}, p={p:.1e}')
            ax.legend(fontsize=9, loc='upper left')

        ax.set_xlabel(f"Single-color {metrics_to_plot[col]}")
        ax.set_ylabel(f"Multi-color Depth-sep {metrics_to_plot[col]}")
        ax.set_title(f"{source}: {metrics_to_plot[col]}")

plt.tight_layout()
plt.savefig(output_dir / "full_comparison_scatter.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# Bar chart comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for col, metric in enumerate(["Junctions", "Endpoints", "Length"]):
    ax = axes[col]
    x_pos = np.arange(3)
    width = 0.35

    single_means = []
    multi_means = []
    single_stds = []
    multi_stds = []

    for source in ["GT", "LBBDM", "PBBDM"]:
        s = results[source]["single"]
        m = results[source]["multi"]
        if metric == "Junctions":
            sv = [x["junctions"] for x in s]
            mv = [x["junctions"] for x in m]
        elif metric == "Endpoints":
            sv = [x["endpoints"] for x in s]
            mv = [x["endpoints_corrected"] for x in m]
        else:
            sv = [x["length"] for x in s]
            mv = [x["length"] for x in m]
        single_means.append(np.mean(sv))
        multi_means.append(np.mean(mv))
        single_stds.append(np.std(sv))
        multi_stds.append(np.std(mv))

    bars1 = ax.bar(x_pos - width/2, single_means, width, yerr=single_stds,
                   capsize=4, label='Single-color (flat)', color=colors_single, alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, multi_means, width, yerr=multi_stds,
                   capsize=4, label='Multi-color (depth-sep)', color=colors_multi, alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["GT", "LBBDM", "PBBDM"])
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend()

    for bar, m_val in zip(bars1, single_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{m_val:.0f}", ha='center', va='bottom', fontsize=9)
    for bar, m_val in zip(bars2, multi_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{m_val:.0f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "full_comparison_bars.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved:")
print("  - full_comparison_scatter.png")
print("  - full_comparison_bars.png")
