"""
Full pipeline:
1. Run comparison with bridging on all samples (GT, LBBDM, PBBDM)
2. Statistical test figures
3. Publication-ready representative figure
"""

import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from scipy.spatial.distance import cdist
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data")
output_dir = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output")


# ============================================================
# Core functions (same as before + bridging)
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
    h, s = hsv[:,:,0], hsv[:,:,1]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fg = (gray > 15) & (s > 30)
    bottom = ((h <= 12) | (h >= 155)) & fg
    middle = ((h >= 13) & (h <= 55)) & fg
    top = ((h >= 85) & (h <= 140)) & fg
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    def clean(m):
        u = m.astype(np.uint8)*255
        u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k, iterations=2)
        u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k, iterations=1)
        return remove_small_objects(u > 0, min_size=min_area)
    return clean(bottom), clean(middle), clean(top)


def get_vessel_mask(img_rgb, min_area=50):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    vessel = gray > 15
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    u = vessel.astype(np.uint8)*255
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k, iterations=2)
    u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k, iterations=1)
    return remove_small_objects(u > 0, min_size=min_area)


def compute_metrics(skeleton):
    total_length = np.sum(skeleton)
    if total_length == 0:
        return {"length": 0, "junctions": 0, "endpoints": 0}
    skel_u8 = skeleton.astype(np.uint8)
    ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel_u8, -1, ks) * skel_u8
    return {
        "length": int(total_length),
        "junctions": int(np.sum(neighbors >= 3)),
        "endpoints": int(np.sum(neighbors == 1))
    }


def bridge_skeletons(skels, masks, radius=15):
    layer_pairs = [(0,1),(1,2)]
    connected = np.zeros_like(skels[0], dtype=np.uint8)
    for s in skels:
        connected |= s.astype(np.uint8)

    for li, lj in layer_pairs:
        skel_u8_i = skels[li].astype(np.uint8)
        skel_u8_j = skels[lj].astype(np.uint8)
        ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
        ni = cv2.filter2D(skel_u8_i, -1, ks) * skel_u8_i
        nj = cv2.filter2D(skel_u8_j, -1, ks) * skel_u8_j
        ep_i = np.argwhere(ni == 1)
        ep_j = np.argwhere(nj == 1)
        if len(ep_i) == 0 or len(ep_j) == 0:
            continue

        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
        md_j = cv2.dilate(masks[lj].astype(np.uint8)*255, dk, iterations=1) > 0
        md_i = cv2.dilate(masks[li].astype(np.uint8)*255, dk, iterations=1) > 0

        vi = np.array([p for p in ep_i if md_j[p[0], p[1]]])
        vj = np.array([p for p in ep_j if md_i[p[0], p[1]]])
        if len(vi) == 0 or len(vj) == 0:
            continue

        dists = cdist(vi, vj)
        used_i, used_j = set(), set()
        for idx in np.argsort(dists.ravel()):
            ii, jj = idx // dists.shape[1], idx % dists.shape[1]
            if dists[ii,jj] > radius: break
            if ii in used_i or jj in used_j: continue
            cv2.line(connected, (int(vi[ii][1]),int(vi[ii][0])), (int(vj[jj][1]),int(vj[jj][0])), 1, 1)
            used_i.add(ii); used_j.add(jj)

    return skeletonize(connected > 0)


def analyze_multi_bridged(img_rgb, baseline_y):
    crop = img_rgb[:baseline_y, :]
    bottom, middle, top = separate_layers_hsv(crop)
    masks = [bottom, middle, top]
    skels = [skeletonize(m) for m in masks]

    # Per-layer metrics (before bridging)
    layer_metrics = [compute_metrics(s) for s in skels]
    sep_j = sum(m["junctions"] for m in layer_metrics)
    sep_l = sum(m["length"] for m in layer_metrics)

    # Bridged skeleton
    bridged = bridge_skeletons(skels, masks, radius=15)
    m_bridged = compute_metrics(bridged)

    return {
        "sep_junctions": sep_j,
        "sep_length": sep_l,
        "bridged_junctions": m_bridged["junctions"],
        "bridged_endpoints": m_bridged["endpoints"],
        "bridged_length": m_bridged["length"],
    }


def analyze_single(img_rgb, baseline_y):
    crop = img_rgb[:baseline_y, :]
    mask = get_vessel_mask(crop)
    skel = skeletonize(mask)
    return compute_metrics(skel)


# ============================================================
# Process all common samples
# ============================================================
gt_multi_dir = BASE / "Multi-color" / "Pix2pix"
gt_single_dir = BASE / "Single-color" / "Pix2pix"

all_sets = [
    {f.stem.replace("_real_B","") for f in gt_multi_dir.glob("*_real_B.png")},
    {f.stem.replace("_real_B","") for f in gt_single_dir.glob("*_real_B.png")},
    {d.name for d in (BASE/"Multi-color"/"LBBDM").iterdir() if d.is_dir()},
    {d.name for d in (BASE/"Single-color"/"LBBDM").iterdir() if d.is_dir()},
    {d.name for d in (BASE/"Multi-color"/"PBBDM").iterdir() if d.is_dir()},
    {d.name for d in (BASE/"Single-color"/"PBBDM").iterdir() if d.is_dir()},
]
common = sorted(set.intersection(*all_sets))
print(f"Common samples: {len(common)}")

results = {src: {"single": [], "multi": []} for src in ["GT","LBBDM","PBBDM"]}
valid_sids = []

for i, sid in enumerate(common):
    if (i+1) % 50 == 0: print(f"  {i+1}/{len(common)}...")

    gt_m = np.array(Image.open(gt_multi_dir / f"{sid}_real_B.png").convert("RGB"))
    by = detect_baseline_row(gt_m)
    if by < 30: continue
    valid_sids.append(sid)

    def load(p): return np.array(Image.open(p).convert("RGB"))

    paths = {
        "GT": (gt_multi_dir / f"{sid}_real_B.png", gt_single_dir / f"{sid}_real_B.png"),
        "LBBDM": (BASE/"Multi-color"/"LBBDM"/sid/"output_0.png", BASE/"Single-color"/"LBBDM"/sid/"output_0.png"),
        "PBBDM": (BASE/"Multi-color"/"PBBDM"/sid/"output_0.png", BASE/"Single-color"/"PBBDM"/sid/"output_0.png"),
    }

    for src, (mp, sp) in paths.items():
        results[src]["multi"].append(analyze_multi_bridged(load(mp), by))
        results[src]["single"].append(analyze_single(load(sp), by))

n = len(results["GT"]["single"])
print(f"Processed: {n} samples")


# ============================================================
# Statistical summary
# ============================================================
print("\n" + "="*100)
print("FINAL COMPARISON: Single-color (flat) vs Multi-color (depth-separated, bridged)")
print("="*100)

stat_data = {}  # for plotting

for src in ["GT","LBBDM","PBBDM"]:
    s = results[src]["single"]
    m = results[src]["multi"]

    s_j = [x["junctions"] for x in s]
    m_j = [x["bridged_junctions"] for x in m]
    s_e = [x["endpoints"] for x in s]
    m_e = [x["bridged_endpoints"] for x in m]
    s_l = [x["length"] for x in s]
    m_l = [x["bridged_length"] for x in m]

    t_j, p_j = stats.ttest_rel(s_j, m_j)
    t_e, p_e = stats.ttest_rel(s_e, m_e)
    t_l, p_l = stats.ttest_rel(s_l, m_l)
    # Wilcoxon as well
    w_j, wp_j = stats.wilcoxon(s_j, m_j)
    w_e, wp_e = stats.wilcoxon(s_e, m_e)
    w_l, wp_l = stats.wilcoxon(s_l, m_l)

    stat_data[src] = {
        "s_j": s_j, "m_j": m_j,
        "s_e": s_e, "m_e": m_e,
        "s_l": s_l, "m_l": m_l,
        "p_j": p_j, "p_e": p_e, "p_l": p_l,
        "wp_j": wp_j, "wp_e": wp_e, "wp_l": wp_l,
    }

    print(f"\n--- {src} ---")
    print(f"{'Metric':<15} {'Single':<20} {'Multi (bridged)':<20} {'t-test p':<12} {'Wilcoxon p':<12}")
    print("-"*79)
    print(f"{'Junctions':<15} {np.mean(s_j):>6.1f} ± {np.std(s_j):>5.1f}   {np.mean(m_j):>6.1f} ± {np.std(m_j):>5.1f}   {p_j:.2e}    {wp_j:.2e}")
    print(f"{'Endpoints':<15} {np.mean(s_e):>6.1f} ± {np.std(s_e):>5.1f}   {np.mean(m_e):>6.1f} ± {np.std(m_e):>5.1f}   {p_e:.2e}    {wp_e:.2e}")
    print(f"{'Length':<15} {np.mean(s_l):>6.1f} ± {np.std(s_l):>5.1f}   {np.mean(m_l):>6.1f} ± {np.std(m_l):>5.1f}   {p_l:.2e}    {wp_l:.2e}")


# ============================================================
# Figure A: Statistical comparison (paired plots)
# ============================================================
def pval_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "n.s."


fig, axes = plt.subplots(3, 3, figsize=(16, 14))
metric_labels = ["Junctions", "Endpoints", "Vessel Length (px)"]
metric_keys = [("s_j","m_j","p_j"), ("s_e","m_e","p_e"), ("s_l","m_l","p_l")]

for row, src in enumerate(["GT","LBBDM","PBBDM"]):
    for col, (sk, mk, pk) in enumerate(metric_keys):
        ax = axes[row, col]
        sv = stat_data[src][sk]
        mv = stat_data[src][mk]
        pv = stat_data[src][pk]

        ax.scatter(sv, mv, alpha=0.35, s=12, c='#3366cc', edgecolors='none')
        maxv = max(max(sv) if sv else 1, max(mv) if mv else 1) * 1.1
        ax.plot([0, maxv], [0, maxv], 'k--', alpha=0.3, lw=1)

        if len(sv) > 2:
            slope, intercept, r, p_r, se = stats.linregress(sv, mv)
            xf = np.linspace(0, maxv, 100)
            ax.plot(xf, slope*xf+intercept, 'r-', alpha=0.6, lw=1.5)
            ax.text(0.05, 0.92, f'R²={r**2:.3f}\n{pval_stars(pv)} (p={pv:.1e})',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel(f"Single-color")
        ax.set_ylabel(f"Multi-color (depth-sep)")
        ax.set_title(f"{src}: {metric_labels[col]}")

plt.tight_layout()
plt.savefig(output_dir / "stat_scatter_bridged.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nSaved: stat_scatter_bridged.png")


# ============================================================
# Figure B: Bar chart with significance
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
colors_s = '#7BC67E'
colors_m = '#5B9BD5'

for col, (metric_name, sk, mk, pk) in enumerate([
    ("Junctions", "s_j", "m_j", "p_j"),
    ("Endpoints", "s_e", "m_e", "p_e"),
    ("Vessel Length", "s_l", "m_l", "p_l"),
]):
    ax = axes[col]
    x = np.arange(3)
    w = 0.3

    s_means, m_means, s_stds, m_stds, pvals = [], [], [], [], []
    for src in ["GT","LBBDM","PBBDM"]:
        s_means.append(np.mean(stat_data[src][sk]))
        m_means.append(np.mean(stat_data[src][mk]))
        s_stds.append(np.std(stat_data[src][sk]))
        m_stds.append(np.std(stat_data[src][mk]))
        pvals.append(stat_data[src][pk])

    b1 = ax.bar(x-w/2, s_means, w, yerr=s_stds, capsize=3,
                label='Single-color', color=colors_s, alpha=0.85, edgecolor='white')
    b2 = ax.bar(x+w/2, m_means, w, yerr=m_stds, capsize=3,
                label='Multi-color\n(depth-sep)', color=colors_m, alpha=0.85, edgecolor='white')

    # Significance brackets
    for i_x in range(3):
        max_h = max(s_means[i_x]+s_stds[i_x], m_means[i_x]+m_stds[i_x])
        bracket_h = max_h * 1.08
        ax.plot([i_x-w/2, i_x-w/2, i_x+w/2, i_x+w/2],
                [bracket_h, bracket_h*1.03, bracket_h*1.03, bracket_h],
                'k-', lw=0.8)
        ax.text(i_x, bracket_h*1.04, pval_stars(pvals[i_x]),
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["GT", "LBBDM", "PBBDM"])
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name)
    ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig(output_dir / "stat_bars_bridged.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: stat_bars_bridged.png")


# ============================================================
# Figure C: Publication figure — representative sample
# ============================================================
rep_sid = "23-13-512"  # well-balanced sample

def dilate(s, t=4):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t,t))
    return cv2.dilate(s.astype(np.uint8)*255, k, iterations=1) > 0

def make_depth_skel_overlay(img_crop, baseline_y):
    crop = img_crop[:baseline_y, :]
    dim = (crop * 0.3).astype(np.uint8)
    bottom, middle, top = separate_layers_hsv(crop)
    masks = [bottom, middle, top]
    skels = [skeletonize(m) for m in masks]
    bridged = bridge_skeletons(skels, masks, radius=15)

    # Color bridged skeleton by layer
    vis = dim.copy()
    thick = dilate(bridged, 5)
    colors = [(255,80,80),(255,255,60),(80,140,255)]
    for si, mask in enumerate(masks):
        region = thick & mask
        vis[region] = colors[si]
    # Uncolored bridged parts
    unc = thick & ~(masks[0]|masks[1]|masks[2])
    vis[unc] = (0,255,200)
    return vis, compute_metrics(bridged)


def make_single_skel_overlay(img_crop, baseline_y):
    crop = img_crop[:baseline_y, :]
    dim = (crop * 0.3).astype(np.uint8)
    mask = get_vessel_mask(crop)
    skel = skeletonize(mask)
    vis = dim.copy()
    vis[dilate(skel, 5)] = (0,255,0)
    return vis, compute_metrics(skel)


fig, axes = plt.subplots(3, 5, figsize=(25, 15))

sources = ["GT", "LBBDM", "PBBDM"]
source_paths = {
    "GT": {
        "multi": gt_multi_dir / f"{rep_sid}_real_B.png",
        "single": gt_single_dir / f"{rep_sid}_real_B.png",
    },
    "LBBDM": {
        "multi": BASE/"Multi-color"/"LBBDM"/rep_sid/"output_0.png",
        "single": BASE/"Single-color"/"LBBDM"/rep_sid/"output_0.png",
    },
    "PBBDM": {
        "multi": BASE/"Multi-color"/"PBBDM"/rep_sid/"output_0.png",
        "single": BASE/"Single-color"/"PBBDM"/rep_sid/"output_0.png",
    },
}

gt_m_img = np.array(Image.open(source_paths["GT"]["multi"]).convert("RGB"))
by = detect_baseline_row(gt_m_img)

for row, src in enumerate(sources):
    multi_img = np.array(Image.open(source_paths[src]["multi"]).convert("RGB"))
    single_img = np.array(Image.open(source_paths[src]["single"]).convert("RGB"))

    # Col 0: Multi-color original
    axes[row, 0].imshow(multi_img[:by, :])
    axes[row, 0].set_title(f"{src}: Multi-color GT" if src == "GT" else f"{src}: Multi-color Prediction")

    # Col 1: Single-color original
    axes[row, 1].imshow(single_img[:by, :])
    axes[row, 1].set_title(f"{src}: Single-color GT" if src == "GT" else f"{src}: Single-color Prediction")

    # Col 2: Single skeleton
    s_vis, s_m = make_single_skel_overlay(single_img, by)
    axes[row, 2].imshow(s_vis)
    axes[row, 2].set_title(f"Single Skeleton\nJ={s_m['junctions']}, E={s_m['endpoints']}, L={s_m['length']}")

    # Col 3: Multi-color layer masks
    crop = multi_img[:by, :]
    bottom, middle, top = separate_layers_hsv(crop)
    vis_layers = np.zeros_like(crop)
    vis_layers[bottom] = [255,80,80]
    vis_layers[middle] = [255,255,60]
    vis_layers[top] = [80,140,255]
    axes[row, 3].imshow(vis_layers)
    axes[row, 3].set_title("Depth Layer Separation\n(R=Bottom, Y=Mid, B=Top)")

    # Col 4: Depth-separated bridged skeleton
    m_vis, m_m = make_depth_skel_overlay(multi_img, by)
    axes[row, 4].imshow(m_vis)
    axes[row, 4].set_title(f"Depth-sep Skeleton (bridged)\nJ={m_m['junctions']}, E={m_m['endpoints']}, L={m_m['length']}")

for ax in axes.flat:
    ax.axis("off")

# Row labels
for row, src in enumerate(sources):
    axes[row, 0].text(-0.02, 0.5, src, transform=axes[row,0].transAxes,
                      fontsize=16, fontweight='bold', va='center', ha='right', rotation=90)

plt.suptitle(f"Representative Sample: {rep_sid}", fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(output_dir / "paper_figure_representative.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: paper_figure_representative.png")

print("\nAll figures saved.")
