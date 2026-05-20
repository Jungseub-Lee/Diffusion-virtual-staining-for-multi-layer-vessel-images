"""
Final analysis: skeleton on full image, but exclude baseline region from metric counting.
Baseline = dense horizontal vessel band at bottom (ECM interface).
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
# Baseline detection: create exclusion mask (not crop)
# ============================================================
def make_baseline_exclusion_mask(img_rgb, threshold_ratio=0.35):
    """
    Returns a boolean mask where True = valid region (above baseline),
    False = baseline region to exclude from counting.
    Skeleton is still computed on full image.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    vessel = gray > 15
    h, w = vessel.shape
    row_density = np.sum(vessel, axis=1) / w

    # Scan from bottom: find where continuous dense region starts
    baseline_top = h
    for y in range(h - 1, -1, -1):
        if row_density[y] < threshold_ratio:
            baseline_top = y
            break

    # Create mask: everything above baseline_top is valid
    valid_mask = np.ones((h, w), dtype=bool)
    valid_mask[baseline_top:, :] = False
    return valid_mask


# ============================================================
# Core functions
# ============================================================
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


def compute_metrics_masked(skeleton, valid_mask):
    """Compute metrics but only count features within valid_mask."""
    skel_u8 = skeleton.astype(np.uint8)
    ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel_u8, -1, ks) * skel_u8

    # Only count within valid region
    valid_skel = skeleton & valid_mask
    valid_neighbors = neighbors * valid_mask.astype(np.uint8)

    return {
        "length": int(np.sum(valid_skel)),
        "junctions": int(np.sum(valid_neighbors >= 3)),
        "endpoints": int(np.sum(valid_neighbors == 1)),
    }


def bridge_skeletons(skels, masks, radius=15):
    layer_pairs = [(0,1),(1,2)]
    connected = np.zeros_like(skels[0], dtype=np.uint8)
    for s in skels:
        connected |= s.astype(np.uint8)
    bridges = []

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
            y1,x1 = int(vi[ii][0]), int(vi[ii][1])
            y2,x2 = int(vj[jj][0]), int(vj[jj][1])
            cv2.line(connected, (x1,y1), (x2,y2), 1, 1)
            bridges.append(((y1,x1),(y2,x2)))
            used_i.add(ii); used_j.add(jj)

    return skeletonize(connected > 0), bridges


def analyze_multi(img_rgb, valid_mask):
    bottom, middle, top = separate_layers_hsv(img_rgb)
    masks = [bottom, middle, top]
    skels = [skeletonize(m) for m in masks]
    bridged, bridges = bridge_skeletons(skels, masks, radius=15)
    return compute_metrics_masked(bridged, valid_mask), masks, skels, bridged, bridges


def analyze_single(img_rgb, valid_mask):
    mask = get_vessel_mask(img_rgb)
    skel = skeletonize(mask)
    return compute_metrics_masked(skel, valid_mask), skel


# ============================================================
# Find common samples
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


# ============================================================
# Process all
# ============================================================
def load(p): return np.array(Image.open(p).convert("RGB"))

results = {src: {"single": [], "multi": []} for src in ["GT","LBBDM","PBBDM"]}

for i, sid in enumerate(common):
    if (i+1) % 50 == 0: print(f"  {i+1}/{len(common)}...")

    # Use GT multi-color to detect baseline (same FOV for all)
    gt_m = load(gt_multi_dir / f"{sid}_real_B.png")
    valid_mask = make_baseline_exclusion_mask(gt_m)

    paths = {
        "GT": (gt_multi_dir/f"{sid}_real_B.png", gt_single_dir/f"{sid}_real_B.png"),
        "LBBDM": (BASE/"Multi-color"/"LBBDM"/sid/"output_0.png", BASE/"Single-color"/"LBBDM"/sid/"output_0.png"),
        "PBBDM": (BASE/"Multi-color"/"PBBDM"/sid/"output_0.png", BASE/"Single-color"/"PBBDM"/sid/"output_0.png"),
    }

    for src, (mp, sp) in paths.items():
        m_metrics, _, _, _, _ = analyze_multi(load(mp), valid_mask)
        s_metrics, _ = analyze_single(load(sp), valid_mask)
        results[src]["multi"].append(m_metrics)
        results[src]["single"].append(s_metrics)

n = len(results["GT"]["single"])
print(f"Processed: {n}")


# ============================================================
# Stats
# ============================================================
print("\n" + "="*100)
print("FINAL: Skeleton on full image, baseline excluded from counting")
print("="*100)

stat_data = {}
for src in ["GT","LBBDM","PBBDM"]:
    s = results[src]["single"]
    m = results[src]["multi"]
    d = {}
    for key, label in [("junctions","j"),("endpoints","e"),("length","l")]:
        sv = [x[key] for x in s]
        mv = [x[key] for x in m]
        t, p = stats.ttest_rel(sv, mv)
        w, wp = stats.wilcoxon(sv, mv)
        d[f"s_{label}"] = sv
        d[f"m_{label}"] = mv
        d[f"p_{label}"] = p
        d[f"wp_{label}"] = wp
    stat_data[src] = d

    s_j,m_j = d["s_j"],d["m_j"]
    s_e,m_e = d["s_e"],d["m_e"]
    s_l,m_l = d["s_l"],d["m_l"]
    print(f"\n--- {src} ---")
    print(f"{'Metric':<12} {'Single':<20} {'Multi (depth-sep)':<20} {'t-test p':<12} {'Wilcoxon p'}")
    print("-"*76)
    print(f"{'Junctions':<12} {np.mean(s_j):>6.1f} +/- {np.std(s_j):>5.1f}   {np.mean(m_j):>6.1f} +/- {np.std(m_j):>5.1f}   {d['p_j']:.2e}    {d['wp_j']:.2e}")
    print(f"{'Endpoints':<12} {np.mean(s_e):>6.1f} +/- {np.std(s_e):>5.1f}   {np.mean(m_e):>6.1f} +/- {np.std(m_e):>5.1f}   {d['p_e']:.2e}    {d['wp_e']:.2e}")
    print(f"{'Length':<12} {np.mean(s_l):>6.1f} +/- {np.std(s_l):>5.1f}   {np.mean(m_l):>6.1f} +/- {np.std(m_l):>5.1f}   {d['p_l']:.2e}    {d['wp_l']:.2e}")


# ============================================================
# Figure 1: Scatter plots
# ============================================================
def pval_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "n.s."

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
mlabels = ["Junctions","Endpoints","Vessel Length (px)"]
mkeys = [("s_j","m_j","p_j"),("s_e","m_e","p_e"),("s_l","m_l","p_l")]

for row, src in enumerate(["GT","LBBDM","PBBDM"]):
    for col, (sk,mk,pk) in enumerate(mkeys):
        ax = axes[row,col]
        sv,mv = stat_data[src][sk], stat_data[src][mk]
        pv = stat_data[src][pk]
        ax.scatter(sv, mv, alpha=0.35, s=12, c='#3366cc', edgecolors='none')
        maxv = max(max(sv), max(mv)) * 1.1
        ax.plot([0,maxv],[0,maxv],'k--',alpha=0.3,lw=1)
        slope,intercept,r,_,_ = stats.linregress(sv,mv)
        xf = np.linspace(0,maxv,100)
        ax.plot(xf, slope*xf+intercept, 'r-', alpha=0.6, lw=1.5)
        ax.text(0.05, 0.92, f'R\u00b2={r**2:.3f}\n{pval_stars(pv)} (p={pv:.1e})',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.set_xlabel("Single-color")
        ax.set_ylabel("Multi-color (depth-sep)")
        ax.set_title(f"{src}: {mlabels[col]}")

plt.tight_layout()
plt.savefig(output_dir / "final_scatter.png", dpi=200, bbox_inches="tight")
plt.close()


# ============================================================
# Figure 2: Bar chart
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
cs, cm = '#7BC67E', '#5B9BD5'

for col, (mname,sk,mk,pk) in enumerate([
    ("Junctions","s_j","m_j","p_j"),
    ("Endpoints","s_e","m_e","p_e"),
    ("Vessel Length","s_l","m_l","p_l"),
]):
    ax = axes[col]
    x = np.arange(3); w = 0.3
    sm_,mm_,ss_,ms_,pv_ = [],[],[],[],[]
    for src in ["GT","LBBDM","PBBDM"]:
        sm_.append(np.mean(stat_data[src][sk]))
        mm_.append(np.mean(stat_data[src][mk]))
        ss_.append(np.std(stat_data[src][sk]))
        ms_.append(np.std(stat_data[src][mk]))
        pv_.append(stat_data[src][pk])
    ax.bar(x-w/2,sm_,w,yerr=ss_,capsize=3,label='Single-color',color=cs,alpha=.85,edgecolor='white')
    ax.bar(x+w/2,mm_,w,yerr=ms_,capsize=3,label='Multi-color\n(depth-sep)',color=cm,alpha=.85,edgecolor='white')
    for ix in range(3):
        mh = max(sm_[ix]+ss_[ix], mm_[ix]+ms_[ix])
        bh = mh*1.08
        ax.plot([ix-w/2,ix-w/2,ix+w/2,ix+w/2],[bh,bh*1.03,bh*1.03,bh],'k-',lw=.8)
        ax.text(ix, bh*1.04, pval_stars(pv_[ix]), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["GT","LBBDM","PBBDM"])
    ax.set_ylabel(mname); ax.set_title(mname); ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig(output_dir / "final_bars.png", dpi=200, bbox_inches="tight")
plt.close()


# ============================================================
# Figure 3: Representative sample with baseline mask shown
# ============================================================
rep_sid = "23-13-512"

def dilate(s, t=4):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t,t))
    return cv2.dilate(s.astype(np.uint8)*255, k, iterations=1) > 0

gt_m = load(gt_multi_dir / f"{rep_sid}_real_B.png")
valid_mask = make_baseline_exclusion_mask(gt_m)

fig, axes = plt.subplots(3, 6, figsize=(30, 15))

source_paths = {
    "GT": (gt_multi_dir/f"{rep_sid}_real_B.png", gt_single_dir/f"{rep_sid}_real_B.png"),
    "LBBDM": (BASE/"Multi-color"/"LBBDM"/rep_sid/"output_0.png", BASE/"Single-color"/"LBBDM"/rep_sid/"output_0.png"),
    "PBBDM": (BASE/"Multi-color"/"PBBDM"/rep_sid/"output_0.png", BASE/"Single-color"/"PBBDM"/rep_sid/"output_0.png"),
}

for row, src in enumerate(["GT","LBBDM","PBBDM"]):
    mp, sp = source_paths[src]
    mi, si = load(mp), load(sp)

    # Col 0: Multi-color original with baseline region dimmed
    vis0 = mi.copy()
    vis0[~valid_mask] = (vis0[~valid_mask] * 0.3).astype(np.uint8)
    # Draw baseline boundary line
    baseline_y = np.argmin(valid_mask[:, valid_mask.shape[1]//2])
    if baseline_y == 0:
        # find the transition
        for yy in range(valid_mask.shape[0]-1, -1, -1):
            if valid_mask[yy, valid_mask.shape[1]//2]:
                baseline_y = yy
                break
    cv2.line(vis0, (0, baseline_y), (vis0.shape[1], baseline_y), (255,255,0), 2)
    axes[row,0].imshow(vis0)
    axes[row,0].set_title(f"{'Multi-color GT' if src=='GT' else f'{src} Multi-color'}\n(yellow line = baseline boundary)")

    # Col 1: Single-color original
    vis1 = si.copy()
    vis1[~valid_mask] = (vis1[~valid_mask] * 0.3).astype(np.uint8)
    cv2.line(vis1, (0, baseline_y), (vis1.shape[1], baseline_y), (255,255,0), 2)
    axes[row,1].imshow(vis1)
    axes[row,1].set_title(f"{'Single-color GT' if src=='GT' else f'{src} Single-color'}")

    # Col 2: Single-color skeleton (valid region highlighted)
    s_metrics, skel_s = analyze_single(si, valid_mask)
    dim_s = (si * 0.25).astype(np.uint8)
    vis2 = dim_s.copy()
    vis2[dilate(skel_s, 5) & valid_mask] = (0,255,0)
    vis2[dilate(skel_s, 5) & ~valid_mask] = (0,80,0)  # dim green for excluded
    axes[row,2].imshow(vis2)
    axes[row,2].set_title(f"Single Skeleton\nJ={s_metrics['junctions']}, E={s_metrics['endpoints']}, L={s_metrics['length']}")

    # Col 3: Layer separation
    bottom, middle, top = separate_layers_hsv(mi)
    vis3 = np.zeros_like(mi)
    vis3[bottom] = [255,80,80]
    vis3[middle] = [255,255,60]
    vis3[top] = [80,140,255]
    vis3[~valid_mask] = (vis3[~valid_mask] * 0.3).astype(np.uint8)
    axes[row,3].imshow(vis3)
    axes[row,3].set_title("Depth Layer Separation\n(R=Bottom, Y=Mid, B=Top)")

    # Col 4: Depth-sep skeleton (bridged)
    m_metrics, masks, skels, bridged, bridges = analyze_multi(mi, valid_mask)
    dim_m = (mi * 0.25).astype(np.uint8)
    vis4 = dim_m.copy()
    thick = dilate(bridged, 5)
    colors = [(255,80,80),(255,255,60),(80,140,255)]
    for si_idx, mask in enumerate(masks):
        vis4[thick & mask & valid_mask] = colors[si_idx]
        vis4[thick & mask & ~valid_mask] = tuple(c//3 for c in colors[si_idx])
    unc = thick & ~(masks[0]|masks[1]|masks[2])
    vis4[unc & valid_mask] = (0,255,200)
    axes[row,4].imshow(vis4)
    axes[row,4].set_title(f"Depth-sep Skeleton (bridged)\nJ={m_metrics['junctions']}, E={m_metrics['endpoints']}, L={m_metrics['length']}")

    # Col 5: Side-by-side metric comparison (text)
    axes[row,5].axis("off")
    txt = (f"{src} — Sample {rep_sid}\n\n"
           f"{'Metric':<12} {'Single':>8} {'Multi':>8} {'Diff':>8}\n"
           f"{'-'*38}\n"
           f"{'Junctions':<12} {s_metrics['junctions']:>8d} {m_metrics['junctions']:>8d} {m_metrics['junctions']-s_metrics['junctions']:>+8d}\n"
           f"{'Endpoints':<12} {s_metrics['endpoints']:>8d} {m_metrics['endpoints']:>8d} {m_metrics['endpoints']-s_metrics['endpoints']:>+8d}\n"
           f"{'Length':<12} {s_metrics['length']:>8d} {m_metrics['length']:>8d} {m_metrics['length']-s_metrics['length']:>+8d}\n")
    axes[row,5].text(0.1, 0.6, txt, transform=axes[row,5].transAxes,
                     fontsize=12, fontfamily='monospace', va='top',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.9))

for ax in axes[:,:5].flat: ax.axis("off")
for row, src in enumerate(["GT","LBBDM","PBBDM"]):
    axes[row,0].text(-0.02, 0.5, src, transform=axes[row,0].transAxes,
                     fontsize=16, fontweight='bold', va='center', ha='right', rotation=90)

plt.suptitle(f"Representative Sample: {rep_sid} (baseline excluded from metrics)", fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(output_dir / "paper_figure_final.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nAll saved:")
print("  final_scatter.png, final_bars.png, paper_figure_final.png")
