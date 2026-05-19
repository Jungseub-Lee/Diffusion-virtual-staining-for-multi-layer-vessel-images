"""
Full comparison WITHOUT baseline removal.
Single-color flat vs Multi-color depth-separated (with bridging).
GT, LBBDM, PBBDM.
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


def analyze_multi(img_rgb):
    bottom, middle, top = separate_layers_hsv(img_rgb)
    masks = [bottom, middle, top]
    skels = [skeletonize(m) for m in masks]
    bridged = bridge_skeletons(skels, masks, radius=15)
    return compute_metrics(bridged)


def analyze_single(img_rgb):
    mask = get_vessel_mask(img_rgb)
    skel = skeletonize(mask)
    return compute_metrics(skel)


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
# Process
# ============================================================
results = {src: {"single": [], "multi": []} for src in ["GT","LBBDM","PBBDM"]}

def load(p): return np.array(Image.open(p).convert("RGB"))

for i, sid in enumerate(common):
    if (i+1) % 50 == 0: print(f"  {i+1}/{len(common)}...")

    paths = {
        "GT": (gt_multi_dir/f"{sid}_real_B.png", gt_single_dir/f"{sid}_real_B.png"),
        "LBBDM": (BASE/"Multi-color"/"LBBDM"/sid/"output_0.png", BASE/"Single-color"/"LBBDM"/sid/"output_0.png"),
        "PBBDM": (BASE/"Multi-color"/"PBBDM"/sid/"output_0.png", BASE/"Single-color"/"PBBDM"/sid/"output_0.png"),
    }

    for src, (mp, sp) in paths.items():
        results[src]["multi"].append(analyze_multi(load(mp)))
        results[src]["single"].append(analyze_single(load(sp)))

n = len(results["GT"]["single"])
print(f"Processed: {n}")

# ============================================================
# Stats
# ============================================================
print("\n" + "="*100)
print("FINAL (no baseline removal): Single vs Multi-color depth-separated (bridged)")
print("="*100)

stat_data = {}
for src in ["GT","LBBDM","PBBDM"]:
    s = results[src]["single"]
    m = results[src]["multi"]

    d = {}
    for key_s, key_m, label in [
        ("junctions","junctions","Junctions"),
        ("endpoints","endpoints","Endpoints"),
        ("length","length","Length"),
    ]:
        sv = [x[key_s] for x in s]
        mv = [x[key_m] for x in m]
        t, p = stats.ttest_rel(sv, mv)
        w, wp = stats.wilcoxon(sv, mv)
        d[f"s_{label[0].lower()}"] = sv
        d[f"m_{label[0].lower()}"] = mv
        d[f"p_{label[0].lower()}"] = p
        d[f"wp_{label[0].lower()}"] = wp

    stat_data[src] = d

    print(f"\n--- {src} ---")
    s_j, m_j = d["s_j"], d["m_j"]
    s_e, m_e = d["s_e"], d["m_e"]
    s_l, m_l = d["s_l"], d["m_l"]
    print(f"{'Metric':<12} {'Single':<20} {'Multi (bridged)':<20} {'t-test p':<12} {'Wilcoxon p'}")
    print("-"*76)
    print(f"{'Junctions':<12} {np.mean(s_j):>6.1f} +/- {np.std(s_j):>5.1f}   {np.mean(m_j):>6.1f} +/- {np.std(m_j):>5.1f}   {d['p_j']:.2e}    {d['wp_j']:.2e}")
    print(f"{'Endpoints':<12} {np.mean(s_e):>6.1f} +/- {np.std(s_e):>5.1f}   {np.mean(m_e):>6.1f} +/- {np.std(m_e):>5.1f}   {d['p_e']:.2e}    {d['wp_e']:.2e}")
    print(f"{'Length':<12} {np.mean(s_l):>6.1f} +/- {np.std(s_l):>5.1f}   {np.mean(m_l):>6.1f} +/- {np.std(m_l):>5.1f}   {d['p_l']:.2e}    {d['wp_l']:.2e}")


# ============================================================
# Figure 1: Scatter plots (3x3)
# ============================================================
def pval_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "n.s."

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
mlabels = ["Junctions", "Endpoints", "Vessel Length (px)"]
mkeys = [("s_j","m_j","p_j"), ("s_e","m_e","p_e"), ("s_l","m_l","p_l")]

for row, src in enumerate(["GT","LBBDM","PBBDM"]):
    for col, (sk, mk, pk) in enumerate(mkeys):
        ax = axes[row, col]
        sv = stat_data[src][sk]
        mv = stat_data[src][mk]
        pv = stat_data[src][pk]

        ax.scatter(sv, mv, alpha=0.35, s=12, c='#3366cc', edgecolors='none')
        maxv = max(max(sv), max(mv)) * 1.1
        ax.plot([0, maxv], [0, maxv], 'k--', alpha=0.3, lw=1)

        slope, intercept, r, p_r, se = stats.linregress(sv, mv)
        xf = np.linspace(0, maxv, 100)
        ax.plot(xf, slope*xf+intercept, 'r-', alpha=0.6, lw=1.5)
        ax.text(0.05, 0.92, f'R\u00b2={r**2:.3f}\n{pval_stars(pv)} (p={pv:.1e})',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel("Single-color")
        ax.set_ylabel("Multi-color (depth-sep)")
        ax.set_title(f"{src}: {mlabels[col]}")

plt.tight_layout()
plt.savefig(output_dir / "final_scatter_no_crop.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nSaved: final_scatter_no_crop.png")


# ============================================================
# Figure 2: Bar chart with significance
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
cs, cm = '#7BC67E', '#5B9BD5'

for col, (mname, sk, mk, pk) in enumerate([
    ("Junctions","s_j","m_j","p_j"),
    ("Endpoints","s_e","m_e","p_e"),
    ("Vessel Length","s_l","m_l","p_l"),
]):
    ax = axes[col]
    x = np.arange(3); w = 0.3
    sm, mm, ss, ms, pv = [],[],[],[],[]
    for src in ["GT","LBBDM","PBBDM"]:
        sm.append(np.mean(stat_data[src][sk]))
        mm.append(np.mean(stat_data[src][mk]))
        ss.append(np.std(stat_data[src][sk]))
        ms.append(np.std(stat_data[src][mk]))
        pv.append(stat_data[src][pk])

    ax.bar(x-w/2, sm, w, yerr=ss, capsize=3, label='Single-color', color=cs, alpha=.85, edgecolor='white')
    ax.bar(x+w/2, mm, w, yerr=ms, capsize=3, label='Multi-color\n(depth-sep)', color=cm, alpha=.85, edgecolor='white')

    for ix in range(3):
        mh = max(sm[ix]+ss[ix], mm[ix]+ms[ix])
        bh = mh * 1.08
        ax.plot([ix-w/2, ix-w/2, ix+w/2, ix+w/2], [bh, bh*1.03, bh*1.03, bh], 'k-', lw=.8)
        ax.text(ix, bh*1.04, pval_stars(pv[ix]), ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(["GT","LBBDM","PBBDM"])
    ax.set_ylabel(mname); ax.set_title(mname); ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig(output_dir / "final_bars_no_crop.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: final_bars_no_crop.png")


# ============================================================
# Figure 3: Representative sample (paper figure)
# ============================================================
rep_sid = "23-13-512"

def dilate(s, t=4):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t,t))
    return cv2.dilate(s.astype(np.uint8)*255, k, iterations=1) > 0

def make_depth_vis(img):
    dim = (img * 0.3).astype(np.uint8)
    bottom, middle, top = separate_layers_hsv(img)
    masks = [bottom, middle, top]
    skels = [skeletonize(m) for m in masks]
    bridged = bridge_skeletons(skels, masks, radius=15)
    vis = dim.copy()
    thick = dilate(bridged, 5)
    colors = [(255,80,80),(255,255,60),(80,140,255)]
    for si, mask in enumerate(masks):
        vis[thick & mask] = colors[si]
    vis[thick & ~(masks[0]|masks[1]|masks[2])] = (0,255,200)
    return vis, compute_metrics(bridged)

def make_single_vis(img):
    dim = (img * 0.3).astype(np.uint8)
    mask = get_vessel_mask(img)
    skel = skeletonize(mask)
    vis = dim.copy()
    vis[dilate(skel, 5)] = (0,255,0)
    return vis, compute_metrics(skel)

def make_layer_vis(img):
    bottom, middle, top = separate_layers_hsv(img)
    vis = np.zeros_like(img)
    vis[bottom] = [255,80,80]
    vis[middle] = [255,255,60]
    vis[top] = [80,140,255]
    return vis

fig, axes = plt.subplots(3, 5, figsize=(25, 15))
source_paths = {
    "GT": (gt_multi_dir/f"{rep_sid}_real_B.png", gt_single_dir/f"{rep_sid}_real_B.png"),
    "LBBDM": (BASE/"Multi-color"/"LBBDM"/rep_sid/"output_0.png", BASE/"Single-color"/"LBBDM"/rep_sid/"output_0.png"),
    "PBBDM": (BASE/"Multi-color"/"PBBDM"/rep_sid/"output_0.png", BASE/"Single-color"/"PBBDM"/rep_sid/"output_0.png"),
}

for row, src in enumerate(["GT","LBBDM","PBBDM"]):
    mp, sp = source_paths[src]
    mi, si = load(mp), load(sp)

    axes[row,0].imshow(mi)
    axes[row,0].set_title(f"{'Multi-color GT' if src=='GT' else f'{src} Multi-color'}")

    axes[row,1].imshow(si)
    axes[row,1].set_title(f"{'Single-color GT' if src=='GT' else f'{src} Single-color'}")

    sv, sm = make_single_vis(si)
    axes[row,2].imshow(sv)
    axes[row,2].set_title(f"Single Skeleton\nJ={sm['junctions']}, E={sm['endpoints']}, L={sm['length']}")

    axes[row,3].imshow(make_layer_vis(mi))
    axes[row,3].set_title("Depth Layer Separation\n(R=Bottom, Y=Mid, B=Top)")

    mv, mm = make_depth_vis(mi)
    axes[row,4].imshow(mv)
    axes[row,4].set_title(f"Depth-sep Skeleton (bridged)\nJ={mm['junctions']}, E={mm['endpoints']}, L={mm['length']}")

for ax in axes.flat: ax.axis("off")
for row, src in enumerate(["GT","LBBDM","PBBDM"]):
    axes[row,0].text(-0.02, 0.5, src, transform=axes[row,0].transAxes,
                     fontsize=16, fontweight='bold', va='center', ha='right', rotation=90)

plt.suptitle(f"Representative Sample: {rep_sid}", fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(output_dir / "paper_figure_no_crop.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: paper_figure_no_crop.png")

# ============================================================
def find_ep_jn(skeleton):
    skel_u8 = skeleton.astype(np.uint8)
    ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel_u8, -1, ks) * skel_u8
    return np.argwhere(neighbors == 1), np.argwhere(neighbors >= 3), neighbors

# Debug: endpoint vis for representative sample (no crop)
# ============================================================
fig, axes = plt.subplots(1, 5, figsize=(30, 6))
img = load(gt_multi_dir/f"{rep_sid}_real_B.png")
dim = (img * 0.25).astype(np.uint8)

bottom, middle, top = separate_layers_hsv(img)
masks = [bottom, middle, top]
skels = [skeletonize(m) for m in masks]
layer_colors = [(255,80,80),(255,255,60),(80,140,255)]

# Col 0: Original
axes[0].imshow(img); axes[0].set_title(f"{rep_sid}: Original GT")

# Col 1: Layer skeletons with endpoints
vis1 = dim.copy()
all_ep_count = 0
for si, skel in enumerate(skels):
    vis1[dilate(skel, 4)] = layer_colors[si]
    ep, _, _ = find_ep_jn(skel)
    all_ep_count += len(ep)
    for y,x in ep:
        cv2.circle(vis1, (x,y), 6, (255,255,255), 2)
axes[1].imshow(vis1); axes[1].set_title(f"Layer Skeletons\nAll endpoints: {all_ep_count}")

# Col 2: Endpoint classification
vis2 = dim.copy()
for si, skel in enumerate(skels):
    vis2[dilate(skel, 4)] = layer_colors[si]

true_total, boundary_total = 0, 0
for si, skel in enumerate(skels):
    other_union = np.zeros_like(masks[0], dtype=bool)
    for j in range(3):
        if j != si: other_union |= masks[j]
    od = cv2.dilate(other_union.astype(np.uint8)*255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)), iterations=1) > 0
    ep, _, _ = find_ep_jn(skel)
    for y,x in ep:
        if od[y,x]:
            cv2.drawMarker(vis2, (x,y), (255,100,100), cv2.MARKER_TILTED_CROSS, 10, 2)
            boundary_total += 1
        else:
            cv2.circle(vis2, (x,y), 6, (0,255,0), -1)
            true_total += 1
axes[2].imshow(vis2)
axes[2].set_title(f"Endpoint Classification\nTrue: {true_total}, Boundary: {boundary_total}")

# Col 3: Bridged
bridged = bridge_skeletons(skels, masks, 15)
vis3 = dim.copy()
thick = dilate(bridged, 5)
for si, mask in enumerate(masks):
    vis3[thick & mask] = layer_colors[si]
vis3[thick & ~(masks[0]|masks[1]|masks[2])] = (0,255,200)
mb = compute_metrics(bridged)
axes[3].imshow(vis3)
axes[3].set_title(f"After Bridging\nJ={mb['junctions']}, E={mb['endpoints']}, L={mb['length']}")

# Col 4: Single-color
si_img = load(gt_single_dir/f"{rep_sid}_real_B.png")
sv, sm = make_single_vis(si_img)
axes[4].imshow(sv)
axes[4].set_title(f"Single-color Skeleton\nJ={sm['junctions']}, E={sm['endpoints']}, L={sm['length']}")

for ax in axes: ax.axis("off")
plt.tight_layout()
plt.savefig(output_dir / "endpoint_debug_no_crop.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: endpoint_debug_no_crop.png")
