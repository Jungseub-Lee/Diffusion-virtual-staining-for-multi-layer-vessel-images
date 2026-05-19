"""
FIXED analysis:
- Junctions: per-layer independent counting (NO cross-layer junctions)
- Endpoints: bridged skeleton (reduce false boundary endpoints)
- Length: per-layer sum
- Baseline: improved detection
"""
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from scipy.spatial.distance import cdist
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data")
output_dir = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output")

def load(p): return np.array(Image.open(p).convert("RGB"))

def separate_layers(img, min_area=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s = hsv[:,:,0], hsv[:,:,1]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fg = (gray > 15) & (s > 30)
    b = ((h<=12)|(h>=155))&fg; m = ((h>=13)&(h<=55))&fg; t = ((h>=85)&(h<=140))&fg
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    def c(mask):
        u = mask.astype(np.uint8)*255
        u = cv2.morphologyEx(u,cv2.MORPH_CLOSE,k,iterations=2)
        u = cv2.morphologyEx(u,cv2.MORPH_OPEN,k,iterations=1)
        return remove_small_objects(u>0, min_size=min_area)
    return c(b),c(m),c(t)

def get_vessel_mask(img, min_area=50):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v = gray > 15
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    u = v.astype(np.uint8)*255
    u = cv2.morphologyEx(u,cv2.MORPH_CLOSE,k,iterations=2)
    u = cv2.morphologyEx(u,cv2.MORPH_OPEN,k,iterations=1)
    return remove_small_objects(u>0, min_size=min_area)

def get_baseline_y(img, thr=0.35, min_run=5):
    """
    Improved baseline detection: find continuous dense band at bottom.
    Requires at least min_run consecutive dense rows.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v = gray > 15
    h, w = v.shape
    rd = np.sum(v, axis=1) / w

    # Scan from bottom, find first row below threshold
    # Must have min_run consecutive dense rows to count as baseline
    run = 0
    for y in range(h-1, -1, -1):
        if rd[y] >= thr:
            run += 1
        else:
            if run >= min_run:
                return y  # baseline starts at y+1
            run = 0

    # If entire image is dense, use bottom 15% as baseline
    if run >= min_run:
        return max(int(h * 0.15), 10)

    return h  # no baseline detected

def make_valid_mask(img):
    h, w = img.shape[:2]
    by = get_baseline_y(img)
    vm = np.ones((h,w), dtype=bool)
    vm[by:,:] = False
    return vm, by

def find_features(skeleton):
    su = skeleton.astype(np.uint8)
    ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su,-1,ks)*su
    return np.argwhere(nb==1), np.argwhere(nb>=3)

def count_in_mask(coords, mask):
    return sum(1 for y,x in coords if mask[y,x])

def bridge_skeletons(skels, masks, radius=15):
    connected = np.zeros_like(skels[0], dtype=np.uint8)
    for s in skels: connected |= s.astype(np.uint8)
    for li,lj in [(0,1),(1,2)]:
        ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
        ni = cv2.filter2D(skels[li].astype(np.uint8),-1,ks)*skels[li].astype(np.uint8)
        nj = cv2.filter2D(skels[lj].astype(np.uint8),-1,ks)*skels[lj].astype(np.uint8)
        ei,ej = np.argwhere(ni==1), np.argwhere(nj==1)
        if len(ei)==0 or len(ej)==0: continue
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
        dj = cv2.dilate(masks[lj].astype(np.uint8)*255,dk,iterations=1)>0
        di = cv2.dilate(masks[li].astype(np.uint8)*255,dk,iterations=1)>0
        vi = np.array([p for p in ei if dj[p[0],p[1]]])
        vj = np.array([p for p in ej if di[p[0],p[1]]])
        if len(vi)==0 or len(vj)==0: continue
        dists = cdist(vi,vj)
        ui,uj = set(),set()
        for idx in np.argsort(dists.ravel()):
            ii,jj = idx//dists.shape[1], idx%dists.shape[1]
            if dists[ii,jj]>radius: break
            if ii in ui or jj in uj: continue
            cv2.line(connected,(int(vi[ii][1]),int(vi[ii][0])),(int(vj[jj][1]),int(vj[jj][0])),1,1)
            ui.add(ii); uj.add(jj)
    return skeletonize(connected>0)

def dilate(s, t=5):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
    return cv2.dilate(s.astype(np.uint8)*255,k,iterations=1)>0


def analyze_multi_correct(img, vm):
    """
    CORRECT multi-color analysis:
    - Junctions: per-layer independent (no cross-layer)
    - Endpoints: from bridged skeleton
    - Length: per-layer sum
    """
    bottom, middle, top = separate_layers(img)
    masks = [bottom, middle, top]
    skels = [skeletonize(m) for m in masks]

    # Junctions & Length: per-layer independent
    total_j = 0
    total_l = 0
    for skel in skels:
        ep, jn = find_features(skel)
        total_j += count_in_mask(jn, vm)
        total_l += int(np.sum(skel & vm))

    # Endpoints: from bridged skeleton (layer-connected)
    bridged = bridge_skeletons(skels, masks)
    b_ep, _ = find_features(bridged)
    total_e = count_in_mask(b_ep, vm)

    return {
        "junctions": total_j,
        "endpoints": total_e,
        "length": total_l,
    }, masks, skels, bridged


def analyze_single_correct(img, vm):
    mask = get_vessel_mask(img)
    skel = skeletonize(mask)
    ep, jn = find_features(skel)
    return {
        "junctions": count_in_mask(jn, vm),
        "endpoints": count_in_mask(ep, vm),
        "length": int(np.sum(skel & vm)),
    }, skel


# ============================================================
# Full dataset comparison
# ============================================================
gt_multi_dir = BASE/"Multi-color"/"Pix2pix"
gt_single_dir = BASE/"Single-color"/"Pix2pix"

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

for i, sid in enumerate(common):
    if (i+1) % 50 == 0: print(f"  {i+1}/{len(common)}...")

    gt_m = load(gt_multi_dir/f"{sid}_real_B.png")
    vm, by = make_valid_mask(gt_m)

    paths = {
        "GT": (gt_multi_dir/f"{sid}_real_B.png", gt_single_dir/f"{sid}_real_B.png"),
        "LBBDM": (BASE/"Multi-color"/"LBBDM"/sid/"output_0.png", BASE/"Single-color"/"LBBDM"/sid/"output_0.png"),
        "PBBDM": (BASE/"Multi-color"/"PBBDM"/sid/"output_0.png", BASE/"Single-color"/"PBBDM"/sid/"output_0.png"),
    }

    for src, (mp, sp) in paths.items():
        m_met, _, _, _ = analyze_multi_correct(load(mp), vm)
        s_met, _ = analyze_single_correct(load(sp), vm)
        results[src]["multi"].append(m_met)
        results[src]["single"].append(s_met)

n = len(results["GT"]["single"])
print(f"Processed: {n}")

# Stats
print("\n" + "="*100)
print("CORRECTED: Junctions=per-layer independent, Endpoints=bridged, Length=per-layer sum")
print("Baseline excluded from all counts")
print("="*100)

def pval_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "n.s."

stat_data = {}
for src in ["GT","LBBDM","PBBDM"]:
    s = results[src]["single"]
    m = results[src]["multi"]
    d = {}
    for key in ["junctions","endpoints","length"]:
        sv = [x[key] for x in s]
        mv = [x[key] for x in m]
        t, p = stats.ttest_rel(sv, mv)
        w, wp = stats.wilcoxon(sv, mv)
        label = key[0]
        d[f"s_{label}"] = sv; d[f"m_{label}"] = mv
        d[f"p_{label}"] = p; d[f"wp_{label}"] = wp
    stat_data[src] = d

    print(f"\n--- {src} ---")
    for key, label in [("junctions","j"),("endpoints","e"),("length","l")]:
        sv, mv = d[f"s_{label}"], d[f"m_{label}"]
        p = d[f"p_{label}"]
        print(f"  {key:<12s} Single: {np.mean(sv):>6.1f} +/- {np.std(sv):>5.1f}  |  Multi: {np.mean(mv):>6.1f} +/- {np.std(mv):>5.1f}  |  p={p:.2e} {pval_stars(p)}")


# ============================================================
# Scatter + Bar plots
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
mlabels = ["Junctions\n(per-layer independent)","Endpoints\n(bridged)","Vessel Length\n(per-layer sum)"]
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
        ax.set_xlabel("Single-color"); ax.set_ylabel("Multi-color (depth-sep)")
        ax.set_title(f"{src}: {mlabels[col]}")

plt.tight_layout()
plt.savefig(output_dir / "corrected_scatter.png", dpi=200, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
cs, cm = '#7BC67E', '#5B9BD5'
for col, (mname,sk,mk,pk) in enumerate([
    ("Junctions (per-layer)","s_j","m_j","p_j"),
    ("Endpoints (bridged)","s_e","m_e","p_e"),
    ("Vessel Length","s_l","m_l","p_l"),
]):
    ax = axes[col]; x = np.arange(3); w = 0.3
    sm_,mm_,ss_,ms_,pv_ = [],[],[],[],[]
    for src in ["GT","LBBDM","PBBDM"]:
        sm_.append(np.mean(stat_data[src][sk])); mm_.append(np.mean(stat_data[src][mk]))
        ss_.append(np.std(stat_data[src][sk])); ms_.append(np.std(stat_data[src][mk]))
        pv_.append(stat_data[src][pk])
    ax.bar(x-w/2,sm_,w,yerr=ss_,capsize=3,label='Single-color',color=cs,alpha=.85,edgecolor='white')
    ax.bar(x+w/2,mm_,w,yerr=ms_,capsize=3,label='Multi-color\n(depth-sep)',color=cm,alpha=.85,edgecolor='white')
    for ix in range(3):
        mh = max(sm_[ix]+ss_[ix],mm_[ix]+ms_[ix]); bh = mh*1.08
        ax.plot([ix-w/2,ix-w/2,ix+w/2,ix+w/2],[bh,bh*1.03,bh*1.03,bh],'k-',lw=.8)
        ax.text(ix, bh*1.04, pval_stars(pv_[ix]), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["GT","LBBDM","PBBDM"])
    ax.set_ylabel(mname); ax.set_title(mname); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / "corrected_bars.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nSaved: corrected_scatter.png, corrected_bars.png")


# ============================================================
# Selected samples visualization with CORRECT junction counting
# ============================================================
sids = ["1-19-716", "14-7-512", "16-18-716", "23-13-512", "1-16-512"]
layer_colors = [(255,80,80),(255,255,60),(80,140,255)]

fig, axes = plt.subplots(len(sids), 5, figsize=(30, 7*len(sids)))

for row, sid in enumerate(sids):
    fl_m = load(gt_multi_dir/f"{sid}_real_B.png")
    fl_s = load(gt_single_dir/f"{sid}_real_B.png")
    bf = load(gt_multi_dir/f"{sid}_real_A.png")
    bf3 = bf if len(bf.shape)==3 else np.stack([bf]*3, axis=-1)
    vm, by = make_valid_mask(fl_m)

    # Single
    s_met, s_skel = analyze_single_correct(fl_s, vm)

    # Multi (correct)
    m_met, masks, skels, bridged = analyze_multi_correct(fl_m, vm)

    dim_s = (fl_s * 0.3).astype(np.uint8)
    dim_m = (fl_m * 0.3).astype(np.uint8)

    def draw_bl(vis, by):
        cv2.line(vis, (0,by), (vis.shape[1],by), (255,255,0), 2)
        vis[by:,:] = (vis[by:,:] * 0.3).astype(np.uint8)
        return vis

    # Col 0: BF + single skeleton
    vis0 = bf3.copy()
    vis0[dilate(s_skel,5)] = (0,255,0)
    vis0 = draw_bl(vis0, by)
    axes[row,0].imshow(vis0)
    axes[row,0].set_title(f"{sid}\nSingle Skeleton on BF\nJ={s_met['junctions']}, E={s_met['endpoints']}, L={s_met['length']}", fontsize=10)

    # Col 1: BF + depth skeleton
    vis1 = bf3.copy()
    bt = dilate(bridged, 5)
    for si, mask in enumerate(masks): vis1[bt & mask] = layer_colors[si]
    vis1[bt & ~(masks[0]|masks[1]|masks[2])] = (0,255,200)
    vis1 = draw_bl(vis1, by)
    axes[row,1].imshow(vis1)
    axes[row,1].set_title(f"Depth-sep Skeleton on BF\nJ={m_met['junctions']}, E={m_met['endpoints']}, L={m_met['length']}", fontsize=10)

    # Col 2: Single skeleton + junctions (above baseline)
    vis2 = dim_s.copy()
    vis2[dilate(s_skel,4)] = (0,200,0)
    s_ep, s_jn = find_features(s_skel)
    for y,x in s_jn:
        if vm[y,x]: cv2.circle(vis2, (x,y), 7, (0,255,255), -1)
    for y,x in s_ep:
        if vm[y,x]: cv2.circle(vis2, (x,y), 5, (255,255,255), 2)
    vis2 = draw_bl(vis2, by)
    axes[row,2].imshow(vis2)
    axes[row,2].set_title(f"Single: J={s_met['junctions']} (cyan)\nE={s_met['endpoints']} (white)", fontsize=10)

    # Col 3: Depth skeleton + per-layer junctions (above baseline)
    vis3 = dim_m.copy()
    bt = dilate(bridged, 4)
    for si, mask in enumerate(masks): vis3[bt & mask] = layer_colors[si]
    vis3[bt & ~(masks[0]|masks[1]|masks[2])] = (0,255,200)
    # Per-layer junctions (NOT cross-layer)
    for si, skel in enumerate(skels):
        _, jn = find_features(skel)
        for y,x in jn:
            if vm[y,x]: cv2.circle(vis3, (x,y), 7, (0,255,255), -1)
    # Endpoints from bridged
    b_ep, _ = find_features(bridged)
    for y,x in b_ep:
        if vm[y,x]: cv2.circle(vis3, (x,y), 5, (255,255,255), 2)
    vis3 = draw_bl(vis3, by)
    axes[row,3].imshow(vis3)
    axes[row,3].set_title(f"Depth: J={m_met['junctions']} (cyan, per-layer)\nE={m_met['endpoints']} (white, bridged)", fontsize=10)

    # Col 4: Multi-color FL reference
    vis4 = fl_m.copy()
    vis4 = draw_bl(vis4, by)
    axes[row,4].imshow(vis4)
    axes[row,4].set_title("Multi-color FL (GT)", fontsize=10)

for ax in axes.flat: ax.axis("off")
plt.tight_layout()
plt.savefig(output_dir / "corrected_selected_samples.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: corrected_selected_samples.png")
