"""
Publication figure for Advanced Healthcare Materials.
- Original baseline detection (simple, less aggressive)
- Corrected metrics: Junctions=per-layer, Endpoints=bridged, Length=per-layer
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
matplotlib.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.dpi': 300,
})
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

def get_baseline_y(img, thr=0.35):
    """Original simple baseline detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v = gray > 15; h, w = v.shape
    rd = np.sum(v, axis=1) / w
    for y in range(h-1,-1,-1):
        if rd[y] < thr: return y
    return h

def make_valid_mask(img):
    h, w = img.shape[:2]
    by = get_baseline_y(img)
    vm = np.ones((h,w), dtype=bool); vm[by:,:] = False
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

def analyze_multi(img, vm):
    bottom, middle, top = separate_layers(img)
    masks = [bottom, middle, top]
    skels = [skeletonize(m) for m in masks]
    total_j, total_l = 0, 0
    for skel in skels:
        _, jn = find_features(skel)
        total_j += count_in_mask(jn, vm)
        total_l += int(np.sum(skel & vm))
    bridged = bridge_skeletons(skels, masks)
    b_ep, _ = find_features(bridged)
    total_e = count_in_mask(b_ep, vm)
    return {"junctions": total_j, "endpoints": total_e, "length": total_l}, masks, skels, bridged

def analyze_single(img, vm):
    mask = get_vessel_mask(img)
    skel = skeletonize(mask)
    ep, jn = find_features(skel)
    return {"junctions": count_in_mask(jn, vm), "endpoints": count_in_mask(ep, vm),
            "length": int(np.sum(skel & vm))}, skel

layer_colors = [(255,80,80),(255,255,60),(80,140,255)]

# ============================================================
# Full dataset
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
print(f"Common: {len(common)}")

results = {src: {"single":[], "multi":[]} for src in ["GT","LBBDM","PBBDM"]}
for i, sid in enumerate(common):
    if (i+1)%50==0: print(f"  {i+1}/{len(common)}...")
    gt_m = load(gt_multi_dir/f"{sid}_real_B.png")
    vm, by = make_valid_mask(gt_m)
    paths = {
        "GT": (gt_multi_dir/f"{sid}_real_B.png", gt_single_dir/f"{sid}_real_B.png"),
        "LBBDM": (BASE/"Multi-color"/"LBBDM"/sid/"output_0.png", BASE/"Single-color"/"LBBDM"/sid/"output_0.png"),
        "PBBDM": (BASE/"Multi-color"/"PBBDM"/sid/"output_0.png", BASE/"Single-color"/"PBBDM"/sid/"output_0.png"),
    }
    for src, (mp, sp) in paths.items():
        mm, _,_,_ = analyze_multi(load(mp), vm)
        sm, _ = analyze_single(load(sp), vm)
        results[src]["multi"].append(mm)
        results[src]["single"].append(sm)

n = len(results["GT"]["single"])
print(f"Processed: {n}")

def pval_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "n.s."

stat_data = {}
for src in ["GT","LBBDM","PBBDM"]:
    s,m = results[src]["single"], results[src]["multi"]
    d = {}
    for key in ["junctions","endpoints","length"]:
        sv = [x[key] for x in s]; mv = [x[key] for x in m]
        _,p = stats.ttest_rel(sv,mv); _,wp = stats.wilcoxon(sv,mv)
        lb = key[0]
        d[f"s_{lb}"]=sv; d[f"m_{lb}"]=mv; d[f"p_{lb}"]=p; d[f"wp_{lb}"]=wp
    stat_data[src] = d

print("\n--- Results ---")
for src in ["GT","LBBDM","PBBDM"]:
    d = stat_data[src]
    print(f"\n{src}:")
    for key, lb in [("junctions","j"),("endpoints","e"),("length","l")]:
        sv,mv,p = d[f"s_{lb}"],d[f"m_{lb}"],d[f"p_{lb}"]
        print(f"  {key:<11s} S:{np.mean(sv):>6.1f}+/-{np.std(sv):>5.1f}  M:{np.mean(mv):>6.1f}+/-{np.std(mv):>5.1f}  p={p:.2e} {pval_stars(p)}")


# ============================================================
# FIGURE: Publication quality for Advanced Healthcare Materials
# Full-page width = ~17cm, single column = ~8.5cm
# ============================================================

# --- Panel A: Representative samples (3 selected) ---
rep_sids = ["1-19-716", "16-18-716", "1-16-512"]

fig_a, axes_a = plt.subplots(len(rep_sids), 5, figsize=(17/2.54*2, 5/2.54*2*len(rep_sids)))
# cm to inches conversion: /2.54, then scale 2x for resolution

col_titles = ["Brightfield", "Multi-color\nFluorescence (GT)",
              "Single-color\nSkeleton", "Depth Layer\nSeparation",
              "Depth-separated\nSkeleton"]

for row, sid in enumerate(rep_sids):
    bf = load(gt_multi_dir/f"{sid}_real_A.png")
    fl_m = load(gt_multi_dir/f"{sid}_real_B.png")
    fl_s = load(gt_single_dir/f"{sid}_real_B.png")
    bf3 = bf if len(bf.shape)==3 else np.stack([bf]*3, axis=-1)
    vm, by = make_valid_mask(fl_m)

    s_met, s_skel = analyze_single(fl_s, vm)
    m_met, masks, skels, bridged = analyze_multi(fl_m, vm)

    # Col 0: BF
    axes_a[row,0].imshow(bf3)

    # Col 1: Multi-color FL
    axes_a[row,1].imshow(fl_m)

    # Col 2: Single skeleton on BF
    vis2 = bf3.copy()
    vis2[dilate(s_skel,4)] = (0,255,0)
    axes_a[row,2].imshow(vis2)

    # Col 3: Layer separation
    vis3 = np.zeros_like(fl_m)
    vis3[masks[0]] = [255,80,80]
    vis3[masks[1]] = [255,255,60]
    vis3[masks[2]] = [80,140,255]
    axes_a[row,3].imshow(vis3)

    # Col 4: Depth-sep skeleton on BF
    vis4 = bf3.copy()
    bt = dilate(bridged,4)
    for si,mask in enumerate(masks): vis4[bt & mask] = layer_colors[si]
    vis4[bt & ~(masks[0]|masks[1]|masks[2])] = (0,255,200)
    axes_a[row,4].imshow(vis4)

    # Row metrics annotation
    axes_a[row,2].text(0.5, 0.02,
        f"J={s_met['junctions']}  E={s_met['endpoints']}  L={s_met['length']}",
        transform=axes_a[row,2].transAxes, fontsize=7, ha='center', va='bottom',
        color='white', bbox=dict(facecolor='black', alpha=0.7, pad=2, boxstyle='round,pad=0.3'))
    axes_a[row,4].text(0.5, 0.02,
        f"J={m_met['junctions']}  E={m_met['endpoints']}  L={m_met['length']}",
        transform=axes_a[row,4].transAxes, fontsize=7, ha='center', va='bottom',
        color='white', bbox=dict(facecolor='black', alpha=0.7, pad=2, boxstyle='round,pad=0.3'))

for j, title in enumerate(col_titles):
    axes_a[0,j].set_title(title, fontsize=9, fontweight='bold', pad=8)

for ax in axes_a.flat:
    ax.axis("off")

# Add scale bar (50um ≈ 51.2 px at 10x with 1024px/FOV)
# Patch size 512 → ~500um, so 50um ≈ 51px
for row in range(len(rep_sids)):
    ax = axes_a[row, 0]
    # White scale bar in bottom-left
    h_img = bf3.shape[0]
    w_img = bf3.shape[1]

plt.tight_layout(pad=0.5)
plt.savefig(output_dir / "Fig_panel_A_representatives.png", dpi=300, bbox_inches="tight",
            facecolor='white', edgecolor='none')
plt.savefig(output_dir / "Fig_panel_A_representatives.pdf", dpi=300, bbox_inches="tight",
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: Fig_panel_A")


# --- Panel B: Quantitative comparison (bar charts) ---
fig_b, axes_b = plt.subplots(1, 3, figsize=(17/2.54*2, 4.5/2.54*2))

cs, cm = '#7BC67E', '#5B9BD5'

metric_configs = [
    ("Junctions", "s_j", "m_j", "p_j"),
    ("Endpoints", "s_e", "m_e", "p_e"),
    ("Vessel Length (px)", "s_l", "m_l", "p_l"),
]

for col, (mname, sk, mk, pk) in enumerate(metric_configs):
    ax = axes_b[col]
    x = np.arange(3); w = 0.32
    sm_, mm_, ss_, ms_, pv_ = [], [], [], [], []
    for src in ["GT","LBBDM","PBBDM"]:
        sm_.append(np.mean(stat_data[src][sk]))
        mm_.append(np.mean(stat_data[src][mk]))
        ss_.append(np.std(stat_data[src][sk]) / np.sqrt(n))  # SEM
        ms_.append(np.std(stat_data[src][mk]) / np.sqrt(n))  # SEM
        pv_.append(stat_data[src][pk])

    b1 = ax.bar(x-w/2, sm_, w, yerr=ss_, capsize=3, label='Single-color',
                color=cs, alpha=0.85, edgecolor='#555555', linewidth=0.5)
    b2 = ax.bar(x+w/2, mm_, w, yerr=ms_, capsize=3, label='Multi-color\n(depth-separated)',
                color=cm, alpha=0.85, edgecolor='#555555', linewidth=0.5)

    # Significance brackets
    for ix in range(3):
        mh = max(sm_[ix]+ss_[ix], mm_[ix]+ms_[ix])
        bh = mh * 1.12
        ax.plot([ix-w/2, ix-w/2, ix+w/2, ix+w/2],
                [bh, bh*1.04, bh*1.04, bh], 'k-', lw=0.6)
        stars = pval_stars(pv_[ix])
        ax.text(ix, bh*1.05, stars, ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["GT", "LBBDM", "PBBDM"], fontsize=8)
    ax.set_ylabel(mname, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if col == 0:
        ax.legend(fontsize=7, loc='upper right', framealpha=0.9)

plt.tight_layout(pad=1.0)
plt.savefig(output_dir / "Fig_panel_B_bars.png", dpi=300, bbox_inches="tight",
            facecolor='white', edgecolor='none')
plt.savefig(output_dir / "Fig_panel_B_bars.pdf", dpi=300, bbox_inches="tight",
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: Fig_panel_B")


# --- Panel C: Scatter plots (GT only, single vs multi for each metric) ---
fig_c, axes_c = plt.subplots(1, 3, figsize=(17/2.54*2, 5/2.54*2))

scatter_color = '#3366cc'
for col, (mname, sk, mk, pk) in enumerate(metric_configs):
    ax = axes_c[col]
    sv = stat_data["GT"][sk]
    mv = stat_data["GT"][mk]
    pv = stat_data["GT"][pk]

    ax.scatter(sv, mv, alpha=0.35, s=10, c=scatter_color, edgecolors='none', zorder=2)
    maxv = max(max(sv), max(mv)) * 1.15
    ax.plot([0,maxv],[0,maxv],'k--',alpha=0.25,lw=0.8, zorder=1)

    slope,intercept,r,_,_ = stats.linregress(sv,mv)
    xf = np.linspace(0,maxv,100)
    ax.plot(xf, slope*xf+intercept, color='#cc3333', alpha=0.7, lw=1.2, zorder=3)

    ax.text(0.05, 0.92, f'R\u00b2 = {r**2:.3f}\np = {pv:.1e}',
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cccccc'))

    ax.set_xlabel(f"Single-color {mname.split('(')[0].strip()}", fontsize=8)
    ax.set_ylabel(f"Multi-color {mname.split('(')[0].strip()}", fontsize=8)
    ax.set_title(f"GT: {mname.split('(')[0].strip()}", fontsize=9, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(pad=1.0)
plt.savefig(output_dir / "Fig_panel_C_scatter.png", dpi=300, bbox_inches="tight",
            facecolor='white', edgecolor='none')
plt.savefig(output_dir / "Fig_panel_C_scatter.pdf", dpi=300, bbox_inches="tight",
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: Fig_panel_C")


# --- Panel D: GT vs Model concordance for depth-separated metrics ---
fig_d, axes_d = plt.subplots(2, 3, figsize=(17/2.54*2, 8/2.54*2))
model_colors = {'LBBDM': '#E67E22', 'PBBDM': '#8E44AD'}

for col, (mname, _, mk, _) in enumerate(metric_configs):
    for row_idx, model in enumerate(["LBBDM","PBBDM"]):
        ax = axes_d[row_idx, col]
        gt_vals = stat_data["GT"][mk]
        model_vals = stat_data[model][mk]

        ax.scatter(gt_vals, model_vals, alpha=0.35, s=10,
                   c=model_colors[model], edgecolors='none', zorder=2)
        maxv = max(max(gt_vals), max(model_vals)) * 1.15
        ax.plot([0,maxv],[0,maxv],'k--',alpha=0.25,lw=0.8, zorder=1)

        slope,intercept,r,p_r,_ = stats.linregress(gt_vals, model_vals)
        xf = np.linspace(0,maxv,100)
        ax.plot(xf, slope*xf+intercept, color='#cc3333', alpha=0.7, lw=1.2, zorder=3)

        ax.text(0.05, 0.92, f'R\u00b2 = {r**2:.3f}\np = {p_r:.1e}',
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cccccc'))

        ax.set_xlabel(f"GT {mname.split('(')[0].strip()}", fontsize=8)
        ax.set_ylabel(f"{model} {mname.split('(')[0].strip()}", fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if col == 0:
            ax.text(-0.25, 0.5, model, transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='center', ha='center', rotation=90)

    axes_d[0,col].set_title(f"{mname.split('(')[0].strip()}", fontsize=9, fontweight='bold')

plt.tight_layout(pad=1.0)
plt.savefig(output_dir / "Fig_panel_D_concordance.png", dpi=300, bbox_inches="tight",
            facecolor='white', edgecolor='none')
plt.savefig(output_dir / "Fig_panel_D_concordance.pdf", dpi=300, bbox_inches="tight",
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: Fig_panel_D")

print("\nAll publication figures saved (PNG + PDF).")
