"""
Combined publication figure for Advanced Healthcare Materials.
Panel a: Simplified schematic (single vs multi-color skeleton concept)
Panel b: Representative samples (BF, FL, single skel, depth skel)
Panel c: Quantitative bar charts (Junction, Length, Endpoint)
"""
import numpy as np
from PIL import Image
from pathlib import Path
import cv2, json, warnings
from skimage.morphology import skeletonize, remove_small_objects
from scipy.spatial.distance import cdist
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})
warnings.filterwarnings('ignore')

BASE = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data")
OUT = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output")

# Load saved metrics
with open(OUT / "metric_results.json") as f:
    data = json.load(f)

# ====== HELPER FUNCTIONS (same as paper_final.py) ======
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

def dilate(s, t=4):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
    return cv2.dilate(s.astype(np.uint8)*255,k,iterations=1)>0

layer_colors = [(230,70,70),(240,220,50),(70,130,240)]

def pval_stars(p):
    if p<0.001: return "***"
    elif p<0.01: return "**"
    elif p<0.05: return "*"
    return "n.s."

# ====================================================================
# FIGURE LAYOUT
# Full-page width ~180mm for AHM
# Layout:
#   Row 1 (top):    Panel a (schematic) - spans full width, ~25% height
#   Row 2 (middle): Panel b (representative images) - full width, ~40% height
#   Row 3 (bottom): Panel c (bar charts) - full width, ~35% height
# ====================================================================

fig = plt.figure(figsize=(180/25.4, 220/25.4))  # 180mm x 220mm

# ====== PANEL a: SCHEMATIC (simplified) ======
# Use matplotlib to draw the schematic directly
ax_schema_left = fig.add_axes([0.02, 0.78, 0.45, 0.20])   # single-color
ax_schema_right = fig.add_axes([0.52, 0.78, 0.45, 0.20])  # multi-color

def draw_vessel_bezier(ax, pts, color, lw, alpha=0.7):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, solid_capstyle='round')

def bz(p0,p1,p2,p3,n=20):
    pts=[]
    for i in range(n+1):
        t=i/n; u=1-t
        pts.append((u**3*p0[0]+3*u**2*t*p1[0]+3*u*t**2*p2[0]+t**3*p3[0],
                     u**3*p0[1]+3*u**2*t*p1[1]+3*u*t**2*p2[1]+t**3*p3[1]))
    return pts

# Vessel paths (y goes up = sprouting)
A_main = bz((2,0),(1.9,0.7),(1.7,1.5),(1.4,2.0),25)+bz((1.4,2.0),(1.2,2.4),(0.9,2.7),(0.6,2.85),15)
A_br1 = bz((1.7,1.5),(2.1,1.9),(2.6,2.3),(3.1,2.6),15)+bz((3.1,2.6),(3.4,2.7),(3.7,2.8),(3.9,2.85),10)
A_br2 = bz((1.4,2.0),(1.1,2.3),(0.8,2.4),(0.5,2.4),10)
A_br3 = bz((2.6,2.3),(2.65,2.55),(2.7,2.7),(2.75,2.85),8)
B_main = bz((3.4,0),(3.3,0.6),(2.9,1.2),(2.5,1.6),20)+bz((2.5,1.6),(2.1,1.9),(1.7,2.2),(1.4,2.35),15)
B_br1 = bz((2.9,1.2),(2.5,1.4),(2.0,1.5),(1.5,1.4),12)+bz((1.5,1.4),(1.2,1.35),(0.8,1.2),(0.5,1.1),10)
B_br2 = bz((2.5,1.6),(2.8,2.0),(3.2,2.3),(3.7,2.45),12)+bz((3.7,2.45),(4.0,2.55),(4.3,2.6),(4.5,2.65),8)
B_br3 = bz((3.3,0.6),(3.6,0.8),(3.9,1.0),(4.2,1.05),10)

allA = [A_main, A_br1, A_br2, A_br3]
allB = [B_main, B_br1, B_br2, B_br3]

# Junction positions (exact branch points)
jA = [(1.7,1.5),(1.4,2.0),(2.6,2.3)]
jB = [(2.9,1.2),(2.5,1.6),(3.3,0.6)]

# Endpoint positions (tips)
eA = [(0.6,2.85),(3.9,2.85),(0.5,2.4),(2.75,2.85)]
eB = [(1.4,2.35),(0.5,1.1),(4.5,2.65),(4.2,1.05)]
rA, rB = (2,0), (3.4,0)

# --- Left: Single-color skeleton ---
ax = ax_schema_left
ax.set_xlim(-0.1, 5.0)
ax.set_ylim(-0.3, 3.2)
ax.set_aspect('equal')
ax.axis('off')

# ECM baseline
ax.fill_between([-0.1, 5.0], -0.3, 0.05, color='#ECF0F1', zorder=0)
ax.axhline(0.05, color='#BDC3C7', lw=1, ls='-')
ax.text(4.8, -0.15, 'ECM', fontsize=6, color='#95A5A6', ha='right', style='italic')

# All vessels in green (skeleton style)
for p in allA: draw_vessel_bezier(ax, p, '#27AE60', 2.5, 0.8)
for p in allB: draw_vessel_bezier(ax, p, '#27AE60', 2.0, 0.7)

# Junctions
for j in jA+jB:
    ax.plot(j[0], j[1], 'o', color='#1E8449', ms=5, mec='white', mew=1, zorder=5)

# Endpoints
for e in eA+eB:
    ax.plot(e[0], e[1], 'o', color='white', ms=4, mec='#1E8449', mew=1.5, zorder=5)
ax.plot(rA[0], rA[1], 'o', color='white', ms=4, mec='#1E8449', mew=1.5, zorder=5)
ax.plot(rB[0], rB[1], 'o', color='white', ms=4, mec='#1E8449', mew=1.5, zorder=5)

# False junction highlight at overlap (~1.8, 2.0)
fj_pos = (1.75, 2.05)
circle = plt.Circle(fj_pos, 0.25, fill=False, ec='#E74C3C', lw=1.5, ls='--', zorder=6)
ax.add_patch(circle)
ax.annotate('False\njunction', xy=fj_pos, xytext=(2.8, 2.7),
            fontsize=6.5, color='#C0392B', fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1),
            zorder=7)

# Merged zone label
ax.annotate('Merged\n(length lost)', xy=(1.55, 1.7), xytext=(0.3, 0.6),
            fontsize=5.5, color='#7D6608', style='italic', ha='center',
            arrowprops=dict(arrowstyle='->', color='#D4A017', lw=0.8),
            bbox=dict(boxstyle='round,pad=0.2', fc='#FEF9E7', ec='#D4A017', lw=0.5),
            zorder=7)

ax.set_title('Single-color Skeleton', fontsize=9, fontweight='bold', color='#1E8449', pad=5)


# --- Right: Multi-color depth-separated skeleton ---
ax = ax_schema_right
ax.set_xlim(-0.1, 5.0)
ax.set_ylim(-0.3, 3.2)
ax.set_aspect('equal')
ax.axis('off')

# ECM baseline
ax.fill_between([-0.1, 5.0], -0.3, 0.05, color='#ECF0F1', zorder=0)
ax.axhline(0.05, color='#BDC3C7', lw=1, ls='-')
ax.text(4.8, -0.15, 'ECM', fontsize=6, color='#95A5A6', ha='right', style='italic')

# Vessel A = Blue (top layer)
for p in allA: draw_vessel_bezier(ax, p, '#2980B9', 2.5, 0.8)
# Vessel B = Red (bottom layer)
for p in allB: draw_vessel_bezier(ax, p, '#C0392B', 2.0, 0.7)

# Yellow transition
yw = bz((2.1, 1.8), (1.95, 1.95), (1.85, 2.05), (1.75, 2.1), 8)
draw_vessel_bezier(ax, yw, '#D4A017', 2, 0.6)

# Blue junctions
for j in jA: ax.plot(j[0], j[1], 'o', color='#2471A3', ms=5, mec='white', mew=1, zorder=5)
# Red junctions
for j in jB: ax.plot(j[0], j[1], 'o', color='#922B21', ms=5, mec='white', mew=1, zorder=5)

# Blue endpoints
for e in eA: ax.plot(e[0], e[1], 'o', color='white', ms=4, mec='#2471A3', mew=1.5, zorder=5)
ax.plot(rA[0], rA[1], 'o', color='white', ms=4, mec='#2471A3', mew=1.5, zorder=5)
# Red endpoints
for e in eB: ax.plot(e[0], e[1], 'o', color='white', ms=4, mec='#922B21', mew=1.5, zorder=5)
ax.plot(rB[0], rB[1], 'o', color='white', ms=4, mec='#922B21', mew=1.5, zorder=5)

# Correct separation highlight
corr_pos = (1.75, 2.05)
circle = plt.Circle(corr_pos, 0.25, fill=False, ec='#27AE60', lw=1.5, ls='--', zorder=6)
ax.add_patch(circle)
ax.annotate('Correctly\nseparated', xy=corr_pos, xytext=(2.8, 2.7),
            fontsize=6.5, color='#1E8449', fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1),
            zorder=7)

# Length recovered label
ax.annotate('2 paths\n(length recovered)', xy=(1.55, 1.7), xytext=(0.3, 0.6),
            fontsize=5.5, color='#196F3D', style='italic', ha='center',
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=0.8),
            bbox=dict(boxstyle='round,pad=0.2', fc='#EAFAF1', ec='#27AE60', lw=0.5),
            zorder=7)

# Color legend
legend_items = [
    mpatches.Patch(color='#C0392B', label='Bottom layer'),
    mpatches.Patch(color='#D4A017', label='Middle layer'),
    mpatches.Patch(color='#2980B9', label='Top layer'),
]
ax.legend(handles=legend_items, fontsize=5.5, loc='lower right',
          framealpha=0.9, edgecolor='#BDC3C7', handlelength=1, handletextpad=0.4)

ax.set_title('Depth-separated Skeleton', fontsize=9, fontweight='bold', color='#2471A3', pad=5)

# Panel label
fig.text(0.01, 0.97, 'a', fontsize=14, fontweight='bold', va='top')


# ====== PANEL b: REPRESENTATIVE IMAGES ======
gt_multi_dir = BASE/"Multi-color"/"Pix2pix"
gt_single_dir = BASE/"Single-color"/"Pix2pix"

rep_sids = ["1-19-716", "16-18-716", "1-16-512"]
n_rep = len(rep_sids)

# 3 rows x 4 cols: BF, Multi-color FL, Single skel on BF, Depth skel on BF
panel_b_top = 0.38
panel_b_h = 0.38
col_w = 0.23
gap_x = 0.01

for row, sid in enumerate(rep_sids):
    bf = load(gt_multi_dir/f"{sid}_real_A.png")
    fl_m = load(gt_multi_dir/f"{sid}_real_B.png")
    fl_s = load(gt_single_dir/f"{sid}_real_B.png")
    bf3 = bf if len(bf.shape)==3 else np.stack([bf]*3, axis=-1)
    vm, by = make_valid_mask(fl_m)

    # Single analysis
    mask_s = get_vessel_mask(fl_s)
    skel_s = skeletonize(mask_s)
    ep_s, jn_s = find_features(skel_s)
    s_j = count_in_mask(jn_s, vm)
    s_e = count_in_mask(ep_s, vm)
    s_l = int(np.sum(skel_s & vm))

    # Multi analysis
    bottom, middle, top_ = separate_layers(fl_m)
    masks = [bottom, middle, top_]
    skels = [skeletonize(m) for m in masks]
    m_j, m_l = 0, 0
    for sk in skels:
        _, jn = find_features(sk)
        m_j += count_in_mask(jn, vm)
        m_l += int(np.sum(sk & vm))
    bridged = bridge_skeletons(skels, masks)
    b_ep, _ = find_features(bridged)
    m_e = count_in_mask(b_ep, vm)

    y_pos = panel_b_top + (n_rep - 1 - row) * panel_b_h / n_rep

    for col in range(4):
        ax = fig.add_axes([0.02 + col*(col_w + gap_x), y_pos, col_w, panel_b_h/n_rep - 0.01])

        if col == 0:
            ax.imshow(bf3)
        elif col == 1:
            ax.imshow(fl_m)
        elif col == 2:
            vis = bf3.copy()
            vis[dilate(skel_s, 5)] = (40, 200, 80)
            ax.imshow(vis)
            ax.text(0.5, 0.03, f'J={s_j}  L={s_l}',
                    transform=ax.transAxes, fontsize=6, ha='center', color='white',
                    bbox=dict(fc='black', alpha=0.6, pad=1.5, boxstyle='round,pad=0.2'))
        elif col == 3:
            vis = bf3.copy()
            bt = dilate(bridged, 5)
            for si, mask in enumerate(masks):
                vis[bt & mask] = layer_colors[si]
            vis[bt & ~(masks[0]|masks[1]|masks[2])] = (0, 220, 180)
            ax.imshow(vis)
            ax.text(0.5, 0.03, f'J={m_j}  L={m_l}',
                    transform=ax.transAxes, fontsize=6, ha='center', color='white',
                    bbox=dict(fc='black', alpha=0.6, pad=1.5, boxstyle='round,pad=0.2'))

        ax.axis('off')

        if row == 0:
            titles = ['Brightfield', 'Multi-color FL\n(GT)', 'Single-color\nSkeleton', 'Depth-separated\nSkeleton']
            ax.set_title(titles[col], fontsize=7.5, fontweight='bold', pad=3)

        if col == 0:
            ax.text(-0.05, 0.5, sid, transform=ax.transAxes, fontsize=6,
                    rotation=90, va='center', ha='right', color='#555555')

fig.text(0.01, 0.76, 'b', fontsize=14, fontweight='bold', va='top')


# ====== PANEL c: QUANTITATIVE BAR CHARTS ======
ax_j = fig.add_axes([0.08, 0.05, 0.25, 0.28])
ax_l = fig.add_axes([0.40, 0.05, 0.25, 0.28])
ax_e = fig.add_axes([0.72, 0.05, 0.25, 0.28])

cs, cm = '#7BC67E', '#5B9BD5'
configs = [
    (ax_j, "Junctions", "junctions"),
    (ax_l, "Total Length (px)", "length"),
    (ax_e, "Endpoints", "endpoints"),
]

for ax, label, key in configs:
    x = np.arange(3); w = 0.3
    sm, mm, ss, ms, pv = [], [], [], [], []
    for src in ["GT", "LBBDM", "PBBDM"]:
        d = data[src][key]
        sm.append(d['single_mean']); mm.append(d['multi_mean'])
        ss.append(d['single_sem']); ms.append(d['multi_sem'])
        pv.append(d['p_ttest'])

    ax.bar(x-w/2, sm, w, yerr=ss, capsize=3, label='Single-color',
           color=cs, alpha=0.85, edgecolor='#555', lw=0.5)
    ax.bar(x+w/2, mm, w, yerr=ms, capsize=3, label='Depth-separated',
           color=cm, alpha=0.85, edgecolor='#555', lw=0.5)

    for ix in range(3):
        mh = max(sm[ix]+ss[ix], mm[ix]+ms[ix])
        bh = mh * 1.08
        ax.plot([ix-w/2, ix-w/2, ix+w/2, ix+w/2],
                [bh, bh*1.03, bh*1.03, bh], 'k-', lw=0.5)
        ax.text(ix, bh*1.04, pval_stars(pv[ix]), ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(["GT", "LBBDM", "PBBDM"], fontsize=7)
    ax.set_ylabel(label, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if key == "junctions":
        ax.legend(fontsize=6, loc='upper left', framealpha=0.9, edgecolor='#ccc')

fig.text(0.01, 0.34, 'c', fontsize=14, fontweight='bold', va='top')

# ====== SAVE ======
plt.savefig(OUT / "Fig_depth_analysis_combined.png", dpi=300, bbox_inches="tight",
            facecolor='white', edgecolor='none')
plt.savefig(OUT / "Fig_depth_analysis_combined.pdf", dpi=300, bbox_inches="tight",
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: Fig_depth_analysis_combined.png/pdf")
