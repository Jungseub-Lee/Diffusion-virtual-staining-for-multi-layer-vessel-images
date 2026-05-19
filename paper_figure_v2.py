"""
Publication figure for Advanced Healthcare Materials.
Panel a: Simplified schematic
Panel b: Representative images
Panel c: Single GT vs Multi GT (raw paired comparison)
Panel d: Multi GT=1.0 normalized, LBBDM/PBBDM ratio
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
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'Arial', 'font.size': 7,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'figure.dpi': 300, 'savefig.dpi': 300,
})
warnings.filterwarnings('ignore')

BASE = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data")
OUT = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output")

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

# ====== COMPUTE ALL METRICS INCLUDING AREA ======
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

def analyze_multi_full(img, vm):
    bottom, middle, top = separate_layers(img)
    masks = [bottom, middle, top]
    skels = [skeletonize(m) for m in masks]
    total_j, total_l, total_a = 0, 0, 0
    for i, sk in enumerate(skels):
        _, jn = find_features(sk)
        total_j += count_in_mask(jn, vm)
        total_l += int(np.sum(sk & vm))
        total_a += int(np.sum(masks[i] & vm))
    bridged = bridge_skeletons(skels, masks)
    b_ep, _ = find_features(bridged)
    total_e = count_in_mask(b_ep, vm)
    return {"junctions": total_j, "endpoints": total_e, "length": total_l, "area": total_a}, masks, skels, bridged

def analyze_single_full(img, vm):
    mask = get_vessel_mask(img)
    sk = skeletonize(mask)
    ep, jn = find_features(sk)
    area = int(np.sum(mask & vm))
    return {"junctions": count_in_mask(jn, vm), "endpoints": count_in_mask(ep, vm),
            "length": int(np.sum(sk & vm)), "area": area}, sk, mask

results = {src: {"single":[], "multi":[]} for src in ["GT","LBBDM","PBBDM"]}
sample_ids = []

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
        mm, _, _, _ = analyze_multi_full(load(mp), vm)
        sm, _, _ = analyze_single_full(load(sp), vm)
        results[src]["multi"].append(mm)
        results[src]["single"].append(sm)
    sample_ids.append(sid)

n = len(common)
print(f"Processed: {n}")

# Compute stats
metrics = ["junctions", "endpoints", "length", "area"]
stat_data = {}
for src in ["GT","LBBDM","PBBDM"]:
    s, m = results[src]["single"], results[src]["multi"]
    d = {}
    for key in metrics:
        sv = [x[key] for x in s]; mv = [x[key] for x in m]
        _, p = stats.ttest_rel(sv, mv)
        _, wp = stats.wilcoxon(sv, mv)
        d[key] = {"sv": sv, "mv": mv, "p": p, "wp": wp}
    stat_data[src] = d

# Print summary
for src in ["GT","LBBDM","PBBDM"]:
    print(f"\n{src}:")
    for key in metrics:
        d = stat_data[src][key]
        sm, mm = np.mean(d["sv"]), np.mean(d["mv"])
        print(f"  {key:<11s} S:{sm:>8.1f}  M:{mm:>8.1f}  p={d['p']:.2e} {pval_stars(d['p'])}")


# ====================================================================
# FIGURE
# ====================================================================
fig = plt.figure(figsize=(180/25.4, 240/25.4))  # 180mm x 240mm

# ====== PANEL a: SCHEMATIC ======
ax_s1 = fig.add_axes([0.03, 0.82, 0.44, 0.16])
ax_s2 = fig.add_axes([0.52, 0.82, 0.44, 0.16])

def bz(p0,p1,p2,p3,n=20):
    pts=[]
    for i in range(n+1):
        t=i/n; u=1-t
        pts.append((u**3*p0[0]+3*u**2*t*p1[0]+3*u*t**2*p2[0]+t**3*p3[0],
                     u**3*p0[1]+3*u**2*t*p1[1]+3*u*t**2*p2[1]+t**3*p3[1]))
    return pts

def draw_v(ax, pts, color, lw, alpha=0.7):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, solid_capstyle='round')

# Vessel paths
A0=bz((2,0),(1.9,0.7),(1.7,1.5),(1.4,2.0),25)+bz((1.4,2.0),(1.2,2.4),(0.9,2.7),(0.6,2.85),12)
A1=bz((1.7,1.5),(2.1,1.9),(2.6,2.3),(3.1,2.6),14)+bz((3.1,2.6),(3.4,2.7),(3.7,2.8),(3.9,2.85),8)
A2=bz((1.4,2.0),(1.1,2.3),(0.8,2.4),(0.5,2.4),10)
A3=bz((2.6,2.3),(2.65,2.55),(2.7,2.7),(2.75,2.85),8)
B0=bz((3.4,0),(3.3,0.6),(2.9,1.2),(2.5,1.6),20)+bz((2.5,1.6),(2.1,1.9),(1.7,2.2),(1.4,2.35),15)
B1=bz((2.9,1.2),(2.5,1.4),(2.0,1.5),(1.5,1.4),12)+bz((1.5,1.4),(1.2,1.35),(0.8,1.2),(0.5,1.1),10)
B2=bz((2.5,1.6),(2.8,2.0),(3.2,2.3),(3.7,2.45),12)+bz((3.7,2.45),(4.0,2.55),(4.3,2.6),(4.5,2.65),8)
B3=bz((3.3,0.6),(3.6,0.8),(3.9,1.0),(4.2,1.05),10)
allA=[A0,A1,A2,A3]; allB=[B0,B1,B2,B3]
jA=[(1.7,1.5),(1.4,2.0),(2.6,2.3)]; jB=[(2.9,1.2),(2.5,1.6),(3.3,0.6)]
eA=[(0.6,2.85),(3.9,2.85),(0.5,2.4),(2.75,2.85)]
eB=[(1.4,2.35),(0.5,1.1),(4.5,2.65),(4.2,1.05)]

for ax_sch, mode in [(ax_s1, "single"), (ax_s2, "multi")]:
    ax_sch.set_xlim(-0.1,5.0); ax_sch.set_ylim(-0.3,3.1)
    ax_sch.set_aspect('equal'); ax_sch.axis('off')
    ax_sch.fill_between([-0.1,5.0],-0.3,0.05,color='#ECF0F1',zorder=0)
    ax_sch.axhline(0.05,color='#BDC3C7',lw=0.8)
    ax_sch.text(4.8,-0.15,'ECM',fontsize=5,color='#95A5A6',ha='right',style='italic')

    if mode == "single":
        for p in allA: draw_v(ax_sch,p,'#27AE60',2.2,0.75)
        for p in allB: draw_v(ax_sch,p,'#27AE60',1.8,0.65)
        for j in jA+jB: ax_sch.plot(j[0],j[1],'o',color='#1E8449',ms=4,mec='white',mew=0.8,zorder=5)
        for e in eA+eB: ax_sch.plot(e[0],e[1],'o',color='white',ms=3,mec='#1E8449',mew=1.2,zorder=5)
        ax_sch.plot(2,0,'o',color='white',ms=3,mec='#1E8449',mew=1.2,zorder=5)
        ax_sch.plot(3.4,0,'o',color='white',ms=3,mec='#1E8449',mew=1.2,zorder=5)
        # False junction
        c=plt.Circle((1.75,2.05),0.22,fill=False,ec='#E74C3C',lw=1.2,ls='--',zorder=6)
        ax_sch.add_patch(c)
        ax_sch.annotate('False junction',xy=(1.75,2.05),xytext=(3.0,2.7),fontsize=5.5,color='#C0392B',fontweight='bold',ha='center',arrowprops=dict(arrowstyle='->',color='#C0392B',lw=0.8),zorder=7)
        ax_sch.annotate('Merged\n(length lost)',xy=(1.55,1.7),xytext=(0.3,0.5),fontsize=5,color='#7D6608',style='italic',ha='center',arrowprops=dict(arrowstyle='->',color='#D4A017',lw=0.6),bbox=dict(boxstyle='round,pad=0.15',fc='#FEF9E7',ec='#D4A017',lw=0.4),zorder=7)
        ax_sch.set_title('Single-color Skeleton',fontsize=8,fontweight='bold',color='#1E8449',pad=3)
    else:
        for p in allA: draw_v(ax_sch,p,'#2980B9',2.2,0.75)
        for p in allB: draw_v(ax_sch,p,'#C0392B',1.8,0.65)
        yw=bz((2.1,1.8),(1.95,1.95),(1.85,2.05),(1.75,2.1),8)
        draw_v(ax_sch,yw,'#D4A017',1.5,0.5)
        for j in jA: ax_sch.plot(j[0],j[1],'o',color='#2471A3',ms=4,mec='white',mew=0.8,zorder=5)
        for j in jB: ax_sch.plot(j[0],j[1],'o',color='#922B21',ms=4,mec='white',mew=0.8,zorder=5)
        for e in eA: ax_sch.plot(e[0],e[1],'o',color='white',ms=3,mec='#2471A3',mew=1.2,zorder=5)
        ax_sch.plot(2,0,'o',color='white',ms=3,mec='#2471A3',mew=1.2,zorder=5)
        for e in eB: ax_sch.plot(e[0],e[1],'o',color='white',ms=3,mec='#922B21',mew=1.2,zorder=5)
        ax_sch.plot(3.4,0,'o',color='white',ms=3,mec='#922B21',mew=1.2,zorder=5)
        c=plt.Circle((1.75,2.05),0.22,fill=False,ec='#27AE60',lw=1.2,ls='--',zorder=6)
        ax_sch.add_patch(c)
        ax_sch.annotate('Correctly\nseparated',xy=(1.75,2.05),xytext=(3.0,2.7),fontsize=5.5,color='#1E8449',fontweight='bold',ha='center',arrowprops=dict(arrowstyle='->',color='#27AE60',lw=0.8),zorder=7)
        ax_sch.annotate('2 paths\n(length recovered)',xy=(1.55,1.7),xytext=(0.3,0.5),fontsize=5,color='#196F3D',style='italic',ha='center',arrowprops=dict(arrowstyle='->',color='#27AE60',lw=0.6),bbox=dict(boxstyle='round,pad=0.15',fc='#EAFAF1',ec='#27AE60',lw=0.4),zorder=7)
        leg=[mpatches.Patch(color='#C0392B',label='Bottom'),mpatches.Patch(color='#D4A017',label='Middle'),mpatches.Patch(color='#2980B9',label='Top')]
        ax_sch.legend(handles=leg,fontsize=4.5,loc='lower right',framealpha=0.9,edgecolor='#BDC3C7',handlelength=0.8,handletextpad=0.3)
        ax_sch.set_title('Depth-separated Skeleton',fontsize=8,fontweight='bold',color='#2471A3',pad=3)

fig.text(0.01,0.98,'a',fontsize=12,fontweight='bold',va='top')


# ====== PANEL b: REPRESENTATIVE IMAGES ======
rep_sids = ["1-19-716", "16-18-716", "1-16-512"]
n_rep = len(rep_sids)
pb_top = 0.52; pb_h = 0.28
col_w = 0.225; gap = 0.008

for row, sid in enumerate(rep_sids):
    bf = load(gt_multi_dir/f"{sid}_real_A.png")
    fl_m = load(gt_multi_dir/f"{sid}_real_B.png")
    fl_s = load(gt_single_dir/f"{sid}_real_B.png")
    bf3 = bf if len(bf.shape)==3 else np.stack([bf]*3, axis=-1)
    vm, by = make_valid_mask(fl_m)

    mask_s = get_vessel_mask(fl_s); skel_s = skeletonize(mask_s)
    ep_s, jn_s = find_features(skel_s)
    s_j = count_in_mask(jn_s, vm); s_l = int(np.sum(skel_s & vm))

    bottom,middle,top_ = separate_layers(fl_m)
    masks_m = [bottom,middle,top_]
    skels_m = [skeletonize(m) for m in masks_m]
    m_j, m_l = 0, 0
    for sk in skels_m:
        _, jn = find_features(sk); m_j += count_in_mask(jn, vm); m_l += int(np.sum(sk & vm))
    bridged = bridge_skeletons(skels_m, masks_m)

    rh = pb_h / n_rep - 0.005
    y_pos = pb_top + (n_rep-1-row) * (pb_h / n_rep)

    for col in range(4):
        ax = fig.add_axes([0.03 + col*(col_w+gap), y_pos, col_w, rh])
        if col==0: ax.imshow(bf3)
        elif col==1: ax.imshow(fl_m)
        elif col==2:
            vis=bf3.copy(); vis[dilate(skel_s,5)]=(40,200,80); ax.imshow(vis)
            ax.text(0.5,0.03,f'J={s_j}  L={s_l}',transform=ax.transAxes,fontsize=5,ha='center',color='white',bbox=dict(fc='black',alpha=0.6,pad=1,boxstyle='round,pad=0.15'))
        elif col==3:
            vis=bf3.copy(); bt=dilate(bridged,5)
            for si,mask in enumerate(masks_m): vis[bt&mask]=layer_colors[si]
            vis[bt&~(masks_m[0]|masks_m[1]|masks_m[2])]=(0,220,180); ax.imshow(vis)
            ax.text(0.5,0.03,f'J={m_j}  L={m_l}',transform=ax.transAxes,fontsize=5,ha='center',color='white',bbox=dict(fc='black',alpha=0.6,pad=1,boxstyle='round,pad=0.15'))
        ax.axis('off')
        if row==0:
            titles=['Brightfield','Multi-color FL\n(GT)','Single-color\nSkeleton','Depth-separated\nSkeleton']
            ax.set_title(titles[col],fontsize=6.5,fontweight='bold',pad=2)
        if col==0:
            ax.text(-0.03,0.5,sid,transform=ax.transAxes,fontsize=5,rotation=90,va='center',ha='right',color='#555')

fig.text(0.01,0.81,'b',fontsize=12,fontweight='bold',va='top')


# ====== PANEL c: Single GT vs Multi GT (paired comparison) ======
# 4 metrics, paired bar chart
metric_labels = ["Junctions", "Length (px)", "Endpoints", "Area (px)"]
metric_keys = ["junctions", "length", "endpoints", "area"]
cs_c, cm_c = '#7BC67E', '#5B9BD5'

pc_y = 0.28; pc_h = 0.20
for mi, (mlab, mkey) in enumerate(zip(metric_labels, metric_keys)):
    ax = fig.add_axes([0.07 + mi*0.24, pc_y, 0.19, pc_h])
    d = stat_data["GT"][mkey]
    sv, mv, p = d["sv"], d["mv"], d["p"]
    sm, mm = np.mean(sv), np.mean(mv)
    se_s, se_m = np.std(sv)/np.sqrt(n), np.std(mv)/np.sqrt(n)

    ax.bar(0, sm, 0.6, yerr=se_s, capsize=3, color=cs_c, alpha=0.85, edgecolor='#555', lw=0.4)
    ax.bar(1, mm, 0.6, yerr=se_m, capsize=3, color=cm_c, alpha=0.85, edgecolor='#555', lw=0.4)

    mh = max(sm+se_s, mm+se_m)
    bh = mh*1.08
    ax.plot([0,0,1,1],[bh,bh*1.03,bh*1.03,bh],'k-',lw=0.5)
    ax.text(0.5, bh*1.04, pval_stars(p), ha='center', va='bottom', fontsize=7)

    ax.set_xticks([0,1])
    ax.set_xticklabels(['Single', 'Multi'], fontsize=6)
    ax.set_ylabel(mlab, fontsize=7)
    ax.set_title(f'GT: {mlab.split("(")[0].strip()}', fontsize=7, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.text(0.01, 0.49, 'c', fontsize=12, fontweight='bold', va='top')
# Legend for panel c
fig.text(0.12, 0.49, 'Single-color', fontsize=6, color=cs_c, fontweight='bold')
fig.text(0.22, 0.49, 'vs', fontsize=6, color='#555')
fig.text(0.26, 0.49, 'Depth-separated', fontsize=6, color=cm_c, fontweight='bold')
fig.text(0.42, 0.49, '(GT fluorescence, n=215, paired t-test)', fontsize=5.5, color='#777')


# ====== PANEL d: Multi GT=1.0 normalized, LBBDM/PBBDM ratio ======
pd_y = 0.04; pd_h = 0.20
colors_model = {'LBBDM': '#E67E22', 'PBBDM': '#8E44AD'}

for mi, (mlab, mkey) in enumerate(zip(metric_labels, metric_keys)):
    ax = fig.add_axes([0.07 + mi*0.24, pd_y, 0.19, pd_h])

    gt_mv = stat_data["GT"][mkey]["mv"]

    # Per-sample ratio: model_multi / gt_multi
    ratios = {}
    for model in ["LBBDM", "PBBDM"]:
        model_mv = stat_data[model][mkey]["mv"]
        r = []
        for g, m in zip(gt_mv, model_mv):
            if g > 0:
                r.append(m / g)
        ratios[model] = r

    x = np.arange(2); w = 0.5
    for ix, model in enumerate(["LBBDM", "PBBDM"]):
        rm = np.mean(ratios[model])
        rs = np.std(ratios[model]) / np.sqrt(len(ratios[model]))
        ax.bar(ix, rm, w, yerr=rs, capsize=3,
               color=colors_model[model], alpha=0.8, edgecolor='#555', lw=0.4)
        # Value label
        ax.text(ix, rm + rs + 0.02, f'{rm:.2f}', ha='center', va='bottom', fontsize=5.5, color='#333')

    ax.axhline(1.0, color='#333', lw=0.8, ls='--', alpha=0.5, zorder=0)
    ax.text(-0.4, 1.01, 'GT', fontsize=5, color='#555', va='bottom', style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(["LBBDM", "PBBDM"], fontsize=6)
    ax.set_ylabel(f'{mlab.split("(")[0].strip()} Ratio', fontsize=6.5)
    ax.set_title(mlab.split("(")[0].strip(), fontsize=7, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(1.3, ax.get_ylim()[1]))

fig.text(0.01, 0.25, 'd', fontsize=12, fontweight='bold', va='top')
fig.text(0.12, 0.25, 'Normalized to Multi-color GT (dashed line = 1.0)', fontsize=6, color='#555')

# ====== SAVE ======
plt.savefig(OUT/"Fig_depth_v2.png", dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
plt.savefig(OUT/"Fig_depth_v2.pdf", dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
plt.close()
print("\nSaved: Fig_depth_v2.png/pdf")
