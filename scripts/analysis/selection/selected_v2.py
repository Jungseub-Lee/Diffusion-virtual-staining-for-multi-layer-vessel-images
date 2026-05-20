"""
Selected samples v2: replace 9-15-512, 18-19-512 with new ones.
Junction analysis with baseline exclusion only.
"""
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data")
output_dir = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output")
gt_multi = BASE/"Multi-color"/"Pix2pix"
gt_single = BASE/"Single-color"/"Pix2pix"

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

def make_valid_mask(img, thr=0.35):
    h, w = img.shape[:2]
    by = get_baseline_y(img, thr)
    vm = np.ones((h,w), dtype=bool); vm[by:,:] = False
    return vm, by

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

def find_features(skeleton):
    su = skeleton.astype(np.uint8)
    ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su,-1,ks)*su
    return np.argwhere(nb==1), np.argwhere(nb>=3)

def dilate(s, t=5):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
    return cv2.dilate(s.astype(np.uint8)*255,k,iterations=1)>0

def count_in_mask(coords, mask):
    return sum(1 for y,x in coords if mask[y,x])

layer_colors = [(255,80,80),(255,255,60),(80,140,255)]

# ============================================================
# Find 2 replacement samples (exclude 9-15-512, 18-19-512)
# ============================================================
keep = {"1-19-716", "16-18-716", "1-16-512"}
exclude = {"9-15-512", "18-19-512"} | keep

single_set = {f.stem.replace("_real_B","") for f in gt_single.glob("*_real_B.png")}
candidates = []
for f in sorted(gt_multi.glob("*_real_B.png")):
    sid = f.stem.replace("_real_B","")
    if sid in exclude or sid not in single_set: continue
    img = load(f)
    vm, by = make_valid_mask(img)
    if by < 60: continue
    roi = img[:by,:]
    bottom, middle, top = separate_layers(roi)
    ba,ma,ta = np.sum(bottom), np.sum(middle), np.sum(top)
    total = ba+ma+ta
    if total < 3000: continue
    balance = min(ba,ma,ta) / (total/3 + 1)
    # Want good sprouts in valid region, not just baseline
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    hcov = np.sum(np.sum(gray_roi>15, axis=1)>5) / roi.shape[0]
    score = balance * total * hcov
    candidates.append((sid, score, balance, total, ba, ma, ta))

candidates.sort(key=lambda x: x[1], reverse=True)
print("Top replacement candidates:")
for c in candidates[:15]:
    print(f"  {c[0]:<15s} score={c[1]:>8.0f} bal={c[2]:.2f} B={c[4]:>6d} M={c[5]:>6d} T={c[6]:>6d}")

# Pick 2 from different chips for diversity
new1 = candidates[0][0]  # top score
new2 = None
chip1 = new1.split("-")[0]
for c in candidates[1:]:
    if c[0].split("-")[0] != chip1:
        new2 = c[0]; break
if new2 is None:
    new2 = candidates[1][0]

print(f"\nReplacements: {new1}, {new2}")

sids = ["1-19-716", new1, "16-18-716", new2, "1-16-512"]
print(f"Final selection: {sids}")


# ============================================================
# Figure 1: BF overlay
# ============================================================
fig1, ax1 = plt.subplots(len(sids), 4, figsize=(24, 6*len(sids)))

for row, sid in enumerate(sids):
    bf = load(gt_multi/f"{sid}_real_A.png")
    fl_m = load(gt_multi/f"{sid}_real_B.png")
    fl_s = load(gt_single/f"{sid}_real_B.png")
    bf3 = bf if len(bf.shape)==3 else np.stack([bf]*3, axis=-1)

    s_skel = skeletonize(get_vessel_mask(fl_s))
    bottom, middle, top = separate_layers(fl_m)
    masks = [bottom, middle, top]
    skels = [skeletonize(m) for m in masks]
    bridged = bridge_skeletons(skels, masks)

    ax1[row,0].imshow(bf3); ax1[row,0].set_title(f"{sid}\nBrightfield", fontsize=11)
    ax1[row,1].imshow(fl_m); ax1[row,1].set_title("Multi-color FL (GT)", fontsize=11)

    vis_s = bf3.copy(); vis_s[dilate(s_skel,5)] = (0,255,0)
    ax1[row,2].imshow(vis_s); ax1[row,2].set_title("Single Skeleton on BF", fontsize=11)

    vis_d = bf3.copy()
    bt = dilate(bridged,5)
    for si,mask in enumerate(masks): vis_d[bt & mask] = layer_colors[si]
    vis_d[bt & ~(masks[0]|masks[1]|masks[2])] = (0,255,200)
    ax1[row,3].imshow(vis_d); ax1[row,3].set_title("Depth-sep Skeleton on BF", fontsize=11)

for ax in ax1.flat: ax.axis("off")
plt.tight_layout()
plt.savefig(output_dir / "selected_v2_bf.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: selected_v2_bf.png")


# ============================================================
# Figure 2: Junction analysis (baseline excluded)
# ============================================================
fig2, ax2 = plt.subplots(len(sids), 4, figsize=(28, 7*len(sids)))

for row, sid in enumerate(sids):
    fl_m = load(gt_multi/f"{sid}_real_B.png")
    fl_s = load(gt_single/f"{sid}_real_B.png")
    vm, by = make_valid_mask(fl_m)

    s_skel = skeletonize(get_vessel_mask(fl_s))
    s_ep, s_jn = find_features(s_skel)
    s_j_valid = count_in_mask(s_jn, vm)
    s_e_valid = count_in_mask(s_ep, vm)

    bottom, middle, top = separate_layers(fl_m)
    masks = [bottom, middle, top]
    skels = [skeletonize(m) for m in masks]
    bridged = bridge_skeletons(skels, masks)
    d_ep, d_jn = find_features(bridged)
    d_j_valid = count_in_mask(d_jn, vm)
    d_e_valid = count_in_mask(d_ep, vm)

    dim_s = (fl_s * 0.3).astype(np.uint8)
    dim_m = (fl_m * 0.3).astype(np.uint8)

    # Draw baseline boundary
    def draw_baseline(vis, by):
        cv2.line(vis, (0,by), (vis.shape[1],by), (255,255,0), 1)
        # dim below baseline
        vis[by:,:] = (vis[by:,:] * 0.3).astype(np.uint8)
        return vis

    # Col 0: Single skeleton + junctions/endpoints (baseline excluded)
    vis0 = dim_s.copy()
    vis0[dilate(s_skel,4)] = (0,200,0)
    for y,x in s_jn:
        if vm[y,x]: cv2.circle(vis0, (x,y), 7, (0,255,255), -1)
    for y,x in s_ep:
        if vm[y,x]: cv2.circle(vis0, (x,y), 6, (255,255,255), 2)
    vis0 = draw_baseline(vis0, by)
    ax2[row,0].imshow(vis0)
    ax2[row,0].set_title(f"{sid} — Single-color\nJ={s_j_valid} (cyan), E={s_e_valid} (white)\n[above baseline only]", fontsize=10)

    # Col 1: Depth skeleton + junctions/endpoints (baseline excluded)
    vis1 = dim_m.copy()
    bth = dilate(bridged,4)
    for si,mask in enumerate(masks): vis1[bth & mask] = layer_colors[si]
    vis1[bth & ~(masks[0]|masks[1]|masks[2])] = (0,255,200)
    for y,x in d_jn:
        if vm[y,x]: cv2.circle(vis1, (x,y), 7, (0,255,255), -1)
    for y,x in d_ep:
        if vm[y,x]: cv2.circle(vis1, (x,y), 6, (255,255,255), 2)
    vis1 = draw_baseline(vis1, by)
    ax2[row,1].imshow(vis1)
    ax2[row,1].set_title(f"Depth-separated (bridged)\nJ={d_j_valid} (cyan), E={d_e_valid} (white)\n[above baseline only]", fontsize=10)

    # Col 2: Multi-color FL reference with baseline
    vis2 = fl_m.copy()
    vis2 = draw_baseline(vis2, by)
    ax2[row,2].imshow(vis2)
    ax2[row,2].set_title("Multi-color FL (GT)\n[yellow line = baseline]", fontsize=10)

    # Col 3: Junction difference map (baseline excluded only)
    vis3 = (fl_m * 0.15).astype(np.uint8)

    # Mark junctions with larger markers for clarity
    for y,x in s_jn:
        if vm[y,x]: cv2.circle(vis3, (x,y), 9, (100,255,100), -1)  # green = single junction
    for y,x in d_jn:
        if vm[y,x]:
            # Check if near any single junction
            near_single = False
            for sy,sx in s_jn:
                if vm[sy,sx] and abs(y-sy)<12 and abs(x-sx)<12:
                    near_single = True; break
            if near_single:
                cv2.circle(vis3, (x,y), 9, (0,255,255), 2)  # cyan ring = shared
            else:
                cv2.circle(vis3, (x,y), 9, (255,80,80), -1)  # red = depth-only NEW

    vis3 = draw_baseline(vis3, by)
    diff = d_j_valid - s_j_valid
    ax2[row,3].imshow(vis3)
    ax2[row,3].set_title(f"Junction Map (above baseline)\n"
                         f"Green=single, Red=depth-new, Cyan=shared\n"
                         f"S:{s_j_valid} → D:{d_j_valid} (diff: {diff:+d})", fontsize=10)

for ax in ax2.flat: ax.axis("off")
plt.tight_layout()
plt.savefig(output_dir / "selected_v2_junctions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: selected_v2_junctions.png")
