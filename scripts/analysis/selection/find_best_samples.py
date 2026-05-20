"""
Score all samples and visualize top candidates on BF.
Criteria: balanced layers, good vessel density, clear sprout structures.
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
gt_multi = BASE / "Multi-color" / "Pix2pix"
gt_single = BASE / "Single-color" / "Pix2pix"

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

def baseline_mask(img, thr=0.35):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v = gray > 15
    h, w = v.shape
    rd = np.sum(v, axis=1) / w
    bt = h
    for y in range(h-1,-1,-1):
        if rd[y] < thr: bt = y; break
    return bt

# Score all samples
multi_files = sorted(gt_multi.glob("*_real_B.png"))
single_set = {f.stem.replace("_real_B","") for f in gt_single.glob("*_real_B.png")}

scores = []
for f in multi_files:
    sid = f.stem.replace("_real_B","")
    if sid not in single_set: continue

    img = load(f)
    bt = baseline_mask(img)
    if bt < 50: continue

    roi = img[:bt, :]
    bottom, middle, top = separate_layers(roi)

    ba = np.sum(bottom)
    ma = np.sum(middle)
    ta = np.sum(top)
    total = ba + ma + ta
    if total < 3000: continue

    # Balance: min layer / avg layer
    avg = total / 3
    balance = min(ba, ma, ta) / (avg + 1)

    # Sprout coverage: fraction of ROI height with vessels
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    vessel_rows = np.sum(gray_roi > 15, axis=1) > 5
    height_coverage = np.sum(vessel_rows) / roi.shape[0]

    score = balance * total * height_coverage
    scores.append((sid, score, balance, total, ba, ma, ta, height_coverage, bt))

scores.sort(key=lambda x: x[1], reverse=True)
print("Top 20 samples:")
for i, s in enumerate(scores[:20]):
    print(f"  {i+1:2d}. {s[0]:<15s} score={s[1]:>10.0f} bal={s[2]:.2f} total={s[3]:>7d} B={s[4]:>6d} M={s[5]:>6d} T={s[6]:>6d} hcov={s[7]:.2f}")

# Visualize top 8 on BF
top_sids = [s[0] for s in scores[:8]]

def get_vessel_mask(img, min_area=50):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v = gray > 15
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    u = v.astype(np.uint8)*255
    u = cv2.morphologyEx(u,cv2.MORPH_CLOSE,k,iterations=2)
    u = cv2.morphologyEx(u,cv2.MORPH_OPEN,k,iterations=1)
    return remove_small_objects(u>0, min_size=min_area)

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

layer_colors = [(255,80,80),(255,255,60),(80,140,255)]

fig, axes = plt.subplots(len(top_sids), 4, figsize=(24, 6*len(top_sids)))

for row, sid in enumerate(top_sids):
    bf = load(gt_multi / f"{sid}_real_A.png")
    fl_m = load(gt_multi / f"{sid}_real_B.png")
    fl_s = load(gt_single / f"{sid}_real_B.png")

    bf3 = bf if len(bf.shape)==3 else np.stack([bf]*3, axis=-1)

    # Single skeleton
    s_skel = skeletonize(get_vessel_mask(fl_s))
    s_thick = dilate(s_skel, 5)

    # Multi depth skeleton
    bottom, middle, top = separate_layers(fl_m)
    masks = [bottom, middle, top]
    skels = [skeletonize(m) for m in masks]
    bridged = bridge_skeletons(skels, masks)
    b_thick = dilate(bridged, 5)

    # Col 0: BF
    axes[row,0].imshow(bf3)
    axes[row,0].set_title(f"{sid}\nBrightfield")

    # Col 1: Multi-color FL
    axes[row,1].imshow(fl_m)
    axes[row,1].set_title("Multi-color FL (GT)")

    # Col 2: Single skeleton on BF
    vis2 = bf3.copy()
    vis2[s_thick] = (0,255,0)
    axes[row,2].imshow(vis2)
    axes[row,2].set_title("Single-color Skeleton on BF")

    # Col 3: Depth skeleton on BF
    vis3 = bf3.copy()
    for si, mask in enumerate(masks):
        vis3[b_thick & mask] = layer_colors[si]
    vis3[b_thick & ~(masks[0]|masks[1]|masks[2])] = (0,255,200)
    axes[row,3].imshow(vis3)
    axes[row,3].set_title("Depth-sep Skeleton on BF")

for ax in axes.flat: ax.axis("off")
plt.tight_layout()
plt.savefig(output_dir / "top8_bf_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: top8_bf_overlay.png")
