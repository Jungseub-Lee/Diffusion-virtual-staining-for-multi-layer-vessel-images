"""
Visualize selected samples: BF overlay + junction point analysis.
Show why junctions increase in depth-separated vs single-color.
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

sids = ["1-19-716", "9-15-512", "16-18-716", "18-19-512", "1-16-512"]

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

def baseline_mask(img, thr=0.35):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v = gray > 15; h, w = v.shape
    rd = np.sum(v, axis=1) / w
    bt = h
    for y in range(h-1,-1,-1):
        if rd[y] < thr: bt = y; break
    vm = np.ones((h,w), dtype=bool); vm[bt:,:] = False
    return vm

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
    return np.argwhere(nb==1), np.argwhere(nb>=3), nb

def dilate(s, t=5):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
    return cv2.dilate(s.astype(np.uint8)*255,k,iterations=1)>0

def metrics_masked(skel, vm):
    su = skel.astype(np.uint8)
    ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su,-1,ks)*su
    vnb = nb * vm.astype(np.uint8)
    vs = skel & vm
    return {
        "length": int(np.sum(vs)),
        "junctions": int(np.sum(vnb>=3)),
        "endpoints": int(np.sum(vnb==1)),
    }

layer_colors = [(255,80,80),(255,255,60),(80,140,255)]
layer_names = ["Bottom","Middle","Top"]

# ============================================================
# Figure 1: BF overlay (5 samples x 4 cols)
# ============================================================
fig1, ax1 = plt.subplots(len(sids), 4, figsize=(24, 6*len(sids)))

for row, sid in enumerate(sids):
    gt_m_path = BASE/"Multi-color"/"Pix2pix"/f"{sid}_real_B.png"
    gt_s_path = BASE/"Single-color"/"Pix2pix"/f"{sid}_real_B.png"
    bf_path = BASE/"Multi-color"/"Pix2pix"/f"{sid}_real_A.png"

    if not gt_m_path.exists():
        print(f"SKIP {sid}: not found"); continue

    bf = load(bf_path)
    fl_m = load(gt_m_path)
    fl_s = load(gt_s_path)
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
plt.savefig(output_dir / "selected_bf_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: selected_bf_overlay.png")


# ============================================================
# Figure 2: Junction analysis (5 samples x 4 cols)
# Why junctions increase in depth-separated
# ============================================================
fig2, ax2 = plt.subplots(len(sids), 4, figsize=(28, 7*len(sids)))

for row, sid in enumerate(sids):
    gt_m_path = BASE/"Multi-color"/"Pix2pix"/f"{sid}_real_B.png"
    gt_s_path = BASE/"Single-color"/"Pix2pix"/f"{sid}_real_B.png"

    if not gt_m_path.exists(): continue

    fl_m = load(gt_m_path)
    fl_s = load(gt_s_path)
    vm = baseline_mask(fl_m)

    # Single
    s_skel = skeletonize(get_vessel_mask(fl_s))
    s_ep, s_jn, _ = find_features(s_skel)
    s_m = metrics_masked(s_skel, vm)

    # Multi depth
    bottom, middle, top = separate_layers(fl_m)
    masks = [bottom, middle, top]
    skels = [skeletonize(m) for m in masks]
    bridged = bridge_skeletons(skels, masks)
    d_ep, d_jn, _ = find_features(bridged)
    d_m = metrics_masked(bridged, vm)

    dim_s = (fl_s * 0.3).astype(np.uint8)
    dim_m = (fl_m * 0.3).astype(np.uint8)

    # Col 0: Single skeleton + junctions (cyan dots) + endpoints (green dots)
    vis0 = dim_s.copy()
    vis0[dilate(s_skel,4)] = (0,200,0)
    for y,x in s_jn:
        if vm[y,x]: cv2.circle(vis0, (x,y), 7, (0,255,255), -1)  # cyan = junction
    for y,x in s_ep:
        if vm[y,x]: cv2.circle(vis0, (x,y), 6, (255,255,255), 2)  # white ring = endpoint
    ax2[row,0].imshow(vis0)
    ax2[row,0].set_title(f"{sid} — Single-color\nJ={s_m['junctions']} (cyan), E={s_m['endpoints']} (white)", fontsize=11)

    # Col 1: Multi-color depth skeleton + junctions + endpoints
    vis1 = dim_m.copy()
    bt = dilate(bridged,4)
    for si,mask in enumerate(masks): vis1[bt & mask] = layer_colors[si]
    vis1[bt & ~(masks[0]|masks[1]|masks[2])] = (0,255,200)
    for y,x in d_jn:
        if vm[y,x]: cv2.circle(vis1, (x,y), 7, (0,255,255), -1)
    for y,x in d_ep:
        if vm[y,x]: cv2.circle(vis1, (x,y), 6, (255,255,255), 2)
    ax2[row,1].imshow(vis1)
    ax2[row,1].set_title(f"Depth-separated (bridged)\nJ={d_m['junctions']} (cyan), E={d_m['endpoints']} (white)", fontsize=11)

    # Col 2: Original multi-color FL for reference
    ax2[row,2].imshow(fl_m)
    ax2[row,2].set_title("Multi-color FL (GT reference)", fontsize=11)

    # Col 3: Junction difference map
    # Show where depth-sep has junctions that single doesn't
    # Blue = single-only junctions, Red = depth-only junctions, Cyan = shared
    vis3 = np.zeros_like(fl_m)

    # Create junction proximity maps (dilated for visibility)
    s_jn_map = np.zeros(fl_m.shape[:2], dtype=bool)
    for y,x in s_jn:
        if vm[y,x]: cv2.circle(s_jn_map.view(np.uint8), (x,y), 8, 1, -1)
    d_jn_map = np.zeros(fl_m.shape[:2], dtype=bool)
    for y,x in d_jn:
        if vm[y,x]: cv2.circle(d_jn_map.view(np.uint8), (x,y), 8, 1, -1)

    # Shared
    shared = s_jn_map & d_jn_map
    # Single-only
    single_only = s_jn_map & ~d_jn_map
    # Depth-only (NEW junctions from depth separation)
    depth_only = d_jn_map & ~s_jn_map

    # Background: dimmed original
    vis3 = (fl_m * 0.2).astype(np.uint8)
    vis3[shared] = (0, 255, 255)      # cyan = shared
    vis3[single_only] = (100, 100, 255)  # blue = single-only
    vis3[depth_only] = (255, 100, 100)   # red = depth-only (new from depth)

    n_shared = np.sum(shared) // 200  # approximate count
    ax2[row,3].imshow(vis3)
    ax2[row,3].set_title(f"Junction Difference Map\n"
                         f"Cyan=shared, Blue=single-only, Red=depth-new\n"
                         f"Single J={s_m['junctions']}, Depth J={d_m['junctions']}, Diff={d_m['junctions']-s_m['junctions']:+d}",
                         fontsize=10)

for ax in ax2.flat: ax.axis("off")
plt.tight_layout()
plt.savefig(output_dir / "selected_junction_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: selected_junction_analysis.png")
