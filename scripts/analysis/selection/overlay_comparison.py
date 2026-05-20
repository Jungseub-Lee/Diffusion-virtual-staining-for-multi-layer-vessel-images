"""
Compare skeleton overlay on BF vs fluorescence for representative samples.
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

rep_sids = ["23-13-512", "14-15-512", "1-15-512"]


def separate_layers_hsv(img, min_area=50):
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

def load(p): return np.array(Image.open(p).convert("RGB"))


for sid in rep_sids:
    bf = load(BASE/"Multi-color"/"Pix2pix"/f"{sid}_real_A.png")
    fl_multi = load(BASE/"Multi-color"/"Pix2pix"/f"{sid}_real_B.png")
    fl_single = load(BASE/"Single-color"/"Pix2pix"/f"{sid}_real_B.png")

    # Convert BF to 3ch for overlay
    if len(bf.shape) == 2:
        bf3 = np.stack([bf]*3, axis=-1)
    else:
        bf3 = bf.copy()

    # Single-color skeleton
    s_mask = get_vessel_mask(fl_single)
    s_skel = skeletonize(s_mask)
    s_thick = dilate(s_skel, 5)

    # Multi-color depth skeleton
    bottom, middle, top = separate_layers_hsv(fl_multi)
    masks = [bottom, middle, top]
    skels = [skeletonize(m) for m in masks]
    bridged = bridge_skeletons(skels, masks)
    b_thick = dilate(bridged, 5)

    layer_colors = [(255,80,80),(255,255,60),(80,140,255)]

    # ============================================================
    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
    fig.suptitle(f"Sample: {sid}", fontsize=18, fontweight='bold')

    # --- Row 0: Overlay on BF ---
    # BF original
    axes[0,0].imshow(bf3)
    axes[0,0].set_title("Brightfield (BF)", fontsize=13)

    # Single skeleton on BF
    bf_single = bf3.copy()
    bf_single[s_thick] = (0, 255, 0)
    axes[0,1].imshow(bf_single)
    axes[0,1].set_title("Single-color Skeleton on BF", fontsize=13)

    # Depth skeleton on BF
    bf_depth = bf3.copy()
    for si, mask in enumerate(masks):
        bf_depth[b_thick & mask] = layer_colors[si]
    bf_depth[b_thick & ~(masks[0]|masks[1]|masks[2])] = (0,255,200)
    axes[0,2].imshow(bf_depth)
    axes[0,2].set_title("Depth-sep Skeleton on BF", fontsize=13)

    # Side by side overlay (split view)
    h_img = bf3.shape[1]
    mid = h_img // 2
    split = bf3.copy()
    split[:, :mid][s_thick[:, :mid]] = (0, 255, 0)
    for si, mask in enumerate(masks):
        split[:, mid:][b_thick[:, mid:] & mask[:, mid:]] = layer_colors[si]
    split[:, mid:][b_thick[:, mid:] & ~(masks[0][:, mid:]|masks[1][:, mid:]|masks[2][:, mid:])] = (0,255,200)
    # Draw divider line
    cv2.line(split, (mid, 0), (mid, split.shape[0]), (255,255,255), 2)
    axes[0,3].imshow(split)
    axes[0,3].set_title("Split: Single (left) | Depth-sep (right) on BF", fontsize=13)

    # --- Row 1: Overlay on Fluorescence ---
    # Fluorescence originals
    axes[1,0].imshow(fl_multi)
    axes[1,0].set_title("Multi-color Fluorescence (GT)", fontsize=13)

    # Single skeleton on single FL (dimmed)
    fl_s_dim = (fl_single * 0.35).astype(np.uint8)
    fl_s_dim[s_thick] = (0, 255, 0)
    axes[1,1].imshow(fl_s_dim)
    axes[1,1].set_title("Single-color Skeleton on FL", fontsize=13)

    # Depth skeleton on multi FL (dimmed)
    fl_m_dim = (fl_multi * 0.35).astype(np.uint8)
    for si, mask in enumerate(masks):
        fl_m_dim[b_thick & mask] = layer_colors[si]
    fl_m_dim[b_thick & ~(masks[0]|masks[1]|masks[2])] = (0,255,200)
    axes[1,2].imshow(fl_m_dim)
    axes[1,2].set_title("Depth-sep Skeleton on FL", fontsize=13)

    # Split on FL
    split_fl = np.zeros_like(bf3)
    fl_s_half = (fl_single * 0.35).astype(np.uint8)
    fl_m_half = (fl_multi * 0.35).astype(np.uint8)
    split_fl[:, :mid] = fl_s_half[:, :mid]
    split_fl[:, mid:] = fl_m_half[:, mid:]
    split_fl[:, :mid][s_thick[:, :mid]] = (0, 255, 0)
    for si, mask in enumerate(masks):
        split_fl[:, mid:][b_thick[:, mid:] & mask[:, mid:]] = layer_colors[si]
    split_fl[:, mid:][b_thick[:, mid:] & ~(masks[0][:, mid:]|masks[1][:, mid:]|masks[2][:, mid:])] = (0,255,200)
    cv2.line(split_fl, (mid, 0), (mid, split_fl.shape[0]), (255,255,255), 2)
    axes[1,3].imshow(split_fl)
    axes[1,3].set_title("Split: Single (left) | Depth-sep (right) on FL", fontsize=13)

    for ax in axes.flat: ax.axis("off")
    axes[0,0].text(-0.02, 0.5, "On BF", transform=axes[0,0].transAxes,
                   fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)
    axes[1,0].text(-0.02, 0.5, "On FL", transform=axes[1,0].transAxes,
                   fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)

    plt.tight_layout()
    plt.savefig(output_dir / f"overlay_bf_vs_fl_{sid}.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: overlay_bf_vs_fl_{sid}.png")

print("Done.")
