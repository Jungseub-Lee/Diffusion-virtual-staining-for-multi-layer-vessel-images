"""
Generate individual PNG images for each layer/step for HTML viewer.
"""
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from skimage.morphology import skeletonize, remove_small_objects

data_dir = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data\Multi-color\Pix2pix")
out = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output\individual")
out.mkdir(exist_ok=True, parents=True)

sample_ids = ["1-0-512", "1-5-512", "1-10-512", "1-15-512"]


def separate_layers(img_rgb, min_area=50):
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


def dilate_skel(skel, t=5):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t,t))
    return cv2.dilate(skel.astype(np.uint8)*255, k, iterations=1) > 0


def save(arr, path):
    Image.fromarray(arr).save(path)


for sid in sample_ids:
    img = np.array(Image.open(data_dir / f"{sid}_real_B.png").convert("RGB"))
    bottom, middle, top = separate_layers(img)
    sb = skeletonize(bottom)
    sm = skeletonize(middle)
    st_ = skeletonize(top)
    tb = dilate_skel(sb, 5)
    tm = dilate_skel(sm, 5)
    tt = dilate_skel(st_, 5)
    dim = (img * 0.3).astype(np.uint8)

    # 1. Original
    save(img, out / f"{sid}_original.png")

    # 2. Each mask - show original pixels on black
    for name, mask in [("bottom", bottom), ("middle", middle), ("top", top)]:
        v = np.zeros_like(img); v[mask] = img[mask]
        save(v, out / f"{sid}_mask_{name}.png")

    # 3. Each mask overlay on dimmed original
    colors = {"bottom": [180,30,30], "middle": [220,220,30], "top": [30,100,255]}
    for name, mask in [("bottom", bottom), ("middle", middle), ("top", top)]:
        ov = dim.copy()
        ov[mask] = np.clip(img[mask]*0.5 + np.array(colors[name])*0.5, 0, 255).astype(np.uint8)
        save(ov, out / f"{sid}_overlay_{name}.png")

    # All masks combined overlay
    ov_all = dim.copy()
    ov_all[bottom] = np.clip(ov_all[bottom].astype(float) + np.array([180,30,30])*0.6, 0, 255).astype(np.uint8)
    ov_all[middle] = np.clip(ov_all[middle].astype(float) + np.array([220,220,30])*0.6, 0, 255).astype(np.uint8)
    ov_all[top] = np.clip(ov_all[top].astype(float) + np.array([30,100,255])*0.6, 0, 255).astype(np.uint8)
    save(ov_all, out / f"{sid}_overlay_all.png")

    # 4. Each skeleton on dimmed original
    skel_colors = {"bottom": [255,80,80], "middle": [255,255,60], "top": [80,140,255]}
    for name, thick in [("bottom", tb), ("middle", tm), ("top", tt)]:
        sk = dim.copy(); sk[thick] = skel_colors[name]
        save(sk, out / f"{sid}_skel_{name}.png")

    # All skeletons combined
    sk_all = dim.copy()
    sk_all[tb] = [255,80,80]
    sk_all[tm] = [255,255,60]
    sk_all[tt] = [80,140,255]
    save(sk_all, out / f"{sid}_skel_all.png")

    print(f"Done: {sid}")

print("All images saved.")
