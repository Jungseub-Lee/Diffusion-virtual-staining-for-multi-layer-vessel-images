"""
Create Figure 1E panel images (thumbnails) for PPT assembly.
Uses a different crop region from previous panels, with bridged skeletons.
"""
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output\paper_panels_depth')
DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')

s1 = np.array(Image.open(DATA / 'Original color stack data' / 'Sample 1_BF and GT(512x512 two images).png'))
bf_raw = s1[:512]
gt_raw = s1[512:]
gt_blur = cv2.GaussianBlur(gt_raw, (7, 7), 0)
h, w = gt_raw.shape[:2]

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

R = gt_blur[:,:,0].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
diff = B - R

# === Build bridged skeletons (same pipeline) ===
vessel_all = remove_small_objects(
    cv2.morphologyEx((cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY) > 10).astype(np.uint8)*255,
                     cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=100)
vessel_dilated = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

r_mask = remove_small_objects(
    cv2.morphologyEx((R > 30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=50)
r_skel = skeletonize(r_mask)
r_filtered = r_skel & (diff < 0)

b_seed = B > 50; b_low = B > 20
b_ll, nl = ndimage.label(b_low)
b_hyst = np.zeros_like(b_low, dtype=bool)
for i in range(1, nl+1):
    reg = b_ll == i
    if np.any(b_seed[reg]): b_hyst |= reg
b_hyst = remove_small_objects(
    cv2.morphologyEx(b_hyst.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0, min_size=50)
b_skel = skeletonize(b_hyst)
b_filtered = b_skel & (diff > 0)

def get_eps(skel, tl=20):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    eps = np.argwhere(nb == 1); res = []
    for ep in eps:
        y0, x0 = ep; path = [(y0,x0)]; vis = {(y0,x0)}; cy,cx = y0,x0
        for _ in range(tl):
            f = False
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    if dy==0 and dx==0: continue
                    ny,nx = cy+dy, cx+dx
                    if 0<=ny<h and 0<=nx<w and skel[ny,nx] and (ny,nx) not in vis:
                        path.append((ny,nx)); vis.add((ny,nx)); cy,cx=ny,nx; f=True; break
                if f: break
            if not f: break
        if len(path) >= 5:
            n = min(len(path), 10)
            t = np.array(path[0], dtype=float) - np.array(path[n-1], dtype=float)
            nm = np.linalg.norm(t)
            if nm > 0: t /= nm
            res.append({'pos':(y0,x0),'tangent':t})
    return res

def bezier(e1, e2):
    p1=np.array(e1['pos'],dtype=float); p2=np.array(e2['pos'],dtype=float)
    d=np.linalg.norm(p2-p1); t1,t2=e1['tangent'],e2['tangent']
    c1=p1+t1*d*0.4; c2=p2+t2*d*0.4; n=max(int(d*1.5),10)
    pts = []
    for t in np.linspace(0,1,n):
        p = (1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2
        pts.append((int(round(p[0])), int(round(p[1]))))
    return pts

def do_bridge(skel, vm):
    eps = get_eps(skel); ct = np.cos(np.radians(60)); cands = []
    for i in range(len(eps)):
        for j in range(i+1, len(eps)):
            p1=np.array(eps[i]['pos'],dtype=float); p2=np.array(eps[j]['pos'],dtype=float)
            d=np.linalg.norm(p2-p1)
            if d<5 or d>80: continue
            d12=(p2-p1)/d; c1=np.dot(eps[i]['tangent'],d12); c2=np.dot(eps[j]['tangent'],-d12)
            if c1<ct or c2<ct: continue
            pts=bezier(eps[i],eps[j])
            ov=sum(1 for(y,x) in pts if 0<=y<h and 0<=x<w and vm[y,x])
            vr=ov/max(len(pts),1)
            if vr<0.7: continue
            sc=(c1+c2)/2-d/80*0.2+vr*0.2
            cands.append((i,j,sc,eps[i],eps[j]))
    cands.sort(key=lambda x:-x[2]); used=set(); br=skel.copy()
    for i,j,sc,e1,e2 in cands:
        if i not in used and j not in used:
            for(y,x) in bezier(e1,e2):
                if 0<=y<h and 0<=x<w: br[y,x]=True
            used.add(i); used.add(j)
    return skeletonize(br)

r_bridged = do_bridge(r_filtered, vessel_dilated)
b_bridged = do_bridge(b_filtered, vessel_dilated)

# === Crop region: (280, 400) — good density, different from usual ===
cy, cx, crop = 280, 400, 90
y0, y1 = max(0, cy-crop), min(h, cy+crop)
x0, x1 = max(0, cx-crop), min(w, cx+crop)

# 1. BF thumbnail
bf_crop = bf_raw[y0:y1, x0:x1]
Image.fromarray(bf_crop).save(OUT / 'fig1e_bf.png')

# 2. Multi-color GT thumbnail
gt_crop = gt_raw[y0:y1, x0:x1]
Image.fromarray(gt_crop).save(OUT / 'fig1e_gt.png')

# 3. Bottom layer (R>B) view
r_view = gt_raw.copy().astype(float)
r_dom = R > B
r_view[~r_dom] = r_view[~r_dom] * 0.12
r_crop = r_view[y0:y1, x0:x1].astype(np.uint8)
Image.fromarray(r_crop).save(OUT / 'fig1e_bot.png')

# 4. Top layer (B>R) view
b_view = gt_raw.copy().astype(float)
b_dom = B > R
b_view[~b_dom] = b_view[~b_dom] * 0.12
b_crop = b_view[y0:y1, x0:x1].astype(np.uint8)
Image.fromarray(b_crop).save(OUT / 'fig1e_top.png')

# 5. Bridged skeleton (R+B combined)
dk3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
skel_vis = np.zeros((h, w, 3), dtype=np.uint8)
r_dil = cv2.dilate(r_bridged.astype(np.uint8), dk4) > 0
b_dil = cv2.dilate(b_bridged.astype(np.uint8), dk4) > 0

# Yellow detection for R skel
hsv = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2HSV)
hue = hsv[:,:,0].astype(float)
sat = hsv[:,:,1].astype(float)
y_dom = cv2.dilate(((hue >= 15) & (hue <= 50) & (sat > 40)).astype(np.uint8), dk4) > 0

r_only = r_dil & ~b_dil
b_only = b_dil & ~r_dil
both = r_dil & b_dil

skel_vis[r_only & ~y_dom] = [230, 70, 70]
skel_vis[r_only & y_dom] = [255, 210, 50]
skel_vis[b_only] = [70, 140, 240]
skel_vis[both] = [180, 100, 220]

skel_crop = skel_vis[y0:y1, x0:x1]
Image.fromarray(skel_crop).save(OUT / 'fig1e_skel.png')

print(f'Crop region: ({y0},{x0})-({y1},{x1})')
print('Saved all fig1e_*.png thumbnails')
