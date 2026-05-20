"""
Regenerate j, k, l panels with strict JN detection + EP/JN markers on combined.
Also regenerate i (single GT) for consistency.
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
gt_raw = s1[512:]
gt_blur = cv2.GaussianBlur(gt_raw, (7, 7), 0)
h, w = gt_raw.shape[:2]
by = int(h * 0.8)

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

R = gt_blur[:,:,0].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
diff = B - R

# === Masks & Skeletons ===
vessel_all = remove_small_objects(
    cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=100)
vessel_dilated = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0
base_skel = skeletonize(vessel_all)

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

# === Bridging ===
def get_eps(skel, tl=20):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    eps = np.argwhere(nb == 1)
    res = []
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
    eps = get_eps(skel)
    ct = np.cos(np.radians(60)); cands = []
    for i in range(len(eps)):
        for j in range(i+1,len(eps)):
            p1=np.array(eps[i]['pos'],dtype=float); p2=np.array(eps[j]['pos'],dtype=float)
            d=np.linalg.norm(p2-p1)
            if d<5 or d>80: continue
            d12=(p2-p1)/d; c1=np.dot(eps[i]['tangent'],d12); c2=np.dot(eps[j]['tangent'],-d12)
            if c1<ct or c2<ct: continue
            pts=bezier(eps[i],eps[j])
            ov=sum(1 for(y,x)in pts if 0<=y<h and 0<=x<w and vm[y,x])
            vr=ov/max(len(pts),1)
            if vr<0.7: continue
            sc=(c1+c2)/2-d/80*0.2+vr*0.2
            cands.append((i,j,sc,eps[i],eps[j]))
    cands.sort(key=lambda x:-x[2]); used=set(); br=skel.copy()
    for i,j,sc,e1,e2 in cands:
        if i not in used and j not in used:
            for(y,x)in bezier(e1,e2):
                if 0<=y<h and 0<=x<w: br[y,x]=True
            used.add(i); used.add(j)
    return skeletonize(br)

r_bridged = do_bridge(r_filtered, vessel_dilated)
b_bridged = do_bridge(b_filtered, vessel_dilated)

# === Strict EP/JN detection ===
def get_strict_ep_jn(skel, region_mask, boundary_y, boundary_margin=5):
    s = skel & region_mask
    su = s.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    length = int(np.sum(s))

    # EP: cluster
    ep_mask = nb == 1
    ep_labels, n_ep = ndimage.label(ep_mask)
    ep_list = []
    for i in range(1, n_ep+1):
        pts = np.argwhere(ep_labels == i)
        cy, cx = pts.mean(axis=0)
        y, x = int(round(cy)), int(round(cx))
        if y < boundary_y - boundary_margin:
            ep_list.append((y, x))

    # JN: cluster + verify >=3 branches
    jn_mask = nb >= 3
    jn_dil = cv2.dilate(jn_mask.astype(np.uint8), k3)
    jn_labels, n_raw = ndimage.label(jn_dil)
    jn_list = []
    for i in range(1, n_raw+1):
        cluster = jn_labels == i
        ys, xs = np.where(cluster)
        pad = 3
        ly0 = max(0, ys.min()-pad); ly1 = min(h, ys.max()+pad+1)
        lx0 = max(0, xs.min()-pad); lx1 = min(w, xs.max()+pad+1)
        local = s[ly0:ly1, lx0:lx1].copy()
        local[cluster[ly0:ly1, lx0:lx1]] = False
        _, nb2 = ndimage.label(local)
        if nb2 >= 3:
            pts = np.argwhere((jn_labels == i) & jn_mask)
            if len(pts) == 0: pts = np.argwhere(jn_labels == i)
            cy, cx = pts.mean(axis=0)
            jn_list.append((int(round(cy)), int(round(cx))))

    return ep_list, jn_list, length

region = np.zeros((h, w), dtype=bool); region[:by] = True

single_ep, single_jn, single_L = get_strict_ep_jn(base_skel, region, by)
r_ep, r_jn, r_L = get_strict_ep_jn(r_bridged, region, by)
b_ep, b_jn, b_L = get_strict_ep_jn(b_bridged, region, by)

# Classify EP
dk5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
r_dil_mask = cv2.dilate(r_bridged.astype(np.uint8), dk5) > 0
b_dil_mask = cv2.dilate(b_bridged.astype(np.uint8), dk5) > 0

def classify_ep(ep_list, other_dil):
    real, conn = [], []
    for (y, x) in ep_list:
        if other_dil[y, x]: conn.append((y, x))
        else: real.append((y, x))
    return real, conn

r_real, r_conn = classify_ep(r_ep, b_dil_mask)
b_real, b_conn = classify_ep(b_ep, r_dil_mask)

print(f'Single GT: L={single_L} EP={len(single_ep)} JN={len(single_jn)}')
print(f'R bridged: L={r_L} EP={len(r_ep)}(real={len(r_real)} conn={len(r_conn)}) JN={len(r_jn)}')
print(f'B bridged: L={b_L} EP={len(b_ep)}(real={len(b_real)} conn={len(b_conn)}) JN={len(b_jn)}')
print(f'R+B:      L={r_L+b_L} EP(real)={len(r_real)+len(b_real)} JN={len(r_jn)+len(b_jn)}')

# === Yellow dominance detection for R skel ===
# In GT, yellow = middle depth. Where Y channel (G in RGB context + high R, low B) dominates
# Use HSV hue: yellow ≈ hue 15-50 in OpenCV (0-180 scale)
hsv = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2HSV)
hue = hsv[:,:,0].astype(float)  # 0-180
sat = hsv[:,:,1].astype(float)  # 0-255
# Yellow dominant: hue in 15-50, saturation decent
yellow_dominant = (hue >= 15) & (hue <= 50) & (sat > 40)

# === Generate panels ===
BG_ALPHA = 0.7  # higher = GT more visible
R_COLOR = [255, 80, 80]
Y_COLOR = [255, 220, 50]     # yellow for Y-dominant R skel pixels
B_COLOR = [80, 150, 255]
EP_COLOR = (255, 255, 0)     # yellow
CONN_COLOR = (0, 255, 255)   # cyan
JN_COLOR = (255, 0, 255)     # magenta
OVERLAP_COLOR = [180, 100, 220]

def draw_markers(vis, ep_real=None, ep_conn=None, jn=None, r=7, t=2):
    if ep_real:
        for (y, x) in ep_real:
            cv2.circle(vis, (x, y), r, EP_COLOR, t)
    if ep_conn:
        for (y, x) in ep_conn:
            cv2.circle(vis, (x, y), r, CONN_COLOR, t)
    if jn:
        for (y, x) in jn:
            cv2.circle(vis, (x, y), r-1, JN_COLOR, t)
    return vis

# All panels: crop to top 80% (baseline area removed)
y_dom_dil = cv2.dilate(yellow_dominant.astype(np.uint8), dk4) > 0

# White veil between GT and skel: blend GT toward white slightly
WHITE_VEIL = 0.45  # 0=no white, 1=full white

def make_base(gt_crop):
    """GT * BG_ALPHA, then blend toward white by WHITE_VEIL"""
    vis = gt_crop.copy().astype(float) * BG_ALPHA
    vis = vis * (1 - WHITE_VEIL) + 255 * WHITE_VEIL
    return vis

# (i) Single GT skel
vis = make_base(gt_raw[:by])
skel_dil = cv2.dilate(base_skel[:by].astype(np.uint8), dk4) > 0
vis[skel_dil] = np.array([0, 200, 0])
draw_markers(vis, ep_real=single_ep, jn=single_jn)
Image.fromarray(vis.astype(np.uint8)).save(OUT / 'i_single.png')
print('Saved i_single.png')

# (j) R bridged — yellow where Y-dominant
vis = make_base(gt_raw[:by])
r_dil_skel = cv2.dilate(r_bridged[:by].astype(np.uint8), dk4) > 0
r_is_yellow = r_dil_skel & y_dom_dil[:by]
r_is_red = r_dil_skel & ~y_dom_dil[:by]
vis[r_is_red] = R_COLOR
vis[r_is_yellow] = Y_COLOR
draw_markers(vis, ep_real=r_real, ep_conn=r_conn, jn=r_jn)
Image.fromarray(vis.astype(np.uint8)).save(OUT / 'j_r_br.png')
print('Saved j_r_br.png')

# (k) B bridged
vis = make_base(gt_raw[:by])
b_dil_sk = cv2.dilate(b_bridged[:by].astype(np.uint8), dk4) > 0
vis[b_dil_sk] = B_COLOR
draw_markers(vis, ep_real=b_real, ep_conn=b_conn, jn=b_jn)
Image.fromarray(vis.astype(np.uint8)).save(OUT / 'k_b_br.png')
print('Saved k_b_br.png')

# (l) R+B combined with overlap transparency + EP/JN markers
vis = make_base(gt_raw[:by])
r_dil = cv2.dilate(r_bridged[:by].astype(np.uint8), dk4) > 0
b_dil = cv2.dilate(b_bridged[:by].astype(np.uint8), dk4) > 0
r_only = r_dil & ~b_dil
b_only = b_dil & ~r_dil
both = r_dil & b_dil
# R skel: split yellow/red
r_only_yellow = r_only & y_dom_dil[:by]
r_only_red = r_only & ~y_dom_dil[:by]
vis[r_only_red] = R_COLOR
vis[r_only_yellow] = Y_COLOR
vis[b_only] = B_COLOR
vis[both] = OVERLAP_COLOR
# Markers: all real EP + all JN
all_real = r_real + b_real
all_conn = r_conn + b_conn
all_jn = r_jn + b_jn
draw_markers(vis, ep_real=all_real, ep_conn=all_conn, jn=all_jn)
Image.fromarray(vis.astype(np.uint8)).save(OUT / 'l_comb.png')
print('Saved l_comb.png')

# === No-GT background versions (black background) ===
def make_base_black(img):
    return np.zeros((img.shape[0], img.shape[1], 3), dtype=float)

# (i_nb) Single GT skel - no BG
vis = make_base_black(gt_raw[:by])
vis[skel_dil] = np.array([0, 200, 0])
draw_markers(vis, ep_real=single_ep, jn=single_jn)
Image.fromarray(vis.astype(np.uint8)).save(OUT / 'i_single_nb.png')
print('Saved i_single_nb.png')

# (j_nb) R bridged - no BG
vis = make_base_black(gt_raw[:by])
r_dil_skel = cv2.dilate(r_bridged[:by].astype(np.uint8), dk4) > 0
r_is_yellow = r_dil_skel & y_dom_dil[:by]
r_is_red = r_dil_skel & ~y_dom_dil[:by]
vis[r_is_red] = R_COLOR
vis[r_is_yellow] = Y_COLOR
draw_markers(vis, ep_real=r_real, ep_conn=r_conn, jn=r_jn)
Image.fromarray(vis.astype(np.uint8)).save(OUT / 'j_r_br_nb.png')
print('Saved j_r_br_nb.png')

# (k_nb) B bridged - no BG
vis = make_base_black(gt_raw[:by])
b_dil_sk = cv2.dilate(b_bridged[:by].astype(np.uint8), dk4) > 0
vis[b_dil_sk] = B_COLOR
draw_markers(vis, ep_real=b_real, ep_conn=b_conn, jn=b_jn)
Image.fromarray(vis.astype(np.uint8)).save(OUT / 'k_b_br_nb.png')
print('Saved k_b_br_nb.png')

# (l_nb) R+B combined - no BG
vis = make_base_black(gt_raw[:by])
vis[r_only_red] = R_COLOR
vis[r_only_yellow] = Y_COLOR
vis[b_only] = B_COLOR
vis[both] = OVERLAP_COLOR
draw_markers(vis, ep_real=all_real, ep_conn=all_conn, jn=all_jn)
Image.fromarray(vis.astype(np.uint8)).save(OUT / 'l_comb_nb.png')
print('Saved l_comb_nb.png')

print('\nDone!')
