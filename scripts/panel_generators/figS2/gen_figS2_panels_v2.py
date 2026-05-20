"""Figure S2 Panel C: GT context + R/B before/bridging/after + combined + crops"""
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
h, w = gt_raw.shape[:2]; by = int(h * 0.8)
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
dk5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

R = gt_blur[:,:,0].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
diff = B - R

vessel_all = remove_small_objects(
    cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=100)
vessel_dilated = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

r_mask = remove_small_objects(
    cv2.morphologyEx((R > 30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=50)
r_skel = skeletonize(r_mask); r_filtered = r_skel & (diff < 0)

b_seed = B > 50; b_low = B > 20
b_ll, nl = ndimage.label(b_low)
b_hyst = np.zeros_like(b_low, dtype=bool)
for i in range(1, nl+1):
    reg = b_ll == i
    if np.any(b_seed[reg]): b_hyst |= reg
b_hyst = remove_small_objects(
    cv2.morphologyEx(b_hyst.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0, min_size=50)
b_skel = skeletonize(b_hyst); b_filtered = b_skel & (diff > 0)

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

def do_bridge_tracked(skel, vm):
    eps = get_eps(skel); ct = np.cos(np.radians(60)); cands = []
    for i in range(len(eps)):
        for j in range(i+1, len(eps)):
            p1=np.array(eps[i]['pos'],dtype=float); p2=np.array(eps[j]['pos'],dtype=float)
            d=np.linalg.norm(p2-p1)
            if d<5 or d>80: continue
            d12=(p2-p1)/d; c1v=np.dot(eps[i]['tangent'],d12); c2v=np.dot(eps[j]['tangent'],-d12)
            if c1v<ct or c2v<ct: continue
            pts=bezier(eps[i],eps[j])
            ov=sum(1 for(y,x)in pts if 0<=y<h and 0<=x<w and vm[y,x])
            vr=ov/max(len(pts),1)
            if vr<0.7: continue
            sc=(c1v+c2v)/2-d/80*0.2+vr*0.2
            cands.append((i,j,sc,eps[i],eps[j],pts))
    cands.sort(key=lambda x:-x[2]); used=set()
    bridges=[]; bridged=skel.copy()
    for i,j,sc,e1,e2,pts in cands:
        if i not in used and j not in used:
            bridges.append({'ep1':e1,'ep2':e2,'pts':pts})
            for(y,x) in pts:
                if 0<=y<h and 0<=x<w: bridged[y,x]=True
            used.add(i); used.add(j)
    bridged = skeletonize(bridged)
    return bridged, bridges

r_bridged, r_bridges = do_bridge_tracked(r_filtered, vessel_dilated)
b_bridged, b_bridges = do_bridge_tracked(b_filtered, vessel_dilated)

R_COL = [230,70,70]; B_COL = [70,140,240]
R_BR = [255,180,50]; B_BR = [50,255,180]
OVERLAP = [180,100,220]
EP_C = (255,255,0); JN_C = (255,0,255); CONN_C = (0,255,255)

def draw_bridge_paths(vis, bridges, color, thickness=2):
    for br in bridges:
        for (py, px) in br['pts']:
            if 0<=py<vis.shape[0] and 0<=px<vis.shape[1]:
                for dy in range(-thickness, thickness+1):
                    for dx in range(-thickness, thickness+1):
                        ny, nx = py+dy, px+dx
                        if 0<=ny<vis.shape[0] and 0<=nx<vis.shape[1]:
                            vis[ny, nx] = color

gt_crop = gt_raw[:by]

# Context images
Image.fromarray(gt_crop).save(OUT / 's2c_gt.png')
r_vis = np.zeros((by, w, 3), dtype=np.uint8)
r_vis[:,:,0] = gt_blur[:by,:,0]; r_vis[:,:,1] = (gt_blur[:by,:,0]*0.3).astype(np.uint8)
Image.fromarray(r_vis).save(OUT / 's2c_r_channel.png')
b_vis = np.zeros((by, w, 3), dtype=np.uint8)
b_vis[:,:,2] = gt_blur[:by,:,2]; b_vis[:,:,1] = (gt_blur[:by,:,2]*0.3).astype(np.uint8)
Image.fromarray(b_vis).save(OUT / 's2c_b_channel.png')

# Per-channel stages
for name, sk, bridges_list, color, br_color in [
    ('r', r_filtered, r_bridges, R_COL, R_BR),
    ('b', b_filtered, b_bridges, B_COL, B_BR)]:
    bridged_sk = r_bridged if name == 'r' else b_bridged

    # Before
    vis = np.zeros((by, w, 3), dtype=np.uint8)
    vis[cv2.dilate(sk[:by].astype(np.uint8), dk4) > 0] = color
    Image.fromarray(vis).save(OUT / f's2c_{name}_before.png')

    # Bridging
    vis2 = vis.copy()
    valid_br = [br for br in bridges_list if br['ep1']['pos'][0] < by and br['ep2']['pos'][0] < by]
    clipped = [{'pts': [(min(py,by-1),px) for py,px in br['pts'] if py < by]} for br in valid_br]
    draw_bridge_paths(vis2, clipped, br_color, thickness=2)
    for br in valid_br:
        for ep in [br['ep1'], br['ep2']]:
            ey, ex = ep['pos']
            if ey < by:
                cv2.circle(vis2, (int(ex), int(ey)), 5, EP_C, -1)
    Image.fromarray(vis2).save(OUT / f's2c_{name}_bridging.png')

    # After
    vis3 = np.zeros((by, w, 3), dtype=np.uint8)
    vis3[cv2.dilate(bridged_sk[:by].astype(np.uint8), dk4) > 0] = color
    Image.fromarray(vis3).save(OUT / f's2c_{name}_after.png')

# Combined with markers
r_dil = cv2.dilate(r_bridged[:by].astype(np.uint8), dk4) > 0
b_dil = cv2.dilate(b_bridged[:by].astype(np.uint8), dk4) > 0
vis_comb = np.zeros((by, w, 3), dtype=np.uint8)
vis_comb[r_dil & ~b_dil] = R_COL
vis_comb[b_dil & ~r_dil] = B_COL
vis_comb[r_dil & b_dil] = OVERLAP

def get_ep_jn(skel_crop):
    su = skel_crop.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    ep_mask = nb == 1; jn_mask = nb >= 3
    ep_labels, n_ep = ndimage.label(ep_mask)
    eps = []
    for i in range(1, n_ep+1):
        pts = np.argwhere(ep_labels == i)
        cy, cx = pts.mean(axis=0)
        y, x = int(round(cy)), int(round(cx))
        if y < by - 5: eps.append((y, x))
    jn_dil = cv2.dilate(jn_mask.astype(np.uint8), k3)
    jn_labels, n_raw = ndimage.label(jn_dil)
    jns = []
    for i in range(1, n_raw+1):
        cluster = jn_labels == i
        ys, xs = np.where(cluster)
        pad = 3
        ly0 = max(0, ys.min()-pad); ly1 = min(by, ys.max()+pad+1)
        lx0 = max(0, xs.min()-pad); lx1 = min(w, xs.max()+pad+1)
        local = skel_crop[ly0:ly1, lx0:lx1].copy()
        local[cluster[ly0:ly1, lx0:lx1]] = False
        _, nb2 = ndimage.label(local)
        if nb2 >= 3:
            pts = np.argwhere((jn_labels == i) & jn_mask)
            if len(pts) == 0: pts = np.argwhere(jn_labels == i)
            cy, cx = pts.mean(axis=0)
            jns.append((int(round(cy)), int(round(cx))))
    return eps, jns

r_ep, r_jn = get_ep_jn(r_bridged[:by])
b_ep, b_jn = get_ep_jn(b_bridged[:by])
r_dil_m = cv2.dilate(r_bridged[:by].astype(np.uint8), dk5) > 0
b_dil_m = cv2.dilate(b_bridged[:by].astype(np.uint8), dk5) > 0
r_real = [(y,x) for y,x in r_ep if not b_dil_m[y,x]]
r_conn = [(y,x) for y,x in r_ep if b_dil_m[y,x]]
b_real = [(y,x) for y,x in b_ep if not r_dil_m[y,x]]
b_conn = [(y,x) for y,x in b_ep if r_dil_m[y,x]]

for y,x in r_real+b_real: cv2.circle(vis_comb, (x,y), 7, EP_C, 2)
for y,x in r_conn+b_conn: cv2.circle(vis_comb, (x,y), 7, CONN_C, 2)
for y,x in r_jn+b_jn: cv2.circle(vis_comb, (x,y), 6, JN_C, 2)
Image.fromarray(vis_comb).save(OUT / 's2c_combined.png')

# Full GT overlay with white veil
BG_ALPHA = 0.7; WHITE_VEIL = 0.4
vis_full = gt_crop.copy().astype(float) * BG_ALPHA
vis_full = vis_full * (1 - WHITE_VEIL) + 255 * WHITE_VEIL
vis_full[r_dil & ~b_dil] = R_COL
vis_full[b_dil & ~r_dil] = B_COL
vis_full[r_dil & b_dil] = OVERLAP
for y,x in r_real+b_real: cv2.circle(vis_full, (x,y), 7, EP_C, 2)
for y,x in r_conn+b_conn: cv2.circle(vis_full, (x,y), 7, CONN_C, 2)
for y,x in r_jn+b_jn: cv2.circle(vis_full, (x,y), 6, JN_C, 2)
Image.fromarray(vis_full.astype(np.uint8)).save(OUT / 's2c_full_gt.png')

# Crop examples
crops = [
    (180, 120, 90, 'crop1'),
    (280, 400, 90, 'crop2'),
    (150, 340, 80, 'crop3'),
]
for cy, cx, pad, tag in crops:
    y0 = max(0, cy-pad); y1 = min(by, cy+pad)
    x0 = max(0, cx-pad); x1 = min(w, cx+pad)
    Image.fromarray(gt_crop[y0:y1, x0:x1]).save(OUT / f's2c_{tag}_gt.png')
    Image.fromarray(vis_comb[y0:y1, x0:x1]).save(OUT / f's2c_{tag}_skel.png')
    Image.fromarray(vis_full.astype(np.uint8)[y0:y1, x0:x1]).save(OUT / f's2c_{tag}_overlay.png')

print(f'R EP={len(r_real)}+{len(r_conn)} JN={len(r_jn)}')
print(f'B EP={len(b_real)}+{len(b_conn)} JN={len(b_jn)}')
print('All S2v2 panels saved!')
