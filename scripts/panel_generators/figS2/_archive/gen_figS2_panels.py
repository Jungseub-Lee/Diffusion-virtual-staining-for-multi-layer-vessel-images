"""
Figure S2: Tangent-guided Gap Bridging Algorithm
Panel C: R skel before/bridging/after, B skel before/bridging/after, Combined, Full image versions
- No yellow — pure Red/Blue only for clarity
- Bridging paths shown in distinct colors (orange for R bridge, teal for B bridge)
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
dk3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

R = gt_blur[:,:,0].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
diff = B - R

# === Build masks and filtered skeletons ===
vessel_all = remove_small_objects(
    cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=100)
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

# === Bridging with path tracking ===
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

def do_bridge_tracked(skel, vm):
    eps = get_eps(skel); ct = np.cos(np.radians(60)); cands = []
    for i in range(len(eps)):
        for j in range(i+1, len(eps)):
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
            cands.append((i,j,sc,eps[i],eps[j],pts))
    cands.sort(key=lambda x:-x[2]); used=set()
    bridges = []; bridged = skel.copy()
    for i,j,sc,e1,e2,pts in cands:
        if i not in used and j not in used:
            bridges.append({'ep1':e1,'ep2':e2,'pts':pts,'score':sc})
            for(y,x) in pts:
                if 0<=y<h and 0<=x<w: bridged[y,x]=True
            used.add(i); used.add(j)
    bridged = skeletonize(bridged)
    return bridged, bridges

r_bridged, r_bridges = do_bridge_tracked(r_filtered, vessel_dilated)
b_bridged, b_bridges = do_bridge_tracked(b_filtered, vessel_dilated)

print(f'R bridges: {len(r_bridges)}, B bridges: {len(b_bridges)}')

# === Colors ===
R_COLOR = [230, 70, 70]
B_COLOR = [70, 140, 240]
R_BRIDGE_COLOR = [255, 180, 50]   # orange
B_BRIDGE_COLOR = [50, 255, 180]   # teal
OVERLAP_COLOR = [180, 100, 220]
EP_COLOR = (255, 255, 0)
JN_COLOR = (255, 0, 255)
CONN_COLOR = (0, 255, 255)

def draw_skel(vis, skel_mask, color, dk=dk4):
    dil = cv2.dilate(skel_mask.astype(np.uint8), dk) > 0
    vis[dil] = color
    return vis, dil

def draw_bridge_paths(vis, bridges, color, thickness=2):
    for br in bridges:
        for (py, px) in br['pts']:
            if 0<=py<vis.shape[0] and 0<=px<vis.shape[1]:
                for dy in range(-thickness, thickness+1):
                    for dx in range(-thickness, thickness+1):
                        ny, nx = py+dy, px+dx
                        if 0<=ny<vis.shape[0] and 0<=nx<vis.shape[1]:
                            vis[ny, nx] = color
    return vis

def draw_ep_markers(vis, bridges, radius=5):
    for br in bridges:
        for ep in [br['ep1'], br['ep2']]:
            ey, ex = ep['pos']
            if 0<=ey<vis.shape[0] and 0<=ex<vis.shape[1]:
                cv2.circle(vis, (int(ex), int(ey)), radius, EP_COLOR, 2)
    return vis

# === Get strict EP/JN ===
def get_strict_ep_jn(skel, region_mask, boundary_y, margin=5):
    s = skel & region_mask
    su = s.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su

    ep_mask = nb == 1
    ep_labels, n_ep = ndimage.label(ep_mask)
    ep_list = []
    for i in range(1, n_ep+1):
        pts = np.argwhere(ep_labels == i)
        cy, cx = pts.mean(axis=0)
        y, x = int(round(cy)), int(round(cx))
        if y < boundary_y - margin:
            ep_list.append((y, x))

    jn_mask = nb >= 3
    jn_dil = cv2.dilate(jn_mask.astype(np.uint8), k3)
    jn_labels, n_raw = ndimage.label(jn_dil)
    jn_list = []
    sh = s.shape[0]
    sw = s.shape[1] if len(s.shape) > 1 else 1
    for i in range(1, n_raw+1):
        cluster = jn_labels == i
        ys, xs = np.where(cluster)
        pad = 3
        ly0 = max(0, ys.min()-pad); ly1 = min(sh, ys.max()+pad+1)
        lx0 = max(0, xs.min()-pad); lx1 = min(sw, xs.max()+pad+1)
        local = s[ly0:ly1, lx0:lx1].copy()
        local[cluster[ly0:ly1, lx0:lx1]] = False
        _, nb2 = ndimage.label(local)
        if nb2 >= 3:
            pts = np.argwhere((jn_labels == i) & jn_mask)
            if len(pts) == 0: pts = np.argwhere(jn_labels == i)
            cy, cx = pts.mean(axis=0)
            jn_list.append((int(round(cy)), int(round(cx))))
    return ep_list, jn_list

region = np.zeros((by, w), dtype=bool); region[:] = True

# =============================================
# PANEL C: Step-by-step bridging visualization
# =============================================

# --- R SKEL: before / bridging / after ---
# Before
vis_r_before = np.zeros((by, w, 3), dtype=np.uint8)
draw_skel(vis_r_before, r_filtered[:by], R_COLOR)
Image.fromarray(vis_r_before).save(OUT / 's2_r_before.png')

# Bridging (show paths)
vis_r_bridging = vis_r_before.copy()
draw_bridge_paths(vis_r_bridging, [{'pts':[(min(py,by-1),px) for py,px in br['pts'] if py<by]} for br in r_bridges], R_BRIDGE_COLOR)
draw_ep_markers(vis_r_bridging, [br for br in r_bridges if br['ep1']['pos'][0]<by])
Image.fromarray(vis_r_bridging).save(OUT / 's2_r_bridging.png')

# After
vis_r_after = np.zeros((by, w, 3), dtype=np.uint8)
draw_skel(vis_r_after, r_bridged[:by], R_COLOR)
Image.fromarray(vis_r_after).save(OUT / 's2_r_after.png')

# --- B SKEL: before / bridging / after ---
vis_b_before = np.zeros((by, w, 3), dtype=np.uint8)
draw_skel(vis_b_before, b_filtered[:by], B_COLOR)
Image.fromarray(vis_b_before).save(OUT / 's2_b_before.png')

vis_b_bridging = vis_b_before.copy()
draw_bridge_paths(vis_b_bridging, [{'pts':[(min(py,by-1),px) for py,px in br['pts'] if py<by]} for br in b_bridges], B_BRIDGE_COLOR)
draw_ep_markers(vis_b_bridging, [br for br in b_bridges if br['ep1']['pos'][0]<by])
Image.fromarray(vis_b_bridging).save(OUT / 's2_b_bridging.png')

vis_b_after = np.zeros((by, w, 3), dtype=np.uint8)
draw_skel(vis_b_after, b_bridged[:by], B_COLOR)
Image.fromarray(vis_b_after).save(OUT / 's2_b_after.png')

# --- COMBINED: R+B after bridging with overlap ---
vis_comb = np.zeros((by, w, 3), dtype=np.uint8)
r_dil = cv2.dilate(r_bridged[:by].astype(np.uint8), dk4) > 0
b_dil = cv2.dilate(b_bridged[:by].astype(np.uint8), dk4) > 0
r_only = r_dil & ~b_dil
b_only = b_dil & ~r_dil
both = r_dil & b_dil
vis_comb[r_only] = R_COLOR
vis_comb[b_only] = B_COLOR
vis_comb[both] = OVERLAP_COLOR

# EP/JN markers on combined
r_ep, r_jn = get_strict_ep_jn(r_bridged[:by], region, by)
b_ep, b_jn = get_strict_ep_jn(b_bridged[:by], region, by)

dk5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
r_dil_mask = cv2.dilate(r_bridged[:by].astype(np.uint8), dk5) > 0
b_dil_mask = cv2.dilate(b_bridged[:by].astype(np.uint8), dk5) > 0

r_real = [(y,x) for y,x in r_ep if not b_dil_mask[y,x]]
r_conn = [(y,x) for y,x in r_ep if b_dil_mask[y,x]]
b_real = [(y,x) for y,x in b_ep if not r_dil_mask[y,x]]
b_conn = [(y,x) for y,x in b_ep if r_dil_mask[y,x]]

all_real = r_real + b_real
all_conn = r_conn + b_conn
all_jn = r_jn + b_jn

for (y,x) in all_real:
    cv2.circle(vis_comb, (x,y), 7, EP_COLOR, 2)
for (y,x) in all_conn:
    cv2.circle(vis_comb, (x,y), 7, CONN_COLOR, 2)
for (y,x) in all_jn:
    cv2.circle(vis_comb, (x,y), 6, JN_COLOR, 2)

Image.fromarray(vis_comb).save(OUT / 's2_combined.png')

# --- COMBINED on GT overlay ---
BG_ALPHA = 0.35
vis_comb_gt = gt_raw[:by].copy().astype(float) * BG_ALPHA
vis_comb_gt[r_only] = R_COLOR
vis_comb_gt[b_only] = B_COLOR
vis_comb_gt[both] = OVERLAP_COLOR
for (y,x) in all_real:
    cv2.circle(vis_comb_gt, (x,y), 7, EP_COLOR, 2)
for (y,x) in all_conn:
    cv2.circle(vis_comb_gt, (x,y), 7, CONN_COLOR, 2)
for (y,x) in all_jn:
    cv2.circle(vis_comb_gt, (x,y), 6, JN_COLOR, 2)
Image.fromarray(vis_comb_gt.astype(np.uint8)).save(OUT / 's2_combined_gt.png')

print(f'R EP: {len(r_real)} real + {len(r_conn)} conn, JN: {len(r_jn)}')
print(f'B EP: {len(b_real)} real + {len(b_conn)} conn, JN: {len(b_jn)}')
print(f'Combined: EP(real)={len(all_real)}, JN={len(all_jn)}')
print('\nAll S2 panels saved!')
