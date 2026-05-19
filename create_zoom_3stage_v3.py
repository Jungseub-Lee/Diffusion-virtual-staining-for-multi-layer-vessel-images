"""
(n) Bridge zoom panels: Before → Bridging → After
- Uses gen_bridge_panels.py style crop regions (per-bridge, zoom_pad=70)
- Before/After have NO text/arrows (text added in PPT)
- Bridging shows bridge path with color-coded lines
- R+B overlap with transparency
- Selects top 3 R + top 3 B bridges by distance (well-separated)
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

# === Bridge detection ===
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

def do_bridge_with_info(skel, vm):
    eps = get_eps(skel); ct = np.cos(np.radians(60)); cands = []
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
            cands.append((i,j,sc,d,eps[i],eps[j],pts))
    cands.sort(key=lambda x:-x[2]); used=set(); bridges=[]
    bridged = skel.copy()
    for i,j,sc,d,e1,e2,pts in cands:
        if i not in used and j not in used:
            bridges.append({'pts':pts,'ep1':e1,'ep2':e2,'score':sc,'dist':d})
            for(y,x) in pts:
                if 0<=y<h and 0<=x<w: bridged[y,x]=True
            used.add(i); used.add(j)
    bridged = skeletonize(bridged)
    return bridged, bridges

r_bridged, r_bridges = do_bridge_with_info(r_filtered, vessel_dilated)
b_bridged, b_bridges = do_bridge_with_info(b_filtered, vessel_dilated)

# === Visualization ===
BG_ALPHA = 0.5
R_COLOR = np.array([255, 80, 80], dtype=float)
B_COLOR = np.array([80, 150, 255], dtype=float)
R_BRIDGE_COLOR = [255, 180, 50]   # orange for R bridges
B_BRIDGE_COLOR = [50, 255, 180]   # teal for B bridges
OVERLAP_BLEND = 0.5

def draw_rb_overlap(bg, r_dil, b_dil):
    vis = bg.copy()
    r_only = r_dil & ~b_dil
    b_only = b_dil & ~r_dil
    both = r_dil & b_dil
    vis[r_only] = R_COLOR
    vis[b_only] = B_COLOR
    vis[both] = (R_COLOR * OVERLAP_BLEND + B_COLOR * (1-OVERLAP_BLEND))
    return vis

# Collect bridges sorted by distance
all_br = []
for b in r_bridges:
    cy = int((b['ep1']['pos'][0]+b['ep2']['pos'][0])/2)
    cx = int((b['ep1']['pos'][1]+b['ep2']['pos'][1])/2)
    if cy < by:
        all_br.append(('R', b, cy, cx))
for b in b_bridges:
    cy = int((b['ep1']['pos'][0]+b['ep2']['pos'][0])/2)
    cx = int((b['ep1']['pos'][1]+b['ep2']['pos'][1])/2)
    if cy < by:
        all_br.append(('B', b, cy, cx))

all_br.sort(key=lambda x: -x[1]['dist'])

# Pick 3 well-separated crops
zoom_pad = 70
selected = []
for ch, br, cy, cx in all_br:
    too_close = False
    for s in selected:
        if abs(cy - s[2]) < zoom_pad and abs(cx - s[3]) < zoom_pad:
            too_close = True; break
    if not too_close:
        selected.append((ch, br, cy, cx))
        if len(selected) >= 3:
            break

print(f"Selected {len(selected)} crop regions:")

# Also identify all bridges within each crop (so Bridging panel shows all)
def bridges_in_crop(bridges, y0, y1, x0, x1, margin=15):
    result = []
    for b in bridges:
        for ep in [b['ep1'], b['ep2']]:
            ey, ex = ep['pos']
            if y0-margin <= ey < y1+margin and x0-margin <= ex < x1+margin:
                result.append(b); break
    return result

for idx, (ch, br, cy, cx) in enumerate(selected):
    y0 = max(0, cy - zoom_pad); y1 = min(h, cy + zoom_pad)
    x0 = max(0, cx - zoom_pad); x1 = min(w, cx + zoom_pad)
    print(f"  Crop {idx+1}: {ch} bridge dist={br['dist']:.0f} region=({y0},{x0})-({y1},{x1})")

    bg = gt_raw[y0:y1, x0:x1].copy().astype(float) * BG_ALPHA

    # Local dilated skeletons
    r_filt_dil = cv2.dilate(r_filtered[y0:y1, x0:x1].astype(np.uint8), dk4) > 0
    b_filt_dil = cv2.dilate(b_filtered[y0:y1, x0:x1].astype(np.uint8), dk4) > 0
    r_br_dil = cv2.dilate(r_bridged[y0:y1, x0:x1].astype(np.uint8), dk4) > 0
    b_br_dil = cv2.dilate(b_bridged[y0:y1, x0:x1].astype(np.uint8), dk4) > 0

    # --- BEFORE: filtered skeletons (pre-bridge), no text/arrows ---
    before = draw_rb_overlap(bg.copy(), r_filt_dil, b_filt_dil)
    Image.fromarray(before.astype(np.uint8)).save(OUT / f'n_zoom_{idx+1}_before.png')

    # --- BRIDGING: filtered + ALL bridge paths in this crop ---
    bridging = draw_rb_overlap(bg.copy(), r_filt_dil, b_filt_dil)
    r_local = bridges_in_crop(r_bridges, y0, y1, x0, x1)
    b_local = bridges_in_crop(b_bridges, y0, y1, x0, x1)
    # Draw bridge paths (thick, color-coded)
    for b in r_local:
        for (py, px) in b['pts']:
            ly, lx = py - y0, px - x0
            if 0 <= ly < bridging.shape[0] and 0 <= lx < bridging.shape[1]:
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = ly+dy, lx+dx
                        if 0 <= ny < bridging.shape[0] and 0 <= nx < bridging.shape[1]:
                            bridging[ny, nx] = R_BRIDGE_COLOR
    for b in b_local:
        for (py, px) in b['pts']:
            ly, lx = py - y0, px - x0
            if 0 <= ly < bridging.shape[0] and 0 <= lx < bridging.shape[1]:
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = ly+dy, lx+dx
                        if 0 <= ny < bridging.shape[0] and 0 <= nx < bridging.shape[1]:
                            bridging[ny, nx] = B_BRIDGE_COLOR
    # Draw endpoint dots
    for b in r_local + b_local:
        for ep in [b['ep1'], b['ep2']]:
            ey, ex = ep['pos'][0]-y0, ep['pos'][1]-x0
            if 0 <= ey < bridging.shape[0] and 0 <= ex < bridging.shape[1]:
                cv2.circle(bridging.astype(np.uint8), (int(ex), int(ey)), 4, (255,255,0), -1)
    Image.fromarray(bridging.astype(np.uint8)).save(OUT / f'n_zoom_{idx+1}_bridging.png')

    # --- AFTER: bridged clean skeletons ---
    after = draw_rb_overlap(bg.copy(), r_br_dil, b_br_dil)
    Image.fromarray(after.astype(np.uint8)).save(OUT / f'n_zoom_{idx+1}_after.png')

    print(f"    R bridges in crop: {len(r_local)}, B bridges: {len(b_local)}")

print("\nDone!")
