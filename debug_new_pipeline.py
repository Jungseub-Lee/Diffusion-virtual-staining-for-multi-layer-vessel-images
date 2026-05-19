"""
New pipeline: skel WITHOUT dominance cutting → bridge → depth label tagging
Step-by-step visualization for 18-19-512
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage

DATA = Path(r'[1] Data')
OUT = Path(r'analysis_output/debug_pipeline')
OUT.mkdir(exist_ok=True)

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
dk2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

sid = '18-19-512'
gt = np.array(Image.open(DATA / 'Multi-color' / 'LSGAN' / f'{sid}_real_B.png'))
gt_blur = cv2.GaussianBlur(gt, (7,7), 0)
h, w = gt.shape[:2]; by = int(h*0.8)
R = gt_blur[:,:,0].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
diff = B - R

# ============================================================
# STEP 1: Masks (same as before)
# ============================================================
r_mask = remove_small_objects(
    cv2.morphologyEx((R>30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)
b_seed = B>50; b_low = B>20
b_lab, b_n = ndimage.label(b_low)
b_mask = np.zeros_like(b_low, dtype=bool)
for i in range(1, b_n+1):
    if np.any(b_seed[b_lab==i]): b_mask |= (b_lab==i)
b_mask = remove_small_objects(
    cv2.morphologyEx(b_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2)>0, min_size=50)

vessel_all = remove_small_objects(
    cv2.morphologyEx((gray>10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=100)
vm = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

# ============================================================
# STEP 2: Skeletonize WITHOUT dominance filter
# ============================================================
r_skel = skeletonize(r_mask)  # full R skeleton
b_skel = skeletonize(b_mask)  # full B skeleton

# OLD (for comparison): with dominance
r_skel_old = r_skel & (diff < 0)
b_skel_old = b_skel & (diff > 0)

# Count fragments
r_labels_new, r_n_new = ndimage.label(r_skel)
b_labels_new, b_n_new = ndimage.label(b_skel)
r_labels_old, r_n_old = ndimage.label(r_skel_old)
b_labels_old, b_n_old = ndimage.label(b_skel_old)

print(f"R skel: new(no dom)={r_n_new} components, old(dom)={r_n_old} components")
print(f"B skel: new(no dom)={b_n_new} components, old(dom)={b_n_old} components")
print(f"R skel pixels: new={np.sum(r_skel)}, old={np.sum(r_skel_old)}")
print(f"B skel pixels: new={np.sum(b_skel)}, old={np.sum(b_skel_old)}")

# ============================================================
# STEP 3: Bridge (on full skel, using channel mask for path check)
# ============================================================
def get_eps_tangent(skel, h, w, tl=20):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    eps = np.argwhere(nb == 1); res = []
    for ep in eps:
        y0, x0 = ep; path = [(y0,x0)]; vis = {(y0,x0)}; cy, cx = y0, x0
        for _ in range(tl):
            f = False
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    if dy==0 and dx==0: continue
                    ny, nx = cy+dy, cx+dx
                    if 0<=ny<h and 0<=nx<w and skel[ny,nx] and (ny,nx) not in vis:
                        path.append((ny,nx)); vis.add((ny,nx)); cy,cx=ny,nx; f=True; break
                if f: break
            if not f: break
        if len(path) >= 5:
            n2 = min(len(path), 10)
            t = np.array(path[0], dtype=float) - np.array(path[n2-1], dtype=float)
            nm = np.linalg.norm(t)
            if nm > 0: t /= nm
            res.append({'pos': (y0,x0), 'tangent': t, 'path': path})
    return res

def bezier(e1, e2):
    p1 = np.array(e1['pos'], dtype=float); p2 = np.array(e2['pos'], dtype=float)
    d = np.linalg.norm(p2-p1); t1, t2 = e1['tangent'], e2['tangent']
    c1 = p1 + t1*d*0.4; c2 = p2 + t2*d*0.4
    n = max(int(d*1.5), 10)
    return [(int(round(((1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2)[0])),
             int(round(((1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2)[1])))
            for t in np.linspace(0, 1, n)]

def simple_bridge(skel, ch_mask, vm, h, w, max_dist=60, max_angle=65):
    """Simple EP-EP bridge with channel mask check."""
    eps = get_eps_tangent(skel, h, w)
    ct = np.cos(np.radians(max_angle))
    cands = []
    for i in range(len(eps)):
        for j in range(i+1, len(eps)):
            p1 = np.array(eps[i]['pos'], dtype=float)
            p2 = np.array(eps[j]['pos'], dtype=float)
            d = np.linalg.norm(p2-p1)
            if d < 3 or d > max_dist: continue
            d12 = (p2-p1)/d
            c1 = np.dot(eps[i]['tangent'], d12)
            c2 = np.dot(eps[j]['tangent'], -d12)
            if c1 < ct or c2 < ct: continue
            pts = bezier(eps[i], eps[j])
            # Channel mask check (70%)
            ch_count = sum(1 for (y,x) in pts if 0<=y<h and 0<=x<w and ch_mask[y,x])
            if ch_count / max(len(pts),1) < 0.7: continue
            # Vessel mask check (70%)
            vm_count = sum(1 for (y,x) in pts if 0<=y<h and 0<=x<w and vm[y,x])
            if vm_count / max(len(pts),1) < 0.7: continue
            sc = (c1+c2)/2 - d/max_dist*0.2
            cands.append((i, j, sc, eps[i], eps[j], pts))
    cands.sort(key=lambda x: -x[2])
    used = set(); bridged = skel.copy()
    for i, j, sc, e1, e2, pts in cands:
        if i not in used and j not in used:
            for (y,x) in pts:
                if 0<=y<h and 0<=x<w: bridged[y,x] = True
            used.add(i); used.add(j)
    return skeletonize(bridged)

r_bridged = simple_bridge(r_skel, r_mask, vm, h, w)
b_bridged = simple_bridge(b_skel, b_mask, vm, h, w)

r_br_labels, r_br_n = ndimage.label(r_bridged)
b_br_labels, b_br_n = ndimage.label(b_bridged)
print(f"\nAfter bridge: R={r_br_n} components, B={b_br_n} components")

# ============================================================
# STEP 4: Depth label tagging (crossing region only)
# ============================================================
crossing = r_mask & b_mask  # where both masks overlap
crossing_dil = cv2.dilate(crossing.astype(np.uint8)*255, k5) > 0

# For each skel pixel, tag:
# R skel: 'R_dominant' where R>B, 'R_under_B' where B>R (in crossing)
# B skel: 'B_dominant' where B>R, 'B_under_R' where R>B (in crossing)

# R skel depth label
r_dominant = r_bridged & (diff <= 0)       # R>B: R vessel is dominant
r_under_b  = r_bridged & (diff > 0)        # B>R: R vessel passes under B
r_in_crossing = r_bridged & crossing_dil   # in crossing zone

# B skel depth label
b_dominant = b_bridged & (diff >= 0)       # B>R: B vessel is dominant
b_under_r  = b_bridged & (diff < 0)        # R>B: B vessel passes under R
b_in_crossing = b_bridged & crossing_dil

print(f"\nR skel: total={np.sum(r_bridged)} dominant={np.sum(r_dominant)} under_B={np.sum(r_under_b)} in_crossing={np.sum(r_in_crossing)}")
print(f"B skel: total={np.sum(b_bridged)} dominant={np.sum(b_dominant)} under_R={np.sum(b_under_r)} in_crossing={np.sum(b_in_crossing)}")

# ============================================================
# STEP 5: EP/JN counting
# ============================================================
def get_jn_ep(skel, h, by):
    roi = np.zeros_like(skel, dtype=bool); roi[:by] = True
    sc = skel & roi; su = sc.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    jm = nb >= 3; jlab, n_raw = ndimage.label(jm)
    jn_pts = []
    for i in range(1, n_raw+1):
        cluster = jlab == i; py, px = np.where(cluster)
        pad = 5
        ymin = max(0, py.min()-pad); ymax = min(h, py.max()+pad+1)
        xmin = max(0, px.min()-pad); xmax = min(skel.shape[1], px.max()+pad+1)
        local = sc[ymin:ymax, xmin:xmax].copy()
        cdil = cv2.dilate(cluster[ymin:ymax, xmin:xmax].astype(np.uint8), k3) > 0
        local[cdil] = False
        _, nbr = ndimage.label(local)
        if nbr >= 3: jn_pts.append((int(py.mean()), int(px.mean())))
    ep_pts = [(y,x) for y,x in np.argwhere((nb==1) & roi) if y < by-3]
    return jn_pts, ep_pts

r_jn, r_ep = get_jn_ep(r_bridged, h, by)
b_jn, b_ep = get_jn_ep(b_bridged, h, by)

# OLD pipeline for comparison
r_jn_old, r_ep_old = get_jn_ep(r_skel_old, h, by)
b_jn_old, b_ep_old = get_jn_ep(b_skel_old, h, by)

print(f"\nNew pipeline (no dom cut):")
print(f"  R: JN={len(r_jn)} EP={len(r_ep)} L={np.sum(r_bridged[:by])}")
print(f"  B: JN={len(b_jn)} EP={len(b_ep)} L={np.sum(b_bridged[:by])}")
print(f"Old pipeline (dom cut, no bridge):")
print(f"  R: JN={len(r_jn_old)} EP={len(r_ep_old)} L={np.sum(r_skel_old[:by])}")
print(f"  B: JN={len(b_jn_old)} EP={len(b_ep_old)} L={np.sum(b_skel_old[:by])}")

# ============================================================
# VISUALIZATION: 6 rows x 3 cols
# ============================================================
fig, axes = plt.subplots(6, 3, figsize=(16, 28))
fig.suptitle(f'{sid} — New Pipeline: No Dominance Cut → Bridge → Depth Tag', fontsize=14, fontweight='bold')

# Row 0: GT, R mask (binary), B mask (binary)
axes[0,0].imshow(gt[:by]); axes[0,0].set_title('GT (multi-color)', fontsize=10)
axes[0,1].imshow(r_mask[:by], cmap='gray'); axes[0,1].set_title(f'R mask (R>30) | {np.sum(r_mask[:by])}px', fontsize=10)
axes[0,2].imshow(b_mask[:by], cmap='gray'); axes[0,2].set_title(f'B mask (hyst 50/20) | {np.sum(b_mask[:by])}px', fontsize=10)

# Row 1: OLD skel (with dominance) vs NEW skel (no dominance)
# OLD R
vis = np.zeros((by,w,3), dtype=np.uint8)
vis[cv2.dilate(r_skel_old[:by].astype(np.uint8), dk2)>0] = [230,70,70]
axes[1,0].imshow(vis); axes[1,0].set_title(f'OLD: R skel & (R>B) | {r_n_old} comp, {np.sum(r_skel_old[:by])}px', fontsize=9)
# NEW R
vis = np.zeros((by,w,3), dtype=np.uint8)
vis[cv2.dilate(r_skel[:by].astype(np.uint8), dk2)>0] = [230,70,70]
axes[1,1].imshow(vis); axes[1,1].set_title(f'NEW: R skel (no dom) | {r_n_new} comp, {np.sum(r_skel[:by])}px', fontsize=9)
# NEW B
vis = np.zeros((by,w,3), dtype=np.uint8)
vis[cv2.dilate(b_skel[:by].astype(np.uint8), dk2)>0] = [70,140,240]
axes[1,2].imshow(vis); axes[1,2].set_title(f'NEW: B skel (no dom) | {b_n_new} comp, {np.sum(b_skel[:by])}px', fontsize=9)

# Row 2: After bridging
vis = np.zeros((by,w,3), dtype=np.uint8)
vis[cv2.dilate(r_bridged[:by].astype(np.uint8), dk2)>0] = [230,70,70]
axes[2,0].imshow(vis); axes[2,0].set_title(f'R bridged | {r_br_n} comp', fontsize=10)
vis = np.zeros((by,w,3), dtype=np.uint8)
vis[cv2.dilate(b_bridged[:by].astype(np.uint8), dk2)>0] = [70,140,240]
axes[2,1].imshow(vis); axes[2,1].set_title(f'B bridged | {b_br_n} comp', fontsize=10)
# Combined
vis = np.zeros((by,w,3), dtype=np.uint8)
r_d = cv2.dilate(r_bridged[:by].astype(np.uint8), dk2)>0
b_d = cv2.dilate(b_bridged[:by].astype(np.uint8), dk2)>0
vis[r_d & ~b_d] = [230,70,70]; vis[b_d & ~r_d] = [70,140,240]; vis[r_d & b_d] = [180,100,220]
axes[2,2].imshow(vis); axes[2,2].set_title('R+B combined (purple=overlap)', fontsize=10)

# Row 3: Depth tagging visualization
# R skel: bright red = R dominant, dark red = R under B
vis = np.zeros((by,w,3), dtype=np.uint8)
r_dom_dil = cv2.dilate(r_dominant[:by].astype(np.uint8), dk2)>0
r_und_dil = cv2.dilate(r_under_b[:by].astype(np.uint8), dk2)>0
vis[r_dom_dil] = [255, 80, 80]   # bright red = dominant
vis[r_und_dil] = [120, 40, 40]   # dark red = under B
axes[3,0].imshow(vis); axes[3,0].set_title(f'R depth: bright=dominant({np.sum(r_dominant[:by])}px) dark=underB({np.sum(r_under_b[:by])}px)', fontsize=8)

# B skel: bright blue = B dominant, dark blue = B under R
vis = np.zeros((by,w,3), dtype=np.uint8)
b_dom_dil = cv2.dilate(b_dominant[:by].astype(np.uint8), dk2)>0
b_und_dil = cv2.dilate(b_under_r[:by].astype(np.uint8), dk2)>0
vis[b_dom_dil] = [80, 150, 255]   # bright blue = dominant
vis[b_und_dil] = [40, 60, 120]    # dark blue = under R
axes[3,1].imshow(vis); axes[3,1].set_title(f'B depth: bright=dominant({np.sum(b_dominant[:by])}px) dark=underR({np.sum(b_under_r[:by])}px)', fontsize=8)

# Crossing region
vis = np.zeros((by,w,3), dtype=np.uint8)
vis[crossing[:by]] = [100, 100, 100]
vis[crossing_dil[:by] & ~crossing[:by]] = [50, 50, 50]
vis[r_dom_dil] = [255, 80, 80]; vis[r_und_dil] = [120, 40, 40]
vis[b_dom_dil] = [80, 150, 255]; vis[b_und_dil] = [40, 60, 120]
vis[r_dom_dil & b_dom_dil] = [180, 100, 220]
axes[3,2].imshow(vis); axes[3,2].set_title(f'Crossing zone (gray) + depth-tagged skels', fontsize=9)

# Row 4: EP/JN on new pipeline
vis = np.zeros((by,w,3), dtype=np.uint8)
vis[cv2.dilate(r_bridged[:by].astype(np.uint8), dk2)>0] = [230,70,70]
for y,x in r_jn: cv2.circle(vis,(x,y),4,(255,0,255),-1)
for y,x in r_ep: cv2.circle(vis,(x,y),3,(255,255,0),-1)
axes[4,0].imshow(vis); axes[4,0].set_title(f'NEW R: JN={len(r_jn)} EP={len(r_ep)}', fontsize=10)

vis = np.zeros((by,w,3), dtype=np.uint8)
vis[cv2.dilate(b_bridged[:by].astype(np.uint8), dk2)>0] = [70,140,240]
for y,x in b_jn: cv2.circle(vis,(x,y),4,(255,0,255),-1)
for y,x in b_ep: cv2.circle(vis,(x,y),3,(255,255,0),-1)
axes[4,1].imshow(vis); axes[4,1].set_title(f'NEW B: JN={len(b_jn)} EP={len(b_ep)}', fontsize=10)

# Combined on GT
vis = gt[:by].copy().astype(float) * 0.5
vis = vis * 0.6 + 255 * 0.4  # white veil
r_d4 = cv2.dilate(r_bridged[:by].astype(np.uint8), dk4)>0
b_d4 = cv2.dilate(b_bridged[:by].astype(np.uint8), dk4)>0
vis[r_d4 & ~b_d4] = [230,70,70]; vis[b_d4 & ~r_d4] = [70,140,240]; vis[r_d4 & b_d4] = [180,100,220]
vis = vis.astype(np.uint8)
for y,x in r_jn+b_jn: cv2.circle(vis,(x,y),5,(255,0,255),2)
for y,x in r_ep+b_ep: cv2.circle(vis,(x,y),5,(255,255,0),2)
axes[4,2].imshow(vis); axes[4,2].set_title(f'R+B on GT | JN={len(r_jn)+len(b_jn)} EP={len(r_ep)+len(b_ep)}', fontsize=10)

# Row 5: Comparison OLD vs NEW metrics
# OLD on GT
vis = gt[:by].copy().astype(float) * 0.5
vis = vis * 0.6 + 255 * 0.4
r_d_old = cv2.dilate(r_skel_old[:by].astype(np.uint8), dk4)>0
b_d_old = cv2.dilate(b_skel_old[:by].astype(np.uint8), dk4)>0
vis[r_d_old & ~b_d_old] = [230,70,70]; vis[b_d_old & ~r_d_old] = [70,140,240]; vis[r_d_old & b_d_old] = [180,100,220]
vis = vis.astype(np.uint8)
for y,x in r_jn_old+b_jn_old: cv2.circle(vis,(x,y),5,(255,0,255),2)
for y,x in r_ep_old+b_ep_old: cv2.circle(vis,(x,y),5,(255,255,0),2)
axes[5,0].imshow(vis)
axes[5,0].set_title(f'OLD (dom cut): R JN={len(r_jn_old)} EP={len(r_ep_old)} | B JN={len(b_jn_old)} EP={len(b_ep_old)}', fontsize=8)

# NEW on GT (same as row 4 col 2)
vis = gt[:by].copy().astype(float) * 0.5
vis = vis * 0.6 + 255 * 0.4
vis[r_d4 & ~b_d4] = [230,70,70]; vis[b_d4 & ~r_d4] = [70,140,240]; vis[r_d4 & b_d4] = [180,100,220]
vis = vis.astype(np.uint8)
for y,x in r_jn+b_jn: cv2.circle(vis,(x,y),5,(255,0,255),2)
for y,x in r_ep+b_ep: cv2.circle(vis,(x,y),5,(255,255,0),2)
axes[5,1].imshow(vis)
axes[5,1].set_title(f'NEW (no cut): R JN={len(r_jn)} EP={len(r_ep)} | B JN={len(b_jn)} EP={len(b_ep)}', fontsize=8)

# Summary text
axes[5,2].axis('off')
axes[5,2].set_facecolor('white')
txt = (f"Comparison ({sid})\n\n"
       f"OLD (dominance cut + no bridge):\n"
       f"  R: {r_n_old} comp, JN={len(r_jn_old)}, EP={len(r_ep_old)}, L={np.sum(r_skel_old[:by])}\n"
       f"  B: {b_n_old} comp, JN={len(b_jn_old)}, EP={len(b_ep_old)}, L={np.sum(b_skel_old[:by])}\n\n"
       f"NEW (no cut + bridge + depth tag):\n"
       f"  R: {r_br_n} comp, JN={len(r_jn)}, EP={len(r_ep)}, L={np.sum(r_bridged[:by])}\n"
       f"  B: {b_br_n} comp, JN={len(b_jn)}, EP={len(b_ep)}, L={np.sum(b_bridged[:by])}\n\n"
       f"Depth tag (R skel):\n"
       f"  R-dominant: {np.sum(r_dominant[:by])}px\n"
       f"  R-under-B: {np.sum(r_under_b[:by])}px\n\n"
       f"Depth tag (B skel):\n"
       f"  B-dominant: {np.sum(b_dominant[:by])}px\n"
       f"  B-under-R: {np.sum(b_under_r[:by])}px")
axes[5,2].text(0.05, 0.95, txt, fontsize=9, fontfamily='monospace', va='top',
               transform=axes[5,2].transAxes)

for ax in axes.flat: ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / f'new_pipeline_{sid}.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'\nSaved new_pipeline_{sid}.png')
print('Done!')
