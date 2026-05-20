"""
New pipeline v2 on Sample 1 (512x512) — full visualization.
R>45 | B ratio>0.55 | No dom cut | Depth transition through yellow.
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

# Load Sample 1 (512x512)
s1 = np.array(Image.open(DATA / 'Original color stack data' / 'Sample 1_BF and GT(512x512 two images).png'))
gt = s1[512:]  # bottom half = GT
gt_blur = cv2.GaussianBlur(gt, (7,7), 0)
h, w = gt.shape[:2]
by = int(h * 0.8)
print(f'Sample 1: {h}x{w}, analysis zone: top {by}px')

R = gt_blur[:,:,0].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
hsv = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2HSV)
hue = hsv[:,:,0].astype(float)
sat = hsv[:,:,1].astype(float)
diff = B - R

# === MASKS ===
r_mask = remove_small_objects(
    cv2.morphologyEx((R>45).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)
ratio = B / (R + B + 1)
b_mask = remove_small_objects(
    cv2.morphologyEx(((ratio > 0.55) & (B > 15)).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)

vessel_all = remove_small_objects(
    cv2.morphologyEx((gray>10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=100)
vm = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

# Yellow zone (raw)
yellow_hue = (hue >= 15) & (hue <= 50) & (sat > 30) & (gray > 15)
yellow_diff = (np.abs(diff) < 15) & (R > 20) & (B > 20)
yellow_zone = yellow_hue | yellow_diff

ovl = np.sum(r_mask[:by] & b_mask[:by])
ovl_pct = ovl / max(np.sum(r_mask[:by]),1) * 100
print(f'R mask: {np.sum(r_mask[:by])}px | B mask: {np.sum(b_mask[:by])}px | overlap: {ovl}px ({ovl_pct:.0f}% of R)')

# === SKEL (no dominance cut) ===
r_skel = skeletonize(r_mask)
b_skel = skeletonize(b_mask)
r_comp = ndimage.label(r_skel)[1]
b_comp = ndimage.label(b_skel)[1]
print(f'R skel: {np.sum(r_skel[:by])}px, {r_comp} comp | B skel: {np.sum(b_skel[:by])}px, {b_comp} comp')

# === EP/JN ===
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
    ep_pts = [(y,x) for y,x in np.argwhere((nb==1) & roi) if y < by-5]
    return jn_pts, ep_pts

r_jn, r_ep = get_jn_ep(r_skel, h, by)
b_jn, b_ep = get_jn_ep(b_skel, h, by)
print(f'R: JN={len(r_jn)} EP={len(r_ep)} | B: JN={len(b_jn)} EP={len(b_ep)}')

# === DEPTH TRANSITION ===
def bezier_pts(e1_pos, e1_tan, e2_pos, e2_tan):
    p1 = np.array(e1_pos, dtype=float); p2 = np.array(e2_pos, dtype=float)
    d = np.linalg.norm(p2-p1)
    c1 = p1 + np.array(e1_tan)*d*0.4; c2 = p2 + np.array(e2_tan)*d*0.4
    n = max(int(d*1.5), 10)
    return [(int(round(((1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2)[0])),
             int(round(((1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2)[1])))
            for t in np.linspace(0, 1, n)]

r_eps_t = [e for e in get_eps_tangent(r_skel, h, w) if e['pos'][0] < by-5]
b_eps_t = [e for e in get_eps_tangent(b_skel, h, w) if e['pos'][0] < by-5]

max_pair_dist = 30  # 512x512 is bigger, allow slightly more
pairs = []
for ri, rep in enumerate(r_eps_t):
    ry, rx = rep['pos']
    for bi, bep in enumerate(b_eps_t):
        by2, bx = bep['pos']
        d = np.sqrt((ry-by2)**2 + (rx-bx)**2)
        if 2 < d < max_pair_dist:
            path = bezier_pts(rep['pos'], rep['tangent'], bep['pos'], bep['tangent'])
            y_count = sum(1 for (py,px) in path if 0<=py<h and 0<=px<w and yellow_zone[py,px])
            y_ratio = y_count / max(len(path), 1)
            vm_count = sum(1 for (py,px) in path if 0<=py<h and 0<=px<w and vm[py,px])
            vm_ratio = vm_count / max(len(path), 1)
            pairs.append({
                'ri': ri, 'bi': bi, 'r_ep': rep, 'b_ep': bep,
                'dist': d, 'path': path,
                'y_ratio': y_ratio, 'vm_ratio': vm_ratio,
                'mid': (int((ry+by2)/2), int((rx+bx)/2))
            })

print(f'\nDepth transition: {len(pairs)} R-B pairs found')
for p in pairs:
    ry, rx = p['r_ep']['pos']; by2, bx = p['b_ep']['pos']
    print(f'  R({ry},{rx})↔B({by2},{bx}) d={p["dist"]:.1f} yellow={p["y_ratio"]:.0%} vm={p["vm_ratio"]:.0%}')

# Apply threshold 30%
thr = 0.3
passed = [p for p in pairs if p['y_ratio'] >= thr and p['vm_ratio'] >= 0.6]
passed.sort(key=lambda x: x['dist'])
used_r = set(); used_b = set(); selected = []
for p in passed:
    if p['ri'] not in used_r and p['bi'] not in used_b:
        selected.append(p); used_r.add(p['ri']); used_b.add(p['bi'])
print(f'Yellow >= {thr:.0%}: pass={len(passed)} selected={len(selected)}')

# === FIGURE: 5 rows x 3 cols ===
BG_A = 0.5; WV = 0.4
fig, axes = plt.subplots(5, 3, figsize=(18, 26))
fig.suptitle(f'Sample 1 (512x512) — New Pipeline v2\nR>45 | B ratio>0.55 | No dom cut | Depth transition (yellow≥30%)',
             fontsize=13, fontweight='bold')

# Row 0: GT, R mask, B mask
axes[0,0].imshow(gt[:by]); axes[0,0].set_title('GT (multi-color)', fontsize=11)
axes[0,1].imshow(r_mask[:by], cmap='gray'); axes[0,1].set_title(f'R mask (R>45) | {np.sum(r_mask[:by])}px', fontsize=10)
axes[0,2].imshow(b_mask[:by], cmap='gray'); axes[0,2].set_title(f'B mask (ratio>0.55) | {np.sum(b_mask[:by])}px', fontsize=10)

# Row 1: Overlap, R skel, B skel
vis = np.zeros((by,w,3), dtype=np.uint8)
vis[r_mask[:by] & ~b_mask[:by]] = [200,60,60]
vis[b_mask[:by] & ~r_mask[:by]] = [60,120,200]
vis[r_mask[:by] & b_mask[:by]] = [200,100,200]
axes[1,0].imshow(vis); axes[1,0].set_title(f'R∩B overlap={ovl_pct:.0f}% of R', fontsize=10)

vis = np.zeros((by,w,3), dtype=np.uint8)
vis[cv2.dilate(r_skel[:by].astype(np.uint8), dk2)>0] = [230,70,70]
axes[1,1].imshow(vis); axes[1,1].set_title(f'R skel | {r_comp} comp', fontsize=10)

vis = np.zeros((by,w,3), dtype=np.uint8)
vis[cv2.dilate(b_skel[:by].astype(np.uint8), dk2)>0] = [70,140,240]
axes[1,2].imshow(vis); axes[1,2].set_title(f'B skel | {b_comp} comp', fontsize=10)

# Row 2: Depth tag + yellow zone
r_dom = r_skel & (diff <= 0); r_und = r_skel & (diff > 0)
b_dom = b_skel & (diff >= 0); b_und = b_skel & (diff < 0)

vis = np.zeros((by,w,3), dtype=np.uint8)
vis[cv2.dilate(r_dom[:by].astype(np.uint8), dk2)>0] = [255,80,80]
vis[cv2.dilate(r_und[:by].astype(np.uint8), dk2)>0] = [120,40,40]
vis[cv2.dilate(b_dom[:by].astype(np.uint8), dk2)>0] = [80,150,255]
vis[cv2.dilate(b_und[:by].astype(np.uint8), dk2)>0] = [40,60,120]
axes[2,0].imshow(vis); axes[2,0].set_title('Depth-tagged skels (bright=dominant, dark=under)', fontsize=9)

vis = np.zeros((by,w,3), dtype=np.uint8)
vis[yellow_hue[:by]] = [220,220,50]
vis[yellow_diff[:by] & ~yellow_hue[:by]] = [160,160,50]
axes[2,1].imshow(vis); axes[2,1].set_title('Yellow zone (raw, no dilation)', fontsize=10)

# Depth transition pairs
vis = gt[:by].copy().astype(float) * 0.5
vis[cv2.dilate(r_skel[:by].astype(np.uint8), dk2)>0] = [230,70,70]
vis[cv2.dilate(b_skel[:by].astype(np.uint8), dk2)>0] = [70,140,240]
vis = vis.astype(np.uint8)
for p in pairs:
    ry, rx = p['r_ep']['pos']; by2, bx = p['b_ep']['pos']
    is_pass = p['y_ratio'] >= thr and p['vm_ratio'] >= 0.6
    color = (0,255,0) if is_pass else (255,80,80)
    cv2.line(vis, (rx,ry), (bx,by2), color, 1, cv2.LINE_AA)
    cv2.circle(vis, (rx,ry), 3, (255,100,100), -1)
    cv2.circle(vis, (bx,by2), 3, (100,150,255), -1)
    my, mx = p['mid']
    if my < by:
        cv2.putText(vis, f"{p['y_ratio']:.0%}", (mx+4,my-2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
axes[2,2].imshow(vis); axes[2,2].set_title(f'R-B pairs ({len(pairs)}): green=pass, red=fail', fontsize=9)

# Row 3: R+B on GT with EP/JN
vis = gt[:by].copy().astype(float) * BG_A
vis = vis * (1-WV) + 255*WV
r_d4 = cv2.dilate(r_skel[:by].astype(np.uint8), dk4)>0
b_d4 = cv2.dilate(b_skel[:by].astype(np.uint8), dk4)>0
vis[r_d4 & ~b_d4] = [230,70,70]; vis[b_d4 & ~r_d4] = [70,140,240]; vis[r_d4 & b_d4] = [180,100,220]
vis = vis.astype(np.uint8)
for y,x in r_jn+b_jn: cv2.circle(vis,(x,y),6,(255,0,255),2)
for y,x in r_ep+b_ep: cv2.circle(vis,(x,y),6,(255,255,0),2)
axes[3,0].imshow(vis)
axes[3,0].set_title(f'R+B on GT | JN={len(r_jn)+len(b_jn)} EP={len(r_ep)+len(b_ep)}', fontsize=10)

# R on GT
vis = gt[:by].copy().astype(float) * BG_A; vis = vis*(1-WV)+255*WV
vis[r_d4] = [230,70,70]; vis = vis.astype(np.uint8)
for y,x in r_jn: cv2.circle(vis,(x,y),6,(255,0,255),2)
for y,x in r_ep: cv2.circle(vis,(x,y),6,(255,255,0),2)
axes[3,1].imshow(vis); axes[3,1].set_title(f'R on GT | JN={len(r_jn)} EP={len(r_ep)}', fontsize=10)

# B on GT
vis = gt[:by].copy().astype(float) * BG_A; vis = vis*(1-WV)+255*WV
vis[b_d4] = [70,140,240]; vis = vis.astype(np.uint8)
for y,x in b_jn: cv2.circle(vis,(x,y),6,(255,0,255),2)
for y,x in b_ep: cv2.circle(vis,(x,y),6,(255,255,0),2)
axes[3,2].imshow(vis); axes[3,2].set_title(f'B on GT | JN={len(b_jn)} EP={len(b_ep)}', fontsize=10)

# Row 4: Final with depth transitions + summary
vis = gt[:by].copy().astype(float) * BG_A; vis = vis*(1-WV)+255*WV
vis[r_d4 & ~b_d4] = [230,70,70]; vis[b_d4 & ~r_d4] = [70,140,240]; vis[r_d4 & b_d4] = [180,100,220]
vis = vis.astype(np.uint8)
for p in selected:
    for py, px in p['path']:
        if 0<=py<by and 0<=px<w:
            for dy in range(-1,2):
                for dx in range(-1,2):
                    ny, nx = py+dy, px+dx
                    if 0<=ny<by and 0<=nx<w: vis[ny,nx] = [255,220,50]
    my, mx = p['mid']
    if my < by: cv2.circle(vis, (mx,my), 7, (0,255,255), 2)
for y,x in r_jn+b_jn: cv2.circle(vis,(x,y),6,(255,0,255),2)
for y,x in r_ep+b_ep: cv2.circle(vis,(x,y),6,(255,255,0),2)
axes[4,0].imshow(vis)
axes[4,0].set_title(f'Final: R+B + transitions({len(selected)}) + EP/JN', fontsize=10)

# Combined skel on black
vis = np.zeros((by,w,3), dtype=np.uint8)
r_d2 = cv2.dilate(r_skel[:by].astype(np.uint8), dk2)>0
b_d2 = cv2.dilate(b_skel[:by].astype(np.uint8), dk2)>0
vis[r_d2 & ~b_d2] = [230,70,70]; vis[b_d2 & ~r_d2] = [70,140,240]; vis[r_d2 & b_d2] = [180,100,220]
for p in selected:
    for py, px in p['path']:
        if 0<=py<by and 0<=px<w: vis[py,px] = [255,220,50]
axes[4,1].imshow(vis); axes[4,1].set_title('Combined skel (R+B+transition)', fontsize=10)

# Summary
axes[4,2].axis('off')
r_L = np.sum(r_skel[:by]); b_L = np.sum(b_skel[:by])
txt = (f"Sample 1 (512x512)\n\n"
       f"R mask: R>45, {np.sum(r_mask[:by])}px\n"
       f"B mask: ratio>0.55, {np.sum(b_mask[:by])}px\n"
       f"Overlap: {ovl}px ({ovl_pct:.0f}% of R)\n\n"
       f"R skel: L={r_L} JN={len(r_jn)} EP={len(r_ep)}\n"
       f"B skel: L={b_L} JN={len(b_jn)} EP={len(b_ep)}\n"
       f"R+B:    L={r_L+b_L} JN={len(r_jn)+len(b_jn)} EP={len(r_ep)+len(b_ep)}\n\n"
       f"Depth transitions:\n"
       f"  Pairs found: {len(pairs)}\n"
       f"  Yellow>=30%: {len(passed)}\n"
       f"  Selected: {len(selected)}\n\n"
       f"R depth: dom={np.sum(r_dom[:by])}px under={np.sum(r_und[:by])}px\n"
       f"B depth: dom={np.sum(b_dom[:by])}px under={np.sum(b_und[:by])}px")
axes[4,2].text(0.05, 0.95, txt, fontsize=10, fontfamily='monospace', va='top',
               transform=axes[4,2].transAxes)

for ax in axes.flat: ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'newpipe_v2_sample1_512.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'\nSaved newpipe_v2_sample1_512.png')
print('Done!')
