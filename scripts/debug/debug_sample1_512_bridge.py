"""
Sample 1 (512x512): Full pipeline with bridging + depth transition.
R>45 | B ratio>0.55 | No dom cut | EP-EP bridge | Depth transition.
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

s1 = np.array(Image.open(DATA / 'Original color stack data' / 'Sample 1_BF and GT(512x512 two images).png'))
gt = s1[512:]
gt_blur = cv2.GaussianBlur(gt, (7,7), 0)
h, w = gt.shape[:2]; by = int(h*0.8)

R = gt_blur[:,:,0].astype(np.float32); B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
hsv = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2HSV)
hue = hsv[:,:,0].astype(float); sat = hsv[:,:,1].astype(float)
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

# === SKEL (no dom cut) ===
r_skel_raw = skeletonize(r_mask)
b_skel_raw = skeletonize(b_mask)

# === EP tangent ===
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

def bezier_pts(e1_pos, e1_tan, e2_pos, e2_tan):
    p1 = np.array(e1_pos, dtype=float); p2 = np.array(e2_pos, dtype=float)
    d = np.linalg.norm(p2-p1)
    c1 = p1 + np.array(e1_tan)*d*0.4; c2 = p2 + np.array(e2_tan)*d*0.4
    n = max(int(d*1.5), 10)
    return [(int(round(((1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2)[0])),
             int(round(((1-t)**3*p1 + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p2)[1])))
            for t in np.linspace(0, 1, n)]

# === BRIDGING: EP-EP within same channel ===
def bridge_same_channel(skel, ch_mask, vm, h, w, max_dist=80, max_angle=65):
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
            c1v = np.dot(eps[i]['tangent'], d12)
            c2v = np.dot(eps[j]['tangent'], -d12)
            if c1v < ct or c2v < ct: continue
            pts = bezier(eps[i], eps[j])
            # Channel mask 70%
            ch_count = sum(1 for (y,x) in pts if 0<=y<h and 0<=x<w and ch_mask[y,x])
            if ch_count / max(len(pts),1) < 0.7: continue
            # Vessel mask 70%
            vm_count = sum(1 for (y,x) in pts if 0<=y<h and 0<=x<w and vm[y,x])
            if vm_count / max(len(pts),1) < 0.7: continue
            sc = (c1v+c2v)/2 - d/max_dist*0.2
            cands.append((i, j, sc, eps[i], eps[j], pts))
    cands.sort(key=lambda x: -x[2])
    used = set(); bridged = skel.copy(); bridges = []
    for i, j, sc, e1, e2, pts in cands:
        if i not in used and j not in used:
            for (y,x) in pts:
                if 0<=y<h and 0<=x<w: bridged[y,x] = True
            used.add(i); used.add(j)
            bridges.append({'ep1': e1, 'ep2': e2, 'pts': pts})
    return skeletonize(bridged), bridges

r_bridged, r_bridges = bridge_same_channel(r_skel_raw, r_mask, vm, h, w)
b_bridged, b_bridges = bridge_same_channel(b_skel_raw, b_mask, vm, h, w)

print(f'R bridges: {len(r_bridges)} | B bridges: {len(b_bridges)}')

# === EP/JN after bridging ===
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

r_jn_pre, r_ep_pre = get_jn_ep(r_skel_raw, h, by)
b_jn_pre, b_ep_pre = get_jn_ep(b_skel_raw, h, by)
r_jn_post, r_ep_post = get_jn_ep(r_bridged, h, by)
b_jn_post, b_ep_post = get_jn_ep(b_bridged, h, by)

print(f'Before bridge: R JN={len(r_jn_pre)} EP={len(r_ep_pre)} | B JN={len(b_jn_pre)} EP={len(b_ep_pre)}')
print(f'After bridge:  R JN={len(r_jn_post)} EP={len(r_ep_post)} | B JN={len(b_jn_post)} EP={len(b_ep_post)}')

# === DEPTH TRANSITION (R EP ↔ B EP through yellow) ===
r_eps_t = [e for e in get_eps_tangent(r_bridged, h, w) if e['pos'][0] < by-5]
b_eps_t = [e for e in get_eps_tangent(b_bridged, h, w) if e['pos'][0] < by-5]

pairs = []
for ri, rep in enumerate(r_eps_t):
    ry, rx = rep['pos']
    for bi, bep in enumerate(b_eps_t):
        by2, bx = bep['pos']
        d = np.sqrt((ry-by2)**2 + (rx-bx)**2)
        if 2 < d < 30:
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

thr = 0.3
passed = [p for p in pairs if p['y_ratio'] >= thr and p['vm_ratio'] >= 0.6]
passed.sort(key=lambda x: x['dist'])
used_r = set(); used_b = set(); conn_selected = []
for p in passed:
    if p['ri'] not in used_r and p['bi'] not in used_b:
        conn_selected.append(p); used_r.add(p['ri']); used_b.add(p['bi'])

print(f'\nDepth transition: {len(pairs)} pairs, {len(passed)} pass(>=30%), {len(conn_selected)} selected')
for p in conn_selected:
    ry, rx = p['r_ep']['pos']; by2, bx = p['b_ep']['pos']
    print(f'  R({ry},{rx})↔B({by2},{bx}) d={p["dist"]:.1f} y={p["y_ratio"]:.0%}')

# Connection points: reduce EP count
conn_r_eps = set(p['ri'] for p in conn_selected)
conn_b_eps = set(p['bi'] for p in conn_selected)
r_real_ep = [ep for i, ep in enumerate(r_eps_t) if i not in conn_r_eps and ep['pos'] in [(y,x) for y,x in r_ep_post]]
b_real_ep = [ep for i, ep in enumerate(b_eps_t) if i not in conn_b_eps and ep['pos'] in [(y,x) for y,x in b_ep_post]]

# Simpler: just count
r_ep_conn = len([p for p in conn_selected])  # R EPs used as connection
b_ep_conn = len([p for p in conn_selected])  # B EPs used as connection

print(f'\nFinal EP count:')
print(f'  R: total EP={len(r_ep_post)}, connection={r_ep_conn}, real={len(r_ep_post)-r_ep_conn}')
print(f'  B: total EP={len(b_ep_post)}, connection={b_ep_conn}, real={len(b_ep_post)-b_ep_conn}')

# === FIGURE: 5 rows x 3 cols ===
BG_A = 0.5; WV = 0.4
fig, axes = plt.subplots(5, 3, figsize=(18, 26))
fig.suptitle('Sample 1 (512x512) — Full Pipeline: Bridge + Depth Transition',
             fontsize=13, fontweight='bold')

# Row 0: GT, R mask, B mask
axes[0,0].imshow(gt[:by]); axes[0,0].set_title('GT', fontsize=11)
axes[0,1].imshow(r_mask[:by], cmap='gray'); axes[0,1].set_title(f'R mask (R>45)', fontsize=10)
axes[0,2].imshow(b_mask[:by], cmap='gray'); axes[0,2].set_title(f'B mask (ratio>0.55)', fontsize=10)

# Row 1: Before bridge — R, B, R+B on GT
for col, (sk, jns, eps, color, label) in enumerate([
    (r_skel_raw, r_jn_pre, r_ep_pre, [230,70,70], 'R'),
    (b_skel_raw, b_jn_pre, b_ep_pre, [70,140,240], 'B')]):
    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(sk[:by].astype(np.uint8), dk2)>0] = color
    for y,x in jns: cv2.circle(vis,(x,y),5,(255,0,255),-1)
    for y,x in eps: cv2.circle(vis,(x,y),4,(255,255,0),-1)
    axes[1,col].imshow(vis)
    axes[1,col].set_title(f'Before: {label} JN={len(jns)} EP={len(eps)}', fontsize=10)

vis = gt[:by].copy().astype(float)*BG_A; vis = vis*(1-WV)+255*WV
r_d = cv2.dilate(r_skel_raw[:by].astype(np.uint8), dk4)>0
b_d = cv2.dilate(b_skel_raw[:by].astype(np.uint8), dk4)>0
vis[r_d & ~b_d] = [230,70,70]; vis[b_d & ~r_d] = [70,140,240]; vis[r_d & b_d] = [180,100,220]
vis = vis.astype(np.uint8)
for y,x in r_jn_pre+b_jn_pre: cv2.circle(vis,(x,y),6,(255,0,255),2)
for y,x in r_ep_pre+b_ep_pre: cv2.circle(vis,(x,y),6,(255,255,0),2)
axes[1,2].imshow(vis)
axes[1,2].set_title(f'Before: R+B JN={len(r_jn_pre)+len(b_jn_pre)} EP={len(r_ep_pre)+len(b_ep_pre)}', fontsize=9)

# Row 2: After bridge — R, B, R+B with bridge paths
for col, (sk_pre, sk_post, bridges, jns, eps, color, br_color, label) in enumerate([
    (r_skel_raw, r_bridged, r_bridges, r_jn_post, r_ep_post, [230,70,70], [255,180,50], 'R'),
    (b_skel_raw, b_bridged, b_bridges, b_jn_post, b_ep_post, [70,140,240], [50,255,180], 'B')]):
    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[cv2.dilate(sk_post[:by].astype(np.uint8), dk2)>0] = color
    # Draw bridge paths
    for br in bridges:
        for py, px in br['pts']:
            if 0<=py<by and 0<=px<w:
                for dy in range(-1,2):
                    for dx in range(-1,2):
                        ny, nx = py+dy, px+dx
                        if 0<=ny<by and 0<=nx<w: vis[ny,nx] = br_color
    for y,x in jns: cv2.circle(vis,(x,y),5,(255,0,255),-1)
    for y,x in eps: cv2.circle(vis,(x,y),4,(255,255,0),-1)
    axes[2,col].imshow(vis)
    axes[2,col].set_title(f'After: {label} JN={len(jns)} EP={len(eps)} br={len(bridges)}', fontsize=10)

vis = gt[:by].copy().astype(float)*BG_A; vis = vis*(1-WV)+255*WV
r_d2 = cv2.dilate(r_bridged[:by].astype(np.uint8), dk4)>0
b_d2 = cv2.dilate(b_bridged[:by].astype(np.uint8), dk4)>0
vis[r_d2 & ~b_d2] = [230,70,70]; vis[b_d2 & ~r_d2] = [70,140,240]; vis[r_d2 & b_d2] = [180,100,220]
vis = vis.astype(np.uint8)
for y,x in r_jn_post+b_jn_post: cv2.circle(vis,(x,y),6,(255,0,255),2)
for y,x in r_ep_post+b_ep_post: cv2.circle(vis,(x,y),6,(255,255,0),2)
axes[2,2].imshow(vis)
axes[2,2].set_title(f'After: R+B JN={len(r_jn_post)+len(b_jn_post)} EP={len(r_ep_post)+len(b_ep_post)}', fontsize=9)

# Row 3: Depth transition visualization
vis = gt[:by].copy().astype(float)*0.5
vis[cv2.dilate(r_bridged[:by].astype(np.uint8), dk2)>0] = [230,70,70]
vis[cv2.dilate(b_bridged[:by].astype(np.uint8), dk2)>0] = [70,140,240]
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
axes[3,0].imshow(vis)
axes[3,0].set_title(f'Depth transition pairs ({len(pairs)}): green=pass({len(passed)})', fontsize=9)

# Yellow zone
vis = np.zeros((by,w,3), dtype=np.uint8)
vis[yellow_hue[:by]] = [220,220,50]
vis[yellow_diff[:by] & ~yellow_hue[:by]] = [160,160,50]
axes[3,1].imshow(vis); axes[3,1].set_title('Yellow zone (raw)', fontsize=10)

# Selected transitions on GT
vis = gt[:by].copy().astype(float)*BG_A; vis = vis*(1-WV)+255*WV
vis[r_d2 & ~b_d2] = [230,70,70]; vis[b_d2 & ~r_d2] = [70,140,240]; vis[r_d2 & b_d2] = [180,100,220]
vis = vis.astype(np.uint8)
for p in conn_selected:
    for py, px in p['path']:
        if 0<=py<by and 0<=px<w:
            for dy in range(-1,2):
                for dx in range(-1,2):
                    ny, nx = py+dy, px+dx
                    if 0<=ny<by and 0<=nx<w: vis[ny,nx] = [255,220,50]
    my, mx = p['mid']
    if my < by: cv2.circle(vis, (mx,my), 7, (0,255,255), 2)
for y,x in r_jn_post+b_jn_post: cv2.circle(vis,(x,y),6,(255,0,255),2)
for y,x in r_ep_post+b_ep_post: cv2.circle(vis,(x,y),6,(255,255,0),2)
axes[3,2].imshow(vis)
axes[3,2].set_title(f'Selected transitions ({len(conn_selected)}): yellow=bridge, cyan=conn.pt', fontsize=9)

# Row 4: Final combined + summary
# Depth-tagged combined
vis = np.zeros((by,w,3), dtype=np.uint8)
r_dom = cv2.dilate((r_bridged & (diff<=0))[:by].astype(np.uint8), dk2)>0
r_und = cv2.dilate((r_bridged & (diff>0))[:by].astype(np.uint8), dk2)>0
b_dom = cv2.dilate((b_bridged & (diff>=0))[:by].astype(np.uint8), dk2)>0
b_und = cv2.dilate((b_bridged & (diff<0))[:by].astype(np.uint8), dk2)>0
vis[r_dom] = [255,80,80]; vis[r_und] = [120,40,40]
vis[b_dom] = [80,150,255]; vis[b_und] = [40,60,120]
for p in conn_selected:
    for py, px in p['path']:
        if 0<=py<by and 0<=px<w: vis[py,px] = [255,220,50]
axes[4,0].imshow(vis); axes[4,0].set_title('Depth-tagged + transitions', fontsize=10)

# Final on GT
vis = gt[:by].copy().astype(float)*BG_A; vis = vis*(1-WV)+255*WV
vis[r_d2 & ~b_d2] = [230,70,70]; vis[b_d2 & ~r_d2] = [70,140,240]; vis[r_d2 & b_d2] = [180,100,220]
vis = vis.astype(np.uint8)
for p in conn_selected:
    for py, px in p['path']:
        if 0<=py<by and 0<=px<w:
            for dy in range(-1,2):
                for dx in range(-1,2):
                    ny, nx = py+dy, px+dx
                    if 0<=ny<by and 0<=nx<w: vis[ny,nx] = [255,220,50]
    my, mx = p['mid']
    if my < by: cv2.circle(vis, (mx,my), 7, (0,255,255), 2)
# Mark connection points differently from regular EPs
for y,x in r_jn_post+b_jn_post: cv2.circle(vis,(x,y),6,(255,0,255),2)
for y,x in r_ep_post+b_ep_post: cv2.circle(vis,(x,y),6,(255,255,0),2)
axes[4,1].imshow(vis); axes[4,1].set_title('Final: R+B + transitions + EP/JN', fontsize=10)

# Summary
axes[4,2].axis('off')
r_L = np.sum(r_bridged[:by]); b_L = np.sum(b_bridged[:by])
txt = (f"Sample 1 (512x512) Summary\n\n"
       f"Before bridge:\n"
       f"  R: JN={len(r_jn_pre)} EP={len(r_ep_pre)} L={np.sum(r_skel_raw[:by])}\n"
       f"  B: JN={len(b_jn_pre)} EP={len(b_ep_pre)} L={np.sum(b_skel_raw[:by])}\n\n"
       f"After bridge (R br={len(r_bridges)}, B br={len(b_bridges)}):\n"
       f"  R: JN={len(r_jn_post)} EP={len(r_ep_post)} L={r_L}\n"
       f"  B: JN={len(b_jn_post)} EP={len(b_ep_post)} L={b_L}\n\n"
       f"Depth transitions:\n"
       f"  Pairs: {len(pairs)} | Pass(>=30%): {len(passed)}\n"
       f"  Selected: {len(conn_selected)}\n\n"
       f"Final (after connection):\n"
       f"  R+B: JN={len(r_jn_post)+len(b_jn_post)}\n"
       f"        EP total={len(r_ep_post)+len(b_ep_post)}\n"
       f"        EP conn={len(conn_selected)}\n"
       f"        EP real={len(r_ep_post)+len(b_ep_post)-2*len(conn_selected)}\n"
       f"        L={r_L+b_L}")
axes[4,2].text(0.05, 0.95, txt, fontsize=10, fontfamily='monospace', va='top',
               transform=axes[4,2].transAxes)

for ax in axes.flat: ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'sample1_512_full_pipeline.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('\nSaved sample1_512_full_pipeline.png')
print('Done!')
