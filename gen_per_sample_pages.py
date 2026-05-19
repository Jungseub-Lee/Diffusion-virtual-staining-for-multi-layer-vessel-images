"""
Generate per-sample comparison pages:
- For each of 5 samples: GT + 5 models
- Each shows: original image, Skel R, Skel B, Combined
- Combined includes connected EP detection (cyan = connected, yellow = real)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2, os
from PIL import Image
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')
DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')
pd = OUT / 'paper_panels_depth' / 'per_sample'
pd.mkdir(parents=True, exist_ok=True)

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
BG_ALPHA = 0.45

def get_endpoints_with_tangent(skel, h, w, trace_len=20):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    endpoints = np.argwhere(nb == 1)
    results = []
    for ep in endpoints:
        y0, x0 = ep
        path = [(y0, x0)]; visited = {(y0, x0)}
        cy, cx = y0, x0
        for _ in range(trace_len):
            found = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0: continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and (ny, nx) not in visited:
                        path.append((ny, nx)); visited.add((ny, nx))
                        cy, cx = ny, nx; found = True; break
                if found: break
            if not found: break
        if len(path) >= 5:
            n_avg = min(len(path), 10)
            tangent = np.array(path[0], dtype=float) - np.array(path[n_avg-1], dtype=float)
            norm = np.linalg.norm(tangent)
            if norm > 0: tangent /= norm
            results.append({'pos': (y0, x0), 'tangent': tangent})
    return results

def _bridge_pts(ep1, ep2):
    p1 = np.array(ep1['pos'], dtype=float)
    p2 = np.array(ep2['pos'], dtype=float)
    t1, t2 = ep1['tangent'], ep2['tangent']
    dist = np.linalg.norm(p2 - p1)
    ctrl1 = p1 + t1 * dist * 0.4
    ctrl2 = p2 + t2 * dist * 0.4
    pts = []
    for t in np.linspace(0, 1, max(int(dist * 1.5), 10)):
        p = (1-t)**3*p1 + 3*(1-t)**2*t*ctrl1 + 3*(1-t)*t**2*ctrl2 + t**3*p2
        pts.append((int(round(p[0])), int(round(p[1]))))
    return pts

def match_and_bridge(skel, vessel_mask, h, w, max_dist=80, max_angle_deg=60):
    eps = get_endpoints_with_tangent(skel, h, w)
    candidates = []
    for i in range(len(eps)):
        for j in range(i+1, len(eps)):
            p1 = np.array(eps[i]['pos'], dtype=float)
            p2 = np.array(eps[j]['pos'], dtype=float)
            dist = np.linalg.norm(p2 - p1)
            if dist < 5 or dist > max_dist: continue
            dir_12 = (p2 - p1) / dist
            cos1 = np.dot(eps[i]['tangent'], dir_12)
            cos2 = np.dot(eps[j]['tangent'], -dir_12)
            if cos1 < np.cos(np.radians(max_angle_deg)) or cos2 < np.cos(np.radians(max_angle_deg)): continue
            pts = _bridge_pts(eps[i], eps[j])
            on_v = sum(1 for (y, x) in pts if 0 <= y < h and 0 <= x < w and vessel_mask[y, x])
            vr = on_v / max(len(pts), 1)
            if vr < 0.7: continue
            score = (cos1 + cos2) / 2 - dist / max_dist * 0.2 + vr * 0.2
            candidates.append((i, j, score, dist, vr))
    candidates.sort(key=lambda x: -x[2])
    used = set(); bridged = skel.copy(); bridges = []
    for i, j, sc, d, vr in candidates:
        if i not in used and j not in used:
            used.add(i); used.add(j)
            pts = _bridge_pts(eps[i], eps[j])
            bridges.append((pts, sc, d, vr, eps[i], eps[j]))
            for (y, x) in pts:
                if 0 <= y < h and 0 <= x < w: bridged[y, x] = True
    return bridged, bridges

def process_full(img_rgb):
    """Full pipeline returning all needed data"""
    img_blur = cv2.GaussianBlur(img_rgb, (7, 7), 0)
    h, w = img_rgb.shape[:2]; by = int(h * 0.8)
    R = img_blur[:,:,0].astype(np.float32)
    B = img_blur[:,:,2].astype(np.float32)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    diff = B - R

    rm = cv2.morphologyEx((R > 30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
    rm = remove_small_objects(rm, min_size=50)
    rs = skeletonize(rm)

    bs_ = B > 50; bl = B > 20
    blab, bn = ndimage.label(bl)
    bm = np.zeros_like(bl, dtype=bool)
    for i in range(1, bn + 1):
        if np.any(bs_[blab == i]): bm |= (blab == i)
    bm = cv2.morphologyEx(bm.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0
    bm = remove_small_objects(bm, min_size=50)
    bsk = skeletonize(bm)

    rf = rs & (diff < 0); bf = bsk & (diff > 0)
    va = remove_small_objects(
        cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=100)
    vm = cv2.dilate(va.astype(np.uint8)*255, k5) > 0

    rb, r_br = match_and_bridge(rf, vm, h, w)
    bb, b_br = match_and_bridge(bf, vm, h, w)
    return rb, bb, r_br, b_br, by, h, w

def skel_jn_ep_full(skel, h, crop_y):
    """Return JN points, real EP points, connected EP points"""
    roi = np.zeros_like(skel, dtype=bool); roi[:crop_y] = True
    sc = skel & roi
    su = sc.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su

    # JN clustering
    jm = nb >= 3
    jlab, jn = ndimage.label(jm)
    jn_pts = []
    for i in range(1, jn + 1):
        cluster = jlab == i
        dil = cv2.dilate(cluster.astype(np.uint8), k3) > 0
        ts = sc.copy(); ts[dil] = False
        local = np.zeros_like(ts)
        py, px = np.where(cluster)
        ymin, ymax = max(0, py.min()-5), min(h, py.max()+6)
        xmin, xmax = max(0, px.min()-5), min(sc.shape[1], px.max()+6)
        local[ymin:ymax, xmin:xmax] = ts[ymin:ymax, xmin:xmax]
        _, nbr = ndimage.label(local)
        if nbr >= 3:
            jn_pts.append((int(py.mean()), int(px.mean())))

    # EP (excluding baseline)
    ep_all = np.argwhere((nb == 1) & roi)
    ep_pts = [(y, x) for y, x in ep_all if y < crop_y - 3]
    return jn_pts, ep_pts

def find_connected_eps(r_skel, b_skel, by, proximity=8):
    """Find EP pairs between R and B that are close → connected (cyan), rest → real (yellow)"""
    _, r_eps = skel_jn_ep_full(r_skel, r_skel.shape[0], by)
    _, b_eps = skel_jn_ep_full(b_skel, b_skel.shape[0], by)

    r_connected = set()
    b_connected = set()
    for ri, (ry, rx) in enumerate(r_eps):
        for bi, (bry, brx) in enumerate(b_eps):
            dist = np.sqrt((ry - bry)**2 + (rx - brx)**2)
            if dist < proximity:
                r_connected.add(ri)
                b_connected.add(bi)

    r_real = [(y, x) for i, (y, x) in enumerate(r_eps) if i not in r_connected]
    r_conn = [(y, x) for i, (y, x) in enumerate(r_eps) if i in r_connected]
    b_real = [(y, x) for i, (y, x) in enumerate(b_eps) if i not in b_connected]
    b_conn = [(y, x) for i, (y, x) in enumerate(b_eps) if i in b_connected]

    return r_real, r_conn, b_real, b_conn

def render_panel(img_crop, skel_crop, color, jn_pts, real_eps, conn_eps=None):
    """Render with JN=magenta, real EP=yellow, connected EP=cyan"""
    vis = img_crop.copy().astype(float) * BG_ALPHA
    vis[cv2.dilate(skel_crop.astype(np.uint8), dk4) > 0] = color
    vis = vis.astype(np.uint8)
    for y, x in jn_pts:
        cv2.circle(vis, (x, y), 4, (255, 0, 255), -1)
    for y, x in real_eps:
        cv2.circle(vis, (x, y), 4, (255, 255, 0), -1)
    if conn_eps:
        for y, x in conn_eps:
            cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
    return vis

def save_img(img, fpath):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img); ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(fpath, dpi=200, bbox_inches='tight', pad_inches=0.01, facecolor='black')
    plt.close()

# ===== Main processing =====
sids = ['1-19-716', '9-15-512', '16-18-716', '18-19-512', '1-16-512']
models = ['LBBDM', 'PBBDM', 'LSGAN', 'Pix2pix', 'WGANGP']

for sid in sids:
    print(f"\n===== {sid} =====")

    # Load GT
    gt_path = None
    for gm in ['LSGAN', 'Pix2pix', 'WGANGP']:
        p = DATA / 'Multi-color' / gm / f'{sid}_real_B.png'
        if p.exists(): gt_path = p; break
    bf_path = None
    for gm in ['LSGAN', 'Pix2pix', 'WGANGP']:
        p = DATA / 'Multi-color' / gm / f'{sid}_real_A.png'
        if p.exists(): bf_path = p; break

    if gt_path is None:
        print(f"  GT not found"); continue

    gt_img = np.array(Image.open(gt_path))
    if gt_img.shape[0] != 512:
        gt_img = cv2.resize(gt_img, (512, 512))

    # Save GT original
    save_img(gt_img, pd / f'{sid}_gt.png')

    # Process GT
    r_br, b_br, _, _, by, h, w = process_full(gt_img)
    r_jn, r_ep = skel_jn_ep_full(r_br, h, by)
    b_jn, b_ep = skel_jn_ep_full(b_br, h, by)
    r_real, r_conn, b_real, b_conn = find_connected_eps(r_br, b_br, by)

    # GT panels
    vis_r = render_panel(gt_img[:by], r_br[:by], [255, 80, 80], r_jn, r_real, r_conn)
    save_img(vis_r, pd / f'{sid}_gt_r.png')

    vis_b = render_panel(gt_img[:by], b_br[:by], [80, 150, 255], b_jn, b_real, b_conn)
    save_img(vis_b, pd / f'{sid}_gt_b.png')

    # Combined
    vis_c = gt_img[:by].copy().astype(float) * BG_ALPHA
    vis_c[cv2.dilate(r_br[:by].astype(np.uint8), dk4) > 0] = [255, 80, 80]
    vis_c[cv2.dilate(b_br[:by].astype(np.uint8), dk4) > 0] = [80, 150, 255]
    vis_c = vis_c.astype(np.uint8)
    for y, x in r_jn + b_jn: cv2.circle(vis_c, (x, y), 4, (255, 0, 255), -1)
    for y, x in r_real + b_real: cv2.circle(vis_c, (x, y), 4, (255, 255, 0), -1)
    for y, x in r_conn + b_conn: cv2.circle(vis_c, (x, y), 4, (0, 255, 255), -1)
    save_img(vis_c, pd / f'{sid}_gt_comb.png')

    print(f"  GT: R(JN={len(r_jn)} EP_real={len(r_real)} EP_conn={len(r_conn)}) "
          f"B(JN={len(b_jn)} EP_real={len(b_real)} EP_conn={len(b_conn)})")

    # Process each model
    for model in models:
        # Load model output
        if model in ['LBBDM', 'PBBDM']:
            mdir = DATA / 'Multi-color' / model / sid
            if not mdir.exists():
                print(f"  {model}: dir not found"); continue
            img_path = mdir / 'output_0.png'
        else:
            img_path = DATA / 'Multi-color' / model / f'{sid}_fake_B.png'

        if not img_path.exists():
            print(f"  {model}: file not found"); continue

        m_img = np.array(Image.open(img_path))
        if m_img.shape[0] != 512:
            m_img = cv2.resize(m_img, (512, 512))

        # Save model original
        save_img(m_img, pd / f'{sid}_{model}.png')

        # Process
        try:
            mr_br, mb_br, _, _, mby, mh, mw = process_full(m_img)
            mr_jn, mr_ep = skel_jn_ep_full(mr_br, mh, mby)
            mb_jn, mb_ep = skel_jn_ep_full(mb_br, mh, mby)
            mr_real, mr_conn, mb_real, mb_conn = find_connected_eps(mr_br, mb_br, mby)

            # R panel
            vis_mr = render_panel(m_img[:mby], mr_br[:mby], [255, 80, 80], mr_jn, mr_real, mr_conn)
            save_img(vis_mr, pd / f'{sid}_{model}_r.png')

            # B panel
            vis_mb = render_panel(m_img[:mby], mb_br[:mby], [80, 150, 255], mb_jn, mb_real, mb_conn)
            save_img(vis_mb, pd / f'{sid}_{model}_b.png')

            # Combined
            vis_mc = m_img[:mby].copy().astype(float) * BG_ALPHA
            vis_mc[cv2.dilate(mr_br[:mby].astype(np.uint8), dk4) > 0] = [255, 80, 80]
            vis_mc[cv2.dilate(mb_br[:mby].astype(np.uint8), dk4) > 0] = [80, 150, 255]
            vis_mc = vis_mc.astype(np.uint8)
            for y, x in mr_jn + mb_jn: cv2.circle(vis_mc, (x, y), 4, (255, 0, 255), -1)
            for y, x in mr_real + mb_real: cv2.circle(vis_mc, (x, y), 4, (255, 255, 0), -1)
            for y, x in mr_conn + mb_conn: cv2.circle(vis_mc, (x, y), 4, (0, 255, 255), -1)
            save_img(vis_mc, pd / f'{sid}_{model}_comb.png')

            print(f"  {model}: R(JN={len(mr_jn)} EP={len(mr_real)}+{len(mr_conn)}c) "
                  f"B(JN={len(mb_jn)} EP={len(mb_real)}+{len(mb_conn)}c)")
        except Exception as e:
            print(f"  {model}: ERROR {e}")

print("\nAll per-sample panels done!")
