"""
v3 panels: all fixes applied
1. (m) bridging before/after/final separate images
2. (n) zoom: before/after pairs, no text on image, no arrows, 3R+3B
3. (j)(k)(l) style panels for 5 extra samples
4. Charts: no value labels, GT removed from chart b
5. Background brightness increased for skel overlay
6. Bridge overview without arrows (for crop source)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2, json, os
from PIL import Image
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')
DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')
pd = OUT / 'paper_panels_depth'
pd.mkdir(exist_ok=True)

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dk3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

BG_ALPHA = 0.45  # brighter background (was 0.2-0.25)

# ===== Core functions =====
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
            results.append({'pos': (y0, x0), 'tangent': tangent, 'path': path})
    return results

def _bridge_pts(ep1, ep2):
    p1 = np.array(ep1['pos'], dtype=float)
    p2 = np.array(ep2['pos'], dtype=float)
    t1, t2 = ep1['tangent'], ep2['tangent']
    dist = np.linalg.norm(p2 - p1)
    ctrl1 = p1 + t1 * dist * 0.4
    ctrl2 = p2 + t2 * dist * 0.4
    n_pts = max(int(dist * 1.5), 10)
    pts = []
    for t in np.linspace(0, 1, n_pts):
        p = (1-t)**3*p1 + 3*(1-t)**2*t*ctrl1 + 3*(1-t)*t**2*ctrl2 + t**3*p2
        pts.append((int(round(p[0])), int(round(p[1]))))
    return pts

def match_and_bridge(skel, vessel_mask, h, w, max_dist=80, max_angle_deg=60):
    eps = get_endpoints_with_tangent(skel, h, w, trace_len=20)
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
            thresh = np.cos(np.radians(max_angle_deg))
            if cos1 < thresh or cos2 < thresh: continue
            pts = _bridge_pts(eps[i], eps[j])
            on_v = sum(1 for (y, x) in pts if 0 <= y < h and 0 <= x < w and vessel_mask[y, x])
            vr = on_v / max(len(pts), 1)
            if vr < 0.7: continue
            score = (cos1 + cos2) / 2 - dist / max_dist * 0.2 + vr * 0.2
            candidates.append((i, j, score, dist, vr))
    candidates.sort(key=lambda x: -x[2])
    used = set(); matches = []
    for i, j, sc, d, vr in candidates:
        if i not in used and j not in used:
            matches.append((i, j, sc, d, vr)); used.add(i); used.add(j)
    bridged = skel.copy(); bridges = []
    for i, j, sc, d, vr in matches:
        pts = _bridge_pts(eps[i], eps[j])
        bridges.append((pts, sc, d, vr, eps[i], eps[j]))
        for (y, x) in pts:
            if 0 <= y < h and 0 <= x < w: bridged[y, x] = True
    return bridged, bridges, eps

def process_image(img_rgb):
    """Full pipeline: R/B skel + dominance filter + bridge"""
    img_blur = cv2.GaussianBlur(img_rgb, (7, 7), 0)
    h, w = img_rgb.shape[:2]
    by = int(h * 0.8)
    R = img_blur[:,:,0].astype(np.float32)
    B = img_blur[:,:,2].astype(np.float32)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    diff = B - R

    r_mask = cv2.morphologyEx((R > 30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
    r_mask = remove_small_objects(r_mask, min_size=50)
    r_skel = skeletonize(r_mask)

    b_seed = B > 50; b_low = B > 20
    b_lab, b_n = ndimage.label(b_low)
    b_mask = np.zeros_like(b_low, dtype=bool)
    for i in range(1, b_n + 1):
        if np.any(b_seed[b_lab == i]): b_mask |= (b_lab == i)
    b_mask = cv2.morphologyEx(b_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0
    b_mask = remove_small_objects(b_mask, min_size=50)
    b_skel = skeletonize(b_mask)

    r_filt = r_skel & (diff < 0)
    b_filt = b_skel & (diff > 0)
    # No fragment removal - keep all skeleton segments

    vessel_all = remove_small_objects(
        cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0, min_size=100)
    vm = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

    r_bridged, r_bridges, r_eps = match_and_bridge(r_filt, vm, h, w)
    b_bridged, b_bridges, b_eps = match_and_bridge(b_filt, vm, h, w)

    return {
        'r_filt': r_filt, 'b_filt': b_filt,
        'r_bridged': r_bridged, 'b_bridged': b_bridged,
        'r_bridges': r_bridges, 'b_bridges': b_bridges,
        'r_eps': r_eps, 'b_eps': b_eps,
        'by': by, 'h': h, 'w': w
    }

def skel_jn_ep(skel, h, crop_y):
    """Count real JN and EP (excluding baseline)"""
    roi = np.zeros_like(skel, dtype=bool); roi[:crop_y] = True
    sc = skel & roi
    su = sc.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    # JN clustering
    jm = nb >= 3
    jlab, jn = ndimage.label(jm)
    real_jn = 0
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
            real_jn += 1
            jn_pts.append((int(py.mean()), int(px.mean())))
    # EP
    ep_all = np.argwhere((nb == 1) & roi)
    ep_pts = [(y, x) for y, x in ep_all if y < crop_y - 3]
    return real_jn, jn_pts, len(ep_pts), ep_pts

EXTRA_BG_ALPHA = 0.7
EXTRA_WHITE_VEIL = 0.4

def render_skel_panel(img_rgb, skel, color, jn_pts, ep_pts, alpha=BG_ALPHA):
    """Render skeleton on GT image with JN (magenta) and EP (yellow)"""
    vis = img_rgb.copy().astype(float) * alpha
    vis[cv2.dilate(skel.astype(np.uint8), dk4) > 0] = color
    vis = vis.astype(np.uint8)
    for y, x in jn_pts:
        cv2.circle(vis, (x, y), 4, (255, 0, 255), -1)
    for y, x in ep_pts:
        cv2.circle(vis, (x, y), 4, (255, 255, 0), -1)
    return vis

def render_skel_panel_bright(img_rgb, skel, color, jn_pts, ep_pts):
    """Render skeleton on brighter GT + white veil"""
    vis = img_rgb.copy().astype(float) * EXTRA_BG_ALPHA
    vis = vis * (1 - EXTRA_WHITE_VEIL) + 255 * EXTRA_WHITE_VEIL
    vis[cv2.dilate(skel.astype(np.uint8), dk4) > 0] = color
    vis = vis.astype(np.uint8)
    for y, x in jn_pts:
        cv2.circle(vis, (x, y), 5, (255, 0, 255), 2)
    for y, x in ep_pts:
        cv2.circle(vis, (x, y), 5, (255, 255, 0), 2)
    return vis

def save_panel(img, path, figsize=(4, 3.3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img); ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='black')
    plt.close()

# =====================================================
# SAMPLE 1 processing
# =====================================================
s1 = np.array(Image.open(DATA / 'Original color stack data' / 'Sample 1_BF and GT(512x512 two images).png'))
gt_raw = s1[512:]
res = process_image(gt_raw)
h_img, w_img = res['h'], res['w']
by = res['by']

# ===== Fix 6: Brighter background for j, k, l panels =====
# Re-generate j, k, l with higher alpha
for tag, skel, color in [
    ('j_r_br', res['r_bridged'], [255, 80, 80]),
    ('k_b_br', res['b_bridged'], [80, 150, 255]),
]:
    jn_n, jn_pts, ep_n, ep_pts = skel_jn_ep(skel, h_img, by)
    vis = render_skel_panel(gt_raw[:by], skel[:by], color, jn_pts, ep_pts)
    save_panel(vis, pd / f'{tag}.png')
    print(f"  {tag}: JN={jn_n} EP={ep_n}")

# l_comb: R+B combined
vis_comb = gt_raw[:by].copy().astype(float) * BG_ALPHA
vis_comb[cv2.dilate(res['r_bridged'][:by].astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis_comb[cv2.dilate(res['b_bridged'][:by].astype(np.uint8), dk4) > 0] = [80, 150, 255]
vis_comb = vis_comb.astype(np.uint8)
r_jn_n, r_jn_pts, r_ep_n, r_ep_pts = skel_jn_ep(res['r_bridged'], h_img, by)
b_jn_n, b_jn_pts, b_ep_n, b_ep_pts = skel_jn_ep(res['b_bridged'], h_img, by)
for y, x in r_jn_pts + b_jn_pts:
    cv2.circle(vis_comb, (x, y), 4, (255, 0, 255), -1)
for y, x in r_ep_pts + b_ep_pts:
    cv2.circle(vis_comb, (x, y), 4, (255, 255, 0), -1)
save_panel(vis_comb, pd / 'l_comb.png')
print("Saved j, k, l panels (brighter)")

# ===== Fix 1: (m) bridging before / after / final as 3 separate images =====
# Before
vis_before = gt_raw[:by].copy().astype(float) * BG_ALPHA
vis_before[cv2.dilate(res['r_filt'][:by].astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis_before[cv2.dilate(res['b_filt'][:by].astype(np.uint8), dk4) > 0] = [80, 150, 255]
vis_before = vis_before.astype(np.uint8)
for ep in res['r_eps']:
    y, x = ep['pos']
    if y < by:
        cv2.circle(vis_before, (x, y), 3, (255, 255, 0), -1)
        ty, tx = ep['tangent']
        cv2.arrowedLine(vis_before, (x, y), (int(x+tx*12), int(y+ty*12)), (255, 255, 0), 1, tipLength=0.3)
for ep in res['b_eps']:
    y, x = ep['pos']
    if y < by:
        cv2.circle(vis_before, (x, y), 3, (0, 255, 255), -1)
        ty, tx = ep['tangent']
        cv2.arrowedLine(vis_before, (x, y), (int(x+tx*12), int(y+ty*12)), (0, 255, 255), 1, tipLength=0.3)
save_panel(vis_before, pd / 'm_before.png')

# After
vis_after = gt_raw[:by].copy().astype(float) * BG_ALPHA
vis_after[cv2.dilate(res['r_bridged'][:by].astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis_after[cv2.dilate(res['b_bridged'][:by].astype(np.uint8), dk4) > 0] = [80, 150, 255]
vis_after = vis_after.astype(np.uint8)
for pts, sc, d, vr, ep1, ep2 in res['r_bridges']:
    for (y, x) in pts:
        if 0 <= y < by and 0 <= x < w_img:
            cv2.circle(vis_after, (x, y), 2, (255, 200, 50), -1)
for pts, sc, d, vr, ep1, ep2 in res['b_bridges']:
    for (y, x) in pts:
        if 0 <= y < by and 0 <= x < w_img:
            cv2.circle(vis_after, (x, y), 2, (50, 255, 200), -1)
save_panel(vis_after, pd / 'm_after.png')

# Final (clean, no arrows/dots)
vis_final = gt_raw[:by].copy().astype(float) * BG_ALPHA
vis_final[cv2.dilate(res['r_bridged'][:by].astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis_final[cv2.dilate(res['b_bridged'][:by].astype(np.uint8), dk4) > 0] = [80, 150, 255]
save_panel(vis_final.astype(np.uint8), pd / 'm_final.png')
print("Saved m_before, m_after, m_final")

# ===== Fix 2: (n) zoom before/after crops — no text, no arrows =====
# Source: same as m_before and m_after but without arrows for crops
vis_before_clean = gt_raw[:by].copy().astype(float) * BG_ALPHA
vis_before_clean[cv2.dilate(res['r_filt'][:by].astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis_before_clean[cv2.dilate(res['b_filt'][:by].astype(np.uint8), dk4) > 0] = [80, 150, 255]
vis_before_clean = vis_before_clean.astype(np.uint8)

vis_after_clean = gt_raw[:by].copy().astype(float) * BG_ALPHA
vis_after_clean[cv2.dilate(res['r_bridged'][:by].astype(np.uint8), dk4) > 0] = [255, 80, 80]
vis_after_clean[cv2.dilate(res['b_bridged'][:by].astype(np.uint8), dk4) > 0] = [80, 150, 255]
vis_after_clean = vis_after_clean.astype(np.uint8)
# Mark bridge paths on after image
for pts, *_ in res['r_bridges']:
    for (y, x) in pts:
        if 0 <= y < by and 0 <= x < w_img:
            cv2.circle(vis_after_clean, (x, y), 2, (0, 255, 0), -1)
for pts, *_ in res['b_bridges']:
    for (y, x) in pts:
        if 0 <= y < by and 0 <= x < w_img:
            cv2.circle(vis_after_clean, (x, y), 2, (0, 255, 0), -1)

# Select 3 R bridges + 3 B bridges (longest distance first)
all_bridges = [(br, 'R') for br in res['r_bridges']] + [(br, 'B') for br in res['b_bridges']]
valid_br = [((pts, sc, d, vr, ep1, ep2), c) for (pts, sc, d, vr, ep1, ep2), c in all_bridges
            if (ep1['pos'][0] + ep2['pos'][0]) / 2 < by]
r_br = sorted([(br, c) for br, c in valid_br if c == 'R'], key=lambda x: -x[0][2])
b_br = sorted([(br, c) for br, c in valid_br if c == 'B'], key=lambda x: -x[0][2])

selected = r_br[:3] + b_br[:3]
zoom_pad = 50

for idx, ((pts, sc, dist, vr, ep1, ep2), color) in enumerate(selected):
    cy = int((ep1['pos'][0] + ep2['pos'][0]) / 2)
    cx = int((ep1['pos'][1] + ep2['pos'][1]) / 2)
    y0 = max(0, cy - zoom_pad); y1 = min(by, cy + zoom_pad)
    x0 = max(0, cx - zoom_pad); x1 = min(w_img, cx + zoom_pad)

    crop_before = vis_before_clean[y0:y1, x0:x1].copy()
    crop_after = vis_after_clean[y0:y1, x0:x1].copy()

    save_panel(crop_before, pd / f'n_zoom_{idx+1}_before.png', figsize=(2.5, 2.5))
    save_panel(crop_after, pd / f'n_zoom_{idx+1}_after.png', figsize=(2.5, 2.5))

print(f"Saved {len(selected)} zoom before/after pairs")

# ===== Fix 4: Charts without value labels, chart b without GT =====
with open(OUT / 'metric_results_fixed.json') as f:
    metrics = json.load(f)

gt = metrics['GT']
gt_s = {k: gt[k]['single_mean'] for k in ['junctions', 'endpoints', 'length']}
gt_m = {k: gt[k]['multi_mean'] for k in ['junctions', 'endpoints', 'length']}
gt_s_sem = {k: gt[k]['single_sem'] for k in ['junctions', 'endpoints', 'length']}
gt_m_sem = {k: gt[k]['multi_sem'] for k in ['junctions', 'endpoints', 'length']}

keys = ['length', 'endpoints', 'junctions']
labels = ['Length', 'Endpoints', 'Junctions']

# Chart 1: Single vs Multi GT (no value labels)
fig1, ax1 = plt.subplots(figsize=(5, 3.5))
x = np.arange(len(labels)); wd = 0.32
s_norm = [1.0] * 3
m_norm = [gt_m[k] / gt_s[k] for k in keys]
s_err = [gt_s_sem[k] / gt_s[k] for k in keys]
m_err = [gt_m_sem[k] / gt_s[k] for k in keys]

ax1.bar(x - wd/2, s_norm, wd, yerr=s_err, label='Single GT',
        color='#4CAF50', alpha=0.85, capsize=4, edgecolor='white', linewidth=0.5)
ax1.bar(x + wd/2, m_norm, wd, yerr=m_err, label='Multi GT',
        color='#FF7043', alpha=0.85, capsize=4, edgecolor='white', linewidth=0.5)
ax1.axhline(1.0, color='gray', ls='--', alpha=0.4, lw=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel('Normalized (Single GT = 1.0)', fontsize=9)
ax1.set_title('Single GT vs Multi GT (N=215, top 80%)', fontweight='bold', fontsize=12)
ax1.legend(fontsize=9, loc='upper left')
ax1.set_ylim(0, 1.8); ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(pd / 'chart1_single_vs_multi.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Chart 2: Multi models (WITHOUT GT, normalized to Multi GT=1.0)
lbbdm = metrics['LBBDM']; pbbdm = metrics['PBBDM']
# Load GAN results
with open(OUT / 'gan_results.json') as f:
    gan_res = json.load(f)

all_models = {}
for name, src in [('LBBDM', lbbdm), ('PBBDM', pbbdm)]:
    all_models[name] = {k: src[k]['multi_mean'] for k in keys}
    all_models[f'{name}_sem'] = {k: src[k]['multi_sem'] for k in keys}
for gm in ['LSGAN', 'Pix2pix', 'WGANGP']:
    if gan_res.get(gm, {}).get('multi'):
        vals = gan_res[gm]['multi']
        all_models[gm] = {k: np.mean([v[k] for v in vals]) for k in keys}
        all_models[f'{gm}_sem'] = {k: np.std([v[k] for v in vals])/np.sqrt(len(vals)) for k in keys}

model_order = ['PBBDM', 'LBBDM', 'LSGAN', 'Pix2pix', 'WGANGP']
model_order = [m for m in model_order if m in all_models]
metric_colors = ['#f39c12', '#e74c3c', '#9b59b6']

fig2, ax2 = plt.subplots(figsize=(6.5, 3.5))
x2 = np.arange(len(model_order)); wd2 = 0.22
for midx, (key, mlabel) in enumerate(zip(keys, labels)):
    gt_ref = gt_m[key]
    vals = [all_models[m][key] / gt_ref for m in model_order]
    errs = [all_models.get(f'{m}_sem', {}).get(key, 0) / gt_ref for m in model_order]
    ax2.bar(x2 + (midx - 1) * wd2, vals, wd2, yerr=errs,
            label=mlabel, color=metric_colors[midx], alpha=0.8, capsize=3,
            edgecolor='white', linewidth=0.5)
ax2.axhline(1.0, color='gray', ls='--', alpha=0.4, lw=0.8)
ax2.set_xticks(x2)
ax2.set_xticklabels([f'{m}\n(Multi)' for m in model_order], fontsize=9)
ax2.set_ylabel('Normalized (Multi GT = 1.0)', fontsize=9)
ax2.set_title('Virtual Staining Models vs Multi GT', fontweight='bold', fontsize=12)
ax2.legend(fontsize=8, loc='upper right')
ax2.set_ylim(0, 1.5); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(pd / 'chart2_multi_vs_models.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved charts (no value labels, chart2 without GT)")

# ===== Fix 5: 5 extra samples =====
extra_sids = ['1-19-716', '9-15-512', '16-18-716', '18-19-512', '1-16-512']
multi_dir = DATA / 'Multi-color'
single_dir = DATA / 'Single-color'

# Find LBBDM/PBBDM multi images for these sample IDs
for model_dir_name in ['LBBDM', 'PBBDM']:
    model_path = multi_dir / model_dir_name
    if not model_path.exists():
        continue
    for sid in extra_sids:
        fname = f'{sid}_real_B.png'
        fpath = model_path / fname
        if not fpath.exists():
            # Try GT directly
            continue

# Actually, use GT multi-color images from the model directories
# real_B = GT multi-color
print("\n===== Processing 5 extra samples =====")
for sid in extra_sids:
    # GT multi-color is in GAN directories as real_B
    gt_path = None
    for model_name in ['LSGAN', 'Pix2pix', 'WGANGP']:
        candidate = multi_dir / model_name / f'{sid}_real_B.png'
        if candidate.exists():
            gt_path = candidate; break
    if gt_path is None:
        print(f"  {sid}: GT not found, skipping")
        continue

    gt_img = np.array(Image.open(gt_path))
    if gt_img.ndim == 2:
        print(f"  {sid}: grayscale, skipping")
        continue

    # Resize if needed
    if gt_img.shape[0] != 512:
        gt_img = cv2.resize(gt_img, (512, 512))

    print(f"  {sid}: shape={gt_img.shape}")
    r = process_image(gt_img)
    h_s, w_s, by_s = r['h'], r['w'], r['by']

    # Skel R panel (bright GT + white veil)
    rjn, rjp, rep_n, rept = skel_jn_ep(r['r_bridged'], h_s, by_s)
    vis_r = render_skel_panel_bright(gt_img[:by_s], r['r_bridged'][:by_s], [255, 80, 80], rjp, rept)
    save_panel(vis_r, pd / f'extra_{sid}_r.png')

    # Skel B panel (bright GT + white veil)
    bjn, bjp, bep_n, bept = skel_jn_ep(r['b_bridged'], h_s, by_s)
    vis_b = render_skel_panel_bright(gt_img[:by_s], r['b_bridged'][:by_s], [80, 150, 255], bjp, bept)
    save_panel(vis_b, pd / f'extra_{sid}_b.png')

    # Combined R+B with connected EP detection
    dk5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    r_dil_m = cv2.dilate(r['r_bridged'][:by_s].astype(np.uint8), dk5) > 0
    b_dil_m = cv2.dilate(r['b_bridged'][:by_s].astype(np.uint8), dk5) > 0
    r_dil_sk = cv2.dilate(r['r_bridged'][:by_s].astype(np.uint8), dk4) > 0
    b_dil_sk = cv2.dilate(r['b_bridged'][:by_s].astype(np.uint8), dk4) > 0

    # Classify EP: real vs connected (where one skel meets the other)
    r_real_ep = [(y, x) for y, x in rept if not b_dil_m[y, x]]
    r_conn_ep = [(y, x) for y, x in rept if b_dil_m[y, x]]
    b_real_ep = [(y, x) for y, x in bept if not r_dil_m[y, x]]
    b_conn_ep = [(y, x) for y, x in bept if r_dil_m[y, x]]

    # Build combined vis with bright GT + white veil
    vis_c = gt_img[:by_s].copy().astype(float) * EXTRA_BG_ALPHA
    vis_c = vis_c * (1 - EXTRA_WHITE_VEIL) + 255 * EXTRA_WHITE_VEIL
    r_only = r_dil_sk & ~b_dil_sk
    b_only = b_dil_sk & ~r_dil_sk
    both = r_dil_sk & b_dil_sk
    vis_c[r_only] = [255, 80, 80]
    vis_c[b_only] = [80, 150, 255]
    vis_c[both] = [180, 100, 220]  # purple overlap
    vis_c = vis_c.astype(np.uint8)

    # Draw markers: JN=magenta, real EP=yellow, connected EP=cyan
    for y, x in rjp + bjp:
        cv2.circle(vis_c, (x, y), 5, (255, 0, 255), 2)
    for y, x in r_real_ep + b_real_ep:
        cv2.circle(vis_c, (x, y), 5, (255, 255, 0), 2)
    for y, x in r_conn_ep + b_conn_ep:
        cv2.circle(vis_c, (x, y), 5, (0, 255, 255), 2)
    save_panel(vis_c, pd / f'extra_{sid}_comb.png')

    print(f"    R: JN={rjn} EP={rep_n}(real={len(r_real_ep)} conn={len(r_conn_ep)})")
    print(f"    B: JN={bjn} EP={bep_n}(real={len(b_real_ep)} conn={len(b_conn_ep)})")

print("\nAll v3 panels generated!")
