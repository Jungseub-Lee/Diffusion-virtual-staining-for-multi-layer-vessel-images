"""
Generate all panels for paper figure v2.
- Zoomed bridging detail
- Bar chart 1: Single GT vs Multi GT
- Bar chart 2: Multi GT vs all models (LBBDM, PBBDM, LSGAN, Pix2pix, WGANGP)
- GAN quantification on-the-fly
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2, json, glob, os
from PIL import Image
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')
DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')
pd_dir = OUT / 'paper_panels_depth'
pd_dir.mkdir(exist_ok=True)

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dk3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ===== Sample 1 processing for bridging panels =====
s1 = np.array(Image.open(DATA / 'Original color stack data' / 'Sample 1_BF and GT(512x512 two images).png'))
gt_raw = s1[512:]
gt_blur = cv2.GaussianBlur(gt_raw, (7, 7), 0)
h_img, w_img = gt_raw.shape[:2]
by = int(h_img * 0.8)

R = gt_blur[:,:,0].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
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

def skel_endpoints(skel):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    return np.argwhere(nb == 1)

def get_tangent(skel, ep, n=12):
    y, x = ep; visited = {(y, x)}; path = [(y, x)]
    for _ in range(n):
        cy, cx = path[-1]; found = False
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                    if skel[ny, nx] and (ny, nx) not in visited:
                        visited.add((ny, nx)); path.append((ny, nx)); found = True; break
            if found: break
        if not found: break
    if len(path) < 3: return np.array([0.0, 0.0])
    pts = np.array(path); t = pts[0] - pts[-1]
    norm = np.linalg.norm(t)
    return t / norm if norm > 0 else np.array([0.0, 0.0])

def bridge_endpoints(skel, mask, max_gap=40, min_align=0.3):
    eps = skel_endpoints(skel)
    if len(eps) < 2: return skel.copy(), []
    tangents = [get_tangent(skel, ep) for ep in eps]
    bridged = skel.copy(); used = set(); bridges = []
    for i in range(len(eps)):
        if i in used: continue
        best_j, best_score = -1, -1
        for j in range(i + 1, len(eps)):
            if j in used: continue
            d = np.linalg.norm(eps[i].astype(float) - eps[j].astype(float))
            if d > max_gap or d < 3: continue
            direction = (eps[j].astype(float) - eps[i].astype(float))
            direction /= np.linalg.norm(direction)
            score = (abs(np.dot(tangents[i], direction)) + abs(np.dot(tangents[j], -direction))) / 2
            if score > min_align and score > best_score:
                mid_y, mid_x = int((eps[i][0]+eps[j][0])/2), int((eps[i][1]+eps[j][1])/2)
                if 0 <= mid_y < mask.shape[0] and 0 <= mid_x < mask.shape[1] and mask[mid_y, mid_x]:
                    best_j, best_score = j, score
        if best_j >= 0:
            used.add(i); used.add(best_j)
            y0, x0 = eps[i]; y1, x1 = eps[best_j]
            n_pts = max(int(np.linalg.norm(eps[i].astype(float) - eps[best_j].astype(float))) * 2, 20)
            for t in np.linspace(0, 1, n_pts):
                yy, xx = int(y0 + t*(y1-y0)), int(x0 + t*(x1-x0))
                if 0 <= yy < skel.shape[0] and 0 <= xx < skel.shape[1]:
                    bridged[yy, xx] = True
            bridges.append(((y0, x0), (y1, x1)))
    return bridged, bridges

vessel_mask = cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
r_bridged, r_bridges = bridge_endpoints(r_filt, vessel_mask)
b_bridged, b_bridges = bridge_endpoints(b_filt, vessel_mask)
print(f"R bridges: {len(r_bridges)}, B bridges: {len(b_bridges)}")

# ===== Zoomed bridging panels =====
all_bridges = [(b, 'R') for b in r_bridges] + [(b, 'B') for b in b_bridges]
valid_bridges = [((p1,p2),c) for (p1,p2),c in all_bridges if (p1[0]+p2[0])/2 < by]
print(f"Valid bridges for zoom: {len(valid_bridges)}")

zoom_pad = 50
for idx in range(min(4, len(valid_bridges))):
    (p1, p2), color = valid_bridges[idx]
    cy, cx = int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)
    y0, y1 = max(0, cy-zoom_pad), min(by, cy+zoom_pad)
    x0, x1 = max(0, cx-zoom_pad), min(w_img, cx+zoom_pad)
    vis = gt_raw[y0:y1, x0:x1].copy().astype(float) * 0.3
    if color == 'R':
        vis[cv2.dilate(r_filt[y0:y1, x0:x1].astype(np.uint8), dk3) > 0] = [255, 80, 80]
        bridge_only = r_bridged[y0:y1, x0:x1] & ~r_filt[y0:y1, x0:x1]
        vis[cv2.dilate(bridge_only.astype(np.uint8), dk3) > 0] = [255, 255, 100]
    else:
        vis[cv2.dilate(b_filt[y0:y1, x0:x1].astype(np.uint8), dk3) > 0] = [80, 150, 255]
        bridge_only = b_bridged[y0:y1, x0:x1] & ~b_filt[y0:y1, x0:x1]
        vis[cv2.dilate(bridge_only.astype(np.uint8), dk3) > 0] = [255, 255, 100]
    fig_i, ax_i = plt.subplots(figsize=(3, 3))
    ax_i.imshow(vis.astype(np.uint8)); ax_i.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(pd_dir / f'zoom_bridge_{idx+1}.png', dpi=300, bbox_inches='tight',
                pad_inches=0.02, facecolor='black')
    plt.close()
print("Saved zoom bridge panels")

# ===== GAN model quantification =====
def quantify_single_image(img, crop_ratio=0.8):
    """Quantify single-color skeleton: JN, EP, Length (top crop_ratio only)"""
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        g = img
    g = cv2.GaussianBlur(g, (7, 7), 0)
    h_i = g.shape[0]; crop_y = int(h_i * crop_ratio)
    mask = cv2.morphologyEx((g > 10).astype(np.uint8)*255, k3, cv2.MORPH_CLOSE, iterations=2) > 0
    mask = remove_small_objects(mask, min_size=50)
    skel = skeletonize(mask)
    # Crop to top portion
    roi = np.zeros_like(skel); roi[:crop_y] = True
    skel_crop = skel & roi
    su = skel_crop.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    # Junction clustering
    jn_mask = nb >= 3
    jn_lab, jn_n = ndimage.label(jn_mask)
    real_jn = 0
    for i in range(1, jn_n + 1):
        cluster = jn_lab == i
        dilated = cv2.dilate(cluster.astype(np.uint8), k3) > 0
        test_skel = skel_crop.copy()
        test_skel[dilated] = False
        local = np.zeros_like(test_skel)
        py, px = np.where(cluster)
        ymin, ymax = max(0, py.min()-5), min(skel_crop.shape[0], py.max()+6)
        xmin, xmax = max(0, px.min()-5), min(skel_crop.shape[1], px.max()+6)
        local[ymin:ymax, xmin:xmax] = test_skel[ymin:ymax, xmin:xmax]
        _, n_branches = ndimage.label(local)
        if n_branches >= 3:
            real_jn += 1
    # Endpoints
    ep_pts = np.argwhere((nb == 1) & roi)
    # Exclude baseline endpoints (near crop boundary)
    real_ep = sum(1 for y, x in ep_pts if y < crop_y - 3)
    length = int(np.sum(skel_crop))
    return {'junctions': real_jn, 'endpoints': real_ep, 'length': length}

def quantify_multi_image(img, crop_ratio=0.8):
    """Quantify multi-color: R+B channel separation, per-channel skel"""
    img_blur = cv2.GaussianBlur(img, (7, 7), 0)
    h_i = img.shape[0]; crop_y = int(h_i * crop_ratio)
    R_ch = img_blur[:,:,0].astype(np.float32)
    B_ch = img_blur[:,:,2].astype(np.float32)
    gray_ch = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    diff_ch = B_ch - R_ch

    # R skel
    rm = cv2.morphologyEx((R_ch > 30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
    rm = remove_small_objects(rm, min_size=50)
    rs = skeletonize(rm)
    rf = rs & (diff_ch < 0)

    # B skel (hysteresis)
    bs_ = B_ch > 50; bl = B_ch > 20
    blab, bn = ndimage.label(bl)
    bm = np.zeros_like(bl, dtype=bool)
    for i in range(1, bn + 1):
        if np.any(bs_[blab == i]): bm |= (blab == i)
    bm = cv2.morphologyEx(bm.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0
    bm = remove_small_objects(bm, min_size=50)
    bsk = skeletonize(bm)
    bf = bsk & (diff_ch > 0)

    # Bridge
    vm = cv2.morphologyEx((gray_ch > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
    rb, _ = bridge_endpoints(rf, vm)
    bb, _ = bridge_endpoints(bf, vm)

    roi = np.zeros((h_i, img.shape[1]), dtype=bool); roi[:crop_y] = True
    total_jn, total_ep, total_len = 0, 0, 0
    for skel_ch in [rb, bb]:
        sc = skel_ch & roi
        su = sc.astype(np.uint8)
        ker = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
        nb = cv2.filter2D(su, -1, ker) * su
        jm = nb >= 3
        jlab, jn = ndimage.label(jm)
        rjn = 0
        for i in range(1, jn + 1):
            cluster = jlab == i
            dil = cv2.dilate(cluster.astype(np.uint8), k3) > 0
            ts = sc.copy(); ts[dil] = False
            local = np.zeros_like(ts)
            py, px = np.where(cluster)
            ymin, ymax = max(0, py.min()-5), min(h_i, py.max()+6)
            xmin, xmax = max(0, px.min()-5), min(img.shape[1], px.max()+6)
            local[ymin:ymax, xmin:xmax] = ts[ymin:ymax, xmin:xmax]
            _, nbr = ndimage.label(local)
            if nbr >= 3: rjn += 1
        ep_pts = np.argwhere((nb == 1) & roi)
        rep = sum(1 for y, x in ep_pts if y < crop_y - 3)
        total_jn += rjn; total_ep += rep; total_len += int(np.sum(sc))
    return {'junctions': total_jn, 'endpoints': total_ep, 'length': total_len}

# ===== Process all GAN models =====
gan_models = ['LSGAN', 'Pix2pix', 'WGANGP']
gan_results = {}

for model_name in gan_models:
    print(f"\nProcessing {model_name}...")
    single_vals = []
    multi_vals = []

    # Single-color
    single_dir = DATA / 'Single-color' / model_name
    fakes = sorted([str(single_dir / f) for f in os.listdir(single_dir)
                     if 'fake_B' in f and '512' in f])
    for fp in fakes:
        img = np.array(Image.open(fp))
        try:
            q = quantify_single_image(img)
            single_vals.append(q)
        except:
            pass

    # Multi-color
    multi_dir = DATA / 'Multi-color' / model_name
    fakes = sorted([str(multi_dir / f) for f in os.listdir(multi_dir)
                     if 'fake_B' in f and '512' in f])
    for fp in fakes:
        img = np.array(Image.open(fp))
        try:
            q = quantify_multi_image(img)
            multi_vals.append(q)
        except:
            pass

    gan_results[model_name] = {
        'single': single_vals,
        'multi': multi_vals,
    }
    print(f"  Single: {len(single_vals)} samples, Multi: {len(multi_vals)} samples")
    if single_vals:
        sm = {k: np.mean([v[k] for v in single_vals]) for k in ['junctions','endpoints','length']}
        print(f"  Single mean: J={sm['junctions']:.1f} EP={sm['endpoints']:.1f} L={sm['length']:.1f}")
    if multi_vals:
        mm = {k: np.mean([v[k] for v in multi_vals]) for k in ['junctions','endpoints','length']}
        print(f"  Multi mean: J={mm['junctions']:.1f} EP={mm['endpoints']:.1f} L={mm['length']:.1f}")

# Save GAN results
with open(OUT / 'gan_results.json', 'w') as f:
    json.dump(gan_results, f, indent=2)
print("\nSaved gan_results.json")

# ===== Bar Chart 1: Single GT vs Multi GT =====
with open(OUT / 'metric_results_fixed.json') as f:
    metrics = json.load(f)

gt = metrics['GT']
gt_s = {k: gt[k]['single_mean'] for k in ['junctions', 'endpoints', 'length']}
gt_m = {k: gt[k]['multi_mean'] for k in ['junctions', 'endpoints', 'length']}
gt_s_sem = {k: gt[k]['single_sem'] for k in ['junctions', 'endpoints', 'length']}
gt_m_sem = {k: gt[k]['multi_sem'] for k in ['junctions', 'endpoints', 'length']}

fig1, ax1 = plt.subplots(figsize=(5, 3.5))
labels = ['Length', 'Endpoints', 'Junctions']
keys = ['length', 'endpoints', 'junctions']
x = np.arange(len(labels)); wd = 0.32

s_norm = [1.0] * 3
m_norm = [gt_m[k] / gt_s[k] for k in keys]
s_err = [gt_s_sem[k] / gt_s[k] for k in keys]
m_err = [gt_m_sem[k] / gt_s[k] for k in keys]

bars1 = ax1.bar(x - wd/2, s_norm, wd, yerr=s_err, label='Single GT',
                color='#4CAF50', alpha=0.85, capsize=4, edgecolor='white', linewidth=0.5)
bars2 = ax1.bar(x + wd/2, m_norm, wd, yerr=m_err, label='Multi GT',
                color='#FF7043', alpha=0.85, capsize=4, edgecolor='white', linewidth=0.5)

for bars, vals in [(bars1, s_norm), (bars2, m_norm)]:
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax1.axhline(1.0, color='gray', ls='--', alpha=0.4, lw=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel('Normalized (Single GT = 1.0)', fontsize=9)
ax1.set_title('Single GT vs Multi GT (N=215)', fontweight='bold', fontsize=12)
ax1.legend(fontsize=9, loc='upper left')
ax1.set_ylim(0, 2.0); ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(pd_dir / 'chart1_single_vs_multi.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved chart1_single_vs_multi.png")

# ===== Bar Chart 2: Multi GT vs ALL models =====
lbbdm = metrics['LBBDM']
pbbdm = metrics['PBBDM']

# Gather all model data for multi
all_models = {}
all_models['GT'] = {k: gt[k]['multi_mean'] for k in ['junctions','endpoints','length']}
all_models['GT_sem'] = {k: gt[k]['multi_sem'] for k in ['junctions','endpoints','length']}
all_models['LBBDM'] = {k: lbbdm[k]['multi_mean'] for k in ['junctions','endpoints','length']}
all_models['LBBDM_sem'] = {k: lbbdm[k]['multi_sem'] for k in ['junctions','endpoints','length']}
all_models['PBBDM'] = {k: pbbdm[k]['multi_mean'] for k in ['junctions','endpoints','length']}
all_models['PBBDM_sem'] = {k: pbbdm[k]['multi_sem'] for k in ['junctions','endpoints','length']}

for gm in gan_models:
    if gan_results[gm]['multi']:
        vals = gan_results[gm]['multi']
        all_models[gm] = {k: np.mean([v[k] for v in vals]) for k in ['junctions','endpoints','length']}
        all_models[f'{gm}_sem'] = {k: np.std([v[k] for v in vals])/np.sqrt(len(vals)) for k in ['junctions','endpoints','length']}

model_names = ['GT', 'PBBDM', 'LBBDM', 'LSGAN', 'Pix2pix', 'WGANGP']
model_names = [m for m in model_names if m in all_models]
model_colors = {'GT': '#2ecc71', 'PBBDM': '#3498db', 'LBBDM': '#9b59b6',
                'LSGAN': '#e67e22', 'Pix2pix': '#e74c3c', 'WGANGP': '#95a5a6'}

fig2, ax2 = plt.subplots(figsize=(8, 4))
x = np.arange(len(model_names)); wd = 0.22
metric_colors = ['#f39c12', '#e74c3c', '#9b59b6']

for midx, (key, mlabel) in enumerate(zip(keys, labels)):
    gt_ref = all_models['GT'][key]
    vals = [all_models[m][key] / gt_ref for m in model_names]
    errs = [all_models.get(f'{m}_sem', {}).get(key, 0) / gt_ref for m in model_names]
    bars = ax2.bar(x + (midx - 1) * wd, vals, wd, yerr=errs,
                   label=mlabel, color=metric_colors[midx], alpha=0.8, capsize=3,
                   edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

ax2.axhline(1.0, color='gray', ls='--', alpha=0.4, lw=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([f'{m}\n(Multi)' for m in model_names], fontsize=9)
ax2.set_ylabel('Normalized (Multi GT = 1.0)', fontsize=9)
ax2.set_title('Multi-color: GT vs Virtual Staining Models', fontweight='bold', fontsize=12)
ax2.legend(fontsize=8, loc='upper right')
ax2.set_ylim(0, 1.8); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(pd_dir / 'chart2_multi_vs_models.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved chart2_multi_vs_models.png")

# Print summary table
print("\n===== MULTI-COLOR NORMALIZED (GT=1.0) =====")
print(f"{'Model':<12} {'Length':>8} {'EP':>8} {'JN':>8}")
for m in model_names:
    l = all_models[m]['length'] / all_models['GT']['length']
    e = all_models[m]['endpoints'] / all_models['GT']['endpoints']
    j = all_models[m]['junctions'] / all_models['GT']['junctions']
    print(f"{m:<12} {l:>8.2f} {e:>8.2f} {j:>8.2f}")

print("\nAll panels generated!")
