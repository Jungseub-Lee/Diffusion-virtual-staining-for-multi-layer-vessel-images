"""
Per-layer (R=Bottom, B=Top) scatter plot concordance:
  GT multi vs each model (PBBDM, LBBDM, WGANGP, Pix2pix, LSGAN)
  Using RGB channel-based separation (R>B → bottom, B>R → top)
  With bridging, strict JN, baseline exclusion (bottom 20%)
  Individual scatter PNGs + combined figure
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import cv2, warnings, json
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import label as ndlabel
from scipy import stats, ndimage
warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')
OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')
PANEL_DIR = OUT / 'scatter_panels'
PANEL_DIR.mkdir(exist_ok=True)

def load(p): return np.array(Image.open(p).convert('RGB'))

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def rgb_channel_skel(img, crop_frac=0.20):
    """Extract R-dominant and B-dominant skeletons using RGB channels."""
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    h, w = img.shape[:2]
    by = int(h * (1 - crop_frac))
    region = np.zeros((h, w), dtype=bool); region[:by] = True

    R = blur[:,:,0].astype(np.float32)
    B = blur[:,:,2].astype(np.float32)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    diff = B - R

    # Vessel mask
    vessel_all = remove_small_objects(
        cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0,
        min_size=50)
    vessel_dilated = cv2.dilate(vessel_all.astype(np.uint8)*255, k5) > 0

    # R mask & skeleton
    r_mask = remove_small_objects(
        cv2.morphologyEx((R > 30).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0,
        min_size=30)
    r_skel = skeletonize(r_mask)
    r_filt = r_skel & (diff < 0)  # R dominant

    # B mask & skeleton (hysteresis)
    b_seed = B > 50; b_low = B > 20
    b_ll, nl = ndimage.label(b_low)
    b_hyst = np.zeros_like(b_low, dtype=bool)
    for i in range(1, nl+1):
        reg = b_ll == i
        if np.any(b_seed[reg]): b_hyst |= reg
    b_hyst = remove_small_objects(
        cv2.morphologyEx(b_hyst.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0,
        min_size=30)
    b_skel = skeletonize(b_hyst)
    b_filt = b_skel & (diff > 0)  # B dominant

    # Bridge
    r_br = bridge_skel(r_filt, vessel_dilated, h, w)
    b_br = bridge_skel(b_filt, vessel_dilated, h, w)

    # Strict EP/JN
    r_ep, r_jn, r_L = strict_metrics(r_br, region, h, w, by)
    b_ep, b_jn, b_L = strict_metrics(b_br, region, h, w, by)

    # Area
    r_area = int(np.sum(r_mask & region & (diff < 0)))
    b_area = int(np.sum(b_hyst & region & (diff > 0)))

    return {
        'R': {'area': r_area, 'length': r_L, 'junctions': r_jn, 'endpoints': r_ep},
        'B': {'area': b_area, 'length': b_L, 'junctions': b_jn, 'endpoints': b_ep},
    }

def get_eps(skel, h, w, tl=20):
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

def bridge_skel(skel, vm, h, w):
    eps = get_eps(skel, h, w)
    ct = np.cos(np.radians(60)); cands = []
    for i in range(len(eps)):
        for j in range(i+1,len(eps)):
            p1=np.array(eps[i]['pos'],dtype=float); p2=np.array(eps[j]['pos'],dtype=float)
            d=np.linalg.norm(p2-p1)
            if d<5 or d>80: continue
            d12=(p2-p1)/d; c1=np.dot(eps[i]['tangent'],d12); c2=np.dot(eps[j]['tangent'],-d12)
            if c1<ct or c2<ct: continue
            # Bezier
            t1,t2=eps[i]['tangent'],eps[j]['tangent']
            cp1=p1+t1*d*0.4; cp2=p2+t2*d*0.4; n=max(int(d*1.5),10)
            pts=[]
            for t in np.linspace(0,1,n):
                p=(1-t)**3*p1+3*(1-t)**2*t*cp1+3*(1-t)*t**2*cp2+t**3*p2
                pts.append((int(round(p[0])),int(round(p[1]))))
            ov=sum(1 for(y,x)in pts if 0<=y<h and 0<=x<w and vm[y,x])
            vr=ov/max(len(pts),1)
            if vr<0.7: continue
            sc=(c1+c2)/2-d/80*0.2+vr*0.2
            cands.append((i,j,sc,pts))
    cands.sort(key=lambda x:-x[2]); used=set(); br=skel.copy()
    for i,j,sc,pts in cands:
        if i not in used and j not in used:
            for(y,x)in pts:
                if 0<=y<h and 0<=x<w: br[y,x]=True
            used.add(i); used.add(j)
    return skeletonize(br)

def strict_metrics(skel, region, h, w, by, boundary_margin=5):
    s = skel & region
    su = s.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    length = int(np.sum(s))

    # EP (exclude baseline)
    ep_mask = nb == 1
    ep_labels, n_ep = ndimage.label(ep_mask)
    ep_count = 0
    for i in range(1, n_ep+1):
        pts = np.argwhere(ep_labels == i)
        cy = pts.mean(axis=0)[0]
        if cy < by - boundary_margin:
            ep_count += 1

    # JN (strict: cluster + verify >=3 branches)
    jn_mask = nb >= 3
    jn_dil = cv2.dilate(jn_mask.astype(np.uint8), k3)
    jn_labels, n_raw = ndimage.label(jn_dil)
    jn_count = 0
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
            jn_count += 1

    return ep_count, jn_count, length

# === Find common samples ===
gt_dir = BASE / 'Multi-color' / 'Pix2pix'
models_info = {
    'PBBDM': {'dir': BASE / 'Multi-color' / 'PBBDM', 'type': 'subdir'},
    'LBBDM': {'dir': BASE / 'Multi-color' / 'LBBDM', 'type': 'subdir'},
    'WGANGP': {'dir': BASE / 'Multi-color' / 'WGANGP', 'type': 'flat'},
    'Pix2pix': {'dir': BASE / 'Multi-color' / 'Pix2pix', 'type': 'flat'},
    'LSGAN': {'dir': BASE / 'Multi-color' / 'LSGAN', 'type': 'flat'},
}

gt_sids = {f.stem.replace('_real_B','') for f in gt_dir.glob('*_real_B.png')}
common = sorted(gt_sids)
# Check all models exist
for mname, minfo in models_info.items():
    if minfo['type'] == 'subdir':
        model_sids = {d.name for d in minfo['dir'].iterdir() if d.is_dir()}
    else:
        model_sids = {f.stem.replace('_fake_B','') for f in minfo['dir'].glob('*_fake_B.png')}
    common = sorted(set(common) & model_sids)

print(f'Common samples: {len(common)}')

# === Process all samples ===
data = {src: {'R': {'area':[],'length':[],'junctions':[],'endpoints':[]},
              'B': {'area':[],'length':[],'junctions':[],'endpoints':[]}}
        for src in ['GT'] + list(models_info.keys())}

for idx, sid in enumerate(common):
    if idx % 30 == 0: print(f'  {idx}/{len(common)}...')

    # GT
    gt_img = load(gt_dir / f'{sid}_real_B.png')
    gt_metrics = rgb_channel_skel(gt_img)
    for layer in ['R', 'B']:
        for met in ['area', 'length', 'junctions', 'endpoints']:
            data['GT'][layer][met].append(gt_metrics[layer][met])

    # Models
    for mname, minfo in models_info.items():
        if minfo['type'] == 'subdir':
            mpath = minfo['dir'] / sid / 'output_0.png'
        else:
            mpath = minfo['dir'] / f'{sid}_fake_B.png'

        if not mpath.exists():
            for met in ['area', 'length', 'junctions', 'endpoints']:
                data[mname]['R'][met].append(0)
                data[mname]['B'][met].append(0)
            continue

        mod_img = load(mpath)
        mod_metrics = rgb_channel_skel(mod_img)
        for layer in ['R', 'B']:
            for met in ['area', 'length', 'junctions', 'endpoints']:
                data[mname][layer][met].append(mod_metrics[layer][met])

# Save raw data
json_data = {}
for src in data:
    json_data[src] = {}
    for layer in ['R', 'B']:
        json_data[src][layer] = {met: data[src][layer][met] for met in data[src][layer]}
with open(OUT / 'perlayer_rgb_scatter_data.json', 'w') as f:
    json.dump(json_data, f)
print('Saved perlayer_rgb_scatter_data.json')

# === Generate scatter plots ===
metrics = ['area', 'length', 'junctions', 'endpoints']
met_titles = ['Vessel Area', 'Vessel Length', 'Junctions', 'Endpoints']
model_names = ['PBBDM', 'LBBDM', 'WGANGP', 'Pix2pix', 'LSGAN']
layer_colors = {'R': '#CC4444', 'B': '#4477CC'}
layer_markers = {'R': 'o', 'B': 's'}
layer_labels = {'R': 'Bottom (R>B)', 'B': 'Top (B>R)'}

# Individual scatter plots
for mname in model_names:
    for met, mtitle in zip(metrics, met_titles):
        fig, ax = plt.subplots(figsize=(4, 4))

        r2_texts = []
        for layer in ['R', 'B']:
            gt_v = np.array(data['GT'][layer][met])
            mod_v = np.array(data[mname][layer][met])
            ax.scatter(gt_v, mod_v, c=layer_colors[layer], marker=layer_markers[layer],
                      s=20, alpha=0.35, edgecolors='none', zorder=2, label=layer_labels[layer])

            valid = gt_v > 0
            if np.sum(valid) > 2:
                _, _, r_val, p_val, _ = stats.linregress(gt_v[valid], mod_v[valid])
                r2 = r_val**2
                p_str = 'p<0.001' if p_val < 0.001 else f'p={p_val:.3f}'
                r2_texts.append((layer, r2, p_str))

        # Identity line
        all_vals = np.concatenate([
            data['GT']['R'][met], data['GT']['B'][met],
            data[mname]['R'][met], data[mname]['B'][met]
        ])
        maxval = np.max(all_vals) * 1.08 if len(all_vals) > 0 else 1
        ax.plot([0, maxval], [0, maxval], color='#888888', linewidth=1, linestyle='--', zorder=1)

        # R2 text
        y_text = 0.97
        for layer, r2, p_str in r2_texts:
            ax.text(0.03, y_text, f'{layer_labels[layer][:3]}: R²={r2:.3f}, {p_str}',
                   transform=ax.transAxes, fontsize=8, va='top', ha='left',
                   color=layer_colors[layer], fontweight='bold')
            y_text -= 0.08

        ax.set_xlim(0, maxval)
        ax.set_ylim(0, maxval)
        ax.set_aspect('equal')
        ax.set_xlabel('Ground Truth', fontsize=10)
        ax.set_ylabel(f'{mname} Predicted', fontsize=10)
        ax.set_title(f'{mname} — {mtitle}', fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=8)
        if met == metrics[0]:
            ax.legend(fontsize=7, loc='lower right')

        plt.tight_layout()
        plt.savefig(PANEL_DIR / f'scatter_{mname}_{met}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

# Combined figure (5 models × 4 metrics)
fig, axes = plt.subplots(5, 4, figsize=(16, 20))
fig.suptitle('Per-layer Concordance: GT vs Virtual Staining Models\n'
             '(RGB Channel Separation: R=Bottom, B=Top)',
             fontsize=14, fontweight='bold', y=0.995)

for row, mname in enumerate(model_names):
    for col, (met, mtitle) in enumerate(zip(metrics, met_titles)):
        ax = axes[row, col]

        for layer in ['R', 'B']:
            gt_v = np.array(data['GT'][layer][met])
            mod_v = np.array(data[mname][layer][met])
            ax.scatter(gt_v, mod_v, c=layer_colors[layer], marker=layer_markers[layer],
                      s=15, alpha=0.3, edgecolors='none', zorder=2)

            valid = gt_v > 0
            if np.sum(valid) > 2:
                _, _, r_val, p_val, _ = stats.linregress(gt_v[valid], mod_v[valid])
                r2 = r_val**2
                p_str = 'p<0.001' if p_val < 0.001 else f'p={p_val:.3f}'
                color = layer_colors[layer]
                y_off = 0.97 if layer == 'R' else 0.89
                ax.text(0.03, y_off, f'{layer}: R²={r2:.3f}',
                       transform=ax.transAxes, fontsize=7, va='top', ha='left',
                       color=color, fontweight='bold')

        all_vals = np.concatenate([
            data['GT']['R'][met], data['GT']['B'][met],
            data[mname]['R'][met], data[mname]['B'][met]
        ])
        maxval = np.max(all_vals) * 1.08 if len(all_vals) > 0 else 1
        ax.plot([0, maxval], [0, maxval], color='#888888', linewidth=1, linestyle='--', zorder=1)
        ax.set_xlim(0, maxval); ax.set_ylim(0, maxval)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

        if row == 0: ax.set_title(mtitle, fontsize=11, fontweight='bold')
        if row == 4: ax.set_xlabel('Ground Truth', fontsize=9)
        if col == 0: ax.set_ylabel(f'{mname}\nPredicted', fontsize=9, fontweight='bold')

legend_elements = [
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#CC4444', markersize=8, label='Bottom (R>B)'),
    plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='#4477CC', markersize=8, label='Top (B>R)'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, 0.975), frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.965])
plt.savefig(OUT / 'Figure_perlayer_concordance_rgb.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print(f'\nSaved {len(model_names)*len(metrics)} individual scatter plots')
print('Saved Figure_perlayer_concordance_rgb.png')
print('Done!')
