import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import cv2, warnings
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import binary_fill_holes, label
from scipy import stats
warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')
OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')

def load(p): return np.array(Image.open(p).convert('RGB'))
def fill_layer_holes(mask, overall):
    labeled, n = label(mask)
    filled = mask.copy()
    for i in range(1, n+1):
        comp = labeled == i
        filled |= binary_fill_holes(comp) & ~comp & overall
    return filled
def separate_layers_hf(img, min_area=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s = hsv[:,:,0], hsv[:,:,1]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fg = (gray > 15) & (s > 30)
    ov = gray > 15
    b = ((h <= 12) | (h >= 155)) & fg
    m = ((h >= 13) & (h <= 55)) & fg
    t = ((h >= 85) & (h <= 140)) & fg
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    def c(mask):
        u = cv2.morphologyEx(mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k, iterations=2)
        u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k, iterations=1)
        return fill_layer_holes(remove_small_objects(u > 0, min_size=min_area), ov)
    return c(b), c(m), c(t)
def find_features(skel):
    su = skel.astype(np.uint8)
    ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ks) * su
    return np.argwhere(nb == 1), np.argwhere(nb >= 3)
def count_in_mask(coords, mask):
    return sum(1 for y,x in coords if mask[y,x]) if len(coords) else 0

gt_multi_dir = BASE / 'Multi-color' / 'Pix2pix'
all_sets = [
    {f.stem.replace('_real_B','') for f in gt_multi_dir.glob('*_real_B.png')},
    {d.name for d in (BASE/'Multi-color'/'LBBDM').iterdir() if d.is_dir()},
    {d.name for d in (BASE/'Multi-color'/'PBBDM').iterdir() if d.is_dir()},
]
common = sorted(set.intersection(*all_sets))
crop_frac = 0.20
layer_names = ['Bottom (Red)', 'Middle (Yellow)', 'Top (Blue)']
layer_colors_plot = ['#CC4444', '#CCAA22', '#4477CC']

print(f'Processing {len(common)} samples...')

# Collect per-layer data: GT vs LBBDM vs PBBDM
# For each sample and each layer: area, length, junctions, endpoints
data = {src: {li: {'area':[], 'length':[], 'junctions':[], 'endpoints':[]} for li in range(3)}
        for src in ['GT', 'LBBDM', 'PBBDM']}

for idx, sid in enumerate(common):
    if idx % 50 == 0:
        print(f'  {idx}/{len(common)}...')
    gt_img = load(gt_multi_dir / f'{sid}_real_B.png')
    h, w = gt_img.shape[:2]
    cut = int(h * (1 - crop_frac))
    vm = np.ones((h, w), dtype=bool)
    vm[cut:, :] = False

    paths = {
        'GT': gt_multi_dir / f'{sid}_real_B.png',
        'LBBDM': BASE / 'Multi-color' / 'LBBDM' / sid / 'output_0.png',
        'PBBDM': BASE / 'Multi-color' / 'PBBDM' / sid / 'output_0.png',
    }

    for src, mp in paths.items():
        img_m = load(mp)
        layers = list(separate_layers_hf(img_m))
        for li in range(3):
            skel = skeletonize(layers[li])
            ep, jn = find_features(skel)
            data[src][li]['area'].append(int(np.sum(layers[li] & vm)))
            data[src][li]['length'].append(int(np.sum(skel & vm)))
            data[src][li]['junctions'].append(count_in_mask(jn, vm))
            data[src][li]['endpoints'].append(count_in_mask(ep, vm))

print('Data collected. Generating scatter plots...')

metrics = ['area', 'length', 'junctions', 'endpoints']
metric_labels = ['Vessel Area (px)', 'Skeleton Length (px)', 'Junctions', 'Endpoints']

for model_src in ['LBBDM', 'PBBDM']:
    # Version 1: R/B only (2 layers)
    for version, layers_to_use, suffix in [
        ('R+B', [0, 2], 'rb'),
        ('R+Y+B', [0, 1, 2], 'ryb'),
    ]:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'{model_src} vs GT — Per-layer concordance ({version})', fontsize=14, fontweight='bold')

        for mi, (met, met_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[mi]
            all_gt, all_model = [], []

            for li in layers_to_use:
                gt_vals = np.array(data['GT'][li][met])
                model_vals = np.array(data[model_src][li][met])
                ax.scatter(gt_vals, model_vals, c=layer_colors_plot[li],
                          s=15, alpha=0.4, label=layer_names[li], edgecolors='none')
                all_gt.extend(gt_vals.tolist())
                all_model.extend(model_vals.tolist())

            # Overall R2
            all_gt = np.array(all_gt)
            all_model = np.array(all_model)
            mask_valid = all_gt > 0
            if np.sum(mask_valid) > 2:
                slope, intercept, r, p, stderr = stats.linregress(all_gt[mask_valid], all_model[mask_valid])
                r2 = r ** 2
                # Fit line
                xfit = np.linspace(0, np.max(all_gt) * 1.05, 100)
                ax.plot(xfit, slope * xfit + intercept, 'k--', linewidth=1, alpha=0.6)
            else:
                r2 = 0

            # Per-layer R2
            r2_texts = []
            for li in layers_to_use:
                gt_v = np.array(data['GT'][li][met])
                mod_v = np.array(data[model_src][li][met])
                valid = gt_v > 0
                if np.sum(valid) > 2:
                    _, _, r_li, _, _ = stats.linregress(gt_v[valid], mod_v[valid])
                    r2_texts.append(f'{layer_names[li][:3]}: R\u00b2={r_li**2:.3f}')

            # Identity line
            maxval = max(np.max(all_gt), np.max(all_model)) * 1.05 if len(all_gt) > 0 else 1
            ax.plot([0, maxval], [0, maxval], 'gray', linewidth=0.8, alpha=0.5, linestyle=':')

            ax.set_xlabel(f'GT {met_label}', fontsize=9)
            ax.set_ylabel(f'{model_src} {met_label}', fontsize=9)
            ax.set_title(f'{met_label}\nAll R\u00b2={r2:.3f}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=7, loc='upper left')
            ax.set_xlim(0, maxval)
            ax.set_ylim(0, maxval)
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Per-layer R2 text
            for ti, txt in enumerate(r2_texts):
                ax.text(0.98, 0.15 - ti * 0.07, txt, transform=ax.transAxes,
                       fontsize=7, ha='right', va='top', color=layer_colors_plot[layers_to_use[ti]])

        plt.tight_layout()
        fname = f'scatter_perlayer_{model_src}_{suffix}.png'
        plt.savefig(str(OUT / fname), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'  Saved {fname}')

print('\nAll scatter plots generated.')
