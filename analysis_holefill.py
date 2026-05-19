import numpy as np, json
from pathlib import Path
from PIL import Image
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import binary_fill_holes, label
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')
OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')

def load(p):
    return np.array(Image.open(p).convert('RGB'))

def fill_layer_holes(mask, overall_vessel_mask):
    labeled, n = label(mask)
    filled = mask.copy()
    for i in range(1, n+1):
        component = labeled == i
        component_filled = binary_fill_holes(component)
        new_pixels = component_filled & ~component & overall_vessel_mask
        filled |= new_pixels
    return filled

def separate_layers_holefilled(img, min_area=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s = hsv[:,:,0], hsv[:,:,1]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fg = (gray > 15) & (s > 30)
    overall = gray > 15
    b = ((h <= 12) | (h >= 155)) & fg
    m = ((h >= 13) & (h <= 55)) & fg
    t = ((h >= 85) & (h <= 140)) & fg
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    def c(mask):
        u = mask.astype(np.uint8) * 255
        u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k, iterations=2)
        u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k, iterations=1)
        cleaned = remove_small_objects(u > 0, min_size=min_area)
        return fill_layer_holes(cleaned, overall)
    return c(b), c(m), c(t)

def separate_layers_exclusive(img, min_area=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s = hsv[:,:,0], hsv[:,:,1]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fg = (gray > 15) & (s > 30)
    b = ((h <= 12) | (h >= 155)) & fg
    m = ((h >= 13) & (h <= 55)) & fg
    t = ((h >= 85) & (h <= 140)) & fg
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    def c(mask):
        u = mask.astype(np.uint8) * 255
        u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k, iterations=2)
        u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k, iterations=1)
        return remove_small_objects(u > 0, min_size=min_area)
    return c(b), c(m), c(t)

def get_vessel_mask(img, min_area=50):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v = gray > 15
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    u = v.astype(np.uint8) * 255
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k, iterations=2)
    u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k, iterations=1)
    return remove_small_objects(u > 0, min_size=min_area)

def find_features(skeleton):
    su = skeleton.astype(np.uint8)
    ks = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ks) * su
    return np.argwhere(nb == 1), np.argwhere(nb >= 3)

def count_in_mask(coords, mask):
    if len(coords) == 0:
        return 0
    return sum(1 for y, x in coords if mask[y, x])

def count_true_endpoints_perlayer(skels, masks, vm, radius=15):
    adjacent = [(0,1), (1,2)]
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
    dilated = [cv2.dilate(m.astype(np.uint8)*255, dk, iterations=1) > 0 for m in masks]
    total = 0
    for li in range(3):
        if np.sum(skels[li]) == 0:
            continue
        ep_coords, _ = find_features(skels[li])
        for ey, ex in ep_coords:
            if not vm[ey, ex]:
                continue
            is_boundary = False
            for la, lb_ in adjacent:
                if li == la and dilated[lb_][ey, ex]:
                    is_boundary = True
                    break
                elif li == lb_ and dilated[la][ey, ex]:
                    is_boundary = True
                    break
            if not is_boundary:
                total += 1
    return total

# Quick test on 4 samples
gt_multi_dir = BASE / 'Multi-color' / 'Pix2pix'
samples_test = ['1-5-512', '1-10-512', '1-16-512', '9-15-512']
print('=== Hole-fill effect on sample junctions ===')
for sid in samples_test:
    img = load(gt_multi_dir / f'{sid}_real_B.png')
    h, w = img.shape[:2]
    cutoff = int(h * 0.9)
    vm = np.ones((h, w), dtype=bool)
    vm[cutoff:, :] = False

    old_masks = list(separate_layers_exclusive(img))
    old_skels = [skeletonize(m) for m in old_masks]
    old_j = sum(count_in_mask(find_features(sk)[1], vm) for sk in old_skels)

    new_masks = list(separate_layers_holefilled(img))
    new_skels = [skeletonize(m) for m in new_masks]
    new_j = sum(count_in_mask(find_features(sk)[1], vm) for sk in new_skels)

    filled = sum(np.sum(new_masks[i] & ~old_masks[i]) for i in range(3))
    print(f'  {sid}: OLD_J={old_j} -> NEW_J={new_j} (delta={old_j - new_j}, filled={filled}px)')

# Full analysis
print('\n=== Running full analysis with hole-filling... ===')
gt_single_dir = BASE / 'Single-color' / 'Pix2pix'
all_sets = [
    {f.stem.replace('_real_B', '') for f in gt_multi_dir.glob('*_real_B.png')},
    {f.stem.replace('_real_B', '') for f in gt_single_dir.glob('*_real_B.png')},
    {d.name for d in (BASE / 'Multi-color' / 'LBBDM').iterdir() if d.is_dir()},
    {d.name for d in (BASE / 'Single-color' / 'LBBDM').iterdir() if d.is_dir()},
    {d.name for d in (BASE / 'Multi-color' / 'PBBDM').iterdir() if d.is_dir()},
    {d.name for d in (BASE / 'Single-color' / 'PBBDM').iterdir() if d.is_dir()},
]
common = sorted(set.intersection(*all_sets))
crop_frac = 0.10

results = {src: {'single': [], 'multi': []} for src in ['GT', 'LBBDM', 'PBBDM']}
for idx, sid in enumerate(common):
    if idx % 50 == 0:
        print(f'  {idx}/{len(common)}...')
    gt_m = load(gt_multi_dir / f'{sid}_real_B.png')
    h, w = gt_m.shape[:2]
    cutoff = int(h * (1 - crop_frac))
    vm = np.ones((h, w), dtype=bool)
    vm[cutoff:, :] = False

    paths = {
        'GT': (gt_multi_dir / f'{sid}_real_B.png', gt_single_dir / f'{sid}_real_B.png'),
        'LBBDM': (BASE / 'Multi-color' / 'LBBDM' / sid / 'output_0.png',
                   BASE / 'Single-color' / 'LBBDM' / sid / 'output_0.png'),
        'PBBDM': (BASE / 'Multi-color' / 'PBBDM' / sid / 'output_0.png',
                   BASE / 'Single-color' / 'PBBDM' / sid / 'output_0.png'),
    }
    for src, (mp, sp) in paths.items():
        img_m = load(mp)
        layer_masks = list(separate_layers_holefilled(img_m))
        layer_skels = [skeletonize(m) for m in layer_masks]
        m_j = sum(count_in_mask(find_features(sk)[1], vm) for sk in layer_skels)
        m_l = sum(int(np.sum(sk & vm)) for sk in layer_skels)
        m_e = count_true_endpoints_perlayer(layer_skels, layer_masks, vm)
        m_a = sum(int(np.sum(m & vm)) for m in layer_masks)
        results[src]['multi'].append({
            'sid': sid, 'junctions': m_j, 'endpoints': m_e, 'length': m_l, 'area': m_a
        })

        img_s = load(sp)
        mask_s = get_vessel_mask(img_s)
        skel_s = skeletonize(mask_s)
        ep_s, jn_s = find_features(skel_s)
        results[src]['single'].append({
            'sid': sid,
            'junctions': count_in_mask(jn_s, vm),
            'endpoints': count_in_mask(ep_s, vm),
            'length': int(np.sum(skel_s & vm)),
            'area': int(np.sum(mask_s & vm))
        })

print('\n' + '=' * 80)
print('PANEL C: Single GT vs Multi GT (hole-filled)')
print('=' * 80)
s = results['GT']['single']
m = results['GT']['multi']
for key in ['junctions', 'endpoints', 'length', 'area']:
    sv = [x[key] for x in s]
    mv = [x[key] for x in m]
    t, p = stats.ttest_rel(sv, mv)
    ratio = np.mean(mv) / (np.mean(sv) + 0.001)
    print(f'  {key:>11s}: S={np.mean(sv):.1f}  M={np.mean(mv):.1f}  ratio={ratio:.2f}  p={p:.1e}')

print('\n' + '=' * 80)
print('PANEL D: Model Multi / GT Multi')
print('=' * 80)
gt_m_data = results['GT']['multi']
for src in ['LBBDM', 'PBBDM']:
    print(f'\n--- {src} ---')
    m_data = results[src]['multi']
    for key in ['junctions', 'endpoints', 'length', 'area']:
        gt_vals = [x[key] for x in gt_m_data]
        model_vals = [x[key] for x in m_data]
        ratios = [m_ / g for g, m_ in zip(gt_vals, model_vals) if g > 0]
        ratios = np.array(ratios)
        t_stat, p_val = stats.ttest_1samp(ratios, 1.0)
        _, _, r, _, _ = stats.linregress(gt_vals, model_vals)
        sig = 'ns' if p_val > 0.05 else '*' if p_val > 0.01 else '**' if p_val > 0.001 else '***'
        print(f'  {key:>11s}: ratio={np.mean(ratios):.3f}+-{np.std(ratios)/len(ratios)**0.5:.3f}  p={p_val:.1e}({sig})  R2={r**2:.3f}')

with open(OUT / 'final_results_holefilled_crop10.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved.')
