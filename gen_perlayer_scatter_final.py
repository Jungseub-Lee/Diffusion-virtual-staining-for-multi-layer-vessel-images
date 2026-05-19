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
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV); h,s=hsv[:,:,0],hsv[:,:,1]
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY); fg=(gray>15)&(s>30); ov=gray>15
    b=((h<=12)|(h>=155))&fg; m=((h>=13)&(h<=55))&fg; t=((h>=85)&(h<=140))&fg
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    def c(mask):
        u=cv2.morphologyEx(mask.astype(np.uint8)*255,cv2.MORPH_CLOSE,k,iterations=2)
        u=cv2.morphologyEx(u,cv2.MORPH_OPEN,k,iterations=1)
        return fill_layer_holes(remove_small_objects(u>0,min_size=min_area),ov)
    return c(b),c(m),c(t)
def find_features(skel):
    su=skel.astype(np.uint8)
    ks=np.array([[1,1,1],[1,0,1],[1,1,1]],dtype=np.uint8)
    nb=cv2.filter2D(su,-1,ks)*su
    return np.argwhere(nb==1),np.argwhere(nb>=3)
def count_in_mask(coords,mask):
    return sum(1 for y,x in coords if mask[y,x]) if len(coords) else 0

gt_multi_dir=BASE/'Multi-color'/'Pix2pix'
all_sets=[
    {f.stem.replace('_real_B','') for f in gt_multi_dir.glob('*_real_B.png')},
    {d.name for d in (BASE/'Multi-color'/'LBBDM').iterdir() if d.is_dir()},
    {d.name for d in (BASE/'Multi-color'/'PBBDM').iterdir() if d.is_dir()},
]
common=sorted(set.intersection(*all_sets))
crop_frac=0.20

data={src:{li:{'area':[],'length':[],'junctions':[],'endpoints':[]} for li in range(3)}
      for src in ['GT','LBBDM','PBBDM']}

print(f'Processing {len(common)} samples...')
for idx, sid in enumerate(common):
    if idx%50==0: print(f'  {idx}/{len(common)}...')
    gt_img=load(gt_multi_dir/f'{sid}_real_B.png')
    h,w=gt_img.shape[:2]; cut=int(h*(1-crop_frac))
    vm=np.ones((h,w),dtype=bool); vm[cut:,:]=False
    paths={'GT':gt_multi_dir/f'{sid}_real_B.png',
           'LBBDM':BASE/'Multi-color'/'LBBDM'/sid/'output_0.png',
           'PBBDM':BASE/'Multi-color'/'PBBDM'/sid/'output_0.png'}
    for src,mp in paths.items():
        img_m=load(mp); layers=list(separate_layers_hf(img_m))
        for li in range(3):
            skel=skeletonize(layers[li]); ep,jn=find_features(skel)
            data[src][li]['area'].append(int(np.sum(layers[li]&vm)))
            data[src][li]['length'].append(int(np.sum(skel&vm)))
            data[src][li]['junctions'].append(count_in_mask(jn,vm))
            data[src][li]['endpoints'].append(count_in_mask(ep,vm))

print('Generating figure...')

metrics = ['area', 'length', 'junctions', 'endpoints']
met_titles = ['Vessel Area', 'Vessel Length', 'Vessel Junctions', 'Vessel Endpoints']
models = ['LBBDM', 'PBBDM']
row_labels = ['LBBDM\nPredicted', 'PBBDM\nPredicted']
layers_use = [0, 2]  # R, B
layer_colors = ['#CC4444', '#4477CC']
layer_markers = ['o', 's']
layer_labels = ['Bottom (Red)', 'Top (Blue)']

fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))

# Title
fig.suptitle('Depth-aware Quantitative Concordance of Vascular Morphology\n'
             'Between Ground Truth and Predicted Virtual Staining (Per-layer)',
             fontsize=13, fontweight='bold', y=0.99)

# Legend at top
legend_elements = [
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#CC4444', markersize=8, label='Bottom layer (Red)'),
    plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='#4477CC', markersize=8, label='Top layer (Blue)'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, 0.955), frameon=False)

for row, model in enumerate(models):
    for col, (met, mtitle) in enumerate(zip(metrics, met_titles)):
        ax = axes[row, col]

        all_gt, all_mod = [], []
        r2_per_layer = []

        for li_idx, li in enumerate(layers_use):
            gt_v = np.array(data['GT'][li][met])
            mod_v = np.array(data[model][li][met])
            ax.scatter(gt_v, mod_v, c=layer_colors[li_idx], marker=layer_markers[li_idx],
                      s=18, alpha=0.35, edgecolors='none', zorder=2)
            all_gt.extend(gt_v.tolist())
            all_mod.extend(mod_v.tolist())

            # Per-layer R2
            valid = gt_v > 0
            if np.sum(valid) > 2:
                _, _, r_li, p_li, _ = stats.linregress(gt_v[valid], mod_v[valid])
                r2_per_layer.append((layer_labels[li_idx][:3], r_li**2, p_li))

        all_gt = np.array(all_gt)
        all_mod = np.array(all_mod)

        # Identity line
        maxval = max(np.max(all_gt), np.max(all_mod)) * 1.08 if len(all_gt) > 0 else 1
        ax.plot([0, maxval], [0, maxval], color='#888888', linewidth=1, linestyle='--', zorder=1)

        # R2 text annotations (matching reference style)
        y_text = 0.97
        for lname, r2_val, p_val in r2_per_layer:
            p_str = 'p<0.001' if p_val < 0.001 else f'p={p_val:.3f}'
            color = '#CC4444' if 'Bot' in lname else '#4477CC'
            ax.text(0.03, y_text, f'{lname}: R\u00b2={r2_val:.2f}, {p_str}',
                   transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
                   color=color, fontweight='bold')
            y_text -= 0.08

        ax.set_xlim(0, maxval)
        ax.set_ylim(0, maxval)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)

        # Column titles
        if row == 0:
            ax.set_title(mtitle, fontsize=12, fontweight='bold', pad=8)

        # X label only on bottom row
        if row == 1:
            ax.set_xlabel('Ground Truth', fontsize=10)

        # Y label
        if col == 0:
            ax.set_ylabel(f'[{model}]\nPredicted', fontsize=10, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.935], h_pad=1.0, w_pad=0.8)
plt.savefig(str(OUT / 'Figure_perlayer_concordance.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

img = Image.open(str(OUT / 'Figure_perlayer_concordance.png'))
print(f'Saved: {img.size[0]}x{img.size[1]}, ratio={img.size[0]/img.size[1]:.3f}')
print('Done.')
