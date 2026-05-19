import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import cv2, json, warnings
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
def get_vessel_mask(img, min_area=50):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY); v=gray>15
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    u=cv2.morphologyEx(v.astype(np.uint8)*255,cv2.MORPH_CLOSE,k,iterations=2)
    u=cv2.morphologyEx(u,cv2.MORPH_OPEN,k,iterations=1)
    return remove_small_objects(u>0,min_size=min_area)
def dilate_skel(s, t=3):
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
    return cv2.dilate(s.astype(np.uint8)*255,k,iterations=1)>0
def find_features(skel):
    su=skel.astype(np.uint8)
    ks=np.array([[1,1,1],[1,0,1],[1,1,1]],dtype=np.uint8)
    nb=cv2.filter2D(su,-1,ks)*su
    return np.argwhere(nb==1),np.argwhere(nb>=3)
def count_in_mask(coords,mask):
    return sum(1 for y,x in coords if mask[y,x]) if len(coords) else 0
def count_true_ep(skels,masks,vm,radius=15):
    adj=[(0,1),(1,2)]
    dk=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
    dil=[cv2.dilate(m.astype(np.uint8)*255,dk,iterations=1)>0 for m in masks]
    tot=0
    for li in range(3):
        if np.sum(skels[li])==0: continue
        ep,_=find_features(skels[li])
        for ey,ex in ep:
            if not vm[ey,ex]: continue
            bnd=False
            for a,b_ in adj:
                if li==a and dil[b_][ey,ex]: bnd=True; break
                elif li==b_ and dil[a][ey,ex]: bnd=True; break
            if not bnd: tot+=1
    return tot

layer_colors=[(230,70,70),(240,220,50),(70,130,240)]
gt_multi_dir=BASE/'Multi-color'/'Pix2pix'
gt_single_dir=BASE/'Single-color'/'Pix2pix'

# === Representative images for 18-19-512 ===
sid = '18-19-512'
bf = load(gt_multi_dir/f'{sid}_real_A.png')
gt_m = load(gt_multi_dir/f'{sid}_real_B.png')
gt_s = load(gt_single_dir/f'{sid}_real_B.png')
mask_s = get_vessel_mask(gt_s); skel_s = skeletonize(mask_s)
bf_single = bf.copy(); bf_single[dilate_skel(skel_s, 4)] = (0, 200, 0)
lm = list(separate_layers_hf(gt_m)); ls_ = [skeletonize(m) for m in lm]
bf_multi = bf.copy()
for li, sk in enumerate(ls_):
    bf_multi[dilate_skel(sk, 4)] = layer_colors[li]
Image.fromarray(bf).save(OUT/f'panel_b_{sid}_bf.png')
Image.fromarray(gt_m).save(OUT/f'panel_b_{sid}_fl.png')
Image.fromarray(bf_single).save(OUT/f'panel_b_{sid}_single.png')
Image.fromarray(bf_multi).save(OUT/f'panel_b_{sid}_multi.png')
print(f'Saved panel_b images for {sid}')

# === Full analysis crop 20% ===
all_sets=[
    {f.stem.replace('_real_B','') for f in gt_multi_dir.glob('*_real_B.png')},
    {f.stem.replace('_real_B','') for f in gt_single_dir.glob('*_real_B.png')},
    {d.name for d in (BASE/'Multi-color'/'LBBDM').iterdir() if d.is_dir()},
    {d.name for d in (BASE/'Single-color'/'LBBDM').iterdir() if d.is_dir()},
    {d.name for d in (BASE/'Multi-color'/'PBBDM').iterdir() if d.is_dir()},
    {d.name for d in (BASE/'Single-color'/'PBBDM').iterdir() if d.is_dir()},
]
common=sorted(set.intersection(*all_sets))
crop_frac = 0.20
metrics_keys = ['junctions','endpoints','length','area']

all_data = {src: {'single':[], 'multi':[]} for src in ['GT','LBBDM','PBBDM']}
for idx, sid_c in enumerate(common):
    if idx % 50 == 0: print(f'Processing {idx}/{len(common)}...')
    gt_m_img = load(gt_multi_dir/f'{sid_c}_real_B.png')
    h,w = gt_m_img.shape[:2]; cut=int(h*(1-crop_frac))
    vm=np.ones((h,w),dtype=bool); vm[cut:,:]=False
    paths={
        'GT':(gt_multi_dir/f'{sid_c}_real_B.png',gt_single_dir/f'{sid_c}_real_B.png'),
        'LBBDM':(BASE/'Multi-color'/'LBBDM'/sid_c/'output_0.png',BASE/'Single-color'/'LBBDM'/sid_c/'output_0.png'),
        'PBBDM':(BASE/'Multi-color'/'PBBDM'/sid_c/'output_0.png',BASE/'Single-color'/'PBBDM'/sid_c/'output_0.png'),
    }
    for src,(mp,sp) in paths.items():
        img_m=load(mp)
        lm_c=list(separate_layers_hf(img_m))
        ls_c=[skeletonize(m) for m in lm_c]
        mj=sum(count_in_mask(find_features(sk)[1],vm) for sk in ls_c)
        ml=sum(int(np.sum(sk&vm)) for sk in ls_c)
        me=count_true_ep(ls_c,lm_c,vm)
        ma=sum(int(np.sum(m&vm)) for m in lm_c)
        all_data[src]['multi'].append({'junctions':mj,'endpoints':me,'length':ml,'area':ma})
        img_s=load(sp); ms_c=get_vessel_mask(img_s); ss_c=skeletonize(ms_c)
        es_c,js_c=find_features(ss_c)
        all_data[src]['single'].append({'junctions':count_in_mask(js_c,vm),'endpoints':count_in_mask(es_c,vm),'length':int(np.sum(ss_c&vm)),'area':int(np.sum(ms_c&vm))})

# === Panel C: Single/MultiGT normalized ===
met_labels = ['Junctions', 'Endpoints', 'Length', 'Area']
panel_c = {}
for key in metrics_keys:
    gt_m_vals = [x[key] for x in all_data['GT']['multi']]
    gt_s_vals = [x[key] for x in all_data['GT']['single']]
    # Normalize to Single GT = 1.0: compute Multi/Single per sample
    m_rats = [m/s if s>0 else np.nan for s,m in zip(gt_s_vals, gt_m_vals)]
    m_rats = np.array([r for r in m_rats if not np.isnan(r)])
    t,p = stats.ttest_1samp(m_rats, 1.0)
    sig='***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    panel_c[key] = {'mean': float(np.mean(m_rats)), 'sem': float(np.std(m_rats)/len(m_rats)**0.5), 'p': float(p), 'sig': sig}
    print(f'Panel C {key}: Multi/Single = {np.mean(m_rats):.3f}+-{np.std(m_rats)/len(m_rats)**0.5:.3f} p={p:.1e}({sig})')

# === Panel D ===
panel_d = {}
for src in ['LBBDM','PBBDM']:
    panel_d[src] = {}
    for key in metrics_keys:
        gt_vals = [x[key] for x in all_data['GT']['multi']]
        model_vals = [x[key] for x in all_data[src]['multi']]
        rats = np.array([m/g for g,m in zip(gt_vals, model_vals) if g>0])
        t,p = stats.ttest_1samp(rats, 1.0)
        _,_,r,_,_ = stats.linregress(gt_vals, model_vals)
        sig='ns' if p>0.05 else '*' if p>0.01 else '**' if p>0.001 else '***'
        panel_d[src][key] = {'mean': float(np.mean(rats)), 'sem': float(np.std(rats)/len(rats)**0.5), 'r2': float(r**2), 'p': float(p), 'sig': sig}
        print(f'Panel D {src} {key}: {np.mean(rats):.3f} R2={r**2:.3f} p={p:.1e}({sig})')

# === Generate Panel C chart ===
fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
x = np.arange(4)
width = 0.35
m_means = [panel_c[k]['mean'] for k in metrics_keys]
m_sems = [panel_c[k]['sem'] for k in metrics_keys]

bars_s = ax.bar(x - width/2, [1.0]*4, width, color='#5CB85C', edgecolor='black', linewidth=0.8, label='Single-color')
bars_m = ax.bar(x + width/2, m_means, width, yerr=m_sems, color='#F0AD4E', edgecolor='black', linewidth=0.8, capsize=3, label='Multi-color')
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(met_labels, fontsize=11, fontweight='bold')
ax.set_ylabel('Normalized to Single-color GT', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(0, 1.35)
plt.tight_layout()
plt.savefig(str(OUT/'panel_c_norm.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# === Generate Panel D chart ===
fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
width = 0.3
l_means = [panel_d['LBBDM'][k]['mean'] for k in metrics_keys]
l_sems = [panel_d['LBBDM'][k]['sem'] for k in metrics_keys]
l_r2s = [panel_d['LBBDM'][k]['r2'] for k in metrics_keys]
l_sigs = [panel_d['LBBDM'][k]['sig'] for k in metrics_keys]
p_means = [panel_d['PBBDM'][k]['mean'] for k in metrics_keys]
p_sems = [panel_d['PBBDM'][k]['sem'] for k in metrics_keys]
p_r2s = [panel_d['PBBDM'][k]['r2'] for k in metrics_keys]
p_sigs = [panel_d['PBBDM'][k]['sig'] for k in metrics_keys]

bars_l = ax.bar(x-width/2, l_means, width, yerr=l_sems, color='#D4762C', edgecolor='black', linewidth=0.8, capsize=3, label='LBBDM')
bars_p = ax.bar(x+width/2, p_means, width, yerr=p_sems, color='#8B6AAE', edgecolor='black', linewidth=0.8, capsize=3, label='PBBDM')
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(met_labels, fontsize=11, fontweight='bold')
ax.set_ylabel('Model / GT Multi ratio', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(0, 1.35)
plt.tight_layout()
plt.savefig(str(OUT/'panel_d_norm.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print('\nAll done.')
