"""
B mask method comparison:
1. Current: Hysteresis (seed=50, ext=20)
2. B/(R+B+1) ratio > threshold
3. B-R difference > threshold
4. HSV Hue in blue range
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
dk2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))

sids = ['18-19-512', '1-19-716', '1-16-512']

for sid in sids:
    gt_path = None
    for m in ['LSGAN','Pix2pix','WGANGP']:
        p = DATA / 'Multi-color' / m / f'{sid}_real_B.png'
        if p.exists(): gt_path = p; break
    if gt_path is None: continue

    gt = np.array(Image.open(gt_path))
    gt_blur = cv2.GaussianBlur(gt, (7,7), 0)
    h, w = gt.shape[:2]; by = int(h*0.8)
    R = gt_blur[:,:,0].astype(np.float32)
    G = gt_blur[:,:,1].astype(np.float32)
    B = gt_blur[:,:,2].astype(np.float32)
    gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2HSV)
    hue = hsv[:,:,0].astype(float)  # 0-180
    sat = hsv[:,:,1].astype(float)

    # R mask (fixed: R>45)
    r_mask = remove_small_objects(
        cv2.morphologyEx((R>45).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)

    # ============================================================
    # Method 1: Current hysteresis (seed=50, ext=20)
    # ============================================================
    b_seed = B > 50; b_low = B > 20
    b_lab, b_n = ndimage.label(b_low)
    m1 = np.zeros_like(b_low, dtype=bool)
    for i in range(1, b_n+1):
        if np.any(b_seed[b_lab==i]): m1 |= (b_lab==i)
    m1 = remove_small_objects(
        cv2.morphologyEx(m1.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2)>0, min_size=50)

    # ============================================================
    # Method 2: B/(R+B+1) ratio
    # ============================================================
    ratio = B / (R + B + 1)
    m2_thresholds = [0.45, 0.50, 0.55, 0.60]
    m2_masks = {}
    for thr in m2_thresholds:
        raw = ratio > thr
        m2 = remove_small_objects(
            cv2.morphologyEx((raw & (B > 15)).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)
        m2_masks[thr] = m2

    # ============================================================
    # Method 3: B-R difference
    # ============================================================
    diff = B - R
    m3_thresholds = [0, 5, 10, 20]
    m3_masks = {}
    for thr in m3_thresholds:
        raw = diff > thr
        m3 = remove_small_objects(
            cv2.morphologyEx((raw & (B > 15)).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)
        m3_masks[thr] = m3

    # ============================================================
    # Method 4: HSV Hue blue range
    # ============================================================
    m4_ranges = [(90,140), (85,145), (80,150), (100,135)]
    m4_masks = {}
    for (lo, hi) in m4_ranges:
        raw = (hue >= lo) & (hue <= hi) & (sat > 30) & (B > 15)
        m4 = remove_small_objects(
            cv2.morphologyEx(raw.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2)>0, min_size=50)
        m4_masks[(lo,hi)] = m4

    # ============================================================
    # FIGURE: 4 methods comparison
    # ============================================================
    # Row 0: GT, R mask, Current B (hyst)
    # Row 1: Method 2 (ratio) x4
    # Row 2: Method 3 (B-R) x4
    # Row 3: Method 4 (Hue) x4
    # Row 4: Best of each + overlap with R

    fig, axes = plt.subplots(5, 4, figsize=(20, 22))
    fig.suptitle(f'{sid} — B Mask Method Comparison (R mask: R>45)', fontsize=14, fontweight='bold')

    # Row 0: GT, R mask, Current B, R∩B overlap
    axes[0,0].imshow(gt[:by]); axes[0,0].set_title('GT', fontsize=10)
    axes[0,1].imshow(r_mask[:by], cmap='gray'); axes[0,1].set_title(f'R mask (R>45) | {np.sum(r_mask[:by])}px', fontsize=9)
    axes[0,2].imshow(m1[:by], cmap='gray'); axes[0,2].set_title(f'M1: Hyst(50/20) | {np.sum(m1[:by])}px', fontsize=9)
    # Overlap
    ovl = r_mask[:by] & m1[:by]
    vis = np.zeros((by,w,3), dtype=np.uint8)
    vis[r_mask[:by] & ~m1[:by]] = [200,60,60]; vis[m1[:by] & ~r_mask[:by]] = [60,120,200]; vis[ovl] = [200,100,200]
    axes[0,3].imshow(vis); axes[0,3].set_title(f'M1 overlap: {np.sum(ovl)}px ({np.sum(ovl)/max(np.sum(r_mask[:by]),1)*100:.0f}% of R)', fontsize=8)

    # Row 1: Method 2 (ratio)
    for col, thr in enumerate(m2_thresholds):
        m = m2_masks[thr]
        ovl = r_mask[:by] & m[:by]
        vis = np.zeros((by,w,3), dtype=np.uint8)
        vis[r_mask[:by] & ~m[:by]] = [200,60,60]; vis[m[:by] & ~r_mask[:by]] = [60,120,200]; vis[ovl] = [200,100,200]
        axes[1,col].imshow(vis)
        ovl_pct = np.sum(ovl)/max(np.sum(r_mask[:by]),1)*100
        axes[1,col].set_title(f'M2: ratio>{thr} | B={np.sum(m[:by])}px\novl={ovl_pct:.0f}%', fontsize=8)

    # Row 2: Method 3 (B-R)
    for col, thr in enumerate(m3_thresholds):
        m = m3_masks[thr]
        ovl = r_mask[:by] & m[:by]
        vis = np.zeros((by,w,3), dtype=np.uint8)
        vis[r_mask[:by] & ~m[:by]] = [200,60,60]; vis[m[:by] & ~r_mask[:by]] = [60,120,200]; vis[ovl] = [200,100,200]
        axes[2,col].imshow(vis)
        ovl_pct = np.sum(ovl)/max(np.sum(r_mask[:by]),1)*100
        axes[2,col].set_title(f'M3: B-R>{thr} | B={np.sum(m[:by])}px\novl={ovl_pct:.0f}%', fontsize=8)

    # Row 3: Method 4 (Hue)
    for col, (lo,hi) in enumerate(m4_ranges):
        m = m4_masks[(lo,hi)]
        ovl = r_mask[:by] & m[:by]
        vis = np.zeros((by,w,3), dtype=np.uint8)
        vis[r_mask[:by] & ~m[:by]] = [200,60,60]; vis[m[:by] & ~r_mask[:by]] = [60,120,200]; vis[ovl] = [200,100,200]
        axes[3,col].imshow(vis)
        ovl_pct = np.sum(ovl)/max(np.sum(r_mask[:by]),1)*100
        axes[3,col].set_title(f'M4: Hue[{lo}-{hi}] | B={np.sum(m[:by])}px\novl={ovl_pct:.0f}%', fontsize=8)

    # Row 4: Best candidates — show mask + skel on GT
    best_candidates = [
        ('M1: Hyst(50/20)', m1),
        ('M2: ratio>0.55', m2_masks[0.55]),
        ('M3: B-R>10', m3_masks[10]),
        ('M4: Hue[90-140]', m4_masks[(90,140)]),
    ]
    for col, (label, m) in enumerate(best_candidates):
        sk = skeletonize(m)
        vis = gt[:by].copy().astype(float) * 0.4
        vis[cv2.dilate(sk[:by].astype(np.uint8), dk4)>0] = [70,140,240]
        # Also show R skel
        r_sk = skeletonize(r_mask)
        vis[cv2.dilate(r_sk[:by].astype(np.uint8), dk4)>0] = [230,70,70]
        ovl_sk = cv2.dilate(sk[:by].astype(np.uint8), dk4) & cv2.dilate(r_sk[:by].astype(np.uint8), dk4)
        vis[ovl_sk>0] = [200,100,200]
        axes[4,col].imshow(vis.astype(np.uint8))
        n_comp = ndimage.label(sk)[1]
        axes[4,col].set_title(f'{label}\nskel: {np.sum(sk[:by])}px, {n_comp} comp', fontsize=8)

    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT / f'bmask_methods_{sid}.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # Print summary
    print(f'\n{"="*60}')
    print(f'{sid} (R mask: R>45, {np.sum(r_mask[:by])}px)')
    print(f'{"="*60}')
    print(f'{"Method":<25} {"B mask px":>10} {"Overlap px":>10} {"Ovl % of R":>10}')
    print(f'{"-"*60}')
    all_methods = [('M1: Hyst(50/20)', m1)]
    for thr in m2_thresholds: all_methods.append((f'M2: ratio>{thr}', m2_masks[thr]))
    for thr in m3_thresholds: all_methods.append((f'M3: B-R>{thr}', m3_masks[thr]))
    for (lo,hi) in m4_ranges: all_methods.append((f'M4: Hue[{lo}-{hi}]', m4_masks[(lo,hi)]))
    for label, m in all_methods:
        ovl = np.sum(r_mask[:by] & m[:by])
        ovl_pct = ovl / max(np.sum(r_mask[:by]),1) * 100
        print(f'{label:<25} {np.sum(m[:by]):>10} {ovl:>10} {ovl_pct:>9.0f}%')

print('\nDone!')
