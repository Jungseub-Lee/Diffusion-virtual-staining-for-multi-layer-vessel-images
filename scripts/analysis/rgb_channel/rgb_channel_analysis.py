"""
Sample 1: R, B 채널에서 직접 skeleton 추출.
Base skeleton은 참고용으로만.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
import warnings
warnings.filterwarnings('ignore')

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')
DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')

s1 = np.array(Image.open(DATA / 'Original color stack data' / 'Sample 1_BF and GT(512x512 two images).png'))
gt_raw = s1[512:]
gt_blur = cv2.GaussianBlur(gt_raw, (7, 7), 0)
h_img, w_img = gt_raw.shape[:2]

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
dk3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

R = gt_blur[:,:,0].astype(np.float32)
G = gt_blur[:,:,1].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)

# ===== R channel mask → skeleton =====
# R 채널 threshold로 vessel mask
r_thresh_vals = [30, 50, 70]
b_thresh_vals = [30, 50, 70]

def make_mask(channel, thresh, min_size=50):
    mask = channel > thresh
    mask = cv2.morphologyEx(mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
    mask = remove_small_objects(mask, min_size=min_size)
    return mask

def skel_features(skel):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    endpoints = np.argwhere(nb == 1)
    junctions = np.argwhere(nb >= 3)
    return endpoints, junctions

# Base skeleton (참고)
vessel_all = remove_small_objects(
    cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0,
    min_size=100)
base_skel = skeletonize(vessel_all)

# ===== Figure 1: R channel — threshold별 mask + skeleton =====
fig1, axes1 = plt.subplots(3, 4, figsize=(32, 20))
fig1.suptitle('R Channel: Threshold → Mask → Skeleton', fontsize=16, fontweight='bold')

for row, thresh in enumerate(r_thresh_vals):
    r_mask = make_mask(R, thresh)
    r_skel = skeletonize(r_mask)
    eps, jns = skel_features(r_skel)

    # R channel heatmap
    axes1[row,0].imshow(R, cmap='Reds', vmin=0, vmax=255)
    axes1[row,0].set_title(f'R channel (thresh={thresh})', fontsize=12)

    # Mask
    axes1[row,1].imshow(r_mask, cmap='gray')
    axes1[row,1].set_title(f'R mask (>{thresh}): {np.sum(r_mask)} px', fontsize=12)

    # Skeleton on mask
    skel_on_mask = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    skel_on_mask[r_mask] = [80, 30, 30]
    skel_on_mask[cv2.dilate(r_skel.astype(np.uint8), dk3) > 0] = [255, 100, 100]
    axes1[row,2].imshow(skel_on_mask)
    axes1[row,2].set_title(f'R skeleton: EP={len(eps)} JN={len(jns)}', fontsize=12)

    # Skeleton on GT
    r_on_gt = gt_raw.copy().astype(float) * 0.4
    r_on_gt[cv2.dilate(r_skel.astype(np.uint8), dk4) > 0] = [255, 80, 80]
    # base skel 참고용 (얇게)
    base_thin = cv2.dilate(base_skel.astype(np.uint8), dk3) > 0
    r_on_gt[base_thin & ~(cv2.dilate(r_skel.astype(np.uint8), dk4) > 0)] = \
        r_on_gt[base_thin & ~(cv2.dilate(r_skel.astype(np.uint8), dk4) > 0)] * 0.5 + np.array([0,80,0]) * 0.5
    axes1[row,3].imshow(r_on_gt.astype(np.uint8))
    axes1[row,3].set_title(f'R skel(red) + base ref(green) on GT', fontsize=11)

for ax in axes1.flat: ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'rgb_R_channel_skeleton.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved rgb_R_channel_skeleton.png')

# ===== Figure 2: B channel — threshold별 mask + skeleton =====
fig2, axes2 = plt.subplots(3, 4, figsize=(32, 20))
fig2.suptitle('B Channel: Threshold → Mask → Skeleton', fontsize=16, fontweight='bold')

for row, thresh in enumerate(b_thresh_vals):
    b_mask = make_mask(B, thresh)
    b_skel = skeletonize(b_mask)
    eps, jns = skel_features(b_skel)

    axes2[row,0].imshow(B, cmap='Blues', vmin=0, vmax=255)
    axes2[row,0].set_title(f'B channel (thresh={thresh})', fontsize=12)

    axes2[row,1].imshow(b_mask, cmap='gray')
    axes2[row,1].set_title(f'B mask (>{thresh}): {np.sum(b_mask)} px', fontsize=12)

    skel_on_mask = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    skel_on_mask[b_mask] = [30, 40, 80]
    skel_on_mask[cv2.dilate(b_skel.astype(np.uint8), dk3) > 0] = [100, 150, 255]
    axes2[row,2].imshow(skel_on_mask)
    axes2[row,2].set_title(f'B skeleton: EP={len(eps)} JN={len(jns)}', fontsize=12)

    b_on_gt = gt_raw.copy().astype(float) * 0.4
    b_on_gt[cv2.dilate(b_skel.astype(np.uint8), dk4) > 0] = [80, 130, 255]
    base_thin = cv2.dilate(base_skel.astype(np.uint8), dk3) > 0
    b_on_gt[base_thin & ~(cv2.dilate(b_skel.astype(np.uint8), dk4) > 0)] = \
        b_on_gt[base_thin & ~(cv2.dilate(b_skel.astype(np.uint8), dk4) > 0)] * 0.5 + np.array([0,80,0]) * 0.5
    axes2[row,3].imshow(b_on_gt.astype(np.uint8))
    axes2[row,3].set_title(f'B skel(blue) + base ref(green) on GT', fontsize=11)

for ax in axes2.flat: ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'rgb_B_channel_skeleton.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved rgb_B_channel_skeleton.png')

# ===== Figure 3: R+B skeleton 합쳐서 비교 (중간 threshold) =====
best_thresh = 50
r_mask = make_mask(R, best_thresh)
b_mask = make_mask(B, best_thresh)
r_skel = skeletonize(r_mask)
b_skel = skeletonize(b_mask)

r_eps, r_jns = skel_features(r_skel)
b_eps, b_jns = skel_features(b_skel)
base_eps, base_jns = skel_features(base_skel)

fig3, axes3 = plt.subplots(2, 3, figsize=(27, 16))
fig3.suptitle(f'R/B Channel Skeleton (thresh={best_thresh}) vs Base', fontsize=16, fontweight='bold')

# GT
axes3[0,0].imshow(gt_raw)
axes3[0,0].set_title('GT', fontsize=13)

# R+B skel on GT
rb_on_gt = gt_raw.copy().astype(float) * 0.4
rb_on_gt[cv2.dilate(r_skel.astype(np.uint8), dk4) > 0] = [255, 80, 80]
rb_on_gt[cv2.dilate(b_skel.astype(np.uint8), dk4) > 0] = [80, 130, 255]
# overlap
both_skel = cv2.dilate(r_skel.astype(np.uint8), dk4) & cv2.dilate(b_skel.astype(np.uint8), dk4)
rb_on_gt[both_skel > 0] = [200, 50, 200]
axes3[0,1].imshow(rb_on_gt.astype(np.uint8))
axes3[0,1].set_title(f'R skel(red) + B skel(blue)\noverlap(purple) on GT', fontsize=12)

# Base skel on GT (참고)
base_on_gt = gt_raw.copy().astype(float) * 0.4
base_on_gt[cv2.dilate(base_skel.astype(np.uint8), dk4) > 0] = [0, 255, 0]
axes3[0,2].imshow(base_on_gt.astype(np.uint8))
axes3[0,2].set_title('Base skeleton (ref)', fontsize=13)

# Skeleton only
rb_skel_vis = np.zeros((h_img, w_img, 3), dtype=np.uint8)
rb_skel_vis[cv2.dilate(r_skel.astype(np.uint8), dk3) > 0] = [230, 70, 70]
rb_skel_vis[cv2.dilate(b_skel.astype(np.uint8), dk3) > 0] = [70, 130, 240]
rb_skel_vis[(cv2.dilate(r_skel.astype(np.uint8), dk3) > 0) &
            (cv2.dilate(b_skel.astype(np.uint8), dk3) > 0)] = [200, 50, 200]
axes3[1,0].imshow(rb_skel_vis)
axes3[1,0].set_title('R skel + B skel', fontsize=13)

# Base skel only
base_vis = np.zeros((h_img, w_img, 3), dtype=np.uint8)
base_vis[cv2.dilate(base_skel.astype(np.uint8), dk3) > 0] = [200, 200, 200]
axes3[1,1].imshow(base_vis)
axes3[1,1].set_title('Base skeleton', fontsize=13)

# Metrics
crop_y = int(h_img * 0.8)
def count_above(eps, jns, skel):
    j = np.sum(jns[:,0] < crop_y) if len(jns) > 0 else 0
    e = np.sum(eps[:,0] < crop_y) if len(eps) > 0 else 0
    vm = np.ones((h_img, w_img), dtype=bool); vm[crop_y:] = False
    l = int(np.sum(skel & vm))
    return j, l, e

jr, lr, er = count_above(r_eps, r_jns, r_skel)
jb, lb, eb = count_above(b_eps, b_jns, b_skel)
jbase, lbase, ebase = count_above(base_eps, base_jns, base_skel)

axes3[1,2].axis('off')
axes3[1,2].set_facecolor('black')
txt = (f"R channel skel:\n  J={jr}  L={lr}  E={er}\n\n"
       f"B channel skel:\n  J={jb}  L={lb}  E={eb}\n\n"
       f"R + B sum:\n  J={jr+jb}  L={lr+lb}  E={er+eb}\n\n"
       f"Base skeleton (ref):\n  J={jbase}  L={lbase}  E={ebase}")
axes3[1,2].text(0.1, 0.8, txt, fontsize=14, transform=axes3[1,2].transAxes,
                fontfamily='monospace', color='white', va='top')
axes3[1,2].set_title('Metrics', fontsize=13)

for ax in axes3.flat: ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'rgb_RB_skeleton_combined.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nR skel: J={jr} L={lr} E={er}")
print(f"B skel: J={jb} L={lb} E={eb}")
print(f"R+B:    J={jr+jb} L={lr+lb} E={er+eb}")
print(f"Base:   J={jbase} L={lbase} E={ebase}")
print('\nSaved rgb_RB_skeleton_combined.png')
print('Done!')
