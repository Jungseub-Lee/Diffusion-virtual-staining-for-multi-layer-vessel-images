"""
B channel: Hysteresis skel → B intensity가 진한 곳만 남기기
R channel: thresh=30 skel → R intensity가 진한 곳만 남기기
그리고 연결 문제 분석
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
import warnings
warnings.filterwarnings('ignore')

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')
DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')

s1 = np.array(Image.open(DATA / 'Original color stack data' / 'Sample 1_BF and GT(512x512 two images).png'))
gt_raw = s1[512:]
gt_blur = cv2.GaussianBlur(gt_raw, (7, 7), 0)
h, w = gt_raw.shape[:2]

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dk3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

R = gt_blur[:,:,0].astype(np.float32)
G = gt_blur[:,:,1].astype(np.float32)
B = gt_blur[:,:,2].astype(np.float32)
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)

# ===== R skeleton (thresh=30) =====
r_mask = R > 30
r_mask = cv2.morphologyEx(r_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
r_mask = remove_small_objects(r_mask, min_size=50)
r_skel = skeletonize(r_mask)

# ===== B skeleton (hysteresis seed=50, low=20) =====
b_seed = B > 50
b_low = B > 20
b_seed_l, n_s = ndimage.label(b_seed)
b_low_l, n_l = ndimage.label(b_low)
b_hyst = np.zeros_like(b_low, dtype=bool)
for i in range(1, n_l + 1):
    region = b_low_l == i
    if np.any(b_seed[region]):
        b_hyst |= region
b_hyst = cv2.morphologyEx(b_hyst.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0
b_hyst = remove_small_objects(b_hyst, min_size=50)
b_skel = skeletonize(b_hyst)

# Base skel (참고)
vessel_all = remove_small_objects(
    cv2.morphologyEx((gray > 10).astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0,
    min_size=100)
base_skel = skeletonize(vessel_all)

# ===== Skeleton pixel별 channel intensity 측정 =====
# skeleton 위치에서 주변 5x5 영역의 평균 intensity
def get_local_intensity(channel, skel, radius=3):
    """각 skel pixel 주변 radius 영역의 channel 평균값"""
    # channel을 blur해서 local average로
    kernel_size = 2 * radius + 1
    local_avg = cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)
    intensity_map = np.zeros_like(channel)
    intensity_map[skel] = local_avg[skel]
    return intensity_map

r_on_rskel = get_local_intensity(R, r_skel, radius=4)
b_on_bskel = get_local_intensity(B, b_skel, radius=4)

# ===== Intensity threshold별 skeleton 필터링 =====
intensity_thresholds = [30, 50, 70, 90]

fig, axes = plt.subplots(4, 4, figsize=(32, 28))
fig.suptitle('Skeleton filtered by channel intensity (keep only "dark/intense" regions)',
             fontsize=16, fontweight='bold')

for row, int_thresh in enumerate(intensity_thresholds):
    # B skel: B값이 int_thresh 이상인 곳만 남기기
    b_blur9 = cv2.GaussianBlur(B, (9, 9), 0)
    b_filtered = b_skel & (b_blur9 >= int_thresh)

    # R skel: R값이 int_thresh 이상인 곳만 남기기
    r_filtered = r_skel & (cv2.GaussianBlur(R, (9, 9), 0) >= int_thresh)

    # Panel 1: B skel on B channel
    b_vis = np.stack([B*0.3, B*0.3, B*0.8], axis=-1).astype(np.uint8)
    b_vis[cv2.dilate(b_skel.astype(np.uint8), dk3) > 0] = [100, 100, 200]  # full skel dim
    b_vis[cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0] = [50, 200, 255]  # filtered bright
    axes[row, 0].imshow(b_vis)
    axes[row, 0].set_title(f'B skel: intensity≥{int_thresh}\ncyan=kept, dim=removed', fontsize=11)

    # Panel 2: R skel on R channel
    r_vis = np.stack([R*0.8, R*0.3, R*0.3], axis=-1).astype(np.uint8)
    r_vis[cv2.dilate(r_skel.astype(np.uint8), dk3) > 0] = [200, 100, 100]
    r_vis[cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0] = [255, 230, 50]
    axes[row, 1].imshow(r_vis)
    axes[row, 1].set_title(f'R skel: intensity≥{int_thresh}\nyellow=kept, dim=removed', fontsize=11)

    # Panel 3: Filtered R + B on GT
    on_gt = gt_raw.copy().astype(float) * 0.3
    on_gt[cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0] = [255, 80, 80]
    on_gt[cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0] = [80, 150, 255]
    both = (cv2.dilate(r_filtered.astype(np.uint8), dk4) > 0) & \
           (cv2.dilate(b_filtered.astype(np.uint8), dk4) > 0)
    on_gt[both] = [200, 50, 200]
    axes[row, 2].imshow(on_gt.astype(np.uint8))
    axes[row, 2].set_title(f'Filtered R(red)+B(blue) on GT\nintensity≥{int_thresh}', fontsize=11)

    # Panel 4: Gap analysis — where skeleton was removed
    gap_vis = np.zeros((h, w, 3), dtype=np.uint8)
    # Kept parts
    gap_vis[cv2.dilate(r_filtered.astype(np.uint8), dk3) > 0] = [200, 60, 60]
    gap_vis[cv2.dilate(b_filtered.astype(np.uint8), dk3) > 0] = [60, 120, 220]
    # Removed parts (gaps) — 이 부분이 crossing에서 빠진 구간
    r_gap = r_skel & ~r_filtered
    b_gap = b_skel & ~b_filtered
    gap_vis[cv2.dilate(r_gap.astype(np.uint8), dk4) > 0] = [255, 200, 100]  # R gap orange
    gap_vis[cv2.dilate(b_gap.astype(np.uint8), dk4) > 0] = [100, 255, 200]  # B gap cyan

    r_gap_px = int(np.sum(r_gap))
    b_gap_px = int(np.sum(b_gap))
    axes[row, 3].imshow(gap_vis)
    axes[row, 3].set_title(f'Gaps: R(orange)={r_gap_px}px B(cyan)={b_gap_px}px\n'
                           f'→ crossing에서 빠진 구간', fontsize=11)

for ax in axes.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'rgb_intensity_filtered_skel.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved rgb_intensity_filtered_skel.png')

# ===== Figure 2: Crossing 연결 문제 분석 =====
# intensity=50 기준으로 상세 분석
int_t = 50
b_filt = b_skel & (cv2.GaussianBlur(B, (9, 9), 0) >= int_t)
r_filt = r_skel & (cv2.GaussianBlur(R, (9, 9), 0) >= int_t)
r_gap = r_skel & ~r_filt
b_gap = b_skel & ~b_filt

fig2, axes2 = plt.subplots(2, 3, figsize=(27, 16))
fig2.suptitle(f'Crossing Gap Analysis (intensity threshold={int_t})', fontsize=16, fontweight='bold')

# Full view: gaps highlighted
full_vis = gt_raw.copy().astype(float) * 0.3
full_vis[cv2.dilate(r_filt.astype(np.uint8), dk4) > 0] = [255, 80, 80]
full_vis[cv2.dilate(b_filt.astype(np.uint8), dk4) > 0] = [80, 150, 255]
full_vis[cv2.dilate(r_gap.astype(np.uint8), dk4) > 0] = [255, 200, 50]  # R gap
full_vis[cv2.dilate(b_gap.astype(np.uint8), dk4) > 0] = [50, 255, 200]  # B gap
axes2[0, 0].imshow(full_vis.astype(np.uint8))
axes2[0, 0].set_title('Full: R(red) B(blue)\nR-gap(yellow) B-gap(cyan)', fontsize=12)

# R channel intensity along R skeleton
r_int_vis = np.zeros((h, w, 3), dtype=np.uint8)
r_blur = cv2.GaussianBlur(R, (9, 9), 0)
for y, x in np.argwhere(r_skel):
    val = r_blur[y, x]
    # intensity를 color로 (dark red → bright yellow)
    r_int_vis[y, x] = [min(255, int(val * 2)), min(255, int(val * 1.5)), 0]
r_int_vis_d = cv2.dilate(r_int_vis, dk3)
axes2[0, 1].imshow(r_int_vis_d)
axes2[0, 1].set_title('R intensity along R skel\n(dark=low, bright=high)', fontsize=12)

# B channel intensity along B skeleton
b_int_vis = np.zeros((h, w, 3), dtype=np.uint8)
b_blur = cv2.GaussianBlur(B, (9, 9), 0)
for y, x in np.argwhere(b_skel):
    val = b_blur[y, x]
    b_int_vis[y, x] = [0, min(255, int(val * 1.5)), min(255, int(val * 2))]
b_int_vis_d = cv2.dilate(b_int_vis, dk3)
axes2[0, 2].imshow(b_int_vis_d)
axes2[0, 2].set_title('B intensity along B skel\n(dark=low, bright=high)', fontsize=12)

# Key insight: at crossings, R skel has low R → gap is where BLUE vessel crosses over
# Similarly, B skel has low B → gap is where RED vessel crosses over
# So: R-gap location ≈ B vessel crossing, B-gap location ≈ R vessel crossing

# R gap이 B skel과 겹치는지, B gap이 R skel과 겹치는지
r_gap_dilated = cv2.dilate(r_gap.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))) > 0
b_gap_dilated = cv2.dilate(b_gap.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))) > 0

# R gap 근처에 B skel이 있는가?
r_gap_near_b = r_gap_dilated & b_filt
b_gap_near_r = b_gap_dilated & r_filt

cross_vis = np.zeros((h, w, 3), dtype=np.uint8)
cross_vis[cv2.dilate(r_filt.astype(np.uint8), dk3) > 0] = [200, 60, 60]
cross_vis[cv2.dilate(b_filt.astype(np.uint8), dk3) > 0] = [60, 120, 200]
# Crossing points: where gaps of one channel meet the other channel's skel
cross_vis[cv2.dilate(r_gap_near_b.astype(np.uint8), dk4) > 0] = [255, 255, 0]
cross_vis[cv2.dilate(b_gap_near_r.astype(np.uint8), dk4) > 0] = [0, 255, 255]
axes2[1, 0].imshow(cross_vis)
axes2[1, 0].set_title('Crossing detection:\nR-gap∩B-skel(yellow) B-gap∩R-skel(cyan)', fontsize=11)

# What it means conceptually
concept = np.zeros((h, w, 3), dtype=np.uint8)
# R vessel = R-filtered (definite) + R-gap bridged through crossing
# B vessel = B-filtered (definite) + B-gap bridged through crossing
concept[cv2.dilate(r_filt.astype(np.uint8), dk4) > 0] = [255, 80, 80]
concept[cv2.dilate(b_filt.astype(np.uint8), dk4) > 0] = [80, 150, 255]
# At crossing: both vessels pass through → both get the gap segment
concept[cv2.dilate(r_gap.astype(np.uint8), dk4) > 0] = [255, 150, 150]  # R gap = still R vessel
concept[cv2.dilate(b_gap.astype(np.uint8), dk4) > 0] = [150, 180, 255]  # B gap = still B vessel
axes2[1, 1].imshow(concept)
axes2[1, 1].set_title('Concept: keep gaps as vessel\nlight=crossing segment', fontsize=12)

# base skel reference
base_vis = gt_raw.copy().astype(float) * 0.3
base_vis[cv2.dilate(base_skel.astype(np.uint8), dk4) > 0] = [0, 255, 0]
axes2[1, 2].imshow(base_vis.astype(np.uint8))
axes2[1, 2].set_title('Base skel (reference)', fontsize=12)

for ax in axes2.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'rgb_crossing_gap_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved rgb_crossing_gap_analysis.png')
print('Done!')
