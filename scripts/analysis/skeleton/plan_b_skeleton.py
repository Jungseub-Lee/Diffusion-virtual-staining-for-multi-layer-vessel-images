"""
Plan B: Base skeleton + per-pixel color + 색상 영역 분리 분석 (Sample 1).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import label
import warnings
warnings.filterwarnings('ignore')

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')
DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')
OUT.mkdir(exist_ok=True)

# Load Sample 1 GT
s1 = np.array(Image.open(DATA / 'Original color stack data' / 'Sample 1_BF and GT(512x512 two images).png'))
gt_raw = s1[512:]
h_img, w_img = gt_raw.shape[:2]

# Gaussian blur 전처리
gt = cv2.GaussianBlur(gt_raw, (7, 7), 0)

hsv = cv2.cvtColor(gt, cv2.COLOR_RGB2HSV)
hue = hsv[:,:,0].astype(np.float32)
sat = hsv[:,:,1].astype(np.float32)
gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# ===== 1. Vessel mask + base skeleton =====
vessel_mask = (gray > 10) & (sat > 15)  # threshold 낮춤
vessel_mask = cv2.morphologyEx(vessel_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
vessel_mask = remove_small_objects(vessel_mask, min_size=100)
base_skel = skeletonize(vessel_mask)

# ===== 2. 색상 영역 마스크 — RGB 채널 기반 (non-exclusive) =====
# RGB 채널 직접 사용: 보라/마젠타 pixel이 R과 B 양쪽에 포함됨
gt_float = gt.astype(np.float32)
r_ch = gt_float[:,:,0]
g_ch = gt_float[:,:,1]
b_ch = gt_float[:,:,2]

fg = vessel_mask  # 이미 만든 vessel mask 사용

# Red 영역: R 채널이 dominant하거나 significant
# (빨간색, 주황, 마젠타/보라 포함)
r_mask = (r_ch > 40) & (r_ch > g_ch * 1.2) & fg

# Blue 영역: B 채널이 dominant하거나 significant
# (파란색, 시안, 마젠타/보라 포함)
b_mask = (b_ch > 40) & (b_ch > g_ch * 1.0) & fg

# Yellow 영역: R+G 둘 다 높고 B 낮음
y_mask = (r_ch > 40) & (g_ch > 40) & (b_ch < g_ch * 0.8) & fg

# Morphological cleanup
r_mask = cv2.morphologyEx(r_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
y_mask = cv2.morphologyEx(y_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
b_mask = cv2.morphologyEx(b_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
r_mask = remove_small_objects(r_mask > 0, min_size=30)
y_mask = remove_small_objects(y_mask > 0, min_size=30)
b_mask = remove_small_objects(b_mask > 0, min_size=30)

print(f"R: {np.sum(r_mask)} px, Y: {np.sum(y_mask)} px, B: {np.sum(b_mask)} px")

# ===== 3. 조합 영역 =====
ry_mask = r_mask | y_mask
by_mask = b_mask | y_mask
rb_overlap = r_mask & b_mask  # 이제 보라/마젠타 pixel이 여기 포함됨

# R+B overlap을 개별 component로 분리
rb_labels, n_rb = label(rb_overlap)
print(f"R∩B overlap: {np.sum(rb_overlap)} pixels, {n_rb} components")

# ===== 4. Per-pixel skeleton coloring =====
def classify_pixel(py, px, hue, sat, radius=5):
    h_img, w_img = hue.shape
    y0, y1 = max(0, py-radius), min(h_img, py+radius+1)
    x0, x1 = max(0, px-radius), min(w_img, px+radius+1)
    local_hue = hue[y0:y1, x0:x1]
    local_sat = sat[y0:y1, x0:x1]
    valid = local_sat > 15
    if not np.any(valid):
        return 'Y'
    h_vals = local_hue[valid]
    rc = np.sum((h_vals <= 15) | (h_vals >= 150))
    yc = np.sum((h_vals >= 16) & (h_vals <= 55))
    bc = np.sum((h_vals >= 80) & (h_vals <= 145))
    if rc >= yc and rc >= bc: return 'R'
    if bc >= yc: return 'B'
    return 'Y'

skel_pts = np.argwhere(base_skel)
colored_skel_mask = {
    'R': np.zeros((h_img, w_img), dtype=bool),
    'Y': np.zeros((h_img, w_img), dtype=bool),
    'B': np.zeros((h_img, w_img), dtype=bool),
}
for (py, px) in skel_pts:
    c = classify_pixel(py, px, hue, sat, radius=5)
    colored_skel_mask[c][py, px] = True

def skel_features(skel):
    su = skel.astype(np.uint8)
    ker = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    endpoints = set(map(tuple, np.argwhere(nb == 1)))
    junctions = set(map(tuple, np.argwhere(nb >= 3)))
    return endpoints, junctions

endpoints, junctions = skel_features(base_skel)
print(f"Base skeleton: {np.sum(base_skel)} px, {len(endpoints)} EP, {len(junctions)} JN")

# ===== VISUALIZATION =====
dk3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dk4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

# --- Figure 1: 색상 영역 분리 ---
fig1, axes1 = plt.subplots(2, 4, figsize=(32, 14))
fig1.suptitle('Sample 1: Color Region Separation (Gaussian blur k=7, lower threshold)',
              fontsize=16, fontweight='bold')

# Row 1: GT, R, Y, B 개별
axes1[0,0].imshow(gt)
axes1[0,0].set_title('GT (blurred)', fontsize=13)

# R only: GT에서 R 영역만
r_img = gt_raw.copy(); r_img[~r_mask] = 0
axes1[0,1].imshow(r_img)
axes1[0,1].set_title(f'Red only ({np.sum(r_mask)} px)', fontsize=13)

# Y only
y_img = gt_raw.copy(); y_img[~y_mask] = 0
axes1[0,2].imshow(y_img)
axes1[0,2].set_title(f'Yellow only ({np.sum(y_mask)} px)', fontsize=13)

# B only
b_img = gt_raw.copy(); b_img[~b_mask] = 0
axes1[0,3].imshow(b_img)
axes1[0,3].set_title(f'Blue only ({np.sum(b_mask)} px)', fontsize=13)

# Row 2: R+Y, B+Y, R∩B overlap, R∩B components
ry_img = gt_raw.copy(); ry_img[~ry_mask] = 0
axes1[1,0].imshow(ry_img)
axes1[1,0].set_title(f'Red + Yellow ({np.sum(ry_mask)} px)', fontsize=13)

by_img = gt_raw.copy(); by_img[~by_mask] = 0
axes1[1,1].imshow(by_img)
axes1[1,1].set_title(f'Blue + Yellow ({np.sum(by_mask)} px)', fontsize=13)

# R∩B overlap on GT
rb_img = gt_raw.copy(); rb_img[~rb_overlap] = 0
axes1[1,2].imshow(rb_img)
axes1[1,2].set_title(f'Red ∩ Blue overlap ({np.sum(rb_overlap)} px)', fontsize=13)

# R∩B components individually colored
rb_comp_vis = np.zeros((*gt_raw.shape[:2], 3), dtype=np.uint8)
cmap_tab = plt.cm.tab10(np.linspace(0, 1, 10))
for i in range(1, n_rb + 1):
    comp = rb_labels == i
    color = (np.array(cmap_tab[i % 10][:3]) * 255).astype(np.uint8)
    rb_comp_vis[comp] = color
axes1[1,3].imshow(rb_comp_vis)
axes1[1,3].set_title(f'R∩B components ({n_rb} regions)', fontsize=13)

for ax in axes1.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'plan_b_color_regions.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved plan_b_color_regions.png')

# --- Figure 2: R∩B 개별 component zoom ---
if n_rb > 0:
    # 크기순 정렬
    comp_sizes = [(i, np.sum(rb_labels == i)) for i in range(1, n_rb + 1)]
    comp_sizes.sort(key=lambda x: -x[1])
    n_show = min(12, n_rb)

    ncols = 4
    nrows = (n_show + ncols - 1) // ncols
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(24, 6 * nrows))
    if nrows == 1:
        axes3 = axes3[np.newaxis, :]
    fig3.suptitle(f'R∩B Overlap: {n_rb} Individual Components (by size)',
                  fontsize=14, fontweight='bold')

    for ci in range(n_show):
        row, col = ci // ncols, ci % ncols
        comp_id, comp_size = comp_sizes[ci]
        comp = rb_labels == comp_id

        # Bounding box + padding
        ys, xs = np.where(comp)
        pad = 30
        y0, y1 = max(0, ys.min()-pad), min(h_img, ys.max()+pad+1)
        x0, x1 = max(0, xs.min()-pad), min(w_img, xs.max()+pad+1)
        sl = (slice(y0, y1), slice(x0, x1))

        # GT crop + R∩B overlay
        crop = gt_raw[sl].copy().astype(float)
        overlay = np.zeros((y1-y0, x1-x0, 4))

        # Show R region in this area (semi-transparent red)
        local_r = r_mask[sl]
        local_b = b_mask[sl]
        local_rb = comp[sl]

        # Background: GT dimmed
        bg = crop * 0.4

        # R area = red tint, B area = blue tint, overlap = magenta
        vis = bg.copy()
        vis[local_r & ~local_rb] = np.clip(vis[local_r & ~local_rb] * 0.3 + np.array([200,50,50]) * 0.7, 0, 255)
        vis[local_b & ~local_rb] = np.clip(vis[local_b & ~local_rb] * 0.3 + np.array([50,80,200]) * 0.7, 0, 255)
        vis[local_rb] = np.clip(vis[local_rb] * 0.2 + np.array([220,50,220]) * 0.8, 0, 255)

        axes3[row, col].imshow(vis.astype(np.uint8))
        axes3[row, col].set_title(f'#{ci+1} ({comp_size} px)\nR=red, B=blue, R∩B=magenta', fontsize=10)
        axes3[row, col].axis('off')

    # Hide unused
    for ci in range(n_show, nrows * ncols):
        row, col = ci // ncols, ci % ncols
        axes3[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(OUT / 'plan_b_rb_overlap_components.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved plan_b_rb_overlap_components.png')

# --- Figure 3: Skeleton comparison (기존 유지) ---
fig2, axes2 = plt.subplots(2, 3, figsize=(27, 16))
fig2.suptitle('Sample 1: Base Skeleton + Color (blur k=7)', fontsize=16, fontweight='bold')

axes2[0,0].imshow(gt_raw)
axes2[0,0].set_title('Original GT (raw)', fontsize=13)

base_on_gt = gt_raw.copy().astype(float) * 0.5
base_on_gt[cv2.dilate(base_skel.astype(np.uint8), dk4) > 0] = [0, 255, 0]
axes2[0,1].imshow(base_on_gt.astype(np.uint8))
axes2[0,1].set_title(f'Base Skeleton\n{len(endpoints)} EP, {len(junctions)} JN', fontsize=13)

colored_on_gt = gt_raw.copy().astype(float) * 0.5
for c_label, c_rgb in [('R', [230,70,70]), ('Y', [240,220,50]), ('B', [70,130,240])]:
    c_dilated = cv2.dilate(colored_skel_mask[c_label].astype(np.uint8), dk4) > 0
    colored_on_gt[c_dilated] = c_rgb
axes2[0,2].imshow(colored_on_gt.astype(np.uint8))
axes2[0,2].set_title('Base Skel + Per-pixel Color', fontsize=13)

# Old per-color skeletons
sk_r_old = skeletonize(r_mask); sk_y_old = skeletonize(y_mask); sk_b_old = skeletonize(b_mask)
old_on_gt = gt_raw.copy().astype(float) * 0.5
old_on_gt[cv2.dilate(sk_r_old.astype(np.uint8), dk4) > 0] = [230, 70, 70]
old_on_gt[cv2.dilate(sk_y_old.astype(np.uint8), dk4) > 0] = [240, 220, 50]
old_on_gt[cv2.dilate(sk_b_old.astype(np.uint8), dk4) > 0] = [70, 130, 240]
axes2[1,0].imshow(old_on_gt.astype(np.uint8))
axes2[1,0].set_title('OLD: Per-Color Skeletons on GT', fontsize=13)

new_skel_vis = np.zeros((*gt_raw.shape[:2], 3), dtype=np.uint8)
for c_label, c_rgb in [('R', [230,70,70]), ('Y', [240,220,50]), ('B', [70,130,240])]:
    new_skel_vis[cv2.dilate(colored_skel_mask[c_label].astype(np.uint8), dk3) > 0] = c_rgb
axes2[1,1].imshow(new_skel_vis)
axes2[1,1].set_title('NEW: Colored Base Skeleton', fontsize=13)

old_skel_vis = np.zeros_like(gt_raw)
old_skel_vis[cv2.dilate(sk_r_old.astype(np.uint8), dk3) > 0] = [230, 70, 70]
old_skel_vis[cv2.dilate(sk_y_old.astype(np.uint8), dk3) > 0] = [240, 220, 50]
old_skel_vis[cv2.dilate(sk_b_old.astype(np.uint8), dk3) > 0] = [70, 130, 240]
axes2[1,2].imshow(old_skel_vis)
axes2[1,2].set_title('OLD: Per-Color Skeletons', fontsize=13)

crop_y = int(h_img * 0.8)
vm = np.ones((h_img, w_img), dtype=bool); vm[crop_y:] = False
def count_metrics(skel):
    eps, jns = skel_features(skel)
    j = sum(1 for p in jns if p[0] < crop_y)
    e = sum(1 for p in eps if p[0] < crop_y)
    l = int(np.sum(skel & vm))
    return j, l, e

j_base, l_base, e_base = count_metrics(base_skel)
j_old = sum(count_metrics(sk)[0] for sk in [sk_r_old, sk_y_old, sk_b_old])
l_old = sum(count_metrics(sk)[1] for sk in [sk_r_old, sk_y_old, sk_b_old])
e_old = sum(count_metrics(sk)[2] for sk in [sk_r_old, sk_y_old, sk_b_old])
print(f"\nMetrics: New J={j_base} L={l_base} E={e_base} | Old J={j_old} L={l_old} E={e_old}")

for ax in axes2.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'plan_b_skeleton_sample1.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved plan_b_skeleton_sample1.png')
print('Done!')
