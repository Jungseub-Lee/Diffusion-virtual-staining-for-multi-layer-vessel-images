"""
하단 20%: base skel (single color, 높낮이 구분 X)
상단 80%: R/B separated skel
경계에서 자연스럽게 연결되는지 확인
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
total = R + G + B + 1e-6
b_ratio = B / total
r_ratio = R / total

# Zone boundary
boundary_y = int(h * 0.8)  # 상단 80% = 0~boundary_y, 하단 20% = boundary_y~h

# ===== All skeletons =====
# Base skel (전체)
vessel_mask = gray > 10
vessel_mask = cv2.morphologyEx(vessel_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
vessel_mask = remove_small_objects(vessel_mask, min_size=100)
base_skel = skeletonize(vessel_mask)

# R skel
r_mask = R > 30
r_mask = cv2.morphologyEx(r_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
r_mask = remove_small_objects(r_mask, min_size=50)
r_skel = skeletonize(r_mask)

# B skel (hysteresis)
b_seed = B > 50
b_low = B > 20
b_low_l, n_l = ndimage.label(b_low)
b_hyst = np.zeros_like(b_low, dtype=bool)
for i in range(1, n_l + 1):
    region = b_low_l == i
    if np.any(b_seed[region]):
        b_hyst |= region
b_hyst = cv2.morphologyEx(b_hyst.astype(np.uint8)*255, cv2.MORPH_CLOSE, k5, iterations=2) > 0
b_hyst = remove_small_objects(b_hyst, min_size=50)
b_skel = skeletonize(b_hyst)

# Ratio 필터 (상단 80%에만 적용)
ratio_thresh = 0.35
b_filt = b_skel & (b_ratio >= ratio_thresh)
r_filt = r_skel & (r_ratio >= ratio_thresh)

# ===== 경계 연결 분석 =====
# 경계선 근처 (+/- 10px)에서 각 skeleton의 pixel 위치 확인
margin = 10
boundary_zone = slice(boundary_y - margin, boundary_y + margin)

base_at_boundary = np.argwhere(base_skel[boundary_zone, :])
r_at_boundary = np.argwhere(r_filt[boundary_zone, :])
b_at_boundary = np.argwhere(b_filt[boundary_zone, :])
r_full_at_boundary = np.argwhere(r_skel[boundary_zone, :])
b_full_at_boundary = np.argwhere(b_skel[boundary_zone, :])

print(f"Boundary zone (y={boundary_y}±{margin}):")
print(f"  Base skel pixels: {len(base_at_boundary)}")
print(f"  R skel (full) pixels: {len(r_full_at_boundary)}")
print(f"  B skel (full) pixels: {len(b_full_at_boundary)}")
print(f"  R skel (ratio filtered) pixels: {len(r_at_boundary)}")
print(f"  B skel (ratio filtered) pixels: {len(b_at_boundary)}")

# ===== Composite skeleton: 하단=base, 상단=R/B =====
# Method A: 단순 zone split
composite_r = np.zeros((h, w), dtype=bool)
composite_b = np.zeros((h, w), dtype=bool)
composite_base = np.zeros((h, w), dtype=bool)

# 상단 80%: R/B filtered skel
composite_r[:boundary_y, :] = r_filt[:boundary_y, :]
composite_b[:boundary_y, :] = b_filt[:boundary_y, :]
# 하단 20%: base skel
composite_base[boundary_y:, :] = base_skel[boundary_y:, :]

# Method B: transition zone 포함 — 경계 근처에서 R/B skel이 base skel과 만나는지
# 상단: R/B filtered, 하단: base, 경계 근처: R/B full skel로 bridge
transition_margin = 15
composite_r_b = np.zeros((h, w), dtype=bool)
composite_b_b = np.zeros((h, w), dtype=bool)
composite_base_b = np.zeros((h, w), dtype=bool)

# 상단 (0 ~ boundary-transition): R/B filtered
composite_r_b[:boundary_y - transition_margin, :] = r_filt[:boundary_y - transition_margin, :]
composite_b_b[:boundary_y - transition_margin, :] = b_filt[:boundary_y - transition_margin, :]
# Transition zone: R/B full skel (ratio filter 없이)
composite_r_b[boundary_y - transition_margin:boundary_y, :] = \
    r_skel[boundary_y - transition_margin:boundary_y, :]
composite_b_b[boundary_y - transition_margin:boundary_y, :] = \
    b_skel[boundary_y - transition_margin:boundary_y, :]
# 하단: base skel
composite_base_b[boundary_y:, :] = base_skel[boundary_y:, :]

# ===== Figure 1: Zone connection 분석 =====
fig, axes = plt.subplots(2, 4, figsize=(36, 16))
fig.suptitle('Zone Connection: Bottom 20% base skel ↔ Top 80% R/B skel', fontsize=16, fontweight='bold')

# Panel 1: GT with boundary line
gt_vis = gt_raw.copy()
cv2.line(gt_vis, (0, boundary_y), (w, boundary_y), (255, 255, 0), 2)
axes[0,0].imshow(gt_vis)
axes[0,0].set_title(f'GT + boundary (y={boundary_y})\ntop 80% = multi, bottom 20% = single', fontsize=11)

# Panel 2: All skeletons overlaid with boundary
all_vis = np.zeros((h, w, 3), dtype=np.uint8)
# base skel everywhere (dim)
all_vis[cv2.dilate(base_skel.astype(np.uint8), dk3) > 0] = [60, 60, 60]
# R/B skel
all_vis[cv2.dilate(r_skel.astype(np.uint8), dk3) > 0] = [150, 50, 50]
all_vis[cv2.dilate(b_skel.astype(np.uint8), dk3) > 0] = [50, 80, 150]
# boundary line
all_vis[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[0,1].imshow(all_vis)
axes[0,1].set_title('All skels: base(gray) R(red) B(blue)\nyellow=boundary', fontsize=11)

# Panel 3: Zoom at boundary
zoom_y0 = boundary_y - 40
zoom_y1 = min(h, boundary_y + 40)
zoom_vis = gt_raw[zoom_y0:zoom_y1, :].copy().astype(float) * 0.3
# base skel
base_z = cv2.dilate(base_skel[zoom_y0:zoom_y1, :].astype(np.uint8), dk4) > 0
zoom_vis[base_z] = [0, 200, 0]
# R skel
r_z = cv2.dilate(r_filt[zoom_y0:zoom_y1, :].astype(np.uint8), dk4) > 0
zoom_vis[r_z] = [255, 80, 80]
# B skel
b_z = cv2.dilate(b_filt[zoom_y0:zoom_y1, :].astype(np.uint8), dk4) > 0
zoom_vis[b_z] = [80, 150, 255]
# boundary
bl = boundary_y - zoom_y0
if 0 <= bl < zoom_y1 - zoom_y0:
    zoom_vis[bl-1:bl+1, :] = [255, 255, 0]
axes[0,2].imshow(zoom_vis.astype(np.uint8))
axes[0,2].set_title('Zoom at boundary\nbase(green) R(red) B(blue)', fontsize=11)

# Panel 4: x좌표별 skeleton 존재 여부 at boundary
boundary_profile = np.zeros((3, w), dtype=bool)
for x in range(w):
    if base_skel[boundary_y, x]:
        boundary_profile[0, x] = True
    if r_skel[boundary_y, x]:
        boundary_profile[1, x] = True
    if b_skel[boundary_y, x]:
        boundary_profile[2, x] = True

axes[0,3].barh([2], [np.sum(boundary_profile[0])], color='green', alpha=0.7, label='base')
axes[0,3].barh([1], [np.sum(boundary_profile[1])], color='red', alpha=0.7, label='R skel')
axes[0,3].barh([0], [np.sum(boundary_profile[2])], color='blue', alpha=0.7, label='B skel')
axes[0,3].set_yticks([0, 1, 2])
axes[0,3].set_yticklabels(['B skel', 'R skel', 'Base'])
axes[0,3].set_title(f'Skel pixels at y={boundary_y}\nbase={np.sum(boundary_profile[0])} R={np.sum(boundary_profile[1])} B={np.sum(boundary_profile[2])}', fontsize=11)
axes[0,3].legend()

# --- Row 2: Composite methods ---

# Method A: 단순 split
comp_a = gt_raw.copy().astype(float) * 0.3
comp_a[cv2.dilate(composite_r.astype(np.uint8), dk4) > 0] = [255, 80, 80]
comp_a[cv2.dilate(composite_b.astype(np.uint8), dk4) > 0] = [80, 150, 255]
comp_a[cv2.dilate(composite_base.astype(np.uint8), dk4) > 0] = [0, 220, 0]
comp_a[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[1,0].imshow(comp_a.astype(np.uint8))
axes[1,0].set_title('Method A: simple split\ntop=R/B filtered, bottom=base', fontsize=11)

# Method B: transition zone
comp_b = gt_raw.copy().astype(float) * 0.3
comp_b[cv2.dilate(composite_r_b.astype(np.uint8), dk4) > 0] = [255, 80, 80]
comp_b[cv2.dilate(composite_b_b.astype(np.uint8), dk4) > 0] = [80, 150, 255]
comp_b[cv2.dilate(composite_base_b.astype(np.uint8), dk4) > 0] = [0, 220, 0]
# transition zone 표시
comp_b[boundary_y-transition_margin-1:boundary_y-transition_margin+1, :] = [200, 200, 0]
comp_b[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[1,1].imshow(comp_b.astype(np.uint8))
axes[1,1].set_title(f'Method B: transition zone ({transition_margin}px)\ntop=filtered, trans=full, bottom=base', fontsize=11)

# Method C: base skel 전체 유지 + 상단에서 R/B 색칠
# base skel의 topology를 유지하면서 상단 pixel만 R/B로 색칠
comp_c_vis = gt_raw.copy().astype(float) * 0.3
# 하단: base skel (green)
base_lower = base_skel.copy()
base_lower[:boundary_y, :] = False
comp_c_vis[cv2.dilate(base_lower.astype(np.uint8), dk4) > 0] = [0, 220, 0]

# 상단: base skel pixel을 R/B ratio로 색칠
base_upper = base_skel.copy()
base_upper[boundary_y:, :] = False
for y, x in np.argwhere(base_upper):
    r_val = r_ratio[y, x]
    b_val = b_ratio[y, x]
    if r_val > b_val:
        comp_c_vis[y, x] = [255, 80, 80]
    else:
        comp_c_vis[y, x] = [80, 150, 255]

# dilate로 보이게
base_upper_r = base_upper & (r_ratio > b_ratio)
base_upper_b = base_upper & (b_ratio >= r_ratio)
comp_c_vis[cv2.dilate(base_upper_r.astype(np.uint8), dk4) > 0] = [255, 80, 80]
comp_c_vis[cv2.dilate(base_upper_b.astype(np.uint8), dk4) > 0] = [80, 150, 255]
comp_c_vis[cv2.dilate(base_lower.astype(np.uint8), dk4) > 0] = [0, 220, 0]
comp_c_vis[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes[1,2].imshow(comp_c_vis.astype(np.uint8))
axes[1,2].set_title('Method C: base skel topology\ntop=colored by ratio, bottom=green', fontsize=11)

# Comparison: all 3 methods connectivity check
# 경계에서 끊김 개수 세기
def count_boundary_breaks(upper_skel, lower_skel, by):
    """경계선에서 upper와 lower가 연결되지 않는 곳 세기"""
    breaks = 0
    connections = 0
    # upper에서 경계 직전(by-1) pixel
    for x in range(w):
        upper_exists = False
        lower_exists = False
        for dy in range(1, 6):  # 경계 위 5px 이내
            if by - dy >= 0 and upper_skel[by - dy, x]:
                upper_exists = True
                break
        for dy in range(0, 6):  # 경계 아래 5px 이내
            if by + dy < h and lower_skel[by + dy, x]:
                lower_exists = True
                break
        if upper_exists and lower_exists:
            connections += 1
        elif upper_exists and not lower_exists:
            breaks += 1
        elif not upper_exists and lower_exists:
            breaks += 1
    return connections, breaks

# Method A breaks
upper_a = composite_r | composite_b
conn_a, brk_a = count_boundary_breaks(upper_a, composite_base, boundary_y)

# Method B breaks
upper_b = composite_r_b | composite_b_b
conn_b, brk_b = count_boundary_breaks(upper_b, composite_base_b, boundary_y)

# Method C: base skel이므로 끊김 없음
conn_c = int(np.sum(base_skel[boundary_y, :]))

axes[1,3].axis('off')
txt = (f"=== Boundary Connection ===\n\n"
       f"Method A (simple split):\n"
       f"  connections={conn_a}, breaks={brk_a}\n\n"
       f"Method B (transition zone):\n"
       f"  connections={conn_b}, breaks={brk_b}\n\n"
       f"Method C (base skel + color):\n"
       f"  continuous (base skel)\n"
       f"  boundary pixels={conn_c}\n\n"
       f"→ Method C는 base skel 기반이라\n"
       f"  끊김 자체가 불가능")
axes[1,3].text(0.05, 0.85, txt, fontsize=13, transform=axes[1,3].transAxes,
               fontfamily='monospace', color='white', va='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

for ax in axes.flat:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'skel_zone_connection.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved skel_zone_connection.png')

# ===== Figure 2: Method C 상세 — base skel + ratio coloring =====
fig2, axes2 = plt.subplots(1, 4, figsize=(36, 8))
fig2.suptitle('Method C Detail: Base skeleton colored by R/B ratio (top 80%)', fontsize=16, fontweight='bold')

# Full base skel colored
full_colored = gt_raw.copy().astype(float) * 0.3
base_r = base_skel & (r_ratio > b_ratio)
base_b = base_skel & (b_ratio >= r_ratio)

# 상단만 색칠
base_r_upper = base_r.copy(); base_r_upper[boundary_y:, :] = False
base_b_upper = base_b.copy(); base_b_upper[boundary_y:, :] = False
base_lower_only = base_skel.copy(); base_lower_only[:boundary_y, :] = False

full_colored[cv2.dilate(base_r_upper.astype(np.uint8), dk4) > 0] = [255, 80, 80]
full_colored[cv2.dilate(base_b_upper.astype(np.uint8), dk4) > 0] = [80, 150, 255]
full_colored[cv2.dilate(base_lower_only.astype(np.uint8), dk4) > 0] = [0, 220, 0]
full_colored[boundary_y-1:boundary_y+1, :] = [255, 255, 0]
axes2[0].imshow(full_colored.astype(np.uint8))
axes2[0].set_title('Base skel: top colored, bottom green', fontsize=12)

# R ratio on base skel
r_ratio_skel = np.zeros((h, w), dtype=np.float32)
r_ratio_skel[base_skel] = r_ratio[base_skel]
r_d = cv2.dilate(r_ratio_skel, dk4)
r_d[cv2.dilate(base_skel.astype(np.uint8), dk4) == 0] = np.nan
axes2[1].imshow(r_d, cmap='Reds', vmin=0, vmax=0.6)
axes2[1].set_title('R ratio on base skel', fontsize=12)

# B ratio on base skel
b_ratio_skel = np.zeros((h, w), dtype=np.float32)
b_ratio_skel[base_skel] = b_ratio[base_skel]
b_d = cv2.dilate(b_ratio_skel, dk4)
b_d[cv2.dilate(base_skel.astype(np.uint8), dk4) == 0] = np.nan
axes2[2].imshow(b_d, cmap='Blues', vmin=0, vmax=0.6)
axes2[2].set_title('B ratio on base skel', fontsize=12)

# GT ref
axes2[3].imshow(gt_raw)
axes2[3].set_title('GT original', fontsize=12)

for ax in axes2:
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'skel_method_c_detail.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved skel_method_c_detail.png')
print('Done!')
