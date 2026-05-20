"""
Heatmap-based vessel separation analysis.
Use the per-pixel average depth map to understand vessel boundaries.
Analyze hue gradient to distinguish natural transitions from crossings.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import label, gaussian_filter
import warnings
warnings.filterwarnings('ignore')

DATA = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data\Original color stack data')
OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')

# Load Sample 1 layers
s1_layers_img = np.array(Image.open(DATA / 'Sample 1_GT and 8 layer images(256x256 nine images).png'))
s1_bf_gt = np.array(Image.open(DATA / 'Sample 1_BF and GT(512x512 two images).png'))
gt_img = s1_bf_gt[512:]  # GT is bottom half
layers = [s1_layers_img[256*i:256*(i+1)] for i in range(1, 9)]

# Upscale layers to 512x512
layers_up = [cv2.resize(l, (512, 512), interpolation=cv2.INTER_LANCZOS4) for l in layers]

# ===== 1. Per-pixel depth map (weighted average layer) =====
h, w = 512, 512
depth_map = np.zeros((h, w), dtype=np.float64)
weight_sum = np.zeros((h, w), dtype=np.float64)

for i, layer in enumerate(layers_up):
    g = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY).astype(np.float64)
    # Use intensity as weight, layer index as depth
    depth_map += g * (i + 1)
    weight_sum += g

avg_depth = np.where(weight_sum > 10, depth_map / weight_sum, 0)

# ===== 2. Hue from GT image =====
hsv = cv2.cvtColor(gt_img, cv2.COLOR_RGB2HSV)
hue = hsv[:,:,0].astype(np.float32)
sat = hsv[:,:,1].astype(np.float32)
gray = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)
vessel_mask = gray > 15

# ===== 3. Hue gradient magnitude =====
# Handle hue wraparound (red at 0 and 179)
hue_rad = hue * np.pi / 90  # Convert to radians (0-180 → 0-2π)
hue_cos = np.cos(hue_rad) * vessel_mask
hue_sin = np.sin(hue_rad) * vessel_mask

# Gradient of cos and sin components
grad_cos_x = cv2.Sobel(hue_cos, cv2.CV_64F, 1, 0, ksize=5)
grad_cos_y = cv2.Sobel(hue_cos, cv2.CV_64F, 0, 1, ksize=5)
grad_sin_x = cv2.Sobel(hue_sin, cv2.CV_64F, 1, 0, ksize=5)
grad_sin_y = cv2.Sobel(hue_sin, cv2.CV_64F, 0, 1, ksize=5)

hue_gradient = np.sqrt(grad_cos_x**2 + grad_cos_y**2 + grad_sin_x**2 + grad_sin_y**2)
hue_gradient = hue_gradient * vessel_mask

# Normalize
hue_grad_norm = hue_gradient / (np.percentile(hue_gradient[vessel_mask], 99) + 1e-6)
hue_grad_norm = np.clip(hue_grad_norm, 0, 1)

# ===== 4. Depth gradient (from layer depth map) =====
depth_gradient_x = cv2.Sobel(avg_depth, cv2.CV_64F, 1, 0, ksize=5)
depth_gradient_y = cv2.Sobel(avg_depth, cv2.CV_64F, 0, 1, ksize=5)
depth_gradient = np.sqrt(depth_gradient_x**2 + depth_gradient_y**2)
depth_gradient = depth_gradient * (avg_depth > 0)

depth_grad_norm = depth_gradient / (np.percentile(depth_gradient[avg_depth > 0], 99) + 1e-6)
depth_grad_norm = np.clip(depth_grad_norm, 0, 1)

# ===== 5. Classification: high gradient = crossing boundary =====
# Smooth the gradient maps
hue_grad_smooth = gaussian_filter(hue_grad_norm, sigma=3) * vessel_mask
depth_grad_smooth = gaussian_filter(depth_grad_norm, sigma=3) * (avg_depth > 0)

# Crossing indicators: high hue gradient AND presence of both R and B nearby
r_mask = ((hue <= 12) | (hue >= 155)) & vessel_mask & (sat > 30)
b_mask = ((hue >= 85) & (hue <= 140)) & vessel_mask & (sat > 30)
y_mask = ((hue >= 13) & (hue <= 55)) & vessel_mask & (sat > 30)

k_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
r_nearby = cv2.dilate(r_mask.astype(np.uint8), k_near) > 0
b_nearby = cv2.dilate(b_mask.astype(np.uint8), k_near) > 0

# Crossing zone: high gradient + both R and B nearby
crossing_candidate = (hue_grad_smooth > 0.3) & r_nearby & b_nearby & vessel_mask

# Transition zone: gradual change with Y in between
y_nearby = cv2.dilate(y_mask.astype(np.uint8), k_near) > 0
transition_candidate = (hue_grad_smooth > 0.1) & (hue_grad_smooth < 0.5) & y_nearby & vessel_mask

# ===== 6. Visualization =====
fig, axes = plt.subplots(3, 4, figsize=(28, 18))
fig.suptitle('Sample 1: Heatmap-based Vessel Analysis', fontsize=16, fontweight='bold')

# Row 1: Basic maps
axes[0,0].imshow(gt_img); axes[0,0].set_title('Original GT', fontsize=12)

im1 = axes[0,1].imshow(avg_depth, cmap='coolwarm', vmin=1, vmax=8)
axes[0,1].imshow(1 - (avg_depth > 0).astype(float), cmap='gray', alpha=0.5*(avg_depth==0))
plt.colorbar(im1, ax=axes[0,1], fraction=0.046, shrink=0.8)
axes[0,1].set_title('Depth Map (1=Top, 8=Bottom)', fontsize=12)

# Hue map (shifted for visualization)
hue_vis = np.zeros((*hue.shape, 3))
hue_vis[r_mask] = [1, 0.3, 0.3]
hue_vis[y_mask] = [1, 1, 0.3]
hue_vis[b_mask] = [0.3, 0.5, 1]
axes[0,2].imshow(hue_vis); axes[0,2].set_title('Hue Classification (R/Y/B)', fontsize=12)

# Layer composition (how many layers contribute per pixel)
layer_count = np.zeros((h, w))
for layer in layers_up:
    g = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
    layer_count += (g > 15).astype(float)
im3 = axes[0,3].imshow(layer_count, cmap='hot', vmin=0, vmax=8)
plt.colorbar(im3, ax=axes[0,3], fraction=0.046, shrink=0.8)
axes[0,3].set_title('Layer Count per Pixel\n(>1 = overlap)', fontsize=12)

# Row 2: Gradients
im4 = axes[1,0].imshow(hue_grad_smooth, cmap='hot', vmin=0, vmax=0.8)
plt.colorbar(im4, ax=axes[1,0], fraction=0.046, shrink=0.8)
axes[1,0].set_title('Hue Gradient (smoothed)\nHigh = abrupt color change', fontsize=11)

im5 = axes[1,1].imshow(depth_grad_smooth, cmap='hot', vmin=0, vmax=0.8)
plt.colorbar(im5, ax=axes[1,1], fraction=0.046, shrink=0.8)
axes[1,1].set_title('Depth Gradient (smoothed)\nHigh = depth jump', fontsize=11)

# Combined gradient
combined_grad = (hue_grad_smooth * 0.5 + depth_grad_smooth * 0.5) * vessel_mask
im6 = axes[1,2].imshow(combined_grad, cmap='hot', vmin=0, vmax=0.8)
plt.colorbar(im6, ax=axes[1,2], fraction=0.046, shrink=0.8)
axes[1,2].set_title('Combined Gradient', fontsize=12)

# Crossing vs transition overlay
overlay = gt_img.copy().astype(np.float64) / 255
crossing_vis = np.zeros((*gt_img.shape[:2], 4))
crossing_vis[crossing_candidate] = [1, 0, 1, 0.5]  # magenta
crossing_vis[transition_candidate & ~crossing_candidate] = [0, 1, 0, 0.3]  # green
axes[1,3].imshow(overlay)
axes[1,3].imshow(crossing_vis)
axes[1,3].set_title('Crossing (magenta) vs Transition (green)\non Original', fontsize=11)

# Row 3: Contour-based approach
# Use depth map to extract contours at different depth levels
axes[2,0].imshow(gt_img)
contour_colors = [(0,0,255), (0,255,255), (0,255,0), (255,255,0), (255,165,0), (255,0,0)]
depth_levels = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
for dl, color in zip(depth_levels, contour_colors):
    level_mask = (avg_depth > dl - 0.5) & (avg_depth <= dl + 0.5)
    level_u8 = (level_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(level_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis_temp = np.zeros_like(gt_img)
    cv2.drawContours(vis_temp, contours, -1, color, 1)
    axes[2,0].imshow(vis_temp, alpha=0.4*(vis_temp.sum(axis=2)>0).astype(float))
axes[2,0].set_title('Depth Contours on GT', fontsize=12)

# Layer overlap regions
overlap_map = np.zeros_like(gt_img)
# Top layers (1-3) vs Bottom layers (6-8) overlap
top_presence = np.zeros((h,w))
for i in [0,1,2]:
    top_presence += (cv2.cvtColor(layers_up[i], cv2.COLOR_RGB2GRAY) > 15).astype(float)
bot_presence = np.zeros((h,w))
for i in [5,6,7]:
    bot_presence += (cv2.cvtColor(layers_up[i], cv2.COLOR_RGB2GRAY) > 15).astype(float)

top_only = (top_presence > 0) & (bot_presence == 0)
bot_only = (bot_presence > 0) & (top_presence == 0)
both = (top_presence > 0) & (bot_presence > 0)

overlap_map[top_only] = [70, 130, 240]
overlap_map[bot_only] = [230, 70, 70]
overlap_map[both] = [255, 0, 255]
axes[2,1].imshow(overlap_map)
axes[2,1].set_title(f'Top(blue) vs Bottom(red) overlap\nMagenta = both present ({np.sum(both)}px)', fontsize=11)

# High-gradient contours overlaid on masks
axes[2,2].imshow(hue_vis)
high_grad = hue_grad_smooth > 0.35
contour_img = np.zeros((h, w), dtype=np.uint8)
contour_img[high_grad & vessel_mask] = 255
contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    pts = c.squeeze()
    if len(pts.shape) == 2 and len(pts) > 5:
        for pt in pts:
            axes[2,2].plot(pt[0], pt[1], 'w.', ms=1)
axes[2,2].set_title('High Gradient Boundaries\non Hue Classification', fontsize=11)

# Proposed smart boundary: use gradient to refine crossing zones
# At crossing zones, split vessels based on which depth they belong to
smart_r = r_mask.copy()
smart_b = b_mask.copy()
# In crossing zones (both R and B nearby + high gradient),
# assign pixels to the layer with the closest depth
crossing_pixels = crossing_candidate & vessel_mask
for y_px in range(h):
    for x_px in range(w):
        if not crossing_pixels[y_px, x_px]:
            continue
        d = avg_depth[y_px, x_px]
        if d > 4.5:  # closer to bottom
            smart_r[y_px, x_px] = True
            smart_b[y_px, x_px] = False
        elif d < 4.5:  # closer to top
            smart_b[y_px, x_px] = True
            smart_r[y_px, x_px] = False

smart_vis = np.zeros_like(gt_img)
smart_vis[smart_r] = [230, 70, 70]
smart_vis[y_mask] = [240, 220, 50]
smart_vis[smart_b] = [70, 130, 240]
axes[2,3].imshow(smart_vis)
axes[2,3].set_title('Depth-refined Masks\n(crossing pixels reassigned by depth)', fontsize=11)

for ax_row in axes:
    for ax in ax_row:
        ax.axis('off')

plt.tight_layout()
plt.savefig(OUT / 'heatmap_analysis_sample1.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved heatmap_analysis_sample1.png')

# ===== Additional: zoomed views of crossing regions =====
# Find the most prominent crossing regions
crossing_labeled, n_cross = label(crossing_candidate)
crossing_sizes = [(i, np.sum(crossing_labeled == i)) for i in range(1, n_cross + 1)]
crossing_sizes.sort(key=lambda x: -x[1])

n_zoom = min(4, len(crossing_sizes))
if n_zoom > 0:
    fig2, axes2 = plt.subplots(n_zoom, 5, figsize=(25, 5*n_zoom))
    if n_zoom == 1:
        axes2 = axes2[np.newaxis, :]
    fig2.suptitle('Zoomed Crossing Regions', fontsize=14, fontweight='bold')

    for zi in range(n_zoom):
        idx = crossing_sizes[zi][0]
        ys, xs = np.where(crossing_labeled == idx)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        pad = 60
        y0, y1 = max(0, cy-pad), min(h, cy+pad)
        x0, x1 = max(0, cx-pad), min(w, cx+pad)

        axes2[zi,0].imshow(gt_img[y0:y1, x0:x1])
        axes2[zi,0].set_title(f'GT (region {zi+1})', fontsize=10)

        axes2[zi,1].imshow(avg_depth[y0:y1, x0:x1], cmap='coolwarm', vmin=1, vmax=8)
        axes2[zi,1].set_title('Depth Map', fontsize=10)

        axes2[zi,2].imshow(hue_grad_smooth[y0:y1, x0:x1], cmap='hot', vmin=0, vmax=0.8)
        axes2[zi,2].set_title('Hue Gradient', fontsize=10)

        axes2[zi,3].imshow(hue_vis[y0:y1, x0:x1])
        cross_z = np.zeros((y1-y0, x1-x0, 4))
        cross_z[crossing_candidate[y0:y1, x0:x1]] = [1, 0, 1, 0.5]
        axes2[zi,3].imshow(cross_z)
        axes2[zi,3].set_title('Crossing zone (magenta)', fontsize=10)

        # Layer overlap in this region
        ov_z = np.zeros((y1-y0, x1-x0, 3))
        ov_z[top_only[y0:y1, x0:x1]] = [0.3, 0.5, 1]
        ov_z[bot_only[y0:y1, x0:x1]] = [1, 0.3, 0.3]
        ov_z[both[y0:y1, x0:x1]] = [1, 0, 1]
        axes2[zi,4].imshow(ov_z)
        axes2[zi,4].set_title('Top/Bot overlap', fontsize=10)

        for ax in axes2[zi]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUT / 'heatmap_crossing_zoom_sample1.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved heatmap_crossing_zoom_sample1.png')

print('Done!')
