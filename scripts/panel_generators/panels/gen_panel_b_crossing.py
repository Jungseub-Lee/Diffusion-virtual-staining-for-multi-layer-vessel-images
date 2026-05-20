import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import binary_fill_holes, label

OUT = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output')
BASE = Path(r'C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data')

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

sid = '18-19-512'
gt_multi_dir = BASE/'Multi-color'/'Pix2pix'
gt_single_dir = BASE/'Single-color'/'Pix2pix'

bf = load(gt_multi_dir/f'{sid}_real_A.png')
gt_m = load(gt_multi_dir/f'{sid}_real_B.png')
gt_s = load(gt_single_dir/f'{sid}_real_B.png')

# Layer colors (BGR-ish for overlay): Red, Yellow, Blue
layer_colors_rgb = [(230, 70, 70), (240, 220, 50), (70, 130, 240)]

# === Single skeleton on BF ===
mask_s = get_vessel_mask(gt_s)
skel_s = skeletonize(mask_s)
# Dilate for visibility
k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
skel_s_thick = cv2.dilate(skel_s.astype(np.uint8)*255, k_d, iterations=1) > 0

bf_single = bf.copy()
bf_single[skel_s_thick] = (0, 200, 0)
Image.fromarray(bf_single).save(OUT/f'panel_b_{sid}_single.png')
print('Single skeleton saved')

# === Multi depth-separated skeleton on BF with proper crossing ===
bottom, middle, top = separate_layers_hf(gt_m)
masks = [bottom, middle, top]
skels = [skeletonize(m) for m in masks]

# Thickness for skeleton lines
thickness = 4
k_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness*2+1, thickness*2+1))

# Create thick skeleton masks per layer
skel_thick = [cv2.dilate(sk.astype(np.uint8)*255, k_thick, iterations=1) > 0 for sk in skels]

# Find overlap between red(bottom) and blue(top) thick skeletons
# These are the crossing regions where we need special handling
overlap_rb = skel_thick[0] & skel_thick[2]  # red & blue overlap
overlap_ry = skel_thick[0] & skel_thick[1]  # red & yellow overlap
overlap_yb = skel_thick[1] & skel_thick[2]  # yellow & blue overlap

# Strategy: Draw layers in order bottom(red) → middle(yellow) → top(blue)
# But at crossing points, we interleave to show both layers
# Method: checkerboard pattern at overlap regions

bf_multi = bf.copy()
h_img, w_img = bf.shape[:2]

# First pass: draw all three layers normally
# Order: red first (bottom), then yellow, then blue on top
for li in [0, 1, 2]:
    color = layer_colors_rgb[li]
    bf_multi[skel_thick[li]] = color

# Now handle red-blue crossings: create alternating pattern so both are visible
# Use a diagonal stripe pattern
stripe_period = 6  # pixels per stripe
yy, xx = np.mgrid[:h_img, :w_img]
# Diagonal stripes: (x + y) mod period < period/2 → red, else → blue
stripe_mask = ((xx + yy) % stripe_period) < (stripe_period // 2)

# At red-blue overlap: alternate between red and blue
rb_red_zone = overlap_rb & stripe_mask
rb_blue_zone = overlap_rb & ~stripe_mask
bf_multi[rb_red_zone] = layer_colors_rgb[0]  # red
bf_multi[rb_blue_zone] = layer_colors_rgb[2]  # blue

# At red-yellow overlap: alternate
ry_red_zone = overlap_ry & stripe_mask
ry_yel_zone = overlap_ry & ~stripe_mask
bf_multi[ry_red_zone] = layer_colors_rgb[0]  # red
bf_multi[ry_yel_zone] = layer_colors_rgb[1]  # yellow

# At yellow-blue overlap: alternate
yb_yel_zone = overlap_yb & stripe_mask
yb_blue_zone = overlap_yb & ~stripe_mask
bf_multi[yb_yel_zone] = layer_colors_rgb[1]  # yellow
bf_multi[yb_blue_zone] = layer_colors_rgb[2]  # blue

# Add thin outline around each skeleton for better separation
k_outline = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness*2+3, thickness*2+3))
for li in [0, 1, 2]:
    outline = cv2.dilate(skels[li].astype(np.uint8)*255, k_outline, iterations=1) > 0
    outline_only = outline & ~skel_thick[li]
    # Darken outline slightly for depth
    bf_multi[outline_only] = np.clip(bf_multi[outline_only].astype(np.int16) - 40, 0, 255).astype(np.uint8)

Image.fromarray(bf_multi).save(OUT/f'panel_b_{sid}_multi.png')
print('Multi depth-sep skeleton with crossing saved')

# Also save a zoomed version of a crossing region for verification
# Find largest overlap region
if np.sum(overlap_rb) > 0:
    ys, xs = np.where(overlap_rb)
    cy, cx = int(np.mean(ys)), int(np.mean(xs))
    crop_r = 60
    y0, y1 = max(0, cy-crop_r), min(h_img, cy+crop_r)
    x0, x1 = max(0, cx-crop_r), min(w_img, cx+crop_r)
    zoom = bf_multi[y0:y1, x0:x1]
    zoom_up = cv2.resize(zoom, (zoom.shape[1]*4, zoom.shape[0]*4), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(zoom_up).save(OUT/f'panel_b_{sid}_crossing_zoom.png')
    print(f'Crossing zoom saved (center: {cy},{cx}, overlap pixels: {np.sum(overlap_rb)})')
else:
    print('No red-blue overlap found')

# Print overlap stats
print(f'Red-Blue overlap: {np.sum(overlap_rb)} px')
print(f'Red-Yellow overlap: {np.sum(overlap_ry)} px')
print(f'Yellow-Blue overlap: {np.sum(overlap_yb)} px')
