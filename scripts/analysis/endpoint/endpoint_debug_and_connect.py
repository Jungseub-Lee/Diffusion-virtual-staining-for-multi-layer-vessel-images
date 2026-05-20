"""
Part 1: Visualize endpoint correction (boundary vs true endpoints)
Part 2: Improve layer connection by bridging nearby endpoints across layers
Part 3: Pick best representative sample for paper figure
"""

import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\[1] Data")
output_dir = Path(r"C:\Users\seub1\Desktop\[Paper] Diffusion virtual staining paper\analysis_output")

# Representative samples with varying vessel density
vis_samples = ["1-0-512", "1-5-512", "1-10-512", "1-15-512"]


def detect_baseline_row(img_rgb, threshold_ratio=0.35):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    vessel = gray > 15
    h, w = vessel.shape
    row_density = np.sum(vessel, axis=1) / w
    baseline_top = h
    for y in range(h - 1, -1, -1):
        if row_density[y] < threshold_ratio:
            baseline_top = y
            break
    return max(0, baseline_top - 5)


def separate_layers_hsv(img_rgb, min_area=50):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fg = (gray > 15) & (s > 30)
    bottom = ((h <= 12) | (h >= 155)) & fg
    middle = ((h >= 13) & (h <= 55)) & fg
    top = ((h >= 85) & (h <= 140)) & fg
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    def clean(m):
        u = m.astype(np.uint8) * 255
        u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k, iterations=2)
        u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k, iterations=1)
        return remove_small_objects(u > 0, min_size=min_area)
    return clean(bottom), clean(middle), clean(top)


def get_vessel_mask(img_rgb, min_area=50):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    vessel = gray > 15
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    u = vessel.astype(np.uint8) * 255
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, k, iterations=2)
    u = cv2.morphologyEx(u, cv2.MORPH_OPEN, k, iterations=1)
    return remove_small_objects(u > 0, min_size=min_area)


def find_endpoints_and_junctions(skeleton):
    skel_u8 = skeleton.astype(np.uint8)
    kernel_sum = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel_u8, -1, kernel_sum) * skel_u8
    ep_coords = np.argwhere(neighbors == 1)   # (y, x)
    jn_coords = np.argwhere(neighbors >= 3)   # (y, x)
    return ep_coords, jn_coords, neighbors


def bridge_layer_skeletons(skels, masks, bridge_radius=15):
    """
    Connect skeletons across layers by drawing bridges between
    nearby endpoints of adjacent layers.

    Adjacent pairs: bottom<->middle, middle<->top

    Returns: connected skeleton (union), list of bridge lines drawn
    """
    layer_names = ["bottom", "middle", "top"]
    adjacent_pairs = [(0, 1), (1, 2)]  # bottom-middle, middle-top

    # Start with combined skeleton
    connected = np.zeros_like(skels[0], dtype=np.uint8)
    for s in skels:
        connected |= s.astype(np.uint8)

    bridges = []  # store bridge info for visualization

    for li, lj in adjacent_pairs:
        ep_i, _, _ = find_endpoints_and_junctions(skels[li])
        ep_j, _, _ = find_endpoints_and_junctions(skels[lj])

        if len(ep_i) == 0 or len(ep_j) == 0:
            continue

        # Also check: endpoint must be near the OTHER layer's mask (not just skeleton)
        # This ensures we only bridge where layers actually meet
        other_mask_dilated_j = cv2.dilate(masks[lj].astype(np.uint8)*255,
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_radius*2+1, bridge_radius*2+1)),
                                          iterations=1) > 0
        other_mask_dilated_i = cv2.dilate(masks[li].astype(np.uint8)*255,
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_radius*2+1, bridge_radius*2+1)),
                                          iterations=1) > 0

        # Filter endpoints of layer i that are near layer j's mask
        valid_ep_i = []
        for y, x in ep_i:
            if other_mask_dilated_j[y, x]:
                valid_ep_i.append([y, x])
        valid_ep_i = np.array(valid_ep_i) if valid_ep_i else np.empty((0, 2))

        # Filter endpoints of layer j that are near layer i's mask
        valid_ep_j = []
        for y, x in ep_j:
            if other_mask_dilated_i[y, x]:
                valid_ep_j.append([y, x])
        valid_ep_j = np.array(valid_ep_j) if valid_ep_j else np.empty((0, 2))

        if len(valid_ep_i) == 0 or len(valid_ep_j) == 0:
            continue

        # Find closest pairs within bridge_radius
        dists = cdist(valid_ep_i, valid_ep_j)

        used_i = set()
        used_j = set()

        # Greedy matching: closest pairs first
        flat_indices = np.argsort(dists.ravel())
        for idx in flat_indices:
            ii = idx // dists.shape[1]
            jj = idx % dists.shape[1]
            d = dists[ii, jj]

            if d > bridge_radius:
                break
            if ii in used_i or jj in used_j:
                continue

            # Draw bridge line on connected skeleton
            y1, x1 = int(valid_ep_i[ii][0]), int(valid_ep_i[ii][1])
            y2, x2 = int(valid_ep_j[jj][0]), int(valid_ep_j[jj][1])
            cv2.line(connected, (x1, y1), (x2, y2), 1, thickness=1)

            bridges.append({
                "from_layer": layer_names[li],
                "to_layer": layer_names[lj],
                "from": (y1, x1),
                "to": (y2, x2),
                "distance": float(d)
            })

            used_i.add(ii)
            used_j.add(jj)

    # Re-skeletonize the connected result to clean up
    connected_skel = skeletonize(connected > 0)
    return connected_skel, bridges


def compute_metrics(skeleton):
    total_length = np.sum(skeleton)
    if total_length == 0:
        return {"length": 0, "junctions": 0, "endpoints": 0}
    skel_u8 = skeleton.astype(np.uint8)
    kernel_sum = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel_u8, -1, kernel_sum) * skel_u8
    return {
        "length": int(total_length),
        "junctions": int(np.sum(neighbors >= 3)),
        "endpoints": int(np.sum(neighbors == 1))
    }


def dilate(s, t=4):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
    return cv2.dilate(s.astype(np.uint8)*255, k, iterations=1) > 0


# ============================================================
# Part 1 & 2: Visualize endpoint correction + bridging
# ============================================================
gt_multi_dir = BASE / "Multi-color" / "Pix2pix"

fig, axes = plt.subplots(len(vis_samples), 5, figsize=(30, 7 * len(vis_samples)))

for row, sid in enumerate(vis_samples):
    img = np.array(Image.open(gt_multi_dir / f"{sid}_real_B.png").convert("RGB"))
    baseline_y = detect_baseline_row(img)
    crop = img[:baseline_y, :]
    dim = (crop * 0.25).astype(np.uint8)

    bottom, middle, top = separate_layers_hsv(crop)
    masks = [bottom, middle, top]

    skel_b = skeletonize(bottom)
    skel_m = skeletonize(middle)
    skel_t = skeletonize(top)
    skels = [skel_b, skel_m, skel_t]

    layer_colors = [(255,80,80), (255,255,60), (80,140,255)]
    layer_names = ["Bottom", "Middle", "Top"]

    # --- Col 0: Original ---
    axes[row, 0].imshow(crop)
    axes[row, 0].set_title(f"{sid}\nOriginal GT", fontsize=12)

    # --- Col 1: Per-layer skeletons with ALL endpoints marked ---
    vis1 = dim.copy()
    for si, skel in enumerate(skels):
        thick = dilate(skel, 4)
        vis1[thick] = layer_colors[si]
        ep, jn, _ = find_endpoints_and_junctions(skel)
        for y, x in ep:
            cv2.circle(vis1, (x, y), 6, (255, 255, 255), 2)  # white circle = all endpoints

    m_raw = sum(compute_metrics(s)["endpoints"] for s in skels)
    axes[row, 1].imshow(vis1)
    axes[row, 1].set_title(f"Layer Skeletons\nAll endpoints (white): {m_raw}", fontsize=12)

    # --- Col 2: Endpoints classified (true=green, boundary=white hollow) ---
    vis2 = dim.copy()
    for si, skel in enumerate(skels):
        thick = dilate(skel, 4)
        vis2[thick] = layer_colors[si]

    true_ep_total = 0
    boundary_ep_total = 0
    for si, skel in enumerate(skels):
        other_masks = [masks[j] for j in range(3) if j != si]
        other_union = np.zeros_like(masks[0], dtype=bool)
        for om in other_masks:
            other_union |= om
        other_dilated = cv2.dilate(other_union.astype(np.uint8)*255,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
                                   iterations=1) > 0

        ep, _, _ = find_endpoints_and_junctions(skel)
        for y, x in ep:
            if other_dilated[y, x]:
                # Boundary endpoint (near other layer) - mark as hollow red X
                cv2.drawMarker(vis2, (x, y), (255, 100, 100), cv2.MARKER_TILTED_CROSS, 10, 2)
                boundary_ep_total += 1
            else:
                # True endpoint - mark as solid green circle
                cv2.circle(vis2, (x, y), 6, (0, 255, 0), -1)
                true_ep_total += 1

    axes[row, 2].imshow(vis2)
    axes[row, 2].set_title(f"Endpoint Classification\nTrue (green): {true_ep_total}, Boundary (red X): {boundary_ep_total}", fontsize=12)

    # --- Col 3: After bridging ---
    connected_skel, bridges = bridge_layer_skeletons(skels, masks, bridge_radius=15)
    vis3 = dim.copy()

    # Draw connected skeleton in white first
    thick_conn = dilate(connected_skel, 4)

    # Color the connected skeleton by original layer membership
    for si, mask in enumerate(masks):
        # Pixels of connected skeleton that fall within this layer's mask
        layer_region = thick_conn & mask
        vis3[layer_region] = layer_colors[si]

    # Remaining (bridged regions not in any mask) in cyan
    uncolored = thick_conn & ~(masks[0] | masks[1] | masks[2])
    vis3[uncolored] = (0, 255, 200)

    # Draw bridge lines prominently
    for b in bridges:
        y1, x1 = b["from"]
        y2, x2 = b["to"]
        cv2.line(vis3, (x1, y1), (x2, y2), (0, 255, 200), 3)
        cv2.circle(vis3, (x1, y1), 5, (255, 255, 255), -1)
        cv2.circle(vis3, (x2, y2), 5, (255, 255, 255), -1)

    m_connected = compute_metrics(connected_skel)
    axes[row, 3].imshow(vis3)
    axes[row, 3].set_title(f"After Bridging ({len(bridges)} bridges)\n"
                           f"J={m_connected['junctions']}, E={m_connected['endpoints']}, L={m_connected['length']}",
                           fontsize=12)

    # --- Col 4: Comparison summary ---
    # Single-color skeleton for reference
    single_img = np.array(Image.open(BASE / "Single-color" / "Pix2pix" / f"{sid}_real_B.png").convert("RGB"))
    single_crop = single_img[:baseline_y, :]
    single_mask = get_vessel_mask(single_crop)
    skel_single = skeletonize(single_mask)
    m_single = compute_metrics(skel_single)

    dim_s = (single_crop * 0.25).astype(np.uint8)
    vis4 = dim_s.copy()
    vis4[dilate(skel_single, 4)] = (0, 255, 0)

    axes[row, 4].imshow(vis4)
    axes[row, 4].set_title(f"Single-color Skeleton\n"
                           f"J={m_single['junctions']}, E={m_single['endpoints']}, L={m_single['length']}",
                           fontsize=12)

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.savefig(output_dir / "endpoint_bridging_debug.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: endpoint_bridging_debug.png")


# ============================================================
# Part 3: Find best representative sample
# ============================================================
# Score samples by vessel density and visual quality
print("\nScoring representative samples...")
gt_multi_dir = BASE / "Multi-color" / "Pix2pix"
gt_single_dir = BASE / "Single-color" / "Pix2pix"

all_multi = sorted([f.stem.replace("_real_B","") for f in gt_multi_dir.glob("*_real_B.png")])
lbbdm_multi = {d.name for d in (BASE / "Multi-color" / "LBBDM").iterdir() if d.is_dir()}
pbbdm_multi = {d.name for d in (BASE / "Multi-color" / "PBBDM").iterdir() if d.is_dir()}
lbbdm_single = {d.name for d in (BASE / "Single-color" / "LBBDM").iterdir() if d.is_dir()}
pbbdm_single = {d.name for d in (BASE / "Single-color" / "PBBDM").iterdir() if d.is_dir()}
single_set = {f.stem.replace("_real_B","") for f in gt_single_dir.glob("*_real_B.png")}

common_all = sorted(set(all_multi) & single_set & lbbdm_multi & pbbdm_multi & lbbdm_single & pbbdm_single)

scores = []
for sid in common_all:
    img = np.array(Image.open(gt_multi_dir / f"{sid}_real_B.png").convert("RGB"))
    baseline_y = detect_baseline_row(img)
    if baseline_y < 50:
        continue
    crop = img[:baseline_y, :]
    bottom, middle, top = separate_layers_hsv(crop)

    # Score: want samples with good representation in all 3 layers
    b_area = np.sum(bottom)
    m_area = np.sum(middle)
    t_area = np.sum(top)

    # Prefer balanced layers (all 3 present) and moderate density
    min_layer = min(b_area, m_area, t_area)
    total = b_area + m_area + t_area
    balance = min_layer / (total + 1) * 3  # closer to 1 = more balanced

    scores.append((sid, total, balance, b_area, m_area, t_area))

# Sort by balance * total (want balanced AND dense)
scores.sort(key=lambda x: x[2] * x[1], reverse=True)

print("\nTop 10 representative samples (balanced + dense):")
for s in scores[:10]:
    print(f"  {s[0]}: total={s[1]:>8d}, balance={s[2]:.3f}, B={s[3]:>6d} M={s[4]:>6d} T={s[5]:>6d}")

best_samples = [s[0] for s in scores[:5]]
print(f"\nSelected for paper figure: {best_samples}")

# Save best sample IDs
with open(output_dir / "best_representative_samples.txt", "w") as f:
    for s in scores[:10]:
        f.write(f"{s[0]}\t{s[1]}\t{s[2]:.4f}\t{s[3]}\t{s[4]}\t{s[5]}\n")

print("Saved: best_representative_samples.txt")
