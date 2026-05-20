"""
Tangent-Guided Gap Bridging for Depth-Separated Vessel Skeletons
================================================================

When depth-encoded fluorescence images are separated into per-layer channels
(e.g., Bottom = R>B, Top = B>R), the resulting skeletons often have gaps at
vessel crossing regions where the opposing channel dominates.

This script demonstrates the bridging algorithm that reconnects those gaps
by matching skeleton endpoints based on their local tangent direction.

Pipeline:
    1. Load multi-color GT → Gaussian blur → RGB channel separation
    2. Build per-layer masks (R-dominant, B-dominant via hysteresis)
    3. Skeletonize each layer independently
    4. Detect endpoints and estimate tangent directions
    5. Match compatible endpoint pairs (angle + distance + vessel overlap)
    6. Connect matches with cubic Bézier curves
    7. Re-skeletonize to ensure 1-pixel width

Author: Jungseub Lee (Seoul National University)
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage


# ============================================================
# 1. Configuration
# ============================================================
GAUSSIAN_KERNEL = 7          # Smoothing before channel separation
R_THRESHOLD = 30             # Red channel mask threshold
B_SEED_THRESH = 50           # Blue hysteresis: high (seed) threshold
B_LOW_THRESH = 20            # Blue hysteresis: low (extension) threshold
MIN_VESSEL_SIZE = 50         # Remove small objects below this pixel count
BRIDGE_MAX_DIST = 80         # Maximum gap distance (pixels) for bridging
BRIDGE_MAX_ANGLE = 60        # Maximum angle deviation (degrees) from tangent
BRIDGE_VESSEL_OVERLAP = 0.7  # Minimum fraction of bridge on vessel mask
TANGENT_TRACE_LEN = 20       # Pixels traced back along skeleton for tangent
BASELINE_FRACTION = 0.8      # Analyze only top 80% of image (exclude baseline)


# ============================================================
# 2. Load and Preprocess
# ============================================================
DATA = Path(r"[1] Data/Original color stack data")
img = np.array(Image.open(DATA / "Sample 1_BF and GT(512x512 two images).png"))
gt_raw = img[512:]  # Bottom half = fluorescence GT
gt_blur = cv2.GaussianBlur(gt_raw, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)

H, W = gt_raw.shape[:2]
analysis_y = int(H * BASELINE_FRACTION)  # Only analyze above this line

R = gt_blur[:, :, 0].astype(np.float32)  # Red channel
B = gt_blur[:, :, 2].astype(np.float32)  # Blue channel
gray = cv2.cvtColor(gt_blur, cv2.COLOR_RGB2GRAY)
diff = B - R  # Positive = blue dominant, Negative = red dominant

# Morphological kernels
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ============================================================
# 3. Build Per-Layer Vessel Masks
# ============================================================

# --- Bottom layer (R > B): simple threshold on R channel ---
r_binary = (R > R_THRESHOLD).astype(np.uint8) * 255
r_closed = cv2.morphologyEx(r_binary, cv2.MORPH_CLOSE, k3, iterations=2) > 0
r_mask = remove_small_objects(r_closed, min_size=MIN_VESSEL_SIZE)

# --- Top layer (B > R): hysteresis threshold on B channel ---
# Seed: high-confidence blue pixels (B > 50)
# Extension: lower-confidence pixels (B > 20) connected to seeds
b_seed = B > B_SEED_THRESH
b_low = B > B_LOW_THRESH
b_labels, n_components = ndimage.label(b_low)

b_mask_hyst = np.zeros_like(b_low, dtype=bool)
for i in range(1, n_components + 1):
    component = b_labels == i
    if np.any(b_seed[component]):  # Keep only components that contain a seed pixel
        b_mask_hyst |= component

b_closed = cv2.morphologyEx(b_mask_hyst.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k5, iterations=2) > 0
b_mask = remove_small_objects(b_closed, min_size=MIN_VESSEL_SIZE)

# --- Combined vessel mask (for bridge validation) ---
vessel_all = remove_small_objects(
    cv2.morphologyEx((gray > 10).astype(np.uint8) * 255, cv2.MORPH_CLOSE, k3, iterations=2) > 0,
    min_size=100,
)
vessel_dilated = cv2.dilate(vessel_all.astype(np.uint8) * 255, k5) > 0


# ============================================================
# 4. Skeletonize and Apply Dominance Filter
# ============================================================

# Skeletonize each layer mask
r_skeleton = skeletonize(r_mask)
b_skeleton = skeletonize(b_mask)

# Apply dominance filter: keep only skeleton pixels where the correct channel dominates
# This removes skeleton segments that "leaked" into the opposing layer's territory
r_filtered = r_skeleton & (diff < 0)   # Keep where R > B
b_filtered = b_skeleton & (diff > 0)   # Keep where B > R

print(f"R skeleton: {np.sum(r_filtered)} px  |  B skeleton: {np.sum(b_filtered)} px")


# ============================================================
# 5. Endpoint Detection and Tangent Estimation
# ============================================================

def find_endpoints_with_tangent(skeleton):
    """
    Find all skeleton endpoints (pixels with exactly 1 neighbor)
    and estimate their local tangent direction by tracing back
    along the skeleton.

    Returns list of dicts: {'pos': (y, x), 'tangent': (ty, tx)}
    The tangent points OUTWARD from the endpoint (away from the skeleton).
    """
    skel_uint8 = skeleton.astype(np.uint8)
    neighbor_kernel = np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_uint8, -1, neighbor_kernel) * skel_uint8
    endpoint_coords = np.argwhere(neighbor_count == 1)

    endpoints = []
    for (ey, ex) in endpoint_coords:
        # Trace back along the skeleton from this endpoint
        path = [(ey, ex)]
        visited = {(ey, ex)}
        cy, cx = ey, ex

        for _ in range(TANGENT_TRACE_LEN):
            found_next = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < H and 0 <= nx < W
                            and skeleton[ny, nx]
                            and (ny, nx) not in visited):
                        path.append((ny, nx))
                        visited.add((ny, nx))
                        cy, cx = ny, nx
                        found_next = True
                        break
                if found_next:
                    break
            if not found_next:
                break

        # Need at least 5 pixels to estimate a reliable tangent
        if len(path) < 5:
            continue

        # Tangent = direction from interior toward endpoint (outward-pointing)
        n_pts = min(len(path), 10)
        tangent = np.array(path[0], dtype=float) - np.array(path[n_pts - 1], dtype=float)
        norm = np.linalg.norm(tangent)
        if norm > 0:
            tangent /= norm
        endpoints.append({"pos": (ey, ex), "tangent": tangent})

    return endpoints


# ============================================================
# 6. Bézier Bridge Construction
# ============================================================

def bezier_curve(ep1, ep2):
    """
    Generate a cubic Bézier curve connecting two endpoints.

    Control points are placed along each endpoint's tangent direction
    at 40% of the gap distance, creating a smooth curve that respects
    the local vessel direction at both ends.

    Returns list of (y, x) integer pixel coordinates along the curve.
    """
    p1 = np.array(ep1["pos"], dtype=float)
    p2 = np.array(ep2["pos"], dtype=float)
    t1 = ep1["tangent"]
    t2 = ep2["tangent"]
    d = np.linalg.norm(p2 - p1)

    # Control points: extend along tangent, 40% of gap distance
    c1 = p1 + t1 * d * 0.4
    c2 = p2 + t2 * d * 0.4

    # Sample points along the Bézier curve
    n_samples = max(int(d * 1.5), 10)
    curve_points = []
    for t in np.linspace(0, 1, n_samples):
        point = ((1 - t) ** 3 * p1
                 + 3 * (1 - t) ** 2 * t * c1
                 + 3 * (1 - t) * t ** 2 * c2
                 + t ** 3 * p2)
        curve_points.append((int(round(point[0])), int(round(point[1]))))

    return curve_points


# ============================================================
# 7. Endpoint Matching and Bridge Execution
# ============================================================

def bridge_skeleton(skeleton, vessel_mask):
    """
    Find and connect compatible endpoint pairs in a skeleton.

    Matching criteria:
        - Distance between endpoints: 5 to BRIDGE_MAX_DIST pixels
        - Angular compatibility: both tangents must point toward
          each other (cos > cos(BRIDGE_MAX_ANGLE))
        - Vessel overlap: >= BRIDGE_VESSEL_OVERLAP of bridge pixels
          must lie on the vessel mask

    Scoring:
        score = (cos1 + cos2) / 2           # angular alignment
                - distance / max_dist * 0.2  # prefer shorter gaps
                + vessel_ratio * 0.2          # prefer on-vessel bridges

    Selection: greedy best-first, each endpoint used at most once.
    """
    endpoints = find_endpoints_with_tangent(skeleton)
    cos_threshold = np.cos(np.radians(BRIDGE_MAX_ANGLE))

    # Score all valid pairs
    candidates = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            p1 = np.array(endpoints[i]["pos"], dtype=float)
            p2 = np.array(endpoints[j]["pos"], dtype=float)
            dist = np.linalg.norm(p2 - p1)

            if dist < 5 or dist > BRIDGE_MAX_DIST:
                continue

            # Check angular compatibility
            direction_1to2 = (p2 - p1) / dist
            cos1 = np.dot(endpoints[i]["tangent"], direction_1to2)   # EP1 tangent toward EP2
            cos2 = np.dot(endpoints[j]["tangent"], -direction_1to2)  # EP2 tangent toward EP1

            if cos1 < cos_threshold or cos2 < cos_threshold:
                continue

            # Check vessel mask overlap
            bridge_pts = bezier_curve(endpoints[i], endpoints[j])
            on_vessel = sum(
                1 for (y, x) in bridge_pts
                if 0 <= y < H and 0 <= x < W and vessel_mask[y, x]
            )
            vessel_ratio = on_vessel / max(len(bridge_pts), 1)

            if vessel_ratio < BRIDGE_VESSEL_OVERLAP:
                continue

            score = (cos1 + cos2) / 2 - dist / BRIDGE_MAX_DIST * 0.2 + vessel_ratio * 0.2
            candidates.append({
                "i": i, "j": j,
                "score": score,
                "dist": dist,
                "ep1": endpoints[i],
                "ep2": endpoints[j],
            })

    # Greedy matching: best score first, no endpoint reuse
    candidates.sort(key=lambda c: -c["score"])
    used = set()
    bridges = []
    bridged = skeleton.copy()

    for c in candidates:
        if c["i"] not in used and c["j"] not in used:
            # Draw bridge onto skeleton
            for (y, x) in bezier_curve(c["ep1"], c["ep2"]):
                if 0 <= y < H and 0 <= x < W:
                    bridged[y, x] = True
            used.add(c["i"])
            used.add(c["j"])
            bridges.append(c)

    # Re-skeletonize to ensure 1-pixel width after bridge insertion
    bridged = skeletonize(bridged)

    return bridged, bridges


# ============================================================
# 8. Run Bridging
# ============================================================

r_bridged, r_bridges = bridge_skeleton(r_filtered, vessel_dilated)
b_bridged, b_bridges = bridge_skeleton(b_filtered, vessel_dilated)

print(f"\nBridging results:")
print(f"  R skeleton: {len(r_bridges)} bridges applied")
for br in r_bridges:
    print(f"    dist={br['dist']:.0f}px  score={br['score']:.3f}")
print(f"  B skeleton: {len(b_bridges)} bridges applied")
for br in b_bridges:
    print(f"    dist={br['dist']:.0f}px  score={br['score']:.3f}")


# ============================================================
# 9. Quantification (Junction / Endpoint / Length)
# ============================================================

def quantify_skeleton(skeleton, label=""):
    """Count junctions, endpoints, and total length in the analysis region."""
    region = skeleton.copy()
    region[analysis_y:] = False  # Exclude baseline area

    skel_uint8 = region.astype(np.uint8)
    neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nb = cv2.filter2D(skel_uint8, -1, neighbor_kernel) * skel_uint8

    length = int(np.sum(region))

    # Endpoints: neighbor count == 1, exclude near baseline
    ep_mask = nb == 1
    ep_labels, n_ep = ndimage.label(ep_mask)
    ep_count = 0
    for i in range(1, n_ep + 1):
        pts = np.argwhere(ep_labels == i)
        cy = pts.mean(axis=0)[0]
        if cy < analysis_y - 5:
            ep_count += 1

    # Junctions: neighbor count >= 3, cluster and verify >= 3 branches
    jn_mask = nb >= 3
    jn_dilated = cv2.dilate(jn_mask.astype(np.uint8), k3)
    jn_labels, n_raw = ndimage.label(jn_dilated)
    jn_count = 0
    for i in range(1, n_raw + 1):
        cluster = jn_labels == i
        ys, xs = np.where(cluster)
        pad = 3
        ly0, ly1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
        lx0, lx1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
        local = region[ly0:ly1, lx0:lx1].copy()
        local[cluster[ly0:ly1, lx0:lx1]] = False
        _, n_branches = ndimage.label(local)
        if n_branches >= 3:
            jn_count += 1

    print(f"  {label:>12s}:  Length={length:5d}  EP={ep_count:3d}  JN={jn_count:3d}")
    return {"length": length, "endpoints": ep_count, "junctions": jn_count}


# Single-color baseline skeleton
base_skel = skeletonize(
    remove_small_objects(
        cv2.morphologyEx((gray > 10).astype(np.uint8) * 255, cv2.MORPH_CLOSE, k3, iterations=2) > 0,
        min_size=100,
    )
)

print(f"\nQuantification (top {BASELINE_FRACTION*100:.0f}% of image, N=1 sample):")
single = quantify_skeleton(base_skel, "Single GT")
r_stats = quantify_skeleton(r_bridged, "R (bottom)")
b_stats = quantify_skeleton(b_bridged, "B (top)")
print(f"  {'R+B sum':>12s}:  Length={r_stats['length']+b_stats['length']:5d}  "
      f"EP={r_stats['endpoints']+b_stats['endpoints']:3d}  "
      f"JN={r_stats['junctions']+b_stats['junctions']:3d}")
