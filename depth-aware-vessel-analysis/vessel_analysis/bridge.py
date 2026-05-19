"""
bridge.py — Tangent-guided gap bridging for skeleton fragments.

At vessel crossing regions, the dominance filter (R>B or B>R) removes skeleton
pixels that belong to the other depth layer. This creates gaps where a vessel
passes through a crossing. These gaps need to be reconnected.

Algorithm: Tangent-Guided Endpoint Matching with Bezier Bridging
================================================================

1. ENDPOINT DETECTION
   Find all skeleton endpoints (pixels with exactly 1 neighbor).

2. TANGENT ESTIMATION
   Trace backward along each endpoint's branch for N pixels (default 20)
   to compute the local tangent direction. This tells us which direction
   the vessel was heading before it was interrupted.

3. CANDIDATE MATCHING
   For each pair of endpoints (ep1, ep2):
   a. Distance check: 5 < dist(ep1, ep2) < max_dist (default 80px)
   b. Angle check: the tangent at ep1 must point toward ep2, and
      vice versa. Both angles must be within max_angle_deg (default 60°).
   c. Vessel mask check: the bridge path must lie within the vessel
      mask (>= 70% of bridge pixels on vessel).

4. SCORING & GREEDY ASSIGNMENT
   Score = mean(cos_angle1, cos_angle2) - dist_penalty + vessel_ratio_bonus
   Sort by score descending, greedily assign (each endpoint used once).

5. BEZIER CURVE BRIDGE
   Connect matched endpoints with a cubic Bezier curve using tangent
   directions as control points, producing a smooth, anatomically
   plausible vessel path through the crossing.

6. RE-SKELETONIZATION
   After all bridges are drawn, re-skeletonize to ensure 1-pixel width.

         ep1 ──t1──►
                        ╲
                          ╲  Bezier bridge
                            ╲
                  ◄──t2── ep2
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize


def _get_endpoints_with_tangent(
    skel: np.ndarray, trace_len: int = 20
) -> list:
    """
    Find all skeleton endpoints and compute their local tangent directions.

    Parameters
    ----------
    skel : np.ndarray
        Binary skeleton image (bool or uint8).
    trace_len : int
        Number of pixels to trace back from endpoint for tangent estimation.

    Returns
    -------
    list of dict
        Each dict contains:
        - 'pos': (y, x) tuple of endpoint position
        - 'tangent': unit vector pointing outward from the endpoint
        - 'path': list of (y, x) positions traced from endpoint
    """
    h, w = skel.shape
    su = skel.astype(np.uint8)
    ker = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su
    endpoints = np.argwhere(nb == 1)

    results = []
    for ep_pos in endpoints:
        y0, x0 = ep_pos
        path = [(y0, x0)]
        visited = {(y0, x0)}
        cy, cx = y0, x0

        # Trace backward along the branch
        for _ in range(trace_len):
            found = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and skel[ny, nx]
                        and (ny, nx) not in visited
                    ):
                        path.append((ny, nx))
                        visited.add((ny, nx))
                        cy, cx = ny, nx
                        found = True
                        break
                if found:
                    break
            if not found:
                break

        # Compute tangent from traced path
        if len(path) >= 5:
            n_avg = min(len(path), 10)
            tangent = np.array(path[0], dtype=float) - np.array(
                path[n_avg - 1], dtype=float
            )
            norm = np.linalg.norm(tangent)
            if norm > 0:
                tangent /= norm
            results.append({"pos": (y0, x0), "tangent": tangent, "path": path})

    return results


def _bezier_bridge(ep1: dict, ep2: dict) -> list:
    """
    Generate a cubic Bezier curve connecting two endpoints.

    The control points are placed along each endpoint's tangent direction
    at 40% of the inter-endpoint distance, ensuring the bridge follows
    the natural vessel curvature.

    Parameters
    ----------
    ep1, ep2 : dict
        Endpoint dicts with 'pos' and 'tangent' keys.

    Returns
    -------
    list of (y, x) tuples
        Pixel coordinates along the Bezier curve.
    """
    p1 = np.array(ep1["pos"], dtype=float)
    p2 = np.array(ep2["pos"], dtype=float)
    t1, t2 = ep1["tangent"], ep2["tangent"]
    dist = np.linalg.norm(p2 - p1)

    # Control points along tangent directions
    ctrl1 = p1 + t1 * dist * 0.4
    ctrl2 = p2 + t2 * dist * 0.4

    # Sample Bezier curve
    n_pts = max(int(dist * 1.5), 10)
    pts = []
    for t in np.linspace(0, 1, n_pts):
        p = (
            (1 - t) ** 3 * p1
            + 3 * (1 - t) ** 2 * t * ctrl1
            + 3 * (1 - t) * t ** 2 * ctrl2
            + t ** 3 * p2
        )
        pts.append((int(round(p[0])), int(round(p[1]))))

    return pts


def bridge_gaps(
    skel: np.ndarray,
    vessel_mask: np.ndarray,
    max_dist: int = 80,
    max_angle_deg: float = 60.0,
    min_vessel_ratio: float = 0.7,
    trace_len: int = 20,
    re_skeletonize: bool = True,
) -> tuple:
    """
    Bridge gaps in a skeleton using tangent-guided endpoint matching.

    Parameters
    ----------
    skel : np.ndarray
        Binary skeleton image (bool).
    vessel_mask : np.ndarray
        Binary vessel mask (should be slightly dilated for tolerance).
    max_dist : int
        Maximum distance between endpoints to consider for bridging.
    max_angle_deg : float
        Maximum allowed angle deviation from tangent direction (degrees).
    min_vessel_ratio : float
        Minimum fraction of bridge pixels that must lie on the vessel mask.
    trace_len : int
        Pixels to trace for tangent estimation.
    re_skeletonize : bool
        If True, re-skeletonize after bridging to ensure 1-pixel width.

    Returns
    -------
    tuple (bridged_skeleton, n_bridges, bridge_info)
        - bridged_skeleton: skeleton with gaps bridged (bool)
        - n_bridges: number of bridges created
        - bridge_info: list of dicts with bridge details for visualization
    """
    h, w = skel.shape
    eps = _get_endpoints_with_tangent(skel, trace_len=trace_len)

    # Score all candidate pairs
    cos_thresh = np.cos(np.radians(max_angle_deg))
    candidates = []

    for i in range(len(eps)):
        for j in range(i + 1, len(eps)):
            p1 = np.array(eps[i]["pos"], dtype=float)
            p2 = np.array(eps[j]["pos"], dtype=float)
            dist = np.linalg.norm(p2 - p1)

            if dist < 5 or dist > max_dist:
                continue

            # Angle compatibility check
            dir_12 = (p2 - p1) / dist
            cos1 = np.dot(eps[i]["tangent"], dir_12)
            cos2 = np.dot(eps[j]["tangent"], -dir_12)

            if cos1 < cos_thresh or cos2 < cos_thresh:
                continue

            # Generate bridge path and check vessel mask coverage
            pts = _bezier_bridge(eps[i], eps[j])
            on_vessel = sum(
                1
                for (y, x) in pts
                if 0 <= y < h and 0 <= x < w and vessel_mask[y, x]
            )
            vessel_ratio = on_vessel / max(len(pts), 1)

            if vessel_ratio < min_vessel_ratio:
                continue

            score = (cos1 + cos2) / 2 - dist / max_dist * 0.2 + vessel_ratio * 0.2
            candidates.append(
                {
                    "i": i,
                    "j": j,
                    "score": score,
                    "dist": dist,
                    "vessel_ratio": vessel_ratio,
                    "ep1": eps[i],
                    "ep2": eps[j],
                }
            )

    # Greedy assignment (best score first, each endpoint used once)
    candidates.sort(key=lambda x: -x["score"])
    used = set()
    matches = []

    for c in candidates:
        if c["i"] not in used and c["j"] not in used:
            matches.append(c)
            used.add(c["i"])
            used.add(c["j"])

    # Draw bridges
    bridged = skel.copy()
    bridge_info = []

    for m in matches:
        pts = _bezier_bridge(m["ep1"], m["ep2"])
        for y, x in pts:
            if 0 <= y < h and 0 <= x < w:
                bridged[y, x] = True
        bridge_info.append(
            {
                "ep1_pos": m["ep1"]["pos"],
                "ep2_pos": m["ep2"]["pos"],
                "ep1_tangent": m["ep1"]["tangent"],
                "ep2_tangent": m["ep2"]["tangent"],
                "dist": m["dist"],
                "score": m["score"],
                "path": pts,
            }
        )

    # Re-skeletonize to ensure 1-pixel width
    if re_skeletonize:
        bridged = skeletonize(bridged)

    return bridged, len(matches), bridge_info
