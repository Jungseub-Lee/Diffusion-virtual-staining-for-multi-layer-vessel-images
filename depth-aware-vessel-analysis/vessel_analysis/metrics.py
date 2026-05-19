"""
metrics.py — Morphological quantification of vessel skeletons.

Computes three key metrics:
- Length: total skeleton pixel count (proxy for vessel length)
- Endpoints (EP): terminal points of vessel branches
- Junctions (JN): bifurcation/crossing points where 3+ branches meet

Includes robust junction detection that:
1. Clusters adjacent junction pixels into single points
2. Verifies each cluster by removing it and counting remaining fragments
3. Only counts clusters that produce >= 3 fragments (true bifurcations)

Also provides endpoint classification for combined R+B analysis:
- Real EP: true vessel terminations
- Connected EP: endpoints that touch the other channel's skeleton
  (indicating the vessel continues in a different depth layer)
"""

import numpy as np
import cv2
from scipy import ndimage


def quantify_skeleton(
    skel: np.ndarray,
    region_mask: np.ndarray = None,
    boundary_y: int = None,
    boundary_margin: int = 5,
) -> dict:
    """
    Compute morphological metrics from a skeleton.

    Uses clustered junction/endpoint detection for accurate counting:
    adjacent junction pixels are grouped into one, and each group is
    verified by topology (must produce >= 3 fragments when removed).

    Parameters
    ----------
    skel : np.ndarray
        Binary skeleton image (bool).
    region_mask : np.ndarray, optional
        Mask defining the analysis region. If None, uses entire image.
    boundary_y : int, optional
        Y-coordinate of the baseline boundary. Endpoints within
        boundary_margin of this line are excluded (cut artifacts).
    boundary_margin : int
        Pixel margin around boundary_y for endpoint exclusion.

    Returns
    -------
    dict with keys:
        - 'length': int, total skeleton pixels in region
        - 'endpoints': list of (y, x) endpoint centroids
        - 'junctions': list of (y, x) junction centroids
        - 'n_endpoints': int, count of endpoints
        - 'n_junctions': int, count of junctions
    """
    if region_mask is not None:
        s = skel & region_mask
    else:
        s = skel.copy()

    su = s.astype(np.uint8)
    ker = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nb = cv2.filter2D(su, -1, ker) * su

    length = int(np.sum(s))

    # --- Endpoint detection with clustering ---
    ep_mask = nb == 1
    ep_labels, n_ep = ndimage.label(ep_mask)
    ep_centroids = []
    for i in range(1, n_ep + 1):
        pts = np.argwhere(ep_labels == i)
        cy, cx = pts.mean(axis=0)
        ep_centroids.append((int(round(cy)), int(round(cx))))

    # Exclude endpoints near baseline boundary
    if boundary_y is not None:
        ep_centroids = [
            (y, x)
            for (y, x) in ep_centroids
            if y < boundary_y - boundary_margin
        ]

    # --- Junction detection with clustering and verification ---
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    jn_mask = nb >= 3
    jn_dilated = cv2.dilate(jn_mask.astype(np.uint8), k3)
    jn_labels, n_jn_raw = ndimage.label(jn_dilated)
    jn_centroids = []

    for i in range(1, n_jn_raw + 1):
        cluster_mask = (jn_labels == i) & jn_mask
        pts = np.argwhere(cluster_mask)
        if len(pts) == 0:
            pts = np.argwhere(jn_labels == i)
        cy, cx = pts.mean(axis=0)
        jy, jx = int(round(cy)), int(round(cx))

        # Topology verification: remove cluster and count remaining fragments
        cluster_all = jn_labels == i
        ys, xs = np.where(cluster_all)
        pad = 3
        ly0 = max(0, ys.min() - pad)
        ly1 = min(s.shape[0], ys.max() + pad + 1)
        lx0 = max(0, xs.min() - pad)
        lx1 = min(s.shape[1], xs.max() + pad + 1)

        local_skel = s[ly0:ly1, lx0:lx1].copy()
        local_cluster = cluster_all[ly0:ly1, lx0:lx1]
        local_skel[local_cluster] = False

        _, n_branches = ndimage.label(local_skel)
        if n_branches >= 3:
            jn_centroids.append((jy, jx))

    return {
        "length": length,
        "endpoints": ep_centroids,
        "junctions": jn_centroids,
        "n_endpoints": len(ep_centroids),
        "n_junctions": len(jn_centroids),
    }


def classify_endpoints(
    ep_list: list,
    other_skel: np.ndarray,
    proximity_radius: int = 5,
) -> tuple:
    """
    Classify endpoints as 'real' or 'connected' based on proximity to
    another channel's skeleton.

    When combining R and B skeletons, some endpoints of one channel
    touch the other channel's skeleton. These are not true vessel
    terminations — they indicate inter-layer connections and should
    be excluded from the endpoint count.

    Parameters
    ----------
    ep_list : list of (y, x) tuples
        Endpoint positions to classify.
    other_skel : np.ndarray
        Skeleton of the other channel (bool).
    proximity_radius : int
        Dilation radius for proximity detection. Default 5.

    Returns
    -------
    tuple (real_eps, connected_eps)
        - real_eps: list of (y, x) — true vessel terminations
        - connected_eps: list of (y, x) — inter-layer connection points
    """
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (proximity_radius * 2 + 1, proximity_radius * 2 + 1)
    )
    other_dilated = cv2.dilate(other_skel.astype(np.uint8), k) > 0

    real_eps = []
    connected_eps = []

    for y, x in ep_list:
        if 0 <= y < other_dilated.shape[0] and 0 <= x < other_dilated.shape[1]:
            if other_dilated[y, x]:
                connected_eps.append((y, x))
            else:
                real_eps.append((y, x))
        else:
            real_eps.append((y, x))

    return real_eps, connected_eps
