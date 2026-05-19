"""
visualize.py — Visualization utilities for depth-aware vessel analysis.

Provides functions to overlay skeletons, endpoints, junctions, and bridge
paths on fluorescence images for qualitative assessment.
"""

import numpy as np
import cv2


def overlay_skeleton(
    img_rgb: np.ndarray,
    skel: np.ndarray,
    color: tuple = (0, 255, 0),
    bg_alpha: float = 0.5,
    skel_width: int = 4,
) -> np.ndarray:
    """
    Overlay a skeleton on an RGB image with adjustable background brightness.

    Parameters
    ----------
    img_rgb : np.ndarray
        Background RGB image.
    skel : np.ndarray
        Binary skeleton.
    color : tuple
        RGB color for the skeleton.
    bg_alpha : float
        Background brightness multiplier (0=black, 1=original).
    skel_width : int
        Dilation kernel size for skeleton visibility.

    Returns
    -------
    np.ndarray
        RGB overlay image (uint8).
    """
    vis = img_rgb.copy().astype(float) * bg_alpha
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (skel_width, skel_width))
    skel_dilated = cv2.dilate(skel.astype(np.uint8), dk) > 0
    vis[skel_dilated] = color
    return vis.astype(np.uint8)


def overlay_rb_skeletons(
    img_rgb: np.ndarray,
    r_skel: np.ndarray,
    b_skel: np.ndarray,
    bg_alpha: float = 0.5,
    skel_width: int = 4,
    r_color: tuple = (255, 80, 80),
    b_color: tuple = (80, 150, 255),
) -> np.ndarray:
    """
    Overlay R and B channel skeletons on an RGB image.

    Parameters
    ----------
    img_rgb : np.ndarray
        Background image.
    r_skel, b_skel : np.ndarray
        R and B channel skeletons.
    bg_alpha : float
        Background brightness.
    skel_width : int
        Dilation size.
    r_color, b_color : tuple
        Colors for R and B skeletons.

    Returns
    -------
    np.ndarray
        RGB overlay image.
    """
    vis = img_rgb.copy().astype(float) * bg_alpha
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (skel_width, skel_width))
    r_dil = cv2.dilate(r_skel.astype(np.uint8), dk) > 0
    b_dil = cv2.dilate(b_skel.astype(np.uint8), dk) > 0
    vis[r_dil] = r_color
    vis[b_dil] = b_color
    return vis.astype(np.uint8)


def draw_markers(
    vis: np.ndarray,
    endpoints: list = None,
    junctions: list = None,
    connected_eps: list = None,
    ep_color: tuple = (255, 255, 0),
    jn_color: tuple = (255, 0, 255),
    conn_color: tuple = (0, 255, 255),
    radius: int = 7,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw endpoint and junction markers on a visualization image.

    Parameters
    ----------
    vis : np.ndarray
        Image to draw on (modified in place and returned).
    endpoints : list of (y, x)
        Real endpoints (yellow circles by default).
    junctions : list of (y, x)
        Junction points (magenta circles).
    connected_eps : list of (y, x)
        Connected endpoints (cyan circles).
    ep_color, jn_color, conn_color : tuple
        RGB colors for each marker type.
    radius, thickness : int
        Circle drawing parameters.

    Returns
    -------
    np.ndarray
        Image with markers drawn.
    """
    out = vis.copy()
    if endpoints:
        for y, x in endpoints:
            cv2.circle(out, (x, y), radius, ep_color, thickness)
    if connected_eps:
        for y, x in connected_eps:
            cv2.circle(out, (x, y), radius, conn_color, thickness)
    if junctions:
        for y, x in junctions:
            cv2.circle(out, (x, y), radius - 1, jn_color, thickness)
    return out


def draw_bridges(
    vis: np.ndarray,
    bridge_info: list,
    color: tuple = (255, 200, 0),
    arrow_scale: float = 15.0,
) -> np.ndarray:
    """
    Draw bridge paths with directional arrows on a visualization.

    Parameters
    ----------
    vis : np.ndarray
        Image to draw on.
    bridge_info : list of dict
        Bridge information from bridge_gaps().
    color : tuple
        RGB color for bridge arrows.
    arrow_scale : float
        Length of tangent direction arrows.

    Returns
    -------
    np.ndarray
        Image with bridges drawn.
    """
    out = vis.copy()
    for b in bridge_info:
        y1, x1 = b["ep1_pos"]
        y2, x2 = b["ep2_pos"]
        t1 = b["ep1_tangent"]
        t2 = b["ep2_tangent"]

        # Draw tangent arrows
        ay1 = int(y1 + t1[0] * arrow_scale)
        ax1 = int(x1 + t1[1] * arrow_scale)
        ay2 = int(y2 + t2[0] * arrow_scale)
        ax2 = int(x2 + t2[1] * arrow_scale)
        cv2.arrowedLine(out, (x1, y1), (ax1, ay1), color, 2, tipLength=0.3)
        cv2.arrowedLine(out, (x2, y2), (ax2, ay2), color, 2, tipLength=0.3)

        # Draw bridge curve
        if "path" in b:
            for k in range(len(b["path"]) - 1):
                py1, px1 = b["path"][k]
                py2, px2 = b["path"][k + 1]
                cv2.line(out, (px1, py1), (px2, py2), color, 1)

    return out
