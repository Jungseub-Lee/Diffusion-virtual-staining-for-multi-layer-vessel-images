"""
core.py — Image preprocessing, vessel mask extraction, and skeletonization.

This module handles the fundamental operations for depth-aware vessel analysis:
1. Gaussian blur preprocessing to reduce surface noise
2. R/B channel-based vessel mask extraction with hysteresis thresholding
3. Dominance filtering (R>B vs B>R) for depth-layer separation
4. Skeletonization of separated vessel masks

The key insight is that multi-color fluorescence GT images encode depth
information via color channels: R (bottom layer) and B (top layer). By
analyzing raw RGB channel intensities rather than hue-based classification,
crossing regions where both R and B are significant are naturally captured
in both channel masks, enabling correct skeleton topology through crossings.
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage


def preprocess(img_rgb: np.ndarray, blur_kernel: int = 7) -> np.ndarray:
    """
    Apply Gaussian blur to reduce surface noise in fluorescence images.

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image (H, W, 3), uint8.
    blur_kernel : int
        Gaussian kernel size (must be odd). Default 7.

    Returns
    -------
    np.ndarray
        Blurred RGB image.
    """
    if blur_kernel > 0:
        return cv2.GaussianBlur(img_rgb, (blur_kernel, blur_kernel), 0)
    return img_rgb.copy()


def detect_baseline(img_rgb: np.ndarray, threshold_ratio: float = 0.35) -> int:
    """
    Detect the baseline boundary row where fluorescence signal becomes unstable.

    The bottom region of fluorescence images typically contains the substrate
    boundary with irregular morphology. This function finds the row where
    vessel density drops, indicating the transition to the unstable region.

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image (H, W, 3).
    threshold_ratio : float
        Fraction of row width that must be vessel pixels to be considered
        above baseline. Default 0.35.

    Returns
    -------
    int
        Row index of the baseline boundary. Vessel analysis should use
        only rows [0, baseline_y).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    vessel = gray > 15
    h, w = vessel.shape
    row_density = np.sum(vessel, axis=1) / w
    baseline_y = h
    for y in range(h - 1, -1, -1):
        if row_density[y] < threshold_ratio:
            baseline_y = y
            break
    return max(0, baseline_y - 5)


def extract_vessel_mask(img_rgb: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    Extract a binary vessel mask from any fluorescence image (single or multi-color).

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image.
    min_area : int
        Minimum connected component area to retain. Default 100.

    Returns
    -------
    np.ndarray
        Binary vessel mask (bool).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    vessel = gray > 10
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel = cv2.morphologyEx(vessel.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
    return remove_small_objects(vessel, min_size=min_area)


def extract_channel_skeleton(
    img_blur: np.ndarray,
    channel: str = "R",
    r_thresh: int = 30,
    b_seed_thresh: int = 50,
    b_low_thresh: int = 20,
    dominance_margin: int = 0,
    min_area: int = 50,
) -> tuple:
    """
    Extract a depth-layer skeleton from a specific RGB channel.

    For the R channel (bottom-layer vessels):
        Simple threshold on R intensity → morphological cleanup → skeleton
        → filter to keep only pixels where R > B + margin.

    For the B channel (top-layer vessels):
        Hysteresis thresholding (seed at high B, expand to low B) to handle
        crossing regions where blue intensity drops due to red vessel overlap
        → morphological cleanup → skeleton → filter to B > R + margin.

    Parameters
    ----------
    img_blur : np.ndarray
        Gaussian-blurred RGB image.
    channel : str
        "R" for bottom-layer or "B" for top-layer extraction.
    r_thresh : int
        Threshold for R channel mask. Default 30.
    b_seed_thresh : int
        High-confidence threshold for B channel seeds. Default 50.
    b_low_thresh : int
        Low threshold for B channel hysteresis expansion. Default 20.
    dominance_margin : int
        Additional margin for dominance filter (R > B + margin). Default 0.
    min_area : int
        Minimum connected component area. Default 50.

    Returns
    -------
    tuple (skeleton, mask, filtered_skeleton)
        - skeleton: full channel skeleton before dominance filtering
        - mask: binary vessel mask for the channel
        - filtered_skeleton: skeleton filtered by channel dominance
    """
    R = img_blur[:, :, 0].astype(np.float32)
    B = img_blur[:, :, 2].astype(np.float32)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    if channel.upper() == "R":
        # Simple threshold on R channel
        mask = R > r_thresh
        mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k3, iterations=2) > 0
        mask = remove_small_objects(mask, min_size=min_area)
        skel = skeletonize(mask)
        # Dominance filter: keep skeleton pixels where R > B
        diff = R - B  # positive where R dominant
        filtered = skel & (diff > dominance_margin)

    elif channel.upper() == "B":
        # Hysteresis thresholding for B channel
        # Seed regions: high B confidence
        seed = B > b_seed_thresh
        # Low threshold expansion
        low = B > b_low_thresh
        low_labeled, n_components = ndimage.label(low)
        mask = np.zeros_like(low, dtype=bool)
        for i in range(1, n_components + 1):
            region = low_labeled == i
            if np.any(seed[region]):
                mask |= region
        mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k5, iterations=2) > 0
        mask = remove_small_objects(mask, min_size=min_area)
        skel = skeletonize(mask)
        # Dominance filter: keep skeleton pixels where B > R
        diff = B - R  # positive where B dominant
        filtered = skel & (diff > dominance_margin)

    else:
        raise ValueError(f"channel must be 'R' or 'B', got '{channel}'")

    return skel, mask, filtered
