"""
Depth-aware Vessel Network Analysis
====================================

A tool for analyzing multi-color fluorescence microscopy images of vascular
networks with depth-encoded color channels (R/B separation).

Modules:
    core    - Image preprocessing, vessel mask extraction, skeletonization
    bridge  - Tangent-guided gap bridging for skeleton fragments
    metrics - Morphological quantification (length, endpoints, junctions)
    visualize - Visualization utilities for skeleton overlays
"""

__version__ = "1.0.0"
__author__ = "Seungbum Baik"

from .core import preprocess, extract_vessel_mask, extract_channel_skeleton
from .bridge import bridge_gaps
from .metrics import quantify_skeleton, classify_endpoints
