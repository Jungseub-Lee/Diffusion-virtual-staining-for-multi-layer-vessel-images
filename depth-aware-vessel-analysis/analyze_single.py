"""
analyze_single.py — Analyze a single multi-color fluorescence image.

Demonstrates the full depth-aware vessel analysis pipeline:
1. Preprocess (Gaussian blur)
2. Extract R/B channel skeletons with dominance filtering
3. Bridge gaps using tangent-guided endpoint matching
4. Quantify morphological metrics (length, EP, JN)
5. Classify endpoints (real vs connected)
6. Generate visualization panels

Usage:
    python analyze_single.py --image path/to/multi_color.png --output results/
    python analyze_single.py --image path/to/multi_color.png --output results/ --no-viz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from skimage.morphology import skeletonize

from vessel_analysis.core import (
    preprocess,
    detect_baseline,
    extract_vessel_mask,
    extract_channel_skeleton,
)
from vessel_analysis.bridge import bridge_gaps
from vessel_analysis.metrics import quantify_skeleton, classify_endpoints
from vessel_analysis.visualize import (
    overlay_rb_skeletons,
    overlay_skeleton,
    draw_markers,
    draw_bridges,
)


def analyze(img_rgb: np.ndarray, params: dict = None) -> dict:
    """
    Run the full depth-aware vessel analysis pipeline on a single image.

    Parameters
    ----------
    img_rgb : np.ndarray
        Multi-color fluorescence image (H, W, 3), uint8.
    params : dict, optional
        Override default parameters.

    Returns
    -------
    dict
        Complete analysis results including metrics, skeletons, and bridges.
    """
    p = {
        "blur_kernel": 7,
        "r_thresh": 30,
        "b_seed_thresh": 50,
        "b_low_thresh": 20,
        "dominance_margin": 0,
        "bridge_max_dist": 80,
        "bridge_max_angle": 60.0,
        "bridge_vessel_ratio": 0.7,
        "baseline_fraction": 0.8,
    }
    if params:
        p.update(params)

    h, w = img_rgb.shape[:2]

    # 1. Preprocess
    img_blur = preprocess(img_rgb, blur_kernel=p["blur_kernel"])

    # 2. Detect baseline boundary
    baseline_y = int(h * p["baseline_fraction"])
    region_mask = np.zeros((h, w), dtype=bool)
    region_mask[:baseline_y, :] = True

    # 3. Extract vessel masks
    vessel_mask = extract_vessel_mask(img_blur)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vessel_mask_dilated = cv2.dilate(vessel_mask.astype(np.uint8) * 255, k5) > 0

    # 4. Single-color skeleton (baseline reference)
    base_skel = skeletonize(vessel_mask)

    # 5. R/B channel skeletons with dominance filtering
    _, _, r_filtered = extract_channel_skeleton(
        img_blur,
        channel="R",
        r_thresh=p["r_thresh"],
        dominance_margin=p["dominance_margin"],
    )
    _, _, b_filtered = extract_channel_skeleton(
        img_blur,
        channel="B",
        b_seed_thresh=p["b_seed_thresh"],
        b_low_thresh=p["b_low_thresh"],
        dominance_margin=p["dominance_margin"],
    )

    # 6. Bridge gaps
    r_bridged, r_n_bridges, r_bridge_info = bridge_gaps(
        r_filtered,
        vessel_mask_dilated,
        max_dist=p["bridge_max_dist"],
        max_angle_deg=p["bridge_max_angle"],
        min_vessel_ratio=p["bridge_vessel_ratio"],
    )
    b_bridged, b_n_bridges, b_bridge_info = bridge_gaps(
        b_filtered,
        vessel_mask_dilated,
        max_dist=p["bridge_max_dist"],
        max_angle_deg=p["bridge_max_angle"],
        min_vessel_ratio=p["bridge_vessel_ratio"],
    )

    # 7. Quantify
    single_metrics = quantify_skeleton(base_skel, region_mask, boundary_y=baseline_y)
    r_metrics = quantify_skeleton(r_bridged, region_mask, boundary_y=baseline_y)
    b_metrics = quantify_skeleton(b_bridged, region_mask, boundary_y=baseline_y)

    # 8. Classify endpoints
    r_real_ep, r_conn_ep = classify_endpoints(r_metrics["endpoints"], b_bridged)
    b_real_ep, b_conn_ep = classify_endpoints(b_metrics["endpoints"], r_bridged)

    return {
        "params": p,
        "baseline_y": baseline_y,
        "skeletons": {
            "base": base_skel,
            "r_filtered": r_filtered,
            "b_filtered": b_filtered,
            "r_bridged": r_bridged,
            "b_bridged": b_bridged,
        },
        "bridges": {
            "r_info": r_bridge_info,
            "b_info": b_bridge_info,
            "r_count": r_n_bridges,
            "b_count": b_n_bridges,
        },
        "metrics": {
            "single": {
                "length": single_metrics["length"],
                "endpoints": single_metrics["n_endpoints"],
                "junctions": single_metrics["n_junctions"],
            },
            "r_channel": {
                "length": r_metrics["length"],
                "endpoints": r_metrics["n_endpoints"],
                "junctions": r_metrics["n_junctions"],
                "real_ep": len(r_real_ep),
                "connected_ep": len(r_conn_ep),
                "bridges": r_n_bridges,
            },
            "b_channel": {
                "length": b_metrics["length"],
                "endpoints": b_metrics["n_endpoints"],
                "junctions": b_metrics["n_junctions"],
                "real_ep": len(b_real_ep),
                "connected_ep": len(b_conn_ep),
                "bridges": b_n_bridges,
            },
            "combined": {
                "length": r_metrics["length"] + b_metrics["length"],
                "real_endpoints": len(r_real_ep) + len(b_real_ep),
                "connected_endpoints": len(r_conn_ep) + len(b_conn_ep),
                "junctions": r_metrics["n_junctions"] + b_metrics["n_junctions"],
            },
        },
        "markers": {
            "r_real_ep": r_real_ep,
            "r_conn_ep": r_conn_ep,
            "b_real_ep": b_real_ep,
            "b_conn_ep": b_conn_ep,
            "r_junctions": r_metrics["junctions"],
            "b_junctions": b_metrics["junctions"],
            "single_ep": single_metrics["endpoints"],
            "single_jn": single_metrics["junctions"],
        },
        "vessel_mask": vessel_mask,
    }


def save_visualizations(
    img_rgb: np.ndarray, results: dict, output_dir: Path
) -> None:
    """Save visualization panels as individual PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    skels = results["skeletons"]
    markers = results["markers"]
    by = results["baseline_y"]

    # (a) Single GT skeleton
    vis = overlay_skeleton(img_rgb[:by], skels["base"][:by], color=(0, 200, 0), bg_alpha=0.5)
    vis = draw_markers(vis, endpoints=markers["single_ep"], junctions=markers["single_jn"])
    Image.fromarray(vis).save(output_dir / "single_gt_skeleton.png")

    # (b) R skeleton
    vis = overlay_skeleton(img_rgb[:by], skels["r_bridged"][:by], color=(255, 80, 80), bg_alpha=0.5)
    vis = draw_markers(vis, endpoints=markers["r_real_ep"], junctions=markers["r_junctions"],
                       connected_eps=markers["r_conn_ep"])
    Image.fromarray(vis).save(output_dir / "r_skeleton.png")

    # (c) B skeleton
    vis = overlay_skeleton(img_rgb[:by], skels["b_bridged"][:by], color=(80, 150, 255), bg_alpha=0.5)
    vis = draw_markers(vis, endpoints=markers["b_real_ep"], junctions=markers["b_junctions"],
                       connected_eps=markers["b_conn_ep"])
    Image.fromarray(vis).save(output_dir / "b_skeleton.png")

    # (d) Combined R+B
    vis = overlay_rb_skeletons(img_rgb[:by], skels["r_bridged"][:by], skels["b_bridged"][:by], bg_alpha=0.5)
    all_real = markers["r_real_ep"] + markers["b_real_ep"]
    all_conn = markers["r_conn_ep"] + markers["b_conn_ep"]
    all_jn = markers["r_junctions"] + markers["b_junctions"]
    vis = draw_markers(vis, endpoints=all_real, junctions=all_jn, connected_eps=all_conn)
    Image.fromarray(vis).save(output_dir / "combined_rb_skeleton.png")

    # (e) R bridging detail
    vis = overlay_skeleton(img_rgb[:by], skels["r_filtered"][:by], color=(180, 60, 60), bg_alpha=0.3)
    vis = draw_bridges(vis, results["bridges"]["r_info"])
    Image.fromarray(vis).save(output_dir / "r_bridging_detail.png")

    # (f) B bridging detail
    vis = overlay_skeleton(img_rgb[:by], skels["b_filtered"][:by], color=(60, 100, 180), bg_alpha=0.3)
    vis = draw_bridges(vis, results["bridges"]["b_info"])
    Image.fromarray(vis).save(output_dir / "b_bridging_detail.png")

    print(f"  Saved 6 visualization panels to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Depth-aware vessel network analysis for a single image."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to multi-color fluorescence image")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--blur", type=int, default=7, help="Gaussian blur kernel size")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization output")
    args = parser.parse_args()

    img_rgb = np.array(Image.open(args.image).convert("RGB"))
    output_dir = Path(args.output)

    print(f"Analyzing: {args.image}")
    print(f"  Image size: {img_rgb.shape[1]}x{img_rgb.shape[0]}")

    results = analyze(img_rgb, params={"blur_kernel": args.blur})

    # Print metrics
    m = results["metrics"]
    print(f"\n{'=' * 60}")
    print(f"  Depth-aware Vessel Analysis Results")
    print(f"  (baseline excluded: bottom {int((1 - results['params']['baseline_fraction']) * 100)}%)")
    print(f"{'=' * 60}")
    print(f"\n{'':28s} {'Length':>7s} {'EP':>5s} {'JN':>5s}")
    print(f"{'-' * 50}")
    print(f"{'Single GT (flat)':28s} {m['single']['length']:7d} {m['single']['endpoints']:5d} {m['single']['junctions']:5d}")
    print(f"{'R channel (bottom)':28s} {m['r_channel']['length']:7d} {m['r_channel']['endpoints']:5d} {m['r_channel']['junctions']:5d}")
    print(f"{'B channel (top)':28s} {m['b_channel']['length']:7d} {m['b_channel']['endpoints']:5d} {m['b_channel']['junctions']:5d}")
    print(f"{'R+B combined':28s} {m['combined']['length']:7d} {m['combined']['real_endpoints']:5d} {m['combined']['junctions']:5d}")
    print(f"\n  R bridges: {m['r_channel']['bridges']}")
    print(f"  B bridges: {m['b_channel']['bridges']}")
    print(f"  Connected EP (excluded): {m['combined']['connected_endpoints']}")

    # Save metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_serializable = {k: v for k, v in m.items()}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"\n  Saved metrics.json to {output_dir}/")

    # Save visualizations
    if not args.no_viz:
        save_visualizations(img_rgb, results, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
