"""
analyze_batch.py — Batch comparison of GT vs virtual staining models.

Processes all samples across multiple models (LBBDM, PBBDM, GANs) and
compares single-color (flat) vs multi-color (depth-separated) vessel metrics.

Usage:
    python analyze_batch.py --data-dir path/to/data --output results/
    python analyze_batch.py --data-dir path/to/data --output results/ --models LBBDM PBBDM
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

from vessel_analysis.core import preprocess, detect_baseline, extract_vessel_mask, extract_channel_skeleton
from vessel_analysis.bridge import bridge_gaps
from vessel_analysis.metrics import quantify_skeleton, classify_endpoints
from skimage.morphology import skeletonize
import cv2


def analyze_multi(img_rgb: np.ndarray, baseline_y: int, params: dict) -> dict:
    """Analyze a multi-color image with depth-aware separation."""
    h, w = img_rgb.shape[:2]
    img_blur = preprocess(img_rgb, blur_kernel=params.get("blur_kernel", 7))

    region_mask = np.zeros((h, w), dtype=bool)
    region_mask[:baseline_y, :] = True

    vessel_mask = extract_vessel_mask(img_blur)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vessel_dilated = cv2.dilate(vessel_mask.astype(np.uint8) * 255, k5) > 0

    _, _, r_filt = extract_channel_skeleton(img_blur, "R")
    _, _, b_filt = extract_channel_skeleton(img_blur, "B")

    r_br, _, _ = bridge_gaps(r_filt, vessel_dilated)
    b_br, _, _ = bridge_gaps(b_filt, vessel_dilated)

    r_m = quantify_skeleton(r_br, region_mask, boundary_y=baseline_y)
    b_m = quantify_skeleton(b_br, region_mask, boundary_y=baseline_y)

    r_real, r_conn = classify_endpoints(r_m["endpoints"], b_br)
    b_real, b_conn = classify_endpoints(b_m["endpoints"], r_br)

    return {
        "length": r_m["length"] + b_m["length"],
        "endpoints": len(r_real) + len(b_real),
        "junctions": r_m["n_junctions"] + b_m["n_junctions"],
        "r_length": r_m["length"],
        "b_length": b_m["length"],
    }


def analyze_single(img_rgb: np.ndarray, baseline_y: int, params: dict) -> dict:
    """Analyze a single-color image (flat projection)."""
    h, w = img_rgb.shape[:2]
    region_mask = np.zeros((h, w), dtype=bool)
    region_mask[:baseline_y, :] = True

    vessel_mask = extract_vessel_mask(img_rgb)
    skel = skeletonize(vessel_mask)
    m = quantify_skeleton(skel, region_mask, boundary_y=baseline_y)

    return {
        "length": m["length"],
        "endpoints": m["n_endpoints"],
        "junctions": m["n_junctions"],
    }


def find_samples(data_dir: Path, models: list) -> list:
    """Find sample IDs common across all models."""
    # Use GAN directory (Pix2pix) for GT sample discovery
    gt_dir = data_dir / "Multi-color" / "Pix2pix"
    if not gt_dir.exists():
        gt_dir = data_dir / "Multi-color" / "LSGAN"

    gt_samples = set()
    for f in gt_dir.glob("*_real_B.png"):
        gt_samples.add(f.stem.replace("_real_B", ""))

    # Find common samples across models
    common = gt_samples.copy()
    for model in models:
        model_dir = data_dir / "Multi-color" / model
        if not model_dir.exists():
            continue
        if model in ["LBBDM", "PBBDM"]:
            model_samples = {d.name for d in model_dir.iterdir() if d.is_dir()}
        else:
            model_samples = {f.stem.replace("_fake_B", "") for f in model_dir.glob("*_fake_B.png")}
        common &= model_samples

    return sorted(common)


def load_image(path: Path) -> np.ndarray:
    """Load an image as RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description="Batch vessel analysis comparison.")
    parser.add_argument("--data-dir", type=str, required=True, help="Root data directory")
    parser.add_argument("--output", type=str, default="batch_results", help="Output directory")
    parser.add_argument("--models", nargs="+", default=["LBBDM", "PBBDM", "LSGAN", "Pix2pix", "WGANGP"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {"blur_kernel": 7}

    samples = find_samples(data_dir, args.models)
    print(f"Found {len(samples)} common samples")

    # Initialize results
    results = {"GT": {"single": [], "multi": []}}
    for model in args.models:
        results[model] = {"single": [], "multi": []}

    # Process
    gt_ref_dir = data_dir / "Multi-color" / "Pix2pix"
    if not gt_ref_dir.exists():
        gt_ref_dir = data_dir / "Multi-color" / "LSGAN"

    for idx, sid in enumerate(samples):
        if (idx + 1) % 25 == 0:
            print(f"  Processing {idx + 1}/{len(samples)}: {sid}")

        # Load GT and detect baseline
        gt_multi = load_image(gt_ref_dir / f"{sid}_real_B.png")
        baseline_y = detect_baseline(gt_multi)
        if baseline_y < 30:
            continue

        # GT
        gt_single_dir = data_dir / "Single-color" / "Pix2pix"
        if not gt_single_dir.exists():
            gt_single_dir = data_dir / "Single-color" / "LSGAN"
        gt_single = load_image(gt_single_dir / f"{sid}_real_B.png")

        results["GT"]["single"].append(analyze_single(gt_single, baseline_y, params))
        results["GT"]["multi"].append(analyze_multi(gt_multi, baseline_y, params))

        # Models
        for model in args.models:
            try:
                if model in ["LBBDM", "PBBDM"]:
                    multi_path = data_dir / "Multi-color" / model / sid / "output_0.png"
                    single_path = data_dir / "Single-color" / model / sid / "output_0.png"
                else:
                    multi_path = data_dir / "Multi-color" / model / f"{sid}_fake_B.png"
                    single_path = data_dir / "Single-color" / model / f"{sid}_fake_B.png"

                if multi_path.exists() and single_path.exists():
                    m_img = load_image(multi_path)
                    s_img = load_image(single_path)
                    results[model]["single"].append(analyze_single(s_img, baseline_y, params))
                    results[model]["multi"].append(analyze_multi(m_img, baseline_y, params))
            except Exception as e:
                print(f"    Warning: {model}/{sid} failed: {e}")

    # Compute summary statistics
    n = len(results["GT"]["single"])
    print(f"\nProcessed {n} samples successfully")

    summary = {}
    for source in ["GT"] + args.models:
        if not results[source]["single"]:
            continue
        s = results[source]["single"]
        m = results[source]["multi"]
        summary[source] = {
            "n": len(s),
            "single": {
                "length": {"mean": np.mean([x["length"] for x in s]), "std": np.std([x["length"] for x in s])},
                "endpoints": {"mean": np.mean([x["endpoints"] for x in s]), "std": np.std([x["endpoints"] for x in s])},
                "junctions": {"mean": np.mean([x["junctions"] for x in s]), "std": np.std([x["junctions"] for x in s])},
            },
            "multi": {
                "length": {"mean": np.mean([x["length"] for x in m]), "std": np.std([x["length"] for x in m])},
                "endpoints": {"mean": np.mean([x["endpoints"] for x in m]), "std": np.std([x["endpoints"] for x in m])},
                "junctions": {"mean": np.mean([x["junctions"] for x in m]), "std": np.std([x["junctions"] for x in m])},
            },
        }

    # Print results table
    print(f"\n{'=' * 90}")
    print(f"  BATCH COMPARISON: Single-color (flat) vs Multi-color (depth-separated)")
    print(f"  N = {n} samples")
    print(f"{'=' * 90}")

    for source in ["GT"] + args.models:
        if source not in summary:
            continue
        ss = summary[source]["single"]
        ms = summary[source]["multi"]
        print(f"\n--- {source} (n={summary[source]['n']}) ---")
        print(f"  {'Metric':<15} {'Single':>20s} {'Multi (depth)':>20s}")
        print(f"  {'-' * 55}")
        for metric in ["length", "endpoints", "junctions"]:
            sv = f"{ss[metric]['mean']:.1f} +/- {ss[metric]['std']:.1f}"
            mv = f"{ms[metric]['mean']:.1f} +/- {ms[metric]['std']:.1f}"
            print(f"  {metric:<15} {sv:>20s} {mv:>20s}")

    # Save results
    with open(output_dir / "batch_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nSaved batch_results.json to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
