# Depth-aware Vessel Network Analysis

Morphological analysis tool for multi-color fluorescence microscopy images of vascular networks with depth-encoded color channels.

## Overview

Multi-color fluorescence ground truth (GT) images encode vascular depth information through color channels:
- **R channel** (red): bottom-layer vessels
- **B channel** (blue): top-layer vessels
- **Y channel** (yellow): intermediate-layer vessels

This tool separates overlapping vessels by depth layer and quantifies morphological features (length, endpoints, junctions) for each layer independently — enabling accurate comparison between GT and virtual staining results.

## Key Algorithm: Tangent-Guided Gap Bridging

When vessels from different depth layers cross each other, the dominance filter (R>B or B>R) creates gaps in the skeleton. Our bridging algorithm reconnects these gaps:

1. **Endpoint detection** — find skeleton terminals
2. **Tangent estimation** — trace backward to compute vessel direction
3. **Candidate matching** — pair endpoints by directional compatibility and distance
4. **Bezier bridging** — connect matched pairs with smooth cubic curves
5. **Vessel mask validation** — ensure bridges follow actual vessel paths

```
    ep1 ──t1──►
                    ╲
                      ╲  Bezier bridge
                        ╲
          ◄──t2── ep2
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/depth-aware-vessel-analysis.git
cd depth-aware-vessel-analysis
pip install -e .
```

## Usage

### Single Image Analysis

```bash
python analyze_single.py --image path/to/multi_color_gt.png --output results/
```

Output:
- `metrics.json` — quantified morphological metrics
- `single_gt_skeleton.png` — flat projection skeleton
- `r_skeleton.png` — bottom-layer (R) skeleton with EP/JN markers
- `b_skeleton.png` — top-layer (B) skeleton with EP/JN markers
- `combined_rb_skeleton.png` — depth-separated overlay
- `r_bridging_detail.png` / `b_bridging_detail.png` — bridge visualization

### Batch Comparison (GT vs Virtual Staining)

```bash
python analyze_batch.py \
    --data-dir path/to/data/ \
    --output batch_results/ \
    --models LBBDM PBBDM LSGAN Pix2pix WGANGP
```

Expected data structure:
```
data/
├── Multi-color/
│   ├── Pix2pix/          # GAN outputs: {sid}_fake_B.png, {sid}_real_B.png
│   ├── LSGAN/
│   ├── WGANGP/
│   ├── LBBDM/            # Diffusion outputs: {sid}/output_0.png
│   └── PBBDM/
└── Single-color/
    ├── Pix2pix/
    ├── LSGAN/
    ├── WGANGP/
    ├── LBBDM/
    └── PBBDM/
```

## Pipeline

```
Multi-color GT image
        │
        ├── Gaussian blur (kernel=7)
        │
        ├── R channel extraction ──► R mask (thresh=30) ──► R skeleton
        │                                                      │
        │                                                      ├── Dominance filter (R>B)
        │                                                      │
        │                                                      ├── Gap bridging
        │                                                      │
        │                                                      └── Quantify (L, EP, JN)
        │
        ├── B channel extraction ──► B mask (hysteresis 50/20) ──► B skeleton
        │                                                            │
        │                                                            ├── Dominance filter (B>R)
        │                                                            │
        │                                                            ├── Gap bridging
        │                                                            │
        │                                                            └── Quantify (L, EP, JN)
        │
        └── Combined R+B ──► EP classification (real vs connected)
                           ──► Final depth-aware metrics
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Length** | Total skeleton pixels (vessel length proxy) |
| **Endpoints (EP)** | Vessel termination points. "Connected EP" (touching other channel) are excluded |
| **Junctions (JN)** | Bifurcation points where ≥3 branches meet. Verified by topology |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `blur_kernel` | 7 | Gaussian blur kernel size |
| `r_thresh` | 30 | R channel mask threshold |
| `b_seed_thresh` | 50 | B channel hysteresis seed threshold |
| `b_low_thresh` | 20 | B channel hysteresis low threshold |
| `dominance_margin` | 0 | R-B dominance margin |
| `bridge_max_dist` | 80 | Maximum bridge distance (px) |
| `bridge_max_angle` | 60 | Maximum angle deviation (degrees) |
| `baseline_fraction` | 0.8 | Fraction of image height for analysis region |

## Module Structure

```
vessel_analysis/
├── __init__.py       # Package entry point
├── core.py           # Preprocessing, mask extraction, skeletonization
├── bridge.py         # Tangent-guided gap bridging algorithm
├── metrics.py        # Morphological quantification (L, EP, JN)
└── visualize.py      # Skeleton overlay and marker visualization
```

## Citation

If you use this tool in your research, please cite:

```bibtex
@article{baik2026depth,
  title={Depth-aware Vessel Network Analysis for Virtual Staining Evaluation},
  author={Baik, Seungbum and others},
  year={2026}
}
```

## License

MIT License
