# Diffusion Virtual Staining for Multi-Layer Vessel Images

Analysis code and final figures for:

> **Latent Brownian Bridge Diffusion for Depth-Aware Virtual Staining of Angiogenic Microvasculature in Microphysiological Systems**
> Jungseub Lee†, Minh Huyen Le†, Huy Hieu Pham, Noo Li Jeon
> †Equal contribution. In preparation for submission to *Medical Image Analysis (MedIA)*.

This repository contains the **downstream analysis pipeline, figure-generation scripts, and final manuscript figures** for the paper. The companion repository [`minhhuyenle/AngioLBBDM`](https://github.com/minhhuyenle/AngioLBBDM) hosts the LBBDM model training and inference code.

---

## What the paper does

Single-plane brightfield (BF) images of angiogenic microvascular networks in a microfluidic MPS are translated into **depth-encoded fluorescence (FL) images** by a Latent Brownian Bridge Diffusion Model (LBBDM). The depth-encoded target preserves axial vascular organization as color within a compact 2D representation, enabling **layer-resolved morphometric analysis from label-free input** — without destructive endpoint staining.

Beyond image-level fidelity, the framework is evaluated by downstream quantification of vessel area, total vessel length, junctions, and endpoints at both the whole-image and per-layer levels.

### Key results

| Metric | Multi-color (depth-aware) | Single-color |
|---|---|---|
| MAE | **0.0351** | 0.0166 |
| MSE | **0.0096** | 0.0055 |
| PSNR (dB) | **20.5350** | 23.0176 |
| SSIM | 0.8569 | 0.9536 |
| LPIPS | **0.1424** | **0.1258** |
| Vessel area R² (global) | **0.92** | 0.90 |
| Vessel length R² (global) | **0.95** | 0.94 |
| Vessel area R² (per-layer, bottom/top) | 0.903 / 0.979 | — |
| Junction MAE (within ±3) | 1.6 / 1.4 (85% / 87%) | — |
| Endpoint MAE (within ±3) | 2.7 / 3.0 (71% / 71%) | — |
| Inference time | 9 s/sample (LBBDM) vs 60 s (PBBDM) | — |

LBBDM outperforms representative GAN baselines (LSGAN, Pix2pix, WGAN-GP) and a pixel-space Brownian Bridge model (PBBDM), with the clearest advantage in the more challenging multi-color depth-aware task.

---

## Repository structure

```
scripts/                           Source code
├── ppt_builders/                  PPTX builders, grouped per paper figure
│   ├── fig1_schematic/            Fig 1 overview schematic (Node.js)
│   ├── fig1e_pipeline/            Fig 1E conceptual pipeline
│   ├── fig1_dataset/              Fig 1B dataset construction
│   ├── fig4_benchmarks/           Fig 4 box-plot benchmarks
│   ├── fig5_qualitative/          Fig 5 qualitative comparison
│   ├── fig6_scatter/              Fig 6 global concordance scatter
│   ├── figS2_bridging/            Fig S2 gap-bridging examples
│   └── _misc/                     General-purpose builders
│
├── analysis/                      Python analysis pipelines
│   ├── morphometric/              Vessel area, length, junctions, endpoints
│   ├── rgb_channel/               RGB channel decomposition (depth encoding)
│   ├── skeleton/                  Skeletonization + tangent-guided bridging
│   ├── depth/                     Depth color analysis
│   ├── separation/                Contour/gradient/watershed separation
│   ├── endpoint/                  Endpoint detection
│   └── selection/                 Representative-sample selection
│
├── panel_generators/              Python scripts that render panel images
│   ├── per_layer/                 Per-layer metrics & scatter
│   ├── figS2/                     Fig S2 panel composition
│   ├── panels/                    General panel rendering
│   ├── figures/                   Main paper-figure renderers
│   ├── zoom/                      3-stage zoom panels
│   └── fig1e/                     Fig 1E figure assembly
│
├── viewer/                        Local HTML viewer for browsing samples
└── debug/                         Debugging scripts (kept for reference)

figures/                           Final PPTX figures (organized per paper figure)
├── fig1_overview/                 Fig 1 schematic + Fig 1E pipeline
├── fig2_depth_aware/              Fig 2 depth-aware skeletonization
├── fig4_benchmarks/               Fig 4 quantitative benchmarks
├── fig5_qualitative/              Fig 5 qualitative comparison
├── fig6_global/                   Fig 6 global concordance (assets in panels/)
├── fig7_per_layer/                Fig 7 per-layer concordance
└── figS2_bridging/                Fig S2 bridging
                                   (each contains _archive/ for older versions)

panels/                            Rendered panel components (PNG/PDF)
├── fig_main/                      Top-level Fig_panel_* and Fig_depth_*
├── per_sample/                    Per-sample renders
├── per_layer_depth/               Layer-resolved depth panels
├── scatter/{v1, v2}/              Scatter-plot panel components
├── perlayer_scatter/              Per-layer concordance scatters
├── s2_bridging/                   Fig S2 layer panels
├── stats/                         Stat bars / scatters
├── selected/                      Selected-sample overlays
└── misc/                          Other rendered outputs

results/                           Quantitative results & paper-relevant text
├── final_results.json             Top-level paper numbers
├── final_results_holefilled_crop10.json
├── best_representative_samples.txt
├── latex_updates.tex              Manuscript-side text snippets
└── *_results.json                 Model-comparison and per-metric breakdowns

depth-aware-vessel-analysis/       Independent Python subproject for
                                   layer-resolved vessel quantification
                                   (see its own README.md)
```

Older intermediate versions of code and figures are stored under `_archive/` subfolders within their respective categories — accessible for traceability but visually separated from the current files.

---

## Figure → file mapping

| Paper figure | Final PPTX | Builder script |
|---|---|---|
| **Fig 1** Overview workflow | `figures/fig1_overview/depth_schematic_v9.pptx` | `scripts/ppt_builders/fig1_schematic/create_schematic_v9.js` |
| **Fig 1E** Conceptual pipeline | `figures/fig1_overview/Figure_1E_Pipeline_v3b.pptx` | `scripts/ppt_builders/fig1e_pipeline/create_fig1e_ppt_v3.js` |
| **Fig 2** Depth-aware skeletonization | `figures/fig2_depth_aware/Figure_DepthAware_v2.pptx` | `depth-aware-vessel-analysis/` + `scripts/ppt_builders/fig1_dataset/` |
| **Fig 4** Benchmarks (box plots) | `figures/fig4_benchmarks/Figure_A4_final.pptx` | `scripts/ppt_builders/fig4_benchmarks/build_a4_final.js` |
| **Fig 5** Qualitative comparison | `figures/fig5_qualitative/Figure_Complete_v3.pptx` | `scripts/ppt_builders/fig5_qualitative/build_complete_ppt.js` |
| **Fig 6** Global concordance | (panels in `panels/perlayer_scatter/`) | `scripts/ppt_builders/fig6_scatter/build_scatter_ppt_v2.js` |
| **Fig 7** Per-layer concordance | `figures/fig7_per_layer/Figure_Perlayer_Concordance_v12.pptx` | `scripts/panel_generators/per_layer/gen_perlayer_metrics_v3.py` |
| **Fig S2** Bridging | `figures/figS2_bridging/Figure_S2_Bridging_v2.pptx` | `scripts/ppt_builders/figS2_bridging/create_figS2_ppt_v2.js` |

---

## Data availability

Paired brightfield / fluorescence images and model-output predictions (~382 MB) are **not stored in this repository.** They will be deposited at a public archive (Zenodo / Figshare) on publication and linked here.

Per-model output directory layout used by the analysis pipeline:

```
[1] Data/
├── Multi-color/{LBBDM, LSGAN, PBBDM, Pix2pix, WGANGP}/<sample_id>/output_*.png
└── Single-color/{LBBDM, LSGAN, PBBDM, Pix2pix, WGANGP}/<sample_id>/output_*.png
```

---

## Reproducing the analysis

1. Obtain the paired dataset (see Data availability) and place it at `[1] Data/`.
2. Install Python dependencies for the `depth-aware-vessel-analysis` subproject:
   ```
   pip install -r depth-aware-vessel-analysis/requirements.txt
   ```
3. Install Node.js dependencies for PPTX builders:
   ```
   npm install
   ```
4. Run the morphometric analysis (vessel area, length, junctions, endpoints):
   ```
   python scripts/analysis/morphometric/final_analysis.py
   ```
5. Regenerate figures by running the corresponding builder script in `scripts/ppt_builders/`.

For model training/inference, see the companion repository: [`minhhuyenle/AngioLBBDM`](https://github.com/minhhuyenle/AngioLBBDM).

---

## Citation

```
@article{lee2026lbbdm,
  title   = {Latent Brownian Bridge Diffusion for Depth-Aware Virtual Staining
             of Angiogenic Microvasculature in Microphysiological Systems},
  author  = {Lee, Jungseub and Le, Minh Huyen and Pham, Huy Hieu and Jeon, Noo Li},
  journal = {Medical Image Analysis},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## License

See [LICENSE](LICENSE). Contact the corresponding author (Noo Li Jeon) for use beyond the license terms.
