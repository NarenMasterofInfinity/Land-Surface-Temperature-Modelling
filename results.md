# Results Summary (Two Training Approaches)

Date: 2026-02-05
Project root: `/home/naren-root/Documents/FYP2/Project`

This file summarizes results for the two training modes mentioned:
1) **MODIS/VIIRS used for training** (weak supervision or direct low‑res guidance).
2) **MODIS/VIIRS discarded** (training on Landsat-only supervision).

Sources: `metrics/`, `algorithms.md`, and model-specific eval CSVs.

---

## Approach 1 — MODIS/VIIRS used in training

### A) CNN LR/HR (weak supervision using MODIS/VIIRS)
From `metrics/cnn_lr_hr_report.csv` (rows with `split=weak`).

| Model | RMSE | SSIM | PSNR | SAM | CC | Notes |
|---|---:|---:|---:|---:|---:|---|
| `cnn_lr_hr` | 5.9065 | 0.8802 | 18.1530 | 8.67e-07 | 0.2414 | weak split |
| `cnn_lr_hr_convnext` | 6.1079 | 0.8845 | 18.0285 | 2.71e-07 | 0.2279 | weak split |
| `cnn_lr_hr_resnet` | 5.8422 | 0.9047 | 18.2248 | 1.98e-07 | 0.2495 | weak split |
| `cnn_lr_hr_hrnet_200` | 5.6693 | 0.9414 | 18.5925 | 5.79e-07 | 0.3334 | weak split |

### B) Fusion baselines (MODIS/VIIRS)
Mean RMSE over eval dates.

| Method | MODIS RMSE | n | VIIRS RMSE | n | Source |
|---|---:|---:|---:|---:|---|
| STARFM | 5.0887 | 12 | 7.2242 | 28 | `metrics/fusion_baselines/starfm/*/starfm_eval_metrics.csv` |
| USTARFM | 7.5020 | 15 | 6.7359 | 24 | `metrics/fusion_baselines/ustarfm_*/*_eval_metrics.csv` |
| FSDAF | 6.1397 | 18 | 7.3813 | 34 | `metrics/fusion_baselines/fsdaf/*/fsdaf_eval_metrics.csv` |

### C) Linear baselines (feature set includes MODIS/VIIRS)
Observed mean metrics from `algorithms.md` (based on `metrics/linear_baselines/ALL_linear_metrics.csv`).

| Model | RMSE | SSIM | PSNR | SAM | CC | ERGAS |
|---|---:|---:|---:|---:|---:|---:|
| OLS | 5.7299 | 0.7636 | 21.3948 | 0.000197 | 0.4988 | 0.4626 |
| Ridge | 5.8096 | 0.7568 | 21.2533 | 0.000201 | 0.4979 | 0.4699 |
| Lasso | 5.8101 | 0.7568 | 21.2525 | 0.000201 | 0.4978 | 0.4700 |
| ElasticNet | 5.7980 | 0.7578 | 21.2746 | 0.000201 | 0.4986 | 0.4689 |

### D) Tree baselines (explicit MODIS/VIIRS monthly aggregates)
Observed mean metrics from `algorithms.md` (based on `metrics/tree_baselines/ALL_tree_metrics.csv`).

| Model | RMSE | SSIM | PSNR | SAM | CC | ERGAS |
|---|---:|---:|---:|---:|---:|---:|
| DecisionTree | 0.4679 | 0.6037 | 10.5131 | 0.000000 | 0.0926 | 30961.3959 |
| LightGBM | 0.3878 | 0.5621 | 13.2615 | 0.349561 | 0.1388 | 17051.5562 |

### E) CNN input ablations that include MODIS/VIIRS
Mean metrics across eval dates, computed from each run’s `cnn_eval_metrics.csv`.

| Run | RMSE | SSIM | PSNR | CC | Source |
|---|---:|---:|---:|---:|---|
| `cnn_default` | 6.1657 | 0.8720 | 23.3546 | 0.5522 | `metrics/deep_baselines/cnn/cnn_eval_metrics.csv` |
| `cnn_modis_lst` | 7.4523 | 0.8245 | 21.3938 | 0.0959 | `metrics/deep_baselines/cnn/modis_lst/cnn_eval_metrics.csv` |
| `cnn_viirs_lst` | 6.9478 | 0.8248 | 21.9331 | 0.0988 | `metrics/deep_baselines/cnn/viirs_lst/cnn_eval_metrics.csv` |
| `cnn_era5_modis` | 6.7241 | 0.8274 | 22.2775 | 0.1686 | `metrics/deep_baselines/cnn/era5_meteorology_modis/cnn_eval_metrics.csv` |
| `cnn_era5_viirs` | 6.7984 | 0.8226 | 22.1599 | 0.1304 | `metrics/deep_baselines/cnn/era5_meteorology_viirs/cnn_eval_metrics.csv` |
| `cnn_veg_modis` | 6.5142 | 0.8699 | 22.8445 | 0.5655 | `metrics/deep_baselines/cnn/vegetation_indices_modis/cnn_eval_metrics.csv` |
| `cnn_veg_viirs` | 6.2610 | 0.8722 | 23.3332 | 0.5305 | `metrics/deep_baselines/cnn/vegetation_indices_viirs/cnn_eval_metrics.csv` |
| `cnn_builtup_modis` | 7.0476 | 0.8255 | 21.9122 | 0.4048 | `metrics/deep_baselines/cnn/builtup_proxies_modis/cnn_eval_metrics.csv` |

---

## Approach 2 — MODIS/VIIRS discarded (Landsat-only supervision)

### A) CNN LR/HR (Landsat-only split)
From `metrics/cnn_lr_hr_report.csv` (rows with `split=landsat`).

| Model | RMSE | SSIM | PSNR | SAM | CC | Notes |
|---|---:|---:|---:|---:|---:|---|
| `cnn_lr_hr` | 6.7303 | 0.8460 | 21.4544 | 0.0 | 0.4926 | landsat split |
| `cnn_lr_hr_convnext` | 6.9591 | 0.8384 | 22.5049 | 0.006143 | 0.4997 | landsat split |
| `cnn_lr_hr_resnet` | 6.6256 | 0.8527 | 21.6089 | 0.0 | 0.5602 | landsat split |
| `cnn_lr_hr_hrnet_200` | 6.6565 | 0.8295 | 21.5721 | 0.0 | 0.5064 | landsat split |

### B) Deep baselines (test RMSE; inputs not logged here)
From `metrics/rmse_comparison.csv` (test RMSE values).

| Model | Test RMSE | Source |
|---|---:|---|
| `cnn` | 11.7538 | `metrics/deep_baselines/cnn/cnn_eval_metrics.csv` |
| `unet` | 12.2146 | `metrics/adv_deep/unet_resnet/unet_eval_metrics.csv` |
| `resnet` | 15.3306 | `metrics/deep_baselines/resnet/resnet_eval_metrics.csv` |
| `convext` | 13.8741 | `metrics/deep_baselines/convext/convext_eval_metrics.csv` |
| `mlp` | 179.1007 | `metrics/deep_baselines/mlp/mlp_eval_metrics.csv` |

---

## HRNet (cnn_lr_hr_hrnet_200)

From `metrics/cnn_lr_hr_report.csv`.

| Split | RMSE | SSIM | PSNR | SAM | CC |
|---|---:|---:|---:|---:|---:|
| overall | 5.7743 | 0.9295 | 18.9094 | 5.17e-07 | 0.3520 |
| weak | 5.6693 | 0.9414 | 18.5925 | 5.79e-07 | 0.3334 |
| landsat | 6.6565 | 0.8295 | 21.5721 | 0.0 | 0.5064 |

---

## XAI — Integrated Gradients (IG) & DeepLIFT

Computed for `cnn_lr_hr_hrnet_200` from:
- IG: `figures/deep_baselines/cnn_lr_hr/cnn_lr_hr_hrnet_200/ig/hrnet_ig_channel_importance.csv`
- DeepLIFT: `figures/deep_baselines/cnn_lr_hr/cnn_lr_hr_hrnet_200/deeplift/hrnet_deeplift_channel_importance.csv`

Top-10 channels by mean importance (overall):

**IG (Top 10)**
| Rank | Channel | Importance |
|---|---|---:|
| 1 | `era5_1` | 1.451788e-05 |
| 2 | `era5_4` | 1.173267e-05 |
| 3 | `era5_3` | 9.639173e-06 |
| 4 | `era5_0` | 9.317302e-06 |
| 5 | `era5_6` | 6.865970e-06 |
| 6 | `world_0` | 3.317616e-06 |
| 7 | `era5_7` | 3.010517e-06 |
| 8 | `s2_10` | 2.130764e-06 |
| 9 | `s2_12` | 1.593947e-06 |
| 10 | `s2_1` | 1.248932e-06 |

**DeepLIFT (Top 10)**
| Rank | Channel | Importance |
|---|---|---:|
| 1 | `era5_1` | 1.720483e-05 |
| 2 | `era5_4` | 1.471588e-05 |
| 3 | `era5_3` | 1.181955e-05 |
| 4 | `era5_0` | 1.137899e-05 |
| 5 | `era5_6` | 8.064076e-06 |
| 6 | `world_0` | 3.862978e-06 |
| 7 | `era5_7` | 3.595451e-06 |
| 8 | `s2_10` | 2.321770e-06 |
| 9 | `s2_12` | 1.771072e-06 |
| 10 | `s2_1` | 1.407107e-06 |

Per-date XAI outputs are also available:
- IG: `figures/deep_baselines/cnn_lr_hr/cnn_lr_hr_hrnet_200/ig/ig_channel_importance_YYYY-MM-DD.csv`
- DeepLIFT: `figures/deep_baselines/cnn_lr_hr/cnn_lr_hr_hrnet_200/deeplift/deeplift_channel_importance_YYYY-MM-DD.csv`

---

## Gaps / Missing Artifacts

- No recorded metrics were found for `landsat_only_baseline/*` runs in `metrics/` or `logs/`.
- `metrics/deep_baselines/cnn/builtup_proxies_viirs/` has no `cnn_eval_metrics.csv`.
- If there are additional “MODIS/VIIRS‑discarded” runs (e.g., explicit inputs without modis/viirs), their metrics aren’t captured in the current `metrics/` tree.

---

## Data Source Validity (Coverage of Valid Pixels)

Computed from `metrics/deep_baselines/valid_px_report.csv` across **261 dates**.  
For each source, validity is summarized as:
- `mean_valid_frac` = mean(valid_px / max_valid_px)
- `total_valid_frac` = sum(valid_px) / (max_valid_px * n_used)

| Source | max_valid_px | mean_valid_frac | total_valid_frac |
|---|---:|---:|---:|
| Landsat | 2,748,643 | 0.5124 | 0.5124 |
| ERA5 | 2,128,496 | 1.0000 | 1.0000 |
| Sentinel-1 | 1,398,463 | 0.9760 | 0.9760 |
| Sentinel-2 | 1,399,502 | 0.9618 | 0.9618 |
| MODIS | 1,357 | 0.1965 | 0.1965 |
| VIIRS | 1,357 | 0.1874 | 0.1874 |
| DEM | 1,400,609 | 1.0000 | 1.0000 |
| WorldCover | 2,763,838 | 1.0000 | 1.0000 |
| Dynamic World | 2,764,071 | 1.0000 | 1.0000 |

Notes:
- MODIS/VIIRS valid pixels are **very sparse** relative to HR grids (max 1,357 valid px in this report), so mean valid fraction is low.
- `modis_tif_valid_px` and `viirs_tif_valid_px` matched their respective `*_valid_px` distributions and are omitted to avoid duplication.
