# Project History and Research README

Date: 2026-03-12
Project root: `/home/naren-root/Documents/FYP2/Project`

This document is a consolidated history of the work done so far on the Madurai land-surface-temperature (LST) super-resolution and fusion pipeline. It is meant to be the single place to understand:

- what data we built and how we cleaned it
- what model families were tried
- what metrics we observed
- what changed over time
- what worked, what failed, and what still needs caution
- what we learned from daytime and nighttime LST analysis

It does not replace the focused docs such as `results.md`, `architecture.md`, `data_preproc.md`, `inference_insights.md`, or the per-run CSVs in `metrics/`. Instead, it stitches them together into one project narrative.

---

## 1. Objective

The project goal is to reconstruct or predict high-resolution LST over Madurai by combining:

- sparse but high-resolution Landsat supervision
- coarse but frequent MODIS and VIIRS thermal observations
- meteorology from ERA5
- spatial context from Sentinel-1, Sentinel-2, DEM, WorldCover, and Dynamic World

Over time, the work evolved across three related targets:

1. 30 m high-resolution daytime LST reconstruction
2. 1 km daily Landsat-equivalent LST prediction using day and night coarse sensors
3. 2025 inference and hotspot analysis, including paired day-night cooling patterns

The persistent challenge across all efforts has been the same: Landsat gives the best target quality, but it is sparse in time, while MODIS and VIIRS are temporally rich but spatially coarse and often quality-limited.

---

## 2. Data Assets and Canonical Representations

The repo contains two main Zarr views of the data:

- `madurai.zarr`
  - product-native organization
  - each source keeps its own CRS and transform metadata
  - used when native-grid handling matters
- `madurai_30m.zarr` / `madurai_30m_new.zarr`
  - canonical Landsat-aligned 30 m grid
  - used by most high-resolution models and reports

### 2.1 Core sources used in the project

- Landsat LST labels at 30 m
- MODIS day/night thermal products
- VIIRS day/night thermal products
- ERA5 meteorology
- Sentinel-1
- Sentinel-2
- DEM
- WorldCover
- Dynamic World

### 2.2 Dataset inventory snapshot

From `30m_report/report.md`:

- Landsat labels: `4029 x 2 x 1610 x 1724`
- ERA5: `4029 x 8 x 1610 x 1724`
- Sentinel-1: `133 x 6 x 1610 x 1724`
- Sentinel-2: `133 x 17 x 1610 x 1724`
- DEM: `1 x 3 x 1610 x 1724`

Missingness from the same report highlights why the task is difficult:

| Variable | Missing fraction |
|---|---:|
| Landsat | `0.9269` |
| Sentinel-1 | `0.5327` |
| Sentinel-2 | `0.6809` |
| ERA5 | `0.3659` |
| DEM | `0.4954` |
| WorldCover | `0.0043` |
| Dynamic World | `0.0042` |

Interpretation:

- high-quality supervision is the scarcest resource in the project
- Sentinel coverage is useful but sparse relative to daily modeling goals
- static layers are reliable
- meteorology is much more available than HR remote sensing

---

## 3. Data Handling Decisions and Tricks

This section is one of the most important parts of the project. A large fraction of progress came from getting the data handling right rather than changing the network alone.

### 3.1 Canonical 30 m grid

Main choice:

- Landsat was treated as the canonical 30 m grid
- all supported products were resampled onto that grid in `make_canonical_30m.py`

Rules:

- continuous variables use bilinear resampling
- categorical variables use nearest-neighbor resampling
- valid masks always use nearest-neighbor resampling
- invalid pixels are converted to `NaN` before resampling to avoid interpolation bleed

This was essential. Without valid-mask propagation and NaN-aware resampling, down/up-sampled artifacts become hard to separate from real thermal signal.

### 3.2 Native-grid preservation

We did not discard source-native geometry:

- `madurai.zarr` keeps each product on its own native grid
- CRS and affine transform are written into attrs
- first-valid-file reference grids are recorded for each product

This allowed experiments that respect coarse sensor structure while also supporting a canonical 30 m modeling path.

### 3.3 ERA5 affine override

ERA5 required explicit override handling in canonicalization:

- origin and pixel size are manually defined
- CRS is fixed to `EPSG:4326`

This avoided relying on inconsistent source metadata and prevented downstream georegistration drift.

### 3.4 No-data and validity policy

Source-specific no-data handling became a major stabilizer:

- Landsat nodata sentinel: `149`
- MODIS nodata sentinel: `-9999`
- VIIRS LST nodata sentinel: `0.0`
- floating-point arrays: validity derived from `isfinite`
- categorical arrays: valid by default unless explicitly cleaned later

This looks mundane, but it materially affected training stability and metric trustworthiness.

### 3.5 Quality-aware filtering

As the work matured, we increasingly moved from "use whatever overlaps" to "use only high-quality overlap."

Key tricks:

- patch-level valid fraction filtering
- sensor-level valid fraction thresholds
- strong-supervision-only filtering on Landsat dates
- explicit coarse sensor selection rather than averaging everything blindly
- quality-aware sensor gating in later architectures

The most recent and best-performing strong-only setup uses:

- Landsat as the only target
- MODIS as the only coarse sensor
- `--min-sensor-valid-frac 0.30`
- `--patch-valid-frac-min 0.40`

Launcher:

```bash
./scripts/run_arch_v2_strong_hq_modis.sh
```

### 3.6 Monthly aggregation for some baselines

The linear and tree baselines worked at a coarser supervision unit:

- supervision unit = month
- MODIS/VIIRS aggregated within month, usually median after QC/cloud handling
- explicit month-level missingness flags were added

This made the baseline experiments feasible and memory-safe, but they are not directly comparable to the best later strong-only patch training in all respects.

### 3.7 Imputation and sample control

For classical baselines and tabularized experiments:

- train-set median imputation was used for features
- invalid targets were dropped, not imputed
- reservoir sampling and tile sampling were used to stay within RAM limits
- chronological splits were preferred over random splits

### 3.8 Coverage insight from valid-pixel reports

From `metrics/deep_baselines/valid_px_report.csv` across 261 dates:

| Source | Mean valid fraction |
|---|---:|
| Landsat | `0.5124` |
| ERA5 | `1.0000` |
| Sentinel-1 | `0.9760` |
| Sentinel-2 | `0.9618` |
| MODIS | `0.1965` |
| VIIRS | `0.1874` |
| DEM | `1.0000` |
| WorldCover | `1.0000` |
| Dynamic World | `1.0000` |

The coarse thermal products are the sparsest spatially on the HR grid, which is why quality-aware use matters so much.

---

## 4. Chronology of Modeling Efforts

The project did not jump directly to the final strong model. It moved through several phases.

### 4.1 Phase A: Simple baselines

These were used to establish whether the problem had learnable signal at all.

#### Linear models

From `results.md` / `metrics/linear_baselines/ALL_linear_metrics.csv`:

| Model | RMSE | SSIM | PSNR | CC | ERGAS |
|---|---:|---:|---:|---:|---:|
| OLS | `5.7299` | `0.7636` | `21.3948` | `0.4988` | `0.4626` |
| Ridge | `5.8096` | `0.7568` | `21.2533` | `0.4979` | `0.4699` |
| Lasso | `5.8101` | `0.7568` | `21.2525` | `0.4978` | `0.4700` |
| ElasticNet | `5.7980` | `0.7578` | `21.2746` | `0.4986` | `0.4689` |

Takeaway:

- simple regressors were competitive enough to prove signal exists
- but they could not model spatial structure, residual detail, or quality-conditioned behavior

#### Tree baselines

From `results.md` / `metrics/tree_baselines/ALL_tree_metrics.csv`:

| Model | RMSE | SSIM | PSNR | CC | ERGAS |
|---|---:|---:|---:|---:|---:|
| DecisionTree | `0.4679` | `0.6037` | `10.5131` | `0.0926` | `30961.3959` |
| LightGBM | `0.3878` | `0.5621` | `13.2615` | `0.1388` | `17051.5562` |

Interpretation:

- these RMSE values look unrealistically strong relative to everything else
- the accompanying SSIM, CC, and ERGAS values are poor
- these runs should be treated as diagnostic baselines, not trustworthy headline results

### 4.2 Phase B: CNN input ablations with MODIS/VIIRS-guided training

Early deep baselines explored direct feature mixes.

From `results.md`:

| Run | RMSE | SSIM | PSNR | CC |
|---|---:|---:|---:|---:|
| `cnn_default` | `6.1657` | `0.8720` | `23.3546` | `0.5522` |
| `cnn_modis_lst` | `7.4523` | `0.8245` | `21.3938` | `0.0959` |
| `cnn_viirs_lst` | `6.9478` | `0.8248` | `21.9331` | `0.0988` |
| `cnn_era5_modis` | `6.7241` | `0.8274` | `22.2775` | `0.1686` |
| `cnn_era5_viirs` | `6.7984` | `0.8226` | `22.1599` | `0.1304` |
| `cnn_veg_modis` | `6.5142` | `0.8699` | `22.8445` | `0.5655` |
| `cnn_veg_viirs` | `6.2610` | `0.8722` | `23.3332` | `0.5305` |
| `cnn_builtup_modis` | `7.0476` | `0.8255` | `21.9122` | `0.4048` |

Takeaway:

- adding MODIS or VIIRS naively as direct input did not guarantee gains
- the model benefitted more from structured context than from raw coarse thermal injection
- this pushed the work toward explicit fusion/refinement ideas

### 4.3 Phase C: `cnn_lr_hr` family

This was the main early HR family and includes plain CNN, ResNet, ConvNeXt, and HRNet variants.

From `metrics/cnn_lr_hr_report.csv`:

| Model | Weak RMSE | Landsat-only RMSE | Overall RMSE |
|---|---:|---:|---:|
| `cnn_lr_hr` | `5.9065` | `6.7303` | `5.9941` |
| `cnn_lr_hr_convnext` | `6.1079` | `6.9591` | `6.1984` |
| `cnn_lr_hr_resnet` | `5.8422` | `6.6256` | `5.9255` |
| `cnn_lr_hr_hrnet_200` | `5.6693` | `6.6565` | `5.7743` |

What this taught us:

- under weak supervision, HRNet was the best within this family
- once evaluated on Landsat-only dates, performance degraded noticeably
- the family was useful, but it was still not the clean "strong Landsat target + coarse-sensor refinement" setup we ultimately wanted

Important note about the HRNet codepath:

- the target falls back to weak supervision when Landsat is absent
- MODIS and VIIRS are not used as a clean dedicated conditioning branch in the way later fusion models do

So HRNet was a meaningful step, but not the final direction.

### 4.4 Phase D: Strong-only deep baselines

Stronger supervision was then isolated more explicitly.

From saved runs:

| Run | Best epoch | Best `val_rmse_ls` |
|---|---:|---:|
| `plain_resnet_strong/resnet_strong_qaware_seed42` | `21` | `6.5602` |
| `hrnet_strong/hrnet_strong_seed42` | `6` | `6.8374` |

Takeaway:

- strong supervision alone did not solve the problem
- quality-aware ResNet beat strong HRNet, but neither was good enough
- this created the need for a more explicit two-stage thermal prior plus refinement design

### 4.5 Phase E: `arch_v1`

`arch_v1` introduced a more deliberate architecture:

- a low-capacity thermal base branch
- a high-resolution residual branch
- gating between residual heads
- grouped modality embeddings for S2, S1, DEM, WorldCover, Dynamic World

This is documented in `documentation/arch_v1_architecture.md`.

Best saved `arch_v1` runs:

| Run | Best epoch | Best `val_rmse_ls` |
|---|---:|---:|
| `additive_bias_strong_lr1e4_seed42` | `70` | `6.4843` |
| `attention_strong_lr1e4_seed42` | `57` | `6.5744` |
| `additive_bias_qgate_coarsefusion_strong_lr1e4_seed42` | `66` | `6.6380` |
| `additive_bias_qgate_strong_lr1e4_seed42` | `25` | `6.6929` |
| `additive_bias_mean_gate_strong_lr1e4_seed42` | `28` | `6.7517` |

Takeaway:

- `arch_v1` improved architecture quality and clarified the decomposition into base field + residual detail
- but numerically it still remained in the same broad band as the stronger ResNet baseline
- the big jump had not happened yet

### 4.6 Phase F: `arch_v2` STARFM-refine models

This is where the project made its clearest jump.

Core idea:

1. build a coarse thermal prior using STARFM-like fusion
2. optionally add bias / mean-group quality gating
3. learn a residual refinement against Landsat on strong dates

Best saved `arch_v2` runs:

| Run | Best epoch | Best `val_rmse_ls` |
|---|---:|---:|
| `starfm_addbias_mean_gate_hq_modis_seed42` | `121` | `4.5910` |
| `starfm_addbias_mean_gate_seed42` | `106` | `4.9080` |
| `starfm_addbias_mean_gate_evalfix_seed42` | `80` | `5.7438` |
| `starfm_addbias_mean_gate_temporal2025_seed42` | `66` | `6.1979` |
| `starfm_addbias_mean_gate_v2b_seed42` | `30` | `6.5674` |

This is the strongest progression in the whole repo:

- best strong HRNet: `6.8374`
- best strong ResNet: `6.5602`
- best `arch_v1`: `6.4843`
- earlier `arch_v2`: `4.9080`
- latest HQ-MODIS `arch_v2`: `4.5910`

#### Training-curve behavior of the latest best run

For `starfm_addbias_mean_gate_hq_modis_seed42`:

- best epoch: `121`
- best `val_rmse_ls`: `4.590966`
- final epoch logged: `146`
- final `val_rmse_ls`: `4.641147`

Convergence landmarks:

- below `10`: epoch `17`
- below `8`: epoch `20`
- below `6`: epoch `24`
- below `5`: epoch `36`
- below `4.8`: epoch `54`
- below `4.7`: epoch `60`
- below `4.6`: epoch `100`

This is a stable curve, not a one-epoch fluke.

#### Why the latest run is the current best direction

- it is strong-supervision-only
- it uses only high-quality Landsat targets
- it uses a single coarse sensor rather than blending unreliable signals
- it bakes data-quality filtering into sample selection
- it outperforms all earlier strong architectures in the repo

### 4.7 Phase G: 1 km day/night BaseNet work

The project also explored daily 1 km Landsat-equivalent prediction, especially to incorporate day and night thermal behavior more directly.

From `basenet_1km/results/run_summary.md`:

- best validation RMSE: `6.185286`
- features: `101`
- train range: `2020-01-04` to `2020-07-30`
- val range: `2023-01-04` to `2023-03-01`
- test range: `2024-01-07` to `2024-02-24`

From `good_archi/results/run_summary.md`:

- best validation RMSE: `3.975568`
- features: `101`
- train range: `2020-01-04` to `2022-12-27`
- val range: `2023-01-04` to `2023-12-30`
- test range: `2024-01-07` to `2024-12-24`

Feature set highlights:

- MODIS day
- MODIS night
- VIIRS day
- VIIRS night
- per-sensor validity and QC scores
- day-night differences
- lag features
- neighborhood statistics
- ERA5 bands and deltas
- static terrain and land-cover fractions

Takeaway:

- this line of work is important because day/night thermal structure is central to urban heat behavior
- the richer QC-aware feature engineering in `good_archi` was a major gain over the earlier 1 km base run
- the 1 km pipeline is not the same task as the 30 m patch-refinement models, but it is a valuable complementary direction

### 4.8 Phase H: Thermal base + residual network architecture consolidation

`architecture.md` documents the more mature thermal LR/HR formulation:

- BaseNet takes coarse spatiotemporal inputs
- UpsampleHead lifts coarse predictions to HR
- ResidualNet applies HR correction using Sentinel, DEM, land cover, and ERA5 context
- final prediction is `base_hr + residual`

This codified the project’s core modeling insight:

- large-scale temperature structure should be modeled by coarse thermal and meteorological context
- high-resolution sensors should specialize in residual correction, not carry the full thermal burden

---

## 5. Fusion Baseline Findings

The classical fusion baselines were important because they provided an interpretable non-neural reference.

Mean RMSE over eval dates:

| Method | MODIS RMSE | MODIS n | VIIRS RMSE | VIIRS n |
|---|---:|---:|---:|---:|
| STARFM | `5.0887` | `12` | `7.2242` | `28` |
| FSDAF | `6.1397` | `18` | `7.3813` | `34` |
| uSTARFM | `7.5020` | `15` | `6.7359` | `24` |

Main lessons:

- STARFM with MODIS was the strongest classical fusion baseline in this repo
- MODIS usually behaved better than VIIRS for the STARFM path
- learned refinement on top of a good coarse prior beats classical fusion alone

This is one reason the current best training path uses MODIS only.

---

## 6. Sensor Choice: MODIS vs VIIRS

This question came up repeatedly, and the answer depends on task definition.

### 6.1 What the repo metrics suggest

- for classical STARFM fusion, MODIS clearly beats VIIRS in RMSE
- for the current best strong-only `arch_v2` recipe, MODIS produced the best validated run
- day/night verification is more mixed: nighttime chosen-sensor logic often benefits from VIIRS

### 6.2 Quality-retention behavior

In the older `arch_v2` quality report, counts above valid-fraction thresholds were:

| Threshold | MODIS dates/rows | VIIRS dates/rows |
|---|---:|---:|
| `>= 0.2` | `1253` | `810` |
| `>= 0.3` | `1052` | `584` |
| `>= 0.4` | `835` | `404` |
| `>= 0.5` | `452` | `206` |

Interpretation:

- under stricter quality thresholds, MODIS retains substantially more usable support than VIIRS
- for strong training with explicit quality filtering, this matters more than nominal coverage counts alone

### 6.3 Working conclusion

- if the goal is the best current strong-only daytime 30 m training setup, choose MODIS
- if the goal is flexible 2025 day/night inference, keep both sensors and choose per date when warranted

---

## 7. Explainability Findings

From integrated gradients and DeepLIFT on `cnn_lr_hr_hrnet_200`:

Top channels were dominated by:

- ERA5 bands, especially `era5_1`, `era5_4`, `era5_3`, `era5_0`
- then some static and Sentinel-2 channels

Takeaway:

- meteorology was consistently one of the strongest drivers
- fine-scale optical context mattered, but after coarse thermal and meteorological structure
- this matches the later architectural separation into a thermal base plus residual correction

---

## 8. Daytime and Nighttime LST Progress

This project is not only about fitting validation RMSE. It is also about building usable city-scale temperature maps and extracting interpretable urban heat structure.

### 8.1 2025 daytime inference coverage

From `outputs/arch_v2_render/starfm_addbias_mean_gate_v2b_2025/meta/run_summary.txt`:

- input mode: `day`
- year: `2025`
- available dates: `259`
- STARFM MODIS dates: `134`
- STARFM VIIRS dates: `125`
- training stats available dates: `226`
- train dates: `158`
- val dates: `22`

### 8.2 Daytime verification by month

From `metrics/day_render_verify/starfm_addbias_mean_gate_v2b_2025_verify/summary_by_month.csv`:

| Month | Mean RMSE |
|---|---:|
| Jan | `2.8759` |
| Feb | `3.4902` |
| Mar | `4.9021` |
| Apr | `6.3795` |
| May | `7.0435` |
| Jun | `6.7534` |
| Jul | `10.5529` |
| Aug | `9.5159` |
| Sep | `6.7517` |
| Nov | `6.0549` |
| Dec | `3.3222` |

Average of monthly mean RMSE values: about `6.1493`.

Interpretation:

- the summer/monsoon portion is much harder than winter
- the model is clearly season-sensitive
- annual deployment quality should be discussed seasonally, not as one single number

### 8.3 Nighttime verification

From `metrics/night_render_verify/starfm_addbias_mean_gate_v2b_night_2025_fix1_verify/summary_by_sensor.csv`:

| Sensor mode | Dates | Mean RMSE | Median RMSE | Mean MAE |
|---|---:|---:|---:|---:|
| chosen | `106` | `3.7224` | `3.8604` | `2.9787` |
| modis | `106` | `4.4830` | `4.1585` | `3.6319` |
| viirs | `106` | `3.2941` | `2.6428` | `2.7112` |

Interpretation:

- nighttime performance can favor VIIRS
- allowing per-date chosen-sensor logic is better than forcing MODIS at night
- sensor choice should remain task-dependent rather than globally fixed

---

## 9. Hotspot and Actionable Insight Results

From `inference_insights.md` and the 2025 actionable-insights outputs:

- `259` daytime dates processed
- `106` night dates available
- `99` matched day-night pairs
- `435` chronic hotspot regions extracted

Thermal pattern summary:

- mean daily city baseline: `34.32 C`
- baseline range: `27.00 C` to `49.09 C`
- median hotspot anomaly: `6.18 C`
- 90th percentile hotspot anomaly: `14.75 C`
- median paired day-night cooling: `6.35 C`
- 90th percentile paired day-night cooling: `13.48 C`
- only `2` paired dates showed negative mean cooling

Interpretation:

- daytime hotspot ranking is more statistically supported than cooling-deficit ranking because daytime dates are more numerous
- cooling analysis is still useful, but it is built on a smaller matched subset
- the project now supports not only prediction, but also urban-heat interpretation and intervention ranking

The hotspot analysis identified several persistent hotspot archetypes:

- dense-built hotspots
- low-built but thermally intense regions
- weak-night-cooling regions

That is an important shift in the project: from pure super-resolution toward applied urban-climate analytics.

---

## 10. Known Metric Caveats and Failure Modes

Not every metric artifact in the repo should be trusted equally.

### 10.1 Broken `landsat_valid_frac` in old quality CSVs

In older `arch_v2` quality files, `landsat_valid_frac` sometimes contains values that are not true fractions. Treat that field as unreliable unless recomputed.

What still appears usable:

- `modis_valid_frac`
- `viirs_valid_frac`

### 10.2 Flat `val_base_rmse_ls` in some old `arch_v2` runs

For example, in `starfm_addbias_mean_gate_seed42`, `val_base_rmse_ls` is effectively constant and very large across epochs.

Interpretation:

- this is expected to be flatter than the learned refinement metric because the base is fixed
- the magnitude suggests the logging path is not a clean standalone measure of base quality
- do not use old `val_base_rmse_ls` as the primary model-selection metric

### 10.3 Tree baseline metrics are internally inconsistent

Extremely low RMSE combined with poor CC, SSIM, and massive ERGAS indicates those results should not be treated as headline evidence.

### 10.4 Direct comparison across tasks needs care

The repo mixes:

- 30 m patch refinement
- monthly aggregate baselines
- 1 km daily prediction
- 2025 city-scale inference

These are related, but not numerically interchangeable.

---

## 11. What We Learned Overall

Across the full project history, the main insights are:

1. Data handling mattered as much as model design.
2. Landsat-only strong supervision is the cleanest target for trustworthy learning.
3. MODIS/VIIRS are valuable, but only when quality-aware and used in the right role.
4. A coarse thermal prior plus HR residual correction works much better than asking one network to do everything directly.
5. MODIS is the best single-sensor choice for the current strong-only daytime training recipe.
6. VIIRS remains very useful, especially in nighttime evaluation and date-wise chosen-sensor inference.
7. Meteorology is consistently important; explainability and architecture both support that.
8. Seasonal behavior is strong, so evaluation must remain month-aware.
9. The project has progressed from baseline prediction to credible city-scale hotspot and cooling analysis.

---

## 12. Current Best Direction

For the user request of:

- only high-quality Landsat target
- only high-quality coarse sensor
- strong supervision only

the current best verified answer in this repo is:

- model family: `arch_v2`
- run: `starfm_addbias_mean_gate_hq_modis_seed42`
- best `val_rmse_ls`: `4.5910`
- sensor choice: MODIS only

Training launcher:

```bash
./scripts/run_arch_v2_strong_hq_modis.sh
```

This script uses:

- `--coarse-sensor modis`
- `--min-sensor-valid-frac 0.30`
- `--patch-valid-frac-min 0.40`
- strong-supervision-only `arch_v2` training

This is the strongest 30 m training recipe currently evidenced by the saved metrics.

---

## 13. Key Reference Files

For deeper detail, use these alongside this README:

- `results.md`
- `data_preproc.md`
- `architecture.md`
- `documentation/arch_v1_architecture.md`
- `documentation/baselines/linear_baselines.md`
- `documentation/baselines/tree.md`
- `30m_report/report.md`
- `inference_insights.md`
- `good_archi/README.md`
- `good_archi/results/run_summary.md`
- `basenet_1km/results/run_summary.md`

Metric artifacts used heavily in this summary:

- `metrics/cnn_lr_hr_report.csv`
- `metrics/plain_resnet_strong/resnet_strong_qaware_seed42/metrics.csv`
- `metrics/hrnet_strong/hrnet_strong_seed42/metrics.csv`
- `metrics/arch_v1/*/metrics.csv`
- `metrics/arch_v2/*/metrics.csv`
- `metrics/fusion_baselines/*/*_eval_metrics.csv`
- `metrics/day_render_verify/*/summary_by_month.csv`
- `metrics/night_render_verify/*/summary_by_sensor.csv`

---

## 14. Suggested Next Steps

The next sensible steps are:

1. standardize one canonical experiment ledger so each run has config, split policy, and trusted headline metrics in one schema
2. recompute old quality CSVs where `landsat_valid_frac` is malformed
3. keep separate scoreboards for:
   - 30 m strong-supervision training
   - 1 km day/night daily prediction
   - 2025 deployment verification
4. evaluate whether the latest HQ-MODIS `arch_v2` logic should also be adapted for nighttime, where VIIRS may be the better primary sensor
5. extend hotspot analysis using the improved best checkpoint if it is not already the deployed one

---

## 15. Bottom Line

The project has moved from exploratory baselines to a defensible strong-supervision thermal fusion pipeline.

The most important progress markers are:

- from weakly guided CNNs to strong-only targets
- from direct coarse-sensor injection to quality-aware coarse priors
- from generic deep models to thermal-base-plus-residual architectures
- from baseline modeling to usable day/night urban-heat insights

At the moment, the clearest best 30 m training result is the high-quality MODIS-only strong `arch_v2` run with validation RMSE `4.5910`.
