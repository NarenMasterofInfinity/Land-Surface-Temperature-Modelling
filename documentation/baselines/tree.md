# Tree-based baselines (LightGBM, optional RF/XGB)

## Goal
Train pixel-wise nonlinear baselines for 30 m LST reconstruction using tree-based regressors.

## Data sources
Works on any of:
- `madurai.zarr`
- `madurai_30m.zarr`
- `madurai_alphaearth_30m.zarr`

Requirements:
- A time coordinate (`time/date/t/datetime`)
- A target LST variable
- Feature variables (defaults to all non-mask, non-target vars)

## Train/test protocol
- **Monthly supervision**: Landsat labels are monthly composites.
- Time-based split on months: first `(1 - test_frac)` portion for training, last `test_frac` for test.
- Default: 80/20.

## Sampling strategy (memory-safe)
- Random spatial tiles per time step.
- Reservoir sampling to cap the total number of samples.
- Defaults tuned for 8–16 GB RAM.

## Temporal alignment (FINAL policy)
- Supervision unit = month (YYYY-MM).
- MODIS/VIIRS are aggregated within the same month (median after daily QC/cloud masking).
- Missingness flags are added:
  - `has_modis_month`, `n_modis_obs_month`
  - `has_viirs_month`, `n_viirs_obs_month`
- A month is kept if Landsat is valid and **(MODIS or VIIRS) is available**.

## Imputation
- Compute per-feature medians on train data.
- Replace NaN/Inf in train/test/full-map using train medians.
- Do not impute target; invalid target pixels are dropped.

## Models
Primary:
- LightGBM (`lgb.LGBMRegressor`)

Optional:
- RandomForest (`--use-rf`)
- XGBoost (`--use-xgb`) if installed

## Metrics
Same as linear baselines:
- RMSE, SSIM, PSNR, SAM, CC, ERGAS

Saved to:
- `metrics/tree_baselines/{model}_metrics.csv`
- `metrics/tree_baselines/ALL_tree_metrics.csv`
- `metrics/tree_baselines/run_config_and_summary.json`

## Figures
Per model:
- LST actual/pred/error map for a date before 2026-01-01.
- Metric time series + mean bars.

Maps are masked to the Madurai ROI polygon.

## Logging
Logs are written to `logs/tree_baselines_*.log`.

## Run
Example:
```bash
python baselines/tree/tree_baselines.py \
  --dataset madurai_30m \
  --target labels_30m/landsat/band_01 \
  --features <list> \
  --test-frac 0.2
```
