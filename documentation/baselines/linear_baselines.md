<!-- ===================================================================
File: /home/naren-root/Documents/FYP2/Project/documentation/baselines/linear_baselines.md
=================================================================== -->

# Linear baselines for high-resolution LST (Madurai)

## Goal
Reconstruct **high-resolution LST at 30 m** with the **best achievable cadence** given the available time axis in the Zarr stores.  
Target output format: per-date 30 m LST maps + quantitative evaluation.

Because “daily 30 m LST” is not guaranteed by the inputs, linear baselines are defined as:

- **If the Zarr store has daily-ish time steps:** produce predictions per time step (daily-ish).
- **If the store is monthly-ish / sparse:** produce predictions per time step (monthly-ish / sparse).
- The script prints a cadence guess (`daily-ish`, `monthly-ish`, etc.) based on median time delta.

## Data sources (Zarr)
The baselines work on any of:
- `madurai.zarr`
- `madurai_30m.zarr`
- `madurai_alphaearth_30m.zarr`

The baseline script expects:
- A **time** coordinate (or equivalent: `time/date/t/datetime`)
- A **target** LST variable (recommended to provide explicitly via `--target`)
- Feature variables (defaults to “all numeric vars excluding qc/quality/mask/flags”)

## Handling NaN/Inf/NoData
All arrays may contain NaN/Inf/no-data.

In the baseline:
- Valid pixels are those where **target is finite**.
- Feature NaN/Inf are **median-imputed** using training-set medians (per feature).
- Metrics are computed only over pixels where both prediction and target are finite.

## Linear baseline models implemented
These are deliberately simple and fast:
1. **OLS** (LinearRegression)
2. **Ridge** (L2-regularized)
3. **Lasso** (L1-regularized)
4. **ElasticNet** (L1+L2)

All models:
- Use **StandardScaler** on features
- Fit on sampled pixels from training times (to stay feasible at 30 m)

## Train/test protocol
- Split by time (chronological):
  - Train = earliest `(1 - test_frac)` portion
  - Test = latest `test_frac` portion (default 0.2)

This matches the intended use: train on historical high-res availability and evaluate on future times.

## Metrics saved
For each model, for each test time step, the script computes and saves:

- **RMSE**
- **SSIM**
- **PSNR**
- **SAM** (Spectral Angle Mapper; meaningful primarily for multi-channel targets such as [Day, Night])
- **CC** (Pearson correlation on valid pixels)
- **ERGAS** (defaults to ratio = 1000/30, configurable)

Outputs are saved in:
`/home/naren-root/Documents/FYP2/Project/metrics/linear_baselines`

Files:
- `{model}_metrics.csv` (per-time metrics)
- `ALL_linear_metrics.csv` (combined)
- `run_config_and_summary.json` (configuration + mean/std summary)

## How to run
Install dependencies:
- `pip install scikit-learn scikit-image`

Run (recommended: explicitly set target once):
```bash
cd /home/naren-root/Documents/FYP2/Project/baselines/linear

python run_linear_baselines.py \
  --dataset madurai_30m \
  --start 2019-01-01 --end 2020-12-31 \
  --target <YOUR_LST_TARGET_VAR>
