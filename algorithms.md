# Algorithms Summary

This file tracks the algorithms used, their inputs/targets, and observed metrics.
Update this file after each new baseline/model run.

---

## Linear Baselines (pixel-wise)

**Models**
- OLS (LinearRegression)
- Ridge
- Lasso
- ElasticNet

**Target**
- `labels_30m/landsat/band_01` (Landsat 30 m LST)

**Input Features (default rules)**
- All numeric variables except:
  - `labels_30m/landsat/band_01` (target)
  - `labels_30m/landsat/band_02` (QA/aux)
  - Any cloud/QC/mask/flag bands
  - `static_30m/worldcover/band_01` (categorical)
  - `static_30m/dynamic_world/band_01` (categorical)
- If a custom `--features` list is provided, it is used as-is.

**Metrics (per time step)**
- RMSE, SSIM, PSNR, SAM, CC, ERGAS

**Outputs**
- Metrics CSVs: `metrics/linear_baselines/{model}_metrics.csv`
- Combined CSV: `metrics/linear_baselines/ALL_linear_metrics.csv`
- Figures: `metrics/linear_baselines/figures/*`
- Models: `models/linear_{model}.joblib`

**Observed Mean Metrics (from `ALL_linear_metrics.csv`)**

| Model | RMSE | SSIM | PSNR | SAM | CC | ERGAS |
|---|---:|---:|---:|---:|---:|---:|
| OLS | 5.729907 | 0.763625 | 21.394806 | 0.000197 | 0.498792 | 0.462561 |
| Ridge | 5.809584 | 0.756823 | 21.253289 | 0.000201 | 0.497860 | 0.469933 |
| Lasso | 5.810052 | 0.756756 | 21.252466 | 0.000201 | 0.497821 | 0.469976 |
| ElasticNet | 5.797998 | 0.757800 | 21.274597 | 0.000201 | 0.498553 | 0.468883 |

---

## Tree-Based Baselines (pixel-wise)

**Models**
- DecisionTreeRegressor (DT)
- LightGBM (LGBM)
- Optional: RandomForest (`--use-rf`)
- Optional: XGBoost (`--use-xgb`)

**Target**
- `labels_30m/landsat/band_01` (Landsat 30 m LST)

**Input Features (default rules)**
- All numeric variables except:
  - `labels_30m/landsat/band_01` (target)
  - `labels_30m/landsat/band_02` (QA/aux)
  - Any cloud/QC/mask/flag bands
  - `static_30m/worldcover/band_01` (categorical)
  - `static_30m/dynamic_world/band_01` (categorical)
- If a custom `--features` list is provided, it is used as-is.

**Temporal Alignment (FINAL policy)**
- Supervision unit = **month (YYYY-MM)** based on Landsat composites.
- MODIS/VIIRS aggregated within the same month (median of daily masked values).
- Missingness flags appended as features:
  - `has_modis_month`, `n_modis_obs_month`
  - `has_viirs_month`, `n_viirs_obs_month`
- Month is kept if Landsat exists and **(MODIS or VIIRS)** has coverage.

**Metrics (per time step)**
- RMSE, SSIM, PSNR, SAM, CC, ERGAS

**Outputs**
- Metrics CSVs: `metrics/tree_baselines/{model}_metrics.csv`
- Combined CSV: `metrics/tree_baselines/ALL_tree_metrics.csv`
- Feature importance: `metrics/tree_baselines/feature_importance_{model}.csv`
- Figures: `metrics/tree_baselines/figures/*`
- Models: `models/tree_{model}.joblib`

**Observed Mean Metrics (from `ALL_tree_metrics.csv`)**

| Model | RMSE | SSIM | PSNR | SAM | CC | ERGAS |
|---|---:|---:|---:|---:|---:|---:|
| DecisionTree | 0.467913 | 0.603671 | 10.513063 | 0.000000 | 0.092579 | 30961.395932 |
| LightGBM | 0.387765 | 0.562103 | 13.261502 | 0.349561 | 0.138829 | 17051.556201 |

---

## Metric Definitions

**RMSE (Root Mean Squared Error)**  
Measures average magnitude of error in the same units as LST. Lower is better.  
\( \sqrt{\frac{1}{N}\sum_i (y_i - \hat{y}_i)^2} \)

**SSIM (Structural Similarity Index)**  
Captures similarity in local structure, contrast, and luminance between maps. Higher is better; ~1 means very similar.

**PSNR (Peak Signal-to-Noise Ratio)**  
Measures fidelity relative to signal range; higher means less error.  
\( 10 \log_{10} \left(\frac{MAX^2}{MSE}\right) \), where \(MAX\) is data range.

**SAM (Spectral Angle Mapper)**  
Measures angular similarity between vectors (per-pixel). Lower is better; 0 means perfect alignment.  
Mean angle between prediction and target vectors:  
\( \arccos \left( \frac{\langle \mathbf{y}, \hat{\mathbf{y}} \rangle}{\|\mathbf{y}\|\|\hat{\mathbf{y}}\|} \right) \)

**CC (Pearson Correlation Coefficient)**  
Measures linear correlation between prediction and target; ranges \([-1, 1]\), higher is better.  
\( \frac{\mathrm{cov}(y, \hat{y})}{\sigma_y \sigma_{\hat{y}}} \)

**ERGAS**  
Normalized global error; lower is better. Accounts for resolution ratio and channel means.  
\( \frac{100}{r} \sqrt{ \frac{1}{C} \sum_c \frac{\mathrm{RMSE}_c^2}{\mu_c^2} } \)  
where \(r\) is coarse/fine resolution ratio.
