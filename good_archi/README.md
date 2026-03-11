# BaseNet 1km Pipeline (`good_archi`)

This module trains a QC-aware BaseNet to predict daily 1 km Landsat-equivalent LST using MODIS + VIIRS fusion with residual correction.

## Layout

- `config.yaml`: run config
- `train_basenet.py`, `evaluate_basenet.py`: main entrypoints
- `dataset_basenet.py`, `model_basenet.py`, `utils_*`: core modules
- `logs/`, `results/`, `checkpoints/`: runtime artifacts (same style as `patch_restnet`)

## Train

From repo root:

```bash
python good_archi/train_basenet.py --config good_archi/config.yaml
```

Optional overrides:

```bash
python good_archi/train_basenet.py --config good_archi/config.yaml --zarr_path /path/to/madurai.zarr --zarr_30m_path /path/to/madurai_30m.zarr
```

Smoke run:

```bash
python good_archi/train_basenet.py --config good_archi/config.yaml --smoke
```

## Evaluate

```bash
python good_archi/evaluate_basenet.py --base_dir good_archi
```

Ensemble training (multi-seed):

```bash
python good_archi/train_basenet.py --config good_archi/config_ft_full.yaml --seeds 42,123,999
```

Ensemble evaluation from manifest:

```bash
python good_archi/evaluate_basenet.py --base_dir good_archi --ensemble_manifest good_archi/results/ensemble_manifest.json
```

## Outputs

Each run writes:

- `logs/train.log`
- `checkpoints/best.pt`, `checkpoints/last.pt`
- `results/metrics.json`
- `results/metrics.csv`
- `results/predictions_sample/*.npy`
- `results/config_resolved.yaml`
- `results/run_summary.md`
- `splits.json`
- `tensorboard/` (if tensorboard available)

## Troubleshooting

### RMSE is too high (units/nodata)

If RMSE is very large (for example 30+ C), this is usually a data-scale issue.

The pipeline now performs:

- nodata sanitization using array attrs + configured sentinels
- unit harmonization (scale/offset + Kelvin to Celsius heuristics)
- hard sanity checks that abort on unrealistic target/feature ranges
- median imputation with NaN-mask features
- debug export: `results/debug_sample.csv`

Check these files first:

- `logs/train_basenet.log` for per-band conversion decisions
- `results/config_resolved.yaml` for exact thresholds/sentinels used
- `results/debug_sample.csv` for pointwise `y_true` vs `y_pred`

## Model Architectures

Set `model.arch` in `config.yaml`:

- `mlp`: baseline residual MLP correction
- `ft_transformer`: lightweight FT-Transformer correction
- `moe`: mixture-of-experts correction head

Optional advanced loss controls:

- `training.loss_type`: `huber` or `nll`
- `model.use_heteroscedastic`: enable `log_sigma` prediction for NLL
- `training.aux_delta_loss`: auxiliary residual supervision term
