# Actionable Insights

This folder implements a practical urban-heat decision-support pipeline on top of daily 30 m LST predictions.

Input:
- a directory of daily `.npy` LST maps
- a directory of nightly `.npy` LST maps
- an ROI mask
- optionally `madurai_30m.zarr` for land-cover suitability masks

Outputs:
- daily baseline and hotspot summary CSVs
- hotspot frequency / excess-heat / score maps as `.npy` and `.png`
- connected hotspot region tables and maps
- intervention cooling proxy maps
- ranked actionable-insight CSVs and figures

Default output location:
- `actionable_insights/outputs/<tag>/`

Run:

```bash
python3 actionable_insights/run_pipeline.py \
  --pred-dir outputs/arch_v2_render/starfm_addbias_mean_gate_v2b_2025/npy \
  --night-pred-dir outputs/arch_v2_render/starfm_addbias_mean_gate_v2b_night_2025_fix1/npy \
  --roi-mask outputs/arch_v2_render/starfm_addbias_mean_gate_v2b_2025/meta/madurai_roi_mask.npy \
  --root-30m madurai_30m.zarr \
  --out-root actionable_insights/outputs \
  --year 2025 \
  --min-valid-observations 20 \
  --min-cooling-observations 20 \
  --tag arch_v2_2025
```

Notes:
- The pipeline now filters inputs to one year. By default that year is `2025`.
- Pixels with too few valid day or day-night observations are excluded from persistence and cooling-deficit scoring.
- Intervention outputs are proxy rankings. The CSV now labels these values as `proxy_*` and ranks regions with `priority_score`.
- The current intervention engine is a proxy simulator. It does not rerun the thermal model.
- The outputs are still decision-support useful because they rank persistent hotspot zones by severity, nocturnal cooling deficit, suitability, and proxy cooling impact under simple intervention scenarios.
