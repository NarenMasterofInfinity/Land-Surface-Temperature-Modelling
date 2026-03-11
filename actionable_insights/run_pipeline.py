from __future__ import annotations

import argparse
from pathlib import Path

try:
    from actionable_insights.pipeline import PipelineConfig, run_pipeline
except ImportError:
    from pipeline import PipelineConfig, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate actionable urban heat insights from daily 30m LST maps.")
    parser.add_argument("--pred-dir", required=True, help="Directory containing daily lst_YYYYMMDD.npy predictions")
    parser.add_argument("--night-pred-dir", required=True, help="Directory containing nightly lst_YYYYMMDD.npy predictions")
    parser.add_argument("--roi-mask", required=True, help="Path to ROI mask .npy")
    parser.add_argument("--root-30m", default="madurai_30m.zarr", help="Optional 30m zarr for static land-cover suitability")
    parser.add_argument(
        "--out-root",
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Output root for actionable insights runs",
    )
    parser.add_argument("--tag", default="default")
    parser.add_argument("--year", type=int, default=2025, help="Only process predictions from this year")
    parser.add_argument("--threshold-mode", choices=["percentile", "zscore"], default="percentile")
    parser.add_argument("--hotspot-percentile", type=float, default=90.0)
    parser.add_argument("--hotspot-zscore", type=float, default=1.5)
    parser.add_argument("--chronic-frequency-threshold", type=float, default=0.30)
    parser.add_argument("--min-valid-observations", type=int, default=20)
    parser.add_argument("--min-cooling-observations", type=int, default=20)
    parser.add_argument("--min-region-pixels", type=int, default=25)
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8)
    parser.add_argument("--sample-png-every", type=int, default=30)
    parser.add_argument("--vmin", type=float, default=10.0)
    parser.add_argument("--vmax", type=float, default=70.0)
    args = parser.parse_args()

    config = PipelineConfig(
        pred_dir=args.pred_dir,
        night_pred_dir=args.night_pred_dir,
        out_root=args.out_root,
        roi_mask=args.roi_mask,
        root_30m=args.root_30m,
        tag=args.tag,
        year=args.year,
        threshold_mode=args.threshold_mode,
        hotspot_percentile=args.hotspot_percentile,
        hotspot_zscore=args.hotspot_zscore,
        chronic_frequency_threshold=args.chronic_frequency_threshold,
        min_valid_observations=args.min_valid_observations,
        min_cooling_observations=args.min_cooling_observations,
        min_region_pixels=args.min_region_pixels,
        connectivity=args.connectivity,
        sample_png_every=args.sample_png_every,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    outputs = run_pipeline(config)
    print(f"actionable insights written to: {outputs['root']}")


if __name__ == "__main__":
    main()
