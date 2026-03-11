from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from osgeo import gdal


def _load_grid_meta(path: Path) -> Tuple[str, List[float], int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    attrs = data["attributes"]
    return str(attrs["crs"]), [float(v) for v in attrs["transform"]], int(attrs["width"]), int(attrs["height"])


def _grid_bounds(transform: List[float], width: int, height: int) -> Tuple[float, float, float, float]:
    a, _, c, _, e, f, _, _, _ = transform
    xmin = c
    ymax = f
    xmax = c + a * width
    ymin = f + e * height
    return xmin, ymin, xmax, ymax


def _warp_to_grid(src_path: Path, dst_crs: str, transform: List[float], width: int, height: int) -> np.ndarray:
    xmin, ymin, xmax, ymax = _grid_bounds(transform, width, height)
    warp_ds = gdal.Warp(
        "",
        str(src_path),
        format="MEM",
        dstSRS=dst_crs,
        outputBounds=(xmin, ymin, xmax, ymax),
        width=width,
        height=height,
        resampleAlg=gdal.GRA_NearestNeighbour,
        multithread=True,
    )
    if warp_ds is None:
        raise RuntimeError(f"Failed to warp {src_path}")
    arr = warp_ds.ReadAsArray().astype(np.float32, copy=False)
    return arr


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _density_class(frac: float) -> str:
    if not np.isfinite(frac):
        return "unknown"
    if frac >= 0.75:
        return "very_high"
    if frac >= 0.50:
        return "high"
    if frac >= 0.25:
        return "medium"
    if frac >= 0.10:
        return "low"
    return "very_low"


def export_building_density(
    run_root: Path,
    grid_meta: Path,
    worldcover_tif: Path,
    dynamic_world_tif: Path,
    out_csv: Path,
) -> Path:
    dst_crs, transform, width, height = _load_grid_meta(grid_meta)
    label_map = np.load(run_root / "step4_regions" / "region_id_map.npy")
    location_rows = _read_csv_rows(run_root / "step4_regions" / "hotspot_region_locations.csv")

    worldcover = _warp_to_grid(worldcover_tif, dst_crs, transform, width, height)
    dynamic_world = _warp_to_grid(dynamic_world_tif, dst_crs, transform, width, height)

    if worldcover.shape != label_map.shape or dynamic_world.shape != label_map.shape:
        raise RuntimeError(
            f"Shape mismatch after warp: label_map={label_map.shape}, worldcover={worldcover.shape}, dynamic_world={dynamic_world.shape}"
        )

    out_rows: List[Dict[str, object]] = []
    for row in location_rows:
        region_id = int(row["region_id"])
        mask = label_map == region_id
        if not np.any(mask):
            continue

        wc_vals = worldcover[mask]
        dw_vals = dynamic_world[mask]

        wc_valid = np.isfinite(wc_vals)
        dw_valid = np.isfinite(dw_vals)
        wc_built_frac = float(np.mean(wc_vals[wc_valid] == 50)) if np.any(wc_valid) else np.nan
        dw_built_frac = float(np.mean(dw_vals[dw_valid] == 6)) if np.any(dw_valid) else np.nan
        built_density_frac = float(np.nanmean([wc_built_frac, dw_built_frac]))
        built_pixels_consensus = int(np.sum((wc_vals == 50) | (dw_vals == 6)))

        out_rows.append(
            {
                **row,
                "worldcover_built_fraction": wc_built_frac,
                "dynamic_world_built_fraction": dw_built_frac,
                "built_density_fraction": built_density_frac,
                "built_density_percent": built_density_frac * 100.0 if np.isfinite(built_density_frac) else np.nan,
                "built_density_class": _density_class(built_density_frac),
                "built_pixels_consensus": built_pixels_consensus,
                "region_pixels_checked": int(mask.sum()),
            }
        )

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify hotspot building density from built-up land-cover rasters.")
    parser.add_argument("--run-root", required=True, help="Actionable insights run root")
    parser.add_argument("--grid-meta", default="madurai_30m.zarr/grid/zarr.json")
    parser.add_argument(
        "--worldcover-tif",
        default="/home/naren-root/Documents/FYP2/data/static_data/worldcover/WorldCover_2021_Madurai_10m.tif",
    )
    parser.add_argument(
        "--dynamic-world-tif",
        default="/home/naren-root/Documents/FYP2/data/static_data/dynamic_world/DynamicWorld_mode_2018-2025_Madurai_10m.tif",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path. Defaults to <run-root>/step4_regions/hotspot_region_building_density.csv",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else run_root / "step4_regions" / "hotspot_region_building_density.csv"
    )
    out = export_building_density(
        run_root=run_root,
        grid_meta=Path(args.grid_meta),
        worldcover_tif=Path(args.worldcover_tif),
        dynamic_world_tif=Path(args.dynamic_world_tif),
        out_csv=out_csv,
    )
    print(out)


if __name__ == "__main__":
    main()
