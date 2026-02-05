#!/usr/bin/env python3
"""
Compute validity summary across full Zarr time axes (exclude full-nan days).

Outputs:
  - stdout: markdown table rows
  - data file: metrics/validity_full_summary.csv
"""
from pathlib import Path
import csv
import numpy as np
import zarr

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"
OUT_CSV = PROJECT_ROOT / "metrics" / "validity_full_summary.csv"

LANDSAT_NODATA = 149
LANDSAT_MIN_VALID_K = 273.0
MODIS_NODATA = -9999.0
VIIRS_LST_NODATA = 0.0
MODIS_MIN_VALID_C = 0.0


def _valid_all_channels(arr):
    if arr.ndim < 3:
        return np.isfinite(arr)
    return np.isfinite(arr).all(axis=0)


def _landsat_valid(arr):
    arr = np.where(arr == LANDSAT_NODATA, np.nan, arr)
    arr = np.where(arr < LANDSAT_MIN_VALID_K, np.nan, arr)
    return np.isfinite(arr)


def _modis_valid(arr):
    if arr.shape[0] < 2:
        return _valid_all_channels(arr)
    lst = arr[0]
    if arr.shape[0] >= 6:
        mask = arr[4]
        lst = np.where(mask == 1, lst, np.nan)
    else:
        mask = arr[1]
        lst = np.where(mask == 0, lst, np.nan)
    lst = np.where(lst == MODIS_NODATA, np.nan, lst)
    lst = np.where(lst <= MODIS_MIN_VALID_C, np.nan, lst)
    return np.isfinite(lst)


def _viirs_valid(arr):
    if arr.shape[0] < 2:
        return _valid_all_channels(arr)
    lst = arr[0]
    if arr.shape[0] >= 6:
        mask = arr[4]
        lst = np.where(mask == 1, lst, np.nan)
    elif arr.shape[0] >= 4:
        mask = arr[2]
        lst = np.where(mask <= 1, lst, np.nan)
    else:
        mask = arr[1]
        lst = np.where(mask <= 1, lst, np.nan)
    lst = np.where(lst == VIIRS_LST_NODATA, np.nan, lst)
    return np.isfinite(lst)


def summarize_counts(name, counts):
    counts = [c for c in counts if c > 0]
    if not counts:
        return {
            "source": name,
            "n_used": 0,
            "max_valid_px": 0,
            "mean_valid_frac": 0.0,
            "total_valid_frac": 0.0,
        }
    max_v = max(counts)
    fracs = [c / max_v if max_v else 0.0 for c in counts]
    n = len(fracs)
    mean = sum(fracs) / n if n else 0.0
    total = (sum(counts) / (max_v * n)) if (max_v and n) else 0.0
    return {
        "source": name,
        "n_used": n,
        "max_valid_px": max_v,
        "mean_valid_frac": mean,
        "total_valid_frac": total,
    }


def main():
    root_30m = zarr.open_group(str(ROOT_30M), mode="r")
    root_daily = zarr.open_group(str(ROOT_DAILY), mode="r")

    landsat = root_30m["labels_30m"]["landsat"]["data"]
    landsat_valid = root_30m["labels_30m"]["landsat"]["valid"]
    era5_valid = root_30m["products_30m"]["era5"]["valid"]
    s1_valid = root_30m["products_30m"]["sentinel1"]["valid"]
    s2_valid = root_30m["products_30m"]["sentinel2"]["valid"]
    modis = root_daily["products"]["modis"]["data"]
    viirs = root_daily["products"]["viirs"]["data"]

    static_dem = root_30m["static_30m"]["dem"]["data"][0]
    static_world = root_30m["static_30m"]["worldcover"]["data"][0]
    static_dyn = root_30m["static_30m"]["dynamic_world"]["data"][0]

    landsat_counts = []
    for t in range(landsat.shape[0]):
        try:
            v = landsat_valid[t, 0]
            c = int(np.count_nonzero(v))
        except Exception:
            c = int(np.count_nonzero(_landsat_valid(landsat[t, 0])))
        landsat_counts.append(c)

    era5_counts = []
    for t in range(era5_valid.shape[0]):
        v = era5_valid[t, 0]
        era5_counts.append(int(np.count_nonzero(v)))

    s1_counts = []
    for t in range(s1_valid.shape[0]):
        v = s1_valid[t, 0]
        s1_counts.append(int(np.count_nonzero(v)))

    s2_counts = []
    for t in range(s2_valid.shape[0]):
        v = s2_valid[t, 0]
        s2_counts.append(int(np.count_nonzero(v)))

    modis_counts = []
    for t in range(modis.shape[0]):
        modis_counts.append(int(np.count_nonzero(_modis_valid(modis[t]))))

    viirs_counts = []
    for t in range(viirs.shape[0]):
        viirs_counts.append(int(np.count_nonzero(_viirs_valid(viirs[t]))))

    static_dem_count = int(np.count_nonzero(_valid_all_channels(static_dem)))
    static_world_count = int(np.count_nonzero(_valid_all_channels(static_world)))
    static_dyn_count = int(np.count_nonzero(_valid_all_channels(static_dyn)))

    rows = [
        summarize_counts("Landsat", landsat_counts),
        summarize_counts("ERA5", era5_counts),
        summarize_counts("Sentinel-1", s1_counts),
        summarize_counts("Sentinel-2", s2_counts),
        summarize_counts("MODIS", modis_counts),
        summarize_counts("VIIRS", viirs_counts),
        summarize_counts("DEM", [static_dem_count]),
        summarize_counts("WorldCover", [static_world_count]),
        summarize_counts("Dynamic World", [static_dyn_count]),
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["source", "n_used", "max_valid_px", "mean_valid_frac", "total_valid_frac"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Markdown table for easy paste
    print("| Source | n_used (nonzero) | max_valid_px | mean_valid_frac | total_valid_frac |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['source']} | {r['n_used']} | {r['max_valid_px']} | {r['mean_valid_frac']:.4f} | {r['total_valid_frac']:.4f} |"
        )


if __name__ == "__main__":
    main()
