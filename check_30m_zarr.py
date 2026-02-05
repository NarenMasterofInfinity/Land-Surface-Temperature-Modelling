from pathlib import Path

import numpy as np
import pandas as pd
import zarr

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"
COMMON_DATES = PROJECT_ROOT / "common_dates.csv"

LANDSAT_NODATA = 149
LANDSAT_MIN_VALID_K = 273.0
GENERIC_NODATA = -9999
MODIS_C_TO_K = 273.15
VIIRS_LST_NODATA = 0.0


def _to_str(arr):
    return np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
    )


def _init_stats():
    return {
        "count": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "min": None,
        "max": None,
        "nan": 0,
        "inf": 0,
    }


def _update_stats(stats, arr):
    arr = np.asarray(arr, dtype=np.float64)
    stats["nan"] += int(np.isnan(arr).sum())
    stats["inf"] += int(np.isinf(arr).sum())

    finite = np.isfinite(arr)
    if not finite.any():
        return
    vals = arr[finite]
    stats["count"] += int(vals.size)
    stats["sum"] += float(vals.sum())
    stats["sumsq"] += float((vals * vals).sum())
    vmin = float(vals.min())
    vmax = float(vals.max())
    if stats["min"] is None or vmin < stats["min"]:
        stats["min"] = vmin
    if stats["max"] is None or vmax > stats["max"]:
        stats["max"] = vmax


def _finalize_stats(stats):
    if stats["count"] == 0:
        return {
            "count": 0,
            "nan": stats["nan"],
            "inf": stats["inf"],
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    mean = stats["sum"] / stats["count"]
    var = stats["sumsq"] / stats["count"] - mean * mean
    std = float(np.sqrt(max(var, 0.0)))
    return {
        "count": stats["count"],
        "nan": stats["nan"],
        "inf": stats["inf"],
        "min": stats["min"],
        "max": stats["max"],
        "mean": mean,
        "std": std,
    }


def _print_stats(name, stats):
    print(
        f"{name}: count={stats['count']} nan={stats['nan']} inf={stats['inf']} "
        f"min={stats['min']} max={stats['max']} mean={stats['mean']} std={stats['std']}"
    )


def _apply_landsat_nodata(arr):
    arr = np.where(arr == LANDSAT_NODATA, np.nan, arr)
    return np.where(arr < LANDSAT_MIN_VALID_K, np.nan, arr)


def _apply_generic_nodata(arr):
    return np.where(arr == GENERIC_NODATA, np.nan, arr)


def _apply_modis_lst(arr):
    arr = _apply_generic_nodata(arr)
    return arr + MODIS_C_TO_K


def _apply_viirs_lst(arr):
    arr = _apply_generic_nodata(arr)
    return np.where(arr == VIIRS_LST_NODATA, np.nan, arr)


def main():
    root_30m = zarr.open_group(str(ROOT_30M), mode="r")
    root_daily = zarr.open_group(str(ROOT_DAILY), mode="r")

    common_df = pd.read_csv(COMMON_DATES)
    common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna()
    common_dates = pd.DatetimeIndex(common_dates).sort_values()

    daily_raw = _to_str(root_30m["time"]["daily"][:])
    monthly_raw = _to_str(root_30m["time"]["monthly"][:])

    daily_times = pd.to_datetime(daily_raw, format="%Y_%m_%d", errors="coerce").dropna()
    monthly_times = pd.to_datetime(monthly_raw, format="%Y_%m", errors="coerce").dropna()

    daily_idx = np.flatnonzero(daily_times.isin(common_dates))
    month_index = pd.DatetimeIndex(common_dates.to_period("M").to_timestamp())
    monthly_idx = np.flatnonzero(monthly_times.isin(month_index))

    print(f"daily_idx={len(daily_idx)} monthly_idx={len(monthly_idx)}")

    sources = {
        "daily/landsat_30m": (root_30m["labels_30m"]["landsat"]["data"], daily_idx),
        "daily/era5_30m": (root_30m["products_30m"]["era5"]["data"], daily_idx),
        "monthly/sentinel1": (root_30m["products_30m"]["sentinel1"]["data"], monthly_idx),
        "monthly/sentinel2": (root_30m["products_30m"]["sentinel2"]["data"], monthly_idx),
    }

    for name, (arr, idxs) in sources.items():
        stats = _init_stats()
        for i in idxs:
            slab = arr[int(i)]
            if "landsat" in name and slab.ndim >= 3:
                slab = slab[:1]
                slab = _apply_landsat_nodata(slab)
            slab = _apply_generic_nodata(slab)
            _update_stats(stats, slab)
        _print_stats(name, _finalize_stats(stats))

    modis_arr = root_daily["products"]["modis"]["data"]
    viirs_arr = root_daily["products"]["viirs"]["data"]

    modis_lst = _init_stats()
    modis_cloud = _init_stats()
    for i in daily_idx:
        slab = modis_arr[int(i)]
        if slab.shape[0] >= 2:
            lst = _apply_modis_lst(slab[:2])
            _update_stats(modis_lst, lst)
        if slab.shape[0] >= 6:
            cloud = _apply_generic_nodata(slab[4:6])
            _update_stats(modis_cloud, cloud)
    _print_stats("daily/modis_lst", _finalize_stats(modis_lst))
    _print_stats("daily/modis_cloud", _finalize_stats(modis_cloud))

    viirs_lst = _init_stats()
    viirs_cloud = _init_stats()
    for i in daily_idx:
        slab = viirs_arr[int(i)]
        if slab.shape[0] >= 2:
            lst = _apply_viirs_lst(slab[:2])
            _update_stats(viirs_lst, lst)
        if slab.shape[0] >= 4:
            cloud = _apply_generic_nodata(slab[2:4])
            _update_stats(viirs_cloud, cloud)
    _print_stats("daily/viirs_lst", _finalize_stats(viirs_lst))
    _print_stats("daily/viirs_cloud", _finalize_stats(viirs_cloud))


if __name__ == "__main__":
    main()
