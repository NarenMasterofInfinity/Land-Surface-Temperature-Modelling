from pathlib import Path

import numpy as np
import pandas as pd
import zarr

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
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


def _apply_source_rules(path, arr):
    arr = _apply_generic_nodata(arr)
    if "landsat" in path:
        arr = _apply_landsat_nodata(arr)
    if "modis" in path:
        arr = arr + MODIS_C_TO_K
    return arr


def _iter_arrays(group, prefix=""):
    items = None
    if hasattr(group, "items"):
        items = group.items()
    elif hasattr(group, "members"):
        items = group.members()

    if items is not None:
        for name, item in items:
            path = f"{prefix}{name}" if prefix else name
            if isinstance(item, zarr.Array):
                yield path, item
            elif isinstance(item, zarr.Group):
                yield from _iter_arrays(item, f"{path}/")
        return

    if hasattr(group, "array_keys") and hasattr(group, "group_keys"):
        for name in group.array_keys():
            path = f"{prefix}{name}" if prefix else name
            yield path, group[name]
        for name in group.group_keys():
            child = group[name]
            path = f"{prefix}{name}" if prefix else name
            yield from _iter_arrays(child, f"{path}/")


def main():
    root = zarr.open_group(str(ROOT_DAILY), mode="r")

    common_df = pd.read_csv(COMMON_DATES)
    common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna()
    common_dates = pd.DatetimeIndex(common_dates).sort_values()

    daily_raw = _to_str(root["time"]["daily"][:])
    daily_times = pd.to_datetime(daily_raw, format="%Y_%m_%d", errors="coerce").dropna()
    daily_idx = np.flatnonzero(daily_times.isin(common_dates))

    monthly_times = None
    monthly_idx = None
    if "time" in root and "monthly" in root["time"]:
        monthly_raw = _to_str(root["time"]["monthly"][:])
        monthly_times = pd.to_datetime(monthly_raw, format="%Y_%m", errors="coerce").dropna()
        month_index = pd.DatetimeIndex(common_dates.to_period("M").to_timestamp())
        monthly_idx = np.flatnonzero(monthly_times.isin(month_index))

    print(f"daily_idx={len(daily_idx)} monthly_idx={len(monthly_idx) if monthly_idx is not None else 0}")

    for path, arr in _iter_arrays(root):
        if path.startswith("time/") or "/valid" in path:
            continue
        if arr.dtype.kind not in "fiu":
            continue
        if arr.ndim == 0:
            continue

        idxs = None
        if arr.shape[0] == len(daily_times):
            idxs = daily_idx
        elif monthly_times is not None and arr.shape[0] == len(monthly_times):
            idxs = monthly_idx
        else:
            continue

        if "landsat" in path:
            stats = _init_stats()
            for i in idxs:
                slab = arr[int(i)]
                if slab.ndim >= 3:
                    slab = slab[:1]
                slab = _apply_source_rules(path, slab)
                _update_stats(stats, slab)
            _print_stats(path, _finalize_stats(stats))
            continue

        if "modis" in path:
            lst_stats = _init_stats()
            cloud_stats = _init_stats()
            for i in idxs:
                slab = arr[int(i)]
                if slab.shape[0] >= 2:
                    lst = _apply_modis_lst(slab[:2])
                    _update_stats(lst_stats, lst)
                if slab.shape[0] >= 6:
                    cloud = _apply_generic_nodata(slab[4:6])
                    _update_stats(cloud_stats, cloud)
            _print_stats(f"{path}/lst", _finalize_stats(lst_stats))
            _print_stats(f"{path}/cloud", _finalize_stats(cloud_stats))
            continue

        if "viirs" in path:
            lst_stats = _init_stats()
            cloud_stats = _init_stats()
            for i in idxs:
                slab = arr[int(i)]
                if slab.shape[0] >= 2:
                    lst = _apply_viirs_lst(slab[:2])
                    _update_stats(lst_stats, lst)
                if slab.shape[0] >= 4:
                    cloud = _apply_generic_nodata(slab[2:4])
                    _update_stats(cloud_stats, cloud)
            _print_stats(f"{path}/lst", _finalize_stats(lst_stats))
            _print_stats(f"{path}/cloud", _finalize_stats(cloud_stats))
            continue

        stats = _init_stats()
        for i in idxs:
            slab = arr[int(i)]
            slab = _apply_source_rules(path, slab)
            _update_stats(stats, slab)
        _print_stats(path, _finalize_stats(stats))


if __name__ == "__main__":
    main()
