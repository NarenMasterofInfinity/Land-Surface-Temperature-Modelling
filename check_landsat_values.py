from pathlib import Path

import numpy as np
import zarr

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"

LANDSAT_NODATA = 149
LANDSAT_MIN_VALID_K = 273.0
LANDSAT_MIN_VALID_K_STRICT = 283.0


def update_stats(stats, arr):
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
    stats["min"] = vmin if stats["min"] is None else min(stats["min"], vmin)
    stats["max"] = vmax if stats["max"] is None else max(stats["max"], vmax)


def finalize_stats(stats):
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


def main():
    root = zarr.open_group(str(ROOT_30M), mode="r")
    arr = root["labels_30m"]["landsat"]["data"]

    stats_raw = {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None, "nan": 0, "inf": 0}
    stats_clean = {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None, "nan": 0, "inf": 0}
    stats_strict = {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None, "nan": 0, "inf": 0}
    odd = {"lt0": 0, "lt273": 0, "lt283": 0, "gt400": 0, "eq149": 0}

    for t in range(arr.shape[0]):
        slab = arr[t, 0]
        odd["eq149"] += int((slab == LANDSAT_NODATA).sum())
        odd["lt0"] += int((slab < 0).sum())
        odd["lt273"] += int((slab < LANDSAT_MIN_VALID_K).sum())
        odd["lt283"] += int((slab < LANDSAT_MIN_VALID_K_STRICT).sum())
        odd["gt400"] += int((slab > 400).sum())

        update_stats(stats_raw, slab)
        cleaned = np.where(slab == LANDSAT_NODATA, np.nan, slab)
        cleaned = np.where(cleaned < LANDSAT_MIN_VALID_K, np.nan, cleaned)
        update_stats(stats_clean, cleaned)
        cleaned_strict = np.where(cleaned < LANDSAT_MIN_VALID_K_STRICT, np.nan, cleaned)
        update_stats(stats_strict, cleaned_strict)

    print("raw stats:", finalize_stats(stats_raw))
    print("clean stats:", finalize_stats(stats_clean))
    print("strict stats (<283 masked):", finalize_stats(stats_strict))
    print("odd counts:", odd)


if __name__ == "__main__":
    main()
