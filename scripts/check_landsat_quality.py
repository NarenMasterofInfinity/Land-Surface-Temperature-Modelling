from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import zarr

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"

LST_MIN_C = 10.0
LST_MAX_C = 70.0
DATE_VALID_FRAC_MIN = 0.15
DATE_MED_MIN_C = 10.0
DATE_MED_MAX_C = 60.0


def _get_landsat_scale_offset(root):
    try:
        g = root["labels_30m"]["landsat"]
        attrs = dict(g.attrs)
        scale = attrs.get("scale_factor", attrs.get("scale", 1.0))
        offset = attrs.get("add_offset", attrs.get("offset", 0.0))
        if scale is None:
            scale = 1.0
        if offset is None:
            offset = 0.0
        return float(scale), float(offset)
    except Exception:
        return 1.0, 0.0


def _iter_chunks(shape, chunks):
    H, W = shape
    ch_y, ch_x = chunks
    for y0 in range(0, H, ch_y):
        y1 = min(H, y0 + ch_y)
        for x0 in range(0, W, ch_x):
            x1 = min(W, x0 + ch_x)
            yield slice(y0, y1), slice(x0, x1)


def _landsat_to_celsius(arr, scale, offset):
    arr = arr.astype(np.float32, copy=False)
    arr = np.where(arr == 149, np.nan, arr)
    if scale != 1.0 or offset != 0.0:
        arr = arr * scale + offset
    if np.isfinite(arr).any() and np.nanmedian(arr) > 200:
        arr = arr - 273.15
    return arr


def _apply_range_mask(arr):
    valid = np.isfinite(arr) & (arr >= LST_MIN_C) & (arr <= LST_MAX_C)
    out = np.where(valid, arr, np.nan)
    return out, valid


def landsat_date_stats(root, t_idx, scale, offset):
    arr2d = root["labels_30m"]["landsat"]["data"][t_idx, 0, :, :]
    shape = arr2d.shape
    if hasattr(arr2d, "chunks") and arr2d.chunks is not None:
        chunks = arr2d.chunks
        if len(chunks) >= 2:
            chunks = (chunks[-2], chunks[-1])
        else:
            chunks = shape
    else:
        chunks = (min(256, shape[0]), min(256, shape[1]))

    vals = []
    n_total = 0
    n_valid = 0
    for ys, xs in _iter_chunks(shape, chunks):
        block = np.asarray(arr2d[ys, xs])
        block = _landsat_to_celsius(block, scale, offset)
        n_total += int(np.isfinite(block).sum())
        v = block[np.isfinite(block)]
        if v.size:
            vals.append(v.astype(np.float32, copy=False))
        block_filt, _ = _apply_range_mask(block)
        n_valid += int(np.isfinite(block_filt).sum())

    if vals:
        all_vals = np.concatenate(vals, axis=0)
        p1, p5, p95, p99 = np.percentile(all_vals, [1, 5, 95, 99])
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        median_all = float(np.median(all_vals))
    else:
        vmin = vmax = p1 = p5 = p95 = p99 = median_all = float("nan")

    valid_fraction = (n_valid / n_total) if n_total > 0 else 0.0
    if vals and n_valid > 0:
        all_vals = np.concatenate(vals, axis=0)
        all_vals = all_vals[(all_vals >= LST_MIN_C) & (all_vals <= LST_MAX_C)]
        median_valid = float(np.median(all_vals)) if all_vals.size else float("nan")
    else:
        median_valid = float("nan")

    removed = n_total - n_valid
    return {
        "t": int(t_idx),
        "min": vmin,
        "p1": float(p1),
        "p5": float(p5),
        "p95": float(p95),
        "p99": float(p99),
        "max": vmax,
        "median": median_all,
        "median_valid": median_valid,
        "valid_fraction": float(valid_fraction),
        "n_total": int(n_total),
        "n_valid": int(n_valid),
        "n_removed": int(removed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "metrics" / "landsat_quality.csv"))
    args = ap.parse_args()

    root = zarr.open_group(str(ROOT_30M), mode="r")
    scale, offset = _get_landsat_scale_offset(root)

    daily_raw = root["time"]["daily"][:]
    daily_str = np.array([
        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in daily_raw
    ])
    daily_times = pd.to_datetime(daily_str, format="%Y_%m_%d", errors="coerce").dropna()

    rows = []
    for t in range(len(daily_times)):
        stats = landsat_date_stats(root, t, scale, offset)
        stats["date"] = str(pd.Timestamp(daily_times[int(t)]).date())
        stats["drop_flag"] = bool(
            stats["valid_fraction"] < DATE_VALID_FRAC_MIN
            or not np.isfinite(stats["median_valid"])
            or stats["median_valid"] < DATE_MED_MIN_C
            or stats["median_valid"] > DATE_MED_MAX_C
        )
        rows.append(stats)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
