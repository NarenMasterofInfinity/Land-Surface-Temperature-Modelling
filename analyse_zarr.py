#!/usr/bin/env python3
"""
Analyze a Zarr geospatial datacube:
- variables, dimensions, dtype, chunks, compression
- spatial resolution (x/y coords or geotransform attrs)
- missing data (NaN + optional nodata)
- stats (min/max/mean/std + percentiles)
- saves outputs (json/csv/markdown)

Usage:
  python analyze_zarr.py --zarr /path/to/cube.zarr --out /path/to/outdir
Optional:
  --sample 1.0        # fraction of time slices to sample (0<sample<=1), default 1.0 (full)
  --time_dim time     # name of time dimension if not 'time'
  --y_dim y           # name of y dimension if not 'y' or 'lat'
  --x_dim x           # name of x dimension if not 'x' or 'lon'
  --nodata -9999      # treat this value as missing (in addition to NaNs)
"""

import argparse, json, os, math, datetime
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError as e:
    raise SystemExit("Missing dependency: xarray. Install via: pip install xarray zarr numcodecs pandas numpy") from e

try:
    import zarr
except ImportError as e:
    raise SystemExit("Missing dependency: zarr. Install via: pip install zarr") from e


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def infer_xy_names(ds, x_dim=None, y_dim=None):
    # Prefer explicit
    if x_dim and y_dim and x_dim in ds.dims and y_dim in ds.dims:
        return x_dim, y_dim

    # Heuristics
    y_candidates = [y_dim, "y", "lat", "latitude", "northing", "row"]
    x_candidates = [x_dim, "x", "lon", "longitude", "easting", "col"]

    y_name = next((c for c in y_candidates if c and (c in ds.dims or c in ds.coords)), None)
    x_name = next((c for c in x_candidates if c and (c in ds.dims or c in ds.coords)), None)

    return x_name, y_name


def infer_time_name(ds, time_dim=None):
    if time_dim and time_dim in ds.dims:
        return time_dim
    for cand in [time_dim, "time", "t", "date", "datetime"]:
        if cand and cand in ds.dims:
            return cand
    return None


def estimate_resolution_from_coords(ds, x_name, y_name):
    """
    Returns dict with pixel size in coordinate units (often meters for projected CRS).
    """
    out = {"x_res": None, "y_res": None, "units": None, "method": None}

    if not x_name or not y_name:
        return out

    # coords may be in ds.coords even if dims are different
    if x_name in ds.coords:
        x = ds.coords[x_name].values
    elif x_name in ds.dims:
        x = np.arange(ds.sizes[x_name])
    else:
        x = None

    if y_name in ds.coords:
        y = ds.coords[y_name].values
    elif y_name in ds.dims:
        y = np.arange(ds.sizes[y_name])
    else:
        y = None

    if x is None or y is None:
        return out

    def median_step(arr):
        if arr.ndim != 1 or arr.size < 2:
            return None
        diffs = np.diff(arr.astype("float64"))
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size == 0:
            return None
        return float(np.median(np.abs(diffs)))

    x_res = median_step(x)
    y_res = median_step(y)

    units = None
    if x_name in ds.coords and hasattr(ds.coords[x_name], "attrs"):
        units = ds.coords[x_name].attrs.get("units", None)
    if units is None and y_name in ds.coords and hasattr(ds.coords[y_name], "attrs"):
        units = ds.coords[y_name].attrs.get("units", None)

    out.update({"x_res": x_res, "y_res": y_res, "units": units, "method": "coords"})
    return out


def estimate_resolution_from_attrs(ds):
    """
    Looks for GDAL-style geotransform or rasterio transform info in attrs.
    """
    out = {"x_res": None, "y_res": None, "units": None, "method": None}

    # common patterns in attrs
    a = dict(ds.attrs) if hasattr(ds, "attrs") else {}

    # rasterio-style: transform as tuple/list or string
    transform = a.get("transform") or a.get("geotransform") or a.get("GeoTransform") or a.get("geo_transform")
    if transform is not None:
        # GeoTransform: [x0, dx, rx, y0, ry, dy] dy is usually negative
        if isinstance(transform, (list, tuple)) and len(transform) >= 6:
            dx = _safe_float(transform[1])
            dy = _safe_float(transform[5])
            if dx is not None and dy is not None:
                out.update({"x_res": abs(dx), "y_res": abs(dy), "units": None, "method": "attrs:geotransform"})
                return out
        # if string, try parse numbers
        if isinstance(transform, str):
            nums = []
            for tok in transform.replace(",", " ").replace("[", " ").replace("]", " ").split():
                v = _safe_float(tok)
                if v is not None:
                    nums.append(v)
            if len(nums) >= 6:
                out.update({"x_res": abs(nums[1]), "y_res": abs(nums[5]), "units": None, "method": "attrs:geotransform"})
                return out

    return out


def zarr_store_metadata(zarr_path: str):
    """
    Pull variable chunk/compressor metadata directly from the zarr store.
    Recurses into subgroups so products/* arrays are captured.
    """
    g = zarr.open_group(zarr_path, mode="r")
    meta = {}

    def walk(group, prefix=""):
        for k, v in group.arrays():
            key = f"{prefix}{k}" if prefix else k
            try:
                compressors = v.compressors
            except Exception:
                compressors = None
            meta[key] = {
                "zarr_shape": list(v.shape),
                "zarr_chunks": list(v.chunks),
                "zarr_dtype": str(v.dtype),
                "zarr_compressor": repr(compressors),
                "zarr_filters": repr(v.filters),
                "zarr_order": v.order,
            }
        for k, sub in group.groups():
            sub_prefix = f"{prefix}{k}/" if prefix else f"{k}/"
            walk(sub, sub_prefix)

    walk(g)
    return meta


def compute_stats_da(da, nodata=None, percentiles=(1, 5, 25, 50, 75, 95, 99), sample_time_slices=None, time_name=None):
    """
    Computes stats without loading the entire cube aggressively.
    Uses chunked computation if dask is available; otherwise falls back to numpy for selected slices.
    """
    # Decide slicing for sampling
    if sample_time_slices is not None and time_name and time_name in da.dims:
        da_work = da.isel({time_name: sample_time_slices})
    else:
        da_work = da

    # Mask nodata if provided
    if nodata is not None:
        da_work = da_work.where(da_work != nodata)

    # Use xarray reductions; will use dask if available
    finite = np.isfinite(da_work)
    count_total = int(np.prod([da_work.sizes[d] for d in da_work.dims]))
    # Note: if da is lazy, this triggers compute; that's intended.
    count_finite = int(finite.sum().compute() if hasattr(finite.sum(), "compute") else finite.sum().values)
    count_missing = count_total - count_finite

    # For stats, compute with masking
    da_f = da_work.where(np.isfinite(da_work))
    reducers = {
        "min": da_f.min(),
        "max": da_f.max(),
        "mean": da_f.mean(),
        "std": da_f.std(),
    }

    out = {}
    for k, val in reducers.items():
        v = val.compute() if hasattr(val, "compute") else val.values
        out[k] = float(v) if np.isfinite(v) else None

    # Percentiles: xarray quantile is dask-aware
    try:
        q = da_f.quantile(np.array(percentiles) / 100.0, skipna=True)
        qv = q.compute() if hasattr(q, "compute") else q.values
        out["percentiles"] = {f"p{p}": (float(qv[i]) if np.isfinite(qv[i]) else None) for i, p in enumerate(percentiles)}
    except Exception:
        # fallback: numpy on flattened sample (dangerous for huge arrays; but we already sampled time if configured)
        arr = da_f.values
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out["percentiles"] = {f"p{p}": float(np.percentile(arr, p)) for p in percentiles}
        else:
            out["percentiles"] = {f"p{p}": None for p in percentiles}

    out["count_total"] = count_total
    out["count_missing"] = count_missing
    out["missing_fraction"] = float(count_missing / count_total) if count_total else None

    return out


def _iter_chunk_slices(shape, chunks):
    ranges = [range(0, shape[d], chunks[d]) for d in range(len(shape))]
    for idxs in itertools.product(*ranges):
        slices = tuple(slice(i, min(i + chunks[d], shape[d])) for d, i in enumerate(idxs))
        yield slices


def _chunk_slice_from_index(index, grid_shape, chunks, shape):
    idxs = np.unravel_index(index, grid_shape)
    return tuple(
        slice(i * chunks[d], min((i + 1) * chunks[d], shape[d]))
        for d, i in enumerate(idxs)
    )


def compute_stats_zarr_array(arr, nodata=None, percentiles=(1, 5, 25, 50, 75, 95, 99),
                             sample_time_slices=None, sample_fraction=1.0, max_samples=200000,
                             max_chunks=200, max_elements=50000000):
    """
    Computes stats for a zarr array by iterating over chunks or sampled time slices.
    Uses approximate percentiles based on a bounded random sample.
    """
    sample_note = None
    if sample_time_slices is None and sample_fraction < 1.0 and arr.ndim >= 1:
        T = arr.shape[0]
        k = max(1, int(math.ceil(T * sample_fraction)))
        sample_time_slices = np.linspace(0, T - 1, k).astype(int).tolist()
        sample_note = "sampled_time_slices"

    count_total = 0
    count_missing = 0
    n = 0
    mean = 0.0
    M2 = 0.0
    min_v = None
    max_v = None
    samples = []
    rng = np.random.default_rng(0)

    def update_with_vals(vals):
        nonlocal n, mean, M2, min_v, max_v, samples
        if vals.size == 0:
            return
        vmin = float(vals.min())
        vmax = float(vals.max())
        min_v = vmin if min_v is None else min(min_v, vmin)
        max_v = vmax if max_v is None else max(max_v, vmax)

        n_chunk = int(vals.size)
        mean_chunk = float(vals.mean())
        M2_chunk = float(((vals - mean_chunk) ** 2).sum())
        if n == 0:
            n = n_chunk
            mean = mean_chunk
            M2 = M2_chunk
        else:
            delta = mean_chunk - mean
            total = n + n_chunk
            mean = mean + delta * n_chunk / total
            M2 = M2 + M2_chunk + (delta ** 2) * n * n_chunk / total
            n = total

        if len(samples) < max_samples:
            remaining = max_samples - len(samples)
            k = min(remaining, min(10000, vals.size))
            if vals.size <= k:
                samples.extend(vals.tolist())
            else:
                idx = rng.choice(vals.size, size=k, replace=False)
                samples.extend(vals[idx].tolist())

    if sample_time_slices is not None and arr.ndim >= 1:
        for i in sample_time_slices:
            if i < 0 or i >= arr.shape[0]:
                continue
            data = arr[(i,) + (slice(None),) * (arr.ndim - 1)]
            count_total += data.size
            if nodata is not None:
                mask = np.isfinite(data) & (data != nodata)
            else:
                mask = np.isfinite(data)
            count_missing += int(data.size - mask.sum())
            update_with_vals(data[mask].astype("float64", copy=False))
            if max_elements and count_total >= max_elements:
                sample_note = "sampled_time_slices_max_elements"
                break
    else:
        chunks = arr.chunks if arr.chunks else arr.shape
        grid_shape = [int(math.ceil(arr.shape[d] / chunks[d])) for d in range(arr.ndim)]
        total_chunks = int(np.prod(grid_shape)) if grid_shape else 0
        if max_chunks and total_chunks > max_chunks:
            sample_note = "sampled_chunks"
            chunk_indices = rng.choice(total_chunks, size=max_chunks, replace=False)
            iter_slices = (_chunk_slice_from_index(i, grid_shape, chunks, arr.shape) for i in chunk_indices)
        else:
            iter_slices = _iter_chunk_slices(arr.shape, chunks)
        for slices in iter_slices:
            data = arr[slices]
            count_total += data.size
            if nodata is not None:
                mask = np.isfinite(data) & (data != nodata)
            else:
                mask = np.isfinite(data)
            count_missing += int(data.size - mask.sum())
            update_with_vals(data[mask].astype("float64", copy=False))
            if max_elements and count_total >= max_elements:
                if sample_note is None:
                    sample_note = "max_elements"
                else:
                    sample_note = f"{sample_note}_max_elements"
                break

    out = {
        "min": float(min_v) if min_v is not None else None,
        "max": float(max_v) if max_v is not None else None,
        "mean": float(mean) if n else None,
        "std": float(math.sqrt(M2 / n)) if n else None,
        "count_total": int(count_total),
        "count_missing": int(count_missing),
        "missing_fraction": float(count_missing / count_total) if count_total else None,
    }

    if samples:
        qv = np.percentile(np.asarray(samples, dtype="float64"), percentiles)
        out["percentiles"] = {f"p{p}": float(qv[i]) for i, p in enumerate(percentiles)}
        out["percentiles_note"] = "approx_from_samples"
    else:
        out["percentiles"] = {f"p{p}": None for p in percentiles}
        out["percentiles_note"] = "no_finite_values"
    if sample_note:
        out["sample_note"] = sample_note

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True, help="Path to .zarr directory")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--sample", type=float, default=1.0, help="Fraction of time slices to sample (0<sample<=1)")
    ap.add_argument("--max_chunks", type=int, default=200, help="Max zarr chunks to scan for stats (0 for no limit)")
    ap.add_argument("--max_elements"    , type=int, default=50000000, help="Max elements to scan for stats (0 for no limit)")
    ap.add_argument("--time_dim", default=None, help="Time dimension name (optional)")
    ap.add_argument("--x_dim", default=None, help="X dimension/coord name (optional)")
    ap.add_argument("--y_dim", default=None, help="Y dimension/coord name (optional)")
    ap.add_argument("--nodata", default=None, help="Nodata value to treat as missing (optional)")
    args = ap.parse_args()

    zarr_path = args.zarr
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    nodata = _safe_float(args.nodata) if args.nodata is not None else None

    # Open dataset with xarray
    ds = xr.open_zarr(zarr_path, consolidated=False)

    # Infer axes
    time_name = infer_time_name(ds, args.time_dim)
    x_name, y_name = infer_xy_names(ds, args.x_dim, args.y_dim)

    # Resolution estimation
    res_coords = estimate_resolution_from_coords(ds, x_name, y_name)
    res_attrs = estimate_resolution_from_attrs(ds)

    # Prefer coords if present, else attrs
    resolution = res_coords if (res_coords["x_res"] and res_coords["y_res"]) else res_attrs

    # Sample time indices if asked
    sample_slices = None
    if args.sample < 1.0 and time_name and time_name in ds.dims:
        T = ds.sizes[time_name]
        k = max(1, int(math.ceil(T * args.sample)))
        idx = np.linspace(0, T - 1, k).astype(int)
        sample_slices = idx.tolist()

    # Zarr low-level metadata (chunks/compression)
    zmeta = zarr_store_metadata(zarr_path)
    data_vars = list(ds.data_vars)
    if not data_vars:
        data_vars = list(zmeta.keys())

    variables_rows = []
    summary = {
        "zarr_path": str(zarr_path),
        "generated_at": datetime.datetime.now().isoformat(),
        "dims": dict(ds.sizes),
        "coords": list(ds.coords),
        "data_vars": data_vars,
        "attrs": dict(ds.attrs),
        "inferred_axes": {"time": time_name, "x": x_name, "y": y_name},
        "resolution_estimate": resolution,
        "nodata_used": nodata,
        "sample_time_fraction": args.sample,
        "sample_time_indices": sample_slices,
        "variables": {},
    }

    # Analyze each variable
    if ds.data_vars:
        for var in ds.data_vars:
            da = ds[var]
            info = {
                "dims": list(da.dims),
                "shape": [da.sizes[d] for d in da.dims],
                "dtype": str(da.dtype),
                "attrs": dict(da.attrs),
            }

            # Attach zarr array metadata when possible
            if var in zmeta:
                info.update(zmeta[var])

            # Only compute stats for numeric arrays
            is_numeric = np.issubdtype(da.dtype, np.number)
            if is_numeric:
                try:
                    stats = compute_stats_da(
                        da,
                        nodata=nodata,
                        sample_time_slices=sample_slices,
                        time_name=time_name
                    )
                except Exception as e:
                    stats = {"error": str(e)}
            else:
                stats = {"note": "non-numeric; stats skipped"}

            info["stats"] = stats
            summary["variables"][var] = info

            # Flat row for CSV
            row = {
                "var": var,
                "dtype": info.get("dtype"),
                "dims": ",".join(info.get("dims", [])),
                "shape": "x".join(map(str, info.get("shape", []))),
                "chunks": str(info.get("zarr_chunks", None)),
                "compressor": info.get("zarr_compressor", None),
                "missing_fraction": stats.get("missing_fraction", None) if isinstance(stats, dict) else None,
                "min": stats.get("min", None) if isinstance(stats, dict) else None,
                "max": stats.get("max", None) if isinstance(stats, dict) else None,
                "mean": stats.get("mean", None) if isinstance(stats, dict) else None,
                "std": stats.get("std", None) if isinstance(stats, dict) else None,
            }
            variables_rows.append(row)

        # Include zarr arrays not represented as xarray data_vars
        for var, meta in zmeta.items():
            if var in summary["variables"]:
                continue
            info = {
                "dims": [],
                "shape": meta.get("zarr_shape", []),
                "dtype": meta.get("zarr_dtype"),
                "attrs": {},
            }
            info.update(meta)
            try:
                dtype_obj = np.dtype(info.get("dtype"))
            except Exception:
                dtype_obj = None
            if dtype_obj is not None and np.issubdtype(dtype_obj, np.number):
                try:
                    stats = compute_stats_zarr_array(
                        zarr.open_array(os.path.join(zarr_path, var), mode="r"),
                        nodata=nodata,
                        sample_time_slices=sample_slices,
                        sample_fraction=args.sample,
                        max_chunks=(args.max_chunks if args.max_chunks > 0 else None),
                        max_elements=(args.max_elements if args.max_elements > 0 else None)
                    )
                except Exception as e:
                    stats = {"error": str(e)}
            else:
                stats = {"note": "non-numeric; stats skipped"}
            info["stats"] = stats
            summary["variables"][var] = info
            variables_rows.append({
                "var": var,
                "dtype": info.get("dtype"),
                "dims": "",
                "shape": "x".join(map(str, info.get("shape", []))),
                "chunks": str(info.get("zarr_chunks", None)),
                "compressor": info.get("zarr_compressor", None),
                "missing_fraction": None,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            })
    else:
        for var, meta in zmeta.items():
            info = {
                "dims": [],
                "shape": meta.get("zarr_shape", []),
                "dtype": meta.get("zarr_dtype"),
                "attrs": {},
            }
            info.update(meta)
            try:
                dtype_obj = np.dtype(info.get("dtype"))
            except Exception:
                dtype_obj = None
            if dtype_obj is not None and np.issubdtype(dtype_obj, np.number):
                try:
                    stats = compute_stats_zarr_array(
                        zarr.open_array(os.path.join(zarr_path, var), mode="r"),
                        nodata=nodata,
                        sample_time_slices=sample_slices,
                        sample_fraction=args.sample,
                        max_chunks=(args.max_chunks if args.max_chunks > 0 else None),
                        max_elements=(args.max_elements if args.max_elements > 0 else None)
                    )
                except Exception as e:
                    stats = {"error": str(e)}
            else:
                stats = {"note": "non-numeric; stats skipped"}
            info["stats"] = stats
            summary["variables"][var] = info
            variables_rows.append({
                "var": var,
                "dtype": info.get("dtype"),
                "dims": "",
                "shape": "x".join(map(str, info.get("shape", []))),
                "chunks": str(info.get("zarr_chunks", None)),
                "compressor": info.get("zarr_compressor", None),
                "missing_fraction": None,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            })

    # Save JSON
    json_path = outdir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save CSV
    df = pd.DataFrame(variables_rows)
    if "var" in df.columns:
        df = df.sort_values("var")
    else:
        df = pd.DataFrame(columns=["var", "dtype", "dims", "shape", "chunks", "compressor",
                                   "missing_fraction", "min", "max", "mean", "std"])
    csv_path = outdir / "variables.csv"
    df.to_csv(csv_path, index=False)

    # Save Markdown report
    md_path = outdir / "report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Zarr Dataset Report\n\n")
        f.write(f"- Path: `{zarr_path}`\n")
        f.write(f"- Generated: `{summary['generated_at']}`\n")
        f.write(f"- Dims: `{summary['dims']}`\n")
        f.write(f"- Inferred axes: `{summary['inferred_axes']}`\n")
        f.write(f"- Resolution estimate: `{summary['resolution_estimate']}`\n")
        f.write(f"- Nodata used: `{nodata}`\n")
        f.write(f"- Time sampling: `{args.sample}` (indices: `{sample_slices}`)\n\n")

        f.write("## Variables\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")

        # Add percentile snippets
        f.write("## Percentiles (per variable)\n\n")
        for var in summary["variables"]:
            stats = summary["variables"][var].get("stats", {})
            if isinstance(stats, dict) and "percentiles" in stats:
                f.write(f"### {var}\n\n")
                f.write(f"- missing_fraction: {stats.get('missing_fraction')}\n")
                f.write(f"- percentiles: {stats.get('percentiles')}\n\n")

    print(f"Saved:\n- {json_path}\n- {csv_path}\n- {md_path}")


if __name__ == "__main__":
    main()
