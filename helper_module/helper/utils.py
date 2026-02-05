"""
utils.py — Data access helpers for Madurai Zarr stores + Landsat date handling.

Supports:
- madurai.zarr (e.g., MODIS/VIIRS/Landsat monthly features)
- madurai_30m.zarr (features on 30m grid)
- madurai_alphaearth_30m.zarr (AlphaEarth features on 30m grid)
- landsat_dates.json (multiple acquisition dates per month)

Core capabilities:
- Load full dataset or subsets of variables
- Load by date range (time slicing) with flexible date parsing
- Landsat monthly -> choose one date per month (strategy-based) OR keep all dates
- Align/merge multiple zarr datasets safely
- Convert to NumPy tensors (stacked) with consistent dimension ordering
- Patch extraction (optional) for training pipelines

Assumptions:
- Zarr stores are xarray-compatible (opened via xarray.open_zarr)
- Time dimension is usually named "time" (if present); code handles alternatives.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import zarr
try:
    import dask.array as da
except Exception:  # pragma: no cover - dask optional
    da = None

# ----------------------------
# Types
# ----------------------------



ZarrKey = Literal["madurai", "madurai_30m", "madurai_alphaearth_30m"]

MonthPickStrategy = Literal[
    "first",          # earliest acquisition date in that month
    "last",           # latest acquisition date in that month
    "median",         # median acquisition date in that month
    "closest_to_mid", # closest to 15th of month
    "random_seeded",  # deterministic random based on seed
]

TimeDimCandidate = ("time", "date", "t", "datetime")


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class DataPaths:
    """Locations for zarr stores + landsat_dates.json."""
    madurai_zarr: Union[str, Path]
    madurai_30m_zarr: Union[str, Path]
    madurai_alphaearth_30m_zarr: Union[str, Path]
    landsat_dates_json: Union[str, Path]

    @staticmethod
    def from_root(root: Union[str, Path]) -> "DataPaths":
        root = Path(root)
        return DataPaths(
            madurai_zarr=root / "madurai.zarr",
            madurai_30m_zarr=root / "madurai_30m.zarr",
            madurai_alphaearth_30m_zarr=root / "madurai_alphaearth_30m.zarr",
            landsat_dates_json=root / "landsat_dates.json",
        )

# ----------------------------
# Fixed project root (DO NOT CHANGE)
# ----------------------------

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project").resolve()

DEFAULT_PATHS = DataPaths(
    madurai_zarr=PROJECT_ROOT / "madurai.zarr",
    madurai_30m_zarr=PROJECT_ROOT / "madurai_30m.zarr",
    madurai_alphaearth_30m_zarr=PROJECT_ROOT / "madurai_alphaearth_30m.zarr",
    landsat_dates_json=PROJECT_ROOT / "landsat_dates.json",
)

# ----------------------------
# Date utilities
# ----------------------------

def _to_timestamp(x: Union[str, date, datetime, np.datetime64, pd.Timestamp]) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, np.datetime64):
        return pd.Timestamp(x)
    if isinstance(x, datetime):
        return pd.Timestamp(x)
    if isinstance(x, date):
        return pd.Timestamp(datetime(x.year, x.month, x.day))
    if isinstance(x, str):
        # Accept "YYYY-MM-DD", "YYYY-MM", ISO strings
        # If YYYY-MM, interpret as first day of month
        s = x.strip()
        if len(s) == 7 and s[4] == "-":
            return pd.Timestamp(f"{s}-01")
        return pd.Timestamp(s)
    raise TypeError(f"Unsupported date type: {type(x)}")


def month_floor(ts: Union[str, date, datetime, np.datetime64, pd.Timestamp]) -> pd.Timestamp:
    t = _to_timestamp(ts)
    return pd.Timestamp(year=t.year, month=t.month, day=1)

def open_zarr_tree(path: str | Path, *, include_vars: Optional[Sequence[str]] = None) -> xr.Dataset:
    """
    Recursively open all array-containing groups in a Zarr store
    and merge them into a single xarray.Dataset.
    """
    if zarr is None:
        raise RuntimeError("zarr is required to open tree stores without xarray metadata.")

    path = str(Path(path).resolve())
    root = zarr.open_group(path, mode="r")

    datasets: List[xr.Dataset] = []

    include_map: Optional[Dict[str, Optional[set]]] = None
    if include_vars:
        include_map = {}
        for v in include_vars:
            if not v:
                continue
            if "/" not in v:
                include_map.setdefault(v, None)
                continue
            gp, bn = v.rsplit("/", 1)
            if gp not in include_map:
                include_map[gp] = set()
            if include_map[gp] is not None:
                include_map[gp].add(bn)

    def _parse_label_time(labels: List[str]) -> Optional[np.ndarray]:
        if not labels:
            return None
        if all(re.fullmatch(r"\d{4}_\d{2}_\d{2}", s) for s in labels):
            return pd.to_datetime(labels, format="%Y_%m_%d")
        if all(re.fullmatch(r"\d{4}_\d{2}", s) for s in labels):
            return pd.to_datetime(labels, format="%Y_%m")
        if all(re.fullmatch(r"\d{4}", s) for s in labels):
            return pd.to_datetime(labels, format="%Y")
        try:
            parsed = pd.to_datetime(labels, errors="coerce")
            if not np.all(pd.isna(parsed)):
                return parsed
        except Exception:
            pass
        return None

    def _maybe_parse_time(values: np.ndarray) -> np.ndarray:
        decoded = [_decode_name(v) for v in values.tolist()]
        parsed = _parse_label_time(decoded)
        if parsed is not None:
            return parsed
        try:
            parsed_any = pd.to_datetime(decoded, errors="coerce")
            if not np.all(pd.isna(parsed_any)):
                return parsed_any
        except Exception:
            pass
        return np.array(decoded)

    def _time_coords() -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        if "time" not in root:
            return out
        tgroup = root["time"]
        for key in ("monthly", "daily", "annual"):
            if key in tgroup:
                vals = tgroup[key][:]
                vals = _maybe_parse_time(vals)
                out[len(vals)] = vals
        return out

    time_by_len = _time_coords()
    default_time = None
    try:
        date_start = root.attrs.get("date_start")
        if date_start:
            default_time = _to_timestamp(str(date_start))
    except Exception:
        default_time = None
    if default_time is None:
        default_time = pd.Timestamp("1970-01-01")

    def _lazy_array(zarr_array):
        if da is None:
            raise RuntimeError(
                "open_zarr_tree requires dask to avoid loading full arrays into memory. "
                "Install with: pip install 'dask[array]'"
            )
        chunks = getattr(zarr_array, "chunks", None) or getattr(zarr_array, "chunksize", None)
        try:
            return da.from_array(zarr_array, chunks=chunks or "auto")
        except Exception:
            return da.from_array(zarr_array, chunks="auto")

    def _coords_for_shape(
        shape: Tuple[int, ...],
        band_names: Optional[List[str]],
        time_vals: Optional[np.ndarray],
    ) -> Tuple[List[str], Dict[str, Any]]:
        coords: Dict[str, Any] = {}
        if len(shape) == 4:
            dims = ["time", "band", "y", "x"]
            coords["time"] = time_vals if time_vals is not None else time_by_len.get(shape[0], np.arange(shape[0]))
            coords["band"] = band_names if band_names else np.arange(shape[1])
            coords["y"] = np.arange(shape[2])
            coords["x"] = np.arange(shape[3])
            return dims, coords
        if len(shape) == 3:
            if band_names and len(band_names) == shape[0]:
                dims = ["band", "y", "x"]
                coords["band"] = band_names
                coords["y"] = np.arange(shape[1])
                coords["x"] = np.arange(shape[2])
                return dims, coords
            dims = ["time", "y", "x"]
            coords["time"] = time_vals if time_vals is not None else time_by_len.get(shape[0], np.arange(shape[0]))
            coords["y"] = np.arange(shape[1])
            coords["x"] = np.arange(shape[2])
            return dims, coords
        if len(shape) == 2:
            return ["y", "x"], {"y": np.arange(shape[0]), "x": np.arange(shape[1])}
        dims = [f"dim_{i}" for i in range(len(shape))]
        coords = {d: np.arange(n) for d, n in zip(dims, shape)}
        return dims, coords

    def _add_group_vars(group_path: str, group) -> None:
        if "data" not in group:
            return
        if include_map is not None and group_path not in include_map:
            return
        data = group["data"]

        band_names: Optional[List[str]] = None
        if "band_names" in group:
            try:
                band_names = [_decode_name(n) for n in group["band_names"][:].tolist()]
            except Exception:
                band_names = None

        time_vals: Optional[np.ndarray] = None
        if "labels" in group:
            try:
                labels = [_decode_name(n) for n in group["labels"][:].tolist()]
                parsed = _parse_label_time(labels)
                if parsed is not None:
                    time_vals = parsed
                elif len(labels) == 1:
                    time_vals = np.array([default_time])
                else:
                    time_vals = np.array(labels)
            except Exception:
                time_vals = None

        dims, coords = _coords_for_shape(tuple(data.shape), band_names, time_vals)
        data_arr = _lazy_array(data)
        da_xr = xr.DataArray(data_arr, dims=dims, coords=coords)

        if band_names and "band" in da_xr.dims:
            if include_map is None or include_map.get(group_path) is None:
                wanted = None
            else:
                wanted = include_map.get(group_path, set())
            for i, name in enumerate(band_names):
                if wanted is not None and name not in wanted:
                    continue
                var_name = f"{group_path}/{name}"
                datasets.append(da_xr.isel(band=i).rename(var_name).to_dataset())
        else:
            if band_names is None and "band" in da_xr.dims:
                wanted = None
                if include_map is not None and include_map.get(group_path) is not None:
                    wanted = include_map.get(group_path, set())
                for i in range(int(data.shape[1])):
                    var_name = f"{group_path}/band_{i}"
                    if wanted is not None and f"band_{i}" not in wanted:
                        continue
                    datasets.append(da_xr.isel(band=i).rename(var_name).to_dataset())
                return
            datasets.append(da_xr.rename(group_path).to_dataset())

    def _walk(group, prefix: str = "") -> None:
        for name, sub in group.groups():
            group_path = f"{prefix}{name}"
            _add_group_vars(group_path, sub)
            _walk(sub, group_path + "/")

    _walk(root)

    if not datasets:
        raise RuntimeError("No arrays found in Zarr store")

    return xr.merge(datasets, compat="override", join="outer")


def month_ceil_exclusive(ts: Union[str, date, datetime, np.datetime64, pd.Timestamp]) -> pd.Timestamp:
    """
    Exclusive upper bound for a month window.
    Example: for 2020-02-XX returns 2020-03-01.
    """
    t = month_floor(ts)
    if t.month == 12:
        return pd.Timestamp(year=t.year + 1, month=1, day=1)
    return pd.Timestamp(year=t.year, month=t.month + 1, day=1)


def parse_date_range(
    start: Optional[Union[str, date, datetime, np.datetime64, pd.Timestamp]],
    end: Optional[Union[str, date, datetime, np.datetime64, pd.Timestamp]],
    *,
    inclusive: Literal["both", "left", "right", "neither"] = "both",
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], str]:
    """
    Returns (start_ts, end_ts, inclusive_str) suitable for xarray.sel(time=slice(...))
    where xarray uses inclusive boundaries in slice, but we provide clarity.
    """
    s = _to_timestamp(start) if start is not None else None
    e = _to_timestamp(end) if end is not None else None
    # xarray slice is inclusive on both ends for label-based selection;
    # keep inclusive flag for downstream custom filtering if needed.
    return s, e, inclusive


# ----------------------------
# Landsat dates handling
# ----------------------------

def load_landsat_dates(path: Union[str, Path]) -> Dict[str, List[pd.Timestamp]]:
    """
    landsat_dates.json should map a month key to multiple acquisition dates.
    Accepted keys:
    - "YYYY-MM" or "YYYY-MM-01" style.
    Values can be:
    - list of strings (ISO datetimes/dates)
    - list of dicts containing {"date": "..."} (we will try best-effort)
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    out: Dict[str, List[pd.Timestamp]] = {}
    for k, v in obj.items():
        mk = k.strip()
        mk = mk[:7] if len(mk) >= 7 else mk  # normalize to YYYY-MM
        dates: List[pd.Timestamp] = []
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    dates.append(_to_timestamp(item))
                elif isinstance(item, dict):
                    # common fields: date, datetime, acq_time, acquisitionDate
                    for field in ("date", "datetime", "acq_time", "acquisitionDate", "acquisition_date"):
                        if field in item:
                            dates.append(_to_timestamp(item[field]))
                            break
                else:
                    # ignore unknown
                    continue
        elif isinstance(v, str):
            dates.append(_to_timestamp(v))
        else:
            continue

        dates = sorted(list({d.normalize() if d.hour == 0 and d.minute == 0 else d for d in dates}))
        if dates:
            out[mk] = dates
    return out


def pick_landsat_date_for_month(
    dates: Sequence[pd.Timestamp],
    strategy: MonthPickStrategy,
    *,
    seed: int = 42,
) -> pd.Timestamp:
    if not dates:
        raise ValueError("Empty dates list for month.")
    ds = sorted(dates)

    if strategy == "first":
        return ds[0]
    if strategy == "last":
        return ds[-1]
    if strategy == "median":
        return ds[len(ds) // 2]
    if strategy == "closest_to_mid":
        # target ~ 15th
        mid = pd.Timestamp(year=ds[0].year, month=ds[0].month, day=15)
        return min(ds, key=lambda x: abs((x - mid).days))
    if strategy == "random_seeded":
        rng = np.random.default_rng(seed + ds[0].year * 100 + ds[0].month)
        return ds[int(rng.integers(0, len(ds)))]
    raise ValueError(f"Unknown strategy: {strategy}")


def months_in_range(
    start: Union[str, date, datetime, np.datetime64, pd.Timestamp],
    end: Union[str, date, datetime, np.datetime64, pd.Timestamp],
) -> List[str]:
    s = month_floor(start)
    e = month_floor(end)
    months = []
    cur = s
    while cur <= e:
        months.append(cur.strftime("%Y-%m"))
        cur = month_ceil_exclusive(cur)
    return months


# ----------------------------
# Zarr opening / indexing
# ----------------------------

def _resolve_path(p: Union[str, Path]) -> str:
    return str(Path(p).expanduser().resolve())


def open_zarr(
    path: Union[str, Path],
    *,
    consolidated: Optional[bool] = None,
    chunks: Union[str, Dict[str, int], None] = "auto",
    decode_cf: bool = True,
    mask_and_scale: bool = True,
) -> xr.Dataset:
    """
    Opens a zarr store as an xarray Dataset.
    consolidated: if None, try True then fallback to False.
    """
    path_str = _resolve_path(path)

    def _try(cons: bool) -> xr.Dataset:
        return xr.open_zarr(
            path_str,
            consolidated=cons,
            chunks=chunks,
            decode_cf=decode_cf,
            mask_and_scale=mask_and_scale,
        )

    def _open_with_fallback(cons: bool) -> xr.Dataset:
        try:
            ds = _try(cons)
        except KeyError as exc:
            if "dimension_names" in str(exc):
                return open_zarr_tree(path_str)
            raise
        # Some stores keep arrays only in nested groups; root opens as empty.
        if not ds.data_vars and not ds.coords:
            return open_zarr_tree(path_str)
        return ds

    if consolidated is None:
        try:
            return _open_with_fallback(True)
        except Exception:
            return _open_with_fallback(False)
    return _open_with_fallback(consolidated)


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def list_vars_from_store(path: Union[str, Path]) -> List[str]:
    """
    Fallback variable listing for zarr stores without xarray metadata.
    Uses the presence of <group>/data + <group>/band_names when available.
    """
    if zarr is None:
        return []
    root = zarr.open_group(_resolve_path(path), mode="r")
    out: List[str] = []

    def add_group_vars(group_path: str, group) -> None:
        if "data" not in group:
            return
        names: List[str] = []
        if "band_names" in group:
            try:
                names = [_decode_name(n) for n in group["band_names"][:].tolist()]
            except Exception:
                names = []
        if names:
            out.extend([f"{group_path}/{n}" for n in names])
            return
        try:
            data = group["data"]
            c = int(data.shape[1]) if data.ndim >= 2 else int(data.shape[0])
        except Exception:
            c = 0
        if c <= 1:
            out.append(group_path)
        else:
            out.extend([f"{group_path}/band_{i}" for i in range(c)])

    def walk(group, prefix: str = "") -> None:
        for name, sub in group.groups():
            group_path = f"{prefix}{name}"
            add_group_vars(group_path, sub)
            walk(sub, group_path + "/")

    walk(root)
    return sorted(set(out))


def detect_time_dim(ds: xr.Dataset) -> Optional[str]:
    for name in TimeDimCandidate:
        if name in ds.dims or name in ds.coords:
            return name
    # last resort: any coord with datetime dtype
    for c in ds.coords:
        if np.issubdtype(ds[c].dtype, np.datetime64):
            return c
    return None


def ensure_datetime_index(ds: xr.Dataset, time_dim: Optional[str] = None) -> xr.Dataset:
    """
    Ensures ds[time_dim] is datetime64 index (if present).
    """
    td = time_dim or detect_time_dim(ds)
    if td is None or td not in ds.coords:
        return ds
    if not np.issubdtype(ds[td].dtype, np.datetime64):
        ds = ds.assign_coords({td: pd.to_datetime(ds[td].values)})
    return ds


def list_vars(ds: xr.Dataset) -> List[str]:
    return sorted(list(ds.data_vars.keys()))


def select_vars(
    ds: xr.Dataset,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    *,
    strict: bool = False,
) -> xr.Dataset:
    """
    include: keep only these vars (if provided)
    exclude: drop these vars (if provided)
    strict: if True, raise if any include var is missing
    """
    if include is not None:
        include = list(include)
        missing = [v for v in include if v not in ds.data_vars]
        if strict and missing:
            raise KeyError(f"Variables not found: {missing}")
        keep = [v for v in include if v in ds.data_vars]
        ds = ds[keep]
    if exclude:
        drop = [v for v in exclude if v in ds.data_vars]
        if drop:
            ds = ds.drop_vars(drop)
    return ds


def time_slice(
    ds: xr.Dataset,
    start: Optional[Union[str, date, datetime, np.datetime64, pd.Timestamp]] = None,
    end: Optional[Union[str, date, datetime, np.datetime64, pd.Timestamp]] = None,
    *,
    time_dim: Optional[str] = None,
) -> xr.Dataset:
    td = time_dim or detect_time_dim(ds)
    if td is None or td not in ds.coords:
        return ds

    ds = ensure_datetime_index(ds, td)

    if start is None and end is None:
        return ds

    s = _to_timestamp(start) if start is not None else None
    e = _to_timestamp(end) if end is not None else None

    if s is not None and e is not None:
        return ds.sel({td: slice(s, e)})
    if s is not None:
        return ds.sel({td: slice(s, None)})
    # e is not None
    return ds.sel({td: slice(None, e)})


def drop_all_nan_time(ds: xr.Dataset, *, time_dim: Optional[str] = None) -> xr.Dataset:
    """
    Drops time steps where all variables are NaN everywhere (helpful for bad scenes).
    """
    td = time_dim or detect_time_dim(ds)
    if td is None or td not in ds.dims:
        return ds

    # Build a boolean mask of time steps that have at least one finite value.
    keep = None
    for v in ds.data_vars:
        da = ds[v]
        # Reduce over all dims except time
        dims = [d for d in da.dims if d != td]
        m = np.isfinite(da).any(dim=dims)
        keep = m if keep is None else (keep | m)
    if keep is None:
        return ds
    return ds.sel({td: ds[td].where(keep, drop=True)})


# ----------------------------
# Main Loader
# ----------------------------

@dataclass
class LoadSpec:
    """
    Specifies what to load.
    """
    zarr: ZarrKey
    include_vars: Optional[Sequence[str]] = None
    exclude_vars: Optional[Sequence[str]] = None
    start: Optional[Union[str, date, datetime, np.datetime64, pd.Timestamp]] = None
    end: Optional[Union[str, date, datetime, np.datetime64, pd.Timestamp]] = None
    time_dim: Optional[str] = None
    strict_vars: bool = False
    drop_all_nan: bool = False


class MaduraiData:
    """
    Single entry point to load/merge Madurai datasets with consistent APIs.
    """

    def __init__(
        self,
        paths: DataPaths,
        *,
        zarr_open_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.paths = paths
        self.zarr_open_kwargs = zarr_open_kwargs or {}

        self._cache: Dict[ZarrKey, xr.Dataset] = {}
        self._landsat_dates_cache: Optional[Dict[str, List[pd.Timestamp]]] = None

    # ---- internal ----

    def _zarr_path(self, key: ZarrKey) -> Union[str, Path]:
        if key == "madurai":
            return self.paths.madurai_zarr
        if key == "madurai_30m":
            return self.paths.madurai_30m_zarr
        if key == "madurai_alphaearth_30m":
            return self.paths.madurai_alphaearth_30m_zarr
        raise ValueError(f"Unknown ZarrKey: {key}")

    def _open_cached(self, key: ZarrKey) -> xr.Dataset:
        if key in self._cache:
            return self._cache[key]
        ds = open_zarr(self._zarr_path(key), **self.zarr_open_kwargs)
        ds = ensure_datetime_index(ds, detect_time_dim(ds) or "time")
        self._cache[key] = ds
        return ds

    # ---- public: metadata ----

    def get_dataset(self, key: ZarrKey) -> xr.Dataset:
        return self._open_cached(key)

    def vars(self, key: ZarrKey) -> List[str]:
        if key == "madurai":
            return list_vars_from_store(self._zarr_path(key))
        try:
            ds = self.get_dataset(key)
            names = list_vars(ds)
            if names:
                return names
        except Exception:
            # Some stores lack xarray dimension metadata; fall back to zarr inspection.
            pass
        return list_vars_from_store(self._zarr_path(key))

    def time_dim(self, key: ZarrKey) -> Optional[str]:
        return detect_time_dim(self.get_dataset(key))

    def time_values(self, key: ZarrKey) -> Optional[pd.DatetimeIndex]:
        ds = self.get_dataset(key)
        td = detect_time_dim(ds)
        if td is None or td not in ds.coords:
            return None
        return pd.DatetimeIndex(pd.to_datetime(ds[td].values))

    # ---- public: landsat dates ----

    def landsat_dates(self) -> Dict[str, List[pd.Timestamp]]:
        if self._landsat_dates_cache is None:
            self._landsat_dates_cache = load_landsat_dates(self.paths.landsat_dates_json)
        return self._landsat_dates_cache

    def landsat_month_pick_map(
        self,
        start: Union[str, date, datetime, np.datetime64, pd.Timestamp],
        end: Union[str, date, datetime, np.datetime64, pd.Timestamp],
        *,
        strategy: MonthPickStrategy = "closest_to_mid",
        seed: int = 42,
        require_present: bool = False,
    ) -> Dict[str, pd.Timestamp]:
        """
        For each month in [start, end], pick one Landsat acquisition date using strategy.
        Returns: {"YYYY-MM": picked_date}
        """
        ld = self.landsat_dates()
        out: Dict[str, pd.Timestamp] = {}
        for m in months_in_range(start, end):
            if m not in ld:
                if require_present:
                    raise KeyError(f"No Landsat dates for month {m} in landsat_dates.json")
                continue
            out[m] = pick_landsat_date_for_month(ld[m], strategy=strategy, seed=seed)
        return out

    # ---- public: loading ----

    def load(self, spec: LoadSpec) -> xr.Dataset:
        if spec.zarr == "madurai" and spec.include_vars:
            ds = open_zarr_tree(self._zarr_path(spec.zarr), include_vars=spec.include_vars)
        else:
            ds = self._open_cached(spec.zarr)
        ds2 = select_vars(ds, spec.include_vars, spec.exclude_vars, strict=spec.strict_vars)
        ds2 = time_slice(ds2, spec.start, spec.end, time_dim=spec.time_dim)
        if spec.drop_all_nan:
            ds2 = drop_all_nan_time(ds2, time_dim=spec.time_dim)
        return ds2

    def load_many(
        self,
        specs: Sequence[LoadSpec],
        *,
        join: Literal["outer", "inner", "left", "right", "exact"] = "outer",
        compat: Literal["no_conflicts", "override", "identical", "broadcast_equals", "equals"] = "no_conflicts",
        combine_attrs: Literal["drop", "override", "no_conflicts", "identical"] = "override",
    ) -> xr.Dataset:
        """
        Loads multiple datasets and merges them.
        """
        dsets = [self.load(s) for s in specs]
        # Align before merge to avoid coordinate mismatches; keep chunks lazy.
        aligned = xr.align(*dsets, join=join)
        merged = xr.merge(aligned, compat=compat, combine_attrs=combine_attrs)
        return merged

    # ---- public: tensor conversion ----

    def to_numpy(
        self,
        ds: xr.Dataset,
        *,
        order: Sequence[str] = ("time", "y", "x"),
        channel_dim: str = "channel",
        fill_value: float = np.nan,
        dtype: np.dtype = np.float32,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Convert Dataset -> NumPy tensor with stacked variable channels.

        Returns:
          X: np.ndarray with shape [*order_dims, C] if all dims present,
             otherwise best-effort with available dims.
          channels: list of variable names in channel order
          used_order: list of dims used (subset of `order` that exist)
        """
        if not ds.data_vars:
            raise ValueError("Dataset has no data_vars to convert to numpy.")

        # Determine dims to use
        dims_in_ds = list(ds.dims)
        used_order = [d for d in order if d in dims_in_ds]
        # Stack variables into channel dimension
        channels = list(ds.data_vars.keys())

        # Convert to a single DataArray: [channel, ...dims...]
        da = xr.concat([ds[v] for v in channels], dim=channel_dim)

        # Reorder dims: used_order + [channel]
        target_dims = used_order + [channel_dim]
        da = da.transpose(*[d for d in target_dims if d in da.dims])

        arr = da.data  # may be dask
        arr = np.array(arr)  # materialize

        if np.isnan(fill_value):
            out = arr.astype(dtype, copy=False)
        else:
            out = np.where(np.isfinite(arr), arr, fill_value).astype(dtype, copy=False)

        # Ensure channel is last
        if out.ndim >= 2 and da.dims[-1] != channel_dim:
            # Should not happen due to transpose, but keep safe.
            ch_axis = da.dims.index(channel_dim)
            out = np.moveaxis(out, ch_axis, -1)

        return out, channels, used_order

    # ---- public: patching ----

    def extract_patches(
        self,
        arr: np.ndarray,
        *,
        patch_hw: Tuple[int, int],
        stride_hw: Optional[Tuple[int, int]] = None,
        spatial_axes: Tuple[int, int] = (-3, -2),
    ) -> np.ndarray:
        """
        Extract sliding window patches from a tensor.

        Expected arr shape examples:
          - [T, H, W, C]  (spatial_axes = (1,2) or (-3,-2))
          - [H, W, C]     (spatial_axes = (0,1) or (-3,-2) with adjustment)

        Returns:
          patches: [..., n_patches, ph, pw, C] or [n_patches, ph, pw, C]
        """
        ph, pw = patch_hw
        sh, sw = stride_hw or patch_hw

        a = arr
        if a.ndim < 3:
            raise ValueError("arr must have at least 3 dims (H, W, C) or (T, H, W, C).")

        # Normalize spatial axes to positive indices
        ax_h = spatial_axes[0] % a.ndim
        ax_w = spatial_axes[1] % a.ndim
        if ax_h == ax_w:
            raise ValueError("spatial_axes must refer to two different axes.")

        # Move H,W,C to the end as [..., H, W, C] for simplicity
        # Identify channel axis as last by convention; if not last, user should rearrange before calling.
        if ax_w != a.ndim - 2 or ax_h != a.ndim - 3:
            # bring H to -3 and W to -2
            a = np.moveaxis(a, (ax_h, ax_w), (-3, -2))

        H, W = a.shape[-3], a.shape[-2]
        if H < ph or W < pw:
            raise ValueError(f"Patch {patch_hw} larger than spatial dims {(H, W)}.")

        n_h = 1 + (H - ph) // sh
        n_w = 1 + (W - pw) // sw
        patches = []

        # Iterate deterministically
        for i in range(n_h):
            for j in range(n_w):
                hs = i * sh
                ws = j * sw
                patch = a[..., hs : hs + ph, ws : ws + pw, :]
                patches.append(patch)

        stacked = np.stack(patches, axis=-4)  # insert n_patches before H,W,C
        return stacked


# ----------------------------
# Convenience constructors
# ----------------------------

def make_madurai_data(
    *,
    chunks: Union[str, Dict[str, int], None] = "auto",
    consolidated: Optional[bool] = None,
) -> MaduraiData:
    """
    Factory with fixed project paths.
    """
    zarr_open_kwargs = dict(chunks=chunks, consolidated=consolidated)
    return MaduraiData(DEFAULT_PATHS, zarr_open_kwargs=zarr_open_kwargs)


# ----------------------------
# Practical patterns (ready-to-call)
# ----------------------------

def load_all_data(
    md: MaduraiData,
    *,
    start: Optional[Union[str, date, datetime, np.datetime64, pd.Timestamp]] = None,
    end: Optional[Union[str, date, datetime, np.datetime64, pd.Timestamp]] = None,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    include_30m: bool = True,
    include_alphaearth_30m: bool = True,
    join: Literal["outer", "inner", "left", "right", "exact"] = "outer",
) -> xr.Dataset:
    """
    Load a merged dataset across madurai(.zarr) plus optional 30m stores.
    """
    specs: List[LoadSpec] = [
        LoadSpec("madurai", include_vars=include, exclude_vars=exclude, start=start, end=end),
    ]
    if include_30m:
        specs.append(LoadSpec("madurai_30m", start=start, end=end))
    if include_alphaearth_30m:
        specs.append(LoadSpec("madurai_alphaearth_30m", start=start, end=end))

    return md.load_many(specs, join=join)


def load_subset(
    md: MaduraiData,
    key: ZarrKey,
    *,
    vars_include: Optional[Sequence[str]] = None,
    vars_exclude: Optional[Sequence[str]] = None,
    start: Optional[Union[str, date, datetime, np.datetime64, pd.Timestamp]] = None,
    end: Optional[Union[str, date, datetime, np.datetime64, pd.Timestamp]] = None,
    strict: bool = False,
    drop_all_nan: bool = False,
) -> xr.Dataset:
    """
    Load a single zarr store with variable filtering + date slicing.
    """
    return md.load(
        LoadSpec(
            zarr=key,
            include_vars=vars_include,
            exclude_vars=vars_exclude,
            start=start,
            end=end,
            strict_vars=strict,
            drop_all_nan=drop_all_nan,
        )
    )


def load_landsat_monthly_as_single_date(
    md: MaduraiData,
    ds: xr.Dataset,
    *,
    start: Union[str, date, datetime, np.datetime64, pd.Timestamp],
    end: Union[str, date, datetime, np.datetime64, pd.Timestamp],
    time_dim: Optional[str] = None,
    strategy: MonthPickStrategy = "closest_to_mid",
    seed: int = 42,
    require_present: bool = False,
) -> xr.Dataset:
    """
    If your Landsat-related variables in madurai.zarr are keyed monthly (one per month),
    but you have multiple true acquisition dates per month, this function creates a
    "picked_time" coordinate mapping each month to one chosen acquisition date.

    It does NOT change the data values; it re-labels/augments time to your chosen date,
    which is useful for spatiotemporal alignment.

    Works when ds has a time coord with monthly cadence.
    """
    td = time_dim or detect_time_dim(ds)
    if td is None or td not in ds.coords:
        raise ValueError("Dataset has no time coordinate; cannot apply Landsat month picking.")

    ds = ensure_datetime_index(ds, td)
    picks = md.landsat_month_pick_map(start, end, strategy=strategy, seed=seed, require_present=require_present)

    # Build a mapping from each ds time (month) -> picked date if exists
    tvals = pd.DatetimeIndex(pd.to_datetime(ds[td].values))
    picked = []
    for t in tvals:
        mk = t.strftime("%Y-%m")
        if mk in picks:
            picked.append(picks[mk])
        else:
            picked.append(pd.Timestamp(t))  # fallback: keep as-is
    picked = pd.to_datetime(picked)

    return ds.assign_coords({f"{td}_picked": (td, picked)})


# ----------------------------
# Debug helpers
# ----------------------------

def describe(ds: xr.Dataset, *, max_vars: int = 50) -> str:
    """
    Compact description: dims, coords, variables, dtypes, NaN ratios (quick).
    """
    lines = []
    lines.append("Dims: " + ", ".join([f"{k}={v}" for k, v in ds.dims.items()]))

    coords = list(ds.coords)
    lines.append("Coords: " + (", ".join(coords) if coords else "(none)"))

    vars_ = list(ds.data_vars)
    lines.append(f"Vars ({len(vars_)}): " + ", ".join(vars_[:max_vars]) + ("" if len(vars_) <= max_vars else " ..."))

    # quick dtype summary
    dtypes = {}
    for v in vars_[:max_vars]:
        dtypes.setdefault(str(ds[v].dtype), 0)
        dtypes[str(ds[v].dtype)] += 1
    lines.append("Dtypes: " + ", ".join([f"{k}×{n}" for k, n in dtypes.items()]))

    # quick nan ratio (sample)
    try:
        nan_ratios = []
        for v in vars_[: min(12, len(vars_))]:
            da = ds[v]
            # sample at most 1e6 values
            a = np.array(da.data)
            if a.size > 1_000_000:
                idx = np.random.default_rng(0).choice(a.size, size=1_000_000, replace=False)
                a = a.reshape(-1)[idx]
            r = float(np.mean(~np.isfinite(a)))
            nan_ratios.append((v, r))
        lines.append("NaN/Inf sample ratios: " + ", ".join([f"{v}={r:.3f}" for v, r in nan_ratios]))
    except Exception:
        lines.append("NaN/Inf sample ratios: (skipped; could not materialize)")

    return "\n".join(lines)


# ----------------------------
# Example usage (keep commented)
# ----------------------------
# md = make_madurai_data("/path/to/data_root")
# ds_all = load_all_data(md, start="2019-01-01", end="2020-12-31", include_30m=True, include_alphaearth_30m=True)
# print(describe(ds_all))
#
# ds_m = load_subset(md, "madurai", vars_include=["modis_lst_day", "viirs_lst_day"], start="2019-01", end="2019-12")
# X, channels, used_order = md.to_numpy(ds_m, order=("time", "y", "x"))
# patches = md.extract_patches(X, patch_hw=(64,64), stride_hw=(64,64), spatial_axes=(1,2))
