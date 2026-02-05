from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from helper import detect_time_dim, list_vars_from_store, make_madurai_data, open_zarr_tree, select_vars
from helper.metrics_image import compute_all


PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project").resolve()
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "metrics" / "deep_baselines"
FIG_DIR = OUT_DIR / "shallow_cnn_simple_figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MAX_PRECOMPUTE_GB = 0.5

ERA5_TOKENS = ("era5", "ecmwf")
S1_TOKENS = ("sentinel-1", "sentinel_1", "sentinel1", "/s1/", "s1_", "_s1")
S2_TOKENS = ("sentinel-2", "sentinel_2", "sentinel2", "/s2/", "s2_")
S2_INDEX_TOKENS = (
    "ndvi", "evi", "savi", "msavi",
    "ndwi_water", "ndwi_moisture", "ndmi", "mndwi",
    "ndbi", "ui", "bsi", "ibi", "albedo_proxy",
)
LANDCOVER_TOKENS = ("worldcover", "dynamic_world", "landcover")
DEM_TOKENS = ("dem", "elevation", "srtm")
CLOUD_TOKENS = ("qc", "quality", "mask", "cloud", "flag", "clear_count", "total_count", "cloud_frac")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    var: str
    band: Optional[int] = None


def _zarr_path_for_key(md, key: str) -> Path:
    if key == "madurai":
        return Path(md.paths.madurai_zarr)
    if key == "madurai_30m":
        return Path(md.paths.madurai_30m_zarr)
    if key == "madurai_alphaearth_30m":
        return Path(md.paths.madurai_alphaearth_30m_zarr)
    raise ValueError(f"Unknown dataset key: {key}")


def _is_era5_var(name: str) -> bool:
    return any(tok in name.lower() for tok in ERA5_TOKENS)


def _is_s1_var(name: str) -> bool:
    lower = name.lower()
    if any(tok in lower for tok in S1_TOKENS):
        return True
    if lower.endswith("_vv") or lower.endswith("_vh") or "/vv" in lower or "/vh" in lower:
        return True
    return False


def _is_s2_var(name: str) -> bool:
    lower = name.lower()
    return any(tok in lower for tok in S2_TOKENS) or any(tok in lower for tok in S2_INDEX_TOKENS)


def _is_landcover_var(name: str) -> bool:
    lower = name.lower()
    return any(tok in lower for tok in LANDCOVER_TOKENS)


def _is_cloud_var(name: str) -> bool:
    lower = name.lower()
    return any(tok in lower for tok in CLOUD_TOKENS)


def _is_landsat_var(name: str) -> bool:
    return "landsat" in name.lower()


def _is_daily_var(name: str) -> bool:
    lower = name.lower()
    return "landsat" in lower or "modis" in lower or "viirs" in lower or _is_era5_var(lower)


def _is_dem_var(name: str) -> bool:
    lower = name.lower()
    return any(tok in lower for tok in DEM_TOKENS)


def _is_static_var(name: str) -> bool:
    lower = name.lower()
    return _is_landcover_var(lower) or _is_dem_var(lower)


def _is_lst_source(name: str) -> bool:
    name = name.lower()
    if _is_era5_var(name):
        return False
    return (
        _is_landsat_var(name)
        or "viirs" in name
        or "modis" in name
        or "lst" in name
        or "temperature" in name
        or "temp" in name
        or name.endswith("band_01")
    )


def _select_vars_from_names(vars_list: Sequence[str], target: str) -> Tuple[List[str], List[str]]:
    base = []
    era5 = []
    for v in vars_list:
        if v == target:
            continue
        if _is_cloud_var(v):
            continue
        if _is_landsat_var(v):
            continue
        if _is_era5_var(v):
            era5.append(v)
            continue
        if _is_s1_var(v) or _is_s2_var(v) or _is_landcover_var(v):
            base.append(v)
    if not era5:
        raise RuntimeError("No ERA5 variables found.")
    if not base:
        raise RuntimeError("No S1/S2/Landcover variables found.")
    return base, era5


def _load_subset(md, key: str, vars_include: Sequence[str]) -> xr.Dataset:
    path = _zarr_path_for_key(md, key)
    try:
        ds = xr.open_zarr(path, consolidated=True, chunks="auto", decode_cf=False, mask_and_scale=False)
    except Exception:
        ds = xr.open_zarr(path, consolidated=False, chunks="auto", decode_cf=False, mask_and_scale=False)
    if not ds.data_vars:
        ds = open_zarr_tree(path, include_vars=vars_include)
    return select_vars(ds, include=vars_include, strict=False)


def _apply_common_dates(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    common_dates_path = PROJECT_ROOT / "common_dates.csv"
    if not common_dates_path.exists():
        raise FileNotFoundError("common_dates.csv not found.")
    common_df = pd.read_csv(common_dates_path)
    if "landsat_date" not in common_df.columns:
        raise KeyError("common_dates.csv missing 'landsat_date' column")
    common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna()
    common_dates = pd.DatetimeIndex(common_dates).sort_values()
    filtered = times[times.isin(common_dates)]
    if len(filtered) == 0:
        raise RuntimeError("No dates left after common_dates.csv filtering.")
    return filtered


def _align_to_common_dates(ds: xr.Dataset, td: str, common_dates: pd.DatetimeIndex) -> xr.Dataset:
    month_index = pd.DatetimeIndex(common_dates.to_period("M").to_timestamp())
    out: Dict[str, xr.DataArray] = {}
    for v in ds.data_vars:
        da = ds[v]
        if td not in da.dims:
            out[v] = da
            continue
        if _is_daily_var(v):
            out[v] = da.sel({td: common_dates})
        elif _is_static_var(v):
            out[v] = da
        else:
            monthly = da.sel({td: month_index})
            if monthly.sizes.get(td, 0) != len(month_index):
                monthly = monthly.reindex({td: month_index})
            monthly = monthly.assign_coords({td: common_dates})
            out[v] = monthly
    coords = {k: ds.coords[k] for k in ds.coords if k != td}
    coords[td] = common_dates
    return xr.Dataset(out, coords=coords, attrs=ds.attrs)


def _infer_xy_dims(da: xr.DataArray) -> Tuple[str, str]:
    if "y" in da.dims and "x" in da.dims:
        return "y", "x"
    return da.dims[-2], da.dims[-1]


def _infer_time_dim(ds: xr.Dataset, target: str) -> Optional[str]:
    td = detect_time_dim(ds)
    if td:
        return td
    if target in ds.data_vars:
        dims = ds[target].dims
        for cand in ("time", "date", "t", "datetime"):
            if cand in dims:
                return cand
        for d in dims:
            if d not in ("band", "y", "x"):
                return d
    for cand in ("time", "date", "t", "datetime"):
        if cand in ds.coords or cand in ds.dims:
            return cand
    return None


def _select_time(ds: xr.Dataset, td: str, t: pd.Timestamp) -> xr.Dataset:
    dts = ds.sel({td: t})
    if td in dts.dims and dts.sizes.get(td, 0) == 1:
        dts = dts.isel({td: 0}, drop=True)
    return dts


def _mask_lst_values(data: np.ndarray, var_name: str) -> np.ndarray:
    out = np.array(data, dtype=np.float32, copy=True)
    out[out <= 0] = np.nan
    if _is_landsat_var(var_name):
        out[np.isclose(out, 149.0)] = np.nan
    return out


def _convert_lst_units(data: np.ndarray, var_name: str) -> np.ndarray:
    if not np.isfinite(data).any():
        return data
    if "modis" in var_name.lower():
        return data
    if _is_landsat_var(var_name) or "viirs" in var_name.lower():
        med = float(np.nanmedian(data))
        if med > 200:
            return data - 273.15
    return data


def _expand_feature_specs(ds: xr.Dataset, vars_list: Sequence[str], time_dim: Optional[str]) -> List[FeatureSpec]:
    specs: List[FeatureSpec] = []
    for v in vars_list:
        da = ds[v]
        dims = list(da.dims)
        if time_dim and time_dim in dims:
            dims = [d for d in dims if d != time_dim]
        if "band" in dims:
            band_vals = da.coords.get("band", None)
            if band_vals is None:
                band_vals = np.arange(da.sizes["band"])
            if da.sizes["band"] <= 1:
                specs.append(FeatureSpec(name=v, var=v, band=0))
            else:
                for i, b in enumerate(band_vals.values):
                    specs.append(FeatureSpec(name=f"{v}[{b}]", var=v, band=i))
            continue
        if len(dims) == 3:
            n0 = da.shape[0]
            if n0 <= 1:
                specs.append(FeatureSpec(name=v, var=v, band=None))
            else:
                for i in range(n0):
                    specs.append(FeatureSpec(name=f"{v}[{i}]", var=v, band=i))
            continue
        specs.append(FeatureSpec(name=v, var=v, band=None))
    return specs


def _load_feature_value(dts: xr.Dataset, spec: FeatureSpec, r: int, c: int) -> float:
    da = dts[spec.var]
    if spec.band is not None:
        if "band" in da.dims:
            da = da.isel(band=spec.band)
        else:
            da = da.isel({da.dims[0]: spec.band})
    y_dim, x_dim = _infer_xy_dims(da)
    val = np.asarray(da.isel({y_dim: r, x_dim: c}).data, dtype=np.float32)
    val = np.squeeze(val)
    if _is_lst_source(spec.var):
        val = _mask_lst_values(val, spec.var)
        val = _convert_lst_units(val, spec.var)
    return float(val)


def _load_feature_patch(dts: xr.Dataset, spec: FeatureSpec, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    da = dts[spec.var]
    if spec.band is not None:
        if "band" in da.dims:
            da = da.isel(band=spec.band)
        else:
            da = da.isel({da.dims[0]: spec.band})
    y_dim, x_dim = _infer_xy_dims(da)
    da = da.isel({y_dim: slice(r0, r1), x_dim: slice(c0, c1)})
    arr = np.asarray(da.data, dtype=np.float32)
    if _is_lst_source(spec.var):
        arr = _mask_lst_values(arr, spec.var)
        arr = _convert_lst_units(arr, spec.var)
    return arr


def _load_feature_map(dts: xr.Dataset, spec: FeatureSpec) -> np.ndarray:
    da = dts[spec.var]
    if spec.band is not None:
        if "band" in da.dims:
            da = da.isel(band=spec.band)
        else:
            da = da.isel({da.dims[0]: spec.band})
    y_dim, x_dim = _infer_xy_dims(da)
    da = da.isel({y_dim: slice(None), x_dim: slice(None)})
    arr = np.asarray(da.data, dtype=np.float32)
    if _is_lst_source(spec.var):
        arr = _mask_lst_values(arr, spec.var)
        arr = _convert_lst_units(arr, spec.var)
    return arr


def _load_feature_map_from_ds(ds: xr.Dataset, td: str, spec: FeatureSpec, t: Optional[pd.Timestamp]) -> np.ndarray:
    da = ds[spec.var]
    if td in da.dims:
        if t is not None:
            da = da.sel({td: t})
            if da.sizes.get(td, 0) == 1:
                da = da.isel({td: 0}, drop=True)
        elif _is_static_var(spec.var):
            da = da.isel({td: 0}, drop=True)
    if spec.band is not None:
        if "band" in da.dims:
            da = da.isel(band=spec.band)
        else:
            da = da.isel({da.dims[0]: spec.band})
    y_dim, x_dim = _infer_xy_dims(da)
    da = da.isel({y_dim: slice(None), x_dim: slice(None)})
    arr = np.asarray(da.data, dtype=np.float32)
    if _is_lst_source(spec.var):
        arr = _mask_lst_values(arr, spec.var)
        arr = _convert_lst_units(arr, spec.var)
    return arr


def _split_specs_by_cadence(
    ds: xr.Dataset, td: str, specs: Sequence[FeatureSpec]
) -> Tuple[List[FeatureSpec], List[FeatureSpec], List[FeatureSpec]]:
    daily: List[FeatureSpec] = []
    monthly: List[FeatureSpec] = []
    static: List[FeatureSpec] = []
    for spec in specs:
        da = ds[spec.var]
        if td not in da.dims:
            static.append(spec)
            continue
        if _is_daily_var(spec.var):
            daily.append(spec)
        elif _is_static_var(spec.var):
            static.append(spec)
        else:
            monthly.append(spec)
    return daily, monthly, static


def _sample_center_pixels(
    ds: xr.Dataset,
    td: str,
    times: Sequence[pd.Timestamp],
    target: str,
    features: Sequence[FeatureSpec],
    n_samples: int,
    rng: np.random.Generator,
    patch_size: int,
    max_tries: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.empty((n_samples, len(features)), dtype=np.float32)
    y = np.empty((n_samples,), dtype=np.float32)
    missing = np.zeros((len(features),), dtype=np.int64)
    count = 0
    tries = 0
    times_list = list(times)
    center_cache: Dict[pd.Timestamp, np.ndarray] = {}
    center_idx: Dict[pd.Timestamp, int] = {}
    dts_cache: Dict[pd.Timestamp, xr.Dataset] = {}
    shape_cache: Dict[pd.Timestamp, Tuple[int, int]] = {}

    def _centers_for_time(t: pd.Timestamp) -> np.ndarray:
        if t in center_cache:
            return center_cache[t]
        dts_local = dts_cache.get(t)
        if dts_local is None:
            dts_local = _select_time(ds, td, t)
            dts_cache[t] = dts_local
        da_tgt = dts_local[target]
        y_dim, x_dim = _infer_xy_dims(da_tgt)
        H, W = int(da_tgt.sizes[y_dim]), int(da_tgt.sizes[x_dim])
        shape_cache[t] = (H, W)
        r0, r1 = _center_range(H, patch_size)
        c0, c1 = _center_range(W, patch_size)
        n_centers = max(256, n_samples // max(1, len(times_list)))
        rows = rng.integers(r0, r1 + 1, size=n_centers)
        cols = rng.integers(c0, c1 + 1, size=n_centers)
        centers = np.stack([rows, cols], axis=1)
        center_cache[t] = centers
        center_idx[t] = 0
        return centers

    while count < n_samples and tries < n_samples * max_tries:
        tries += 1
        t = times_list[int(rng.integers(0, len(times_list)))]
        dts = dts_cache.get(t)
        if dts is None:
            dts = _select_time(ds, td, t)
            dts_cache[t] = dts
        centers = _centers_for_time(t)
        idx = center_idx.get(t, 0)
        if idx >= len(centers):
            center_cache.pop(t, None)
            centers = _centers_for_time(t)
            idx = 0
        r, c = centers[idx]
        center_idx[t] = idx + 1
        yv = _load_feature_value(dts, FeatureSpec(target, target), r, c)
        if not np.isfinite(yv):
            continue
        row = []
        for j, spec in enumerate(features):
            val = _load_feature_value(dts, spec, r, c)
            row.append(val)
            if not np.isfinite(val):
                missing[j] += 1
        X[count] = row
        y[count] = yv
        count += 1
    if count == 0:
        raise RuntimeError("No valid training samples found.")
    return X[:count], y[:count], missing


def _compute_train_medians(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    meds = np.nanmedian(X_train, axis=0)
    keep_mask = np.isfinite(meds)
    meds = np.where(keep_mask, meds, 0.0)
    return meds.astype(np.float32), keep_mask


def _estimate_patch_bytes(samples: int, n_features: int, patch_size: int, dtype_bytes: int = 2) -> int:
    if samples <= 0 or n_features <= 0 or patch_size <= 0:
        return 0
    return int(samples * n_features * patch_size * patch_size * dtype_bytes)


def _cap_patch_samples(
    train_samples: int,
    val_samples: int,
    n_features: int,
    patch_size: int,
    max_gb: float,
) -> Tuple[int, int, float]:
    max_bytes = int(max_gb * 1024**3)
    per = _estimate_patch_bytes(1, n_features, patch_size)
    if per <= 0:
        return train_samples, val_samples, 1.0
    total = (train_samples + val_samples) * per
    if total <= max_bytes:
        return train_samples, val_samples, 1.0
    scale = max_bytes / max(1, total)
    new_train = max(1, int(train_samples * scale))
    new_val = max(1, int(val_samples * scale))
    return new_train, new_val, scale


def _sample_patch_arrays(
    ds: xr.Dataset,
    td: str,
    times: Sequence[pd.Timestamp],
    target: str,
    features: Sequence[FeatureSpec],
    n_samples: int,
    rng: np.random.Generator,
    patch_size: int,
    train_medians: np.ndarray,
    max_tries: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    print("sample_patch_arrays: start")
    if n_samples <= 0:
        return (
            np.empty((0, len(features), patch_size, patch_size), dtype=np.float16),
            np.empty((0,), dtype=np.float32),
        )

    X = np.empty((n_samples, len(features), patch_size, patch_size), dtype=np.float16)
    y = np.empty((n_samples,), dtype=np.float32)
    print("sample_patch_arrays: allocated buffers")

    count = 0
    tries = 0
    times_list = list(times)
    if not times_list:
        raise RuntimeError("No times available for sampling.")
    print("sample_patch_arrays: times list ready", len(times_list))
    rng.shuffle(times_list)
    print("sample_patch_arrays: times shuffled")
    samples_per_time = max(1, int(np.ceil(n_samples / max(1, len(times_list)))))
    print("sample_patch_arrays: samples_per_time", samples_per_time)
    daily_specs, monthly_specs, static_specs = _split_specs_by_cadence(ds, td, features)
    static_maps = {spec: _load_feature_map_from_ds(ds, td, spec, None) for spec in static_specs}
    chunk_size = 8
    for i in range(0, len(times_list), chunk_size):
        if count >= n_samples or tries >= n_samples * max_tries:
            break
        chunk_times = times_list[i : i + chunk_size]
        print("sample_patch_arrays: chunk", i, "size", len(chunk_times))
        dts_chunk = ds.sel({td: chunk_times})
        tgt_chunk = _load_feature_map_from_ds(dts_chunk, td, FeatureSpec(target, target), None)
        daily_maps: Dict[FeatureSpec, np.ndarray] = {}
        for spec in daily_specs + monthly_specs:
            daily_maps[spec] = _load_feature_map_from_ds(dts_chunk, td, spec, None)
        for idx, t in enumerate(chunk_times):
            if count >= n_samples or tries >= n_samples * max_tries:
                break
            print("sample_patch_arrays: selecting time", t)
            tgt_map = tgt_chunk[idx]
            H, W = tgt_map.shape
            r0c, r1c = _center_range(H, patch_size)
            c0c, c1c = _center_range(W, patch_size)
            n_take = min(samples_per_time, n_samples - count)
            print("sample_patch_arrays: sampling centers", n_take)
            rows = rng.integers(r0c, r1c + 1, size=n_take)
            cols = rng.integers(c0c, c1c + 1, size=n_take)
            for r, c in zip(rows, cols):
                if count >= n_samples or tries >= n_samples * max_tries:
                    break
                tries += 1
                r0, r1 = _patch_bounds(int(r), patch_size)
                c0, c1 = _patch_bounds(int(c), patch_size)
                print("sample_patch_arrays: slice target patch")
                tgt_patch = tgt_map[r0:r1, c0:c1]
                center = tgt_patch[patch_size // 2, patch_size // 2]
                if not np.isfinite(center):
                    continue
                if abs(float(center)) > 1e6:
                    continue
                for j, spec in enumerate(features):
                    if j == 0:
                        print("sample_patch_arrays: slice feature patch", features[j].name)
                    if spec in daily_maps:
                        fmap = daily_maps[spec][idx]
                    else:
                        fmap = static_maps[spec]
                    patch = fmap[r0:r1, c0:c1]
                    patch = np.nan_to_num(
                        patch,
                        nan=float(train_medians[j]),
                        posinf=float(train_medians[j]),
                        neginf=float(train_medians[j]),
                    )
                    if not np.isfinite(patch).all():
                        patch = np.zeros_like(patch, dtype=np.float32)
                    X[count, j] = patch.astype(np.float16, copy=False)
                y[count] = float(center)
                count += 1
    if count == 0:
        raise RuntimeError("No valid training samples found.")
    return X[:count], y[:count]


def _center_range(size: int, patch_size: int) -> Tuple[int, int]:
    half = patch_size // 2
    if patch_size % 2 == 0:
        min_center = half
        max_center = size - half
    else:
        min_center = half
        max_center = size - half - 1
    if max_center < min_center:
        raise ValueError(f"Patch size {patch_size} too large for dimension {size}.")
    return min_center, max_center


def _patch_bounds(center: int, patch_size: int) -> Tuple[int, int]:
    half = patch_size // 2
    if patch_size % 2 == 0:
        return center - half, center + half
    return center - half, center + half + 1


def _split_train_val_dates(times: Sequence[pd.Timestamp], val_frac: float) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    times = list(times)
    n_val = max(1, int(round(len(times) * val_frac)))
    if len(times) <= n_val:
        return times, times
    return times[:-n_val], times[-n_val:]


class ShallowCNN(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    max_epochs: int,
    lr: float,
    patience: int,
    center_idx: int,
) -> Tuple[nn.Module, float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best = float("inf")
    best_state = None
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            if not torch.isfinite(xb).all() or not torch.isfinite(yb).all():
                print("skip batch: non-finite train batch")
                continue
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            pred = out[:, 0, center_idx, center_idx]
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                print("skip batch: non-finite train loss")
                continue
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.float32)
                if not torch.isfinite(xb).all() or not torch.isfinite(yb).all():
                    continue
                out = model(xb)
                pred = out[:, 0, center_idx, center_idx]
                loss = loss_fn(pred, yb)
                if not torch.isfinite(loss):
                    continue
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}\n")
        if epoch % 5 == 0:
            try:
                sample_x, sample_y = next(iter(val_loader))
                sample_x = sample_x.to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    out = model(sample_x)[:, 0]
                n_show = min(5, out.shape[0])
                for i in range(n_show):
                    center_val = float(out[i, center_idx, center_idx].cpu().item())
                    actual_val = float(sample_y[i].item())
                    input_val = float(sample_x[i, 0, center_idx, center_idx].cpu().item())
                    print(
                        f"epoch={epoch} debug[{i}] input={input_val:.6f} "
                        f"pred={center_val:.6f} actual={actual_val:.6f}"
                    )
            except Exception as exc:
                print(f"epoch={epoch} debug failed: {exc}")
        if np.isfinite(val_loss) and val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def _prepare_padded_features(
    ds: xr.Dataset,
    td: str,
    t: pd.Timestamp,
    feature_specs: Sequence[FeatureSpec],
    train_medians: np.ndarray,
    pad: int,
) -> List[np.ndarray]:
    padded_maps: List[np.ndarray] = []
    for j, spec in enumerate(feature_specs):
        fmap = _load_feature_map_from_ds(ds, td, spec, t)
        fmap = fmap.astype(np.float32, copy=True)
        bad = ~np.isfinite(fmap)
        if bad.any():
            fmap[bad] = train_medians[j]
        padded = np.pad(fmap, ((pad, pad), (pad, pad)), mode="constant", constant_values=float(train_medians[j]))
        padded_maps.append(padded)
    return padded_maps


def predict_full_map_cnn(
    model: nn.Module,
    ds: xr.Dataset,
    td: str,
    t: pd.Timestamp,
    target: str,
    feature_specs: Sequence[FeatureSpec],
    *,
    tile_size: int,
    pad: int,
    device: torch.device,
    train_medians: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = _load_feature_map_from_ds(ds, td, FeatureSpec(target, target), t)
    y_true = _mask_lst_values(y_true, target)
    y_true = _convert_lst_units(y_true, target)
    H, W = y_true.shape
    y_pred = np.full((H, W), np.nan, dtype=np.float32)
    padded_maps = _prepare_padded_features(ds, td, t, feature_specs, train_medians, pad)

    model.eval()
    with torch.no_grad():
        for r0 in range(0, H, tile_size):
            r1 = min(H, r0 + tile_size)
            for c0 in range(0, W, tile_size):
                c1 = min(W, c0 + tile_size)
                tile_h = r1 - r0
                tile_w = c1 - c0
                cols = []
                for fmap in padded_maps:
                    tile = fmap[r0 : r1 + 2 * pad, c0 : c1 + 2 * pad]
                    cols.append(tile)
                X = np.stack(cols, axis=0)
                xb = torch.from_numpy(X[None, ...]).to(device=device, dtype=torch.float32)
                out = model(xb)[0, 0].cpu().numpy()
                y_pred[r0:r1, c0:c1] = out[pad : pad + tile_h, pad : pad + tile_w]
    return y_true, y_pred


def save_prediction_figures(y_true: np.ndarray, y_pred: np.ndarray, time_label: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    vmin = float(np.nanmin(y_true)) if np.isfinite(y_true).any() else 0.0
    vmax = float(np.nanmax(y_true)) if np.isfinite(y_true).any() else 1.0
    axes[0].imshow(y_true, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("truth")
    axes[1].imshow(y_pred, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("pred")
    err = np.abs(y_pred - y_true)
    axes[2].imshow(err, cmap="magma")
    axes[2].set_title("abs_error")
    for ax in axes:
        ax.axis("off")
    out_path = out_dir / f"shallow_cnn_simple_{time_label}_map.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_metric_plots(df: pd.DataFrame, out_dir: Path) -> None:
    times = pd.to_datetime(df["time"], errors="coerce")
    metrics = [c for c in df.columns if c not in ("time", "n_valid")]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for m in metrics:
        ax.plot(times, df[m], marker="o", linewidth=1.5, label=m)
    ax.set_title("metrics over time")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend()
    fig.savefig(out_dir / "shallow_cnn_simple_metrics_timeseries.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    mean_vals = [float(np.nanmean(df[m])) for m in metrics]
    ax.bar(metrics, mean_vals)
    ax.set_title("metrics mean")
    ax.set_ylabel("value")
    fig.savefig(out_dir / "shallow_cnn_simple_metrics_mean.png", dpi=150)
    plt.close(fig)


def run(
    *,
    dataset_key: str,
    start: Optional[str],
    end: Optional[str],
    target: Optional[str],
    test_frac: float,
    median_samples: int,
    patch_size: int,
    samples_per_epoch: int,
    batch_size: int,
    max_epochs: int,
    val_frac: float,
    patience: int,
    lr: float,
    tile_size: int,
    dry_run_day: Optional[str],
) -> None:
    md = make_madurai_data(chunks=None, consolidated=False)
    vars_all = list_vars_from_store(_zarr_path_for_key(md, dataset_key))
    print(vars_all)

    tgt = "products/landsat/band_01"
    if tgt not in vars_all:
        candidates = [v for v in vars_all if "landsat" in v.lower() and "band_01" in v.lower()]
        if len(candidates) == 1:
            tgt = candidates[0]
        else:
            raise KeyError("Target not found.")
    if target and target != tgt:
        tgt = target

    base_features, era5_feats = _select_vars_from_names(vars_all, tgt)
    vars_include = list(dict.fromkeys(list(base_features) + list(era5_feats) + [tgt]))
    print("variables : " , vars_include )
    ds = _load_subset(md, dataset_key, vars_include=vars_include)
    print("subset loaded",  ds)
    td = _infer_time_dim(ds, tgt)
    print("time dimension detected")
    if td is None:
        raise RuntimeError("Dataset has no time dimension.")
    
    times = pd.DatetimeIndex(pd.to_datetime(ds[td].values, errors="coerce"))
    times = times[~times.isna()]
    times = _apply_common_dates(times)
    if dry_run_day:
        if dry_run_day == "random":
            if len(times) == 0:
                raise RuntimeError("No common dates available for random dry run.")
            day = pd.Timestamp(times[np.random.default_rng(RANDOM_SEED).integers(0, len(times))])
        else:
            day = pd.to_datetime(dry_run_day, errors="coerce")
            if pd.isna(day):
                raise ValueError(f"Invalid --dry-run-day: {dry_run_day}")
        times = times[times == day]
        if len(times) == 0:
            raise RuntimeError(f"dry-run day {day.date()} not in common_dates.csv")
    ds = _align_to_common_dates(ds, td, times)
    print("times calculation done")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA not available.")

    feature_specs = _expand_feature_specs(ds, list(base_features) + list(era5_feats), td)
    print("features specs done")
    times_sorted = list(pd.DatetimeIndex(times).sort_values())
    if len(times_sorted) <= 1:
        train_times = times_sorted
        val_times = times_sorted
        test_times = times_sorted
    else:
        n_test = max(1, int(round(len(times_sorted) * test_frac)))
        train_times = times_sorted[:-n_test]
        test_times = times_sorted[-n_test:]
        if not train_times:
            train_times = times_sorted
        train_times, val_times = _split_train_val_dates(train_times, val_frac)
    print("split done")
    rng = np.random.default_rng(RANDOM_SEED)
    train_medians = np.zeros(len(feature_specs), dtype=np.float32)
    train_samples = samples_per_epoch
    val_samples = max(1, samples_per_epoch // 4)
    train_samples, val_samples, _ = _cap_patch_samples(
        train_samples, val_samples, len(feature_specs), patch_size, MAX_PRECOMPUTE_GB
    )
    print("cap patch samples done")
    X_train, y_train = _sample_patch_arrays(
        ds, td, train_times, tgt, feature_specs, train_samples, rng, patch_size, train_medians
    )
    X_val, y_val = _sample_patch_arrays(
        ds, td, val_times, tgt, feature_specs, val_samples, rng, patch_size, train_medians
    )
    train_mask = np.isfinite(X_train).all(axis=(1, 2, 3)) & np.isfinite(y_train)
    val_mask = np.isfinite(X_val).all(axis=(1, 2, 3)) & np.isfinite(y_val)
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_val, y_val = X_val[val_mask], y_val[val_mask]
    print("finite_samples train", X_train.shape[0], "val", X_val.shape[0])
    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise RuntimeError("No finite samples after filtering.")
    print(X_train.shape, X_val.shape)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    print(f"model_config device={device} batch_size={batch_size} patch_size={patch_size} \nsamples_per_epoch={samples_per_epoch} val_frac={val_frac} lr={lr} max_epochs={max_epochs} \npatience={patience} features={len(feature_specs)} train={len(train_ds)} val={len(val_ds)}\n")

    model = ShallowCNN(len(feature_specs)).to(device)
    center_idx = patch_size // 2
    model, best_val = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        max_epochs=max_epochs,
        lr=lr,
        patience=patience,
        center_idx=center_idx,
    )

    rows = []
    pad = 3
    fig_time = str(pd.Timestamp(test_times[-1]).date())
    for t in test_times:
        y_true, y_pred = predict_full_map_cnn(
            model,
            ds,
            td,
            t,
            tgt,
            feature_specs,
            tile_size=tile_size,
            pad=pad,
            device=device,
            train_medians=train_medians,
        )
        m = np.isfinite(y_true) & np.isfinite(y_pred)
        met = compute_all(y_true, y_pred, mask=m, ratio=33.3333333333, channel_axis=None)
        rows.append({"time": str(pd.Timestamp(t).date()), **met, "n_valid": int(np.sum(m))})
        if str(pd.Timestamp(t).date()) == fig_time:
            save_prediction_figures(y_true, y_pred, fig_time, FIG_DIR)
        del y_true, y_pred, m
        gc.collect()

    df = pd.DataFrame(rows)
    metrics_csv = OUT_DIR / "shallow_cnn_simple_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    save_metric_plots(df, FIG_DIR)

    model_path = MODELS_DIR / "shallow_cnn_simple.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "train_medians": train_medians,
            "features": [fs.name for fs in feature_specs],
            "best_val": best_val,
        },
        model_path,
    )

    cfg = {
        "dataset_key": dataset_key,
        "start": start,
        "end": end,
        "target": tgt,
        "features": [fs.name for fs in feature_specs],
        "feature_vars": list(base_features) + list(era5_feats),
        "median_samples": median_samples,
        "samples_per_epoch": samples_per_epoch,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "val_frac": val_frac,
        "patience": patience,
        "lr": lr,
        "device": str(device),
        "best_val": float(best_val),
        "metrics_csv": str(metrics_csv),
    }
    with (OUT_DIR / "shallow_cnn_simple_run.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("Shallow CNN simple (daily Landsat/VIIRS/MODIS + monthly others)")
    p.add_argument("--dataset", default="madurai_30m", choices=["madurai", "madurai_30m", "madurai_alphaearth_30m"])
    p.add_argument("--start", default=None, help="e.g., 2019-01-01")
    p.add_argument("--end", default=None, help="e.g., 2020-12-31")
    p.add_argument("--target", default=None, help="target variable name")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--median-samples", type=int, default=10_000)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--samples-per-epoch", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=2000)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tile-size", type=int, default=128)
    p.add_argument("--dry-run-day", default=None, help="single date for dry run (YYYY-MM-DD or 'random')")
    args = p.parse_args()

    run(
        dataset_key=args.dataset,
        start=args.start,
        end=args.end,
        target=args.target,
        test_frac=args.test_frac,
        median_samples=args.median_samples,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        val_frac=args.val_frac,
        patience=args.patience,
        lr=args.lr,
        tile_size=args.tile_size,
        dry_run_day=args.dry_run_day,
    )
