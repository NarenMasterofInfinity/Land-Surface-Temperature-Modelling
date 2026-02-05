from __future__ import annotations

import json
import logging
import resource
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from helper import (
    detect_time_dim,
    list_vars_from_store,
    make_madurai_data,
    open_zarr,
    select_vars,
    time_slice,
)
from helper.metrics_image import compute_all


PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project").resolve()
OUT_DIR = PROJECT_ROOT / "metrics" / "deep_baselines" / "shallow_cnn_plus_era5"
LOGS_DIR = PROJECT_ROOT / "logs" / "new"
MODELS_DIR = PROJECT_ROOT / "models"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MAX_PRECOMPUTE_GB = 0.5

S2_INDEX_TOKENS = (
    "ndvi", "evi", "savi", "msavi",
    "ndwi_water", "ndwi_moisture", "ndmi", "mndwi",
    "ndbi", "ui", "bsi", "ibi", "albedo_proxy",
)
S1_TOKENS = ("sentinel-1", "sentinel_1", "sentinel1", "/s1/", "s1_", "_s1")
DEM_TOKENS = ("dem", "elevation", "srtm")
THERMAL_TOKENS = ("modis", "viirs")
ERA5_TOKENS = ("era5", "ecmwf")
CLOUD_TOKENS = ("qc", "quality", "mask", "cloud", "flag", "clear_count", "total_count", "cloud_frac")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    var: str
    band: Optional[int] = None


@dataclass
class SampleStats:
    X: np.ndarray
    y: np.ndarray
    missing_counts: np.ndarray


def setup_logging(dataset_key: str, *, start: Optional[str], end: Optional[str]) -> logging.Logger:
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    tag = dataset_key.replace("/", "_")
    range_tag = f"{start or 'na'}_{end or 'na'}".replace(":", "_")
    log_path = LOGS_DIR / f"shallow_cnn_{tag}_{range_tag}_{ts}.log"

    logger = logging.getLogger("shallow_cnn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)sZ | %(levelname)s | %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Shallow CNN run start | dataset=%s | start=%s | end=%s | utc=%s",
                dataset_key, start, end, pd.Timestamp.utcnow().isoformat())
    logger.info("Log file: %s", log_path)
    return logger


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024.0


def _log_mem(logger: logging.Logger, tag: str, device: Optional[torch.device] = None) -> None:
    rss_mb = _rss_mb()
    logger.info("mem_rss_mb=%0.1f stage=%s", rss_mb, tag)
    if device is not None and device.type == "cuda":
        logger.info(
            "cuda_mem_%s alloc=%.2f reserved=%.2f",
            tag,
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
        )


def _is_cloud_var(name: str) -> bool:
    name = name.lower()
    return any(tok in name for tok in CLOUD_TOKENS)


def _is_worldcover(name: str) -> bool:
    name = name.lower()
    return ("worldcover" in name) or ("dynamic_world" in name)


def _is_era5_var(name: str) -> bool:
    return any(tok in name.lower() for tok in ERA5_TOKENS)


def _is_s1_var(name: str) -> bool:
    lower = name.lower()
    if any(tok in lower for tok in S1_TOKENS):
        return True
    if lower.endswith("_vv") or lower.endswith("_vh") or "/vv" in lower or "/vh" in lower:
        return True
    return False


def _is_s2_index_var(name: str) -> bool:
    return any(tok in name.lower() for tok in S2_INDEX_TOKENS)


def _is_dem_var(name: str) -> bool:
    return any(tok in name.lower() for tok in DEM_TOKENS)


def _is_thermal_var(name: str) -> bool:
    lower = name.lower()
    if "landsat" in lower:
        return False
    return any(tok in lower for tok in THERMAL_TOKENS) and ("lst" in lower or "temp" in lower)


def _is_landsat_var(name: str) -> bool:
    return "landsat" in name.lower()


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


def _select_base_features_from_names(
    vars_list: Sequence[str],
    target: str,
    user_features: Optional[Sequence[str]],
) -> List[str]:
    if user_features:
        missing = [v for v in user_features if v not in vars_list]
        if missing:
            raise KeyError(f"Some --features not found: {missing}")
        feats = [
            v for v in user_features
            if not _is_era5_var(v) and not _is_cloud_var(v) and not _is_worldcover(v)
            and not _is_landsat_var(v)
        ]
        if not feats:
            raise ValueError("No usable base features after filtering clouds/WorldCover/ERA5.")
        return feats

    feats = []
    for v in vars_list:
        if v == target:
            continue
        if _is_cloud_var(v) or _is_worldcover(v):
            continue
        if _is_landsat_var(v):
            continue
        if _is_era5_var(v):
            continue
        if _is_s1_var(v) or _is_s2_index_var(v) or _is_dem_var(v) or _is_thermal_var(v):
            feats.append(v)
    if not feats:
        raise ValueError("No base feature variables found. Provide --features explicitly.")
    return feats


def _select_era5_features_from_names(vars_list: Sequence[str]) -> List[str]:
    feats = [v for v in vars_list if _is_era5_var(v)]
    if not feats:
        raise RuntimeError("No ERA5 features detected in the dataset.")
    return feats


def _zarr_path_for_key(md, key: str) -> Path:
    if key == "madurai":
        return Path(md.paths.madurai_zarr)
    if key == "madurai_30m":
        return Path(md.paths.madurai_30m_zarr)
    if key == "madurai_alphaearth_30m":
        return Path(md.paths.madurai_alphaearth_30m_zarr)
    raise ValueError(f"Unknown dataset key: {key}")


def _load_subset_low_mem(
    md,
    key: str,
    *,
    vars_include: Optional[Sequence[str]],
    start: Optional[str],
    end: Optional[str],
    logger: logging.Logger,
) -> xr.Dataset:
    path = _zarr_path_for_key(md, key)
    ds = open_zarr(path, chunks=None, consolidated=False, decode_cf=False, mask_and_scale=False)
    ds = select_vars(ds, include=vars_include, strict=False)
    ds = time_slice(ds, start=start, end=end)
    logger.info("load_subset_mode=open_zarr_nocache key=%s", key)
    return ds


def _apply_common_dates(times: pd.DatetimeIndex, logger: logging.Logger) -> pd.DatetimeIndex:
    common_dates_path = PROJECT_ROOT / "common_dates.csv"
    if not common_dates_path.exists():
        raise FileNotFoundError("common_dates.csv not found; cannot enforce common-date filtering.")
    common_df = pd.read_csv(common_dates_path)
    if "landsat_date" not in common_df.columns:
        raise KeyError("common_dates.csv missing 'landsat_date' column")
    common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna()
    common_dates = pd.DatetimeIndex(common_dates).sort_values()
    filtered = times[times.isin(common_dates)]
    logger.info("common_dates_filtered=%d", len(filtered))
    if len(filtered) == 0:
        raise RuntimeError("No dates left after common_dates.csv filtering.")
    return filtered


def _infer_xy_dims(da: xr.DataArray) -> Tuple[str, str]:
    if "y" in da.dims and "x" in da.dims:
        return "y", "x"
    return da.dims[-2], da.dims[-1]


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


def _load_feature_patch(
    dts: xr.Dataset,
    spec: FeatureSpec,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> np.ndarray:
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
) -> SampleStats:
    X = np.empty((n_samples, len(features)), dtype=np.float32)
    y = np.empty((n_samples,), dtype=np.float32)
    missing = np.zeros((len(features),), dtype=np.int64)
    count = 0
    tries = 0
    times_list = list(times)
    while count < n_samples and tries < n_samples * max_tries:
        tries += 1
        t = times_list[int(rng.integers(0, len(times_list)))]
        dts = _select_time(ds, td, t)
        da_tgt = dts[target]
        y_dim, x_dim = _infer_xy_dims(da_tgt)
        H, W = int(da_tgt.sizes[y_dim]), int(da_tgt.sizes[x_dim])
        if H < patch_size or W < patch_size:
            raise ValueError(f"Patch size {patch_size} larger than spatial dims {(H, W)}.")
        r0, r1 = _center_range(H, patch_size)
        c0, c1 = _center_range(W, patch_size)
        r = int(rng.integers(r0, r1 + 1))
        c = int(rng.integers(c0, c1 + 1))
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
        raise RuntimeError("No valid training samples found (target all NaN/Inf).")
    return SampleStats(X=X[:count], y=y[:count], missing_counts=missing)


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
    *,
    return_target_patch: bool = False,
    max_tries: int = 50,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if n_samples <= 0:
        empty_x = np.empty((0, len(features), patch_size, patch_size), dtype=np.float16)
        empty_y = np.empty((0,), dtype=np.float32)
        empty_tp = np.empty((0, patch_size, patch_size), dtype=np.float32) if return_target_patch else None
        return empty_x, empty_y, empty_tp

    X = np.empty((n_samples, len(features), patch_size, patch_size), dtype=np.float16)
    y = np.empty((n_samples,), dtype=np.float32)
    y_patch = np.empty((n_samples, patch_size, patch_size), dtype=np.float32) if return_target_patch else None

    count = 0
    tries = 0
    times_list = list(times)
    while count < n_samples and tries < n_samples * max_tries:
        tries += 1
        t = times_list[int(rng.integers(0, len(times_list)))]
        dts = _select_time(ds, td, t)
        da_tgt = dts[target]
        y_dim, x_dim = _infer_xy_dims(da_tgt)
        H, W = int(da_tgt.sizes[y_dim]), int(da_tgt.sizes[x_dim])
        if H < patch_size or W < patch_size:
            raise ValueError(f"Patch size {patch_size} larger than spatial dims {(H, W)}.")
        r0, r1 = _center_range(H, patch_size)
        c0, c1 = _center_range(W, patch_size)
        r = int(rng.integers(r0, r1 + 1))
        c = int(rng.integers(c0, c1 + 1))
        r0, r1 = _patch_bounds(r, patch_size)
        c0, c1 = _patch_bounds(c, patch_size)
        tgt_patch = _load_feature_patch(dts, FeatureSpec(target, target), r0, r1, c0, c1)
        center = tgt_patch[patch_size // 2, patch_size // 2]
        if not np.isfinite(center):
            continue
        for j, spec in enumerate(features):
            patch = _load_feature_patch(dts, spec, r0, r1, c0, c1)
            bad = ~np.isfinite(patch)
            if bad.any():
                patch[bad] = train_medians[j]
            X[count, j] = patch.astype(np.float16, copy=False)
        y[count] = float(center)
        if return_target_patch and y_patch is not None:
            y_patch[count] = tgt_patch.astype(np.float32, copy=False)
        count += 1
    if count == 0:
        raise RuntimeError("No valid training samples found (target all NaN/Inf).")
    return X[:count], y[:count], (y_patch[:count] if y_patch is not None else None)


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
    logger: logging.Logger,
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
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            pred = out[:, 0, center_idx, center_idx]
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.float32)
                out = model(xb)
                pred = out[:, 0, center_idx, center_idx]
                loss = loss_fn(pred, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        logger.info("epoch=%d train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)

        if np.isfinite(val_loss) and val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("early_stopping epoch=%d best_val=%.6f", epoch, best)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def evaluate_on_patches(
    model: nn.Module,
    X: np.ndarray,
    y_patches: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, List[float]] = {"rmse": [], "ssim": [], "psnr": [], "sam": [], "cc": [], "ergas": []}
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_patches))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            out = model(xb).cpu().numpy()[:, 0]
            yb_np = yb.numpy()
            for i in range(out.shape[0]):
                met = compute_all(yb_np[i], out[i], mask=None, ratio=33.3333333333, channel_axis=None)
                for k, v in met.items():
                    metrics[k].append(v)
    return {k: float(np.nanmean(v)) if v else float("nan") for k, v in metrics.items()}


def run_variant(
    *,
    ds: xr.Dataset,
    dataset_key: str,
    td: str,
    times: Sequence[pd.Timestamp],
    target: str,
    feature_vars: Sequence[str],
    out_dir: Path,
    start: Optional[str],
    end: Optional[str],
    test_frac: float,
    median_samples: int,
    patch_size: int,
    samples_per_epoch: int,
    batch_size: int,
    max_epochs: int,
    val_frac: float,
    patience: int,
    lr: float,
    logger: logging.Logger,
    device: torch.device,
) -> None:
    feature_specs = _expand_feature_specs(ds, feature_vars, td)
    logger.info("features_total=%d", len(feature_specs))
    if not feature_specs:
        raise RuntimeError("No feature specs.")

    times_sorted = list(pd.DatetimeIndex(times).sort_values())
    n_test = max(1, int(round(len(times_sorted) * test_frac)))
    train_times = times_sorted[:-n_test]
    test_times = times_sorted[-n_test:]
    if not train_times:
        raise RuntimeError("Train split empty; reduce --test-frac.")

    train_times, val_times = _split_train_val_dates(train_times, val_frac)
    logger.info("dates total=%d train=%d val=%d test=%d", len(times_sorted), len(train_times), len(val_times), len(test_times))

    rng = np.random.default_rng(RANDOM_SEED)
    stats = _sample_center_pixels(ds, td, train_times, target, feature_specs, median_samples, rng, patch_size)
    missing_pct = stats.missing_counts / max(1, stats.X.shape[0])
    train_medians, keep_mask = _compute_train_medians(stats.X)
    if not np.all(keep_mask):
        dropped = [feature_specs[i].name for i in range(len(feature_specs)) if not keep_mask[i]]
        logger.info("dropped_all_nan_features=%s", dropped)
        feature_specs = [fs for i, fs in enumerate(feature_specs) if keep_mask[i]]
        train_medians = train_medians[keep_mask]
        missing_pct = missing_pct[keep_mask]

    for fs, pct in zip(feature_specs, missing_pct):
        logger.info("missing_pct[%s]=%.4f", fs.name, float(pct))

    train_samples = samples_per_epoch
    val_samples = max(1, samples_per_epoch // 4)
    train_samples, val_samples, scale = _cap_patch_samples(
        train_samples, val_samples, len(feature_specs), patch_size, MAX_PRECOMPUTE_GB
    )
    if scale < 1.0:
        logger.info("precompute_capped train=%d val=%d cap_gb=%.2f", train_samples, val_samples, MAX_PRECOMPUTE_GB)

    X_train, y_train, _ = _sample_patch_arrays(
        ds,
        td,
        train_times,
        target,
        feature_specs,
        train_samples,
        rng,
        patch_size,
        train_medians,
        return_target_patch=False,
    )
    X_val, y_val, _ = _sample_patch_arrays(
        ds,
        td,
        val_times,
        target,
        feature_specs,
        val_samples,
        rng,
        patch_size,
        train_medians,
        return_target_patch=False,
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    center_idx = patch_size // 2
    model = ShallowCNN(len(feature_specs)).to(device)
    model, best_val = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        max_epochs=max_epochs,
        lr=lr,
        patience=patience,
        logger=logger,
        center_idx=center_idx,
    )
    logger.info("best_val=%.6f", best_val)

    test_samples = max(100, min(2000, max(1, samples_per_epoch // 4)))
    X_test, _, y_test_patches = _sample_patch_arrays(
        ds,
        td,
        test_times,
        target,
        feature_specs,
        test_samples,
        rng,
        patch_size,
        train_medians,
        return_target_patch=True,
    )
    if y_test_patches is None:
        raise RuntimeError("Test patches missing.")

    metrics = evaluate_on_patches(model, X_test, y_test_patches, device=device, batch_size=batch_size)
    metrics["n_test_patches"] = int(X_test.shape[0])

    out_csv = out_dir / "shallow_cnn_plus_era5_metrics.csv"
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)

    try:
        model_path = MODELS_DIR / "shallow_cnn_plus_era5.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "train_medians": train_medians,
                "features": [fs.name for fs in feature_specs],
                "best_val": best_val,
            },
            model_path,
        )
        logger.info("saved_model=%s", model_path)
    except Exception as exc:
        logger.warning("model save failed: %s", exc)

    cfg = {
        "dataset_key": dataset_key,
        "start": start,
        "end": end,
        "time_dim": td,
        "target": target,
        "features": [fs.name for fs in feature_specs],
        "feature_vars": list(feature_vars),
        "missing_pct": {fs.name: float(pct) for fs, pct in zip(feature_specs, missing_pct)},
        "months_total": len(times_sorted),
        "train_months": [str(pd.Timestamp(t).date()) for t in train_times],
        "val_months": [str(pd.Timestamp(t).date()) for t in val_times],
        "test_months": [str(pd.Timestamp(t).date()) for t in test_times],
        "test_frac": test_frac,
        "patch_size": patch_size,
        "median_samples": median_samples,
        "samples_per_epoch": samples_per_epoch,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "val_frac": val_frac,
        "patience": patience,
        "lr": lr,
        "device": str(device),
        "metrics_csv": str(out_csv),
    }
    with (out_dir / "run_config_and_summary.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    logger.info("DONE metrics_csv=%s", out_csv)


def main(
    *,
    dataset_key: str,
    start: Optional[str],
    end: Optional[str],
    target: Optional[str],
    features: Optional[Sequence[str]],
    test_frac: float,
    median_samples: int,
    patch_size: int,
    samples_per_epoch: int,
    batch_size: int,
    max_epochs: int,
    val_frac: float,
    patience: int,
    lr: float,
) -> None:
    logger = setup_logging(dataset_key, start=start, end=end)
    md = make_madurai_data(chunks=None, consolidated=False)

    vars_all = list_vars_from_store(_zarr_path_for_key(md, dataset_key))
    logger.info("vars_total=%d", len(vars_all))

    tgt = "products/landsat/band_01"
    if tgt not in vars_all:
        candidates = [v for v in vars_all if "landsat" in v.lower() and "band_01" in v.lower()]
        if len(candidates) == 1:
            tgt = candidates[0]
            logger.info("Using landsat band_01 target found in dataset: %s", tgt)
        else:
            raise KeyError(
                f"Target '{tgt}' not found in dataset. Candidates: {candidates[:10]} | "
                f"Available vars: {vars_all[:20]}"
            )
    if target and target != tgt:
        logger.info("Ignoring --target=%s; hardcoding target to %s", target, tgt)

    base_features = _select_base_features_from_names(vars_all, tgt, features)
    era5_feats = _select_era5_features_from_names(vars_all)
    vars_include = list(dict.fromkeys(list(base_features) + list(era5_feats) + [tgt]))
    logger.info("vars_include=%d base=%d era5=%d", len(vars_include), len(base_features), len(era5_feats))

    ds = _load_subset_low_mem(md, dataset_key, vars_include=vars_include, start=start, end=end, logger=logger)
    _log_mem(logger, "after_load_subset")

    td = detect_time_dim(ds)
    if td is None:
        raise RuntimeError("Dataset has no time dimension; shallow CNN needs time to split train/test.")
    times = pd.DatetimeIndex(pd.to_datetime(ds[td].values, errors="coerce"))
    times = times[~times.isna()]
    times = _apply_common_dates(times, logger)
    total_times = int(ds[td].sizes.get(td, len(ds[td])))
    logger.info("using_common_dates=%d total_dates=%d", len(times), total_times)
    _log_mem(logger, "after_common_dates")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    run_variant(
        ds=ds,
        dataset_key=dataset_key,
        td=td,
        times=times,
        target=tgt,
        feature_vars=list(base_features) + list(era5_feats),
        out_dir=OUT_DIR,
        start=start,
        end=end,
        test_frac=test_frac,
        median_samples=median_samples,
        patch_size=patch_size,
        samples_per_epoch=samples_per_epoch,
        batch_size=batch_size,
        max_epochs=max_epochs,
        val_frac=val_frac,
        patience=patience,
        lr=lr,
        logger=logger,
        device=device,
    )


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser("Shallow CNN baseline (ERA5 included, simplified)")
    p.add_argument("--dataset", default="madurai_30m", choices=["madurai", "madurai_30m", "madurai_alphaearth_30m"])
    p.add_argument("--start", default=None, help="e.g., 2019-01-01")
    p.add_argument("--end", default=None, help="e.g., 2020-12-31")
    p.add_argument("--target", default=None, help="target variable name (recommended to set once)")
    p.add_argument("--features", default=None, nargs="+", help="base feature vars (optional)")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--median-samples", type=int, default=10_000)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--samples-per-epoch", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=5000)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    main(
        dataset_key=args.dataset,
        start=args.start,
        end=args.end,
        target=args.target,
        features=args.features,
        test_frac=args.test_frac,
        median_samples=args.median_samples,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        val_frac=args.val_frac,
        patience=args.patience,
        lr=args.lr,
    )
