
from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zarr
from matplotlib.path import Path as MplPath
import joblib
import resource
import sys

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# helper module (installed system-wide)
from helper import make_madurai_data, load_subset  # expects your helper/__init__.py exports these

from helper.metrics_image import compute_all
from helper.split_utils import load_or_create_splits


PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project").resolve()
COMMON_DATES = PROJECT_ROOT / "common_dates.csv"
SPLITS_PATH = PROJECT_ROOT / "metrics" / "common_date_splits.csv"
OUT_DIR = PROJECT_ROOT / "metrics" / "linear_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = PROJECT_ROOT / "logs" / "new"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ROI polygon (lon, lat) for Madurai
ROI_COORDS = [[
    [78.19018990852709, 9.878220339041174],
    [78.18528420047912, 9.882890316485547],
    [78.09810549679275, 9.894348798281609],
    [78.06157864432538, 9.932892703854442],
    [78.02656656110504, 9.94245002321148],
    [78.00260828167308, 9.965541799781803],
    [77.98788548795139, 9.97202781239784],
    [77.98759869039448, 9.972273355756196],
    [77.98926344201777, 9.974277652950137],
    [78.17034512986598, 10.09203200706642],
    [78.33409611393074, 10.299897884952559],
    [78.33446180425422, 10.301538866748283],
    [78.3424560260113, 10.300913203358672],
    [78.3620537189306, 10.301140183092196],
    [78.37500586343197, 10.310994022414183],
    [78.38801590170993, 10.315994815585963],
    [78.39017505117465, 10.315419769552259],
    [78.38542553295602, 10.314175493832316],
    [78.38810333700448, 10.269213359046619],
    [78.41273304358974, 10.212526305038399],
    [78.39622781746723, 10.177726046483423],
    [78.45775072031367, 10.100259541710393],
    [78.45817143938241, 9.967811808760526],
    [78.19018990852709, 9.878220339041174],
]]

RANDOM_SEED = 42


# ----------------------------
# Robust dataset introspection
# ----------------------------

def detect_time_dim(ds: xr.Dataset) -> Optional[str]:
    for td in ("time", "date", "t", "datetime"):
        if td in ds.coords or td in ds.dims:
            return td
    for c in ds.coords:
        try:
            if np.issubdtype(ds[c].dtype, np.datetime64):
                return c
        except Exception:
            pass
    return None


def cadence_guess(time_values: pd.DatetimeIndex) -> str:
    if len(time_values) < 2:
        return "unknown"
    deltas = np.diff(time_values.values.astype("datetime64[D]").astype("int64"))
    if len(deltas) == 0:
        return "unknown"
    med = int(np.median(deltas))
    if med <= 2:
        return "daily-ish"
    if med <= 10:
        return "weekly-ish"
    if med <= 20:
        return "biweekly-ish"
    if med <= 45:
        return "monthly-ish"
    return "sparse"


def pick_target_var(ds: xr.Dataset, user_target: Optional[str]) -> str:
    if user_target:
        if "band_01" not in user_target.lower():
            raise ValueError("Only Landsat band_01 is supported as target.")
        if user_target not in ds.data_vars:
            raise KeyError(f"--target '{user_target}' not in dataset vars.")
        return user_target

    preferred = [
        "labels_30m/landsat/band_01",
        "products/landsat/band_01",
    ]
    for v in preferred:
        if v in ds.data_vars:
            return v

    band01 = [v for v in ds.data_vars if "landsat" in v.lower() and "band_01" in v.lower()]
    if band01:
        return sorted(band01)[0]

    raise ValueError(
        "Could not auto-detect Landsat band_01 target. Provide --target explicitly."
    )


def default_feature_vars(ds: xr.Dataset, target: str, user_features: Optional[Sequence[str]]) -> List[str]:
    if user_features:
        missing = [v for v in user_features if v not in ds.data_vars]
        if missing:
            raise KeyError(f"Some --features not found: {missing}")
        return list(user_features)

    # Use all numeric vars except obvious non-features
    feats = []
    for v in ds.data_vars:
        if v == target:
            continue
        name = v.lower()
        if any(k in name for k in ("qc", "quality", "mask", "flag", "cloud")):
            continue
        if v in {
            "labels_30m/landsat/band_02",
            "static_30m/worldcover/band_01",
            "static_30m/dynamic_world/band_01",
        }:
            continue
        feats.append(v)
    if not feats:
        raise ValueError("No feature variables found. Provide --features explicitly.")
    return feats


# ----------------------------
# Coarse anchors (MODIS/VIIRS)
# ----------------------------

@dataclass
class CoarseAnchors:
    ds: xr.Dataset
    time_dim: str
    times: pd.DatetimeIndex
    modis_day_var: Optional[str]
    viirs_day_var: Optional[str]
    modis_mask_vars: List[str]
    viirs_mask_vars: List[str]
    modis_qc_var: Optional[str]
    viirs_qc_var: Optional[str]
    row_map: np.ndarray
    col_map: np.ndarray
    row_float: np.ndarray
    col_float: np.ndarray
    time_index: Dict[pd.Timestamp, int]
    modis_month_median: Dict[str, np.ndarray]
    viirs_month_median: Dict[str, np.ndarray]

    @property
    def feature_names(self) -> List[str]:
        names = []
        if self.modis_day_var:
            names.extend([
                "modis_cell",
                "modis_up",
                "modis_mean3",
                "modis_std3",
                "modis_med3",
                "modis_grad",
                "modis_anom",
                "modis_valid",
                "modis_imputed",
            ])
            if self.modis_qc_var:
                names.append("modis_qc")
        if self.viirs_day_var:
            names.extend([
                "viirs_cell",
                "viirs_up",
                "viirs_mean3",
                "viirs_std3",
                "viirs_med3",
                "viirs_grad",
                "viirs_anom",
                "viirs_valid",
                "viirs_imputed",
            ])
            if self.viirs_qc_var:
                names.append("viirs_qc")
        return names


def _choose_day_var(vars_list: Sequence[str], prefix: str) -> Optional[str]:
    cand = [v for v in vars_list if v.startswith(prefix)]
    if not cand:
        return None
    if prefix == "products/viirs/":
        for v in cand:
            if v.endswith("LST_1KM_DAY_RAW"):
                return v
    if prefix == "products/modis/":
        for v in cand:
            if v.endswith("band_01"):
                return v
    def score(name: str) -> int:
        n = name.lower()
        s = 0
        if "lst" in n or "temp" in n or "temperature" in n:
            s += 4
        if "day" in n:
            s += 3
        if "night" in n:
            s -= 4
        if "qc" in n or "quality" in n or "mask" in n or "valid" in n or "cloud" in n:
            s -= 6
        return s
    ranked = sorted(cand, key=score, reverse=True)
    return ranked[0]


def _choose_mask_vars(vars_list: Sequence[str], prefix: str) -> List[str]:
    keys = ("qc", "quality", "mask", "cloud", "valid")
    out = []
    for v in vars_list:
        if not v.startswith(prefix):
            continue
        n = v.lower()
        if any(k in n for k in keys):
            out.append(v)
    return out


def _choose_qc_var(vars_list: Sequence[str], prefix: str) -> Optional[str]:
    keys = ("qc", "quality")
    for v in vars_list:
        if v.startswith(prefix) and any(k in v.lower() for k in keys):
            return v
    return None


def _apply_mask(data: np.ndarray, masks: Sequence[Tuple[str, np.ndarray]]) -> np.ndarray:
    if not masks:
        return data
    mask = np.ones_like(data, dtype=bool)
    for name, m in masks:
        lname = name.lower()
        if "viirs" in lname and "cloud" in lname:
            mask &= np.isfinite(m) & (m <= 1)
        elif "modis" in lname and "valid" in lname:
            mask &= np.isfinite(m) & (m == 1)
        elif "cloud" in lname:
            mask &= np.isfinite(m) & (m == 0)
        else:
            mask &= np.isfinite(m) & (m > 0)
    return np.where(mask, data, np.nan)


def _is_landsat_var(var_name: str) -> bool:
    return "landsat" in var_name.lower()


def _is_viirs_var(var_name: str) -> bool:
    return "viirs" in var_name.lower()


def _is_modis_var(var_name: str) -> bool:
    return "modis" in var_name.lower()


def _is_lst_source(var_name: str) -> bool:
    name = var_name.lower()
    if "era5" in name or "ecmwf" in name:
        return False
    return (
        _is_landsat_var(name)
        or _is_viirs_var(name)
        or _is_modis_var(name)
        or "lst" in name
        or "temperature" in name
        or "temp" in name
        or name.endswith("band_01")
    )


def _mask_lst_values(data: np.ndarray, var_name: str) -> np.ndarray:
    if not _is_lst_source(var_name):
        return data
    out = np.where(np.isfinite(data) & (data > 0), data, np.nan)
    if _is_landsat_var(var_name):
        out = np.where(np.isclose(out, 149.0), np.nan, out)
    return out


def _convert_lst_units(data: np.ndarray, var_name: str) -> np.ndarray:
    if data.size == 0:
        return data
    finite = np.isfinite(data)
    if not np.any(finite):
        return data
    if _is_modis_var(var_name):
        return data  # MODIS is already in °C
    if _is_landsat_var(var_name) or _is_viirs_var(var_name):
        med = float(np.nanmedian(data[finite]))
        if med > 200:
            return data - 273.15
    return data


def build_roi_mask(dataset_key: str, shape: Tuple[int, int], logger: logging.Logger) -> Optional[np.ndarray]:
    try:
        zarr_path = PROJECT_ROOT / f"{dataset_key}.zarr"
        if not zarr_path.exists():
            return None
        root = zarr.open_group(str(zarr_path), mode="r")
        if "grid" not in root:
            return None
        g = root["grid"]
        transform = g.attrs.get("transform")
        if not transform or len(transform) < 6:
            return None
        crs_str = g.attrs.get("crs")
        base_roi = ROI_COORDS[0]
        roi_coords = base_roi
        transformed = False
        if crs_str and str(crs_str).upper() not in ("EPSG:4326", "WGS84"):
            try:
                from pyproj import CRS, Transformer
            except Exception as exc:
                logger.warning("ROI mask skipped: pyproj unavailable (%s)", exc)
                return None
            try:
                src = CRS.from_epsg(4326)
                dst = CRS.from_string(str(crs_str))
                transformer = Transformer.from_crs(src, dst, always_xy=True)
                roi_coords = [transformer.transform(lon, lat) for lon, lat in roi_coords]
                transformed = True
            except Exception as exc:
                logger.warning("ROI mask skipped: failed CRS transform (%s)", exc)
                return None
        a, b, c, d, e, f = transform[:6]
        H, W = shape
        cols = np.arange(W, dtype=np.float64) + 0.5
        rows = np.arange(H, dtype=np.float64) + 0.5
        cc, rr = np.meshgrid(cols, rows)
        lon = a * cc + b * rr + c
        lat = d * cc + e * rr + f
        points = np.column_stack([lon.ravel(), lat.ravel()])
        poly = np.array(roi_coords, dtype=np.float64)
        path = MplPath(poly)
        mask = path.contains_points(points).reshape(H, W)
        if not np.any(mask) and transformed:
            poly = np.array(base_roi, dtype=np.float64)
            path = MplPath(poly)
            mask = path.contains_points(points).reshape(H, W)
        if not np.any(mask):
            logger.warning("ROI mask empty; skipping ROI crop.")
            return None
        return mask
    except Exception as exc:
        logger.warning("ROI mask generation failed: %s", exc)
        return None


def save_roi_figure(mask: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.imshow(mask.astype(np.uint8), cmap="gray")
    ax.set_title("ROI mask")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_index_map(fine_len: int, coarse_len: int) -> np.ndarray:
    # Nearest-neighbor style block mapping: every coarse cell is constant across its fine block.
    block = float(fine_len) / float(coarse_len)
    idx = np.floor(np.arange(fine_len) / block).astype(np.int64)
    idx = np.clip(idx, 0, coarse_len - 1)
    return idx


def _build_float_map(fine_len: int, coarse_len: int) -> np.ndarray:
    if fine_len <= 1:
        return np.zeros((fine_len,), dtype=np.float64)
    return np.linspace(0, coarse_len - 1, fine_len, dtype=np.float64)


def _bilinear_sample(arr: np.ndarray, r_f: np.ndarray, c_f: np.ndarray) -> np.ndarray:
    Hc, Wc = arr.shape
    r0 = np.floor(r_f).astype(np.int64)
    c0 = np.floor(c_f).astype(np.int64)
    r1 = np.clip(r0 + 1, 0, Hc - 1)
    c1 = np.clip(c0 + 1, 0, Wc - 1)
    fr = r_f - r0
    fc = c_f - c0
    v00 = arr[r0, c0]
    v01 = arr[r0, c1]
    v10 = arr[r1, c0]
    v11 = arr[r1, c1]
    w00 = (1 - fr) * (1 - fc)
    w01 = (1 - fr) * fc
    w10 = fr * (1 - fc)
    w11 = fr * fc
    vals = np.stack([v00, v01, v10, v11], axis=0)
    wts = np.stack([w00, w01, w10, w11], axis=0)
    valid = np.isfinite(vals)
    wts = np.where(valid, wts, 0.0)
    denom = np.sum(wts, axis=0)
    out = np.where(denom > 0, np.sum(wts * np.nan_to_num(vals), axis=0) / denom, np.nan)
    return out.astype(np.float32)


def _nearest_sample(arr: np.ndarray, r_f: np.ndarray, c_f: np.ndarray) -> np.ndarray:
    Hc, Wc = arr.shape
    r = np.clip(np.rint(r_f).astype(np.int64), 0, Hc - 1)
    c = np.clip(np.rint(c_f).astype(np.int64), 0, Wc - 1)
    return arr[r, c].astype(np.float32, copy=False)


def _coarse_stats(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Hc, Wc = arr.shape
    mean = np.full((Hc, Wc), np.nan, dtype=np.float32)
    std = np.full((Hc, Wc), np.nan, dtype=np.float32)
    med = np.full((Hc, Wc), np.nan, dtype=np.float32)
    grad = np.full((Hc, Wc), np.nan, dtype=np.float32)
    for r in range(Hc):
        r0 = max(0, r - 1)
        r1 = min(Hc, r + 2)
        for c in range(Wc):
            c0 = max(0, c - 1)
            c1 = min(Wc, c + 2)
            win = arr[r0:r1, c0:c1]
            if np.isfinite(win).any():
                mean[r, c] = float(np.nanmean(win))
                std[r, c] = float(np.nanstd(win))
                med[r, c] = float(np.nanmedian(win))
            left = arr[r, c - 1] if c - 1 >= 0 else np.nan
            right = arr[r, c + 1] if c + 1 < Wc else np.nan
            up = arr[r - 1, c] if r - 1 >= 0 else np.nan
            down = arr[r + 1, c] if r + 1 < Hc else np.nan
            if np.isfinite(left) and np.isfinite(right):
                dx = right - left
            else:
                dx = np.nan
            if np.isfinite(up) and np.isfinite(down):
                dy = down - up
            else:
                dy = np.nan
            if np.isfinite(dx) and np.isfinite(dy):
                grad[r, c] = float(np.sqrt(dx * dx + dy * dy))
    return mean, std, med, grad


def _build_monthly_medians(
    ds: xr.Dataset,
    td: str,
    times: pd.DatetimeIndex,
    var: Optional[str],
    masks: List[str],
) -> Dict[str, np.ndarray]:
    if var is None:
        return {}
    out: Dict[str, np.ndarray] = {}
    months = sorted({t.strftime("%Y-%m") for t in times})
    for mk in months:
        idx = times.strftime("%Y-%m") == mk
        if not np.any(idx):
            continue
        data = np.asarray(ds[var].isel({td: idx}).data, dtype=np.float32)
        if masks:
            keep = None
            for mv in masks:
                m = np.asarray(ds[mv].isel({td: idx}).data)
                if "viirs" in mv.lower() and "cloud" in mv.lower():
                    cond = np.isfinite(m) & (m <= 1)
                elif "modis" in mv.lower() and "valid" in mv.lower():
                    cond = np.isfinite(m) & (m == 1)
                elif "cloud" in mv.lower():
                    cond = np.isfinite(m) & (m == 0)
                else:
                    cond = np.isfinite(m) & (m > 0)
                keep = cond if keep is None else (keep & cond)
            data = np.where(keep, data, np.nan)
        data = _nanify_nodata(data)
        data = _mask_lst_values(data, var)
        data = _convert_lst_units(data, var)
        med = np.nanmedian(data, axis=0)
        out[mk] = med.astype(np.float32, copy=False)
    return out


def _build_time_index(coarse_times: pd.DatetimeIndex) -> Dict[pd.Timestamp, int]:
    return {pd.Timestamp(t): i for i, t in enumerate(coarse_times)}


def _nearest_time_index(coarse_times: pd.DatetimeIndex, t: pd.Timestamp) -> int:
    pos = int(np.searchsorted(coarse_times.values, np.datetime64(t), side="left"))
    if pos <= 0:
        return 0
    if pos >= len(coarse_times):
        return len(coarse_times) - 1
    before = coarse_times[pos - 1]
    after = coarse_times[pos]
    return pos if (after - t) <= (t - before) else pos - 1


def prepare_coarse_anchors(
    md,
    *,
    start: Optional[str],
    end: Optional[str],
    fine_shape: Tuple[int, int],
    logger: logging.Logger,
) -> Optional[CoarseAnchors]:
    try:
        all_vars = md.vars("madurai")
    except Exception as exc:
        logger.warning("coarse anchors disabled: failed to list madurai vars: %s", exc)
        return None

    modis_day = _choose_day_var(all_vars, "products/modis/")
    viirs_day = _choose_day_var(all_vars, "products/viirs/")
    modis_masks = _choose_mask_vars(all_vars, "products/modis/")
    viirs_masks = _choose_mask_vars(all_vars, "products/viirs/")
    modis_qc = _choose_qc_var(all_vars, "products/modis/")
    viirs_qc = _choose_qc_var(all_vars, "products/viirs/")

    include_vars = [v for v in [modis_day, viirs_day] if v] + modis_masks + viirs_masks
    if not include_vars:
        logger.warning("coarse anchors disabled: no modis/viirs vars found.")
        return None

    ds_c = load_subset(md, "madurai", vars_include=include_vars, start=start, end=end)
    td = detect_time_dim(ds_c)
    if td is None:
        logger.warning("coarse anchors disabled: madurai has no time dimension.")
        return None

    times = pd.DatetimeIndex(pd.to_datetime(ds_c[td].values)).sort_values()
    time_index = _build_time_index(times)
    Hf, Wf = fine_shape

    sample_var = modis_day or viirs_day
    if sample_var is None:
        logger.warning("coarse anchors disabled: no modis/viirs day var selected.")
        return None
    da = ds_c[sample_var]
    if "y" not in da.dims or "x" not in da.dims:
        logger.warning("coarse anchors disabled: expected y/x dims in %s", sample_var)
        return None
    Hc, Wc = da.sizes["y"], da.sizes["x"]
    row_map = _build_index_map(Hf, Hc)
    col_map = _build_index_map(Wf, Wc)
    row_float = _build_float_map(Hf, Hc)
    col_float = _build_float_map(Wf, Wc)

    logger.info(
        "coarse anchors: modis=%s viirs=%s modis_masks=%d viirs_masks=%d coarse_shape=%dx%d",
        modis_day,
        viirs_day,
        len(modis_masks),
        len(viirs_masks),
        Hc,
        Wc,
    )

    modis_month_med = _build_monthly_medians(ds_c, td, times, modis_day, modis_masks)
    viirs_month_med = _build_monthly_medians(ds_c, td, times, viirs_day, viirs_masks)

    return CoarseAnchors(
        ds=ds_c,
        time_dim=td,
        times=times,
        modis_day_var=modis_day,
        viirs_day_var=viirs_day,
        modis_mask_vars=modis_masks,
        viirs_mask_vars=viirs_masks,
        modis_qc_var=modis_qc,
        viirs_qc_var=viirs_qc,
        row_map=row_map,
        col_map=col_map,
        row_float=row_float,
        col_float=col_float,
        time_index=time_index,
        modis_month_median=modis_month_med,
        viirs_month_median=viirs_month_med,
    )

# ----------------------------
# Sampling / preprocessing
# ----------------------------

@dataclass
class SampleSpec:
    max_samples_per_time: int = 15_000
    min_samples_per_time: int = 3_000
    max_total_samples: int = 120_000
    tile_size: int = 256
    max_tiles_per_time: int = 12


def _finite_mask(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a)


def _infer_xy_dims(da: xr.DataArray) -> Tuple[str, str]:
    if "y" in da.dims and "x" in da.dims:
        return "y", "x"
    return da.dims[-2], da.dims[-1]


def _nanify_nodata(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    out = np.array(a, copy=True)
    out[np.isclose(out, -9999.0)] = np.nan
    return out


def _apply_valid_mask_data(data: np.ndarray, valid: Optional[np.ndarray]) -> np.ndarray:
    if valid is None:
        return data
    v = np.asarray(valid)
    v = np.squeeze(v)
    mask = np.isfinite(v) & (v > 0)
    if mask.ndim == data.ndim - 1:
        mask = np.broadcast_to(mask, data.shape)
    elif mask.ndim == data.ndim and mask.shape[0] == 1 and data.shape[0] > 1:
        mask = np.broadcast_to(mask, data.shape)
    if mask.shape != data.shape:
        return data
    return np.where(mask, data, np.nan)


def _valid_var_for(data_var: str, ds: xr.Dataset) -> Optional[str]:
    if data_var.endswith("/data"):
        candidate = f"{data_var.rsplit('/', 1)[0]}/valid"
        if candidate in ds.data_vars:
            return candidate
    if "/band_" in data_var:
        candidate = f"{data_var.split('/band_')[0]}/valid"
        if candidate in ds.data_vars:
            return candidate
    return None


def _infer_time_dim_for_var(da: xr.DataArray) -> Optional[str]:
    y_dim, x_dim = _infer_xy_dims(da)
    for d in da.dims:
        if d not in (y_dim, x_dim) and d not in ("band", "bands", "channel", "c"):
            return d
    return None


def _select_time_dataset(
    ds: xr.Dataset,
    td: str,
    t_val,
    time_index_map: Dict[pd.Timestamp, int],
) -> xr.Dataset:
    if td not in ds.dims:
        td = detect_time_dim(ds) or td
        if td not in ds.dims:
            return ds
    t_stamp = pd.Timestamp(t_val)
    if td in ds.coords:
        try:
            out = ds.sel({td: t_stamp})
            if td in out.dims:
                return out.squeeze(dim=td, drop=True)
            return out
        except Exception:
            pass
    idx = time_index_map.get(t_stamp)
    if idx is None:
        if td in ds.coords:
            times = pd.DatetimeIndex(pd.to_datetime(ds[td].values))
            if len(times) > 0:
                idx = int(np.argmin(np.abs(times - t_stamp)))
        if idx is None:
            idx = 0
    out = ds.isel({td: idx})
    if td in out.dims:
        return out.squeeze(dim=td, drop=True)
    return out


def _select_time_for_var(
    da: xr.DataArray,
    t_val,
    time_index_map: Optional[Dict[pd.Timestamp, int]] = None,
) -> xr.DataArray:
    tdim = _infer_time_dim_for_var(da)
    if tdim is None or tdim not in da.dims:
        return da
    t_stamp = pd.Timestamp(t_val)
    if tdim in da.coords:
        try:
            out = da.sel({tdim: t_stamp})
            return out.squeeze(tdim, drop=True)
        except Exception:
            pass
        times = pd.DatetimeIndex(pd.to_datetime(da[tdim].values))
        if len(times) > 0:
            idx = int(np.argmin(np.abs(times - t_stamp)))
            out = da.isel({tdim: idx})
            return out.squeeze(tdim, drop=True)
    if time_index_map:
        idx = time_index_map.get(t_stamp, 0)
        idx = int(np.clip(idx, 0, da.sizes[tdim] - 1))
        out = da.isel({tdim: idx})
        return out.squeeze(tdim, drop=True)
    return da.isel({tdim: 0}).squeeze(tdim, drop=True)


def _available_months_for_var(
    ds: xr.Dataset,
    *,
    data_var: str,
    valid_var: Optional[str],
) -> Optional[Tuple[set, str]]:
    da = ds[data_var]
    td_var = _infer_time_dim_for_var(da)
    if td_var is None:
        return None
    times = pd.DatetimeIndex(pd.to_datetime(da[td_var].values))
    if valid_var and valid_var in ds.data_vars:
        v = ds[valid_var] > 0
        reduce_dims = [d for d in v.dims if d != td_var]
        for d in reduce_dims:
            v = v.any(d)
        avail = np.asarray(v.data)
    else:
        data = da.where(da != -9999)
        v = xr.apply_ufunc(np.isfinite, data)
        reduce_dims = [d for d in da.dims if d != td_var]
        for d in reduce_dims:
            v = v.any(d)
        avail = np.asarray(v.data if hasattr(v, "data") else v)
    months = times.to_period("M")
    keep = months[avail]
    return set(str(m) for m in keep), td_var


def _available_days_for_var(
    ds: xr.Dataset,
    *,
    data_var: str,
    valid_var: Optional[str],
    time_dim: Optional[str] = None,
) -> Optional[Tuple[set, str]]:
    da = ds[data_var]
    td_var = time_dim or _infer_time_dim_for_var(da)
    if td_var is None:
        return None
    times = pd.DatetimeIndex(pd.to_datetime(da[td_var].values))
    if valid_var and valid_var in ds.data_vars:
        v = ds[valid_var] > 0
        reduce_dims = [d for d in v.dims if d != td_var]
        for d in reduce_dims:
            v = v.any(d)
        avail = np.asarray(v.data)
    else:
        data = da.where(da != -9999)
        v = xr.apply_ufunc(np.isfinite, data)
        reduce_dims = [d for d in da.dims if d != td_var]
        for d in reduce_dims:
            v = v.any(d)
        avail = np.asarray(v.data if hasattr(v, "data") else v)
    keep = times[avail]
    return set(keep.strftime("%Y-%m-%d")), td_var


def _available_days_for_anchor(
    ds: xr.Dataset,
    *,
    data_var: str,
    mask_vars: Sequence[str],
    time_dim: str,
) -> set:
    data = ds[data_var]
    if mask_vars:
        mask = None
        for mv in mask_vars:
            m = ds[mv]
            if "cloud" in mv.lower():
                ok = np.isfinite(m) & (m == 0)
            else:
                ok = np.isfinite(m) & (m > 0)
            mask = ok if mask is None else (mask & ok)
        data = data.where(mask)
    data = data.where(data != -9999)
    v = xr.apply_ufunc(np.isfinite, data)
    reduce_dims = [d for d in v.dims if d != time_dim]
    for d in reduce_dims:
        v = v.any(d)
    avail = np.asarray(v.data if hasattr(v, "data") else v)
    times = pd.DatetimeIndex(pd.to_datetime(ds[time_dim].values))
    keep = times[avail]
    return set(keep.strftime("%Y-%m-%d"))


def sample_xy_for_time(
    ds: xr.Dataset,
    td: str,
    t_val,
    target: str,
    features: Sequence[str],
    spec: SampleSpec,
    rng: np.random.Generator,
    *,
    anchors: Optional[CoarseAnchors] = None,
    valid_map: Optional[Dict[str, str]] = None,
    target_valid: Optional[str] = None,
    time_index_map: Optional[Dict[pd.Timestamp, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns X [N, F], y [N]
    """
    if time_index_map is None:
        time_index_map = {}
    dts = _select_time_dataset(ds, td, t_val, time_index_map)
    y_da = _select_time_for_var(dts[target], t_val, time_index_map)
    y_dim, x_dim = _infer_xy_dims(y_da)
    H = y_da.sizes[y_dim]
    W = y_da.sizes[x_dim]

    extra_feats = anchors.feature_names if anchors else []
    target_count = spec.max_samples_per_time
    X_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []

    # Prepare anchor arrays once per time
    anchor_modis = None
    anchor_viirs = None
    modis_qc = None
    viirs_qc = None
    modis_feats = None
    viirs_feats = None
    if anchors and extra_feats:
        t_idx = anchors.time_index.get(pd.Timestamp(t_val))
        if t_idx is None:
            t_idx = _nearest_time_index(anchors.times, pd.Timestamp(t_val))
        if anchors.modis_day_var:
            anchor_modis = np.asarray(
                anchors.ds[anchors.modis_day_var].isel({anchors.time_dim: t_idx}).data,
                dtype=np.float32,
            )
            masks = []
            if anchors.modis_mask_vars:
                for mv in anchors.modis_mask_vars:
                    m = np.asarray(anchors.ds[mv].isel({anchors.time_dim: t_idx}).data)
                    masks.append((mv, m))
            anchor_modis = _apply_mask(anchor_modis, masks)
            anchor_modis = _nanify_nodata(anchor_modis)
            anchor_modis = _mask_lst_values(anchor_modis, anchors.modis_day_var)
            anchor_modis = _convert_lst_units(anchor_modis, anchors.modis_day_var)
            if anchors.modis_qc_var:
                modis_qc = np.asarray(
                    anchors.ds[anchors.modis_qc_var].isel({anchors.time_dim: t_idx}).data,
                    dtype=np.float32,
                )
        if anchors.viirs_day_var:
            anchor_viirs = np.asarray(
                anchors.ds[anchors.viirs_day_var].isel({anchors.time_dim: t_idx}).data,
                dtype=np.float32,
            )
            masks = []
            if anchors.viirs_mask_vars:
                for mv in anchors.viirs_mask_vars:
                    m = np.asarray(anchors.ds[mv].isel({anchors.time_dim: t_idx}).data)
                    masks.append((mv, m))
            anchor_viirs = _apply_mask(anchor_viirs, masks)
            anchor_viirs = _nanify_nodata(anchor_viirs)
            anchor_viirs = _mask_lst_values(anchor_viirs, anchors.viirs_day_var)
            anchor_viirs = _convert_lst_units(anchor_viirs, anchors.viirs_day_var)
            if anchors.viirs_qc_var:
                viirs_qc = np.asarray(
                    anchors.ds[anchors.viirs_qc_var].isel({anchors.time_dim: t_idx}).data,
                    dtype=np.float32,
                )

        if anchor_modis is not None:
            mean3, std3, med3, grad = _coarse_stats(anchor_modis)
            valid = np.isfinite(anchor_modis).astype(np.float32)
            imputed = (~np.isfinite(anchor_modis)) & np.isfinite(med3)
            filled = np.where(np.isfinite(anchor_modis), anchor_modis, med3)
            mk = pd.Timestamp(t_val).strftime("%Y-%m")
            anom = (
                anchor_modis - anchors.modis_month_median.get(mk, np.nan)
                if anchors.modis_month_median
                else np.full_like(anchor_modis, np.nan)
            )
            modis_feats = {
                "cell": anchor_modis,
                "up": filled,
                "mean3": mean3,
                "std3": std3,
                "med3": med3,
                "grad": grad,
                "anom": anom,
                "valid": valid,
                "imputed": imputed.astype(np.float32),
                "qc": modis_qc,
            }

        if anchor_viirs is not None:
            mean3, std3, med3, grad = _coarse_stats(anchor_viirs)
            valid = np.isfinite(anchor_viirs).astype(np.float32)
            imputed = (~np.isfinite(anchor_viirs)) & np.isfinite(med3)
            filled = np.where(np.isfinite(anchor_viirs), anchor_viirs, med3)
            mk = pd.Timestamp(t_val).strftime("%Y-%m")
            anom = (
                anchor_viirs - anchors.viirs_month_median.get(mk, np.nan)
                if anchors.viirs_month_median
                else np.full_like(anchor_viirs, np.nan)
            )
            viirs_feats = {
                "cell": anchor_viirs,
                "up": filled,
                "mean3": mean3,
                "std3": std3,
                "med3": med3,
                "grad": grad,
                "anom": anom,
                "valid": valid,
                "imputed": imputed.astype(np.float32),
                "qc": viirs_qc,
            }

    for _ in range(spec.max_tiles_per_time):
        if sum(len(yc) for yc in y_chunks) >= target_count:
            break
        r0 = int(rng.integers(0, max(1, H - spec.tile_size + 1)))
        c0 = int(rng.integers(0, max(1, W - spec.tile_size + 1)))
        r1 = min(H, r0 + spec.tile_size)
        c1 = min(W, c0 + spec.tile_size)

        y_tile = np.asarray(
            y_da.isel({y_dim: slice(r0, r1), x_dim: slice(c0, c1)}).data,
            dtype=np.float32,
        )
        y_tile = _nanify_nodata(y_tile)
        y_tile = _mask_lst_values(y_tile, target)
        y_tile = _convert_lst_units(y_tile, target)
        if target_valid and target_valid in dts.data_vars:
            v_da = _select_time_for_var(dts[target_valid], t_val, time_index_map)
            v_tile = np.asarray(
                v_da.isel({y_dim: slice(r0, r1), x_dim: slice(c0, c1)}).data
            )
            y_tile = _apply_valid_mask_data(y_tile, v_tile)
        m_tile = np.isfinite(y_tile)
        if not np.any(m_tile):
            continue

        idx_all = np.flatnonzero(m_tile.reshape(-1))
        if idx_all.size == 0:
            continue
        remaining = target_count - sum(len(yc) for yc in y_chunks)
        n = int(min(remaining, idx_all.size))
        idx = rng.choice(idx_all, size=n, replace=False)

        X_tile = np.zeros((n, len(features) + len(extra_feats)), dtype=np.float32)
        for j, v in enumerate(features):
            a_da = _select_time_for_var(dts[v], t_val, time_index_map)
            a = np.asarray(
                a_da.isel({y_dim: slice(r0, r1), x_dim: slice(c0, c1)}).data,
                dtype=np.float32,
            )
            a = _nanify_nodata(a)
            if valid_map:
                valid_var = valid_map.get(v)
                if valid_var and valid_var in dts.data_vars:
                    v_da = _select_time_for_var(dts[valid_var], t_val, time_index_map)
                    v_tile = np.asarray(
                        v_da.isel({y_dim: slice(r0, r1), x_dim: slice(c0, c1)}).data
                    )
                    a = _apply_valid_mask_data(a, v_tile)
            a = _mask_lst_values(a, v)
            a = _convert_lst_units(a, v)
            a_flat = a.reshape(-1)
            col = a_flat[idx].astype(np.float32, copy=False)
            col[~np.isfinite(col)] = np.nan
            X_tile[:, j] = col

        if anchors and extra_feats:
            row = (idx // (c1 - c0)) + r0
            col = (idx % (c1 - c0)) + c0
            offset = len(features)
            r_f = anchors.row_float[row]
            c_f = anchors.col_float[col]
            if modis_feats is not None:
                X_tile[:, offset] = _nearest_sample(modis_feats["cell"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _bilinear_sample(modis_feats["up"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _bilinear_sample(modis_feats["mean3"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _bilinear_sample(modis_feats["std3"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _bilinear_sample(modis_feats["med3"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _bilinear_sample(modis_feats["grad"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _bilinear_sample(modis_feats["anom"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _nearest_sample(modis_feats["valid"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _nearest_sample(modis_feats["imputed"], r_f, c_f)
                offset += 1
                if anchors.modis_qc_var and modis_feats["qc"] is not None:
                    X_tile[:, offset] = _nearest_sample(modis_feats["qc"], r_f, c_f)
                    offset += 1
            if viirs_feats is not None:
                X_tile[:, offset] = _nearest_sample(viirs_feats["cell"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _bilinear_sample(viirs_feats["up"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _bilinear_sample(viirs_feats["mean3"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _bilinear_sample(viirs_feats["std3"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _bilinear_sample(viirs_feats["med3"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _bilinear_sample(viirs_feats["grad"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _bilinear_sample(viirs_feats["anom"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _nearest_sample(viirs_feats["valid"], r_f, c_f)
                offset += 1
                X_tile[:, offset] = _nearest_sample(viirs_feats["imputed"], r_f, c_f)
                offset += 1
                if anchors.viirs_qc_var and viirs_feats["qc"] is not None:
                    X_tile[:, offset] = _nearest_sample(viirs_feats["qc"], r_f, c_f)

        y_s = y_tile.reshape(-1)[idx].astype(np.float32, copy=False)
        X_chunks.append(X_tile)
        y_chunks.append(y_s)

    if not y_chunks:
        return np.empty((0, len(features) + len(extra_feats)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X = np.concatenate(X_chunks, axis=0)
    y_s = np.concatenate(y_chunks, axis=0)
    if y_s.size < spec.min_samples_per_time and y_s.size > 0:
        # keep what we have rather than forcing large loads
        pass
    return X, y_s


def impute_with_train_medians(X_train: np.ndarray, X_any: np.ndarray) -> np.ndarray:
    meds = np.nanmedian(X_train, axis=0)
    meds = np.where(np.isfinite(meds), meds, 0.0)
    X = np.array(X_any, copy=True)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(meds, np.where(bad)[1])
    return X


# ----------------------------
# Prediction over full map (tile-based)
# ----------------------------

def _tile_to_numpy(da: xr.DataArray, r0: int, r1: int) -> np.ndarray:
    data = da.data
    tile = data[r0:r1, :]
    return np.asarray(tile, dtype=np.float32)


def predict_full_map(
    model: Pipeline,
    dts: xr.Dataset,
    target: str,
    features: Sequence[str],
    *,
    tile_rows: int = 512,
    anchors: Optional[CoarseAnchors] = None,
    time_val=None,
    valid_map: Optional[Dict[str, str]] = None,
    target_valid: Optional[str] = None,
    use_gpu: bool = False,
    td: Optional[str] = None,
    time_index_map: Optional[Dict[pd.Timestamp, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict y over the full spatial grid for one time step.
    Returns (y_true, y_pred) shaped [H, W]
    """
    if td and time_val is not None:
        if time_index_map is None:
            time_index_map = {}
        dts = _select_time_dataset(dts, td, time_val, time_index_map)
    y_da = _select_time_for_var(dts[target], time_val, time_index_map)
    y_true = np.asarray(y_da.data, dtype=np.float32)
    y_true = _nanify_nodata(y_true)
    y_true = _mask_lst_values(y_true, target)
    y_true = _convert_lst_units(y_true, target)
    if target_valid and target_valid in dts.data_vars:
        v = np.asarray(dts[target_valid].data)
        y_true = _apply_valid_mask_data(y_true, v)
    H, W = y_true.shape[-2], y_true.shape[-1]

    y_pred = np.full((H, W), np.nan, dtype=np.float32)

    feature_arrays = [(_select_time_for_var(dts[v], time_val, time_index_map)) for v in features]
    anchor_modis = None
    anchor_viirs = None
    modis_feats = None
    viirs_feats = None
    if anchors and time_val is not None:
        t_idx = anchors.time_index.get(pd.Timestamp(time_val))
        if t_idx is None:
            t_idx = _nearest_time_index(anchors.times, pd.Timestamp(time_val))
        if anchors.modis_day_var:
            anchor_modis = np.asarray(
                anchors.ds[anchors.modis_day_var].isel({anchors.time_dim: t_idx}).data,
                dtype=np.float32,
            )
            masks = []
            if anchors.modis_mask_vars:
                for mv in anchors.modis_mask_vars:
                    m = np.asarray(anchors.ds[mv].isel({anchors.time_dim: t_idx}).data)
                    masks.append((mv, m))
            anchor_modis = _apply_mask(anchor_modis, masks)
            anchor_modis = _nanify_nodata(anchor_modis)
            anchor_modis = _mask_lst_values(anchor_modis, anchors.modis_day_var)
            anchor_modis = _convert_lst_units(anchor_modis, anchors.modis_day_var)
        if anchors.viirs_day_var:
            anchor_viirs = np.asarray(
                anchors.ds[anchors.viirs_day_var].isel({anchors.time_dim: t_idx}).data,
                dtype=np.float32,
            )
            masks = []
            if anchors.viirs_mask_vars:
                for mv in anchors.viirs_mask_vars:
                    m = np.asarray(anchors.ds[mv].isel({anchors.time_dim: t_idx}).data)
                    masks.append((mv, m))
            anchor_viirs = _apply_mask(anchor_viirs, masks)
            anchor_viirs = _nanify_nodata(anchor_viirs)
            anchor_viirs = _mask_lst_values(anchor_viirs, anchors.viirs_day_var)
            anchor_viirs = _convert_lst_units(anchor_viirs, anchors.viirs_day_var)

        if anchor_modis is not None:
            mean3, std3, med3, grad = _coarse_stats(anchor_modis)
            valid = np.isfinite(anchor_modis).astype(np.float32)
            imputed = (~np.isfinite(anchor_modis)) & np.isfinite(med3)
            filled = np.where(np.isfinite(anchor_modis), anchor_modis, med3)
            mk = pd.Timestamp(time_val).strftime("%Y-%m")
            anom = (
                anchor_modis - anchors.modis_month_median.get(mk, np.nan)
                if anchors.modis_month_median
                else np.full_like(anchor_modis, np.nan)
            )
            modis_feats = {
                "cell": anchor_modis,
                "up": filled,
                "mean3": mean3,
                "std3": std3,
                "med3": med3,
                "grad": grad,
                "anom": anom,
                "valid": valid,
                "imputed": imputed.astype(np.float32),
                "qc": None,
            }
            if anchors.modis_qc_var:
                modis_feats["qc"] = np.asarray(
                    anchors.ds[anchors.modis_qc_var].isel({anchors.time_dim: t_idx}).data,
                    dtype=np.float32,
                )
        if anchor_viirs is not None:
            mean3, std3, med3, grad = _coarse_stats(anchor_viirs)
            valid = np.isfinite(anchor_viirs).astype(np.float32)
            imputed = (~np.isfinite(anchor_viirs)) & np.isfinite(med3)
            filled = np.where(np.isfinite(anchor_viirs), anchor_viirs, med3)
            mk = pd.Timestamp(time_val).strftime("%Y-%m")
            anom = (
                anchor_viirs - anchors.viirs_month_median.get(mk, np.nan)
                if anchors.viirs_month_median
                else np.full_like(anchor_viirs, np.nan)
            )
            viirs_feats = {
                "cell": anchor_viirs,
                "up": filled,
                "mean3": mean3,
                "std3": std3,
                "med3": med3,
                "grad": grad,
                "anom": anom,
                "valid": valid,
                "imputed": imputed.astype(np.float32),
                "qc": None,
            }
            if anchors.viirs_qc_var:
                viirs_feats["qc"] = np.asarray(
                    anchors.ds[anchors.viirs_qc_var].isel({anchors.time_dim: t_idx}).data,
                    dtype=np.float32,
                )

    # build features tile by tile (to limit memory)
    for r0 in range(0, H, tile_rows):
        r1 = min(H, r0 + tile_rows)
        # Collect features for this tile
        cols = []
        for da, v in zip(feature_arrays, features):
            tile = _tile_to_numpy(da, r0, r1)
            tile = _nanify_nodata(tile)
            if valid_map:
                valid_var = valid_map.get(da.name)
                if valid_var and valid_var in dts.data_vars:
                    v_da = _select_time_for_var(dts[valid_var], time_val, time_index_map)
                    v_tile = _tile_to_numpy(v_da, r0, r1)
                    tile = _apply_valid_mask_data(tile, v_tile)
            tile = _mask_lst_values(tile, v)
            tile = _convert_lst_units(tile, v)
            cols.append(tile.reshape(-1))
        if anchors and anchors.feature_names:
            include_modis_qc = "modis_qc" in anchors.feature_names
            include_viirs_qc = "viirs_qc" in anchors.feature_names
            modis_count = 9 + (1 if include_modis_qc else 0)
            viirs_count = 9 + (1 if include_viirs_qc else 0)
            rows = np.repeat(np.arange(r0, r1), W)
            cols_idx = np.tile(np.arange(W), r1 - r0)
            r_f = anchors.row_float[rows]
            c_f = anchors.col_float[cols_idx]
            if modis_feats is not None:
                cols.append(_nearest_sample(modis_feats["cell"], r_f, c_f))
                cols.append(_bilinear_sample(modis_feats["up"], r_f, c_f))
                cols.append(_bilinear_sample(modis_feats["mean3"], r_f, c_f))
                cols.append(_bilinear_sample(modis_feats["std3"], r_f, c_f))
                cols.append(_bilinear_sample(modis_feats["med3"], r_f, c_f))
                cols.append(_bilinear_sample(modis_feats["grad"], r_f, c_f))
                cols.append(_bilinear_sample(modis_feats["anom"], r_f, c_f))
                cols.append(_nearest_sample(modis_feats["valid"], r_f, c_f))
                cols.append(_nearest_sample(modis_feats["imputed"], r_f, c_f))
                if include_modis_qc:
                    if modis_feats["qc"] is not None:
                        cols.append(_nearest_sample(modis_feats["qc"], r_f, c_f))
                    else:
                        cols.append(np.full((cols[0].shape[0],), np.nan, dtype=np.float32))
            else:
                for _ in range(modis_count):
                    cols.append(np.full((cols[0].shape[0],), np.nan, dtype=np.float32))
            if viirs_feats is not None:
                cols.append(_nearest_sample(viirs_feats["cell"], r_f, c_f))
                cols.append(_bilinear_sample(viirs_feats["up"], r_f, c_f))
                cols.append(_bilinear_sample(viirs_feats["mean3"], r_f, c_f))
                cols.append(_bilinear_sample(viirs_feats["std3"], r_f, c_f))
                cols.append(_bilinear_sample(viirs_feats["med3"], r_f, c_f))
                cols.append(_bilinear_sample(viirs_feats["grad"], r_f, c_f))
                cols.append(_bilinear_sample(viirs_feats["anom"], r_f, c_f))
                cols.append(_nearest_sample(viirs_feats["valid"], r_f, c_f))
                cols.append(_nearest_sample(viirs_feats["imputed"], r_f, c_f))
                if include_viirs_qc:
                    if viirs_feats["qc"] is not None:
                        cols.append(_nearest_sample(viirs_feats["qc"], r_f, c_f))
                    else:
                        cols.append(np.full((cols[0].shape[0],), np.nan, dtype=np.float32))
            else:
                for _ in range(viirs_count):
                    cols.append(np.full((cols[0].shape[0],), np.nan, dtype=np.float32))
        X_tile = np.stack(cols, axis=1)  # [N, F]
        X_tile[~np.isfinite(X_tile)] = np.nan  # keep nan for imputer within model? pipeline doesn't impute; we do pre-impute outside

        # We stored median-impute before fitting; for full-map we must also impute consistently.
        # Hack: we stored medians on model as attribute.
        meds = getattr(model, "_train_medians", None)
        if meds is None:
            raise RuntimeError("Model missing _train_medians. This script expects it.")
        bad = ~np.isfinite(X_tile)
        if bad.any():
            X_tile[bad] = np.take(meds, np.where(bad)[1])

        if use_gpu:
            try:
                import cupy as cp
            except Exception as exc:
                raise RuntimeError("GPU mode requires cupy to be installed.") from exc
            X_gpu = cp.asarray(X_tile)
            pred_tile = model.predict(X_gpu)
            pred_tile = cp.asnumpy(pred_tile).astype(np.float32, copy=False)
        else:
            pred_tile = model.predict(X_tile).astype(np.float32, copy=False)
        y_pred[r0:r1, :] = pred_tile.reshape((r1 - r0, W))

    return y_true, y_pred


def _robust_vmin_vmax(a: np.ndarray) -> Tuple[float, float]:
    vals = a[np.isfinite(a)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(vals, 2))
    vmax = float(np.nanpercentile(vals, 98))
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def _random_figure_arrays(
    shape: Tuple[int, int],
    rng: np.random.Generator,
    vmin_vmax: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if vmin_vmax is None:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = vmin_vmax
    base = rng.uniform(vmin, vmax, size=shape).astype(np.float32, copy=False)
    noise = rng.normal(0.0, max(1e-6, (vmax - vmin) * 0.05), size=shape).astype(np.float32, copy=False)
    y_true = base
    y_pred = base + noise
    return y_true, y_pred


def save_prediction_figures(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    time_label: str,
    roi_mask: Optional[np.ndarray] = None,
    vmin_vmax: Optional[Tuple[float, float]] = None,
) -> List[Path]:
    outs: List[Path] = []
    if roi_mask is not None:
        y_true = np.where(roi_mask, y_true, np.nan)
        y_pred = np.where(roi_mask, y_pred, np.nan)
        rows = np.any(roi_mask, axis=1)
        cols = np.any(roi_mask, axis=0)
        if np.any(rows) and np.any(cols):
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            y_true = y_true[r0 : r1 + 1, c0 : c1 + 1]
            y_pred = y_pred[r0 : r1 + 1, c0 : c1 + 1]
    err = y_pred - y_true
    if vmin_vmax is None:
        vmin, vmax = _robust_vmin_vmax(y_true)
    else:
        vmin, vmax = vmin_vmax
    emin, emax = _robust_vmin_vmax(err)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    im0 = axes[0].imshow(np.ma.masked_invalid(y_true), cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("Actual LST")
    axes[0].axis("off")
    axes[0].set_facecolor("black")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.ma.masked_invalid(y_pred), cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted LST")
    axes[1].axis("off")
    axes[1].set_facecolor("black")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(np.ma.masked_invalid(err), cmap="coolwarm", vmin=emin, vmax=emax)
    axes[2].set_title("Prediction Error")
    axes[2].axis("off")
    axes[2].set_facecolor("black")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"{model_name} | {time_label}", fontsize=12)
    out_path = FIGURES_DIR / f"{model_name}_{time_label}_map.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    outs.append(out_path)

    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.any(m):
        yt = y_true[m].astype(np.float32, copy=False)
        yp = y_pred[m].astype(np.float32, copy=False)
        bins = 160
        hist, xedges, yedges = np.histogram2d(
            yt,
            yp,
            bins=bins,
            range=[[vmin, vmax], [vmin, vmax]],
        )
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(
            np.log1p(hist.T),
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="magma",
            aspect="equal",
        )
        ax.plot([vmin, vmax], [vmin, vmax], color="white", linewidth=1, alpha=0.8)
        ax.set_title(f"{model_name} | {time_label} | Pred vs True heatmap")
        ax.set_xlabel("True LST")
        ax.set_ylabel("Predicted LST")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log1p(count)")
        out_cmp = FIGURES_DIR / f"{model_name}_{time_label}_compare_heatmap.png"
        fig.savefig(out_cmp, dpi=150)
        plt.close(fig)
        outs.append(out_cmp)

    return outs


def save_metric_plots(df: pd.DataFrame, model_name: str) -> List[Path]:
    outs: List[Path] = []
    df = df.sort_values("time")
    times = pd.to_datetime(df["time"])
    metrics = [m for m in ["rmse", "mae", "ssim", "psnr", "sam", "cc", "ergas"] if m in df.columns]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for m in metrics:
        ax.plot(times, df[m], marker="o", linewidth=1.5, label=m)
    ax.set_title(f"{model_name} metrics over time")
    ax.set_xlabel("time")
    ax.set_ylabel("metric")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    out_ts = FIGURES_DIR / f"{model_name}_metrics_timeseries.png"
    fig.savefig(out_ts, dpi=150)
    plt.close(fig)
    outs.append(out_ts)

    means = df[metrics].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(means.index, means.values)
    ax.set_title(f"{model_name} mean metrics")
    ax.set_ylabel("value")
    ax.grid(True, axis="y", alpha=0.3)
    out_bar = FIGURES_DIR / f"{model_name}_metrics_mean.png"
    fig.savefig(out_bar, dpi=150)
    plt.close(fig)
    outs.append(out_bar)

    return outs


def _append_samples_with_cap(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    Xs: np.ndarray,
    ys: np.ndarray,
    max_total: int,
    rng: np.random.Generator,
) -> None:
    if ys.size == 0:
        return
    X_list.append(Xs)
    y_list.append(ys)
    total = int(sum(len(y) for y in y_list))
    if total <= max_total:
        return
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    idx = rng.choice(X_all.shape[0], size=max_total, replace=False)
    X_list[:] = [X_all[idx]]
    y_list[:] = [y_all[idx]]


class ReservoirSampler:
    def __init__(self, n_features: int, cap: int, rng: np.random.Generator) -> None:
        self.cap = int(cap)
        self.rng = rng
        self.X = np.empty((self.cap, n_features), dtype=np.float32)
        self.y = np.empty((self.cap,), dtype=np.float32)
        self.count = 0
        self.seen = 0

    def add(self, Xs: np.ndarray, ys: np.ndarray) -> None:
        if ys.size == 0:
            return
        for i in range(ys.shape[0]):
            self.seen += 1
            if self.count < self.cap:
                self.X[self.count] = Xs[i]
                self.y[self.count] = ys[i]
                self.count += 1
            else:
                j = int(self.rng.integers(0, self.seen))
                if j < self.cap:
                    self.X[j] = Xs[i]
                    self.y[j] = ys[i]

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[: self.count], self.y[: self.count]


# ----------------------------
# Logging
# ----------------------------

def _sanitize_for_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)


def setup_logging(dataset_key: str, *, start: Optional[str], end: Optional[str]) -> logging.Logger:
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    tag = _sanitize_for_filename(dataset_key)
    range_tag = _sanitize_for_filename(f"{start or 'na'}_{end or 'na'}")
    log_path = LOGS_DIR / f"linear_baselines_{tag}_{range_tag}_{ts}.log"

    logger = logging.getLogger("linear_baselines")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)sZ | %(levelname)s | %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Linear baselines run start | dataset=%s | start=%s | end=%s | utc=%s",
                dataset_key, start, end, pd.Timestamp.utcnow().isoformat())
    logger.info("Log file: %s", log_path)
    return logger


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024.0


def log_mem(logger: logging.Logger, tag: str) -> None:
    logger.info("mem_rss_mb=%0.1f stage=%s", _rss_mb(), tag)


# ----------------------------
# Main
# ----------------------------

def main(
    *,
    dataset_key: str = "madurai_30m",
    start: Optional[str] = None,
    end: Optional[str] = None,
    target: Optional[str] = None,
    features: Optional[List[str]] = None,
    test_frac: float = 0.2,
    ratio_for_ergas: float = 33.3333333333,
    use_coarse_anchors: bool = True,
    use_gpu: bool = False,
    max_samples_per_time: int = 15_000,
    min_samples_per_time: int = 3_000,
    max_total_samples: int = 120_000,
    tile_rows: int = 128,
    random_figures: bool = False,
    figure_count: int = 2,
) -> None:
    logger = setup_logging(dataset_key, start=start, end=end)
    if random_figures:
        logger.info("random_figures=enabled (figures use synthetic data)")
        try:
            zarr_path = PROJECT_ROOT / f"{dataset_key}.zarr"
            root = zarr.open_group(str(zarr_path), mode="r")
            g = root["grid"]
            H = int(g.attrs.get("height"))
            W = int(g.attrs.get("width"))
        except Exception as exc:
            logger.warning("random_figures fallback to metadata: %s", exc)
            md = make_madurai_data(consolidated=False)
            ds_meta = md.get_dataset(dataset_key)
            tgt = pick_target_var(ds_meta, target)
            H = int(ds_meta[tgt].sizes.get("y", ds_meta[tgt].shape[-2]))
            W = int(ds_meta[tgt].sizes.get("x", ds_meta[tgt].shape[-1]))
        roi_mask = None
        try:
            roi_mask = build_roi_mask(dataset_key, (H, W), logger)
        except Exception as exc:
            logger.warning("ROI mask skipped: %s", exc)
        if roi_mask is not None:
            save_roi_figure(roi_mask, FIGURES_DIR / "roi_mask.png")
        rng = np.random.default_rng(RANDOM_SEED)
        y_true, y_pred = _random_figure_arrays((H, W), rng, None)
        fig_paths = save_prediction_figures(
            y_true=y_true,
            y_pred=y_pred,
            model_name="random",
            time_label="random",
            roi_mask=roi_mask,
            vmin_vmax=None,
        )
        logger.info("random_figures_done paths=%s", [str(p) for p in fig_paths])
        return

    md = make_madurai_data(consolidated=False)

    ds_meta = md.get_dataset(dataset_key)
    td = detect_time_dim(ds_meta)
    if td is None:
        raise ValueError(f"{dataset_key} has no time dimension; linear baselines need time to split train/test.")

    # Normalize time
    ds_meta = ds_meta.assign_coords({td: pd.to_datetime(ds_meta[td].values)})
    times = pd.DatetimeIndex(ds_meta[td].values).sort_values()
    time_index_map = {pd.Timestamp(t): i for i, t in enumerate(times)}
    if start:
        times = times[times >= pd.Timestamp(start)]
    if end:
        times = times[times <= pd.Timestamp(end)]
    common_dates_path = PROJECT_ROOT / "common_dates.csv"
    if common_dates_path.exists():
        try:
            common_df = pd.read_csv(common_dates_path)
            common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna()
            common_dates = pd.DatetimeIndex(common_dates).sort_values()
            times = times[times.isin(common_dates)]
            logger.info("common_dates_filtered=%d", len(times))
        except Exception as exc:
            logger.warning("failed to apply common_dates.csv filter: %s", exc)
    else:
        logger.warning("common_dates.csv not found; using full time axis.")

    logger.info(
        "dataset=%s vars=%d times=%d cadence~%s",
        dataset_key,
        len(ds_meta.data_vars),
        len(times),
        cadence_guess(times),
    )
    log_mem(logger, "after_meta_load")

    tgt = pick_target_var(ds_meta, target)
    feats = default_feature_vars(ds_meta, tgt, features)
    logger.info("target=%s", tgt)
    logger.info("n_features=%d", len(feats))

    valid_map = {}
    for v in feats:
        valid_var = _valid_var_for(v, ds_meta)
        if valid_var:
            valid_map[v] = valid_var
    target_valid = _valid_var_for(tgt, ds_meta)

    anchors = None
    all_features = list(feats)
    if use_coarse_anchors:
        fine_shape = (
            int(ds_meta[tgt].sizes.get("y", ds_meta[tgt].shape[-2])),
            int(ds_meta[tgt].sizes.get("x", ds_meta[tgt].shape[-1])),
        )
        anchors = prepare_coarse_anchors(md, start=start, end=end, fine_shape=fine_shape, logger=logger)
        if anchors and anchors.feature_names:
            all_features.extend(anchors.feature_names)
            logger.info("coarse anchor features=%s", ",".join(anchors.feature_names))
        else:
            anchors = None
    logger.info("n_total_features=%d", len(all_features))
    log_mem(logger, "after_anchor_prep")

    include_vars = {tgt, *feats}
    include_vars.update(valid_map.values())
    if target_valid:
        include_vars.add(target_valid)

    availability_spec = SampleSpec(
        max_samples_per_time=3_000,
        min_samples_per_time=1,
        max_total_samples=3_000,
        tile_size=256,
        max_tiles_per_time=4,
    )
    rng = np.random.default_rng(RANDOM_SEED)
    allowed_times = []
    allowed_times = list(times)
    times = pd.DatetimeIndex(allowed_times)
    if len(times) == 0:
        raise RuntimeError("No times left after all-data filtering.")
    logger.info("all_data_days=%d filtered_times=%d", len(times), len(times))

    # Train/test split by shared common splits (train+val vs test)
    splits = load_or_create_splits(COMMON_DATES, SPLITS_PATH)
    train_dates = pd.DatetimeIndex(splits["train"]).normalize()
    val_dates = pd.DatetimeIndex(splits["val"]).normalize()
    test_dates = pd.DatetimeIndex(splits["test"]).normalize()
    times_norm = pd.DatetimeIndex(times).normalize()
    train_times = times[times_norm.isin(train_dates.union(val_dates))]
    test_times = times[times_norm.isin(test_dates)]
    if len(test_times) == 0 or len(train_times) == 0:
        # Fallback: last fraction if split file doesn't match dataset time axis
        n_test = max(1, int(round(len(times) * test_frac)))
        test_times = times[-n_test:]
        train_times = times[:-n_test]
    logger.info("train_times=%d test_times=%d", len(train_times), len(test_times))

    spec = SampleSpec(
        max_samples_per_time=max_samples_per_time,
        min_samples_per_time=min_samples_per_time,
        max_total_samples=max_total_samples,
    )
    logger.info(
        "sampling max_per_time=%d min_per_time=%d max_total=%d tile_rows=%d",
        spec.max_samples_per_time,
        spec.min_samples_per_time,
        spec.max_total_samples,
        tile_rows,
    )

    # Build sampled training set (reservoir to cap memory)
    sampler: Optional[ReservoirSampler] = None
    for t in train_times:
        log_mem(logger, f"train_start {pd.Timestamp(t).date()}")
        dts = load_subset(
            md,
            dataset_key,
            vars_include=sorted(include_vars),
            start=pd.Timestamp(t),
            end=pd.Timestamp(t),
        )
        log_mem(logger, f"train_loaded {pd.Timestamp(t).date()}")
        Xs, ys = sample_xy_for_time(
            dts,
            td,
            t,
            tgt,
            feats,
            spec,
            rng,
            anchors=anchors,
            valid_map=valid_map,
            target_valid=target_valid,
            time_index_map=time_index_map,
        )
        if ys.size == 0:
            del dts, Xs, ys
            gc.collect()
            log_mem(logger, f"train_skip {pd.Timestamp(t).date()}")
            continue
        if sampler is None:
            sampler = ReservoirSampler(Xs.shape[1], spec.max_total_samples, rng)
        sampler.add(Xs, ys)
        del dts, Xs, ys
        gc.collect()
        log_mem(logger, f"train_keep {pd.Timestamp(t).date()}")
    if sampler is None or sampler.count == 0:
        raise RuntimeError("No valid training samples found (target all NaN/Inf).")

    X_train, y_train = sampler.get()
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    logger.info("train_samples=%d train_features=%d", X_train.shape[0], X_train.shape[1])
    train_vmin_vmax = None
    try:
        y_train_f = y_train[np.isfinite(y_train)]
        if y_train_f.size:
            p2, p98 = np.percentile(y_train_f, [2, 98])
            train_vmin_vmax = (float(p2), float(p98))
    except Exception:
        train_vmin_vmax = None

    # Impute using train medians
    train_medians = np.nanmedian(X_train, axis=0)
    train_medians = np.where(np.isfinite(train_medians), train_medians, 0.0)
    X_train = impute_with_train_medians(X_train, X_train)

    # Models (linear baselines)
    if use_gpu:
        try:
            import cupy as cp
            from cuml.preprocessing import StandardScaler as CuStandardScaler
            from cuml.linear_model import LinearRegression as CuLinearRegression
            from cuml.linear_model import Ridge as CuRidge
            from cuml.linear_model import Lasso as CuLasso
            from cuml.linear_model import ElasticNet as CuElasticNet
        except Exception as exc:
            raise RuntimeError("GPU mode requires cupy and cuml to be installed.") from exc

        models = {
            "ols": Pipeline([("scaler", CuStandardScaler()), ("reg", CuLinearRegression())]),
            "ridge": Pipeline([("scaler", CuStandardScaler()), ("reg", CuRidge(alpha=1.0))]),
            "lasso": Pipeline([("scaler", CuStandardScaler()), ("reg", CuLasso(alpha=1e-4, max_iter=5000))]),
            "elasticnet": Pipeline([("scaler", CuStandardScaler()), ("reg", CuElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000))]),
        }
    else:
        models = {
            "ols": Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
            "ridge": Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=RANDOM_SEED))]),
            "lasso": Pipeline([("scaler", StandardScaler()), ("reg", Lasso(alpha=1e-4, random_state=RANDOM_SEED, max_iter=5000))]),
            "elasticnet": Pipeline([("scaler", StandardScaler()), ("reg", ElasticNet(alpha=1e-4, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=5000))]),
        }

    # Fit + evaluate
    all_rows = []
    summary = {}
    figure_paths: Dict[str, List[str]] = {}
    cutoff = pd.Timestamp("2026-01-01")
    pre_cutoff = [t for t in test_times if pd.Timestamp(t) < cutoff]
    metrics_times = list(test_times[:5]) if len(test_times) > 5 else list(test_times)
    if metrics_times:
        figure_times = [str(pd.Timestamp(t).date()) for t in metrics_times[-figure_count:]]
    else:
        figure_times = [str(pd.Timestamp(pre_cutoff[-1] if pre_cutoff else test_times[-1]).date())]
    H = int(ds_meta[tgt].sizes.get("y", ds_meta[tgt].shape[-2]))
    W = int(ds_meta[tgt].sizes.get("x", ds_meta[tgt].shape[-1]))
    roi_mask = None
    try:
        roi_mask = build_roi_mask(dataset_key, (H, W), logger)
    except Exception as exc:
        logger.warning("ROI mask skipped: %s", exc)
    if roi_mask is not None:
        save_roi_figure(roi_mask, FIGURES_DIR / "roi_mask.png")

    train_eval_n = min(20_000, X_train.shape[0])
    train_eval_idx = rng.choice(X_train.shape[0], size=train_eval_n, replace=False)
    X_train_eval = X_train[train_eval_idx]
    y_train_eval = y_train[train_eval_idx]

    for name, model in models.items():
        logger.info("model=%s fit_samples=%d", name, X_train.shape[0])
        log_mem(logger, f"model_fit_start {name}")
        if use_gpu:
            X_fit = cp.asarray(X_train)
            y_fit = cp.asarray(y_train)
            model.fit(X_fit, y_fit)
        else:
            model.fit(X_train, y_train)
        log_mem(logger, f"model_fit_done {name}")

        # attach medians for full-map prediction
        setattr(model, "_train_medians", train_medians)
        if use_gpu:
            logger.info("model=%s saved_model=skipped (gpu)", name)
        else:
            model_path = MODELS_DIR / f"linear_{name}.joblib"
            joblib.dump(model, model_path)
            logger.info("model=%s saved_model=%s", name, model_path)
        y_tr_pred = model.predict(X_train_eval)
        tr_err = y_tr_pred - y_train_eval
        train_rmse = float(np.sqrt(np.mean(tr_err ** 2)))
        summary.setdefault(name, {})
        summary[name]["train_rmse"] = train_rmse
        logger.info("model=%s train_rmse=%.6f", name, train_rmse)

    rows_by_model: Dict[str, List[Dict[str, object]]] = {name: [] for name in models}
    eval_spec = SampleSpec(
        max_samples_per_time=max(1000, max_samples_per_time // 2),
        min_samples_per_time=0,
        max_total_samples=max(10_000, max_total_samples // 6),
    )
    for t in metrics_times:
        log_mem(logger, f"test_start {pd.Timestamp(t).date()}")
        dts = load_subset(
            md,
            dataset_key,
            vars_include=sorted(include_vars),
            start=pd.Timestamp(t),
            end=pd.Timestamp(t),
        )
        log_mem(logger, f"test_loaded {pd.Timestamp(t).date()}")
        for name, model in models.items():
            figure_date = str(pd.Timestamp(t).date())
            is_figure_time = figure_date in figure_times
            if random_figures and is_figure_time:
                y_true, y_pred = _random_figure_arrays((H, W), rng, train_vmin_vmax)
                fig_paths = save_prediction_figures(
                    y_true=y_true,
                    y_pred=y_pred,
                    model_name=name,
                    time_label=figure_date,
                    roi_mask=roi_mask,
                    vmin_vmax=train_vmin_vmax,
                )
                figure_paths.setdefault(name, []).extend([str(p) for p in fig_paths])
                rmse = float("nan")
                rmse_sum = float("nan")
                mae = float("nan")
                n_valid = 0
                rmse_sampled = float("nan")
                n_sampled = 0
                del y_true, y_pred
            else:
                y_true, y_pred = predict_full_map(
                    model,
                    dts,
                    tgt,
                    feats,
                    tile_rows=tile_rows,
                    anchors=anchors,
                    time_val=t,
                    valid_map=valid_map,
                    target_valid=target_valid,
                    use_gpu=use_gpu,
                    td=td,
                    time_index_map=time_index_map,
                )
                m = np.isfinite(y_true) & np.isfinite(y_pred)
                if np.any(m):
                    err = y_pred[m] - y_true[m]
                    rmse = float(np.sqrt(np.mean(err ** 2)))
                    rmse_sum = float(np.sqrt(np.sum(err ** 2)))
                    mae = float(np.mean(np.abs(err)))
                else:
                    rmse = float("nan")
                    rmse_sum = float("nan")
                    mae = float("nan")
                n_valid = int(np.sum(m))
                Xs, ys = sample_xy_for_time(
                    dts,
                    td,
                    t,
                    tgt,
                    feats,
                    eval_spec,
                    rng,
                    anchors=anchors,
                    valid_map=valid_map,
                    target_valid=target_valid,
                    time_index_map=time_index_map,
                )
                if ys.size == 0:
                    rmse_sampled = float("nan")
                    n_sampled = 0
                else:
                    Xs = impute_with_train_medians(X_train, Xs)
                    y_pred_s = model.predict(Xs).astype(np.float32, copy=False)
                    m_s = np.isfinite(ys) & np.isfinite(y_pred_s)
                    if np.any(m_s):
                        err_s = y_pred_s[m_s] - ys[m_s]
                        rmse_sampled = float(np.sqrt(np.mean(err_s ** 2)))
                    else:
                        rmse_sampled = float("nan")
                    n_sampled = int(np.sum(m_s))
                del Xs, ys
                if is_figure_time:
                    fig_paths = save_prediction_figures(
                        y_true=y_true,
                        y_pred=y_pred,
                        model_name=name,
                        time_label=figure_date,
                        roi_mask=roi_mask,
                        vmin_vmax=train_vmin_vmax,
                    )
                    figure_paths.setdefault(name, []).extend([str(p) for p in fig_paths])
                del y_true, y_pred, m
            row = {
                "model": name,
                "time": str(pd.Timestamp(t).date()),
                "rmse": rmse,
                "rmse_sum": rmse_sum,
                "rmse_sampled": rmse_sampled,
                "mae": mae,
                "n_valid": n_valid,
                "n_sampled": n_sampled,
            }
            rows_by_model[name].append(row)
            gc.collect()
        del dts
        gc.collect()
        log_mem(logger, f"test_done {pd.Timestamp(t).date()}")

    for name, rows in rows_by_model.items():
        df = pd.DataFrame(rows)
        out_csv = OUT_DIR / f"{name}_metrics.csv"
        df.to_csv(out_csv, index=False)
        figure_paths.setdefault(name, []).extend([str(p) for p in save_metric_plots(df, name)])

        # aggregate summary
        agg = df[["rmse", "rmse_sum", "mae"]].mean(numeric_only=True).to_dict()
        agg_std = df[["rmse", "rmse_sum", "mae"]].std(numeric_only=True).to_dict()
        test_rmse = float(np.nanmean(df["rmse"])) if "rmse" in df.columns else float("nan")
        summary[name].update(
            {"mean": agg, "std": agg_std, "n_test_times": int(len(df)), "test_rmse": test_rmse}
        )
        logger.info("model=%s saved_metrics=%s", name, out_csv)

        all_rows.extend(rows)

    # Save combined + config
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "ALL_linear_metrics.csv", index=False)

    cfg = {
        "dataset_key": dataset_key,
        "start": start,
        "end": end,
        "time_dim": td,
        "cadence_guess": cadence_guess(times),
        "target": tgt,
        "features": all_features,
        "test_frac": test_frac,
        "ergas_ratio": ratio_for_ergas,
        "seed": RANDOM_SEED,
        "random_figures": random_figures,
        "figure_count": figure_count,
        "out_dir": str(OUT_DIR),
        "figures_dir": str(FIGURES_DIR),
        "figures": figure_paths,
        "summary": summary,
    }
    with (OUT_DIR / "run_config_and_summary.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    if figure_paths:
        logger.info("figures_dir=%s", FIGURES_DIR)
    logger.info("DONE metrics_dir=%s", OUT_DIR)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("Linear baselines for 30m LST reconstruction")
    p.add_argument("--dataset", default="madurai_30m", choices=["madurai", "madurai_30m", "madurai_alphaearth_30m"])
    p.add_argument("--start", default=None, help="e.g., 2019-01-01")
    p.add_argument("--end", default=None, help="e.g., 2020-12-31")
    p.add_argument("--target", default=None, help="target variable name (recommended to set once)")
    p.add_argument("--features", default=None, nargs="+", help="feature variable names (optional)")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--ergas-ratio", type=float, default=33.3333333333, help="coarse_res/fine_res (default 1000/30)")
    p.add_argument("--max-samples-per-time", type=int, default=15_000)
    p.add_argument("--min-samples-per-time", type=int, default=3_000)
    p.add_argument("--max-total-samples", type=int, default=120_000)
    p.add_argument("--tile-rows", type=int, default=128)
    p.add_argument("--no-coarse-anchors", action="store_true", help="disable MODIS/VIIRS coarse anchor features")
    p.add_argument("--gpu", action="store_true", help="use GPU (requires cupy + cuml)")
    p.add_argument("--random-figures", action="store_true", help="use random data for figure generation (debug)")
    p.add_argument("--figure-count", type=int, default=2, help="number of figure dates to save from test set")
    args = p.parse_args()

    main(
        dataset_key=args.dataset,
        start=args.start,
        end=args.end,
        target=args.target,
        features=args.features,
        test_frac=args.test_frac,
        ratio_for_ergas=args.ergas_ratio,
        use_coarse_anchors=not args.no_coarse_anchors,
        use_gpu=args.gpu,
        max_samples_per_time=args.max_samples_per_time,
        min_samples_per_time=args.min_samples_per_time,
        max_total_samples=args.max_total_samples,
        tile_rows=args.tile_rows,
        random_figures=args.random_figures,
        figure_count=args.figure_count,
    )
