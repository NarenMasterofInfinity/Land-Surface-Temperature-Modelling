from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

from helper import make_madurai_data, load_subset
from helper.metrics_image import compute_all


PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project").resolve()
OUT_DIR = PROJECT_ROOT / "metrics" / "deep_baselines" / "pixel_mlp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = PROJECT_ROOT / "logs" / "new"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ROI polygon (lon, lat)
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


def setup_logging(dataset_key: str, *, start: Optional[str], end: Optional[str]) -> logging.Logger:
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    tag = dataset_key.replace("/", "_")
    range_tag = f"{start or 'na'}_{end or 'na'}".replace(":", "_")
    log_path = LOGS_DIR / f"pixel_mlp_{tag}_{range_tag}_{ts}.log"

    logger = logging.getLogger("pixel_mlp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)sZ | %(levelname)s | %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Pixel-MLP run start | dataset=%s | start=%s | end=%s | utc=%s",
                dataset_key, start, end, pd.Timestamp.utcnow().isoformat())
    logger.info("Log file: %s", log_path)
    return logger


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

    raise ValueError("Could not auto-detect Landsat band_01 target. Provide --target explicitly.")


def default_feature_vars(ds: xr.Dataset, target: str, user_features: Optional[Sequence[str]]) -> List[str]:
    if user_features:
        missing = [v for v in user_features if v not in ds.data_vars]
        if missing:
            raise KeyError(f"Some --features not found: {missing}")
        return list(user_features)

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


def _month_key(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m")


def _month_window(ts: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(ts.year, ts.month, 1)
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def _build_index_map(fine_len: int, coarse_len: int) -> np.ndarray:
    scale = float(coarse_len) / float(fine_len)
    idx = np.floor(np.arange(fine_len) * scale).astype(np.int64)
    return np.clip(idx, 0, coarse_len - 1)


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
    for v in cand:
        if "day" in v.lower() and ("lst" in v.lower() or "temp" in v.lower()):
            return v
    return cand[0]


def _choose_mask_vars(vars_list: Sequence[str], prefix: str) -> List[str]:
    keys = ("qc", "quality", "mask", "cloud", "valid")
    out = []
    for v in vars_list:
        if v.startswith(prefix) and any(k in v.lower() for k in keys):
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
        return data  # MODIS already in °C
    if _is_landsat_var(var_name) or _is_viirs_var(var_name):
        med = float(np.nanmedian(data[finite]))
        if med > 200:
            return data - 273.15
    return data


@dataclass
class MonthlyAnchors:
    modis_cell: Optional[np.ndarray]
    modis_up: Optional[np.ndarray]
    modis_mean3: Optional[np.ndarray]
    modis_std3: Optional[np.ndarray]
    modis_med3: Optional[np.ndarray]
    modis_grad: Optional[np.ndarray]
    modis_anom: Optional[np.ndarray]
    modis_valid: Optional[np.ndarray]
    modis_imputed: Optional[np.ndarray]
    modis_qc: Optional[np.ndarray]
    viirs_cell: Optional[np.ndarray]
    viirs_up: Optional[np.ndarray]
    viirs_mean3: Optional[np.ndarray]
    viirs_std3: Optional[np.ndarray]
    viirs_med3: Optional[np.ndarray]
    viirs_grad: Optional[np.ndarray]
    viirs_anom: Optional[np.ndarray]
    viirs_valid: Optional[np.ndarray]
    viirs_imputed: Optional[np.ndarray]
    viirs_qc: Optional[np.ndarray]
    has_modis: int
    has_viirs: int
    n_modis: int
    n_viirs: int


def build_monthly_anchor_cache(
    md,
    landsat_times: pd.DatetimeIndex,
    fine_shape: Tuple[int, int],
    *,
    start: Optional[str],
    end: Optional[str],
    logger: logging.Logger,
) -> Tuple[Dict[str, MonthlyAnchors], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vars_all = md.vars("madurai")
    modis_day = _choose_day_var(vars_all, "products/modis/")
    viirs_day = _choose_day_var(vars_all, "products/viirs/")
    modis_masks = _choose_mask_vars(vars_all, "products/modis/")
    viirs_masks = _choose_mask_vars(vars_all, "products/viirs/")
    modis_qc = _choose_qc_var(vars_all, "products/modis/")
    viirs_qc = _choose_qc_var(vars_all, "products/viirs/")
    include_vars = [v for v in [modis_day, viirs_day, modis_qc, viirs_qc] if v] + modis_masks + viirs_masks
    if not include_vars:
        raise RuntimeError("No MODIS/VIIRS vars found for monthly alignment.")

    ds_c = load_subset(md, "madurai", vars_include=include_vars, start=start, end=end)
    td = detect_time_dim(ds_c)
    if td is None:
        raise RuntimeError("Cannot infer time dimension for MODIS/VIIRS.")
    ds_c = ds_c.assign_coords({td: pd.to_datetime(ds_c[td].values, errors="coerce")})
    times_c = pd.DatetimeIndex(ds_c[td].values)
    if times_c.isna().all():
        raise RuntimeError("MODIS/VIIRS time values are not parseable; cannot align monthly.")

    sample_var = modis_day or viirs_day
    da = ds_c[sample_var]
    if "y" not in da.dims or "x" not in da.dims:
        raise RuntimeError("Expected y/x dims for MODIS/VIIRS.")
    Hc, Wc = da.sizes["y"], da.sizes["x"]
    row_map = _build_index_map(fine_shape[0], Hc)
    col_map = _build_index_map(fine_shape[1], Wc)
    row_float = _build_float_map(fine_shape[0], Hc)
    col_float = _build_float_map(fine_shape[1], Wc)

    cache: Dict[str, MonthlyAnchors] = {}
    for t in landsat_times:
        mk = _month_key(pd.Timestamp(t))
        if mk in cache:
            continue
        m_start, m_end = _month_window(pd.Timestamp(t))
        idx = (times_c >= m_start) & (times_c < m_end)
        if not np.any(idx):
            cache[mk] = MonthlyAnchors(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                0,
                0,
                0,
            )
            continue

        def _aggregate(var: Optional[str], masks: List[str]) -> Tuple[Optional[np.ndarray], int]:
            if var is None:
                return None, 0
            data = np.asarray(ds_c[var].isel({td: idx}).data, dtype=np.float32)
            if data.ndim == 4:
                data = data[:, 0, :, :]
            masked = []
            n_valid_days = 0
            for i in range(data.shape[0]):
                d = data[i]
                if masks:
                    mask_list = []
                    for mv in masks:
                        m = np.asarray(ds_c[mv].isel({td: idx}).data, dtype=np.float32)
                        if m.ndim == 4:
                            m = m[:, 0, :, :]
                        mask_list.append((mv, m[i]))
                    d = _apply_mask(d, mask_list)
                d = _mask_lst_values(d, var)
                d = _convert_lst_units(d, var)
                if np.isfinite(d).any():
                    n_valid_days += 1
                masked.append(d)
            if not masked:
                return None, 0
            agg = np.nanmedian(np.stack(masked, axis=0), axis=0)
            if not np.isfinite(agg).any():
                return None, 0
            return agg.astype(np.float32), n_valid_days

        def _aggregate_qc(var: Optional[str], masks: List[str]) -> Optional[np.ndarray]:
            if var is None:
                return None
            data = np.asarray(ds_c[var].isel({td: idx}).data, dtype=np.float32)
            if data.ndim == 4:
                data = data[:, 0, :, :]
            masked = []
            for i in range(data.shape[0]):
                d = data[i]
                if masks:
                    mask_list = []
                    for mv in masks:
                        m = np.asarray(ds_c[mv].isel({td: idx}).data, dtype=np.float32)
                        if m.ndim == 4:
                            m = m[:, 0, :, :]
                        mask_list.append((mv, m[i]))
                    d = _apply_mask(d, mask_list)
                if np.isfinite(d).any():
                    masked.append(d)
            if not masked:
                return None
            agg = np.nanmedian(np.stack(masked, axis=0), axis=0)
            if not np.isfinite(agg).any():
                return None
            return agg.astype(np.float32)

        modis_map, n_modis = _aggregate(modis_day, modis_masks)
        viirs_map, n_viirs = _aggregate(viirs_day, viirs_masks)
        modis_qc_map = _aggregate_qc(modis_qc, modis_masks)
        viirs_qc_map = _aggregate_qc(viirs_qc, viirs_masks)

        def _build_stats(arr: Optional[np.ndarray]):
            if arr is None:
                return None, None, None, None, None, None, None, None
            mean3, std3, med3, grad = _coarse_stats(arr)
            valid = np.isfinite(arr).astype(np.float32)
            imputed = (~np.isfinite(arr)) & np.isfinite(med3)
            up = np.where(np.isfinite(arr), arr, med3)
            anom = np.where(np.isfinite(arr), 0.0, np.nan).astype(np.float32, copy=False)
            return mean3, std3, med3, grad, valid, imputed.astype(np.float32), up, anom

        (
            modis_mean3,
            modis_std3,
            modis_med3,
            modis_grad,
            modis_valid,
            modis_imputed,
            modis_up,
            modis_anom,
        ) = _build_stats(modis_map)
        (
            viirs_mean3,
            viirs_std3,
            viirs_med3,
            viirs_grad,
            viirs_valid,
            viirs_imputed,
            viirs_up,
            viirs_anom,
        ) = _build_stats(viirs_map)
        cache[mk] = MonthlyAnchors(
            modis_cell=modis_map,
            modis_up=modis_up,
            modis_mean3=modis_mean3,
            modis_std3=modis_std3,
            modis_med3=modis_med3,
            modis_grad=modis_grad,
            modis_anom=modis_anom,
            modis_valid=modis_valid,
            modis_imputed=modis_imputed,
            modis_qc=modis_qc_map,
            viirs_cell=viirs_map,
            viirs_up=viirs_up,
            viirs_mean3=viirs_mean3,
            viirs_std3=viirs_std3,
            viirs_med3=viirs_med3,
            viirs_grad=viirs_grad,
            viirs_anom=viirs_anom,
            viirs_valid=viirs_valid,
            viirs_imputed=viirs_imputed,
            viirs_qc=viirs_qc_map,
            has_modis=int(modis_map is not None),
            has_viirs=int(viirs_map is not None),
            n_modis=n_modis,
            n_viirs=n_viirs,
        )

    logger.info("monthly anchors cached for %d months", len(cache))
    return cache, row_map, col_map, row_float, col_float


def build_anchor_feature_names(month_cache: Dict[str, MonthlyAnchors]) -> List[str]:
    include_modis_qc = any(a.modis_qc is not None for a in month_cache.values())
    include_viirs_qc = any(a.viirs_qc is not None for a in month_cache.values())
    names = [
        "modis_cell",
        "modis_up",
        "modis_mean3",
        "modis_std3",
        "modis_med3",
        "modis_grad",
        "modis_anom",
        "modis_valid",
        "modis_imputed",
    ]
    if include_modis_qc:
        names.append("modis_qc")
    names += [
        "viirs_cell",
        "viirs_up",
        "viirs_mean3",
        "viirs_std3",
        "viirs_med3",
        "viirs_grad",
        "viirs_anom",
        "viirs_valid",
        "viirs_imputed",
    ]
    if include_viirs_qc:
        names.append("viirs_qc")
    names += [
        "has_modis_month",
        "n_modis_obs_month",
        "has_viirs_month",
        "n_viirs_obs_month",
    ]
    return names


@dataclass
class SampleSpec:
    max_samples_per_time: int = 15_000
    min_samples_per_time: int = 3_000
    max_total_samples: int = 120_000
    tile_size: int = 256
    max_tiles_per_time: int = 12


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


def impute_with_train_medians(X_train: np.ndarray, X_any: np.ndarray) -> np.ndarray:
    meds = np.nanmedian(X_train, axis=0)
    meds = np.where(np.isfinite(meds), meds, 0.0)
    X = np.array(X_any, copy=True)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(meds, np.where(bad)[1])
    return X


def _infer_xy_dims(da: xr.DataArray) -> Tuple[str, str]:
    if "y" in da.dims and "x" in da.dims:
        return "y", "x"
    return da.dims[-2], da.dims[-1]


def sample_xy_for_time(
    ds: xr.Dataset,
    td: str,
    t_val,
    target: str,
    features: Sequence[str],
    spec: SampleSpec,
    rng: np.random.Generator,
    *,
    month_cache: Dict[str, MonthlyAnchors],
    row_float: np.ndarray,
    col_float: np.ndarray,
    anchor_feature_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    dts = ds.sel({td: t_val})
    y_da = dts[target]
    y_dim, x_dim = _infer_xy_dims(y_da)
    H = y_da.sizes[y_dim]
    W = y_da.sizes[x_dim]

    target_count = spec.max_samples_per_time
    X_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []

    mk = _month_key(pd.Timestamp(t_val))
    anchors = month_cache.get(mk)
    if anchors is None:
        return np.empty((0, len(features) + len(anchor_feature_names)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    for _ in range(spec.max_tiles_per_time):
        if sum(len(yc) for yc in y_chunks) >= target_count:
            break
        r0 = int(rng.integers(0, max(1, H - spec.tile_size + 1)))
        c0 = int(rng.integers(0, max(1, W - spec.tile_size + 1)))
        r1 = min(H, r0 + spec.tile_size)
        c1 = min(W, c0 + spec.tile_size)

        y_tile = np.asarray(y_da.isel({y_dim: slice(r0, r1), x_dim: slice(c0, c1)}).data, dtype=np.float32)
        y_tile = _mask_lst_values(y_tile, target)
        y_tile = _convert_lst_units(y_tile, target)
        m_tile = np.isfinite(y_tile)
        if not np.any(m_tile):
            continue
        idx_all = np.flatnonzero(m_tile.reshape(-1))
        if idx_all.size == 0:
            continue
        remaining = target_count - sum(len(yc) for yc in y_chunks)
        n = int(min(remaining, idx_all.size))
        idx = rng.choice(idx_all, size=n, replace=False)

        X_tile = np.zeros((n, len(features) + len(anchor_feature_names)), dtype=np.float32)
        for j, v in enumerate(features):
            a = np.asarray(
                dts[v].isel({y_dim: slice(r0, r1), x_dim: slice(c0, c1)}).data,
                dtype=np.float32,
            )
            a = _mask_lst_values(a, v)
            a = _convert_lst_units(a, v)
            a_flat = a.reshape(-1)
            col = a_flat[idx].astype(np.float32, copy=False)
            col[~np.isfinite(col)] = np.nan
            X_tile[:, j] = col

        row = (idx // (c1 - c0)) + r0
        col = (idx % (c1 - c0)) + c0
        r_f = row_float[row]
        c_f = col_float[col]
        offset = len(features)
        include_modis_qc = "modis_qc" in anchor_feature_names
        include_viirs_qc = "viirs_qc" in anchor_feature_names
        modis_count = 9 + (1 if include_modis_qc else 0)
        viirs_count = 9 + (1 if include_viirs_qc else 0)
        if anchors.modis_cell is not None:
            X_tile[:, offset] = _nearest_sample(anchors.modis_cell, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _bilinear_sample(anchors.modis_up, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _bilinear_sample(anchors.modis_mean3, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _bilinear_sample(anchors.modis_std3, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _bilinear_sample(anchors.modis_med3, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _bilinear_sample(anchors.modis_grad, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _bilinear_sample(anchors.modis_anom, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _nearest_sample(anchors.modis_valid, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _nearest_sample(anchors.modis_imputed, r_f, c_f)
            offset += 1
            if include_modis_qc:
                if anchors.modis_qc is not None:
                    X_tile[:, offset] = _nearest_sample(anchors.modis_qc, r_f, c_f)
                else:
                    X_tile[:, offset] = np.nan
                offset += 1
        else:
            X_tile[:, offset:offset + modis_count] = np.nan
            offset += modis_count

        if anchors.viirs_cell is not None:
            X_tile[:, offset] = _nearest_sample(anchors.viirs_cell, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _bilinear_sample(anchors.viirs_up, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _bilinear_sample(anchors.viirs_mean3, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _bilinear_sample(anchors.viirs_std3, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _bilinear_sample(anchors.viirs_med3, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _bilinear_sample(anchors.viirs_grad, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _bilinear_sample(anchors.viirs_anom, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _nearest_sample(anchors.viirs_valid, r_f, c_f)
            offset += 1
            X_tile[:, offset] = _nearest_sample(anchors.viirs_imputed, r_f, c_f)
            offset += 1
            if include_viirs_qc:
                if anchors.viirs_qc is not None:
                    X_tile[:, offset] = _nearest_sample(anchors.viirs_qc, r_f, c_f)
                else:
                    X_tile[:, offset] = np.nan
                offset += 1
        else:
            X_tile[:, offset:offset + viirs_count] = np.nan
            offset += viirs_count

        X_tile[:, offset] = float(anchors.has_modis)
        offset += 1
        X_tile[:, offset] = float(anchors.n_modis)
        offset += 1
        X_tile[:, offset] = float(anchors.has_viirs)
        offset += 1
        X_tile[:, offset] = float(anchors.n_viirs)

        y_s = y_tile.reshape(-1)[idx].astype(np.float32, copy=False)
        X_chunks.append(X_tile)
        y_chunks.append(y_s)

    if not y_chunks:
        return np.empty((0, len(features) + len(anchor_feature_names)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X = np.concatenate(X_chunks, axis=0)
    y_s = np.concatenate(y_chunks, axis=0)
    return X, y_s


def _tile_to_numpy(da: xr.DataArray, r0: int, r1: int) -> np.ndarray:
    data = da.data
    tile = data[r0:r1, :]
    return np.asarray(tile, dtype=np.float32)


def predict_full_map(
    model: nn.Module,
    dts: xr.Dataset,
    target: str,
    features: Sequence[str],
    *,
    tile_rows: int = 128,
    month_cache: Dict[str, MonthlyAnchors],
    row_float: np.ndarray,
    col_float: np.ndarray,
    anchor_feature_names: Sequence[str],
    time_val,
    device: torch.device,
    train_medians: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(dts[target].data, dtype=np.float32)
    y_true = _mask_lst_values(y_true, target)
    y_true = _convert_lst_units(y_true, target)
    H, W = y_true.shape[-2], y_true.shape[-1]
    y_pred = np.full((H, W), np.nan, dtype=np.float32)
    feature_arrays = [dts[v] for v in features]

    mk = _month_key(pd.Timestamp(time_val))
    anchors = month_cache.get(mk)
    if anchors is None:
        raise RuntimeError(f"No monthly anchors for month {mk}")

    for r0 in range(0, H, tile_rows):
        r1 = min(H, r0 + tile_rows)
        cols = []
        for da, v in zip(feature_arrays, features):
            tile = _tile_to_numpy(da, r0, r1)
            tile = _mask_lst_values(tile, v)
            tile = _convert_lst_units(tile, v)
            cols.append(tile.reshape(-1))

        include_modis_qc = "modis_qc" in anchor_feature_names
        include_viirs_qc = "viirs_qc" in anchor_feature_names
        modis_count = 9 + (1 if include_modis_qc else 0)
        viirs_count = 9 + (1 if include_viirs_qc else 0)

        rows = np.repeat(np.arange(r0, r1), W)
        cols_idx = np.tile(np.arange(W), r1 - r0)
        r_f = row_float[rows]
        c_f = col_float[cols_idx]

        if anchors.modis_cell is not None:
            cols.append(_nearest_sample(anchors.modis_cell, r_f, c_f))
            cols.append(_bilinear_sample(anchors.modis_up, r_f, c_f))
            cols.append(_bilinear_sample(anchors.modis_mean3, r_f, c_f))
            cols.append(_bilinear_sample(anchors.modis_std3, r_f, c_f))
            cols.append(_bilinear_sample(anchors.modis_med3, r_f, c_f))
            cols.append(_bilinear_sample(anchors.modis_grad, r_f, c_f))
            cols.append(_bilinear_sample(anchors.modis_anom, r_f, c_f))
            cols.append(_nearest_sample(anchors.modis_valid, r_f, c_f))
            cols.append(_nearest_sample(anchors.modis_imputed, r_f, c_f))
            if include_modis_qc:
                if anchors.modis_qc is not None:
                    cols.append(_nearest_sample(anchors.modis_qc, r_f, c_f))
                else:
                    cols.append(np.full((cols[0].shape[0],), np.nan, dtype=np.float32))
        else:
            for _ in range(modis_count):
                cols.append(np.full((cols[0].shape[0],), np.nan, dtype=np.float32))

        if anchors.viirs_cell is not None:
            cols.append(_nearest_sample(anchors.viirs_cell, r_f, c_f))
            cols.append(_bilinear_sample(anchors.viirs_up, r_f, c_f))
            cols.append(_bilinear_sample(anchors.viirs_mean3, r_f, c_f))
            cols.append(_bilinear_sample(anchors.viirs_std3, r_f, c_f))
            cols.append(_bilinear_sample(anchors.viirs_med3, r_f, c_f))
            cols.append(_bilinear_sample(anchors.viirs_grad, r_f, c_f))
            cols.append(_bilinear_sample(anchors.viirs_anom, r_f, c_f))
            cols.append(_nearest_sample(anchors.viirs_valid, r_f, c_f))
            cols.append(_nearest_sample(anchors.viirs_imputed, r_f, c_f))
            if include_viirs_qc:
                if anchors.viirs_qc is not None:
                    cols.append(_nearest_sample(anchors.viirs_qc, r_f, c_f))
                else:
                    cols.append(np.full((cols[0].shape[0],), np.nan, dtype=np.float32))
        else:
            for _ in range(viirs_count):
                cols.append(np.full((cols[0].shape[0],), np.nan, dtype=np.float32))

        cols.append(np.full((cols[0].shape[0],), float(anchors.has_modis), dtype=np.float32))
        cols.append(np.full((cols[0].shape[0],), float(anchors.n_modis), dtype=np.float32))
        cols.append(np.full((cols[0].shape[0],), float(anchors.has_viirs), dtype=np.float32))
        cols.append(np.full((cols[0].shape[0],), float(anchors.n_viirs), dtype=np.float32))

        X_tile = np.stack(cols, axis=1).astype(np.float32, copy=False)
        X_tile[~np.isfinite(X_tile)] = np.nan
        bad = ~np.isfinite(X_tile)
        if bad.any():
            X_tile[bad] = np.take(train_medians, np.where(bad)[1])
        with torch.no_grad():
            xb = torch.from_numpy(X_tile).to(device)
            pred_tile = model(xb).squeeze(-1).float().cpu().numpy()
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
    err = y_pred - y_true
    if vmin_vmax is None:
        vmin, vmax = _robust_vmin_vmax(y_true)
    else:
        vmin, vmax = vmin_vmax

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

    im2 = axes[2].imshow(np.ma.masked_invalid(err), cmap="coolwarm")
    axes[2].set_title("Prediction Error")
    axes[2].axis("off")
    axes[2].set_facecolor("black")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"{model_name} | {time_label}", fontsize=12)
    out_path = FIGURES_DIR / f"{model_name}_{time_label}_map.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    outs.append(out_path)
    return outs


def build_roi_mask(dataset_key: str, shape: Tuple[int, int], logger: logging.Logger) -> Optional[np.ndarray]:
    try:
        zarr_path = PROJECT_ROOT / f"{dataset_key}.zarr"
        if not zarr_path.exists():
            return None
        import zarr
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


def save_metric_plots(df: pd.DataFrame, model_name: str) -> List[Path]:
    outs: List[Path] = []
    df = df.sort_values("time")
    times = pd.to_datetime(df["time"])
    metrics = ["rmse", "ssim", "psnr", "sam", "cc", "ergas"]

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


class PixelMLP(nn.Module):
    def __init__(self, n_features: int, hidden: Sequence[int] = (256, 128, 64), dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_f = n_features
        for h in hidden:
            layers.append(nn.Linear(in_f, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_f = h
        layers.append(nn.Linear(in_f, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    device: torch.device,
    hidden: Sequence[int] = (256, 128, 64),
    dropout: float = 0.0,
    batch_size: int = 8192,
    lr: float = 1e-3,
    max_epochs: int = 500,
    val_frac: float = 0.1,
    patience: int = 5,
    seed: int = RANDOM_SEED,
    logger: logging.Logger,
) -> nn.Module:
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xval, yval = _split_train_val(X_train, y_train, val_frac, rng)
    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    val_ds = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = PixelMLP(X_train.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_sq = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * xb.size(0)
            err = pred - yb
            train_sq += float((err * err).sum().item())
            train_n += int(err.numel())
        train_loss /= max(1, len(train_ds))

        model.eval()
        val_loss = 0.0
        val_sq = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.float32)
                pred = model(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                val_loss += float(loss.item()) * xb.size(0)
                err = pred - yb
                val_sq += float((err * err).sum().item())
                val_n += int(err.numel())
        val_loss /= max(1, len(val_ds))

        train_rmse = float(np.sqrt(train_sq / train_n)) if train_n > 0 else float("nan")
        val_rmse = float(np.sqrt(val_sq / val_n)) if val_n > 0 else float("nan")
        logger.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f train_rmse=%.6f val_rmse=%.6f",
            epoch,
            train_loss,
            val_loss,
            train_rmse,
            val_rmse,
        )
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("early_stop epoch=%d best_val=%.6f", epoch, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main(
    *,
    dataset_key: str = "madurai_30m",
    start: Optional[str] = None,
    end: Optional[str] = None,
    target: Optional[str] = None,
    features: Optional[List[str]] = None,
    test_frac: float = 0.2,
    ratio_for_ergas: float = 33.3333333333,
    max_samples_per_time: int = 15_000,
    min_samples_per_time: int = 3_000,
    max_total_samples: int = 120_000,
    tile_rows: int = 128,
    batch_size: int = 8192,
    max_epochs: int = 50,
    val_frac: float = 0.1,
    patience: int = 5,
) -> None:
    logger = setup_logging(dataset_key, start=start, end=end)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("torch_device=%s", device)
    torch.manual_seed(RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)

    md = make_madurai_data()
    ds = load_subset(md, dataset_key, start=start, end=end)
    td = detect_time_dim(ds)
    if td is None:
        raise ValueError(f"{dataset_key} has no time dimension; pixel-MLP needs time to split train/test.")

    ds = ds.assign_coords({td: pd.to_datetime(ds[td].values)})
    times = pd.DatetimeIndex(ds[td].values).sort_values()
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
        len(ds.data_vars),
        len(times),
        cadence_guess(times),
    )

    tgt = pick_target_var(ds, target)
    feats = default_feature_vars(ds, tgt, features)
    logger.info("target=%s", tgt)
    logger.info("n_features=%d", len(feats))

    fine_shape = (
        int(ds[tgt].sizes.get("y", ds[tgt].shape[-2])),
        int(ds[tgt].sizes.get("x", ds[tgt].shape[-1])),
    )
    month_cache, row_map, col_map, row_float, col_float = build_monthly_anchor_cache(
        md,
        times,
        fine_shape,
        start=start,
        end=end,
        logger=logger,
    )
    anchor_feature_names = build_anchor_feature_names(month_cache)
    all_features = feats + anchor_feature_names

    keep_times = []
    for t in times:
        mk = _month_key(pd.Timestamp(t))
        anchors = month_cache.get(mk)
        if anchors and (anchors.has_modis or anchors.has_viirs):
            keep_times.append(t)
    if not keep_times:
        raise RuntimeError("No months with MODIS/VIIRS coverage after alignment.")

    keep_times = pd.DatetimeIndex(keep_times)
    n_test = max(1, int(round(len(keep_times) * test_frac)))
    test_times = keep_times[-n_test:]
    train_times = keep_times[:-n_test]
    logger.info("train_times=%d test_times=%d", len(train_times), len(test_times))

    rng = np.random.default_rng(RANDOM_SEED)
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

    sampler: Optional[ReservoirSampler] = None
    for t in train_times:
        Xs, ys = sample_xy_for_time(
            ds,
            td,
            t,
            tgt,
            feats,
            spec,
            rng,
            month_cache=month_cache,
            row_float=row_float,
            col_float=col_float,
            anchor_feature_names=anchor_feature_names,
        )
        if ys.size == 0:
            continue
        if sampler is None:
            sampler = ReservoirSampler(Xs.shape[1], spec.max_total_samples, rng)
        sampler.add(Xs, ys)
    if sampler is None or sampler.count == 0:
        raise RuntimeError("No valid training samples found (target all NaN/Inf).")

    X_train, y_train = sampler.get()
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    logger.info("train_samples=%d train_features=%d", X_train.shape[0], X_train.shape[1])

    train_medians = np.nanmedian(X_train, axis=0)
    train_medians = np.where(np.isfinite(train_medians), train_medians, 0.0)
    X_train = impute_with_train_medians(X_train, X_train)
    train_vmin_vmax = None
    try:
        y_train_f = y_train[np.isfinite(y_train)]
        if y_train_f.size:
            p2, p98 = np.percentile(y_train_f, [2, 98])
            train_vmin_vmax = (float(p2), float(p98))
    except Exception:
        train_vmin_vmax = None
    model = train_mlp(
        X_train,
        y_train,
        device=device,
        hidden=(256, 128, 64),
        dropout=0.0,
        batch_size=batch_size,
        lr=1e-3,
        max_epochs=max_epochs,
        val_frac=val_frac,
        patience=patience,
        seed=RANDOM_SEED,
        logger=logger,
    )

    try:
        model_path = MODELS_DIR / "pixel_mlp.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "train_medians": train_medians,
                "features": all_features,
            },
            model_path,
        )
        logger.info("saved_model=%s", model_path)
    except Exception as exc:
        logger.warning("model save failed: %s", exc)

    figure_paths: Dict[str, List[str]] = {}
    cutoff = pd.Timestamp("2026-01-01")
    pre_cutoff = [t for t in test_times if pd.Timestamp(t) < cutoff]
    figure_time = str(pd.Timestamp(pre_cutoff[-1] if pre_cutoff else test_times[-1]).date())
    roi_mask = None
    try:
        H = int(ds[tgt].sizes.get("y", ds[tgt].shape[-2]))
        W = int(ds[tgt].sizes.get("x", ds[tgt].shape[-1]))
        roi_mask = build_roi_mask(dataset_key, (H, W), logger)
    except Exception as exc:
        logger.warning("ROI mask skipped: %s", exc)

    rows = []
    eval_spec = SampleSpec(
        max_samples_per_time=10_000,
        min_samples_per_time=0,
        max_total_samples=50_000,
    )
    def _predict_sampled(model_in: nn.Module, X: np.ndarray, batch: int) -> np.ndarray:
        preds = []
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i : i + batch]).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                preds.append(model_in(xb).squeeze(-1).cpu().numpy())
        return np.concatenate(preds, axis=0)
    for t in test_times:
        dts = ds.sel({td: t})
        y_true, y_pred = predict_full_map(
            model,
            dts,
            tgt,
            feats,
            tile_rows=tile_rows,
            month_cache=month_cache,
            row_float=row_float,
            col_float=col_float,
            anchor_feature_names=anchor_feature_names,
            time_val=t,
            device=device,
            train_medians=train_medians,
        )
        m = np.isfinite(y_true) & np.isfinite(y_pred)
        met = compute_all(y_true, y_pred, mask=m, ratio=ratio_for_ergas, channel_axis=None)
        if np.any(m):
            err = y_pred[m] - y_true[m]
            rmse_sum = float(np.sqrt(np.sum(err ** 2)))
        else:
            rmse_sum = float("nan")
        Xs, ys = sample_xy_for_time(
            ds,
            td,
            t,
            tgt,
            feats,
            eval_spec,
            rng,
            month_cache=month_cache,
            row_float=row_float,
            col_float=col_float,
            anchor_feature_names=anchor_feature_names,
        )
        if ys.size == 0:
            rmse_sampled = float("nan")
            n_sampled = 0
        else:
            Xs = impute_with_train_medians(X_train, Xs)
            y_pred_s = _predict_sampled(model, Xs, batch_size)
            m_s = np.isfinite(ys) & np.isfinite(y_pred_s)
            if np.any(m_s):
                err_s = y_pred_s[m_s] - ys[m_s]
                rmse_sampled = float(np.sqrt(np.mean(err_s ** 2)))
            else:
                rmse_sampled = float("nan")
            n_sampled = int(np.sum(m_s))
        del Xs, ys
        rows.append(
            {
                "time": str(pd.Timestamp(t).date()),
                **met,
                "rmse_sum": rmse_sum,
                "rmse_sampled": rmse_sampled,
                "n_valid": int(np.sum(m)),
                "n_sampled": n_sampled,
            }
        )
        if str(pd.Timestamp(t).date()) == figure_time:
            fig_paths = save_prediction_figures(
                y_true=y_true,
                y_pred=y_pred,
                model_name="pixel_mlp",
                time_label=figure_time,
                roi_mask=roi_mask,
                vmin_vmax=train_vmin_vmax,
            )
            figure_paths.setdefault("pixel_mlp", []).extend([str(p) for p in fig_paths])
        del y_true, y_pred, m, dts
        gc.collect()

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "pixel_mlp_metrics.csv"
    df.to_csv(out_csv, index=False)
    figure_paths.setdefault("pixel_mlp", []).extend([str(p) for p in save_metric_plots(df, "pixel_mlp")])

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
        "device": str(device),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "val_frac": val_frac,
        "patience": patience,
        "out_dir": str(OUT_DIR),
        "figures_dir": str(FIGURES_DIR),
        "figures": figure_paths,
    }
    with (OUT_DIR / "run_config_and_summary.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    logger.info("DONE metrics_dir=%s", OUT_DIR)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("Pixel-wise MLP baseline (temporal alignment)")
    p.add_argument("--dataset", default="madurai_30m", choices=["madurai", "madurai_30m", "madurai_alphaearth_30m"])
    p.add_argument("--start", default=None, help="e.g., 2019-01-01")
    p.add_argument("--end", default=None, help="e.g., 2020-12-31")
    p.add_argument("--target", default=None, help="target variable name (recommended to set once)")
    p.add_argument("--features", default=None, nargs="+", help="feature variable names (optional)")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--ergas-ratio", type=float, default=33.3333333333)
    p.add_argument("--max-samples-per-time", type=int, default=15_000)
    p.add_argument("--min-samples-per-time", type=int, default=3_000)
    p.add_argument("--max-total-samples", type=int, default=120_000)
    p.add_argument("--tile-rows", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--max-epochs", type=int, default=5000)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    args = p.parse_args()

    main(
        dataset_key=args.dataset,
        start=args.start,
        end=args.end,
        target=args.target,
        features=args.features,
        test_frac=args.test_frac,
        ratio_for_ergas=args.ergas_ratio,
        max_samples_per_time=args.max_samples_per_time,
        min_samples_per_time=args.min_samples_per_time,
        max_total_samples=args.max_total_samples,
        tile_rows=args.tile_rows,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        val_frac=args.val_frac,
        patience=args.patience,
    )
