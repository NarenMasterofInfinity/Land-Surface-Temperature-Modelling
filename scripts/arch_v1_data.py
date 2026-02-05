from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from arch_v1_filters import (
    FilteringConfig,
    apply_range_mask,
    build_weak_label_from_modis_viirs,
    landsat_to_celsius,
)

import numpy as np
import pandas as pd
import zarr
import os


PATCH_VALID_FRAC_MIN = 0.30
DATE_VALID_FRAC_MIN = 0.15
DATE_MED_MIN_C = 10.0
DATE_MED_MAX_C = 60.0
MAX_RESAMPLE_TRIES = 10


def _to_str(arr: np.ndarray) -> np.ndarray:
    return np.array([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr])


def _get_landsat_scale_offset(root) -> Tuple[float, float]:
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


def _iter_chunks(shape: Tuple[int, int], chunks: Tuple[int, int]):
    h, w = shape
    ch_y, ch_x = chunks
    for y0 in range(0, h, ch_y):
        y1 = min(h, y0 + ch_y)
        for x0 in range(0, w, ch_x):
            x1 = min(w, x0 + ch_x)
            yield slice(y0, y1), slice(x0, x1)


def _landsat_to_celsius_np(arr: np.ndarray, scale: float, offset: float, fill_value: float) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    arr = np.where(arr == fill_value, np.nan, arr)
    if scale != 1.0 or offset != 0.0:
        arr = arr * scale + offset
    if np.isfinite(arr).any() and np.nanmedian(arr) > 200:
        arr = arr - 273.15
    return arr


def _apply_range_mask_np(arr: np.ndarray, min_c: float, max_c: float) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(arr) & (arr >= min_c) & (arr <= max_c)
    out = np.where(valid, arr, 0.0)
    return out.astype(np.float32), valid.astype(np.float32)


def _any_valid_landsat(arr2d, scale: float, offset: float, fill_value: float, min_c: float, max_c: float) -> bool:
    if isinstance(arr2d, np.ndarray):
        arr = _landsat_to_celsius_np(arr2d, scale, offset, fill_value)
        _, m = _apply_range_mask_np(arr, min_c, max_c)
        return bool(m.any())
    shape = arr2d.shape
    if hasattr(arr2d, "chunks") and arr2d.chunks is not None:
        chunks = arr2d.chunks
        if len(chunks) >= 2:
            chunks = (chunks[-2], chunks[-1])
        else:
            chunks = shape
    else:
        chunks = (min(256, shape[0]), min(256, shape[1]))
    for ys, xs in _iter_chunks(shape, chunks):
        block = np.asarray(arr2d[ys, xs])
        block = _landsat_to_celsius_np(block, scale, offset, fill_value)
        _, m = _apply_range_mask_np(block, min_c, max_c)
        if np.any(m):
            return True
    return False


def _landsat_date_stats(arr2d, scale: float, offset: float, fill_value: float, min_c: float, max_c: float) -> Dict[str, float]:
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
        block = _landsat_to_celsius_np(block, scale, offset, fill_value)
        finite = np.isfinite(block)
        n_total += int(finite.sum())
        v = block[finite]
        if v.size:
            vals.append(v.astype(np.float32, copy=False))
        block_filt, _ = _apply_range_mask_np(block, min_c, max_c)
        n_valid += int(np.isfinite(block_filt).sum())
    if vals:
        all_vals = np.concatenate(vals, axis=0)
        median_all = float(np.median(all_vals))
    else:
        median_all = float("nan")
    valid_fraction = (n_valid / n_total) if n_total > 0 else 0.0
    if vals and n_valid > 0:
        all_vals = np.concatenate(vals, axis=0)
        all_vals = all_vals[(all_vals >= min_c) & (all_vals <= max_c)]
        median_valid = float(np.median(all_vals)) if all_vals.size else float("nan")
    else:
        median_valid = float("nan")
    return {
        "median": median_all,
        "median_valid": median_valid,
        "valid_fraction": float(valid_fraction),
    }


def _extract_modis_np(modis_lr: np.ndarray) -> np.ndarray:
    if modis_lr.shape[0] >= 6:
        lst = modis_lr[0].astype(np.float32)
        qc = modis_lr[4].astype(np.float32)
    elif modis_lr.shape[0] >= 2:
        lst = modis_lr[0].astype(np.float32)
        qc = modis_lr[1].astype(np.float32)
    else:
        lst = modis_lr[0].astype(np.float32)
        qc = np.zeros_like(lst, dtype=np.float32)
    valid_qc = np.isfinite(qc) & (qc == 1)
    valid_lst = np.isfinite(lst) & (lst != -9999.0) & (lst > 0)
    valid = valid_qc & valid_lst
    lst = np.where(valid, lst, np.nan)
    if np.isfinite(lst).any() and np.nanmedian(lst) > 200:
        lst = lst - 273.15
    return lst


def _extract_viirs_np(viirs_lr: np.ndarray) -> np.ndarray:
    if viirs_lr.shape[0] >= 4:
        lst = viirs_lr[0].astype(np.float32)
        qc = viirs_lr[2].astype(np.float32)
    elif viirs_lr.shape[0] >= 2:
        lst = viirs_lr[0].astype(np.float32)
        qc = viirs_lr[1].astype(np.float32)
    else:
        lst = viirs_lr[0].astype(np.float32)
        qc = np.zeros_like(lst, dtype=np.float32)
    valid_qc = np.isfinite(qc) & (qc <= 1)
    valid_lst = np.isfinite(lst) & (lst != -9999.0) & (lst >= 273.0)
    valid = valid_qc & valid_lst
    lst = np.where(valid, lst, np.nan)
    lst = lst - 273.15
    return lst


def _bilinear_patch(arr: np.ndarray, r_f: np.ndarray, c_f: np.ndarray) -> np.ndarray:
    h_c, w_c = arr.shape
    r0 = np.floor(r_f).astype(np.int64)
    c0 = np.floor(c_f).astype(np.int64)
    r1 = np.clip(r0 + 1, 0, h_c - 1)
    c1 = np.clip(c0 + 1, 0, w_c - 1)
    fr = (r_f - r0)[:, None]
    fc = (c_f - c0)[None, :]
    v00 = arr[r0[:, None], c0[None, :]]
    v01 = arr[r0[:, None], c1[None, :]]
    v10 = arr[r1[:, None], c0[None, :]]
    v11 = arr[r1[:, None], c1[None, :]]
    w00 = (1 - fr) * (1 - fc)
    w01 = (1 - fr) * fc
    w10 = fr * (1 - fc)
    w11 = fr * fc
    vals = np.stack([v00, v01, v10, v11], axis=0)
    wts = np.stack([w00, w01, w10, w11], axis=0)
    valid = np.isfinite(vals)
    wts = np.where(valid, wts, 0.0)
    denom = np.sum(wts, axis=0)
    numer = np.sum(wts * np.nan_to_num(vals), axis=0)
    out = np.zeros_like(denom, dtype=np.float32)
    np.divide(numer, denom, out=out, where=denom > 0)
    out = np.where(denom > 0, out, np.nan)
    return out.astype(np.float32)


def _build_input_stack(comp: Dict[str, np.ndarray], input_keys: List[str]) -> np.ndarray:
    parts = []
    for k in input_keys:
        arr = comp[k]
        if arr.ndim == 2:
            arr = arr[None, ...]
        parts.append(arr)
    return np.concatenate(parts, axis=0)


def _build_time_maps(root_30m):
    daily_arr = root_30m["time"]["daily"]
    monthly_arr = root_30m["time"]["monthly"]
    daily_raw = _to_str(daily_arr[:])
    monthly_raw = _to_str(monthly_arr[:])
    daily_times = pd.to_datetime(daily_raw, format="%Y_%m_%d", errors="coerce").dropna()
    monthly_times = pd.to_datetime(monthly_raw, format="%Y_%m", errors="coerce").dropna()
    month_index = pd.DatetimeIndex(daily_times.to_period("M").to_timestamp())
    monthly_map = {t: i for i, t in enumerate(monthly_times)}
    daily_to_month = []
    for t in month_index:
        daily_to_month.append(monthly_map.get(t, -1))
    daily_to_month = np.array(daily_to_month)
    daily_idx = np.arange(len(daily_times), dtype=int)
    valid_month = daily_to_month >= 0
    daily_idx = daily_idx[valid_month]
    daily_to_month = daily_to_month[valid_month]
    daily_to_month_map = {int(t): int(m) for t, m in zip(daily_idx, daily_to_month)}
    doy = np.array([(t.dayofyear - 1) / 365.0 for t in daily_times], dtype=np.float32)
    return daily_idx, daily_to_month_map, doy


def scan_available_dates(
    root_30m,
    root_daily,
    daily_idx: np.ndarray,
    landsat_scale: float,
    landsat_offset: float,
    landsat_fill: float,
    min_c: float,
    max_c: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_daily = int(root_30m["time"]["daily"].shape[0])
    landsat_present = np.zeros(n_daily, dtype=bool)
    modis_present = np.zeros(n_daily, dtype=bool)
    viirs_present = np.zeros(n_daily, dtype=bool)

    for t in daily_idx:
        t = int(t)
        try:
            modis_lr = root_daily["products"]["modis"]["data"][t, :, :, :]
            modis_lst = _extract_modis_np(modis_lr)
            modis_present[t] = np.isfinite(modis_lst).any()
        except Exception:
            modis_present[t] = False
        try:
            viirs_lr = root_daily["products"]["viirs"]["data"][t, :, :, :]
            viirs_lst = _extract_viirs_np(viirs_lr)
            viirs_present[t] = np.isfinite(viirs_lst).any()
        except Exception:
            viirs_present[t] = False
        try:
            landsat_slice = root_30m["labels_30m"]["landsat"]["data"][t, 0, :, :]
            landsat_present[t] = _any_valid_landsat(
                landsat_slice, landsat_scale, landsat_offset, landsat_fill, min_c, max_c
            )
        except Exception:
            landsat_present[t] = False

    bad_landsat_dates = set()
    for t in daily_idx:
        t = int(t)
        if not landsat_present[t]:
            continue
        stats = _landsat_date_stats(
            root_30m["labels_30m"]["landsat"]["data"][t, 0, :, :],
            landsat_scale,
            landsat_offset,
            landsat_fill,
            min_c,
            max_c,
        )
        if (
            stats["valid_fraction"] < DATE_VALID_FRAC_MIN
            or not np.isfinite(stats["median_valid"])
            or stats["median_valid"] < DATE_MED_MIN_C
            or stats["median_valid"] > DATE_MED_MAX_C
        ):
            bad_landsat_dates.add(t)

    available_idx = [
        int(t)
        for t in daily_idx
        if int(t) not in bad_landsat_dates
        and (landsat_present[int(t)] or modis_present[int(t)] or viirs_present[int(t)])
    ]
    return np.array(available_idx, dtype=int), landsat_present, modis_present, viirs_present


def build_quality_report(
    root_30m,
    root_daily,
    daily_idx: np.ndarray,
    landsat_scale: float,
    landsat_offset: float,
    landsat_fill: float,
    min_c: float,
    max_c: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for t in daily_idx:
        t = int(t)
        row: Dict[str, object] = {"t": t}
        try:
            landsat_slice = root_30m["labels_30m"]["landsat"]["data"][t, 0, :, :]
            stats = _landsat_date_stats(landsat_slice, landsat_scale, landsat_offset, landsat_fill, min_c, max_c)
            row.update(
                {
                    "landsat_valid_frac": stats["valid_fraction"],
                    "landsat_median_valid": stats["median_valid"],
                }
            )
        except Exception:
            row.update({"landsat_valid_frac": 0.0, "landsat_median_valid": float("nan")})

        try:
            modis_lr = root_daily["products"]["modis"]["data"][t, :, :, :]
            modis_lst = _extract_modis_np(modis_lr)
            row["modis_valid_frac"] = float(np.isfinite(modis_lst).mean())
        except Exception:
            row["modis_valid_frac"] = 0.0

        try:
            viirs_lr = root_daily["products"]["viirs"]["data"][t, :, :, :]
            viirs_lst = _extract_viirs_np(viirs_lr)
            row["viirs_valid_frac"] = float(np.isfinite(viirs_lst).mean())
        except Exception:
            row["viirs_valid_frac"] = 0.0

        rows.append(row)
    return rows



class LSTSampleDataset(Dataset):
    """
    Dataset stub. Provide a list of dicts with keys:
    x, y_ls, m_ls, y_wk, m_wk, is_landsat, doy (optional).
    """

    def __init__(
        self,
        samples: Sequence[Dict[str, torch.Tensor]],
        *,
        filter_cfg: Optional[FilteringConfig] = None,
        weak_target_hw: Optional[Tuple[int, int]] = None,
    ):
        self.samples = list(samples)
        self.filter_cfg = filter_cfg or FilteringConfig()
        self.weak_target_hw = weak_target_hw

    @staticmethod
    def _ensure_ch(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.ndim == 2 else x

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = dict(self.samples[idx])

        if "y_ls_raw" in sample:
            y_ls = landsat_to_celsius(
                sample["y_ls_raw"],
                scale=self.filter_cfg.landsat_scale,
                offset=self.filter_cfg.landsat_offset,
                fill_value=self.filter_cfg.landsat_fill,
            )
            y_ls, m_ls_new = apply_range_mask(y_ls, min_c=self.filter_cfg.min_c, max_c=self.filter_cfg.max_c)
            y_ls = self._ensure_ch(y_ls)
            m_ls_new = self._ensure_ch(m_ls_new)
            if "m_ls" in sample:
                sample["m_ls"] = self._ensure_ch(sample["m_ls"]) * m_ls_new
            else:
                sample["m_ls"] = m_ls_new
            sample["y_ls"] = y_ls

        if "modis_lr" in sample or "viirs_lr" in sample:
            if self.weak_target_hw is None:
                raise ValueError("weak_target_hw is required to build weak labels from modis/viirs")
            modis_lr = sample.get("modis_lr")
            viirs_lr = sample.get("viirs_lr")
            y_wk, m_wk = build_weak_label_from_modis_viirs(modis_lr, viirs_lr, self.weak_target_hw)
            sample["y_wk"] = self._ensure_ch(y_wk)
            sample["m_wk"] = self._ensure_ch(m_wk)

        if "y_wk" in sample:
            y_wk = sample["y_wk"]
            y_wk, m_wk_new = apply_range_mask(y_wk, min_c=self.filter_cfg.min_c, max_c=self.filter_cfg.max_c)
            y_wk = self._ensure_ch(y_wk)
            m_wk_new = self._ensure_ch(m_wk_new)
            if "m_wk" in sample:
                sample["m_wk"] = self._ensure_ch(sample["m_wk"]) * m_wk_new
            else:
                sample["m_wk"] = m_wk_new
            sample["y_wk"] = y_wk

        if "y_ls" in sample:
            sample["y_ls"] = self._ensure_ch(sample["y_ls"])
        if "m_ls" in sample:
            sample["m_ls"] = self._ensure_ch(sample["m_ls"])

        return sample


class DummyLSTDataset(Dataset):
    def __init__(self, n: int, in_ch: int, h: int, w: int):
        self.n = n
        self.in_ch = in_ch
        self.h = h
        self.w = w

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.randn(self.in_ch, self.h, self.w)
        is_ls = torch.tensor(1 if idx % 5 == 0 else 0, dtype=torch.float32)
        y_ls = torch.randn(1, self.h, self.w)
        y_wk = torch.randn(1, self.h, self.w)
        m_ls = (torch.rand(1, self.h, self.w) > 0.3).float() * is_ls
        m_wk = (torch.rand(1, self.h, self.w) > 0.3).float() * (1.0 - is_ls)
        doy = torch.rand(1)
        return {
            "x": x,
            "y_ls": y_ls,
            "m_ls": m_ls,
            "y_wk": y_wk,
            "m_wk": m_wk,
            "is_landsat": is_ls,
            "doy": doy,
        }


def collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if vals[0].ndim == 0:
            out[k] = torch.stack([v.view(1) for v in vals], dim=0)
        else:
            out[k] = torch.stack(vals, dim=0)
    return out


class ZarrLSTDataset(Dataset):
    def __init__(
        self,
        *,
        root_30m,
        root_daily,
        allowed_t: np.ndarray,
        daily_to_month_map: Dict[int, int],
        doy_values: np.ndarray,
        landsat_present: np.ndarray,
        modis_present: np.ndarray,
        viirs_present: np.ndarray,
        patch_size: int,
        samples: int,
        seed: int,
        mode: str = "random",
        resample_invalid: bool = True,
        filter_cfg: Optional[FilteringConfig] = None,
    ):
        self.root_30m = root_30m
        self.root_daily = root_daily
        self.allowed_t = np.array(allowed_t, dtype=int)
        self.daily_to_month_map = daily_to_month_map
        self.doy_values = doy_values
        self.landsat_present = landsat_present
        self.modis_present = modis_present
        self.viirs_present = viirs_present
        self.patch_size = int(patch_size)
        self.samples = int(samples)
        self.seed = int(seed)
        self.mode = mode
        self.resample_invalid = resample_invalid
        self.filter_cfg = filter_cfg or FilteringConfig()

        self.g_era5 = root_30m["products_30m"]["era5"]["data"]
        self.g_landsat = root_30m["labels_30m"]["landsat"]["data"]
        self.g_s1 = root_30m["products_30m"]["sentinel1"]["data"]
        self.g_s2 = root_30m["products_30m"]["sentinel2"]["data"]
        self.g_dem = root_30m["static_30m"]["dem"]["data"]
        self.g_world = root_30m["static_30m"]["worldcover"]["data"]
        self.g_dyn = root_30m["static_30m"]["dynamic_world"]["data"]
        self.g_modis = root_daily["products"]["modis"]["data"]
        self.g_viirs = root_daily["products"]["viirs"]["data"]

        landsat_shape = self.g_landsat.shape
        self.h_hr, self.w_hr = landsat_shape[-2], landsat_shape[-1]

        modis_shape = self.g_modis.shape
        viirs_shape = self.g_viirs.shape
        h_lr_modis, w_lr_modis = modis_shape[-2], modis_shape[-1]
        h_lr_viirs, w_lr_viirs = viirs_shape[-2], viirs_shape[-1]
        self.row_float_modis = np.linspace(0, h_lr_modis - 1, self.h_hr, dtype=np.float64)
        self.col_float_modis = np.linspace(0, w_lr_modis - 1, self.w_hr, dtype=np.float64)
        self.row_float_viirs = np.linspace(0, h_lr_viirs - 1, self.h_hr, dtype=np.float64)
        self.col_float_viirs = np.linspace(0, w_lr_viirs - 1, self.w_hr, dtype=np.float64)

        self.items: List[Tuple[int, int, int]] = []
        if self.mode == "fixed":
            rng = np.random.default_rng(self.seed)
            for _ in range(self.samples):
                t = int(rng.choice(self.allowed_t))
                y0 = int(rng.integers(0, self.h_hr - self.patch_size + 1))
                x0 = int(rng.integers(0, self.w_hr - self.patch_size + 1))
                self.items.append((t, y0, x0))

    def __len__(self) -> int:
        return self.samples if self.mode == "random" else len(self.items)

    def channel_sizes(self) -> Dict[str, int]:
        return {
            "era5": int(self.g_era5.shape[1]),
            "s1": int(self.g_s1.shape[1]),
            "s2": int(self.g_s2.shape[1]),
            "dem": int(self.g_dem.shape[1]),
            "world": int(self.g_world.shape[1]),
            "dyn": int(self.g_dyn.shape[1]),
        }

    def _build_inputs(self, t: int, y0: int, x0: int) -> Dict[str, np.ndarray]:
        y1 = y0 + self.patch_size
        x1 = x0 + self.patch_size

        m = int(self.daily_to_month_map.get(int(t), -1))
        if m < 0:
            s1 = np.full((self.g_s1.shape[1], self.patch_size, self.patch_size), np.nan, dtype=np.float32)
            s2 = np.full((self.g_s2.shape[1], self.patch_size, self.patch_size), np.nan, dtype=np.float32)
        else:
            s1 = self.g_s1[m, :, y0:y1, x0:x1]
            s2 = self.g_s2[m, :, y0:y1, x0:x1]

        era5 = self.g_era5[t, :, y0:y1, x0:x1]
        dem = self.g_dem[0, :, y0:y1, x0:x1]
        world = self.g_world[0, :, y0:y1, x0:x1]
        dyn = self.g_dyn[0, :, y0:y1, x0:x1]

        return {
            "era5": era5,
            "s1": s1,
            "s2": s2,
            "dem": dem,
            "world": world,
            "dyn": dyn,
        }

    def _build_targets(self, t: int, y0: int, x0: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        y1 = y0 + self.patch_size
        x1 = x0 + self.patch_size

        if self.landsat_present[t]:
            y_ls_raw = self.g_landsat[t, 0, y0:y1, x0:x1]
            y_ls = _landsat_to_celsius_np(
                y_ls_raw,
                self.filter_cfg.landsat_scale,
                self.filter_cfg.landsat_offset,
                self.filter_cfg.landsat_fill,
            )
            y_ls, m_ls = _apply_range_mask_np(y_ls, self.filter_cfg.min_c, self.filter_cfg.max_c)
            is_ls = 1
        else:
            y_ls = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            m_ls = np.zeros_like(y_ls, dtype=np.float32)
            is_ls = 0

        modis_ok = bool(self.modis_present[t])
        viirs_ok = bool(self.viirs_present[t])
        if modis_ok:
            modis_lr = self.g_modis[t, :, :, :]
            modis_lst = _extract_modis_np(modis_lr)
            r_m = self.row_float_modis[y0:y1]
            c_m = self.col_float_modis[x0:x1]
            modis_up = _bilinear_patch(modis_lst, r_m, c_m)
        else:
            modis_up = None
        if viirs_ok:
            viirs_lr = self.g_viirs[t, :, :, :]
            viirs_lst = _extract_viirs_np(viirs_lr)
            r_v = self.row_float_viirs[y0:y1]
            c_v = self.col_float_viirs[x0:x1]
            viirs_up = _bilinear_patch(viirs_lst, r_v, c_v)
        else:
            viirs_up = None

        if modis_up is None and viirs_up is None:
            y_wk = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            m_wk = np.zeros_like(y_wk, dtype=np.float32)
        else:
            if modis_up is not None and viirs_up is not None:
                stacked = np.stack([modis_up, viirs_up], axis=0)
                valid = np.isfinite(stacked)
                denom = valid.sum(axis=0)
                numer = np.nansum(stacked, axis=0)
                y_wk = np.zeros_like(numer, dtype=np.float32)
                np.divide(numer, denom, out=y_wk, where=denom > 0)
                y_wk = np.where(denom > 0, y_wk, np.nan).astype(np.float32)
            elif modis_up is not None:
                y_wk = modis_up
            else:
                y_wk = viirs_up
            y_wk, m_wk = _apply_range_mask_np(y_wk, self.filter_cfg.min_c, self.filter_cfg.max_c)

        return y_ls, m_ls, y_wk, m_wk, is_ls

    def _build_sample(self, t: int, y0: int, x0: int) -> Dict[str, torch.Tensor]:
        comp = self._build_inputs(t, y0, x0)
        y_ls, m_ls, y_wk, m_wk, is_ls = self._build_targets(t, y0, x0)
        x = _build_input_stack(comp, ["era5", "s1", "s2", "dem", "world", "dyn"])
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        sample = {
            "x": torch.from_numpy(x),
            "y_ls": torch.from_numpy(y_ls[None, ...]),
            "m_ls": torch.from_numpy(m_ls[None, ...]),
            "y_wk": torch.from_numpy(y_wk[None, ...]),
            "m_wk": torch.from_numpy(m_wk[None, ...]),
            "is_landsat": torch.tensor(float(is_ls)),
            "doy": torch.tensor(float(self.doy_values[int(t)])),
        }
        return sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == "fixed":
            t, y0, x0 = self.items[idx]
        else:
            rng = np.random.default_rng(self.seed + idx)
            t = int(rng.choice(self.allowed_t))
            y0 = int(rng.integers(0, self.h_hr - self.patch_size + 1))
            x0 = int(rng.integers(0, self.w_hr - self.patch_size + 1))

        tries = 0
        sample = self._build_sample(t, y0, x0)
        valid_frac = float(sample["m_ls"].mean().item() if sample["is_landsat"] > 0.5 else sample["m_wk"].mean().item())
        while self.resample_invalid and valid_frac < PATCH_VALID_FRAC_MIN and tries < MAX_RESAMPLE_TRIES:
            rng = np.random.default_rng(self.seed + idx + tries + 1)
            t = int(rng.choice(self.allowed_t))
            y0 = int(rng.integers(0, self.h_hr - self.patch_size + 1))
            x0 = int(rng.integers(0, self.w_hr - self.patch_size + 1))
            sample = self._build_sample(t, y0, x0)
            valid_frac = float(
                sample["m_ls"].mean().item() if sample["is_landsat"] > 0.5 else sample["m_wk"].mean().item()
            )
            tries += 1
        return sample


def build_zarr_datasets(
    *,
    root_30m_path: str,
    root_daily_path: str,
    patch_size: int,
    train_frac: float,
    val_frac: float,
    seed: int,
    samples_per_epoch: int,
    samples_val: int,
    quality_csv: Optional[str] = None,
) -> Tuple[ZarrLSTDataset, ZarrLSTDataset, Dict[str, object]]:
    root_30m = zarr.open_group(root_30m_path, mode="r")
    root_daily = zarr.open_group(root_daily_path, mode="r")

    landsat_scale, landsat_offset = _get_landsat_scale_offset(root_30m)
    filter_cfg = FilteringConfig(landsat_scale=landsat_scale, landsat_offset=landsat_offset)

    daily_idx, daily_to_month_map, doy = _build_time_maps(root_30m)
    quality_rows = build_quality_report(
        root_30m,
        root_daily,
        daily_idx,
        landsat_scale,
        landsat_offset,
        filter_cfg.landsat_fill,
        filter_cfg.min_c,
        filter_cfg.max_c,
    )
    if quality_csv is not None:
        os.makedirs(os.path.dirname(quality_csv), exist_ok=True)
        pd.DataFrame(quality_rows).to_csv(quality_csv, index=False)
    available_idx, landsat_present, modis_present, viirs_present = scan_available_dates(
        root_30m,
        root_daily,
        daily_idx,
        landsat_scale,
        landsat_offset,
        filter_cfg.landsat_fill,
        filter_cfg.min_c,
        filter_cfg.max_c,
    )
    if available_idx.size == 0:
        raise RuntimeError("No available dates with landsat/modis/viirs data.")

    rng = np.random.default_rng(seed)
    idx = np.array(available_idx, dtype=int)
    rng.shuffle(idx)
    n_train = int(len(idx) * train_frac)
    n_val = int(len(idx) * val_frac)
    if n_train <= 0 or n_val <= 0 or (len(idx) - n_train - n_val) <= 0:
        raise RuntimeError("Invalid split fractions for available dates.")
    train_dates = idx[:n_train]
    val_dates = idx[n_train : n_train + n_val]

    train_set = ZarrLSTDataset(
        root_30m=root_30m,
        root_daily=root_daily,
        allowed_t=train_dates,
        daily_to_month_map=daily_to_month_map,
        doy_values=doy,
        landsat_present=landsat_present,
        modis_present=modis_present,
        viirs_present=viirs_present,
        patch_size=patch_size,
        samples=samples_per_epoch,
        seed=seed,
        mode="random",
        filter_cfg=filter_cfg,
    )
    val_set = ZarrLSTDataset(
        root_30m=root_30m,
        root_daily=root_daily,
        allowed_t=val_dates,
        daily_to_month_map=daily_to_month_map,
        doy_values=doy,
        landsat_present=landsat_present,
        modis_present=modis_present,
        viirs_present=viirs_present,
        patch_size=patch_size,
        samples=samples_val,
        seed=seed + 17,
        mode="fixed",
        filter_cfg=filter_cfg,
    )
    info = {
        "available_dates": len(available_idx),
        "train_dates": len(train_dates),
        "val_dates": len(val_dates),
        "landsat_present": int(landsat_present.sum()),
        "modis_present": int(modis_present.sum()),
        "viirs_present": int(viirs_present.sum()),
        "quality_csv": quality_csv,
    }
    return train_set, val_set, info
