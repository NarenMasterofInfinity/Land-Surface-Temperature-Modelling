from __future__ import annotations

from pathlib import Path
import argparse
import logging
import sys
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import zarr
import matplotlib.pyplot as plt

import os

from helper import eval_utils

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.deep.cnn_lr_hr.thermal_base.thermal_base_net import ThermalBaseNet, UpsampleHead
from baselines.deep.cnn_lr_hr.thermal_base.thermal_residual_net import ResidualNet

ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"

EPS_Y = 1e-6
LST_MIN_C = 10.0
LST_MAX_C = 70.0
PATCH_VALID_FRAC_MIN = 0.30
DATE_VALID_FRAC_MIN = 0.15
DATE_MED_MIN_C = 10.0
DATE_MED_MAX_C = 60.0
MAX_RESAMPLE_TRIES = 10

ERA5_TOP4 = [1, 4, 3, 0]


_ap = argparse.ArgumentParser()
_ap.add_argument("--run-name", default="thermal_base", help="Run name for output subdir")
_ap.add_argument("--seed", type=int, default=42)
_ap.add_argument("--batch-size", type=int, default=4)
_ap.add_argument("--patch-size", type=int, default=256)
_ap.add_argument("--full-scene", action="store_true")
_ap.add_argument("--samples-per-epoch", type=int, default=1000)
_ap.add_argument("--samples-val", type=int, default=500)
_ap.add_argument("--train-frac", type=float, default=0.7)
_ap.add_argument("--val-frac", type=float, default=0.1)
_ap.add_argument("--stage", choices=["base", "residual", "both"], default="both")
_ap.add_argument("--use-amp", action="store_true", default=True)
_ap.add_argument("--no-amp", action="store_true")
_ap.add_argument("--base-checkpoint", default=None)
_ap.add_argument("--s2-bands", default=None, help="Comma-separated band indices for S2")
_ap.add_argument("--s1-bands", default=None, help="Comma-separated band indices for S1")
_ap.add_argument("--metrics-samples", type=int, default=50)
_ap.add_argument("--sanity-check", action="store_true", help="Run a short data/model sanity check and exit")
_args = _ap.parse_args()

USE_AMP = _args.use_amp and not _args.no_amp
RUN_TAG = _args.run_name
PATCH_SIZE = int(_args.patch_size)
PATCH_H = PATCH_SIZE
PATCH_W = PATCH_SIZE

OUT_DIR = PROJECT_ROOT / "baselines" / "deep" / "cnn_lr_hr" / "thermal_base" / RUN_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)
log_path = OUT_DIR / "train.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("thermal_base")
logger.info("starting thermal_base run=%s", RUN_TAG)

root_30m = zarr.open_group(str(ROOT_30M), mode="r")
root_daily = zarr.open_group(str(ROOT_DAILY), mode="r")


def _to_str(arr):
    return np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
    )


def _qc_mask(qc: np.ndarray) -> np.ndarray:
    qc = qc.astype(np.float32, copy=False)
    finite = np.isfinite(qc)
    if not finite.any():
        return finite
    qc_max = float(np.nanmax(qc))
    if qc_max <= 1:
        return finite & (qc <= 1)
    if qc_max <= 2:
        return finite & (qc <= 2)
    if qc_max <= 3:
        return finite & (qc <= 2)
    if qc_max <= 10:
        return finite & (qc <= 3)
    return finite


def _extract_modis(modis_lr):
    if modis_lr.shape[0] >= 6:
        lst = modis_lr[0].astype(np.float32)
        qc = modis_lr[4].astype(np.float32)
    elif modis_lr.shape[0] >= 2:
        lst = modis_lr[0].astype(np.float32)
        qc = modis_lr[1].astype(np.float32)
    else:
        lst = modis_lr[0].astype(np.float32)
        qc = np.zeros_like(lst, dtype=np.float32)
    valid_qc = _qc_mask(qc)
    if float(np.mean(valid_qc)) < 0.01:
        valid_qc = np.isfinite(qc)
    lst = np.where(np.isfinite(lst) & (lst != -9999.0), lst, np.nan)
    if np.isfinite(lst).any() and np.nanmedian(lst) > 200:
        lst = lst - 273.15
    valid_lst = np.isfinite(lst) & (lst >= LST_MIN_C) & (lst <= LST_MAX_C)
    valid = valid_qc & valid_lst
    lst = np.where(valid, lst, np.nan)
    return lst.astype(np.float32, copy=False), valid.astype(np.float32)


def _extract_viirs(viirs_lr):
    if viirs_lr.shape[0] >= 4:
        lst = viirs_lr[0].astype(np.float32)
        qc = viirs_lr[2].astype(np.float32)
    elif viirs_lr.shape[0] >= 2:
        lst = viirs_lr[0].astype(np.float32)
        qc = viirs_lr[1].astype(np.float32)
    else:
        lst = viirs_lr[0].astype(np.float32)
        qc = np.zeros_like(lst, dtype=np.float32)
    valid_qc = _qc_mask(qc)
    if float(np.mean(valid_qc)) < 0.01:
        valid_qc = np.isfinite(qc)
    lst = np.where(np.isfinite(lst) & (lst != -9999.0), lst, np.nan)
    if np.isfinite(lst).any() and np.nanmedian(lst) > 200:
        lst = lst - 273.15
    valid_lst = np.isfinite(lst) & (lst >= LST_MIN_C) & (lst <= LST_MAX_C)
    valid = valid_qc & valid_lst
    lst = np.where(valid, lst, np.nan)
    return lst.astype(np.float32, copy=False), valid.astype(np.float32)


def _landsat_to_celsius(arr):
    arr = arr.astype(np.float32, copy=False)
    arr = np.where(arr == 149, np.nan, arr)
    g = root_30m["labels_30m"]["landsat"]
    attrs = dict(g.attrs)
    scale = attrs.get("scale_factor", attrs.get("scale", 1.0))
    offset = attrs.get("add_offset", attrs.get("offset", 0.0))
    if scale is None:
        scale = 1.0
    if offset is None:
        offset = 0.0
    if scale != 1.0 or offset != 0.0:
        arr = arr * float(scale) + float(offset)
    if np.isfinite(arr).any() and np.nanmedian(arr) > 200:
        arr = arr - 273.15
    return arr


def _apply_range_mask(arr):
    valid = np.isfinite(arr) & (arr >= LST_MIN_C) & (arr <= LST_MAX_C)
    out = np.where(valid, arr, np.nan)
    return out.astype(np.float32), valid


def _iter_chunks(shape, chunks):
    H, W = shape
    ch_y, ch_x = chunks
    for y0 in range(0, H, ch_y):
        y1 = min(H, y0 + ch_y)
        for x0 in range(0, W, ch_x):
            x1 = min(W, x0 + ch_x)
            yield slice(y0, y1), slice(x0, x1)


def _any_valid_landsat(arr2d):
    if isinstance(arr2d, np.ndarray):
        arr = _landsat_to_celsius(arr2d)
        _, m = _apply_range_mask(arr)
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
        block = _landsat_to_celsius(block)
        _, m = _apply_range_mask(block)
        if np.any(m):
            return True
    return False


def _landsat_date_stats(t_idx):
    arr2d = root_30m["labels_30m"]["landsat"]["data"][t_idx, 0, :, :]
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
    removed = 0
    for ys, xs in _iter_chunks(shape, chunks):
        block = np.asarray(arr2d[ys, xs])
        block = _landsat_to_celsius(block)
        valid = np.isfinite(block)
        removed += int((~valid).sum())
        if np.any(valid):
            vals.append(block[valid])
    if vals:
        all_vals = np.concatenate(vals)
        min_v = float(np.min(all_vals))
        max_v = float(np.max(all_vals))
        p1 = float(np.percentile(all_vals, 1))
        p5 = float(np.percentile(all_vals, 5))
        p95 = float(np.percentile(all_vals, 95))
        p99 = float(np.percentile(all_vals, 99))
        median_valid = float(np.median(all_vals))
        valid_fraction = float(all_vals.size / (shape[0] * shape[1]))
    else:
        min_v = max_v = p1 = p5 = p95 = p99 = float("nan")
        median_valid = float("nan")
        valid_fraction = 0.0
    return {
        "min": min_v,
        "max": max_v,
        "p1": p1,
        "p5": p5,
        "p95": p95,
        "p99": p99,
        "median_valid": median_valid,
        "valid_fraction": valid_fraction,
        "n_removed": removed,
    }


def _build_date_splits(daily_times, available_idx, seed, train_frac, val_frac):
    rng = np.random.default_rng(seed)
    idx = np.array(available_idx, dtype=int)
    rng.shuffle(idx)
    n_train = int(len(idx) * train_frac)
    n_val = int(len(idx) * val_frac)
    if n_train <= 0 or n_val <= 0 or (len(idx) - n_train - n_val) <= 0:
        raise SystemExit("Invalid split fractions for available dates.")
    train_dates = idx[:n_train]
    val_dates = idx[n_train : n_train + n_val]
    test_dates = idx[n_train + n_val :]
    split_rows = []
    for split_name, dates in (("train", train_dates), ("val", val_dates), ("test", test_dates)):
        for t in dates:
            split_rows.append({"split": split_name, "date": daily_times[int(t)].strftime("%Y-%m-%d")})
    split_path = OUT_DIR / "date_splits.csv"
    pd.DataFrame(split_rows).to_csv(split_path, index=False)
    logger.info("saved date splits: %s", split_path)
    return train_dates, val_dates, test_dates


def _build_items(dates, n_samples, seed, H_hr, W_hr):
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n_samples):
        t = int(rng.choice(dates))
        if _args.full_scene:
            y0 = 0
            x0 = 0
        else:
            y0 = int(rng.integers(0, H_hr - PATCH_H + 1))
            x0 = int(rng.integers(0, W_hr - PATCH_W + 1))
        items.append((t, y0, x0))
    return items


@dataclass
class BaseSample:
    modis_frames: np.ndarray
    modis_masks: np.ndarray
    viirs_frames: np.ndarray
    viirs_masks: np.ndarray
    era5: np.ndarray
    doy: np.ndarray
    static: np.ndarray
    target: np.ndarray
    target_valid: np.ndarray


class BaseDataset(Dataset):
    def __init__(
        self,
        items,
        daily_times,
        modis_present,
        viirs_present,
        landsat_present,
        row_float_hr_to_lr,
        col_float_hr_to_lr,
        era5_indices,
        H_hr,
        W_hr,
        H_lr,
        W_lr,
    ):
        self.items = list(items)
        self.daily_times = daily_times
        self.modis_present = modis_present
        self.viirs_present = viirs_present
        self.landsat_present = landsat_present
        self.row_hr = row_float_hr_to_lr
        self.col_hr = col_float_hr_to_lr
        self.era5_indices = era5_indices
        self.H_hr = H_hr
        self.W_hr = W_hr
        self.H_lr = H_lr
        self.W_lr = W_lr

        self.g_modis = root_daily["products"]["modis"]["data"]
        self.g_viirs = root_daily["products"]["viirs"]["data"]
        self.g_landsat = root_30m["labels_30m"]["landsat"]["data"]
        self.g_era5 = root_30m["products_30m"]["era5"]["data"]
        self.g_dem = root_30m["static_30m"]["dem"]["data"]
        self.g_world = root_30m["static_30m"]["worldcover"]["data"]
        self.g_dyn = root_30m["static_30m"]["dynamic_world"]["data"]

        self.coarse_h = max(1, int(round(PATCH_H * (self.H_lr / self.H_hr))))
        self.coarse_w = max(1, int(round(PATCH_W * (self.W_lr / self.W_hr))))
        self.coarse_h = min(self.coarse_h, self.H_lr)
        self.coarse_w = min(self.coarse_w, self.W_lr)
        self.max_resample = 10

    def __len__(self):
        return len(self.items)

    def _resize_patch(self, patch: np.ndarray, mode: str = "bilinear") -> np.ndarray:
        if patch.shape == (self.coarse_h, self.coarse_w):
            return patch.astype(np.float32, copy=False)
        x = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
        if mode == "nearest":
            x = F.interpolate(x, size=(self.coarse_h, self.coarse_w), mode=mode)
        else:
            x = F.interpolate(x, size=(self.coarse_h, self.coarse_w), mode=mode, align_corners=False)
        return x.squeeze(0).squeeze(0).numpy().astype(np.float32, copy=False)

    def _extract_frames(self, t, y0, x0):
        frames_m = []
        frames_v = []
        masks_m = []
        masks_v = []
        for dt in (2, 1, 0):
            tt = t - dt
            if tt < 0:
                frames_m.append(np.full((1, self.coarse_h, self.coarse_w), np.nan, dtype=np.float32))
                frames_v.append(np.full((1, self.coarse_h, self.coarse_w), np.nan, dtype=np.float32))
                masks_m.append(np.zeros((1, self.coarse_h, self.coarse_w), dtype=np.float32))
                masks_v.append(np.zeros((1, self.coarse_h, self.coarse_w), dtype=np.float32))
                continue

            m_ok = bool(self.modis_present[tt])
            v_ok = bool(self.viirs_present[tt])
            if m_ok:
                modis_lr = self.g_modis[tt, :, :, :]
                modis_lst, modis_mask = _extract_modis(modis_lr)
                r0 = int(round(y0 * self.H_lr / self.H_hr))
                c0 = int(round(x0 * self.W_lr / self.W_hr))
                r1 = min(self.H_lr, r0 + self.coarse_h)
                c1 = min(self.W_lr, c0 + self.coarse_w)
                patch = modis_lst[r0:r1, c0:c1]
                mask_patch = modis_mask[r0:r1, c0:c1]
                patch = self._resize_patch(patch, mode="bilinear")
                mask_patch = self._resize_patch(mask_patch, mode="nearest")
            else:
                patch = np.full((self.coarse_h, self.coarse_w), np.nan, dtype=np.float32)
                mask_patch = np.zeros_like(patch, dtype=np.float32)
            frames_m.append(patch[None, ...])
            masks_m.append(mask_patch[None, ...])

            if v_ok:
                viirs_lr = self.g_viirs[tt, :, :, :]
                viirs_lst, viirs_mask = _extract_viirs(viirs_lr)
                r0 = int(round(y0 * self.H_lr / self.H_hr))
                c0 = int(round(x0 * self.W_lr / self.W_hr))
                r1 = min(self.H_lr, r0 + self.coarse_h)
                c1 = min(self.W_lr, c0 + self.coarse_w)
                patch = viirs_lst[r0:r1, c0:c1]
                mask_patch = viirs_mask[r0:r1, c0:c1]
                patch = self._resize_patch(patch, mode="bilinear")
                mask_patch = self._resize_patch(mask_patch, mode="nearest")
            else:
                patch = np.full((self.coarse_h, self.coarse_w), np.nan, dtype=np.float32)
                mask_patch = np.zeros_like(patch, dtype=np.float32)
            frames_v.append(patch[None, ...])
            masks_v.append(mask_patch[None, ...])

        modis_frames = np.stack(frames_m, axis=0)
        modis_masks = np.stack(masks_m, axis=0)
        viirs_frames = np.stack(frames_v, axis=0)
        viirs_masks = np.stack(masks_v, axis=0)
        return modis_frames, modis_masks, viirs_frames, viirs_masks

    def _downsample_target(self, y_hr, valid_hr):
        row = np.linspace(0, y_hr.shape[0] - 1, self.coarse_h, dtype=np.float64)
        col = np.linspace(0, y_hr.shape[1] - 1, self.coarse_w, dtype=np.float64)
        y_lr = _bilinear_patch(y_hr, row, col)
        valid_lr = _bilinear_patch(valid_hr.astype(np.float32), row, col) > 0.5
        valid_lr = valid_lr & np.isfinite(y_lr)
        y_lr = np.nan_to_num(y_lr, nan=0.0, posinf=0.0, neginf=0.0)
        return y_lr.astype(np.float32), valid_lr

    def _accept_sample(self, valid_lr, modis_masks, viirs_masks, static_c) -> bool:
        target_valid_frac = float(np.mean(valid_lr)) if valid_lr.size else 0.0
        modis_valid_frac = float(np.mean(modis_masks)) if modis_masks.size else 0.0
        viirs_valid_frac = float(np.mean(viirs_masks)) if viirs_masks.size else 0.0
        static_valid_frac = float(np.mean(np.isfinite(static_c))) if static_c.size else 0.0
        if target_valid_frac < 0.30:
            return False
        return (modis_valid_frac >= 0.05) or (viirs_valid_frac >= 0.05) or (static_valid_frac >= 0.90)

    def _build_sample(self, t, y0, x0) -> BaseSample:
        modis_frames, modis_masks, viirs_frames, viirs_masks = self._extract_frames(t, y0, x0)

        era5 = self.g_era5[t, :, y0 : y0 + PATCH_H, x0 : x0 + PATCH_W]
        era5 = era5[self.era5_indices]
        era5 = torch.from_numpy(era5).float().unsqueeze(0)
        era5 = F.interpolate(era5, size=(self.coarse_h, self.coarse_w), mode="bilinear", align_corners=False)
        era5 = era5.squeeze(0).numpy()
        era5 = np.nan_to_num(era5, nan=0.0, posinf=0.0, neginf=0.0)

        date = pd.Timestamp(self.daily_times[int(t)])
        doy = float(date.dayofyear)
        sin_doy = np.sin(2 * np.pi * doy / 365.25)
        cos_doy = np.cos(2 * np.pi * doy / 365.25)
        doy_arr = np.array([sin_doy, cos_doy], dtype=np.float32)

        y_hr = self.g_landsat[t, 0, y0 : y0 + PATCH_H, x0 : x0 + PATCH_W]
        y_hr = _landsat_to_celsius(y_hr)
        y_hr, valid_hr = _apply_range_mask(y_hr)
        y_lr, valid_lr = self._downsample_target(y_hr, valid_hr)

        dem = self.g_dem[0, 0, y0 : y0 + PATCH_H, x0 : x0 + PATCH_W]
        world = self.g_world[0, 0, y0 : y0 + PATCH_H, x0 : x0 + PATCH_W]
        dyn = self.g_dyn[0, 0, y0 : y0 + PATCH_H, x0 : x0 + PATCH_W]
        dem_c = self._resize_patch(dem, mode="bilinear")
        world_c = self._resize_patch(world, mode="nearest")
        dyn_c = self._resize_patch(dyn, mode="nearest")
        static_c = np.stack([dem_c, world_c, dyn_c], axis=0).astype(np.float32, copy=False)

        sample = BaseSample(
            modis_frames=modis_frames,
            modis_masks=modis_masks,
            viirs_frames=viirs_frames,
            viirs_masks=viirs_masks,
            era5=era5,
            doy=doy_arr,
            static=static_c,
            target=y_lr,
            target_valid=valid_lr,
        )
        return sample

    def __getitem__(self, idx):
        t, y0, x0 = self.items[idx]
        sample = self._build_sample(t, y0, x0)
        if self._accept_sample(sample.target_valid, sample.modis_masks, sample.viirs_masks, sample.static):
            return sample
        for _ in range(self.max_resample):
            ridx = int(np.random.randint(0, len(self.items)))
            t, y0, x0 = self.items[ridx]
            sample = self._build_sample(t, y0, x0)
            if self._accept_sample(sample.target_valid, sample.modis_masks, sample.viirs_masks, sample.static):
                return sample
        return sample

    def get_components_at(self, t, y0, x0):
        return self.__getitem__(0)


class ResidualDataset(Dataset):
    def __init__(
        self,
        items,
        s2_bands,
        s1_bands,
        H_hr,
        W_hr,
        H_lr,
        W_lr,
        modis_present,
        viirs_present,
        daily_times,
        daily_to_month_map,
    ):
        self.items = list(items)
        self.s2_bands = s2_bands
        self.s1_bands = s1_bands
        self.H_hr = H_hr
        self.W_hr = W_hr
        self.H_lr = H_lr
        self.W_lr = W_lr
        self.modis_present = modis_present
        self.viirs_present = viirs_present
        self.daily_times = daily_times
        self.daily_to_month_map = daily_to_month_map

        self.g_s2 = root_30m["products_30m"]["sentinel2"]["data"]
        self.g_s1 = root_30m["products_30m"]["sentinel1"]["data"]
        self.g_dem = root_30m["static_30m"]["dem"]["data"]
        self.g_world = root_30m["static_30m"]["worldcover"]["data"]
        self.g_dyn = root_30m["static_30m"]["dynamic_world"]["data"]
        self.g_landsat = root_30m["labels_30m"]["landsat"]["data"]
        self.g_era5 = root_30m["products_30m"]["era5"]["data"]
        self.g_modis = root_daily["products"]["modis"]["data"]
        self.g_viirs = root_daily["products"]["viirs"]["data"]
        self.coarse_h = max(1, int(round(PATCH_H * (self.H_lr / self.H_hr))))
        self.coarse_w = max(1, int(round(PATCH_W * (self.W_lr / self.W_hr))))
        self.coarse_h = min(self.coarse_h, self.H_lr)
        self.coarse_w = min(self.coarse_w, self.W_lr)
        self.s2_ch = self.g_s2.shape[1]
        self.s1_ch = self.g_s1.shape[1]
        self.max_resample = 10

    def __len__(self):
        return len(self.items)

    def _accept_sample(self, valid_hr, modis_masks, viirs_masks, static_c) -> bool:
        target_valid_frac = float(np.mean(valid_hr)) if valid_hr.size else 0.0
        modis_valid_frac = float(np.mean(modis_masks)) if modis_masks.size else 0.0
        viirs_valid_frac = float(np.mean(viirs_masks)) if viirs_masks.size else 0.0
        static_valid_frac = float(np.mean(np.isfinite(static_c))) if static_c.size else 0.0
        if target_valid_frac < 0.30:
            return False
        return (modis_valid_frac >= 0.05) or (viirs_valid_frac >= 0.05) or (static_valid_frac >= 0.90)

    def _build_sample(self, t, y0, x0):
        y1 = y0 + PATCH_H
        x1 = x0 + PATCH_W

        m_idx = int(self.daily_to_month_map.get(int(t), -1))
        if m_idx < 0:
            s2 = np.full((self.s2_ch, PATCH_H, PATCH_W), np.nan, dtype=np.float32)
            s1 = np.full((self.s1_ch, PATCH_H, PATCH_W), np.nan, dtype=np.float32)
        else:
            s2 = self.g_s2[m_idx, :, y0:y1, x0:x1]
            s1 = self.g_s1[m_idx, :, y0:y1, x0:x1] if self.g_s1 is not None else None
        if self.s2_bands is not None:
            s2 = s2[self.s2_bands]
        if s1 is not None and self.s1_bands is not None:
            s1 = s1[self.s1_bands]

        dem = self.g_dem[0, :, y0:y1, x0:x1]
        world = self.g_world[0, :, y0:y1, x0:x1]
        dyn = self.g_dyn[0, :, y0:y1, x0:x1]
        lc = world
        dem_c = self._resize_patch(dem[0], mode="bilinear")
        world_c = self._resize_patch(world[0], mode="nearest")
        dyn_c = self._resize_patch(dyn[0], mode="nearest")
        static_c = np.stack([dem_c, world_c, dyn_c], axis=0).astype(np.float32, copy=False)

        era5 = self.g_era5[t, :, y0:y1, x0:x1]
        era5 = era5[ERA5_TOP4]
        era5 = np.nan_to_num(era5, nan=0.0, posinf=0.0, neginf=0.0)

        y_hr = self.g_landsat[t, 0, y0:y1, x0:x1]
        y_hr = _landsat_to_celsius(y_hr)
        y_hr, valid_hr = _apply_range_mask(y_hr)

        modis_frames, modis_masks, viirs_frames, viirs_masks, era5_coarse, doy = self._base_inputs(t, y0, x0)

        return (
            torch.from_numpy(s2).float(),
            torch.from_numpy(s1).float() if s1 is not None else None,
            torch.from_numpy(dem).float(),
            torch.from_numpy(lc).float(),
            torch.from_numpy(era5).float(),
            torch.from_numpy(y_hr).float(),
            torch.from_numpy(valid_hr).bool(),
            torch.from_numpy(modis_frames).float(),
            torch.from_numpy(modis_masks).float(),
            torch.from_numpy(viirs_frames).float(),
            torch.from_numpy(viirs_masks).float(),
            torch.from_numpy(era5_coarse).float(),
            torch.from_numpy(doy).float(),
            torch.from_numpy(static_c).float(),
        ), valid_hr, modis_masks, viirs_masks, static_c

    def __getitem__(self, idx):
        t, y0, x0 = self.items[idx]
        sample, valid_hr, modis_masks, viirs_masks, static_c = self._build_sample(t, y0, x0)
        if self._accept_sample(valid_hr, modis_masks, viirs_masks, static_c):
            return sample
        for _ in range(self.max_resample):
            ridx = int(np.random.randint(0, len(self.items)))
            t, y0, x0 = self.items[ridx]
            sample, valid_hr, modis_masks, viirs_masks, static_c = self._build_sample(t, y0, x0)
            if self._accept_sample(valid_hr, modis_masks, viirs_masks, static_c):
                return sample
        return sample

    def _resize_patch(self, patch: np.ndarray, mode: str = "bilinear") -> np.ndarray:
        if patch.shape == (self.coarse_h, self.coarse_w):
            return patch.astype(np.float32, copy=False)
        x = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
        if mode == "nearest":
            x = F.interpolate(x, size=(self.coarse_h, self.coarse_w), mode=mode)
        else:
            x = F.interpolate(x, size=(self.coarse_h, self.coarse_w), mode=mode, align_corners=False)
        return x.squeeze(0).squeeze(0).numpy().astype(np.float32, copy=False)

    def _base_inputs(self, t, y0, x0):
        frames_m = []
        frames_v = []
        masks_m = []
        masks_v = []
        for dt in (2, 1, 0):
            tt = t - dt
            if tt < 0:
                frames_m.append(np.full((1, self.coarse_h, self.coarse_w), np.nan, dtype=np.float32))
                frames_v.append(np.full((1, self.coarse_h, self.coarse_w), np.nan, dtype=np.float32))
                masks_m.append(np.zeros((1, self.coarse_h, self.coarse_w), dtype=np.float32))
                masks_v.append(np.zeros((1, self.coarse_h, self.coarse_w), dtype=np.float32))
                continue

            if bool(self.modis_present[tt]):
                modis_lr = self.g_modis[tt, :, :, :]
                modis_lst, modis_mask = _extract_modis(modis_lr)
                r0 = int(round(y0 * self.H_lr / self.H_hr))
                c0 = int(round(x0 * self.W_lr / self.W_hr))
                r1 = min(self.H_lr, r0 + self.coarse_h)
                c1 = min(self.W_lr, c0 + self.coarse_w)
                patch = self._resize_patch(modis_lst[r0:r1, c0:c1])
                mask_patch = self._resize_patch(modis_mask[r0:r1, c0:c1])
            else:
                patch = np.full((self.coarse_h, self.coarse_w), np.nan, dtype=np.float32)
                mask_patch = np.zeros_like(patch, dtype=np.float32)
            frames_m.append(patch[None, ...])
            masks_m.append(mask_patch[None, ...])

            if bool(self.viirs_present[tt]):
                viirs_lr = self.g_viirs[tt, :, :, :]
                viirs_lst, viirs_mask = _extract_viirs(viirs_lr)
                r0 = int(round(y0 * self.H_lr / self.H_hr))
                c0 = int(round(x0 * self.W_lr / self.W_hr))
                r1 = min(self.H_lr, r0 + self.coarse_h)
                c1 = min(self.W_lr, c0 + self.coarse_w)
                patch = self._resize_patch(viirs_lst[r0:r1, c0:c1])
                mask_patch = self._resize_patch(viirs_mask[r0:r1, c0:c1])
            else:
                patch = np.full((self.coarse_h, self.coarse_w), np.nan, dtype=np.float32)
                mask_patch = np.zeros_like(patch, dtype=np.float32)
            frames_v.append(patch[None, ...])
            masks_v.append(mask_patch[None, ...])

        modis_frames = np.stack(frames_m, axis=0)
        modis_masks = np.stack(masks_m, axis=0)
        viirs_frames = np.stack(frames_v, axis=0)
        viirs_masks = np.stack(masks_v, axis=0)

        era5 = self.g_era5[t, :, y0 : y0 + PATCH_H, x0 : x0 + PATCH_W]
        era5 = era5[ERA5_TOP4]
        era5 = np.nan_to_num(era5, nan=0.0, posinf=0.0, neginf=0.0)
        era5 = torch.from_numpy(era5).float().unsqueeze(0)
        era5 = F.interpolate(era5, size=(self.coarse_h, self.coarse_w), mode="bilinear", align_corners=False)
        era5 = era5.squeeze(0).numpy()
        era5 = np.nan_to_num(era5, nan=0.0, posinf=0.0, neginf=0.0)

        date = pd.Timestamp(self.daily_times[int(t)])
        doy = float(date.dayofyear)
        sin_doy = np.sin(2 * np.pi * doy / 365.25)
        cos_doy = np.cos(2 * np.pi * doy / 365.25)
        doy_arr = np.array([sin_doy, cos_doy], dtype=np.float32)

        return modis_frames, modis_masks, viirs_frames, viirs_masks, era5, doy_arr


def _bilinear_patch(arr, r_f, c_f):
    Hc, Wc = arr.shape
    r0 = np.floor(r_f).astype(np.int64)
    c0 = np.floor(c_f).astype(np.int64)
    r1 = np.clip(r0 + 1, 0, Hc - 1)
    c1 = np.clip(c0 + 1, 0, Wc - 1)
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
    out = np.where(denom > 0, np.sum(wts * np.nan_to_num(vals), axis=0) / denom, np.nan)
    return out.astype(np.float32)


def _make_optimizer(params_main, params_mamba, lr_main, weight_decay_main):
    return torch.optim.AdamW(
        [
            {"params": params_main, "lr": lr_main, "weight_decay": weight_decay_main},
            {"params": params_mamba, "lr": 0.5 * lr_main, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def _build_cosine_schedule_steps(optimizer, total_steps, warmup_steps, warmup_start, peak_lr, end_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return (warmup_start + (peak_lr - warmup_start) * (step / max(1, warmup_steps - 1))) / peak_lr
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
        cos = 0.5 * (1 + np.cos(np.pi * t))
        return (end_lr + (peak_lr - end_lr) * cos) / peak_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _save_loss_plot(df_hist, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df_hist["epoch"], df_hist["train_loss"], label="train")
    ax.plot(df_hist["epoch"], df_hist["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=150)
    plt.close(fig)


def _save_rmse_plot(df_hist, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df_hist["epoch"], df_hist["train_rmse"], label="train")
    ax.plot(df_hist["epoch"], df_hist["val_rmse"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("rmse")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rmse.png", dpi=150)
    plt.close(fig)


def _log_tensor_stats(name: str, arr: np.ndarray) -> str:
    if arr.size == 0 or not np.isfinite(arr).any():
        return f"{name}(all_nan)"
    return f"{name}(min={np.nanmin(arr):.3f} mean={np.nanmean(arr):.3f} max={np.nanmax(arr):.3f})"


def _safe_min_mean_max(arr: np.ndarray) -> tuple[float, float, float]:
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmin(arr)), float(np.nanmean(arr)), float(np.nanmax(arr))


def _log_base_samples(epoch: int, modis, modis_m, viirs, viirs_m, era5, doy, static, y, pred, prefix: str):
    if epoch % 2 != 0:
        return
    b = 0
    t = 2
    cy = modis.shape[-2] // 2
    cx = modis.shape[-1] // 2
    pred_np = pred[b].detach().cpu().numpy()
    parts = [
        f"{prefix} epoch={epoch} samples:",
        _log_tensor_stats("modis", modis[b, t, 0].cpu().numpy()),
        _log_tensor_stats("modis_m", modis_m[b, t, 0].cpu().numpy()),
        _log_tensor_stats("viirs", viirs[b, t, 0].cpu().numpy()),
        _log_tensor_stats("viirs_m", viirs_m[b, t, 0].cpu().numpy()),
        _log_tensor_stats("era5", era5[b].cpu().numpy()),
        _log_tensor_stats("static", static[b].cpu().numpy()),
        f"doy(sin={float(doy[b,0]):.3f} cos={float(doy[b,1]):.3f})",
        _log_tensor_stats("target", y[b].cpu().numpy()),
        _log_tensor_stats("pred", pred_np),
        f"center target={float(y[b, cy, cx]):.3f} pred={float(pred_np[cy, cx]):.3f}",
    ]
    logger.info(" | ".join(parts))


def _log_residual_samples(epoch: int, s2, s1, dem, lc, era5, base, y, pred, prefix: str):
    if epoch % 2 != 0:
        return
    b = 0
    cy = y.shape[-2] // 2
    cx = y.shape[-1] // 2
    pred_np = pred[b].detach().cpu().numpy()
    parts = [
        f"{prefix} epoch={epoch} samples:",
        _log_tensor_stats("s2", s2[b].cpu().numpy()),
        _log_tensor_stats("s1", s1[b].cpu().numpy()) if s1 is not None else "s1=None",
        _log_tensor_stats("dem", dem[b].cpu().numpy()),
        _log_tensor_stats("lc", lc[b].cpu().numpy()),
        _log_tensor_stats("era5", era5[b].cpu().numpy()),
        _log_tensor_stats("base", base[b].cpu().numpy()),
        _log_tensor_stats("target", y[b].cpu().numpy()),
        _log_tensor_stats("pred", pred_np),
        f"center target={float(y[b, cy, cx]):.3f} pred={float(pred_np[cy, cx]):.3f}",
    ]
    logger.info(" | ".join(parts))


def _collect_metrics(y_true, y_pred, valid_mask):
    met = eval_utils.compute_metrics(y_true, y_pred, roi_mask=valid_mask)
    return {k: float(v) for k, v in met.items()}


def _epoch_metrics(samples, model, device):
    rows = []
    for sample in samples:
        modis = torch.from_numpy(sample.modis_frames).unsqueeze(0).to(device)
        modis_m = torch.from_numpy(sample.modis_masks).unsqueeze(0).to(device)
        viirs = torch.from_numpy(sample.viirs_frames).unsqueeze(0).to(device)
        viirs_m = torch.from_numpy(sample.viirs_masks).unsqueeze(0).to(device)
        era5 = torch.from_numpy(sample.era5).unsqueeze(0).to(device)
        doy = torch.from_numpy(sample.doy).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(modis, viirs, era5, doy, modis_m, viirs_m).squeeze(0).squeeze(0).cpu().numpy()
        y = sample.target
        m = sample.target_valid
        rows.append(_collect_metrics(y, pred, m))
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return df.mean(numeric_only=True).to_dict()


def _eval_base_metrics(base_net, up_head, dataset, n_samples, device, out_path, mu_y, sigma_y, modis_mu, modis_sigma, viirs_mu, viirs_sigma, era5_mu, era5_sigma, static_mu, static_sigma):
    rng = np.random.default_rng(123)
    idx = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    rows = []
    base_net.eval()
    up_head.eval()
    with torch.no_grad():
        for i in idx:
            sample = dataset[i]
            modis = torch.from_numpy(sample.modis_frames).unsqueeze(0).to(device)
            modis_m = torch.from_numpy(sample.modis_masks).unsqueeze(0).to(device)
            viirs = torch.from_numpy(sample.viirs_frames).unsqueeze(0).to(device)
            viirs_m = torch.from_numpy(sample.viirs_masks).unsqueeze(0).to(device)
            era5 = torch.from_numpy(sample.era5).unsqueeze(0).to(device)
            doy = torch.from_numpy(sample.doy).unsqueeze(0).to(device)
            static = torch.from_numpy(sample.static).unsqueeze(0).to(device)
            modis = torch.nan_to_num(modis, nan=0.0, posinf=0.0, neginf=0.0)
            viirs = torch.nan_to_num(viirs, nan=0.0, posinf=0.0, neginf=0.0)
            static = torch.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)
            modis = (modis - modis_mu) / (modis_sigma + EPS_Y)
            viirs = (viirs - viirs_mu) / (viirs_sigma + EPS_Y)
            era5_mu_t = torch.as_tensor(era5_mu, device=device, dtype=era5.dtype)[None, :, None, None]
            era5_sigma_t = torch.as_tensor(era5_sigma, device=device, dtype=era5.dtype)[None, :, None, None]
            era5 = (era5 - era5_mu_t) / (era5_sigma_t + EPS_Y)
            static_mu_t = torch.as_tensor(static_mu, device=device, dtype=static.dtype)[None, :, None, None]
            static_sigma_t = torch.as_tensor(static_sigma, device=device, dtype=static.dtype)[None, :, None, None]
            static = (static - static_mu_t) / (static_sigma_t + EPS_Y)
            pred = base_net(modis, viirs, era5, doy, static, modis_m, viirs_m).squeeze(0).squeeze(0).cpu().numpy()
            pred = pred * (sigma_y + EPS_Y) + mu_y
            if np.any(sample.target_valid):
                rows.append(_collect_metrics(sample.target, pred, sample.target_valid))
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        return df.mean(numeric_only=True).to_dict()
    return {}


def _compute_target_stats(dataset, n_samples, seed=123):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    vals = []
    for i in idx:
        sample = dataset[i]
        m = sample.target_valid
        if np.any(m):
            vals.append(sample.target[m])
    if not vals:
        return 0.0, 1.0
    all_vals = np.concatenate(vals)
    mu = float(np.mean(all_vals))
    sigma = float(np.std(all_vals))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return mu, sigma


def _compute_base_input_stats(dataset, n_samples, seed=123):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    modis_vals = []
    viirs_vals = []
    era5_vals = []
    static_vals = []
    for i in idx:
        sample = dataset[i]
        m = sample.modis_masks > 0
        v = sample.viirs_masks > 0
        if np.any(m):
            modis_vals.append(sample.modis_frames[m])
        if np.any(v):
            viirs_vals.append(sample.viirs_frames[v])
        if sample.era5.size:
            era5_vals.append(sample.era5.reshape(sample.era5.shape[0], -1))
        if sample.static.size:
            static_vals.append(sample.static.reshape(sample.static.shape[0], -1))
    def _stat(vals):
        if not vals:
            return 0.0, 1.0
        arr = np.concatenate([v.reshape(-1) for v in vals], axis=0)
        finite = np.isfinite(arr)
        if not finite.any():
            return 0.0, 1.0
        mu = float(np.mean(arr[finite]))
        sigma = float(np.std(arr[finite]))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        return mu, sigma

    def _stat_ch(vals, ch):
        if not vals:
            return np.zeros(ch, dtype=np.float32), np.ones(ch, dtype=np.float32)
        arr = np.concatenate(vals, axis=1)
        mu = np.zeros(ch, dtype=np.float32)
        sigma = np.ones(ch, dtype=np.float32)
        for i in range(ch):
            finite = np.isfinite(arr[i])
            if not finite.any():
                continue
            mu[i] = float(np.mean(arr[i][finite]))
            s = float(np.std(arr[i][finite]))
            sigma[i] = s if np.isfinite(s) and s > 0 else 1.0
        return mu, sigma
    modis_mu, modis_sigma = _stat(modis_vals)
    viirs_mu, viirs_sigma = _stat(viirs_vals)
    era5_mu, era5_sigma = _stat_ch(era5_vals, dataset[0].era5.shape[0])
    static_mu, static_sigma = _stat_ch(static_vals, dataset[0].static.shape[0])
    return modis_mu, modis_sigma, viirs_mu, viirs_sigma, era5_mu, era5_sigma, static_mu, static_sigma


def _eval_residual_metrics(res_net, base_net, up_head, dataset, n_samples, device, out_path, mu_y, sigma_y, modis_mu, modis_sigma, viirs_mu, viirs_sigma, era5_mu, era5_sigma, static_mu, static_sigma):
    rng = np.random.default_rng(321)
    idx = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    rows = []
    base_net.eval()
    up_head.eval()
    res_net.eval()
    with torch.no_grad():
        for i in idx:
            s2, s1, dem, lc, era5, y, m, modis, modis_m, viirs, viirs_m, era5_c, doy, static_c = dataset[i]
            s2 = s2.unsqueeze(0).to(device)
            s1 = s1.unsqueeze(0).to(device) if s1 is not None else None
            dem = dem.unsqueeze(0).to(device)
            lc = lc.unsqueeze(0).to(device)
            era5 = era5.unsqueeze(0).to(device)
            modis = modis.unsqueeze(0).to(device)
            modis_m = modis_m.unsqueeze(0).to(device)
            viirs = viirs.unsqueeze(0).to(device)
            viirs_m = viirs_m.unsqueeze(0).to(device)
            era5_c = era5_c.unsqueeze(0).to(device)
            doy = doy.unsqueeze(0).to(device)
            static_c = static_c.unsqueeze(0).to(device)

            modis = torch.nan_to_num(modis, nan=0.0, posinf=0.0, neginf=0.0)
            viirs = torch.nan_to_num(viirs, nan=0.0, posinf=0.0, neginf=0.0)
            static = torch.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)
            modis = (modis - modis_mu) / (modis_sigma + EPS_Y)
            viirs = (viirs - viirs_mu) / (viirs_sigma + EPS_Y)
            era5_mu_t = torch.as_tensor(era5_mu, device=device, dtype=era5_c.dtype)[None, :, None, None]
            era5_sigma_t = torch.as_tensor(era5_sigma, device=device, dtype=era5_c.dtype)[None, :, None, None]
            era5_c = (era5_c - era5_mu_t) / (era5_sigma_t + EPS_Y)
            static_mu_t = torch.as_tensor(static_mu, device=device, dtype=static_c.dtype)[None, :, None, None]
            static_sigma_t = torch.as_tensor(static_sigma, device=device, dtype=static_c.dtype)[None, :, None, None]
            static_c = (static_c - static_mu_t) / (static_sigma_t + EPS_Y)

            base = base_net(modis, viirs, era5_c, doy, static_c, modis_m, viirs_m)
            base_hr = up_head(base)
            base_hr = base_hr * (sigma_y + EPS_Y) + mu_y
            pred = res_net(base_hr, s2, s1, dem, lc, era5, return_residual=False).squeeze(0).squeeze(0).cpu().numpy()
            rows.append(_collect_metrics(y.numpy(), pred, m.numpy()))
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        return df.mean(numeric_only=True).to_dict()
    return {}


def _select_indices(raw, name):
    if raw is None:
        return None
    out = []
    for v in raw.split(","):
        v = v.strip()
        if not v:
            continue
        out.append(int(v))
    logger.info("using %s bands: %s", name, out)
    return out


def _split_param_groups(model):
    mamba_params = []
    main_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "mamba" in n.lower() or "temporal" in n.lower():
            mamba_params.append(p)
        else:
            main_params.append(p)
    return main_params, mamba_params


# ---- time indices ----
daily_raw = _to_str(root_30m["time"]["daily"][:])
monthly_raw = _to_str(root_30m["time"]["monthly"][:])

daily_times = pd.to_datetime(daily_raw, format="%Y_%m_%d", errors="coerce").dropna()
monthly_times = pd.to_datetime(monthly_raw, format="%Y_%m", errors="coerce").dropna()

daily_idx = np.arange(len(daily_times), dtype=int)
month_index = pd.DatetimeIndex(daily_times.to_period("M").to_timestamp())
monthly_map = {t: i for i, t in enumerate(monthly_times)}

daily_to_month = []
for t in daily_times:
    m = t.to_period("M").to_timestamp()
    daily_to_month.append(monthly_map.get(m, -1))
daily_to_month = np.array(daily_to_month)
valid_month = daily_to_month >= 0
daily_idx = daily_idx[valid_month]
daily_to_month = daily_to_month[valid_month]
daily_to_month_map = {int(t): int(m) for t, m in zip(daily_idx, daily_to_month)}

landsat_shape = root_30m["labels_30m"]["landsat"]["data"].shape
H_hr, W_hr = landsat_shape[-2], landsat_shape[-1]
if _args.full_scene:
    PATCH_H = int(H_hr)
    PATCH_W = int(W_hr)
modis_shape = root_daily["products"]["modis"]["data"].shape
viirs_shape = root_daily["products"]["viirs"]["data"].shape
H_lr_modis, W_lr_modis = modis_shape[-2], modis_shape[-1]
H_lr_viirs, W_lr_viirs = viirs_shape[-2], viirs_shape[-1]

row_float_modis = np.linspace(0, H_lr_modis - 1, H_hr, dtype=np.float64)
col_float_modis = np.linspace(0, W_lr_modis - 1, W_hr, dtype=np.float64)

# ---- availability scan ----
landsat_present = np.zeros(len(daily_times), dtype=bool)
modis_present = np.zeros(len(daily_times), dtype=bool)
viirs_present = np.zeros(len(daily_times), dtype=bool)

logger.info("scanning availability across %d daily indices", len(daily_idx))
for t in daily_idx:
    t = int(t)
    try:
        modis_lr = root_daily["products"]["modis"]["data"][t, :, :, :]
        modis_lst, _ = _extract_modis(modis_lr)
        modis_present[t] = np.isfinite(modis_lst).any()
    except Exception as exc:
        logger.warning("modis availability failed t=%d (%s)", t, exc)
        modis_present[t] = False
    try:
        viirs_lr = root_daily["products"]["viirs"]["data"][t, :, :, :]
        viirs_lst, _ = _extract_viirs(viirs_lr)
        viirs_present[t] = np.isfinite(viirs_lst).any()
    except Exception as exc:
        logger.warning("viirs availability failed t=%d (%s)", t, exc)
        viirs_present[t] = False
    try:
        landsat_slice = root_30m["labels_30m"]["landsat"]["data"][t, 0, :, :]
        landsat_present[t] = _any_valid_landsat(landsat_slice)
    except Exception as exc:
        logger.warning("landsat availability failed t=%d (%s)", t, exc)
        landsat_present[t] = False

# per-date stats (log only)
for t in daily_idx:
    t = int(t)
    if not landsat_present[t]:
        continue
    stats = _landsat_date_stats(t)
    logger.info(
        "landsat_stats t=%d min=%.2f p1=%.2f p5=%.2f p95=%.2f p99=%.2f max=%.2f "
        "median=%.2f valid_frac=%.3f removed=%d",
        t,
        stats["min"],
        stats["p1"],
        stats["p5"],
        stats["p95"],
        stats["p99"],
        stats["max"],
        stats["median_valid"],
        stats["valid_fraction"],
        stats["n_removed"],
    )

# use common_dates.csv as source of truth for available days
common_path = PROJECT_ROOT / "common_dates.csv"
if not common_path.exists():
    raise SystemExit(f"common_dates.csv not found: {common_path}")
common_df = pd.read_csv(common_path)
common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna().dt.normalize()
common_set = set(common_dates)
daily_norm = pd.DatetimeIndex(daily_times).normalize()
available_idx = [int(t) for t in daily_idx if daily_norm[int(t)] in common_set]
if not available_idx:
    raise SystemExit("No available dates after applying common_dates.csv.")

logger.info(
    "available days=%d from common_dates.csv (landsat=%d modis=%d viirs=%d)",
    len(available_idx),
    int(landsat_present.sum()),
    int(modis_present.sum()),
    int(viirs_present.sum()),
)

train_dates, val_dates, test_dates = _build_date_splits(
    daily_times, available_idx, _args.seed, _args.train_frac, _args.val_frac
)

train_items = _build_items(train_dates, _args.samples_per_epoch, seed=11, H_hr=H_hr, W_hr=W_hr)
val_items = _build_items(val_dates, _args.samples_val, seed=22, H_hr=H_hr, W_hr=W_hr)

s2_bands = _select_indices(_args.s2_bands, "s2")
s1_bands = _select_indices(_args.s1_bands, "s1")

if _args.sanity_check:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ds = BaseDataset(
        train_items,
        daily_times,
        modis_present,
        viirs_present,
        landsat_present,
        row_float_modis,
        col_float_modis,
        ERA5_TOP4,
        H_hr,
        W_hr,
        H_lr_modis,
        W_lr_modis,
    )
    sample = base_ds[0]
    base_net = ThermalBaseNet(era5_k=len(ERA5_TOP4)).to(device)
    base_net.eval()
    with torch.no_grad():
        modis = torch.from_numpy(sample.modis_frames).unsqueeze(0).to(device)
        modis_m = torch.from_numpy(sample.modis_masks).unsqueeze(0).to(device)
        viirs = torch.from_numpy(sample.viirs_frames).unsqueeze(0).to(device)
        viirs_m = torch.from_numpy(sample.viirs_masks).unsqueeze(0).to(device)
        era5 = torch.from_numpy(sample.era5).unsqueeze(0).to(device)
        doy = torch.from_numpy(sample.doy).unsqueeze(0).to(device)
        static = torch.from_numpy(sample.static).unsqueeze(0).to(device)
        pred = base_net(modis, viirs, era5, doy, static, modis_m, viirs_m).squeeze(0).squeeze(0).cpu().numpy()
        logger.info(
            "sanity base pred shape=%s min=%.3f max=%.3f mean=%.3f",
            pred.shape,
            float(np.nanmin(pred)),
            float(np.nanmax(pred)),
            float(np.nanmean(pred)),
        )
    logger.info("sanity check complete")
    raise SystemExit(0)

def _train_base():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_net = ThermalBaseNet(era5_k=len(ERA5_TOP4)).to(device)
    up_head = UpsampleHead(in_ch=1).to(device)

    if _args.batch_size <= 4:
        peak_lr = 2e-4
    else:
        peak_lr = 3e-4
    main_params, mamba_params = _split_param_groups(base_net)
    opt = _make_optimizer(main_params, mamba_params, lr_main=peak_lr, weight_decay_main=1e-4)

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    base_ds = BaseDataset(
        train_items,
        daily_times,
        modis_present,
        viirs_present,
        landsat_present,
        row_float_modis,
        col_float_modis,
        ERA5_TOP4,
        H_hr,
        W_hr,
        H_lr_modis,
        W_lr_modis,
    )
    val_ds = BaseDataset(
        val_items,
        daily_times,
        modis_present,
        viirs_present,
        landsat_present,
        row_float_modis,
        col_float_modis,
        ERA5_TOP4,
        H_hr,
        W_hr,
        H_lr_modis,
        W_lr_modis,
    )

    train_loader = DataLoader(base_ds, batch_size=_args.batch_size, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_ds, batch_size=_args.batch_size, shuffle=False, collate_fn=lambda x: x)
    total_steps = 80 * max(1, len(train_loader))
    warmup_steps = 5 * max(1, len(train_loader))
    scheduler = _build_cosine_schedule_steps(
        opt, total_steps=total_steps, warmup_steps=warmup_steps, warmup_start=1e-5, peak_lr=peak_lr, end_lr=3e-6
    )

    mu_y, sigma_y = _compute_target_stats(base_ds, n_samples=min(500, len(base_ds)))
    modis_mu, modis_sigma, viirs_mu, viirs_sigma, era5_mu, era5_sigma, static_mu, static_sigma = _compute_base_input_stats(
        base_ds, n_samples=min(500, len(base_ds))
    )
    logger.info("base target stats mu=%.3f sigma=%.3f", mu_y, sigma_y)
    logger.info(
        "base input stats modis_mu=%.3f modis_sigma=%.3f viirs_mu=%.3f viirs_sigma=%.3f era5_mu_mean=%.3f era5_sigma_mean=%.3f static_mu_mean=%.3f static_sigma_mean=%.3f",
        modis_mu,
        modis_sigma,
        viirs_mu,
        viirs_sigma,
        float(np.mean(era5_mu)),
        float(np.mean(era5_sigma)),
        float(np.mean(static_mu)),
        float(np.mean(static_sigma)),
    )

    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience = 12
    bad_epochs = 0

    for epoch in range(1, 81):
        base_net.train()
        up_head.train()
        train_losses = []
        train_sq = 0.0
        train_n = 0
        valid_batches = 0
        valid_pix = 0
        total_pix = 0
        stats_batches = 0
        stats_target = []
        stats_pred = []
        optim_steps = 0

        opt.zero_grad(set_to_none=True)
        did_sample_log = False
        did_shape_log = False
        zero_pred_steps = 0
        first_low_valid = None
        for step, batch in enumerate(train_loader, start=1):
            modis = torch.from_numpy(np.stack([b.modis_frames for b in batch], axis=0)).to(device)
            modis_m = torch.from_numpy(np.stack([b.modis_masks for b in batch], axis=0)).to(device)
            viirs = torch.from_numpy(np.stack([b.viirs_frames for b in batch], axis=0)).to(device)
            viirs_m = torch.from_numpy(np.stack([b.viirs_masks for b in batch], axis=0)).to(device)
            era5 = torch.from_numpy(np.stack([b.era5 for b in batch], axis=0)).to(device)
            doy = torch.from_numpy(np.stack([b.doy for b in batch], axis=0)).to(device)
            static = torch.from_numpy(np.stack([b.static for b in batch], axis=0)).to(device)
            y = torch.from_numpy(np.stack([b.target for b in batch], axis=0)).to(device)
            m = torch.from_numpy(np.stack([b.target_valid for b in batch], axis=0)).to(device)

            modis = torch.nan_to_num(modis, nan=0.0, posinf=0.0, neginf=0.0)
            viirs = torch.nan_to_num(viirs, nan=0.0, posinf=0.0, neginf=0.0)
            static = torch.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)
            modis = (modis - modis_mu) / (modis_sigma + EPS_Y)
            viirs = (viirs - viirs_mu) / (viirs_sigma + EPS_Y)
            era5_mu_t = torch.as_tensor(era5_mu, device=device, dtype=era5.dtype)[None, :, None, None]
            era5_sigma_t = torch.as_tensor(era5_sigma, device=device, dtype=era5.dtype)[None, :, None, None]
            era5 = (era5 - era5_mu_t) / (era5_sigma_t + EPS_Y)
            static_mu_t = torch.as_tensor(static_mu, device=device, dtype=static.dtype)[None, :, None, None]
            static_sigma_t = torch.as_tensor(static_sigma, device=device, dtype=static.dtype)[None, :, None, None]
            static = (static - static_mu_t) / (static_sigma_t + EPS_Y)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                pred = base_net(modis, viirs, era5, doy, static, modis_m, viirs_m)
                pred = pred.squeeze(1)
                if not torch.isfinite(pred).all():
                    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                if m.any():
                    y_norm = (y - mu_y) / (sigma_y + EPS_Y)
                    loss = F.smooth_l1_loss(pred[m], y_norm[m])
                else:
                    loss = None

            if m.any():
                pred_std = float(pred[m].std().detach().cpu())
                target_std = float(y[m].std().detach().cpu())
                valid_pix = int(m.sum().item())
            else:
                pred_std = 0.0
                target_std = 0.0
                valid_pix = 0
            modis_v = float(modis_m.mean().detach().cpu())
            viirs_v = float(viirs_m.mean().detach().cpu())
            static_v = float(torch.isfinite(static).float().mean().detach().cpu())
            modis_std = float(modis[modis_m > 0].std().detach().cpu()) if modis_m.any() else 0.0
            viirs_std = float(viirs[viirs_m > 0].std().detach().cpu()) if viirs_m.any() else 0.0
            era5_std = float(era5.std().detach().cpu())
            static_std = float(static.std().detach().cpu())
            logger.info(
                "batch diag pred_std=%.4f target_std=%.4f valid_pix=%d modis_v=%.3f viirs_v=%.3f static_v=%.3f modis_std=%.3f viirs_std=%.3f era5_std=%.3f static_std=%.3f",
                pred_std,
                target_std,
                valid_pix,
                modis_v,
                viirs_v,
                static_v,
                modis_std,
                viirs_std,
                era5_std,
                static_std,
            )
            if pred_std < 1e-3:
                zero_pred_steps += 1
                if first_low_valid is None and (modis_v < 0.01 and viirs_v < 0.01 and static_v < 0.5):
                    first_low_valid = (step, modis_v, viirs_v, static_v, valid_pix)
            else:
                zero_pred_steps = 0
            if zero_pred_steps >= 20:
                if first_low_valid is not None:
                    s, mv, vv, sv, vp = first_low_valid
                    logger.error(
                        "collapse detected: first low-valid batch step=%d modis_v=%.3f viirs_v=%.3f static_v=%.3f valid_pix=%d",
                        s,
                        mv,
                        vv,
                        sv,
                        vp,
                    )
                raise RuntimeError("collapse detected: pred_std ~ 0 for many steps")

            if not did_shape_log:
                pred_c = pred * (sigma_y + EPS_Y) + mu_y
                pmin, pmean, pmax = _safe_min_mean_max(pred_c[0].detach().cpu().numpy())
                tmin, tmean, tmax = _safe_min_mean_max(y[0].detach().cpu().numpy())
                logger.info(
                    "train shapes modis=%s viirs=%s era5=%s static=%s pred=%s y=%s m=%s",
                    tuple(modis.shape),
                    tuple(viirs.shape),
                    tuple(era5.shape),
                    tuple(static.shape),
                    tuple(pred.shape),
                    tuple(y.shape),
                    tuple(m.shape),
                )
                logger.info(
                    "train sample pred(min=%.3f mean=%.3f max=%.3f) target(min=%.3f mean=%.3f max=%.3f)",
                    pmin,
                    pmean,
                    pmax,
                    tmin,
                    tmean,
                    tmax,
                )
                did_shape_log = True

            if (epoch % 2 == 0) and not did_sample_log:
                pred_c = pred * (sigma_y + EPS_Y) + mu_y
                _log_base_samples(epoch, modis, modis_m, viirs, viirs_m, era5, doy, static, y, pred_c, "train/base")
                did_sample_log = True

            if loss is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(list(base_net.parameters()), 1.0)
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                optim_steps += 1

                if m.any():
                    pred_c = pred * (sigma_y + EPS_Y) + mu_y
                    err = (pred_c - y)[m]
                    train_sq += float((err * err).sum().item())
                    train_n += int(err.numel())
                    valid_batches += 1
                    valid_pix += int(m.sum().item())
                    total_pix += int(m.numel())
                    train_losses.append(loss.item())
                    if stats_batches < 3:
                        stats_batches += 1
                        stats_target.append(y[m].detach().float().cpu())
                        stats_pred.append(pred[m].detach().float().cpu())

        base_net.eval()
        up_head.eval()
        val_losses = []
        val_sq = 0.0
        val_n = 0
        with torch.no_grad():
            did_shape_log = False
            for batch in val_loader:
                modis = torch.from_numpy(np.stack([b.modis_frames for b in batch], axis=0)).to(device)
                modis_m = torch.from_numpy(np.stack([b.modis_masks for b in batch], axis=0)).to(device)
                viirs = torch.from_numpy(np.stack([b.viirs_frames for b in batch], axis=0)).to(device)
                viirs_m = torch.from_numpy(np.stack([b.viirs_masks for b in batch], axis=0)).to(device)
                era5 = torch.from_numpy(np.stack([b.era5 for b in batch], axis=0)).to(device)
                doy = torch.from_numpy(np.stack([b.doy for b in batch], axis=0)).to(device)
                static = torch.from_numpy(np.stack([b.static for b in batch], axis=0)).to(device)
                y = torch.from_numpy(np.stack([b.target for b in batch], axis=0)).to(device)
                m = torch.from_numpy(np.stack([b.target_valid for b in batch], axis=0)).to(device)

                modis = torch.nan_to_num(modis, nan=0.0, posinf=0.0, neginf=0.0)
                viirs = torch.nan_to_num(viirs, nan=0.0, posinf=0.0, neginf=0.0)
                static = torch.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)
                modis = (modis - modis_mu) / (modis_sigma + EPS_Y)
                viirs = (viirs - viirs_mu) / (viirs_sigma + EPS_Y)
                era5_mu_t = torch.as_tensor(era5_mu, device=device, dtype=era5.dtype)[None, :, None, None]
                era5_sigma_t = torch.as_tensor(era5_sigma, device=device, dtype=era5.dtype)[None, :, None, None]
                era5 = (era5 - era5_mu_t) / (era5_sigma_t + EPS_Y)
                static_mu_t = torch.as_tensor(static_mu, device=device, dtype=static.dtype)[None, :, None, None]
                static_sigma_t = torch.as_tensor(static_sigma, device=device, dtype=static.dtype)[None, :, None, None]
                static = (static - static_mu_t) / (static_sigma_t + EPS_Y)

                pred = base_net(modis, viirs, era5, doy, static, modis_m, viirs_m).squeeze(1)
                if not torch.isfinite(pred).all():
                    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

                if not did_shape_log:
                    pred_c = pred * (sigma_y + EPS_Y) + mu_y
                    pmin, pmean, pmax = _safe_min_mean_max(pred_c[0].detach().cpu().numpy())
                    tmin, tmean, tmax = _safe_min_mean_max(y[0].detach().cpu().numpy())
                    logger.info(
                        "val shapes modis=%s viirs=%s era5=%s static=%s pred=%s y=%s m=%s",
                        tuple(modis.shape),
                        tuple(viirs.shape),
                        tuple(era5.shape),
                        tuple(static.shape),
                        tuple(pred.shape),
                        tuple(y.shape),
                        tuple(m.shape),
                    )
                    logger.info(
                        "val sample pred(min=%.3f mean=%.3f max=%.3f) target(min=%.3f mean=%.3f max=%.3f)",
                        pmin,
                        pmean,
                        pmax,
                        tmin,
                        tmean,
                        tmax,
                    )
                    did_shape_log = True

                if m.any():
                    y_norm = (y - mu_y) / (sigma_y + EPS_Y)
                    loss = F.smooth_l1_loss(pred[m], y_norm[m])
                else:
                    loss = torch.tensor(0.0, device=device)

                if m.any():
                    pred_c = pred * (sigma_y + EPS_Y) + mu_y
                    err = (pred_c - y)[m]
                    val_sq += float((err * err).sum().item())
                    val_n += int(err.numel())
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        train_rmse = float(np.sqrt(train_sq / train_n)) if train_n > 0 else float("nan")
        val_rmse = float(np.sqrt(val_sq / val_n)) if val_n > 0 else float("nan")
        valid_batch_frac = valid_batches / max(1, len(train_loader))
        valid_pix_frac = valid_pix / max(1, total_pix)
        if stats_target:
            tgt = torch.cat(stats_target, dim=0).numpy()
            prd = torch.cat(stats_pred, dim=0).numpy()
            prd = prd * (sigma_y + EPS_Y) + mu_y
            logger.info(
                "base stats target_mean=%.3f target_std=%.3f target_min=%.3f target_max=%.3f "
                "pred_mean=%.3f pred_std=%.3f pred_min=%.3f pred_max=%.3f",
                float(np.mean(tgt)),
                float(np.std(tgt)),
                float(np.min(tgt)),
                float(np.max(tgt)),
                float(np.mean(prd)),
                float(np.std(prd)),
                float(np.min(prd)),
                float(np.max(prd)),
            )
        logger.info(
            "base epoch=%d train_loss=%.6f val_loss=%.6f train_rmse=%.4f val_rmse=%.4f valid_batches=%.2f valid_pix=%.3f",
            epoch,
            train_loss,
            val_loss,
            train_rmse,
            val_rmse,
            valid_batch_frac,
            valid_pix_frac,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
            }
        )

        if optim_steps > 0:
            if np.isfinite(val_rmse) and val_rmse < best_val:
                best_val = val_rmse
                best_epoch = epoch
                best_state = {
                "base": copy.deepcopy(base_net.state_dict()),
                "up_head": copy.deepcopy(up_head.state_dict()),
                "era5_idx": ERA5_TOP4,
                "target_mu": mu_y,
                "target_sigma": sigma_y,
                "modis_mu": modis_mu,
                "modis_sigma": modis_sigma,
                "viirs_mu": viirs_mu,
                "viirs_sigma": viirs_sigma,
                "era5_mu": era5_mu,
                "era5_sigma": era5_sigma,
                "static_mu": static_mu,
                    "static_sigma": static_sigma,
                }
                torch.save(best_state, OUT_DIR / "base_best.pt")
                logger.info("saved best base model: %s", OUT_DIR / "base_best.pt")
                bad_epochs = 0
            else:
                bad_epochs += 1

        if bad_epochs >= patience:
            logger.info("early stopping base at epoch %d", epoch)
            break

    df_hist = pd.DataFrame(history)
    df_hist.to_csv(OUT_DIR / "base_history.csv", index=False)
    _save_loss_plot(df_hist, OUT_DIR)
    _save_rmse_plot(df_hist, OUT_DIR)

    if best_state is not None:
        base_net.load_state_dict(best_state["base"])
        up_head.load_state_dict(best_state["up_head"])
    eval_path = OUT_DIR / "base_eval_metrics.csv"
    base_metrics = _eval_base_metrics(
        base_net,
        up_head,
        val_ds,
        _args.metrics_samples,
        device,
        eval_path,
        mu_y,
        sigma_y,
        modis_mu,
        modis_sigma,
        viirs_mu,
        viirs_sigma,
        era5_mu,
        era5_sigma,
        static_mu,
        static_sigma,
    )
    if base_metrics:
        logger.info("base eval metrics: %s", base_metrics)
    return base_net, up_head, mu_y, sigma_y, modis_mu, modis_sigma, viirs_mu, viirs_sigma, era5_mu, era5_sigma, static_mu, static_sigma


def _train_residual(base_net, up_head, mu_y, sigma_y, modis_mu, modis_sigma, viirs_mu, viirs_sigma, era5_mu, era5_sigma, static_mu, static_sigma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_net = base_net.to(device)
    up_head = up_head.to(device)

    s2_bands = _select_indices(_args.s2_bands, "s2")
    s1_bands = _select_indices(_args.s1_bands, "s1")

    s2_ch = root_30m["products_30m"]["sentinel2"]["data"].shape[1]
    if s2_bands is not None:
        s2_ch = len(s2_bands)
    s1_ch = root_30m["products_30m"]["sentinel1"]["data"].shape[1]
    if s1_bands is not None:
        s1_ch = len(s1_bands)

    world_labels = root_30m["static_30m"]["worldcover"]["labels"][:]
    num_classes = len(world_labels)

    res_net = ResidualNet(
        s2_ch=s2_ch,
        s1_ch=s1_ch,
        dem_ch=1,
        lc_num_classes=num_classes,
        lc_one_hot=False,
        era5_ch=len(ERA5_TOP4),
        base_ch=1,
    ).to(device)

    train_ds = ResidualDataset(
        train_items,
        s2_bands,
        s1_bands,
        H_hr,
        W_hr,
        H_lr_modis,
        W_lr_modis,
        modis_present,
        viirs_present,
        daily_times,
        daily_to_month_map,
    )
    val_ds = ResidualDataset(
        val_items,
        s2_bands,
        s1_bands,
        H_hr,
        W_hr,
        H_lr_modis,
        W_lr_modis,
        modis_present,
        viirs_present,
        daily_times,
        daily_to_month_map,
    )
    train_loader = DataLoader(train_ds, batch_size=_args.batch_size, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_ds, batch_size=_args.batch_size, shuffle=False, collate_fn=lambda x: x)

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    def _run_phase(phase_name, start_epoch, end_epoch, lr_peak, lr_end, base_train, warmup_start):
        for p in base_net.parameters():
            p.requires_grad = base_train
        for p in up_head.parameters():
            p.requires_grad = base_train

        if base_train:
            opt = torch.optim.AdamW(
                [
                    {"params": res_net.parameters(), "lr": lr_peak, "weight_decay": 5e-4},
                    {"params": list(base_net.parameters()) + list(up_head.parameters()), "lr": lr_peak * 0.05, "weight_decay": 1e-4},
                ],
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            clip_params = list(res_net.parameters()) + list(base_net.parameters()) + list(up_head.parameters())
        else:
            opt = torch.optim.AdamW(res_net.parameters(), lr=lr_peak, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4)
            clip_params = list(res_net.parameters())
        total_steps = (end_epoch - start_epoch + 1) * max(1, len(train_loader))
        warmup_steps = 5 * max(1, len(train_loader))
        scheduler = _build_cosine_schedule_steps(
            opt, total_steps=total_steps, warmup_steps=warmup_steps, warmup_start=warmup_start, peak_lr=lr_peak, end_lr=lr_end
        )

        history = []
        best_val = float("inf")
        patience = 15
        bad_epochs = 0
        best_state = None

        for epoch in range(start_epoch, end_epoch + 1):
            res_net.train()
            if base_train:
                base_net.train()
                up_head.train()
            else:
                base_net.eval()
                up_head.eval()

            train_losses = []
            train_sq = 0.0
            train_n = 0

            opt.zero_grad(set_to_none=True)
            did_sample_log = False
            did_shape_log = False
            zero_pred_steps = 0
            first_low_valid = None
            for step, batch in enumerate(train_loader, start=1):
                s2 = torch.stack([b[0] for b in batch], dim=0).to(device)
                s1_list = [b[1] for b in batch]
                s1 = torch.stack([x for x in s1_list if x is not None], dim=0).to(device) if s1_list[0] is not None else None
                dem = torch.stack([b[2] for b in batch], dim=0).to(device)
                lc = torch.stack([b[3] for b in batch], dim=0).to(device)
                era5 = torch.stack([b[4] for b in batch], dim=0).to(device)
                y = torch.stack([b[5] for b in batch], dim=0).to(device)
                m = torch.stack([b[6] for b in batch], dim=0).to(device)
                modis_frames = torch.stack([b[7] for b in batch], dim=0).to(device)
                modis_masks = torch.stack([b[8] for b in batch], dim=0).to(device)
                viirs_frames = torch.stack([b[9] for b in batch], dim=0).to(device)
                viirs_masks = torch.stack([b[10] for b in batch], dim=0).to(device)
                era5_coarse = torch.stack([b[11] for b in batch], dim=0).to(device)
                doy = torch.stack([b[12] for b in batch], dim=0).to(device)
                static_c = torch.stack([b[13] for b in batch], dim=0).to(device)

                modis_frames = torch.nan_to_num(modis_frames, nan=0.0, posinf=0.0, neginf=0.0)
                viirs_frames = torch.nan_to_num(viirs_frames, nan=0.0, posinf=0.0, neginf=0.0)
                static_c = torch.nan_to_num(static_c, nan=0.0, posinf=0.0, neginf=0.0)
                modis_frames = (modis_frames - modis_mu) / (modis_sigma + EPS_Y)
                viirs_frames = (viirs_frames - viirs_mu) / (viirs_sigma + EPS_Y)
                era5_mu_t = torch.as_tensor(era5_mu, device=device, dtype=era5_coarse.dtype)[None, :, None, None]
                era5_sigma_t = torch.as_tensor(era5_sigma, device=device, dtype=era5_coarse.dtype)[None, :, None, None]
                era5_coarse = (era5_coarse - era5_mu_t) / (era5_sigma_t + EPS_Y)
                static_mu_t = torch.as_tensor(static_mu, device=device, dtype=static_c.dtype)[None, :, None, None]
                static_sigma_t = torch.as_tensor(static_sigma, device=device, dtype=static_c.dtype)[None, :, None, None]
                static_c = (static_c - static_mu_t) / (static_sigma_t + EPS_Y)

                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    base = base_net(modis_frames, viirs_frames, era5_coarse, doy, static_c, modis_masks, viirs_masks)
                    base_hr = up_head(base)
                    base_hr = base_hr * (sigma_y + EPS_Y) + mu_y
                    pred = res_net(base_hr, s2, s1, dem, lc, era5, return_residual=False)
                    pred = pred.squeeze(1)
                    if not torch.isfinite(pred).all():
                        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                    loss = F.smooth_l1_loss(pred[m], y[m]) if m.any() else torch.tensor(0.0, device=device)

                if m.any():
                    pred_std = float(pred[m].std().detach().cpu())
                    target_std = float(y[m].std().detach().cpu())
                    valid_pix = int(m.sum().item())
                else:
                    pred_std = 0.0
                    target_std = 0.0
                    valid_pix = 0
                modis_v = float(modis_masks.mean().detach().cpu())
                viirs_v = float(viirs_masks.mean().detach().cpu())
                static_v = float(torch.isfinite(static_c).float().mean().detach().cpu())
                logger.info(
                    "batch diag pred_std=%.4f target_std=%.4f valid_pix=%d modis_v=%.3f viirs_v=%.3f static_v=%.3f",
                    pred_std,
                    target_std,
                    valid_pix,
                    modis_v,
                    viirs_v,
                    static_v,
                )
                if pred_std < 1e-3:
                    zero_pred_steps += 1
                    if first_low_valid is None and (modis_v < 0.01 and viirs_v < 0.01 and static_v < 0.5):
                        first_low_valid = (step, modis_v, viirs_v, static_v, valid_pix)
                else:
                    zero_pred_steps = 0
                if zero_pred_steps >= 20:
                    if first_low_valid is not None:
                        s, mv, vv, sv, vp = first_low_valid
                        logger.error(
                            "collapse detected: first low-valid batch step=%d modis_v=%.3f viirs_v=%.3f static_v=%.3f valid_pix=%d",
                            s,
                            mv,
                            vv,
                            sv,
                            vp,
                        )
                    raise RuntimeError("collapse detected: pred_std ~ 0 for many steps")

                if not did_shape_log:
                    pmin, pmean, pmax = _safe_min_mean_max(pred[0].detach().cpu().numpy())
                    tmin, tmean, tmax = _safe_min_mean_max(y[0].detach().cpu().numpy())
                    logger.info(
                        "train shapes s2=%s s1=%s dem=%s lc=%s era5=%s base=%s pred=%s y=%s m=%s",
                        tuple(s2.shape),
                        tuple(s1.shape) if s1 is not None else None,
                        tuple(dem.shape),
                        tuple(lc.shape),
                        tuple(era5.shape),
                        tuple(base_hr.shape),
                        tuple(pred.shape),
                        tuple(y.shape),
                        tuple(m.shape),
                    )
                    logger.info(
                        "train sample pred(min=%.3f mean=%.3f max=%.3f) target(min=%.3f mean=%.3f max=%.3f)",
                        pmin,
                        pmean,
                        pmax,
                        tmin,
                        tmean,
                        tmax,
                    )
                    did_shape_log = True

                if (epoch % 2 == 0) and not did_sample_log:
                    _log_residual_samples(epoch, s2, s1, dem, lc, era5, base_hr, y, pred, "train/residual")
                    did_sample_log = True

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                opt.zero_grad(set_to_none=True)

                if m.any():
                    err = (pred - y)[m]
                    train_sq += float((err * err).sum().item())
                    train_n += int(err.numel())
                train_losses.append(loss.item())

            res_net.eval()
            base_net.eval()
            up_head.eval()
            val_losses = []
            val_sq = 0.0
            val_n = 0
            with torch.no_grad():
                did_shape_log = False
                for batch in val_loader:
                    s2 = torch.stack([b[0] for b in batch], dim=0).to(device)
                    s1_list = [b[1] for b in batch]
                    s1 = torch.stack([x for x in s1_list if x is not None], dim=0).to(device) if s1_list[0] is not None else None
                    dem = torch.stack([b[2] for b in batch], dim=0).to(device)
                    lc = torch.stack([b[3] for b in batch], dim=0).to(device)
                    era5 = torch.stack([b[4] for b in batch], dim=0).to(device)
                    y = torch.stack([b[5] for b in batch], dim=0).to(device)
                    m = torch.stack([b[6] for b in batch], dim=0).to(device)
                    modis_frames = torch.stack([b[7] for b in batch], dim=0).to(device)
                    modis_masks = torch.stack([b[8] for b in batch], dim=0).to(device)
                    viirs_frames = torch.stack([b[9] for b in batch], dim=0).to(device)
                    viirs_masks = torch.stack([b[10] for b in batch], dim=0).to(device)
                    era5_coarse = torch.stack([b[11] for b in batch], dim=0).to(device)
                    doy = torch.stack([b[12] for b in batch], dim=0).to(device)
                    static_c = torch.stack([b[13] for b in batch], dim=0).to(device)

                    modis_frames = torch.nan_to_num(modis_frames, nan=0.0, posinf=0.0, neginf=0.0)
                    viirs_frames = torch.nan_to_num(viirs_frames, nan=0.0, posinf=0.0, neginf=0.0)
                    static_c = torch.nan_to_num(static_c, nan=0.0, posinf=0.0, neginf=0.0)
                    modis_frames = (modis_frames - modis_mu) / (modis_sigma + EPS_Y)
                    viirs_frames = (viirs_frames - viirs_mu) / (viirs_sigma + EPS_Y)
                    era5_mu_t = torch.as_tensor(era5_mu, device=device, dtype=era5_coarse.dtype)[None, :, None, None]
                    era5_sigma_t = torch.as_tensor(era5_sigma, device=device, dtype=era5_coarse.dtype)[None, :, None, None]
                    era5_coarse = (era5_coarse - era5_mu_t) / (era5_sigma_t + EPS_Y)
                    static_mu_t = torch.as_tensor(static_mu, device=device, dtype=static_c.dtype)[None, :, None, None]
                    static_sigma_t = torch.as_tensor(static_sigma, device=device, dtype=static_c.dtype)[None, :, None, None]
                    static_c = (static_c - static_mu_t) / (static_sigma_t + EPS_Y)

                    base = base_net(modis_frames, viirs_frames, era5_coarse, doy, static_c, modis_masks, viirs_masks)
                    base_hr = up_head(base)
                    base_hr = base_hr * (sigma_y + EPS_Y) + mu_y
                    pred = res_net(base_hr, s2, s1, dem, lc, era5, return_residual=False)
                    pred = pred.squeeze(1)
                    if not torch.isfinite(pred).all():
                        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                    loss = F.smooth_l1_loss(pred[m], y[m]) if m.any() else torch.tensor(0.0, device=device)

                    if not did_shape_log:
                        pmin, pmean, pmax = _safe_min_mean_max(pred[0].detach().cpu().numpy())
                        tmin, tmean, tmax = _safe_min_mean_max(y[0].detach().cpu().numpy())
                        logger.info(
                            "val shapes s2=%s s1=%s dem=%s lc=%s era5=%s base=%s pred=%s y=%s m=%s",
                            tuple(s2.shape),
                            tuple(s1.shape) if s1 is not None else None,
                            tuple(dem.shape),
                            tuple(lc.shape),
                            tuple(era5.shape),
                            tuple(base_hr.shape),
                            tuple(pred.shape),
                            tuple(y.shape),
                            tuple(m.shape),
                        )
                        logger.info(
                            "val sample pred(min=%.3f mean=%.3f max=%.3f) target(min=%.3f mean=%.3f max=%.3f)",
                            pmin,
                            pmean,
                            pmax,
                            tmin,
                            tmean,
                            tmax,
                        )
                        did_shape_log = True

                    if m.any():
                        err = (pred - y)[m]
                        val_sq += float((err * err).sum().item())
                        val_n += int(err.numel())
                    val_losses.append(loss.item())

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            train_rmse = float(np.sqrt(train_sq / train_n)) if train_n > 0 else float("nan")
            val_rmse = float(np.sqrt(val_sq / val_n)) if val_n > 0 else float("nan")
            logger.info(
                "%s epoch=%d train_loss=%.6f val_loss=%.6f train_rmse=%.4f val_rmse=%.4f",
                phase_name,
                epoch,
                train_loss,
                val_loss,
                train_rmse,
                val_rmse,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                }
            )

            if np.isfinite(val_rmse) and val_rmse < best_val:
                best_val = val_rmse
                best_state = {
                    "residual": copy.deepcopy(res_net.state_dict()),
                    "base": copy.deepcopy(base_net.state_dict()),
                    "up_head": copy.deepcopy(up_head.state_dict()),
                }
                torch.save(best_state, OUT_DIR / f"residual_{phase_name.lower()}_best.pt")
                logger.info("saved best %s model", phase_name)
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= patience:
                logger.info("early stopping %s at epoch %d", phase_name, epoch)
                break

        return history, best_state

    history_r0, state_r0 = _run_phase("R0", 1, 90, lr_peak=2e-4, lr_end=2e-6, base_train=False, warmup_start=1e-5)
    history_r1, state_r1 = _run_phase("R1", 91, 120, lr_peak=5e-5, lr_end=5e-6, base_train=True, warmup_start=5e-6)

    df_hist = pd.DataFrame(history_r0 + history_r1)
    df_hist.to_csv(OUT_DIR / "residual_history.csv", index=False)
    _save_loss_plot(df_hist, OUT_DIR)
    _save_rmse_plot(df_hist, OUT_DIR)

    if state_r1 is not None:
        res_net.load_state_dict(state_r1["residual"])
    elif state_r0 is not None:
        res_net.load_state_dict(state_r0["residual"])

    eval_path = OUT_DIR / "residual_eval_metrics.csv"
    res_metrics = _eval_residual_metrics(
        res_net,
        base_net,
        up_head,
        val_ds,
        _args.metrics_samples,
        device,
        eval_path,
        mu_y,
        sigma_y,
        modis_mu,
        modis_sigma,
        viirs_mu,
        viirs_sigma,
        era5_mu,
        era5_sigma,
        static_mu,
        static_sigma,
    )
    if res_metrics:
        logger.info("residual eval metrics: %s", res_metrics)

    return res_net


base_net = None
up_head = None
mu_y = 0.0
sigma_y = 1.0
modis_mu = 0.0
modis_sigma = 1.0
viirs_mu = 0.0
viirs_sigma = 1.0
era5_mu = np.zeros(len(ERA5_TOP4), dtype=np.float32)
era5_sigma = np.ones(len(ERA5_TOP4), dtype=np.float32)
static_mu = np.zeros(3, dtype=np.float32)
static_sigma = np.ones(3, dtype=np.float32)
if _args.stage in ("base", "both"):
    base_net, up_head, mu_y, sigma_y, modis_mu, modis_sigma, viirs_mu, viirs_sigma, era5_mu, era5_sigma, static_mu, static_sigma = _train_base()
    logger.info("base training complete")

if _args.stage in ("residual", "both"):
    if base_net is None or up_head is None:
        if _args.base_checkpoint is None:
            raise SystemExit("base checkpoint required for residual stage")
        state = torch.load(_args.base_checkpoint, map_location="cpu")
        base_net = ThermalBaseNet(era5_k=len(ERA5_TOP4))
        up_head = UpsampleHead(in_ch=1)
        base_net.load_state_dict(state["base"])
        up_head.load_state_dict(state["up_head"])
        mu_y = float(state.get("target_mu", 0.0))
        sigma_y = float(state.get("target_sigma", 1.0))
        modis_mu = float(state.get("modis_mu", 0.0))
        modis_sigma = float(state.get("modis_sigma", 1.0))
        viirs_mu = float(state.get("viirs_mu", 0.0))
        viirs_sigma = float(state.get("viirs_sigma", 1.0))
        era5_mu = np.asarray(state.get("era5_mu", era5_mu), dtype=np.float32)
        era5_sigma = np.asarray(state.get("era5_sigma", era5_sigma), dtype=np.float32)
        static_mu = np.asarray(state.get("static_mu", static_mu), dtype=np.float32)
        static_sigma = np.asarray(state.get("static_sigma", static_sigma), dtype=np.float32)
    _train_residual(base_net, up_head, mu_y, sigma_y, modis_mu, modis_sigma, viirs_mu, viirs_sigma, era5_mu, era5_sigma, static_mu, static_sigma)
    logger.info("residual training complete")
