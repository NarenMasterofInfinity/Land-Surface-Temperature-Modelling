import argparse
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset, DataLoader


sys.path.append(str(Path(__file__).resolve().parents[3]))
from deep.cnn_lr_hr.thermal_base.thermal_base_net import ThermalBaseNet
from deep.cnn_lr_hr.thermal_base.thermal_residual_net import ResidualNet


def _to_str(arr):
    return np.array([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr])


def _parse_time_raw(raw):
    arr = np.asarray(raw)
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.to_datetime(arr, errors="coerce")
    if np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return pd.to_datetime([], errors="coerce")
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmin > 1e7 and vmax < 3e9 and np.all(np.floor(finite) == finite):
            return pd.to_datetime(finite.astype(np.int64).astype(str), format="%Y%m%d", errors="coerce")
        if vmax > 1e12:
            return pd.to_datetime(arr, unit="ms", errors="coerce")
        if vmax > 1e9:
            return pd.to_datetime(arr, unit="s", errors="coerce")
        if vmax < 100000:
            return pd.to_datetime(arr, unit="D", origin="unix", errors="coerce")
    s = _to_str(arr)
    if s.size > 0:
        first = str(s.flat[0])
        if len(first) == 8 and first.isdigit():
            return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        if len(first) == 10 and first[4] == "-" and first[7] == "-":
            return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        if len(first) == 10 and first[4] == "_" and first[7] == "_":
            return pd.to_datetime(s, format="%Y_%m_%d", errors="coerce")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(s, errors="coerce")


def _read_daily_time(root_daily):
    if "time" in root_daily and "daily" in root_daily["time"]:
        raw = root_daily["time"]["daily"][:]
        return _parse_time_raw(raw)
    raise RuntimeError("daily time not found")


def _read_monthly_time(root_30m):
    if "time" in root_30m and "monthly" in root_30m["time"]:
        raw = root_30m["time"]["monthly"][:]
        return _parse_time_raw(raw)
    return pd.to_datetime([], errors="coerce")


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


def _extract_modis(modis_lr, lst_min, lst_max):
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
    valid_lst = np.isfinite(lst) & (lst >= lst_min) & (lst <= lst_max)
    valid = valid_qc & valid_lst
    lst = np.where(valid, lst, np.nan)
    return lst.astype(np.float32, copy=False), valid.astype(np.float32)


def _extract_viirs(viirs_lr, lst_min, lst_max):
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
    valid_lst = np.isfinite(lst) & (lst >= lst_min) & (lst <= lst_max)
    valid = valid_qc & valid_lst
    lst = np.where(valid, lst, np.nan)
    return lst.astype(np.float32, copy=False), valid.astype(np.float32)


def _landsat_to_celsius(arr, attrs):
    arr = arr.astype(np.float32, copy=False)
    arr = np.where(arr == 149, np.nan, arr)
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


def _apply_range_mask(arr, lst_min, lst_max):
    valid = np.isfinite(arr) & (arr >= lst_min) & (arr <= lst_max)
    out = np.where(valid, arr, np.nan)
    return out.astype(np.float32), valid


def _bilinear_patch(arr, out_h, out_w):
    row = np.linspace(0, arr.shape[0] - 1, out_h, dtype=np.float64)
    col = np.linspace(0, arr.shape[1] - 1, out_w, dtype=np.float64)
    r0 = np.floor(row).astype(np.int64)
    c0 = np.floor(col).astype(np.int64)
    r1 = np.clip(r0 + 1, 0, arr.shape[0] - 1)
    c1 = np.clip(c0 + 1, 0, arr.shape[1] - 1)
    fr = (row - r0)[:, None]
    fc = (col - c0)[None, :]
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
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, np.sum(wts * np.nan_to_num(vals), axis=0) / denom, np.nan)
    return out.astype(np.float32)


def _downsample_nearest(arr, out_h, out_w):
    x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(out_h, out_w), mode="nearest")
    return x.squeeze(0).squeeze(0).numpy().astype(np.float32)


def _tstat(x: torch.Tensor) -> str:
    x = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    return f"min={x.min().item():.3f} mean={x.mean().item():.3f} max={x.max().item():.3f}"


def _build_cosine_schedule_steps(optimizer, total_steps, warmup_steps, warmup_start, peak_lr, end_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return (warmup_start + (peak_lr - warmup_start) * (step / max(1, warmup_steps - 1))) / peak_lr
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
        cos = 0.5 * (1 + np.cos(np.pi * t))
        return (end_lr + (peak_lr - end_lr) * cos) / peak_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class BaseFullScene(Dataset):
    def __init__(self, t_indices, root_30m, root_daily, lst_min, lst_max, era5_idx):
        self.t_indices = list(t_indices)
        self.root_30m = root_30m
        self.root_daily = root_daily
        self.landsat = root_30m["labels_30m"]["landsat"]["data"]
        self.landsat_attrs = dict(root_30m["labels_30m"]["landsat"].attrs)
        self.modis = root_daily["products"]["modis"]["data"]
        self.viirs = root_daily["products"]["viirs"]["data"]
        self.era5 = root_30m["products_30m"]["era5"]["data"]
        self.dem = root_30m["static_30m"]["dem"]["data"][0, 0]
        self.world = root_30m["static_30m"]["worldcover"]["data"][0, 0]
        self.dyn = root_30m["static_30m"]["dynamic_world"]["data"][0, 0]
        self.lst_min = lst_min
        self.lst_max = lst_max
        self.era5_idx = era5_idx
        self.H_lr, self.W_lr = self.modis.shape[-2], self.modis.shape[-1]
        self.daily_times = _read_daily_time(root_daily).to_period("D").to_timestamp()

    def __len__(self):
        return len(self.t_indices)

    def __getitem__(self, idx):
        t = int(self.t_indices[idx])
        frames_m = []
        frames_v = []
        masks_m = []
        masks_v = []
        for dt in (2, 1, 0):
            tt = t - dt
            if tt < 0:
                frames_m.append(np.full((1, self.H_lr, self.W_lr), np.nan, dtype=np.float32))
                frames_v.append(np.full((1, self.H_lr, self.W_lr), np.nan, dtype=np.float32))
                masks_m.append(np.zeros((1, self.H_lr, self.W_lr), dtype=np.float32))
                masks_v.append(np.zeros((1, self.H_lr, self.W_lr), dtype=np.float32))
                continue
            modis_lst, modis_mask = _extract_modis(self.modis[tt], self.lst_min, self.lst_max)
            viirs_lst, viirs_mask = _extract_viirs(self.viirs[tt], self.lst_min, self.lst_max)
            frames_m.append(modis_lst[None, ...])
            frames_v.append(viirs_lst[None, ...])
            masks_m.append(modis_mask[None, ...])
            masks_v.append(viirs_mask[None, ...])

        modis_frames = np.stack(frames_m, axis=0)
        viirs_frames = np.stack(frames_v, axis=0)
        modis_masks = np.stack(masks_m, axis=0)
        viirs_masks = np.stack(masks_v, axis=0)

        y_hr = self.landsat[t, 0]
        y_hr = _landsat_to_celsius(y_hr, self.landsat_attrs)
        y_hr, valid_hr = _apply_range_mask(y_hr, self.lst_min, self.lst_max)
        y_lr = _bilinear_patch(y_hr, self.H_lr, self.W_lr)
        m_lr = _bilinear_patch(valid_hr.astype(np.float32), self.H_lr, self.W_lr) > 0.5
        m_lr = m_lr & np.isfinite(y_lr)
        y_lr = np.where(m_lr, y_lr, np.nan).astype(np.float32, copy=False)

        era5 = self.era5[t]
        era5 = era5[self.era5_idx]
        era5 = np.nan_to_num(era5, nan=0.0, posinf=0.0, neginf=0.0)
        era5 = torch.from_numpy(era5).float().unsqueeze(0)
        era5 = F.interpolate(era5, size=(self.H_lr, self.W_lr), mode="bilinear", align_corners=False)
        era5 = era5.squeeze(0).numpy()

        dem_c = _bilinear_patch(self.dem, self.H_lr, self.W_lr)
        world_c = _downsample_nearest(self.world, self.H_lr, self.W_lr)
        dyn_c = _downsample_nearest(self.dyn, self.H_lr, self.W_lr)
        static = np.stack([dem_c, world_c, dyn_c], axis=0).astype(np.float32, copy=False)

        date = pd.Timestamp(self.daily_times[t])
        doy = float(date.dayofyear)
        sin_doy = np.sin(2 * np.pi * doy / 365.25)
        cos_doy = np.cos(2 * np.pi * doy / 365.25)
        doy_arr = np.array([sin_doy, cos_doy], dtype=np.float32)

        return (
            modis_frames.astype(np.float32),
            modis_masks.astype(np.float32),
            viirs_frames.astype(np.float32),
            viirs_masks.astype(np.float32),
            era5.astype(np.float32),
            static.astype(np.float32),
            y_lr.astype(np.float32),
            m_lr.astype(np.float32),
            doy_arr,
        )


class ResidualFullScene(Dataset):
    def __init__(self, t_indices, root_30m, root_daily, lst_min, lst_max, era5_idx):
        self.t_indices = list(t_indices)
        self.root_30m = root_30m
        self.root_daily = root_daily
        self.s2 = root_30m["products_30m"]["sentinel2"]["data"]
        self.s1 = root_30m["products_30m"]["sentinel1"]["data"]
        self.era5 = root_30m["products_30m"]["era5"]["data"]
        self.dem = root_30m["static_30m"]["dem"]["data"][0, 0]
        self.world = root_30m["static_30m"]["worldcover"]["data"][0, 0]
        self.landsat = root_30m["labels_30m"]["landsat"]["data"]
        self.landsat_attrs = dict(root_30m["labels_30m"]["landsat"].attrs)
        self.lst_min = lst_min
        self.lst_max = lst_max
        self.era5_idx = era5_idx
        self.daily_times = _read_daily_time(root_daily).to_period("D").to_timestamp()
        self.monthly_times = _read_monthly_time(root_30m).to_period("M").to_timestamp()
        self.month_map = {pd.Timestamp(t): i for i, t in enumerate(self.monthly_times) if pd.notna(t)}

    def __len__(self):
        return len(self.t_indices)

    def __getitem__(self, idx):
        t = int(self.t_indices[idx])
        date = pd.Timestamp(self.daily_times[t])
        m_key = date.to_period("M").to_timestamp()
        m_idx = self.month_map.get(m_key, -1)

        if m_idx < 0:
            s2 = np.full_like(self.s2[0], np.nan, dtype=np.float32)
            s1 = np.full_like(self.s1[0], np.nan, dtype=np.float32) if self.s1 is not None else None
        else:
            s2 = self.s2[m_idx]
            s1 = self.s1[m_idx] if self.s1 is not None else None

        dem = self.dem[None, ...]
        lc = self.world[None, ...]

        era5 = self.era5[t]
        era5 = era5[self.era5_idx]
        era5 = np.nan_to_num(era5, nan=0.0, posinf=0.0, neginf=0.0)

        y_hr = self.landsat[t, 0]
        y_hr = _landsat_to_celsius(y_hr, self.landsat_attrs)
        y_hr, valid_hr = _apply_range_mask(y_hr, self.lst_min, self.lst_max)

        # base inputs (coarse)
        frames_m = []
        frames_v = []
        masks_m = []
        masks_v = []
        H_lr, W_lr = self.root_daily["products"]["modis"]["data"].shape[-2:]
        for dt in (2, 1, 0):
            tt = t - dt
            if tt < 0:
                frames_m.append(np.full((1, H_lr, W_lr), np.nan, dtype=np.float32))
                frames_v.append(np.full((1, H_lr, W_lr), np.nan, dtype=np.float32))
                masks_m.append(np.zeros((1, H_lr, W_lr), dtype=np.float32))
                masks_v.append(np.zeros((1, H_lr, W_lr), dtype=np.float32))
                continue
            modis_lst, modis_mask = _extract_modis(self.root_daily["products"]["modis"]["data"][tt], self.lst_min, self.lst_max)
            viirs_lst, viirs_mask = _extract_viirs(self.root_daily["products"]["viirs"]["data"][tt], self.lst_min, self.lst_max)
            frames_m.append(modis_lst[None, ...])
            frames_v.append(viirs_lst[None, ...])
            masks_m.append(modis_mask[None, ...])
            masks_v.append(viirs_mask[None, ...])

        modis_frames = np.stack(frames_m, axis=0).astype(np.float32)
        viirs_frames = np.stack(frames_v, axis=0).astype(np.float32)
        modis_masks = np.stack(masks_m, axis=0).astype(np.float32)
        viirs_masks = np.stack(masks_v, axis=0).astype(np.float32)

        era5_c = torch.from_numpy(era5).float().unsqueeze(0)
        era5_c = F.interpolate(era5_c, size=(H_lr, W_lr), mode="bilinear", align_corners=False)
        era5_c = era5_c.squeeze(0).numpy().astype(np.float32)

        dem_c = _bilinear_patch(self.dem, H_lr, W_lr)
        world_c = _downsample_nearest(self.world, H_lr, W_lr)
        dyn_c = _downsample_nearest(self.root_30m["static_30m"]["dynamic_world"]["data"][0, 0], H_lr, W_lr)
        static_c = np.stack([dem_c, world_c, dyn_c], axis=0).astype(np.float32, copy=False)

        doy = float(date.dayofyear)
        sin_doy = np.sin(2 * np.pi * doy / 365.25)
        cos_doy = np.cos(2 * np.pi * doy / 365.25)
        doy_arr = np.array([sin_doy, cos_doy], dtype=np.float32)

        return (
            s2.astype(np.float32),
            s1.astype(np.float32) if s1 is not None else None,
            dem.astype(np.float32),
            None,
            era5.astype(np.float32),
            y_hr.astype(np.float32),
            valid_hr.astype(np.float32),
            modis_frames,
            modis_masks,
            viirs_frames,
            viirs_masks,
            era5_c,
            static_c,
            doy_arr,
        )


def _stack_base(batch):
    return [torch.from_numpy(np.stack([b[i] for b in batch], axis=0)) for i in range(len(batch[0]))]


def _stack_residual(batch):
    s2 = torch.from_numpy(np.stack([b[0] for b in batch], axis=0))
    s1 = None
    if batch[0][1] is not None:
        s1 = torch.from_numpy(np.stack([b[1] for b in batch], axis=0))
    dem = torch.from_numpy(np.stack([b[2] for b in batch], axis=0))
    lc = None
    era5 = torch.from_numpy(np.stack([b[4] for b in batch], axis=0))
    y = torch.from_numpy(np.stack([b[5] for b in batch], axis=0))
    m = torch.from_numpy(np.stack([b[6] for b in batch], axis=0))
    modis_frames = torch.from_numpy(np.stack([b[7] for b in batch], axis=0))
    modis_masks = torch.from_numpy(np.stack([b[8] for b in batch], axis=0))
    viirs_frames = torch.from_numpy(np.stack([b[9] for b in batch], axis=0))
    viirs_masks = torch.from_numpy(np.stack([b[10] for b in batch], axis=0))
    era5_c = torch.from_numpy(np.stack([b[11] for b in batch], axis=0))
    static_c = torch.from_numpy(np.stack([b[12] for b in batch], axis=0))
    doy = torch.from_numpy(np.stack([b[13] for b in batch], axis=0))
    return s2, s1, dem, lc, era5, y, m, modis_frames, modis_masks, viirs_frames, viirs_masks, era5_c, static_c, doy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-30m", type=str, default="/home/naren-root/Documents/FYP2/Project/madurai_30m.zarr")
    ap.add_argument("--root-daily", type=str, default="/home/naren-root/Documents/FYP2/Project/madurai.zarr")
    ap.add_argument("--common-dates", type=str, default="/home/naren-root/Documents/FYP2/Project/common_dates.csv")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--base-epochs", type=int, default=60)
    ap.add_argument("--residual-r0", type=int, default=90)
    ap.add_argument("--residual-r1", type=int, default=30)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--residual-only", action="store_true")
    ap.add_argument("--base-ckpt", type=str, default="")
    ap.add_argument("--res-downsample", type=int, default=1)
    ap.add_argument("--warmup-epochs-base", type=int, default=8)
    ap.add_argument("--warmup-epochs-res", type=int, default=8)
    ap.add_argument("--grad-clip-base", type=float, default=1.0)
    ap.add_argument("--grad-clip-res", type=float, default=0.5)
    ap.add_argument("--early-patience", type=int, default=12)
    ap.add_argument("--early-delta", type=float, default=0.0)
    ap.add_argument("--min-target-valid", type=float, default=0.30)
    ap.add_argument("--min-modis-valid", type=float, default=0.05)
    ap.add_argument("--min-viirs-valid", type=float, default=0.05)
    ap.add_argument("--min-s2-valid", type=float, default=0.10)
    ap.add_argument("--wd-res", type=float, default=5e-4)
    ap.add_argument("--res-dropout", type=float, default=0.1)
    ap.add_argument("--res-head-ch", type=int, default=16)
    ap.add_argument("--res-checkpoint", action="store_true")
    ap.add_argument("--res-unet", type=str, default="8,16,24,32")
    ap.add_argument("--res-s2-out", type=int, default=16)
    ap.add_argument("--res-dem-out", type=int, default=8)
    ap.add_argument("--res-ckpt", type=str, default="")
    ap.add_argument("--lr-base", type=float, default=3e-4)
    ap.add_argument("--lr-res", type=float, default=3e-4)
    ap.add_argument("--era5-idx", type=str, default="0,1,3,4")
    ap.add_argument("--lst-min", type=float, default=10.0)
    ap.add_argument("--lst-max", type=float, default=70.0)
    ap.add_argument("--out", type=str, default="/home/naren-root/Documents/FYP2/Project/baselines/deep/cnn_lr_hr/thermal_base/good_run")
    args = ap.parse_args()

    era5_idx = [int(x) for x in args.era5_idx.split(",") if x.strip()]

    root_30m = zarr.open_group(args.root_30m, mode="r")
    root_daily = zarr.open_group(args.root_daily, mode="r")
    daily_times = _read_daily_time(root_daily).to_period("D").to_timestamp()
    daily_map = {pd.Timestamp(t).date(): i for i, t in enumerate(daily_times) if pd.notna(t)}

    common = pd.read_csv(args.common_dates)
    if "date" in common.columns:
        dates = pd.to_datetime(common["date"], errors="coerce")
    else:
        dates = pd.to_datetime(common.iloc[:, 0], errors="coerce")
    dates = dates.dropna().dt.to_period("D").dt.to_timestamp()

    t_indices = []
    for d in dates:
        t = daily_map.get(pd.Timestamp(d).date(), None)
        if t is not None:
            t_indices.append(int(t))
    if not t_indices:
        raise RuntimeError("no matching dates between common_dates and daily time")

    rng = np.random.default_rng(42)
    rng.shuffle(t_indices)
    n = len(t_indices)
    n_val = max(1, int(0.1 * n))
    val_idx = t_indices[:n_val]
    train_idx = t_indices[n_val:]

    train_base = BaseFullScene(train_idx, root_30m, root_daily, args.lst_min, args.lst_max, era5_idx)
    val_base = BaseFullScene(val_idx, root_30m, root_daily, args.lst_min, args.lst_max, era5_idx)
    train_loader = DataLoader(train_base, batch_size=args.batch_size, shuffle=True, collate_fn=_stack_base)
    val_loader = DataLoader(val_base, batch_size=args.batch_size, shuffle=False, collate_fn=_stack_base)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_net = ThermalBaseNet(era5_k=len(era5_idx), static_ch=3).to(device)
    opt = torch.optim.AdamW(base_net.parameters(), lr=args.lr_base, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    if args.dummy:
        args.base_epochs = 2
        args.residual_r0 = 2
        args.residual_r1 = 1
    total_steps = args.base_epochs * max(1, len(train_loader))
    warmup_steps = args.warmup_epochs_base * max(1, len(train_loader))
    scheduler = _build_cosine_schedule_steps(opt, total_steps, warmup_steps, 1e-5, args.lr_base, 3e-6)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    if args.residual_only:
        ckpt_path = args.base_ckpt if args.base_ckpt else str(out_dir / "base_best.pt")
        if not Path(ckpt_path).exists():
            raise RuntimeError(f"base checkpoint not found: {ckpt_path}")
        base_net.load_state_dict(torch.load(ckpt_path, map_location=device))
        base_net.eval()
        print(f"loaded base ckpt: {ckpt_path}")
    else:
        bad_epochs = 0
        for epoch in range(1, args.base_epochs + 1):
            t0 = time.perf_counter()
            base_net.train()
            tr_loss = []
            tr_sq = 0.0
            tr_n = 0
            did_train_sample = False
            for modis, modis_m, viirs, viirs_m, era5, static, y, m, doy in train_loader:
                modis = torch.nan_to_num(modis.to(device), nan=0.0)
                viirs = torch.nan_to_num(viirs.to(device), nan=0.0)
                era5 = torch.nan_to_num(era5.to(device), nan=0.0)
                static = torch.nan_to_num(static.to(device), nan=0.0)
                modis_m = torch.nan_to_num(modis_m.to(device), nan=0.0)
                viirs_m = torch.nan_to_num(viirs_m.to(device), nan=0.0)
                y = y.to(device)
                m = m.to(device) > 0
                m = m & torch.isfinite(y)
                y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                doy = doy.to(device)
                target_v = float(m.float().mean().item())
                modis_v = float(modis_m.float().mean().item())
                viirs_v = float(viirs_m.float().mean().item())
                if target_v < args.min_target_valid or (modis_v < args.min_modis_valid and viirs_v < args.min_viirs_valid):
                    continue

                era5 = (era5 - era5.mean(dim=(0, 2, 3), keepdim=True)) / era5.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
                static = (static - static.mean(dim=(0, 2, 3), keepdim=True)) / static.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    pred = base_net(modis, viirs, era5, doy, static, modis_m, viirs_m).squeeze(1)
                    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                    if not m.any():
                        continue
                    loss = F.smooth_l1_loss(pred[m], y[m])
                if not did_train_sample:
                    p = pred[m]
                    yt = y[m]
                    print(
                        "base train sample "
                        f"pred({_tstat(p)}) target({_tstat(yt)}) "
                        f"modis({_tstat(modis)}) viirs({_tstat(viirs)}) "
                        f"era5({_tstat(era5)}) static({_tstat(static)})"
                    )
                    did_train_sample = True

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(base_net.parameters(), args.grad_clip_base)
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()

                err = (pred - y)[m]
                tr_sq += float((err * err).sum().item())
                tr_n += int(err.numel())
                tr_loss.append(loss.item())

            base_net.eval()
            va_loss = []
            va_sq = 0.0
            va_n = 0
            did_val_sample = False
            with torch.no_grad():
                for modis, modis_m, viirs, viirs_m, era5, static, y, m, doy in val_loader:
                    modis = torch.nan_to_num(modis.to(device), nan=0.0)
                    viirs = torch.nan_to_num(viirs.to(device), nan=0.0)
                    era5 = torch.nan_to_num(era5.to(device), nan=0.0)
                    era5 = (era5 - era5.mean(dim=(0, 2, 3), keepdim=True)) / era5.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
                    static = torch.nan_to_num(static.to(device), nan=0.0)
                    modis_m = torch.nan_to_num(modis_m.to(device), nan=0.0)
                    viirs_m = torch.nan_to_num(viirs_m.to(device), nan=0.0)
                    y = y.to(device)
                    m = m.to(device) > 0
                    m = m & torch.isfinite(y)
                    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                    doy = doy.to(device)
                    target_v = float(m.float().mean().item())
                    modis_v = float(modis_m.float().mean().item())
                    viirs_v = float(viirs_m.float().mean().item())
                    if target_v < args.min_target_valid or (modis_v < args.min_modis_valid and viirs_v < args.min_viirs_valid):
                        continue

                    era5 = (era5 - era5.mean(dim=(0, 2, 3), keepdim=True)) / era5.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
                    static = (static - static.mean(dim=(0, 2, 3), keepdim=True)) / static.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)

                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                        pred = base_net(modis, viirs, era5, doy, static, modis_m, viirs_m).squeeze(1)
                        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                        if not m.any():
                            continue
                        loss = F.smooth_l1_loss(pred[m], y[m])
                    if not did_val_sample:
                        p = pred[m]
                        yt = y[m]
                        print(
                            "base val sample "
                            f"pred({_tstat(p)}) target({_tstat(yt)}) "
                            f"modis({_tstat(modis)}) viirs({_tstat(viirs)}) "
                            f"era5({_tstat(era5)}) static({_tstat(static)})"
                        )
                        did_val_sample = True

                    err = (pred - y)[m]
                    va_sq += float((err * err).sum().item())
                    va_n += int(err.numel())
                    va_loss.append(loss.item())

            tr_loss_v = float(np.mean(tr_loss)) if tr_loss else float("nan")
            va_loss_v = float(np.mean(va_loss)) if va_loss else float("nan")
            tr_rmse = float(np.sqrt(tr_sq / tr_n)) if tr_n > 0 else float("nan")
            va_rmse = float(np.sqrt(va_sq / va_n)) if va_n > 0 else float("nan")
            elapsed = time.perf_counter() - t0
            print(
                f"base epoch={epoch} train_loss={tr_loss_v:.6f} val_loss={va_loss_v:.6f} "
                f"train_rmse={tr_rmse:.4f} val_rmse={va_rmse:.4f} time={elapsed:.1f}s"
            )

            if np.isfinite(va_rmse) and va_rmse < best_val:
                best_val = va_rmse
                torch.save(base_net.state_dict(), out_dir / "base_best.pt")
                print("saved best base")
                bad_epochs = 0
            elif np.isfinite(va_rmse):
                if va_rmse > best_val + args.early_delta:
                    bad_epochs += 1
                    if bad_epochs >= args.early_patience:
                        print(f"early stop base at epoch {epoch}")
                        break

    res_device = device if not args.dummy else torch.device("cpu")
    # Residual training
    if args.dummy:
        base_net = base_net.to(res_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    s2_ch = root_30m["products_30m"]["sentinel2"]["data"].shape[1]
    s1_ch = None
    if args.res_downsample != 1:
        raise RuntimeError("res_downsample must be 1 (no res downsampling allowed).")
    if args.residual_only and args.res_ckpt:
        args.res_checkpoint = True
    unet_channels = tuple(int(x) for x in args.res_unet.split(",") if x.strip())
    if len(unet_channels) != 4:
        raise RuntimeError("--res-unet must have 4 comma-separated ints, e.g. 8,16,24,32")
    res_net = ResidualNet(
        s2_ch=s2_ch,
        s1_ch=s1_ch,
        dem_ch=1,
        lc_num_classes=None,
        lc_one_hot=False,
        era5_ch=len(era5_idx),
        base_ch=1,
        unet_channels=unet_channels,
        s2_out=args.res_s2_out,
        s1_out=8,
        dem_out=args.res_dem_out,
        head_dropout=args.res_dropout,
        head_ch=args.res_head_ch,
        use_checkpoint=args.res_checkpoint,
    ).to(res_device)
    if args.res_ckpt:
        if not Path(args.res_ckpt).exists():
            raise RuntimeError(f"res checkpoint not found: {args.res_ckpt}")
        res_net.load_state_dict(torch.load(args.res_ckpt, map_location=res_device))
        print(f"loaded res ckpt: {args.res_ckpt}")
    if args.res_ckpt:
        ckpt_path = args.res_ckpt
        if not Path(ckpt_path).exists():
            raise RuntimeError(f"res checkpoint not found: {ckpt_path}")
        res_state = torch.load(ckpt_path, map_location=res_device)
        res_net.load_state_dict(res_state)
        print(f"loaded res ckpt: {ckpt_path}")
    train_res = ResidualFullScene(train_idx, root_30m, root_daily, args.lst_min, args.lst_max, era5_idx)
    val_res = ResidualFullScene(val_idx, root_30m, root_daily, args.lst_min, args.lst_max, era5_idx)
    train_loader_r = DataLoader(train_res, batch_size=args.batch_size, shuffle=True, collate_fn=_stack_residual)
    val_loader_r = DataLoader(val_res, batch_size=args.batch_size, shuffle=False, collate_fn=_stack_residual)

    def _run_phase(epochs, lr_peak, lr_end, warmup_start):
        for p in base_net.parameters():
            p.requires_grad = False
        params = [{"params": res_net.parameters(), "lr": lr_peak, "weight_decay": args.wd_res}]
        # base_net is frozen for residual training
        opt_r = torch.optim.AdamW(params, betas=(0.9, 0.999), eps=1e-8)
        scaler_r = torch.amp.GradScaler("cuda", enabled=res_device.type == "cuda")
        total_steps = epochs * max(1, len(train_loader_r))
        warmup_steps = args.warmup_epochs_res * max(1, len(train_loader_r))
        sched = _build_cosine_schedule_steps(opt_r, total_steps, warmup_steps, warmup_start, lr_peak, lr_end)

        best_val_r = float("inf")
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            res_net.train()
            base_net.eval()
            tr_sq = 0.0
            tr_n = 0
            did_train_sample = False
            for s2, s1, dem, lc, era5, y, m, modis_frames, modis_masks, viirs_frames, viirs_masks, era5_c, static_c, doy in train_loader_r:
                s2 = torch.nan_to_num(s2.to(res_device), nan=0.0)
                s1 = torch.nan_to_num(s1.to(res_device), nan=0.0) if s1 is not None else None
                dem = torch.nan_to_num(dem.to(res_device), nan=0.0)
                lc = torch.nan_to_num(lc.to(res_device), nan=0.0) if lc is not None else None
                era5 = torch.nan_to_num(era5.to(res_device), nan=0.0)
                modis_frames = torch.nan_to_num(modis_frames.to(res_device), nan=0.0)
                modis_masks = torch.nan_to_num(modis_masks.to(res_device), nan=0.0)
                viirs_frames = torch.nan_to_num(viirs_frames.to(res_device), nan=0.0)
                viirs_masks = torch.nan_to_num(viirs_masks.to(res_device), nan=0.0)
                era5_c = torch.nan_to_num(era5_c.to(res_device), nan=0.0)
                static_c = torch.nan_to_num(static_c.to(res_device), nan=0.0)
                doy = doy.to(res_device)
                y = y.to(res_device)
                m = m.to(res_device) > 0
                m = m & torch.isfinite(y)
                y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                target_v = float(m.float().mean().item())
                s2_v = float(torch.isfinite(s2).float().mean().item())
                if target_v < args.min_target_valid or s2_v < args.min_s2_valid:
                    continue

                era5 = (era5 - era5.mean(dim=(0, 2, 3), keepdim=True)) / era5.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
                era5_c = (era5_c - era5_c.mean(dim=(0, 2, 3), keepdim=True)) / era5_c.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
                static_c = (static_c - static_c.mean(dim=(0, 2, 3), keepdim=True)) / static_c.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
                with torch.no_grad():
                    base_c = base_net(modis_frames, viirs_frames, era5_c, doy, static_c, modis_masks, viirs_masks)
                base_hr = F.interpolate(base_c, size=y.shape[-2:], mode="bilinear", align_corners=False)
                # no residual downsampling allowed
                with torch.amp.autocast("cuda", enabled=res_device.type == "cuda"):
                    pred = res_net(base_hr, s2, s1, dem, lc, era5, return_residual=False).squeeze(1)
                    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                    if not m.any():
                        continue
                    loss = F.smooth_l1_loss(pred[m], y[m])
                if not did_train_sample:
                    p = pred[m]
                    yt = y[m]
                    print(
                        "res train sample "
                        f"pred({_tstat(p)}) target({_tstat(yt)}) "
                        f"s2({_tstat(s2)}) dem({_tstat(dem)}) era5({_tstat(era5)})"
                    )
                    did_train_sample = True

                opt_r.zero_grad(set_to_none=True)
                scaler_r.scale(loss).backward()
                scaler_r.unscale_(opt_r)
                torch.nn.utils.clip_grad_norm_(res_net.parameters(), args.grad_clip_res)
                scale_before = scaler_r.get_scale()
                scaler_r.step(opt_r)
                scaler_r.update()
                if scaler_r.get_scale() >= scale_before:
                    sched.step()

                err = (pred - y)[m]
                tr_sq += float((err * err).sum().item())
                tr_n += int(err.numel())

            # val
            res_net.eval()
            base_net.eval()
            va_sq = 0.0
            va_n = 0
            did_val_sample = False
            with torch.no_grad():
                for s2, s1, dem, lc, era5, y, m, modis_frames, modis_masks, viirs_frames, viirs_masks, era5_c, static_c, doy in val_loader_r:
                    s2 = torch.nan_to_num(s2.to(res_device), nan=0.0)
                    s1 = torch.nan_to_num(s1.to(res_device), nan=0.0) if s1 is not None else None
                    dem = torch.nan_to_num(dem.to(res_device), nan=0.0)
                    lc = torch.nan_to_num(lc.to(res_device), nan=0.0) if lc is not None else None
                    era5 = torch.nan_to_num(era5.to(res_device), nan=0.0)
                    era5 = (era5 - era5.mean(dim=(0, 2, 3), keepdim=True)) / era5.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
                    modis_frames = torch.nan_to_num(modis_frames.to(res_device), nan=0.0)
                    modis_masks = torch.nan_to_num(modis_masks.to(res_device), nan=0.0)
                    viirs_frames = torch.nan_to_num(viirs_frames.to(res_device), nan=0.0)
                    viirs_masks = torch.nan_to_num(viirs_masks.to(res_device), nan=0.0)
                    era5_c = torch.nan_to_num(era5_c.to(res_device), nan=0.0)
                    static_c = torch.nan_to_num(static_c.to(res_device), nan=0.0)
                    doy = doy.to(res_device)
                    y = y.to(res_device)
                    m = m.to(res_device) > 0
                    m = m & torch.isfinite(y)
                    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                    target_v = float(m.float().mean().item())
                    s2_v = float(torch.isfinite(s2).float().mean().item())
                    if target_v < args.min_target_valid or s2_v < args.min_s2_valid:
                        continue

                    era5_c = (era5_c - era5_c.mean(dim=(0, 2, 3), keepdim=True)) / era5_c.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
                    static_c = (static_c - static_c.mean(dim=(0, 2, 3), keepdim=True)) / static_c.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
                    base_c = base_net(modis_frames, viirs_frames, era5_c, doy, static_c, modis_masks, viirs_masks)
                    base_hr = F.interpolate(base_c, size=y.shape[-2:], mode="bilinear", align_corners=False)
                    # no residual downsampling allowed
                    with torch.amp.autocast("cuda", enabled=res_device.type == "cuda"):
                        pred = res_net(base_hr, s2, s1, dem, lc, era5, return_residual=False).squeeze(1)
                        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                        if not m.any():
                            continue
                    if not did_val_sample:
                        p = pred[m]
                        yt = y[m]
                        print(
                            "res val sample "
                            f"pred({_tstat(p)}) target({_tstat(yt)}) "
                            f"s2({_tstat(s2)}) dem({_tstat(dem)}) era5({_tstat(era5)})"
                        )
                        did_val_sample = True
                    err = (pred - y)[m]
                    va_sq += float((err * err).sum().item())
                    va_n += int(err.numel())

            tr_rmse = float(np.sqrt(tr_sq / tr_n)) if tr_n > 0 else float("nan")
            va_rmse = float(np.sqrt(va_sq / va_n)) if va_n > 0 else float("nan")
            elapsed = time.perf_counter() - t0
            print(f"res epoch={epoch} train_rmse={tr_rmse:.4f} val_rmse={va_rmse:.4f} time={elapsed:.1f}s")

            if np.isfinite(va_rmse) and va_rmse < best_val_r:
                best_val_r = va_rmse
                torch.save(res_net.state_dict(), out_dir / "res_best.pt")
                print("saved best residual")
                bad_epochs = 0
            elif np.isfinite(va_rmse):
                if va_rmse > best_val_r + args.early_delta:
                    bad_epochs += 1
                    if bad_epochs >= args.early_patience:
                        print(f"early stop residual at epoch {epoch}")
                        break

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _run_phase(args.residual_r0, lr_peak=args.lr_res, lr_end=2e-6, warmup_start=1e-5)


if __name__ == "__main__":
    main()
