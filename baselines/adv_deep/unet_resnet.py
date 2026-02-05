from pathlib import Path
from datetime import datetime
import copy
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import zarr
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

from helper.split_utils import load_or_create_splits
from helper import eval_utils

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"
COMMON_DATES = PROJECT_ROOT / "common_dates.csv"
SPLITS_PATH = PROJECT_ROOT / "metrics" / "common_date_splits.csv"
OUT_DIR = PROJECT_ROOT / "metrics" / "adv_deep" / "unet_resnet"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models" / "adv_deep" / "unet_resnet"
LOG_DIR = PROJECT_ROOT / "logs" / "new"
LOG_PATH = LOG_DIR / f"unet_resnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

EPS_Y = 1e-6

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

_log_f = open(LOG_PATH, "w", buffering=1)


def log(msg: str) -> None:
    print(msg)
    _log_f.write(str(msg) + "\n")


root_30m = zarr.open_group(str(ROOT_30M), mode="r")
root_daily = zarr.open_group(str(ROOT_DAILY), mode="r")

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


def _to_str(arr):
    return np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
    )


common_df = pd.read_csv(COMMON_DATES)
common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna()
common_dates = pd.DatetimeIndex(common_dates).sort_values()

daily_raw = _to_str(root_30m["time"]["daily"][:])
monthly_raw = _to_str(root_30m["time"]["monthly"][:])

daily_times = pd.to_datetime(daily_raw, format="%Y_%m_%d", errors="coerce").dropna()
monthly_times = pd.to_datetime(monthly_raw, format="%Y_%m", errors="coerce").dropna()

daily_idx = np.flatnonzero(daily_times.isin(common_dates))
month_index = pd.DatetimeIndex(common_dates.to_period("M").to_timestamp())
monthly_idx = np.flatnonzero(monthly_times.isin(month_index))

monthly_map = {t: i for i, t in enumerate(monthly_times)}
daily_to_month = []
for t in daily_times[daily_idx]:
    m = t.to_period("M").to_timestamp()
    daily_to_month.append(monthly_map.get(m, -1))
daily_to_month = np.array(daily_to_month)
valid = daily_to_month >= 0
daily_idx = daily_idx[valid]
daily_to_month = daily_to_month[valid]
daily_to_month_map = {int(t): int(m) for t, m in zip(daily_idx, daily_to_month)}

log(f"daily_idx={len(daily_idx)} monthly_idx={len(monthly_idx)}")

landsat_shape = root_30m["labels_30m"]["landsat"]["data"].shape
H_hr, W_hr = landsat_shape[-2], landsat_shape[-1]

modis_shape = root_daily["products"]["modis"]["data"].shape
viirs_shape = root_daily["products"]["viirs"]["data"].shape
H_lr_modis, W_lr_modis = modis_shape[-2], modis_shape[-1]
H_lr_viirs, W_lr_viirs = viirs_shape[-2], viirs_shape[-1]

row_float_modis = np.linspace(0, H_lr_modis - 1, H_hr, dtype=np.float64)
col_float_modis = np.linspace(0, W_lr_modis - 1, W_hr, dtype=np.float64)
row_float_viirs = np.linspace(0, H_lr_viirs - 1, H_hr, dtype=np.float64)
col_float_viirs = np.linspace(0, W_lr_viirs - 1, W_hr, dtype=np.float64)

patch_size = 128
INPUT_SOURCES = ("era5", "modis", "viirs", "s1", "s2", "dem", "world", "dyn")


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


def _nearest_patch(arr, r_f, c_f):
    Hc, Wc = arr.shape
    r = np.clip(np.rint(r_f).astype(np.int64), 0, Hc - 1)
    c = np.clip(np.rint(c_f).astype(np.int64), 0, Wc - 1)
    return arr[r[:, None], c[None, :]].astype(np.float32)


def _coarse_stats(arr):
    arr = arr.astype(np.float32, copy=False)
    Hc, Wc = arr.shape
    pad = np.pad(arr, ((1, 1), (1, 1)), mode="constant", constant_values=np.nan)
    windows = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            windows.append(pad[1 + dr : 1 + dr + Hc, 1 + dc : 1 + dc + Wc])
    stack = np.stack(windows, axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0).astype(np.float32)
        std = np.nanstd(stack, axis=0).astype(np.float32)
        med = np.nanmedian(stack, axis=0).astype(np.float32)
    left = pad[1 : 1 + Hc, 0:Wc]
    right = pad[1 : 1 + Hc, 2 : 2 + Wc]
    up = pad[0:Hc, 1 : 1 + Wc]
    down = pad[2 : 2 + Hc, 1 : 1 + Wc]
    dx = np.where(np.isfinite(left) & np.isfinite(right), right - left, np.nan)
    dy = np.where(np.isfinite(up) & np.isfinite(down), down - up, np.nan)
    grad = np.where(np.isfinite(dx) & np.isfinite(dy), np.sqrt(dx * dx + dy * dy), np.nan)
    grad = grad.astype(np.float32)
    return mean, std, med, grad


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
    valid_qc = np.isfinite(qc) & (qc == 1)
    valid_lst = np.isfinite(lst) & (lst != -9999.0) & (lst > 0)
    valid = valid_qc & valid_lst
    lst = np.where(valid, lst, np.nan)
    return lst, valid.astype(np.float32), qc


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
    valid_qc = np.isfinite(qc) & (qc <= 1)
    valid_lst = np.isfinite(lst) & (lst != -9999.0) & (lst >= 273.0)
    valid = valid_qc & valid_lst
    lst = np.where(valid, lst, np.nan)
    lst = lst - 273.15
    return lst, valid.astype(np.float32), qc


def _monthly_median_map(g_modis, g_viirs):
    modis_month = {}
    viirs_month = {}
    months = np.unique(daily_to_month)
    for m in months:
        if m < 0:
            continue
        day_idx = daily_idx[daily_to_month == m]
        if day_idx.size == 0:
            continue
        modis_stack = []
        viirs_stack = []
        for t in day_idx:
            modis_lr = g_modis[int(t), :, :, :]
            viirs_lr = g_viirs[int(t), :, :, :]
            modis_lst, _, _ = _extract_modis(modis_lr)
            viirs_lst, _, _ = _extract_viirs(viirs_lr)
            modis_stack.append(modis_lst)
            viirs_stack.append(viirs_lst)
        modis_month[int(m)] = (
            np.nanmedian(np.stack(modis_stack, axis=0), axis=0).astype(np.float32)
            if modis_stack
            else None
        )
        viirs_month[int(m)] = (
            np.nanmedian(np.stack(viirs_stack, axis=0), axis=0).astype(np.float32)
            if viirs_stack
            else None
        )
    return modis_month, viirs_month


MODIS_MONTH_MED, VIIRS_MONTH_MED = _monthly_median_map(
    root_daily["products"]["modis"]["data"],
    root_daily["products"]["viirs"]["data"],
)


class TileDataset(Dataset):
    def __init__(self, items, seed=0, allowed_t=None):
        self.items = list(items)
        self.rng = np.random.default_rng(seed)
        self.allowed_t = np.array(allowed_t) if allowed_t is not None else np.array(daily_idx)

        self.g_era5 = root_30m["products_30m"]["era5"]["data"]
        self.g_landsat = root_30m["labels_30m"]["landsat"]["data"]
        self.g_s1 = root_30m["products_30m"]["sentinel1"]["data"]
        self.g_s2 = root_30m["products_30m"]["sentinel2"]["data"]
        self.g_modis = root_daily["products"]["modis"]["data"]
        self.g_viirs = root_daily["products"]["viirs"]["data"]

        self.g_dem = root_30m["static_30m"]["dem"]["data"]
        self.g_world = root_30m["static_30m"]["worldcover"]["data"]
        self.g_dyn = root_30m["static_30m"]["dynamic_world"]["data"]

    def _build_components(self, t, y0, x0):
        t = int(t)
        y0 = int(y0)
        x0 = int(x0)
        y1 = y0 + patch_size
        x1 = x0 + patch_size

        m = int(daily_to_month_map[t])

        era5 = self.g_era5[t, :, y0:y1, x0:x1]
        s1 = self.g_s1[m, :, y0:y1, x0:x1]
        s2 = self.g_s2[m, :, y0:y1, x0:x1]
        dem = self.g_dem[0, :, y0:y1, x0:x1]
        world = self.g_world[0, :, y0:y1, x0:x1]
        dyn = self.g_dyn[0, :, y0:y1, x0:x1]

        y = self.g_landsat[t, 0, y0:y1, x0:x1]

        modis_lr = self.g_modis[t, :, :, :]
        viirs_lr = self.g_viirs[t, :, :, :]

        m_month = daily_to_month_map.get(int(t), -1)
        modis_month_med = MODIS_MONTH_MED.get(m_month)
        viirs_month_med = VIIRS_MONTH_MED.get(m_month)

        modis_lst, modis_valid, modis_qc = _extract_modis(modis_lr)
        viirs_lst, viirs_valid, viirs_qc = _extract_viirs(viirs_lr)

        modis_mean3, modis_std3, modis_med3, modis_grad = _coarse_stats(modis_lst)
        viirs_mean3, viirs_std3, viirs_med3, viirs_grad = _coarse_stats(viirs_lst)

        modis_imputed = (~np.isfinite(modis_lst)) & np.isfinite(modis_med3)
        viirs_imputed = (~np.isfinite(viirs_lst)) & np.isfinite(viirs_med3)
        modis_up = np.where(np.isfinite(modis_lst), modis_lst, modis_med3)
        viirs_up = np.where(np.isfinite(viirs_lst), viirs_lst, viirs_med3)

        if modis_month_med is not None:
            modis_anom = np.where(np.isfinite(modis_up), modis_up - modis_month_med, np.nan)
        else:
            modis_anom = np.full_like(modis_up, np.nan)
        if viirs_month_med is not None:
            viirs_anom = np.where(np.isfinite(viirs_up), viirs_up - viirs_month_med, np.nan)
        else:
            viirs_anom = np.full_like(viirs_up, np.nan)

        modis_mean = float(np.nanmean(modis_lst)) if np.isfinite(modis_lst).any() else np.nan
        modis_std = float(np.nanstd(modis_lst)) if np.isfinite(modis_lst).any() else np.nan
        viirs_mean = float(np.nanmean(viirs_lst)) if np.isfinite(viirs_lst).any() else np.nan
        viirs_std = float(np.nanstd(viirs_lst)) if np.isfinite(viirs_lst).any() else np.nan
        modis_mean_map = np.full_like(modis_lst, modis_mean, dtype=np.float32)
        modis_std_map = np.full_like(modis_lst, modis_std, dtype=np.float32)
        viirs_mean_map = np.full_like(viirs_lst, viirs_mean, dtype=np.float32)
        viirs_std_map = np.full_like(viirs_lst, viirs_std, dtype=np.float32)

        r_m = row_float_modis[y0:y1]
        c_m = col_float_modis[x0:x1]
        r_v = row_float_viirs[y0:y1]
        c_v = col_float_viirs[x0:x1]

        modis = np.stack(
            [
                _nearest_patch(modis_lst, r_m, c_m),
                _bilinear_patch(modis_up, r_m, c_m),
                _bilinear_patch(modis_mean3, r_m, c_m),
                _bilinear_patch(modis_std3, r_m, c_m),
                _bilinear_patch(modis_med3, r_m, c_m),
                _bilinear_patch(modis_grad, r_m, c_m),
                _bilinear_patch(modis_anom, r_m, c_m),
                _nearest_patch(modis_valid, r_m, c_m),
                _nearest_patch(modis_imputed.astype(np.float32), r_m, c_m),
                _nearest_patch(modis_qc, r_m, c_m),
                _nearest_patch(modis_mean_map, r_m, c_m),
                _nearest_patch(modis_std_map, r_m, c_m),
            ],
            axis=0,
        )
        viirs = np.stack(
            [
                _nearest_patch(viirs_lst, r_v, c_v),
                _bilinear_patch(viirs_up, r_v, c_v),
                _bilinear_patch(viirs_mean3, r_v, c_v),
                _bilinear_patch(viirs_std3, r_v, c_v),
                _bilinear_patch(viirs_med3, r_v, c_v),
                _bilinear_patch(viirs_grad, r_v, c_v),
                _bilinear_patch(viirs_anom, r_v, c_v),
                _nearest_patch(viirs_valid, r_v, c_v),
                _nearest_patch(viirs_imputed.astype(np.float32), r_v, c_v),
                _nearest_patch(viirs_qc, r_v, c_v),
                _nearest_patch(viirs_mean_map, r_v, c_v),
                _nearest_patch(viirs_std_map, r_v, c_v),
            ],
            axis=0,
        )

        y = np.where(y == 149, np.nan, y)
        if np.isfinite(y).any() and np.nanmedian(y) > 200:
            y = y - 273.15

        return {
            "era5": era5,
            "modis": modis,
            "viirs": viirs,
            "s1": s1,
            "s2": s2,
            "dem": dem,
            "world": world,
            "dyn": dyn,
            "y": y,
        }

    def _sample_components(self):
        i = self.rng.integers(0, len(self.allowed_t))
        t = int(self.allowed_t[i])
        y0 = self.rng.integers(0, H_hr - patch_size + 1)
        x0 = self.rng.integers(0, W_hr - patch_size + 1)
        return self._build_components(t, y0, x0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        t, y0, x0 = self.items[idx]
        comp = self._build_components(t, y0, x0)
        x = np.concatenate([comp[name] for name in INPUT_SOURCES], axis=0)
        return torch.from_numpy(x).float(), torch.from_numpy(comp["y"]).float()

    def sample_components(self):
        return self._sample_components()

    def get_components_at(self, t, y0, x0):
        return self._build_components(t, y0, x0)


def fill_nan_nearest(x):
    if torch.isfinite(x).all():
        return x
    x0 = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    h, w = x.shape[-2:]
    x_low = F.interpolate(x0, scale_factor=0.5, mode="nearest")
    x_up = F.interpolate(x_low, size=(h, w), mode="nearest")
    return torch.where(torch.isfinite(x), x, x_up)


def ensure_nchw(x, in_ch=None):
    if x.dim() != 4:
        return x
    if in_ch is None:
        if x.shape[1] > x.shape[-1]:
            return x.permute(0, 3, 1, 2).contiguous()
        return x
    if x.shape[1] == in_ch:
        return x
    if x.shape[-1] == in_ch:
        return x.permute(0, 3, 1, 2).contiguous()
    if x.shape[1] < x.shape[-1]:
        return x.permute(0, 3, 1, 2).contiguous()
    return x


def normalize_batch_global(x, mu, sigma, mask_idx):
    x = ensure_nchw(x, in_ch=x.shape[1])
    x0 = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu_t = torch.as_tensor(mu, device=x0.device, dtype=x0.dtype)[None, :, None, None]
    sigma_t = torch.as_tensor(sigma, device=x0.device, dtype=x0.dtype)[None, :, None, None]
    out = (x0 - mu_t) / sigma_t
    if mask_idx:
        out[:, mask_idx, :, :] = x0[:, mask_idx, :, :]
    return out


def build_items(dates, n_items, seed):
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n_items):
        t = int(rng.choice(dates))
        y0 = int(rng.integers(0, H_hr - patch_size + 1))
        x0 = int(rng.integers(0, W_hr - patch_size + 1))
        items.append((t, y0, x0))
    return items


def concat_inputs(comp):
    return np.concatenate([comp[name] for name in INPUT_SOURCES], axis=0)


def input_shapes_line(comp, total_ch):
    parts = [f"{name}={comp[name].shape}" for name in INPUT_SOURCES]
    parts.append(f"y={comp['y'].shape}")
    parts.append(f"total_ch={total_ch}")
    return "input_shapes " + " ".join(parts)


def get_mask_channel_indices(comp):
    offsets = {}
    idx = 0
    for name in INPUT_SOURCES:
        offsets[name] = idx
        idx += comp[name].shape[0]
    mask_idx = []
    if "modis" in offsets:
        modis_off = offsets["modis"]
        mask_idx.extend(
            [
                modis_off + 7,
                modis_off + 8,
                modis_off + 9,
            ]
        )
    if "viirs" in offsets:
        viirs_off = offsets["viirs"]
        mask_idx.extend(
            [
                viirs_off + 7,
                viirs_off + 8,
                viirs_off + 9,
            ]
        )
    if "world" in offsets:
        world_off = offsets["world"]
        for i in range(comp["world"].shape[0]):
            mask_idx.append(world_off + i)
    if "dyn" in offsets:
        dyn_off = offsets["dyn"]
        for i in range(comp["dyn"].shape[0]):
            mask_idx.append(dyn_off + i)
    return sorted(set(mask_idx))


def compute_input_stats(dataset: TileDataset, items, n_samples, mask_idx):
    rng = np.random.default_rng(123)
    picks = rng.choice(len(items), size=min(n_samples, len(items)), replace=False)
    sum_x = None
    sum_sq = None
    count = None
    for i in picks:
        t, y0, x0 = items[i]
        comp = dataset.get_components_at(t, y0, x0)
        x_raw = concat_inputs(comp)
        x = x_raw.reshape(x_raw.shape[0], -1)
        finite = np.isfinite(x)
        if sum_x is None:
            sum_x = np.zeros(x.shape[0], dtype=np.float64)
            sum_sq = np.zeros(x.shape[0], dtype=np.float64)
            count = np.zeros(x.shape[0], dtype=np.float64)
        for ch in range(x.shape[0]):
            if ch in mask_idx:
                continue
            m = finite[ch]
            if np.any(m):
                vals = x[ch][m]
                sum_x[ch] += float(np.sum(vals))
                sum_sq[ch] += float(np.sum(vals * vals))
                count[ch] += float(vals.size)
    mu = np.zeros_like(sum_x)
    sigma = np.ones_like(sum_x)
    for ch in range(sum_x.shape[0]):
        if ch in mask_idx:
            mu[ch] = 0.0
            sigma[ch] = 1.0
            continue
        if count[ch] > 0:
            mu[ch] = sum_x[ch] / count[ch]
            var = max(0.0, sum_sq[ch] / count[ch] - mu[ch] * mu[ch])
            sigma[ch] = float(np.sqrt(var)) if var > 0 else 1.0
        else:
            mu[ch] = 0.0
            sigma[ch] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)


def sample_values(name, arr, k=5):
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        log(f"{name}: all NaN (skipped)")
        return
    vals = finite.ravel()[:k]
    log(f"{name}: {vals.tolist()}")


def compute_target_stats(dataset: Dataset, n_samples: int = 200) -> tuple:
    ys = []
    for _ in range(n_samples):
        comp = dataset.sample_components()
        y = comp["y"]
        if np.isfinite(y).any():
            ys.append(y[np.isfinite(y)])
    if not ys:
        return 0.0, 1.0
    vals = np.concatenate(ys, axis=0)
    mu = float(np.nanmean(vals))
    sigma = float(np.nanstd(vals))
    if not np.isfinite(sigma) or sigma < EPS_Y:
        sigma = 1.0
    return mu, sigma


def save_loss_plot(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(df["epoch"], df["train_loss"], label="train")
    ax.plot(df["epoch"], df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("unet_resnet loss")
    ax.legend()
    fig.savefig(out_dir / "unet_resnet_loss.png", dpi=150)
    plt.close(fig)


def build_roi_mask(shape):
    try:
        root = zarr.open_group(str(ROOT_30M), mode="r")
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
                log(f"ROI mask skipped: pyproj unavailable ({exc})")
                return None
            try:
                src = CRS.from_epsg(4326)
                dst = CRS.from_string(str(crs_str))
                transformer = Transformer.from_crs(src, dst, always_xy=True)
                roi_coords = [transformer.transform(lon, lat) for lon, lat in roi_coords]
                transformed = True
            except Exception as exc:
                log(f"ROI mask skipped: failed CRS transform ({exc})")
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
            log("ROI mask empty; skipping ROI plot.")
            return None
        return mask
    except Exception as exc:
        log(f"ROI mask generation failed: {exc}")
        return None


def save_roi_figure(mask: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.imshow(mask.astype(np.float32), cmap="viridis")
    ax.set_title("Madurai ROI mask")
    ax.axis("off")
    fig.savefig(out_dir / "unet_resnet_roi_mask.png", dpi=150)
    plt.close(fig)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(x)
        out = self.act(out + identity)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        self.enc1 = ResBlock(in_ch, base)
        self.enc2 = ResBlock(base, base * 2, stride=2)
        self.enc3 = ResBlock(base * 2, base * 4, stride=2)
        self.enc4 = ResBlock(base * 4, base * 8, stride=2)
        self.bottleneck = ResBlock(base * 8, base * 16, stride=2)

        self.dec4 = UpBlock(base * 16, base * 8, base * 8)
        self.dec3 = UpBlock(base * 8, base * 4, base * 4)
        self.dec2 = UpBlock(base * 4, base * 2, base * 2)
        self.dec1 = UpBlock(base * 2, base, base)
        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head(d1)


def _tile_starts(full_size: int, tile: int) -> list:
    if full_size <= tile:
        return [0]
    starts = list(range(0, full_size - tile + 1, tile))
    last = full_size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def predict_full_map(model_in, dataset: TileDataset, t_idx: int, device, in_ch: int, mu_x, sigma_x, mask_idx, mu_y, sigma_y):
    model_in.eval()
    H, W = H_hr, W_hr
    y_pred = np.full((H, W), np.nan, dtype=np.float32)
    y_true = np.full((H, W), np.nan, dtype=np.float32)
    ys = _tile_starts(H, patch_size)
    xs = _tile_starts(W, patch_size)
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                comp = dataset.get_components_at(t_idx, y0, x0)
                x_raw = concat_inputs(comp)
                xb = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
                xb = ensure_nchw(xb, in_ch=in_ch)
                if not torch.isfinite(xb).all():
                    xb = fill_nan_nearest(xb)
                xb = normalize_batch_global(xb, mu_x, sigma_x, mask_idx)
                pred = model_in(xb).squeeze(0).squeeze(0).cpu().numpy()
                pred = pred * (sigma_y + EPS_Y) + mu_y
                y_patch = comp["y"]
                y_true[y0 : y0 + patch_size, x0 : x0 + patch_size] = y_patch
                y_pred[y0 : y0 + patch_size, x0 : x0 + patch_size] = pred
    return y_true, y_pred


def sample_rmse(model_in, dataset: TileDataset, t_idx: int, device, in_ch: int, mu_x, sigma_x, mask_idx, mu_y, sigma_y, *, n_tiles: int, seed: int):
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_tiles):
        y0 = int(rng.integers(0, H_hr - patch_size + 1))
        x0 = int(rng.integers(0, W_hr - patch_size + 1))
        comp = dataset.get_components_at(t_idx, y0, x0)
        x_raw = concat_inputs(comp)
        xb = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
        xb = ensure_nchw(xb, in_ch=in_ch)
        if not torch.isfinite(xb).all():
            xb = fill_nan_nearest(xb)
        xb = normalize_batch_global(xb, mu_x, sigma_x, mask_idx)
        with torch.no_grad():
            pred = model_in(xb).squeeze(0).squeeze(0).cpu().numpy()
            pred = pred * (sigma_y + EPS_Y) + mu_y
        y_true = comp["y"]
        m = np.isfinite(y_true) & np.isfinite(pred)
        if np.any(m):
            err = pred[m] - y_true[m]
            errs.append(err)
    if not errs:
        return float("nan"), 0
    err_all = np.concatenate(errs, axis=0)
    rmse_sampled = float(np.sqrt(np.mean(err_all ** 2))) if err_all.size else float("nan")
    return rmse_sampled, int(err_all.size)


splits = load_or_create_splits(COMMON_DATES, SPLITS_PATH)
train_dates = pd.DatetimeIndex(splits["train"]).normalize()
val_dates = pd.DatetimeIndex(splits["val"]).normalize()
test_dates = pd.DatetimeIndex(splits["test"]).normalize()
daily_norm = pd.DatetimeIndex(daily_times).normalize()
train_dates = [int(t) for t in daily_idx if daily_norm[int(t)] in train_dates]
val_dates = [int(t) for t in daily_idx if daily_norm[int(t)] in val_dates]
test_dates = [int(t) for t in daily_idx if daily_norm[int(t)] in test_dates]

split_rows = []
for split_name, dates in (("train", train_dates), ("val", val_dates), ("test", test_dates)):
    for t in dates:
        split_rows.append({"split": split_name, "date": daily_times[int(t)].strftime("%Y-%m-%d")})
split_path = OUT_DIR / "unet_date_splits.csv"
pd.DataFrame(split_rows).to_csv(split_path, index=False)
log(f"saved date splits: {split_path}")

samples_per_epoch_train = 1000
samples_val = 500
samples_test = 500

train_items = build_items(train_dates, samples_per_epoch_train, seed=11)
val_items = build_items(val_dates, samples_val, seed=22)
test_items = build_items(test_dates, samples_test, seed=33)

train_ds = TileDataset(train_items, seed=1, allowed_t=train_dates)
val_ds = TileDataset(val_items, seed=2, allowed_t=val_dates)
test_ds = TileDataset(test_items, seed=3, allowed_t=test_dates)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_comp = train_ds.get_components_at(*train_items[0])
sample_x = concat_inputs(sample_comp)
in_ch = sample_x.shape[0]
mask_idx = get_mask_channel_indices(sample_comp)
mu_x, sigma_x = compute_input_stats(train_ds, train_items, n_samples=300, mask_idx=mask_idx)
log(f"input_stats computed: ch={in_ch} mask_ch={len(mask_idx)}")
log(input_shapes_line(sample_comp, in_ch))

mu_y, sigma_y = compute_target_stats(train_ds, n_samples=200)
log(f"target_stats mu={mu_y:.6f} sigma={sigma_y:.6f}")

model = UNetResNet(in_ch=in_ch).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.SmoothL1Loss()
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
)

max_epochs = 100

history = []
best_val = float("inf")
best_epoch = 0
best_state = None

for epoch in range(1, max_epochs + 1):
    t0 = time.time()
    model.train()
    train_losses = []
    train_sq = 0.0
    train_n = 0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        xb = ensure_nchw(xb, in_ch=in_ch)

        if not torch.isfinite(xb).all():
            xb = fill_nan_nearest(xb)
        xb = normalize_batch_global(xb, mu_x, sigma_x, mask_idx)

        finite_tgt = torch.isfinite(yb)
        if not finite_tgt.any():
            continue
        yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
        yb = (yb - mu_y) / (sigma_y + EPS_Y)

        opt.zero_grad(set_to_none=True)
        pred = model(xb).squeeze(1)
        loss = loss_fn(pred[finite_tgt], yb[finite_tgt])
        if not torch.isfinite(loss):
            continue
        err = (pred - yb)[finite_tgt]
        train_sq += float((err * err).sum().item())
        train_n += int(err.numel())
        loss.backward()
        opt.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    val_sq = 0.0
    val_n = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb = ensure_nchw(xb, in_ch=in_ch)

            if not torch.isfinite(xb).all():
                xb = fill_nan_nearest(xb)
            xb = normalize_batch_global(xb, mu_x, sigma_x, mask_idx)

            finite_tgt = torch.isfinite(yb)
            if not finite_tgt.any():
                continue
            yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
            yb = (yb - mu_y) / (sigma_y + EPS_Y)

            pred = model(xb).squeeze(1)
            loss = loss_fn(pred[finite_tgt], yb[finite_tgt])
            if not torch.isfinite(loss):
                continue
            err = (pred - yb)[finite_tgt]
            val_sq += float((err * err).sum().item())
            val_n += int(err.numel())
            val_losses.append(loss.item())

    train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
    val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
    train_rmse = float(np.sqrt(train_sq / train_n)) if train_n > 0 else float("nan")
    val_rmse = float(np.sqrt(val_sq / val_n)) if val_n > 0 else float("nan")
    train_rmse_renorm = train_rmse * (sigma_y + EPS_Y) if np.isfinite(train_rmse) else float("nan")
    val_rmse_renorm = val_rmse * (sigma_y + EPS_Y) if np.isfinite(val_rmse) else float("nan")
    epoch_s = float(time.time() - t0)
    log(
        "epoch="
        f"{epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
        f"train_rmse={train_rmse_renorm:.6f} val_rmse={val_rmse_renorm:.6f} "
        f"epoch_s={epoch_s:.1f}"
    )
    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_rmse": train_rmse_renorm,
            "val_rmse": val_rmse_renorm,
            "epoch_time_s": epoch_s,
        }
    )
    lr_scheduler.step(val_loss if np.isfinite(val_loss) else train_loss)

    if np.isfinite(val_loss) and val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        model_path = MODEL_DIR / "unet_resnet_best.pt"
        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": best_state,
                "in_ch": in_ch,
                "target_mu": mu_y,
                "target_sigma": sigma_y,
            },
            model_path,
        )
        log(f"saved best model: {model_path}")

    if epoch % 50 == 0:
        comp = train_ds.sample_components()
        log(f"epoch={epoch} sample_values (unnormalized)")
        for name in list(INPUT_SOURCES) + ["y"]:
            sample_values(name, comp[name])
        x_raw = concat_inputs(comp)
        xb = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
        xb = ensure_nchw(xb, in_ch=in_ch)
        if not torch.isfinite(xb).all():
            xb = fill_nan_nearest(xb)
        xb = normalize_batch_global(xb, mu_x, sigma_x, mask_idx)
        with torch.no_grad():
            pred = model(xb).squeeze(0).squeeze(0).cpu().numpy()
        pred_renorm = pred * (sigma_y + EPS_Y) + mu_y
        sample_values("pred_renorm", pred_renorm)
        sample_values("y_renorm", comp["y"])

if best_state is not None:
    model.load_state_dict(best_state)

df_hist = pd.DataFrame(history)
csv_path = OUT_DIR / "unet_metrics.csv"
df_hist.to_csv(csv_path, index=False)
log(f"saved metrics csv: {csv_path}")
save_loss_plot(df_hist, FIG_DIR)

roi_mask = eval_utils.build_roi_mask(ROOT_30M, (H_hr, W_hr))
if roi_mask is not None:
    eval_utils.save_roi_figure(roi_mask, FIG_DIR / "roi_mask.png")

eval_rows = []
sample_tiles = 8
figure_date = str(pd.Timestamp(daily_times[int(test_dates[-1])]).date())
for t in test_dates:
    y_true, y_pred = predict_full_map(model, train_ds, int(t), device, in_ch, mu_x, sigma_x, mask_idx, mu_y, sigma_y)
    met = eval_utils.compute_metrics(y_true, y_pred, roi_mask=roi_mask)
    rmse_sampled, n_sampled = sample_rmse(
        model,
        train_ds,
        int(t),
        device,
        in_ch,
        mu_x,
        sigma_x,
        mask_idx,
        mu_y,
        sigma_y,
        n_tiles=sample_tiles,
        seed=int(t) + 123,
    )
    eval_rows.append(
        {
            "time": str(pd.Timestamp(daily_times[int(t)]).date()),
            **{k: met[k] for k in ("rmse", "ssim", "psnr", "sam", "cc")},
            "rmse_sum": met["rmse_sum"],
            "rmse_sampled": rmse_sampled,
            "n_valid": met["n_valid"],
            "n_sampled": int(n_sampled),
        }
    )
    if str(pd.Timestamp(daily_times[int(t)]).date()) == figure_date:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
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
        fig.savefig(FIG_DIR / "unet_pred_sample.png", dpi=150)
        plt.close(fig)

metrics_path = OUT_DIR / "unet_eval_metrics.csv"
pd.DataFrame(eval_rows).to_csv(metrics_path, index=False)
if eval_rows:
    keys = ("rmse", "ssim", "psnr", "sam", "cc")
    vals = [float(eval_rows[0][k]) for k in keys]
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    ax.bar(keys, vals)
    ax.set_title("unet metrics")
    ax.set_ylabel("value")
    fig.savefig(FIG_DIR / "unet_metrics_bar.png", dpi=150)
    plt.close(fig)
log(f"saved eval metrics csv: {metrics_path}")
