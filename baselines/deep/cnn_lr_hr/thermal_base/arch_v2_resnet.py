from __future__ import annotations

from pathlib import Path
import argparse
import logging
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import zarr

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"
COMMON_DATES = PROJECT_ROOT / "common_dates.csv"

ERA5_TOP4 = [1, 4, 3, 0]
EPS = 1e-6
LST_MIN_C = 10.0
LST_MAX_C = 70.0

_ap = argparse.ArgumentParser()
_ap.add_argument("--run-name", default="arch_v2_resnet")
_ap.add_argument("--epochs", type=int, default=60)
_ap.add_argument("--batch-size", type=int, default=4)
_ap.add_argument("--samples-per-epoch", type=int, default=1200)
_ap.add_argument("--samples-val", type=int, default=300)
_ap.add_argument("--patch-size", type=int, default=256)
_ap.add_argument("--full-scene", action="store_true", help="Use full-scene LR grid per date")
_ap.add_argument("--seed", type=int, default=42)
_args = _ap.parse_args()

RUN_TAG = _args.run_name
PATCH_SIZE = int(_args.patch_size)
OUT_DIR = PROJECT_ROOT / "baselines" / "deep" / "cnn_lr_hr" / "thermal_base" / RUN_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "train.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("arch_v2_resnet")

root_30m = zarr.open_group(str(ROOT_30M), mode="r")
root_daily = zarr.open_group(str(ROOT_DAILY), mode="r")


def _to_str(arr):
    return np.array([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr])


def _landsat_to_celsius(arr):
    arr = arr.astype(np.float32, copy=False)
    arr = np.where(arr == 149, np.nan, arr)
    g = root_30m["labels_30m"]["landsat"]
    attrs = dict(g.attrs)
    scale = attrs.get("scale_factor", attrs.get("scale", 1.0)) or 1.0
    offset = attrs.get("add_offset", attrs.get("offset", 0.0)) or 0.0
    if scale != 1.0 or offset != 0.0:
        arr = arr * float(scale) + float(offset)
    if np.isfinite(arr).any() and np.nanmedian(arr) > 200:
        arr = arr - 273.15
    return arr


def _apply_range_mask(arr):
    valid = np.isfinite(arr) & (arr >= LST_MIN_C) & (arr <= LST_MAX_C)
    out = np.where(valid, arr, np.nan)
    return out.astype(np.float32), valid


def _extract_modis(modis_lr):
    lst = modis_lr[0].astype(np.float32)
    qc = modis_lr[4].astype(np.float32) if modis_lr.shape[0] >= 6 else modis_lr[1].astype(np.float32)
    valid = np.isfinite(qc) & (qc == 1) & np.isfinite(lst) & (lst != -9999.0) & (lst > 0)
    lst = np.where(valid, lst, np.nan)
    if np.isfinite(lst).any() and np.nanmedian(lst) > 200:
        lst = lst - 273.15
    return lst, valid.astype(np.float32)


def _extract_viirs(viirs_lr):
    lst = viirs_lr[0].astype(np.float32)
    qc = viirs_lr[2].astype(np.float32) if viirs_lr.shape[0] >= 4 else viirs_lr[1].astype(np.float32)
    valid = np.isfinite(qc) & (qc <= 1) & np.isfinite(lst) & (lst != -9999.0) & (lst >= 273.0)
    lst = np.where(valid, lst, np.nan)
    lst = lst - 273.15
    return lst, valid.astype(np.float32)


def _resize(x, size, mode):
    t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    if mode == "nearest":
        t = F.interpolate(t, size=size, mode=mode)
    else:
        t = F.interpolate(t, size=size, mode=mode, align_corners=False)
    return t.squeeze(0).squeeze(0).numpy().astype(np.float32, copy=False)


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


def _nanaware_resample(arr, out_h, out_w):
    r = np.linspace(0, arr.shape[0] - 1, out_h, dtype=np.float64)
    c = np.linspace(0, arr.shape[1] - 1, out_w, dtype=np.float64)
    return _bilinear_patch(arr, r, c)


def _build_items(dates, n_samples, H, W, seed, full_scene):
    if full_scene:
        return [(int(t), 0, 0) for t in dates]
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n_samples):
        t = int(rng.choice(dates))
        y0 = int(rng.integers(0, H - PATCH_SIZE + 1))
        x0 = int(rng.integers(0, W - PATCH_SIZE + 1))
        items.append((t, y0, x0))
    return items


class BaseDataset(Dataset):
    def __init__(self, items, daily_times, H_hr, W_hr, H_lr, W_lr, full_scene: bool):
        self.items = list(items)
        self.daily_times = daily_times
        self.H_hr, self.W_hr = H_hr, W_hr
        self.H_lr, self.W_lr = H_lr, W_lr
        self.full_scene = full_scene
        if full_scene:
            self.coarse_h, self.coarse_w = H_lr, W_lr
        else:
            self.coarse_h = max(1, int(round(PATCH_SIZE * (self.H_lr / self.H_hr))))
            self.coarse_w = max(1, int(round(PATCH_SIZE * (self.W_lr / self.W_hr))))
        self.g_modis = root_daily["products"]["modis"]["data"]
        self.g_viirs = root_daily["products"]["viirs"]["data"]
        self.g_era5 = root_30m["products_30m"]["era5"]["data"]
        self.g_landsat = root_30m["labels_30m"]["landsat"]["data"]
        self.g_dem = root_30m["static_30m"]["dem"]["data"]
        self.g_world = root_30m["static_30m"]["worldcover"]["data"]
        self.g_dyn = root_30m["static_30m"]["dynamic_world"]["data"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        t, y0, x0 = self.items[idx]
        if self.full_scene:
            y0, x0 = 0, 0
            y1, x1 = self.H_hr, self.W_hr
            r0, c0 = 0, 0
            r1, c1 = self.H_lr, self.W_lr
        else:
            y1 = y0 + PATCH_SIZE
            x1 = x0 + PATCH_SIZE
            r0 = int(round(y0 * self.H_lr / self.H_hr))
            c0 = int(round(x0 * self.W_lr / self.W_hr))
            r1 = min(self.H_lr, r0 + self.coarse_h)
            c1 = min(self.W_lr, c0 + self.coarse_w)

        modis_lr = self.g_modis[t, :, :, :]
        viirs_lr = self.g_viirs[t, :, :, :]
        modis_lst, modis_m = _extract_modis(modis_lr)
        viirs_lst, viirs_m = _extract_viirs(viirs_lr)
        modis = modis_lst[r0:r1, c0:c1]
        modis_m = modis_m[r0:r1, c0:c1]
        viirs = viirs_lst[r0:r1, c0:c1]
        viirs_m = viirs_m[r0:r1, c0:c1]
        target_size = (self.coarse_h, self.coarse_w)
        if modis.shape != target_size:
            modis = _resize(modis, target_size, "bilinear")
            modis_m = _resize(modis_m, target_size, "nearest")
        if viirs.shape != target_size:
            viirs = _resize(viirs, target_size, "bilinear")
            viirs_m = _resize(viirs_m, target_size, "nearest")

        era5 = self.g_era5[t, :, y0:y1, x0:x1][ERA5_TOP4]
        era5 = torch.from_numpy(np.nan_to_num(era5, nan=0.0)).float().unsqueeze(0)
        era5 = F.interpolate(era5, size=target_size, mode="bilinear", align_corners=False)
        era5 = era5.squeeze(0).numpy()

        dem = _resize(self.g_dem[0, 0, y0:y1, x0:x1], target_size, "bilinear")
        world = _resize(self.g_world[0, 0, y0:y1, x0:x1], target_size, "nearest")
        dyn = _resize(self.g_dyn[0, 0, y0:y1, x0:x1], target_size, "nearest")

        y_hr = _landsat_to_celsius(self.g_landsat[t, 0, y0:y1, x0:x1])
        y_hr, v_hr = _apply_range_mask(y_hr)
        y_lr = _nanaware_resample(y_hr, target_size[0], target_size[1])
        v_lr = _nanaware_resample(v_hr.astype(np.float32), target_size[0], target_size[1]) > 0.5
        v_lr = v_lr & np.isfinite(y_lr)
        y_lr = np.nan_to_num(y_lr, nan=0.0, posinf=0.0, neginf=0.0)

        x = np.stack([modis, modis_m, viirs, viirs_m, dem, world, dyn], axis=0).astype(np.float32)
        x = np.concatenate([x, era5.astype(np.float32)], axis=0)
        return x, y_lr.astype(np.float32), v_lr.astype(bool)


class BasicBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(4, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(4, ch)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class SimpleResNet(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
        )
        self.block1 = BasicBlock(32)
        self.block2 = BasicBlock(32)
        self.block3 = BasicBlock(32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


def main():
    daily_raw = _to_str(root_30m["time"]["daily"][:])
    daily_times = pd.to_datetime(daily_raw, format="%Y_%m_%d", errors="coerce").dropna()
    if not COMMON_DATES.exists():
        raise SystemExit(f"missing {COMMON_DATES}")
    common_df = pd.read_csv(COMMON_DATES)
    common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna().dt.normalize()
    daily_norm = pd.DatetimeIndex(daily_times).normalize()
    daily_idx = np.arange(len(daily_times), dtype=int)
    available_idx = [int(t) for t in daily_idx if daily_norm[int(t)] in set(common_dates)]
    if not available_idx:
        raise SystemExit("No dates from common_dates.csv matched to daily_times")

    landsat_shape = root_30m["labels_30m"]["landsat"]["data"].shape
    H_hr, W_hr = landsat_shape[-2], landsat_shape[-1]
    modis_shape = root_daily["products"]["modis"]["data"].shape
    H_lr, W_lr = modis_shape[-2], modis_shape[-1]

    rng = np.random.default_rng(_args.seed)
    rng.shuffle(available_idx)
    n_train = int(len(available_idx) * 0.7)
    n_val = int(len(available_idx) * 0.1)
    train_dates = available_idx[:n_train]
    val_dates = available_idx[n_train : n_train + n_val]
    test_dates = available_idx[n_train + n_val :]
    logger.info("dates train=%d val=%d test=%d", len(train_dates), len(val_dates), len(test_dates))

    train_items = _build_items(train_dates, _args.samples_per_epoch, H_hr, W_hr, seed=11, full_scene=_args.full_scene)
    val_items = _build_items(val_dates, _args.samples_val, H_hr, W_hr, seed=22, full_scene=_args.full_scene)

    train_ds = BaseDataset(train_items, daily_times, H_hr, W_hr, H_lr, W_lr, _args.full_scene)
    val_ds = BaseDataset(val_items, daily_times, H_hr, W_hr, H_lr, W_lr, _args.full_scene)

    train_loader = DataLoader(train_ds, batch_size=_args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=_args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleResNet(in_ch=11).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best = float("inf")
    for epoch in range(1, _args.epochs + 1):
        model.train()
        tr_loss = []
        tr_sq = 0.0
        tr_n = 0
        did_log = False
        for xb, yb, mb in train_loader:
            xb = torch.nan_to_num(xb.to(device), nan=0.0)
            yb = yb.to(device)
            mb = mb.to(device)
            pred = model(xb).squeeze(1)
            if not mb.any():
                continue
            if not did_log:
                logger.info("train shapes xb=%s pred=%s yb=%s mb=%s", tuple(xb.shape), tuple(pred.shape), tuple(yb.shape), tuple(mb.shape))
            if not did_log:
                p = pred[0].detach().cpu().numpy()
                t = yb[0].detach().cpu().numpy()
                logger.info(
                    "train sample pred(min=%.3f mean=%.3f max=%.3f) target(min=%.3f mean=%.3f max=%.3f)",
                    float(np.nanmin(p)),
                    float(np.nanmean(p)),
                    float(np.nanmax(p)),
                    float(np.nanmin(t)),
                    float(np.nanmean(t)),
                    float(np.nanmax(t)),
                )
                did_log = True
            loss = F.smooth_l1_loss(pred[mb], yb[mb])
            err = (pred - yb)[mb]
            tr_sq += float((err * err).sum().item())
            tr_n += int(err.numel())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss.append(loss.item())

        model.eval()
        va_loss = []
        va_sq = 0.0
        va_n = 0
        with torch.no_grad():
            did_log = False
            for xb, yb, mb in val_loader:
                xb = torch.nan_to_num(xb.to(device), nan=0.0)
                yb = yb.to(device)
                mb = mb.to(device)
                pred = model(xb).squeeze(1)
                if not mb.any():
                    continue
                if not did_log:
                    logger.info("val shapes xb=%s pred=%s yb=%s mb=%s", tuple(xb.shape), tuple(pred.shape), tuple(yb.shape), tuple(mb.shape))
                if not did_log:
                    p = pred[0].detach().cpu().numpy()
                    t = yb[0].detach().cpu().numpy()
                    logger.info(
                        "val sample pred(min=%.3f mean=%.3f max=%.3f) target(min=%.3f mean=%.3f max=%.3f)",
                        float(np.nanmin(p)),
                        float(np.nanmean(p)),
                        float(np.nanmax(p)),
                        float(np.nanmin(t)),
                        float(np.nanmean(t)),
                        float(np.nanmax(t)),
                    )
                    did_log = True
                loss = F.smooth_l1_loss(pred[mb], yb[mb])
                err = (pred - yb)[mb]
                va_sq += float((err * err).sum().item())
                va_n += int(err.numel())
                va_loss.append(loss.item())

        tr = float(np.mean(tr_loss)) if tr_loss else float("nan")
        va = float(np.mean(va_loss)) if va_loss else float("nan")
        tr_rmse = float(np.sqrt(tr_sq / tr_n)) if tr_n > 0 else float("nan")
        va_rmse = float(np.sqrt(va_sq / va_n)) if va_n > 0 else float("nan")
        logger.info("epoch=%d train_loss=%.6f val_loss=%.6f train_rmse=%.4f val_rmse=%.4f", epoch, tr, va, tr_rmse, va_rmse)
        if np.isfinite(va) and va < best:
            best = va
            torch.save({"model": model.state_dict()}, OUT_DIR / "best.pt")
            logger.info("saved best: %s", OUT_DIR / "best.pt")


if __name__ == "__main__":
    main()
