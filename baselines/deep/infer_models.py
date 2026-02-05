import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import zarr

from helper.metrics_image import compute_all

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"
COMMON_DATES = PROJECT_ROOT / "common_dates.csv"

TARGET_SCALE = 10000.0
LANDSAT_NODATA = 149
LANDSAT_MIN_VALID_K = 273.0


def _to_str(arr):
    return np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
    )


class SimpleCNN(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class PixelMLP(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


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


def normalize_batch(x, in_ch=None):
    x = ensure_nchw(x, in_ch=in_ch)
    finite = torch.isfinite(x)
    x0 = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    denom = finite.sum(dim=(0, 2, 3)).clamp(min=1)
    mean = (x0 * finite).sum(dim=(0, 2, 3)) / denom
    var = ((x0 - mean[None, :, None, None]) ** 2 * finite).sum(dim=(0, 2, 3)) / denom
    std = var.sqrt().clamp(min=1e-6)
    return (x0 - mean[None, :, None, None]) / std[None, :, None, None]


def _apply_modis_mask(modis_lr):
    if modis_lr.shape[0] < 2:
        return modis_lr
    lst = modis_lr[0].astype(np.float32) + 273.15
    mask = modis_lr[1].astype(np.float32)
    valid = mask == 0
    lst = np.where(valid, lst, np.nan)
    return np.stack([lst, mask], axis=0)


def _apply_viirs_mask(viirs_lr):
    if viirs_lr.shape[0] < 2:
        return viirs_lr
    lst = viirs_lr[0].astype(np.float32)
    mask = viirs_lr[1].astype(np.float32)
    valid = mask <= 1
    lst = np.where(valid, lst, np.nan)
    return np.stack([lst, mask], axis=0)


def build_patch(root_30m, root_daily, t, m, patch_size, rng):
    landsat_shape = root_30m["labels_30m"]["landsat"]["data"].shape
    H_hr, W_hr = landsat_shape[-2], landsat_shape[-1]

    modis_shape = root_daily["products"]["modis"]["data"].shape
    viirs_shape = root_daily["products"]["viirs"]["data"].shape
    H_lr_modis, W_lr_modis = modis_shape[-2], modis_shape[-1]
    H_lr_viirs, W_lr_viirs = viirs_shape[-2], viirs_shape[-1]

    y_map_modis = np.floor(np.linspace(0, H_lr_modis - 1, H_hr)).astype(np.int64)
    x_map_modis = np.floor(np.linspace(0, W_lr_modis - 1, W_hr)).astype(np.int64)
    y_map_viirs = np.floor(np.linspace(0, H_lr_viirs - 1, H_hr)).astype(np.int64)
    x_map_viirs = np.floor(np.linspace(0, W_lr_viirs - 1, W_hr)).astype(np.int64)

    y0 = rng.integers(0, H_hr - patch_size + 1)
    x0 = rng.integers(0, W_hr - patch_size + 1)
    y1 = y0 + patch_size
    x1 = x0 + patch_size

    era5 = root_30m["products_30m"]["era5"]["data"][t, :, y0:y1, x0:x1]
    s1 = root_30m["products_30m"]["sentinel1"]["data"][m, :, y0:y1, x0:x1]
    s2 = root_30m["products_30m"]["sentinel2"]["data"][m, :, y0:y1, x0:x1]
    dem = root_30m["static_30m"]["dem"]["data"][0, :, y0:y1, x0:x1]
    world = root_30m["static_30m"]["worldcover"]["data"][0, :, y0:y1, x0:x1]
    dyn = root_30m["static_30m"]["dynamic_world"]["data"][0, :, y0:y1, x0:x1]

    y = root_30m["labels_30m"]["landsat"]["data"][t, 0, y0:y1, x0:x1]
    y = np.where(y == LANDSAT_NODATA, np.nan, y)
    y = np.where(y < LANDSAT_MIN_VALID_K, np.nan, y)

    y_idx_m = y_map_modis[y0:y1]
    x_idx_m = x_map_modis[x0:x1]
    y_idx_v = y_map_viirs[y0:y1]
    x_idx_v = x_map_viirs[x0:x1]

    modis_lr = root_daily["products"]["modis"]["data"][t, :, :, :]
    viirs_lr = root_daily["products"]["viirs"]["data"][t, :, :, :]
    modis_lr = _apply_modis_mask(modis_lr)
    viirs_lr = _apply_viirs_mask(viirs_lr)
    modis = modis_lr[:, y_idx_m][:, :, x_idx_m]
    viirs = viirs_lr[:, y_idx_v][:, :, x_idx_v]

    x = np.concatenate([era5, modis, viirs, s1, s2, dem, world, dyn], axis=0)
    return x.astype(np.float32), y.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="date in YYYY_MM_DD or YYYY-MM-DD")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root_30m = zarr.open_group(str(ROOT_30M), mode="r")
    root_daily = zarr.open_group(str(ROOT_DAILY), mode="r")

    common_df = pd.read_csv(COMMON_DATES)
    common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna()
    common_dates = pd.DatetimeIndex(common_dates).sort_values()

    daily_raw = _to_str(root_30m["time"]["daily"][:])
    daily_times = pd.to_datetime(daily_raw, format="%Y_%m_%d", errors="coerce").dropna()

    date = pd.to_datetime(args.date.replace("_", "-"), errors="coerce")
    if pd.isna(date):
        raise SystemExit("Invalid date format. Use YYYY_MM_DD or YYYY-MM-DD.")
    if date not in daily_times:
        raise SystemExit("Date not found in madurai_30m daily timeline.")
    t = int(np.where(daily_times == date)[0][0])

    month_index = pd.DatetimeIndex(common_dates.to_period("M").to_timestamp())
    monthly_raw = _to_str(root_30m["time"]["monthly"][:])
    monthly_times = pd.to_datetime(monthly_raw, format="%Y_%m", errors="coerce").dropna()
    monthly_map = {t: i for i, t in enumerate(monthly_times)}
    m = int(monthly_map.get(date.to_period("M").to_timestamp()))
    if m < 0:
        raise SystemExit("Month index not found for date.")

    rng = np.random.default_rng(args.seed)
    x_np, y_np = build_patch(root_30m, root_daily, t, m, args.patch_size, rng)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xb = torch.from_numpy(x_np[None, ...]).to(device)
    yb = torch.from_numpy(y_np[None, ...]).to(device)

    # CNN
    cnn_ckpt = PROJECT_ROOT / "models" / "deep_baselines" / "cnn" / "cnn_best.pt"
    cnn = None
    if cnn_ckpt.exists():
        ckpt = torch.load(cnn_ckpt, map_location=device)
        cnn = SimpleCNN(in_ch=int(ckpt["in_ch"])).to(device)
        cnn.load_state_dict(ckpt["model_state_dict"])
        cnn.eval()
        with torch.no_grad():
            x_cnn = ensure_nchw(xb, in_ch=int(ckpt["in_ch"]))
            if not torch.isfinite(x_cnn).all():
                x_cnn = fill_nan_nearest(x_cnn)
            x_cnn = normalize_batch(x_cnn, in_ch=int(ckpt["in_ch"]))
            pred = cnn(x_cnn).squeeze(1).cpu().numpy()[0] * TARGET_SCALE
        y_true = y_np
        metrics = compute_all(y_true, pred)
        metrics = {k: metrics[k] for k in ("rmse", "ssim", "psnr", "sam", "cc")}
        print("cnn metrics:", metrics)
    else:
        print(f"cnn model not found: {cnn_ckpt}")

    # MLP
    mlp_ckpt = PROJECT_ROOT / "models" / "deep_baselines" / "mlp" / "mlp_best.pt"
    if mlp_ckpt.exists():
        ckpt = torch.load(mlp_ckpt, map_location=device)
        mlp = PixelMLP(in_ch=int(ckpt["in_ch"])).to(device)
        mlp.load_state_dict(ckpt["model_state_dict"])
        mlp.eval()
        with torch.no_grad():
            x_mlp = ensure_nchw(xb, in_ch=int(ckpt["in_ch"]))
            if not torch.isfinite(x_mlp).all():
                x_mlp = fill_nan_nearest(x_mlp)
            x_mlp = normalize_batch(x_mlp, in_ch=int(ckpt["in_ch"]))
            x_flat = x_mlp.permute(0, 2, 3, 1).reshape(-1, int(ckpt["in_ch"]))
            pred = mlp(x_flat).reshape(args.patch_size, args.patch_size).cpu().numpy() * TARGET_SCALE
        y_true = y_np
        metrics = compute_all(y_true, pred)
        metrics = {k: metrics[k] for k in ("rmse", "ssim", "psnr", "sam", "cc")}
        print("mlp metrics:", metrics)
    else:
        print(f"mlp model not found: {mlp_ckpt}")

    # LightGBM
    lgb_path = PROJECT_ROOT / "models" / "deep_baselines" / "lightgbm" / "lightgbm_best.txt"
    if lgb_path.exists():
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(lgb_path))
        x_lgb = ensure_nchw(xb, in_ch=xb.shape[1])
        if not torch.isfinite(x_lgb).all():
            x_lgb = fill_nan_nearest(x_lgb)
        x_lgb = normalize_batch(x_lgb, in_ch=xb.shape[1])
        x_flat = x_lgb.permute(0, 2, 3, 1).reshape(-1, xb.shape[1]).cpu().numpy()
        pred = booster.predict(x_flat).reshape(args.patch_size, args.patch_size) * TARGET_SCALE
        y_true = y_np
        metrics = compute_all(y_true, pred)
        metrics = {k: metrics[k] for k in ("rmse", "ssim", "psnr", "sam", "cc")}
        print("lightgbm metrics:", metrics)
    else:
        print(f"lightgbm model not found: {lgb_path}")


if __name__ == "__main__":
    main()
