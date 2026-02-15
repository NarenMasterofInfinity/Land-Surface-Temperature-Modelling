from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import zarr

from helper import eval_utils
PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"

EPS_Y = 1e-6
INPUT_KEYS = ["era5", "s1", "s2", "dem", "world", "dyn"]


_ap = argparse.ArgumentParser()
_ap.add_argument("--run-name", default="cnn_lr_hr_hrnet_200")
_ap.add_argument("--checkpoint", default=None)
_ap.add_argument("--patch-size", type=int, default=256)
_ap.add_argument("--date", required=True, help="YYYY-MM-DD date to predict")
_ap.add_argument("--out-npy", required=True, help="Output .npy path for prediction")
_ap.add_argument("--out-dir", default=None, help="Output folder for PNGs (default: presentation_review)")
_args = _ap.parse_args()

RUN_TAG = _args.run_name
PATCH_SIZE = int(_args.patch_size)

root_30m = zarr.open_group(str(ROOT_30M), mode="r")
root_daily = zarr.open_group(str(ROOT_DAILY), mode="r")

OUT_DIR = Path(_args.out_dir) if _args.out_dir else (PROJECT_ROOT / "presentation_review")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_str(arr):
    return np.array([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr])


def _get_landsat_scale_offset(root):
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


LANDSAT_SCALE, LANDSAT_OFFSET = _get_landsat_scale_offset(root_30m)


def _landsat_to_celsius(arr):
    arr = arr.astype(np.float32, copy=False)
    arr = np.where(arr == 149, np.nan, arr)
    if LANDSAT_SCALE != 1.0 or LANDSAT_OFFSET != 0.0:
        arr = arr * LANDSAT_SCALE + LANDSAT_OFFSET
    if np.isfinite(arr).any() and np.nanmedian(arr) > 200:
        arr = arr - 273.15
    return arr


def _apply_range_mask(arr):
    valid = np.isfinite(arr) & (arr >= 10.0) & (arr <= 70.0)
    out = np.where(valid, arr, np.nan)
    return out.astype(np.float32), valid


def _tile_starts(full_size: int, tile: int) -> list:
    if full_size <= tile:
        return [0]
    starts = list(range(0, full_size - tile + 1, tile))
    last = full_size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


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


def fill_nan_nearest(x):
    if torch.isfinite(x).all():
        return x
    x0 = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    h, w = x.shape[-2:]
    x_low = F.interpolate(x0, scale_factor=0.5, mode="nearest")
    x_up = F.interpolate(x_low, size=(h, w), mode="nearest")
    return torch.where(torch.isfinite(x), x, x_up)


def normalize_batch_global(x, mu, sigma, mask_idx):
    x = ensure_nchw(x, in_ch=x.shape[1])
    x0 = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu_t = torch.as_tensor(mu, device=x0.device, dtype=x0.dtype)[None, :, None, None]
    sigma_t = torch.as_tensor(sigma, device=x0.device, dtype=x0.dtype)[None, :, None, None]
    out = (x0 - mu_t) / sigma_t
    if mask_idx:
        out[:, mask_idx, :, :] = x0[:, mask_idx, :, :]
    return out


def get_mask_channel_indices(comp, input_keys):
    offsets = {}
    idx = 0
    for name in input_keys:
        offsets[name] = idx
        idx += comp[name].shape[0]
    mask_idx = []
    if "world" in offsets:
        world_off = offsets["world"]
        for i in range(comp["world"].shape[0]):
            mask_idx.append(world_off + i)
    if "dyn" in offsets:
        dyn_off = offsets["dyn"]
        for i in range(comp["dyn"].shape[0]):
            mask_idx.append(dyn_off + i)
    return sorted(set(mask_idx))


def build_input_stack(comp, input_keys):
    parts = []
    for k in input_keys:
        arr = comp[k]
        if arr.ndim == 2:
            arr = arr[None, ...]
        parts.append(arr)
    return np.concatenate(parts, axis=0)


class CnnLrHrDataset:
    def __init__(self, daily_to_month_map, H_hr, W_hr):
        self.g_era5 = root_30m["products_30m"]["era5"]["data"]
        self.g_landsat = root_30m["labels_30m"]["landsat"]["data"]
        self.g_s1 = root_30m["products_30m"]["sentinel1"]["data"]
        self.g_s2 = root_30m["products_30m"]["sentinel2"]["data"]
        self.g_dem = root_30m["static_30m"]["dem"]["data"]
        self.g_world = root_30m["static_30m"]["worldcover"]["data"]
        self.g_dyn = root_30m["static_30m"]["dynamic_world"]["data"]
        self.daily_to_month_map = daily_to_month_map
        self.H_hr = H_hr
        self.W_hr = W_hr

    def get_components_at(self, t, y0, x0):
        t = int(t)
        y0 = int(y0)
        x0 = int(x0)
        y1 = y0 + PATCH_SIZE
        x1 = x0 + PATCH_SIZE

        m = int(self.daily_to_month_map.get(int(t), -1))
        if m < 0:
            s1 = np.full((self.g_s1.shape[1], PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)
            s2 = np.full((self.g_s2.shape[1], PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)
        else:
            s1 = self.g_s1[m, :, y0:y1, x0:x1]
            s2 = self.g_s2[m, :, y0:y1, x0:x1]

        era5 = self.g_era5[t, :, y0:y1, x0:x1]
        dem = self.g_dem[0, :, y0:y1, x0:x1]
        world = self.g_world[0, :, y0:y1, x0:x1]
        dyn = self.g_dyn[0, :, y0:y1, x0:x1]

        y = self.g_landsat[t, 0, y0:y1, x0:x1]
        y = _landsat_to_celsius(y)
        y, valid = _apply_range_mask(y)

        return {
            "era5": era5,
            "s1": s1,
            "s2": s2,
            "dem": dem,
            "world": world,
            "dyn": dyn,
            "y": y,
            "target_valid": valid,
        }


class HRNetBasicBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.act(out)


def _make_branch(ch: int, blocks: int) -> nn.Sequential:
    return nn.Sequential(*[HRNetBasicBlock(ch) for _ in range(blocks)])


def _downsample(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )


def _upsample(in_ch: int, out_ch: int, scale: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_ch),
    )


class HRNetSmall(nn.Module):
    def __init__(self, in_ch: int, blocks: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.stage1 = _make_branch(32, blocks)
        self.transition1 = _downsample(32, 64)
        self.stage2_b1 = _make_branch(32, blocks)
        self.stage2_b2 = _make_branch(64, blocks)
        self.fuse2_1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, bias=False), nn.BatchNorm2d(32))
        self.fuse2_2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64))

        self.transition2 = _downsample(64, 128)
        self.stage3_b1 = _make_branch(32, blocks)
        self.stage3_b2 = _make_branch(64, blocks)
        self.stage3_b3 = _make_branch(128, blocks)
        self.fuse3_1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, bias=False), nn.BatchNorm2d(32))
        self.fuse3_1b = nn.Sequential(nn.Conv2d(128, 32, kernel_size=1, bias=False), nn.BatchNorm2d(32))
        self.fuse3_2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64))
        self.fuse3_2b = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        self.fuse3_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.fuse3_3b = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128))

        self.transition3 = _downsample(128, 256)
        self.stage4_b1 = _make_branch(32, blocks)
        self.stage4_b2 = _make_branch(64, blocks)
        self.stage4_b3 = _make_branch(128, blocks)
        self.stage4_b4 = _make_branch(256, blocks)

        self.up_2_to_1 = _upsample(64, 32, scale=2)
        self.up_3_to_1 = _upsample(128, 32, scale=4)
        self.up_4_to_1 = _upsample(256, 32, scale=8)

        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.stage1(x)

        x2 = self.transition1(x1)
        y1 = self.stage2_b1(x1)
        y2 = self.stage2_b2(x2)
        y1 = y1 + F.interpolate(self.fuse2_1(y2), size=y1.shape[-2:], mode="bilinear", align_corners=False)
        y2 = y2 + self.fuse2_2(y1)

        x3 = self.transition2(y2)
        z1 = self.stage3_b1(y1)
        z2 = self.stage3_b2(y2)
        z3 = self.stage3_b3(x3)
        z1 = z1 + F.interpolate(self.fuse3_1(z2), size=z1.shape[-2:], mode="bilinear", align_corners=False)
        z1 = z1 + F.interpolate(self.fuse3_1b(z3), size=z1.shape[-2:], mode="bilinear", align_corners=False)
        z2 = z2 + self.fuse3_2(z1)
        z2 = z2 + F.interpolate(self.fuse3_2b(z3), size=z2.shape[-2:], mode="bilinear", align_corners=False)
        z3 = z3 + self.fuse3_3(z1)
        z3 = z3 + self.fuse3_3b(z2)

        x4 = self.transition3(z3)
        w1 = self.stage4_b1(z1)
        w2 = self.stage4_b2(z2)
        w3 = self.stage4_b3(z3)
        w4 = self.stage4_b4(x4)

        w = w1
        w = w + self.up_2_to_1(w2)
        w = w + self.up_3_to_1(w3)
        w = w + self.up_4_to_1(w4)
        return self.head(w)


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
    if np.isfinite(lst).any() and np.nanmedian(lst) > 200:
        lst = lst - 273.15
    return lst, valid.astype(np.float32)


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
    return lst, valid.astype(np.float32)


def _reduce_channels(arr):
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0]
        is_binary = np.isfinite(arr).all() and np.all((arr == 0) | (arr == 1))
        if is_binary:
            return np.argmax(arr, axis=0).astype(np.float32)
        return np.nanmean(arr, axis=0).astype(np.float32)
    return np.squeeze(arr).astype(np.float32)


def _save_png(
    path: Path,
    arr2d: np.ndarray,
    cmap: str,
    vmin=None,
    vmax=None,
    *,
    add_colorbar: bool = True,
    cbar_label: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    data = arr2d.astype(np.float32, copy=False)
    if vmin is None:
        vmin = float(np.nanmin(data)) if np.isfinite(data).any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cbar.set_label(cbar_label)
    ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _robust_vmin_vmax(arr2d: np.ndarray, lo=2.0, hi=98.0):
    data = arr2d[np.isfinite(arr2d)]
    if data.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(data, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _prep_s1_for_display(arr2d: np.ndarray) -> np.ndarray:
    data = arr2d.astype(np.float32, copy=False)
    if np.isfinite(data).any() and np.nanmedian(data) > 1.0:
        data = 10.0 * np.log10(np.maximum(data, 1e-6))
    return data


def main():
    daily_raw = _to_str(root_30m["time"]["daily"][:])
    monthly_raw = _to_str(root_30m["time"]["monthly"][:])

    daily_times = pd.to_datetime(daily_raw, format="%Y_%m_%d", errors="coerce").dropna()
    monthly_times = pd.to_datetime(monthly_raw, format="%Y_%m", errors="coerce").dropna()

    monthly_map = {t: i for i, t in enumerate(monthly_times)}
    daily_to_month = []
    for t in daily_times:
        m = t.to_period("M").to_timestamp()
        daily_to_month.append(monthly_map.get(m, -1))
    daily_to_month = np.array(daily_to_month)
    valid_month = daily_to_month >= 0
    daily_idx = np.arange(len(daily_times), dtype=int)[valid_month]
    daily_to_month_map = {int(t): int(m) for t, m in zip(daily_idx, daily_to_month[valid_month])}

    landsat_shape = root_30m["labels_30m"]["landsat"]["data"].shape
    H_hr, W_hr = landsat_shape[-2], landsat_shape[-1]

    ds = CnnLrHrDataset(daily_to_month_map, H_hr, W_hr)

    sample_comp = ds.get_components_at(int(daily_idx[0]), 0, 0)
    sample_x = build_input_stack(sample_comp, INPUT_KEYS)
    in_ch = sample_x.shape[0]
    mask_idx = get_mask_channel_indices(sample_comp, INPUT_KEYS)

    model = HRNetSmall(in_ch=in_ch)
    ckpt_path = Path(_args.checkpoint) if _args.checkpoint else (PROJECT_ROOT / "models" / "deep_baselines" / "cnn_lr_hr" / RUN_TAG / f"{RUN_TAG}_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    mu_y = float(ckpt.get("target_mu", 0.0))
    sigma_y = float(ckpt.get("target_sigma", 1.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    mu_x = ckpt.get("input_mu")
    sigma_x = ckpt.get("input_sigma")
    if mu_x is None or sigma_x is None:
        rng = np.random.default_rng(123)
        picks = rng.choice(len(daily_idx), size=min(200, len(daily_idx)), replace=False)
        sum_x = np.zeros(in_ch, dtype=np.float64)
        sum_sq = np.zeros(in_ch, dtype=np.float64)
        count = np.zeros(in_ch, dtype=np.float64)
        for t in picks:
            y0 = int(rng.integers(0, H_hr - PATCH_SIZE + 1))
            x0 = int(rng.integers(0, W_hr - PATCH_SIZE + 1))
            comp = ds.get_components_at(int(t), y0, x0)
            x_raw = build_input_stack(comp, INPUT_KEYS)
            x = x_raw.reshape(x_raw.shape[0], -1)
            finite = np.isfinite(x)
            for ch in range(x.shape[0]):
                if ch in mask_idx:
                    continue
                m = finite[ch]
                if np.any(m):
                    vals = x[ch][m]
                    sum_x[ch] += float(np.sum(vals))
                    sum_sq[ch] += float(np.sum(vals * vals))
                    count[ch] += float(vals.size)
        mu_x = np.zeros_like(sum_x, dtype=np.float32)
        sigma_x = np.ones_like(sum_x, dtype=np.float32)
        for ch in range(sum_x.shape[0]):
            if ch in mask_idx:
                continue
            if count[ch] > 0:
                mu = sum_x[ch] / count[ch]
                var = max(0.0, sum_sq[ch] / count[ch] - mu * mu)
                mu_x[ch] = float(mu)
                sigma_x[ch] = float(np.sqrt(var)) if var > 0 else 1.0
    mu_x = np.asarray(mu_x, dtype=np.float32)
    sigma_x = np.asarray(sigma_x, dtype=np.float32)

    target_date = pd.to_datetime(_args.date, errors="coerce")
    if pd.isna(target_date):
        raise SystemExit(f"invalid --date: {_args.date}")
    daily_norm = pd.DatetimeIndex(daily_times).normalize()
    match = np.where(daily_norm == target_date.normalize())[0]
    if match.size == 0:
        raise SystemExit(f"date not found in daily_times: {target_date.date()}")
    t = int(match[0])

    y_pred = np.full((H_hr, W_hr), np.nan, dtype=np.float32)
    y_true = np.full((H_hr, W_hr), np.nan, dtype=np.float32)
    y_valid = np.zeros((H_hr, W_hr), dtype=bool)
    ys = _tile_starts(H_hr, PATCH_SIZE)
    xs = _tile_starts(W_hr, PATCH_SIZE)

    for y0 in ys:
        for x0 in xs:
            comp = ds.get_components_at(t, y0, x0)
            x_raw = build_input_stack(comp, INPUT_KEYS)
            xb = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
            xb = ensure_nchw(xb, in_ch=in_ch)
            if not torch.isfinite(xb).all():
                xb = fill_nan_nearest(xb)
            xb = normalize_batch_global(xb, mu_x, sigma_x, mask_idx)
            with torch.no_grad():
                pred = model(xb).squeeze(0).squeeze(0).cpu().numpy()
            pred = pred * (sigma_y + EPS_Y) + mu_y
            y_pred[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE] = pred
            y_true[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE] = comp["y"]
            y_valid[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE] = comp["target_valid"]

    out_npy = Path(_args.out_npy)
    out_npy = OUT_DIR / out_npy.name if not out_npy.is_absolute() else out_npy
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, y_pred)
    if np.any(y_valid):
        diff = y_pred[y_valid] - y_true[y_valid]
        rmse = float(np.sqrt(np.nanmean(diff * diff)))
    else:
        rmse = float("nan")
    print(f"saved {out_npy} shape={y_pred.shape} rmse={rmse}")

    # ---- save presentation PNGs ----
    g_modis = root_daily["products"]["modis"]["data"]
    g_viirs = root_daily["products"]["viirs"]["data"]
    g_dem = root_30m["static_30m"]["dem"]["data"]
    g_world = root_30m["static_30m"]["worldcover"]["data"]
    g_dyn = root_30m["static_30m"]["dynamic_world"]["data"]
    g_s1 = root_30m["products_30m"]["sentinel1"]["data"]
    g_s2 = root_30m["products_30m"]["sentinel2"]["data"]

    roi_mask = eval_utils.build_roi_mask(ROOT_30M, (H_hr, W_hr))

    m = int(daily_to_month_map.get(int(t), -1))
    if m < 0:
        s1 = np.full((g_s1.shape[1], H_hr, W_hr), np.nan, dtype=np.float32)
        s2 = np.full((g_s2.shape[1], H_hr, W_hr), np.nan, dtype=np.float32)
    else:
        s1 = g_s1[m]
        s2 = g_s2[m]

    landsat_gt = y_true
    if roi_mask is not None and roi_mask.shape == landsat_gt.shape:
        landsat_gt = np.where(roi_mask, landsat_gt, np.nan)
    modis_lst, _ = _extract_modis(g_modis[t])
    viirs_lst, _ = _extract_viirs(g_viirs[t])
    dem = _reduce_channels(g_dem[0])
    world = _reduce_channels(g_world[0])
    dyn = _reduce_channels(g_dyn[0])
    s1_vis = _reduce_channels(s1)
    s2_vis = _reduce_channels(s2)
    pred_lst = y_pred
    if roi_mask is not None and roi_mask.shape == pred_lst.shape:
        pred_lst = np.where(roi_mask, pred_lst, np.nan)

    _save_png(
        OUT_DIR / f"landsat_gt_{target_date.date()}.png",
        landsat_gt,
        cmap="inferno",
        vmin=10.0,
        vmax=70.0,
        cbar_label="LST (°C)",
    )
    _save_png(
        OUT_DIR / f"predicted_lst_{target_date.date()}.png",
        pred_lst,
        cmap="inferno",
        vmin=10.0,
        vmax=70.0,
        cbar_label="LST (°C)",
    )
    _save_png(
        OUT_DIR / f"modis_lst_{target_date.date()}.png",
        modis_lst,
        cmap="inferno",
        vmin=10.0,
        vmax=70.0,
        cbar_label="LST (°C)",
    )
    _save_png(
        OUT_DIR / f"viirs_lst_{target_date.date()}.png",
        viirs_lst,
        cmap="inferno",
        vmin=10.0,
        vmax=70.0,
        cbar_label="LST (°C)",
    )
    dem_vmin, dem_vmax = _robust_vmin_vmax(dem)
    _save_png(
        OUT_DIR / f"dem_{target_date.date()}.png",
        dem,
        cmap="terrain",
        vmin=dem_vmin,
        vmax=dem_vmax,
        cbar_label="Elevation (m)",
    )
    _save_png(
        OUT_DIR / f"worldcover_{target_date.date()}.png",
        world,
        cmap="tab20",
        add_colorbar=True,
        cbar_label="Class ID",
    )
    _save_png(
        OUT_DIR / f"dynamicworld_{target_date.date()}.png",
        dyn,
        cmap="tab20",
        add_colorbar=True,
        cbar_label="Class ID",
    )
    s1_disp = _prep_s1_for_display(s1_vis)
    s1_vmin, s1_vmax = _robust_vmin_vmax(s1_disp)
    _save_png(
        OUT_DIR / f"sentinel1_{target_date.date()}.png",
        s1_disp,
        cmap="gray",
        vmin=s1_vmin,
        vmax=s1_vmax,
        cbar_label="Backscatter (dB)" if np.isfinite(s1_disp).any() else None,
    )
    s2_vmin, s2_vmax = _robust_vmin_vmax(s2_vis)
    _save_png(
        OUT_DIR / f"sentinel2_{target_date.date()}.png",
        s2_vis,
        cmap="gray",
        vmin=s2_vmin,
        vmax=s2_vmax,
        cbar_label="Reflectance (a.u.)",
    )


if __name__ == "__main__":
    main()
