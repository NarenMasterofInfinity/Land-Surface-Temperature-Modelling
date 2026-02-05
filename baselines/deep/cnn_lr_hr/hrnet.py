from __future__ import annotations

from pathlib import Path
import argparse
import logging
import sys
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import zarr
import matplotlib.pyplot as plt

from helper import eval_utils

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"

BASE_OUT_DIR = PROJECT_ROOT / "metrics" / "deep_baselines" / "cnn_lr_hr"
BASE_MODEL_DIR = PROJECT_ROOT / "models" / "deep_baselines" / "cnn_lr_hr"
BASE_LOG_DIR = PROJECT_ROOT / "logs" / "deep_baselines" / "cnn_lr_hr"
BASE_FIG_DIR = PROJECT_ROOT / "figures" / "deep_baselines" / "cnn_lr_hr"

EPS_Y = 1e-6
INPUT_KEYS = ["era5", "s1", "s2", "dem", "world", "dyn"]

LST_MIN_C = 10.0
LST_MAX_C = 70.0
PATCH_VALID_FRAC_MIN = 0.30
DATE_VALID_FRAC_MIN = 0.15
DATE_MED_MIN_C = 10.0
DATE_MED_MAX_C = 60.0
MAX_RESAMPLE_TRIES = 10


DEFAULT_RUN_NAME = "cnn_lr_hr_hrnet"
MODEL_LABEL = "cnn lr/hr hrnet"

_ap = argparse.ArgumentParser()
_ap.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Run name for output subdir")
_ap.add_argument("--seed", type=int, default=42)
_ap.add_argument("--train-frac", type=float, default=0.7)
_ap.add_argument("--val-frac", type=float, default=0.1)
_ap.add_argument("--max-epochs", type=int, default=100)
_ap.add_argument("--batch-size", type=int, default=8)
_ap.add_argument("--lr", type=float, default=5e-4)
_ap.add_argument("--weight-decay", type=float, default=1e-4)
_ap.add_argument("--grad-clip", type=float, default=1.0)
_ap.add_argument("--warmup-epochs", type=int, default=5)
_ap.add_argument("--resume", action="store_true", help="Resume from checkpoint")
_ap.add_argument("--resume-path", default=None, help="Path to checkpoint (defaults to last.pt)")
_ap.add_argument("--samples-per-epoch", type=int, default=1000)
_ap.add_argument("--samples-val", type=int, default=500)
_ap.add_argument("--samples-test", type=int, default=500)
_ap.add_argument("--patch-size", type=int, default=256)
_ap.add_argument("--n-stats-samples", type=int, default=300)
_ap.add_argument("--n-target-samples", type=int, default=200)
_ap.add_argument("--sample-tiles-eval", type=int, default=8)
_args = _ap.parse_args()

RUN_TAG = _args.run_name
FILE_PREFIX = RUN_TAG
PATCH_SIZE = int(_args.patch_size)

OUT_DIR = BASE_OUT_DIR / RUN_TAG
FIG_DIR = BASE_FIG_DIR / RUN_TAG
MODEL_DIR = BASE_MODEL_DIR / RUN_TAG
LOG_DIR = BASE_LOG_DIR / RUN_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_path = LOG_DIR / f"{RUN_TAG}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(RUN_TAG)
logger.info("starting %s run=%s", MODEL_LABEL, RUN_TAG)


root_30m = zarr.open_group(str(ROOT_30M), mode="r")
root_daily = zarr.open_group(str(ROOT_DAILY), mode="r")


def _get_landsat_scale_offset():
    try:
        g = root_30m["labels_30m"]["landsat"]
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


LANDSAT_SCALE, LANDSAT_OFFSET = _get_landsat_scale_offset()


def _to_str(arr):
    return np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
    )


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


def _iter_chunks(shape, chunks):
    H, W = shape
    ch_y, ch_x = chunks
    for y0 in range(0, H, ch_y):
        y1 = min(H, y0 + ch_y)
        for x0 in range(0, W, ch_x):
            x1 = min(W, x0 + ch_x)
            yield slice(y0, y1), slice(x0, x1)


def _landsat_to_celsius(arr):
    arr = arr.astype(np.float32, copy=False)
    arr = np.where(arr == 149, np.nan, arr)
    if LANDSAT_SCALE != 1.0 or LANDSAT_OFFSET != 0.0:
        arr = arr * LANDSAT_SCALE + LANDSAT_OFFSET
    if np.isfinite(arr).any() and np.nanmedian(arr) > 200:
        arr = arr - 273.15
    return arr


def _apply_range_mask(arr):
    valid = np.isfinite(arr) & (arr >= LST_MIN_C) & (arr <= LST_MAX_C)
    out = np.where(valid, arr, np.nan)
    return out.astype(np.float32), valid


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
    n_total = 0
    n_valid = 0
    n_off = 0
    off_min = float("inf")
    off_max = float("-inf")
    for ys, xs in _iter_chunks(shape, chunks):
        block = np.asarray(arr2d[ys, xs])
        block = _landsat_to_celsius(block)
        finite = np.isfinite(block)
        n_total += int(finite.sum())
        v = block[finite]
        if v.size:
            vals.append(v.astype(np.float32, copy=False))
            off = v[(v < LST_MIN_C) | (v > LST_MAX_C)]
            if off.size:
                n_off += int(off.size)
                off_min = min(off_min, float(np.min(off)))
                off_max = max(off_max, float(np.max(off)))
        block_filt, _ = _apply_range_mask(block)
        n_valid += int(np.isfinite(block_filt).sum())
    if vals:
        all_vals = np.concatenate(vals, axis=0)
        p1, p5, p95, p99 = np.percentile(all_vals, [1, 5, 95, 99])
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        median_all = float(np.median(all_vals))
    else:
        vmin = vmax = p1 = p5 = p95 = p99 = median_all = float("nan")
    valid_fraction = (n_valid / n_total) if n_total > 0 else 0.0
    if vals and n_valid > 0:
        all_vals = np.concatenate(vals, axis=0)
        all_vals = all_vals[(all_vals >= LST_MIN_C) & (all_vals <= LST_MAX_C)]
        median_valid = float(np.median(all_vals)) if all_vals.size else float("nan")
    else:
        median_valid = float("nan")
    removed = n_total - n_valid
    off_min = off_min if np.isfinite(off_min) else float("nan")
    off_max = off_max if np.isfinite(off_max) else float("nan")
    return {
        "min": vmin,
        "max": vmax,
        "p1": float(p1),
        "p5": float(p5),
        "p95": float(p95),
        "p99": float(p99),
        "median": median_all,
        "median_valid": median_valid,
        "n_total": int(n_total),
        "n_valid": int(n_valid),
        "n_removed": int(removed),
        "n_off": int(n_off),
        "off_min": float(off_min),
        "off_max": float(off_max),
        "valid_fraction": float(valid_fraction),
    }


def _build_input_stack(comp, input_keys):
    parts = []
    for k in input_keys:
        arr = comp[k]
        if arr.ndim == 2:
            arr = arr[None, ...]
        parts.append(arr)
    return np.concatenate(parts, axis=0)


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


def compute_input_stats(dataset: Dataset, items, n_samples, mask_idx):
    rng = np.random.default_rng(123)
    picks = rng.choice(len(items), size=min(n_samples, len(items)), replace=False)
    sum_x = None
    sum_sq = None
    count = None
    for i in picks:
        t, y0, x0 = items[i]
        comp = dataset.get_components_at(t, y0, x0)
        x_raw = _build_input_stack(comp, INPUT_KEYS)
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


def compute_target_stats(dataset: Dataset, n_samples: int = 200) -> tuple:
    ys = []
    for _ in range(n_samples):
        comp = dataset.sample_components()
        y = comp["y"]
        valid = comp.get("target_valid")
        if valid is None:
            valid = np.isfinite(y)
        if np.any(valid):
            ys.append(y[valid])
    if not ys:
        return 0.0, 1.0
    vals = np.concatenate(ys, axis=0)
    mu = float(np.nanmean(vals))
    sigma = float(np.nanstd(vals))
    if not np.isfinite(sigma) or sigma < EPS_Y:
        sigma = 1.0
    return mu, sigma


def build_items(dates, n_items, seed, H_hr, W_hr):
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n_items):
        t = int(rng.choice(dates))
        y0 = int(rng.integers(0, H_hr - PATCH_SIZE + 1))
        x0 = int(rng.integers(0, W_hr - PATCH_SIZE + 1))
        items.append((t, y0, x0))
    return items


def save_loss_plot(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(df["epoch"], df["train_loss"], label="train")
    ax.plot(df["epoch"], df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"{MODEL_LABEL} loss")
    ax.legend()
    fig.savefig(out_dir / f"{FILE_PREFIX}_loss.png", dpi=150)
    plt.close(fig)


def save_rmse_plot(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(df["epoch"], df["train_rmse"], label="train")
    ax.plot(df["epoch"], df["val_rmse"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("rmse")
    ax.set_title(f"{MODEL_LABEL} rmse")
    ax.legend()
    fig.savefig(out_dir / f"{FILE_PREFIX}_rmse.png", dpi=150)
    plt.close(fig)


def save_prediction_figure(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    vmin = float(np.nanmin(y_true)) if np.isfinite(y_true).any() else 0.0
    vmax = float(np.nanmax(y_true)) if np.isfinite(y_true).any() else 1.0
    axes[0].imshow(y_true, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"truth ({title})")
    axes[1].imshow(y_pred, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("pred")
    err = np.abs(y_pred - y_true)
    axes[2].imshow(err, cmap="magma")
    axes[2].set_title("abs_error")
    for ax in axes:
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_patch_debug(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_valid: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    vmin = float(np.nanmin(y_true)) if np.isfinite(y_true).any() else 0.0
    vmax = float(np.nanmax(y_true)) if np.isfinite(y_true).any() else 1.0
    axes[0].imshow(y_true, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("target")
    axes[1].imshow(y_pred, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("pred")
    err = np.abs(y_pred - y_true)
    axes[2].imshow(err, cmap="magma")
    axes[2].set_title("abs_error")
    axes[3].imshow(y_valid.astype(np.float32), cmap="gray")
    axes[3].set_title("valid_mask")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_metric_bar(metrics: dict, out_path: Path) -> None:
    keys = list(metrics.keys())
    vals = [float(metrics[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    ax.bar(keys, vals)
    ax.set_title(f"{MODEL_LABEL} metrics")
    ax.set_ylabel("value")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


class ResBlock(nn.Module):
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
        # stem keeps high resolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Stage 1: one branch
        self.stage1 = _make_branch(32, blocks)

        # Stage 2: two branches (32, 64)
        self.transition1 = _downsample(32, 64)
        self.stage2_b1 = _make_branch(32, blocks)
        self.stage2_b2 = _make_branch(64, blocks)
        self.fuse2_1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, bias=False), nn.BatchNorm2d(32))
        self.fuse2_2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64))

        # Stage 3: three branches (32, 64, 128)
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

        # Stage 4: four branches (32, 64, 128, 256)
        self.transition3 = _downsample(128, 256)
        self.stage4_b1 = _make_branch(32, blocks)
        self.stage4_b2 = _make_branch(64, blocks)
        self.stage4_b3 = _make_branch(128, blocks)
        self.stage4_b4 = _make_branch(256, blocks)

        # Fusions to highest resolution for output
        self.up_2_to_1 = _upsample(64, 32, scale=2)
        self.up_3_to_1 = _upsample(128, 32, scale=4)
        self.up_4_to_1 = _upsample(256, 32, scale=8)

        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)

        # stage1
        x1 = self.stage1(x)

        # stage2
        x2 = self.transition1(x1)
        y1 = self.stage2_b1(x1)
        y2 = self.stage2_b2(x2)
        y1 = y1 + F.interpolate(self.fuse2_1(y2), size=y1.shape[-2:], mode="bilinear", align_corners=False)
        y2 = y2 + self.fuse2_2(y1)

        # stage3
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

        # stage4
        x4 = self.transition3(z3)
        w1 = self.stage4_b1(z1)
        w2 = self.stage4_b2(z2)
        w3 = self.stage4_b3(z3)
        w4 = self.stage4_b4(x4)

        # upsample all to highest resolution and sum
        w = w1
        w = w + self.up_2_to_1(w2)
        w = w + self.up_3_to_1(w3)
        w = w + self.up_4_to_1(w4)
        return self.head(w)


class CnnLrHrDataset(Dataset):
    def __init__(
        self,
        items,
        *,
        seed=0,
        allowed_t=None,
        resample_invalid=True,
        landsat_present=None,
        modis_present=None,
        viirs_present=None,
        row_float_modis=None,
        col_float_modis=None,
        row_float_viirs=None,
        col_float_viirs=None,
        daily_to_month_map=None,
        H_hr=0,
        W_hr=0,
    ):
        self.items = list(items)
        self.rng = np.random.default_rng(seed)
        self.allowed_t = np.array(allowed_t) if allowed_t is not None else None
        self.resample_invalid = resample_invalid

        self.g_era5 = root_30m["products_30m"]["era5"]["data"]
        self.g_landsat = root_30m["labels_30m"]["landsat"]["data"]
        self.g_s1 = root_30m["products_30m"]["sentinel1"]["data"]
        self.g_s2 = root_30m["products_30m"]["sentinel2"]["data"]
        self.g_modis = root_daily["products"]["modis"]["data"]
        self.g_viirs = root_daily["products"]["viirs"]["data"]
        self.g_dem = root_30m["static_30m"]["dem"]["data"]
        self.g_world = root_30m["static_30m"]["worldcover"]["data"]
        self.g_dyn = root_30m["static_30m"]["dynamic_world"]["data"]

        self.landsat_present = landsat_present
        self.modis_present = modis_present
        self.viirs_present = viirs_present
        self.row_float_modis = row_float_modis
        self.col_float_modis = col_float_modis
        self.row_float_viirs = row_float_viirs
        self.col_float_viirs = col_float_viirs
        self.daily_to_month_map = daily_to_month_map
        self.H_hr = H_hr
        self.W_hr = W_hr

    def _build_inputs(self, t, y0, x0):
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
        return {
            "era5": era5,
            "s1": s1,
            "s2": s2,
            "dem": dem,
            "world": world,
            "dyn": dyn,
        }

    def _build_target(self, t, y0, x0):
        t = int(t)
        y0 = int(y0)
        x0 = int(x0)
        y1 = y0 + PATCH_SIZE
        x1 = x0 + PATCH_SIZE

        if self.landsat_present[t]:
            y = self.g_landsat[t, 0, y0:y1, x0:x1]
            y = _landsat_to_celsius(y)
            y, valid = _apply_range_mask(y)
            return y, "landsat", valid

        modis_ok = bool(self.modis_present[t])
        viirs_ok = bool(self.viirs_present[t])
        if not (modis_ok or viirs_ok):
            y = np.full((PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)
            valid = np.zeros_like(y, dtype=bool)
            return y, "none", valid

        modis_up = None
        viirs_up = None
        if modis_ok:
            modis_lr = self.g_modis[t, :, :, :]
            modis_lst, _ = _extract_modis(modis_lr)
            r_m = self.row_float_modis[y0:y1]
            c_m = self.col_float_modis[x0:x1]
            modis_up = _bilinear_patch(modis_lst, r_m, c_m)
        if viirs_ok:
            viirs_lr = self.g_viirs[t, :, :, :]
            viirs_lst, _ = _extract_viirs(viirs_lr)
            r_v = self.row_float_viirs[y0:y1]
            c_v = self.col_float_viirs[x0:x1]
            viirs_up = _bilinear_patch(viirs_lst, r_v, c_v)

        if modis_up is not None and viirs_up is not None:
            y = np.nanmean(np.stack([modis_up, viirs_up], axis=0), axis=0)
        elif modis_up is not None:
            y = modis_up
        else:
            y = viirs_up
        y, valid = _apply_range_mask(y.astype(np.float32, copy=False))
        return y, "weak", valid

    def _build_components(self, t, y0, x0):
        comp = self._build_inputs(t, y0, x0)
        y, source, valid = self._build_target(t, y0, x0)
        comp["y"] = y
        comp["target_source"] = source
        comp["target_valid"] = valid
        comp["valid_fraction"] = float(np.mean(valid)) if valid.size else 0.0
        return comp

    def _sample_components(self):
        if self.allowed_t is None or self.allowed_t.size == 0:
            raise RuntimeError("No allowed dates available for sampling")
        i = self.rng.integers(0, len(self.allowed_t))
        t = int(self.allowed_t[i])
        y0 = self.rng.integers(0, self.H_hr - PATCH_SIZE + 1)
        x0 = self.rng.integers(0, self.W_hr - PATCH_SIZE + 1)
        return self._build_components(t, y0, x0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        t, y0, x0 = self.items[idx]
        tries = 0
        comp = self._build_components(t, y0, x0)
        while self.resample_invalid and comp["valid_fraction"] < PATCH_VALID_FRAC_MIN and tries < MAX_RESAMPLE_TRIES:
            if self.allowed_t is None or self.allowed_t.size == 0:
                break
            t = int(self.allowed_t[self.rng.integers(0, len(self.allowed_t))])
            y0 = self.rng.integers(0, self.H_hr - PATCH_SIZE + 1)
            x0 = self.rng.integers(0, self.W_hr - PATCH_SIZE + 1)
            comp = self._build_components(t, y0, x0)
            tries += 1
        x = _build_input_stack(comp, INPUT_KEYS)
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(comp["y"]).float(),
            torch.from_numpy(comp["target_valid"]).bool(),
        )

    def sample_components(self):
        return self._sample_components()

    def get_components_at(self, t, y0, x0):
        return self._build_components(t, y0, x0)


# ---- build time indices ----
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
logger.info("daily_idx=%d monthly_idx=%d", len(daily_idx), len(monthly_times))

# shapes
landsat_shape = root_30m["labels_30m"]["landsat"]["data"].shape
H_hr, W_hr = landsat_shape[-2], landsat_shape[-1]
modis_shape = root_daily["products"]["modis"]["data"].shape
viirs_shape = root_daily["products"]["viirs"]["data"].shape
H_lr_modis, W_lr_modis = modis_shape[-2], modis_shape[-1]
H_lr_viirs, W_lr_viirs = viirs_shape[-2], viirs_shape[-1]

if PATCH_SIZE > H_hr or PATCH_SIZE > W_hr:
    raise SystemExit(f"patch_size {PATCH_SIZE} exceeds scene size {H_hr}x{W_hr}")

row_float_modis = np.linspace(0, H_lr_modis - 1, H_hr, dtype=np.float64)
col_float_modis = np.linspace(0, W_lr_modis - 1, W_hr, dtype=np.float64)
row_float_viirs = np.linspace(0, H_lr_viirs - 1, H_hr, dtype=np.float64)
col_float_viirs = np.linspace(0, W_lr_viirs - 1, W_hr, dtype=np.float64)

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

# Step 1 + 4: per-date Landsat stats + date-level filtering
bad_landsat_dates = set()
total_removed = 0
for t in daily_idx:
    t = int(t)
    if not landsat_present[t]:
        continue
    stats = _landsat_date_stats(t)
    total_removed += stats["n_removed"]
    logger.info(
        "landsat_stats t=%d min=%.2f p1=%.2f p5=%.2f p95=%.2f p99=%.2f max=%.2f "
        "median=%.2f valid_frac=%.3f removed=%d off=%d off_min=%.2f off_max=%.2f",
        t,
        stats["min"],
        stats["p1"],
        stats["p5"],
        stats["p95"],
        stats["p99"],
        stats["max"],
        stats["median"],
        stats["valid_fraction"],
        stats["n_removed"],
        stats["n_off"],
        stats["off_min"],
        stats["off_max"],
    )
    if (
        stats["valid_fraction"] < DATE_VALID_FRAC_MIN
        or not np.isfinite(stats["median_valid"])
        or stats["median_valid"] < DATE_MED_MIN_C
        or stats["median_valid"] > DATE_MED_MAX_C
    ):
        bad_landsat_dates.add(t)

if bad_landsat_dates:
    logger.info("dropping %d landsat dates after QC", len(bad_landsat_dates))
if total_removed > 0:
    logger.info("landsat pixels removed by range filter: %d", total_removed)

available_idx = [
    int(t)
    for t in daily_idx
    if int(t) not in bad_landsat_dates
    and (landsat_present[int(t)] or modis_present[int(t)] or viirs_present[int(t)])
]
if not available_idx:
    raise SystemExit("No available dates with landsat/modis/viirs data.")

logger.info("available days=%d (landsat=%d modis=%d viirs=%d)", len(available_idx), int(landsat_present.sum()), int(modis_present.sum()), int(viirs_present.sum()))

# ---- split dates ----
rng = np.random.default_rng(_args.seed)
idx = np.array(available_idx, dtype=int)
rng.shuffle(idx)

n_train = int(len(idx) * _args.train_frac)
n_val = int(len(idx) * _args.val_frac)
if n_train <= 0 or n_val <= 0 or (len(idx) - n_train - n_val) <= 0:
    raise SystemExit("Invalid split fractions for available dates.")

train_dates = idx[:n_train]
val_dates = idx[n_train : n_train + n_val]
test_dates = idx[n_train + n_val :]

def _count_sources(dates):
    landsat = 0
    weak = 0
    for t in dates:
        if landsat_present[int(t)]:
            landsat += 1
        elif modis_present[int(t)] or viirs_present[int(t)]:
            weak += 1
    return landsat, weak

tr_l, tr_w = _count_sources(train_dates)
va_l, va_w = _count_sources(val_dates)
te_l, te_w = _count_sources(test_dates)
logger.info(
    "split counts train(l=%d w=%d) val(l=%d w=%d) test(l=%d w=%d)",
    tr_l,
    tr_w,
    va_l,
    va_w,
    te_l,
    te_w,
)

split_rows = []
for split_name, dates in (("train", train_dates), ("val", val_dates), ("test", test_dates)):
    for t in dates:
        split_rows.append({"split": split_name, "date": daily_times[int(t)].strftime("%Y-%m-%d")})
split_path = OUT_DIR / f"{FILE_PREFIX}_date_splits.csv"
pd.DataFrame(split_rows).to_csv(split_path, index=False)
logger.info("saved date splits: %s", split_path)

# ---- dataset + loaders ----
train_items = build_items(train_dates, _args.samples_per_epoch, seed=11, H_hr=H_hr, W_hr=W_hr)
val_items = build_items(val_dates, _args.samples_val, seed=22, H_hr=H_hr, W_hr=W_hr)
test_items = build_items(test_dates, _args.samples_test, seed=33, H_hr=H_hr, W_hr=W_hr)

train_ds = CnnLrHrDataset(
    train_items,
    seed=1,
    allowed_t=train_dates,
    resample_invalid=True,
    landsat_present=landsat_present,
    modis_present=modis_present,
    viirs_present=viirs_present,
    row_float_modis=row_float_modis,
    col_float_modis=col_float_modis,
    row_float_viirs=row_float_viirs,
    col_float_viirs=col_float_viirs,
    daily_to_month_map=daily_to_month_map,
    H_hr=H_hr,
    W_hr=W_hr,
)
val_ds = CnnLrHrDataset(
    val_items,
    seed=2,
    allowed_t=val_dates,
    resample_invalid=False,
    landsat_present=landsat_present,
    modis_present=modis_present,
    viirs_present=viirs_present,
    row_float_modis=row_float_modis,
    col_float_modis=col_float_modis,
    row_float_viirs=row_float_viirs,
    col_float_viirs=col_float_viirs,
    daily_to_month_map=daily_to_month_map,
    H_hr=H_hr,
    W_hr=W_hr,
)
test_ds = CnnLrHrDataset(
    test_items,
    seed=3,
    allowed_t=test_dates,
    resample_invalid=False,
    landsat_present=landsat_present,
    modis_present=modis_present,
    viirs_present=viirs_present,
    row_float_modis=row_float_modis,
    col_float_modis=col_float_modis,
    row_float_viirs=row_float_viirs,
    col_float_viirs=col_float_viirs,
    daily_to_month_map=daily_to_month_map,
    H_hr=H_hr,
    W_hr=W_hr,
)

train_loader = DataLoader(train_ds, batch_size=_args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=_args.batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=_args.batch_size, shuffle=False)

sample_comp = train_ds.get_components_at(*train_items[0])
sample_x = _build_input_stack(sample_comp, INPUT_KEYS)
in_ch = sample_x.shape[0]
mask_idx = get_mask_channel_indices(sample_comp, INPUT_KEYS)
mu_x, sigma_x = compute_input_stats(train_ds, train_items, n_samples=_args.n_stats_samples, mask_idx=mask_idx)
mu_y, sigma_y = compute_target_stats(train_ds, n_samples=_args.n_target_samples)
logger.info("input_stats ch=%d mask_ch=%d", in_ch, len(mask_idx))
logger.info("target_stats mu=%.6f sigma=%.6f", mu_y, sigma_y)

# ---- model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HRNetSmall(in_ch=in_ch).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=_args.lr, weight_decay=_args.weight_decay)
loss_fn = nn.SmoothL1Loss()
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt,
    T_max=max(1, _args.max_epochs - _args.warmup_epochs),
    eta_min=1e-6,
)

history = []
best_val = float("inf")
best_epoch = 0
best_state = None
start_epoch = 1

last_ckpt = MODEL_DIR / f"{FILE_PREFIX}_last.pt"
if _args.resume:
    resume_path = Path(_args.resume_path) if _args.resume_path else last_ckpt
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val = ckpt.get("best_val", best_val)
        best_epoch = ckpt.get("best_epoch", best_epoch)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        logger.info("resumed from %s at epoch=%d", resume_path, start_epoch)
    else:
        logger.warning("resume requested but checkpoint not found: %s", resume_path)

for epoch in range(start_epoch, _args.max_epochs + 1):
    if epoch <= _args.warmup_epochs:
        warm_lr = _args.lr * (epoch / max(1, _args.warmup_epochs))
        for pg in opt.param_groups:
            pg["lr"] = warm_lr
    model.train()
    train_losses = []
    train_sq = 0.0
    train_n = 0
    for xb, yb, mb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        xb = ensure_nchw(xb, in_ch=in_ch)
        if not torch.isfinite(xb).all():
            xb = fill_nan_nearest(xb)
        xb = normalize_batch_global(xb, mu_x, sigma_x, mask_idx)

        finite_tgt = mb & torch.isfinite(yb)
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
        if _args.grad_clip and _args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), _args.grad_clip)
        opt.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    val_sq = 0.0
    val_n = 0
    with torch.no_grad():
        for xb, yb, mb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            xb = ensure_nchw(xb, in_ch=in_ch)
            if not torch.isfinite(xb).all():
                xb = fill_nan_nearest(xb)
            xb = normalize_batch_global(xb, mu_x, sigma_x, mask_idx)

            finite_tgt = mb & torch.isfinite(yb)
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
    logger.info(
        "epoch=%d train_loss=%.6f val_loss=%.6f train_rmse=%.6f val_rmse=%.6f",
        epoch,
        train_loss,
        val_loss,
        train_rmse_renorm,
        val_rmse_renorm,
    )
    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_rmse": train_rmse_renorm,
            "val_rmse": val_rmse_renorm,
        }
    )
    if epoch > _args.warmup_epochs:
        lr_scheduler.step()
    if np.isfinite(val_loss) and val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        model_path = MODEL_DIR / f"{FILE_PREFIX}_best.pt"
        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": best_state,
                "in_ch": in_ch,
                "target_mu": mu_y,
                "target_sigma": sigma_y,
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "best_val": best_val,
                "best_epoch": best_epoch,
            },
            model_path,
        )
        logger.info("saved best model: %s", model_path)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "in_ch": in_ch,
            "target_mu": mu_y,
            "target_sigma": sigma_y,
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_val": best_val,
            "best_epoch": best_epoch,
        },
        last_ckpt,
    )


if best_state is not None:
    model.load_state_dict(best_state)

# ---- save training metrics ----
df_hist = pd.DataFrame(history)
csv_path = OUT_DIR / f"{FILE_PREFIX}_metrics.csv"
df_hist.to_csv(csv_path, index=False)
logger.info("saved metrics csv: %s", csv_path)

save_loss_plot(df_hist, FIG_DIR)
save_rmse_plot(df_hist, FIG_DIR)

roi_mask = eval_utils.build_roi_mask(ROOT_30M, (H_hr, W_hr))
if roi_mask is not None:
    eval_utils.save_roi_figure(roi_mask, FIG_DIR / "roi_mask.png")

# ---- evaluation ----

def _tile_starts(full_size: int, tile: int) -> list:
    if full_size <= tile:
        return [0]
    starts = list(range(0, full_size - tile + 1, tile))
    last = full_size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def predict_full_map(model, dataset: CnnLrHrDataset, t_idx: int, device, in_ch: int):
    model.eval()
    H, W = H_hr, W_hr
    y_pred = np.full((H, W), np.nan, dtype=np.float32)
    y_true = np.full((H, W), np.nan, dtype=np.float32)
    y_valid = np.zeros((H, W), dtype=bool)
    ys = _tile_starts(H, PATCH_SIZE)
    xs = _tile_starts(W, PATCH_SIZE)
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                comp = dataset.get_components_at(t_idx, y0, x0)
                x_raw = _build_input_stack(comp, INPUT_KEYS)
                xb = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
                xb = ensure_nchw(xb, in_ch=in_ch)
                if not torch.isfinite(xb).all():
                    xb = fill_nan_nearest(xb)
                xb = normalize_batch_global(xb, mu_x, sigma_x, mask_idx)
                pred = model(xb).squeeze(0).squeeze(0).cpu().numpy()
                pred = pred * (sigma_y + EPS_Y) + mu_y

                y_patch = comp["y"]
                v_patch = comp["target_valid"]
                y_true[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE] = y_patch
                y_pred[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE] = pred
                y_valid[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE] = v_patch
    return y_true, y_pred, y_valid


def sample_rmse(model, dataset: CnnLrHrDataset, t_idx: int, device, in_ch: int, *, n_tiles: int, seed: int):
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_tiles):
        y0 = int(rng.integers(0, H_hr - PATCH_SIZE + 1))
        x0 = int(rng.integers(0, W_hr - PATCH_SIZE + 1))
        comp = dataset.get_components_at(t_idx, y0, x0)
        if comp["valid_fraction"] < PATCH_VALID_FRAC_MIN:
            continue
        x_raw = _build_input_stack(comp, INPUT_KEYS)
        xb = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
        xb = ensure_nchw(xb, in_ch=in_ch)
        if not torch.isfinite(xb).all():
            xb = fill_nan_nearest(xb)
        xb = normalize_batch_global(xb, mu_x, sigma_x, mask_idx)
        with torch.no_grad():
            pred = model(xb).squeeze(0).squeeze(0).cpu().numpy()
            pred = pred * (sigma_y + EPS_Y) + mu_y
        y_true = comp["y"]
        m = comp["target_valid"] & np.isfinite(pred)
        if np.any(m):
            err = pred[m] - y_true[m]
            errs.append(err)
    if not errs:
        return float("nan"), 0
    err_all = np.concatenate(errs, axis=0)
    rmse_sampled = float(np.sqrt(np.mean(err_all ** 2))) if err_all.size else float("nan")
    return rmse_sampled, int(err_all.size)


eval_rows = []
figure_date_supervised = None
figure_date_weak = None
patch_rmse_rows = []
align_rng = np.random.default_rng(123)
align_saved = 0

for t in test_dates:
    t = int(t)
    y_true, y_pred, y_valid = predict_full_map(model, train_ds, t, device, in_ch)
    metric_mask = y_valid if roi_mask is None else (roi_mask & y_valid)
    met = eval_utils.compute_metrics(y_true, y_pred, roi_mask=metric_mask)
    rmse_sampled, n_sampled = sample_rmse(
        model,
        train_ds,
        t,
        device,
        in_ch,
        n_tiles=_args.sample_tiles_eval,
        seed=int(t) + 123,
    )
    target_source = "landsat" if landsat_present[t] else "weak"
    eval_rows.append(
        {
            "time": str(pd.Timestamp(daily_times[int(t)]).date()),
            "target_source": target_source,
            **{k: met[k] for k in ("rmse", "ssim", "psnr", "sam", "cc")},
            "rmse_sum": met["rmse_sum"],
            "rmse_sampled": rmse_sampled,
            "n_valid": met["n_valid"],
            "n_sampled": int(n_sampled),
        }
    )

    if target_source == "landsat" and figure_date_supervised is None:
        figure_date_supervised = str(pd.Timestamp(daily_times[int(t)]).date())
        save_prediction_figure(y_true, y_pred, FIG_DIR / "pred_supervised.png", "landsat")
    if target_source == "weak" and figure_date_weak is None:
        figure_date_weak = str(pd.Timestamp(daily_times[int(t)]).date())
        save_prediction_figure(y_true, y_pred, FIG_DIR / "pred_lr.png", "lowres")

    if align_saved < 10:
        for _ in range(20):
            y0 = int(align_rng.integers(0, H_hr - PATCH_SIZE + 1))
            x0 = int(align_rng.integers(0, W_hr - PATCH_SIZE + 1))
            comp = train_ds.get_components_at(t, y0, x0)
            if comp["valid_fraction"] < PATCH_VALID_FRAC_MIN:
                continue
            y_t = comp["y"]
            y_v = comp["target_valid"]
            x_raw = _build_input_stack(comp, INPUT_KEYS)
            xb = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
            xb = ensure_nchw(xb, in_ch=in_ch)
            if not torch.isfinite(xb).all():
                xb = fill_nan_nearest(xb)
            xb = normalize_batch_global(xb, mu_x, sigma_x, mask_idx)
            with torch.no_grad():
                pred = model(xb).squeeze(0).squeeze(0).cpu().numpy()
            pred = pred * (sigma_y + EPS_Y) + mu_y
            out_path = FIG_DIR / "alignment_checks" / f"align_t{t}_y{y0}_x{x0}.png"
            save_patch_debug(y_t, pred, y_v, out_path, f"align t={t} y0={y0} x0={x0}")
            align_saved += 1
            break
        if align_saved >= 10:
            logger.info("saved %d alignment check patches", align_saved)

    ys = _tile_starts(H_hr, PATCH_SIZE)
    xs = _tile_starts(W_hr, PATCH_SIZE)
    for y0 in ys:
        for x0 in xs:
            y_t = y_true[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
            y_p = y_pred[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
            y_v = y_valid[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
            valid_frac = float(np.mean(y_v)) if y_v.size else 0.0
            if valid_frac < PATCH_VALID_FRAC_MIN:
                continue
            m = y_v & np.isfinite(y_p)
            if not np.any(m):
                continue
            err = y_p[m] - y_t[m]
            rmse = float(np.sqrt(np.mean(err ** 2)))
            patch_rmse_rows.append(
                {
                    "t": t,
                    "y0": int(y0),
                    "x0": int(x0),
                    "rmse": rmse,
                    "valid_fraction": valid_frac,
                    "target_source": target_source,
                }
            )

metrics_path = OUT_DIR / f"{FILE_PREFIX}_eval_metrics.csv"
pd.DataFrame(eval_rows).to_csv(metrics_path, index=False)
logger.info("saved eval metrics csv: %s", metrics_path)


if patch_rmse_rows:
    top = sorted(patch_rmse_rows, key=lambda r: r["rmse"], reverse=True)[:20]
    for i, r in enumerate(top, start=1):
        t = int(r["t"])
        y0 = int(r["y0"])
        x0 = int(r["x0"])
        comp = train_ds.get_components_at(t, y0, x0)
        y_t = comp["y"]
        y_v = comp["target_valid"]
        x_raw = _build_input_stack(comp, INPUT_KEYS)
        xb = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
        xb = ensure_nchw(xb, in_ch=in_ch)
        if not torch.isfinite(xb).all():
            xb = fill_nan_nearest(xb)
        xb = normalize_batch_global(xb, mu_x, sigma_x, mask_idx)
        with torch.no_grad():
            pred = model(xb).squeeze(0).squeeze(0).cpu().numpy()
        pred = pred * (sigma_y + EPS_Y) + mu_y
        out_path = FIG_DIR / "rmse_top_patches" / f"top_{i:02d}_t{t}_y{y0}_x{x0}.png"
        save_patch_debug(
            y_t,
            pred,
            y_v,
            out_path,
            f"top{i:02d} rmse={r['rmse']:.3f} t={t} y0={y0} x0={x0}",
        )
    pd.DataFrame(top).to_csv(OUT_DIR / f"{FILE_PREFIX}_top_patch_rmse.csv", index=False)

if eval_rows:
    first = eval_rows[0]
    save_metric_bar(
        {k: first[k] for k in ("rmse", "ssim", "psnr", "sam", "cc")},
        FIG_DIR / f"{FILE_PREFIX}_metrics_bar.png",
    )
