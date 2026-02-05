from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import torch
from torch.utils.data import Dataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class SplitInfo:
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]
    split_dates: Dict[str, str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _list_dataset_vars(ds: xr.Dataset, prefix: str) -> List[str]:
    return [f"{prefix}{name}" for name in ds.data_vars.keys()]


def _try_open_group(zarr_path: str, group: Optional[str]) -> Optional[xr.Dataset]:
    try:
        return xr.open_zarr(zarr_path, group=group, consolidated=False)
    except Exception:
        return None


def _normalize_dims(da: xr.DataArray) -> xr.DataArray:
    dims = list(da.dims)
    time_dim = None
    for cand in ["time", "t", "date", "datetime"]:
        if cand in dims:
            time_dim = cand
            break
    if time_dim is None:
        time_dim = dims[0]

    spatial_dims = [d for d in dims if d != time_dim]
    if len(spatial_dims) >= 2:
        y_dim, x_dim = spatial_dims[-2], spatial_dims[-1]
    else:
        raise ValueError(f"Expected at least 3 dims (time,y,x), got {dims}")

    rename_map = {}
    if time_dim != "time":
        rename_map[time_dim] = "time"
    if y_dim != "y":
        rename_map[y_dim] = "y"
    if x_dim != "x":
        rename_map[x_dim] = "x"
    if rename_map:
        da = da.rename(rename_map)
    return da


def _select_lst_band(da: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    band_dim = None
    for cand in ["band", "bands", "channel", "channels"]:
        if cand in da.dims:
            band_dim = cand
            break
    if band_dim is None:
        return da

    band_names = None
    for cand in ["band_names", "bands", "band"]:
        if cand in ds:
            band_names = ds[cand].values
            break
    if band_names is None:
        return da

    band_names = [str(v).lower() for v in band_names]
    lst_indices = [i for i, name in enumerate(band_names) if "lst" in name]
    if not lst_indices:
        return da
    return da.isel({band_dim: lst_indices[0]})


def open_zarr_find_lst_var(zarr_path: str) -> Tuple[xr.DataArray, str]:
    ds_root = _try_open_group(zarr_path, None)
    datasets: List[Tuple[xr.Dataset, str]] = []
    if ds_root is not None:
        datasets.append((ds_root, ""))

    ds_landsat = _try_open_group(zarr_path, "products/landsat")
    if ds_landsat is not None:
        datasets.append((ds_landsat, "products/landsat/"))

    candidates: List[Tuple[xr.Dataset, str, str]] = []
    for ds, prefix in datasets:
        for var_name in ds.data_vars.keys():
            full_name = f"{prefix}{var_name}"
            candidates.append((ds, var_name, full_name))

    if not candidates:
        raise RuntimeError("No data variables found in Zarr store.")

    def _score(name: str) -> Tuple[int, int]:
        lower = name.lower()
        return (int("lst" in lower and "landsat" in lower), int("lst" in lower))

    candidates_sorted = sorted(candidates, key=lambda c: _score(c[2]), reverse=True)
    best_ds, best_var, best_full = candidates_sorted[0]
    if "lst" not in best_full.lower():
        available = [c[2] for c in candidates]
        print("Available variables:", available)
        raise RuntimeError("No variable name containing 'lst' was found.")

    da = best_ds[best_var]
    da = _select_lst_band(da, best_ds)
    da = _normalize_dims(da)

    return da, best_full


def monthly_composite(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.dims:
        return da

    time_values = pd.to_datetime(da["time"].values)
    da = da.assign_coords(time=time_values).sortby("time")
    periods = pd.PeriodIndex(time_values, freq="M")
    if len(periods) == len(periods.unique()):
        return da
    return da.resample(time="MS").median(dim="time", skipna=True)


def make_splits(da: xr.DataArray) -> SplitInfo:
    time_values = pd.to_datetime(da["time"].values)
    n_time = len(time_values)
    if n_time < 3:
        raise ValueError("Need at least 3 timesteps for train/val/test split.")

    train_end = int(math.floor(0.7 * n_time))
    val_end = int(math.floor(0.8 * n_time))

    split_dates = {
        "train_start": str(time_values[0]),
        "train_end": str(time_values[train_end - 1]),
        "val_start": str(time_values[train_end]),
        "val_end": str(time_values[val_end - 1]),
        "test_start": str(time_values[val_end]),
        "test_end": str(time_values[-1]),
    }

    return SplitInfo(
        train_idx=list(range(0, train_end)),
        val_idx=list(range(train_end, val_end)),
        test_idx=list(range(val_end, n_time)),
        split_dates=split_dates,
    )


def _compute_valid_fractions(da: xr.DataArray) -> np.ndarray:
    finite = xr.apply_ufunc(np.isfinite, da)
    valid_fraction = finite.mean(dim=("y", "x"), skipna=True).compute()
    return np.asarray(valid_fraction.values, dtype=np.float32)


def _filter_target_indices(
    target_indices: Sequence[int],
    valid_fractions: np.ndarray,
    k: int,
    min_valid_fraction: float,
) -> Tuple[List[int], int]:
    kept = []
    skipped = 0
    for t in target_indices:
        if t < k:
            skipped += 1
            continue
        window = valid_fractions[t - k : t + 1]
        if np.any(window < min_valid_fraction):
            skipped += 1
            continue
        kept.append(t)
    return kept, skipped


class PatchSequenceDataset(Dataset):
    def __init__(
        self,
        da: xr.DataArray,
        target_indices: Sequence[int],
        k: int,
        patch_size: int,
        patches_per_time: int,
        mode: str = "random",
        seed: int = 42,
        min_valid_fraction: float = 0.2,
        valid_fractions: Optional[np.ndarray] = None,
    ) -> None:
        self.da = da
        self.k = k
        self.patch_size = patch_size
        self.patches_per_time = patches_per_time
        self.mode = mode
        self.seed = seed

        self.da = _normalize_dims(self.da)
        if valid_fractions is None:
            valid_fractions = _compute_valid_fractions(self.da)
        self.valid_fractions = valid_fractions

        self.target_indices, skipped = _filter_target_indices(
            target_indices,
            self.valid_fractions,
            k,
            min_valid_fraction,
        )
        self.skipped = skipped

        self.height = int(self.da.sizes["y"])
        self.width = int(self.da.sizes["x"])
        if self.patch_size > self.height or self.patch_size > self.width:
            raise ValueError("Patch size larger than spatial dimensions.")

        self._fixed_samples: List[Tuple[int, int, int]] = []
        if self.mode != "random":
            rng = np.random.default_rng(self.seed)
            for t in self.target_indices:
                for _ in range(self.patches_per_time):
                    y0 = int(rng.integers(0, self.height - self.patch_size + 1))
                    x0 = int(rng.integers(0, self.width - self.patch_size + 1))
                    self._fixed_samples.append((t, y0, x0))

    def __len__(self) -> int:
        if self.mode == "random":
            return max(1, len(self.target_indices) * self.patches_per_time)
        return len(self._fixed_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == "random":
            rng = np.random.default_rng(self.seed + idx)
            t = self.target_indices[idx % len(self.target_indices)]
            y0 = int(rng.integers(0, self.height - self.patch_size + 1))
            x0 = int(rng.integers(0, self.width - self.patch_size + 1))
        else:
            t, y0, x0 = self._fixed_samples[idx]

        y1 = y0 + self.patch_size
        x1 = x0 + self.patch_size

        seq = self.da.isel(time=slice(t - self.k, t), y=slice(y0, y1), x=slice(x0, x1))
        target = self.da.isel(time=t, y=slice(y0, y1), x=slice(x0, x1))

        seq_np = np.asarray(seq.transpose("time", "y", "x").values, dtype=np.float32)
        target_np = np.asarray(target.values, dtype=np.float32)

        mask = np.isfinite(target_np).astype(np.float32)
        seq_np = np.nan_to_num(seq_np, nan=0.0, posinf=0.0, neginf=0.0)
        target_np = np.nan_to_num(target_np, nan=0.0, posinf=0.0, neginf=0.0)

        x = torch.from_numpy(seq_np[:, None, :, :])
        y = torch.from_numpy(target_np[None, :, :])
        m = torch.from_numpy(mask[None, :, :])
        return x, y, m


def masked_losses_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "huber",
    delta: float = 1.0,
    smooth_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = mask.float()
    valid = mask.sum().clamp(min=1.0)
    diff = pred - target

    if loss_type == "l1":
        base = diff.abs()
    else:
        abs_diff = diff.abs()
        base = torch.where(abs_diff < delta, 0.5 * diff**2, delta * (abs_diff - 0.5 * delta))

    loss = (base * mask).sum() / valid
    mae = (diff.abs() * mask).sum() / valid
    rmse = torch.sqrt((diff.pow(2) * mask).sum() / valid)

    if smooth_weight > 0:
        dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        smooth = (dy.abs().mean() + dx.abs().mean())
        loss = loss + smooth_weight * smooth

    return loss, mae, rmse, valid


def save_run_config(out_dir: str, config: Dict[str, object]) -> None:
    path = os.path.join(out_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def save_training_logs(out_dir: str, history: List[Dict[str, float]]) -> None:
    csv_path = os.path.join(out_dir, "metrics.csv")
    json_path = os.path.join(out_dir, "metrics.json")

    with open(csv_path, "w", encoding="utf-8") as f:
        headers = [
            "epoch",
            "train_loss",
            "val_loss",
            "train_mae",
            "val_mae",
            "train_rmse",
            "val_rmse",
            "lr",
            "time_sec",
        ]
        f.write(",".join(headers) + "\n")
        for row in history:
            f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def save_figures(
    figures_dir: str,
    history: List[Dict[str, float]],
    scatter_pred: np.ndarray,
    scatter_true: np.ndarray,
    maps: List[Tuple[str, np.ndarray, np.ndarray]],
) -> None:
    os.makedirs(figures_dir, exist_ok=True)

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_rmse = [row["train_rmse"] for row in history]
    val_rmse = [row["val_rmse"] for row in history]
    train_mae = [row["train_mae"] for row in history]
    val_mae = [row["val_mae"] for row in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_rmse, label="train")
    plt.plot(epochs, val_rmse, label="val")
    plt.xlabel("epoch")
    plt.ylabel("rmse")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "rmse_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_mae, label="train")
    plt.plot(epochs, val_mae, label="val")
    plt.xlabel("epoch")
    plt.ylabel("mae")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "mae_curve.png"), dpi=150)
    plt.close()

    if scatter_pred.size > 0:
        plt.figure(figsize=(5, 5))
        plt.scatter(scatter_true, scatter_pred, s=4, alpha=0.5)
        min_v = float(np.nanmin([scatter_true.min(), scatter_pred.min()]))
        max_v = float(np.nanmax([scatter_true.max(), scatter_pred.max()]))
        plt.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
        plt.xlabel("true")
        plt.ylabel("pred")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "test_scatter.png"), dpi=150)
        plt.close()

    for label, true_map, pred_map in maps:
        err = pred_map - true_map
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        im0 = axes[0].imshow(true_map, cmap="viridis")
        axes[0].set_title("true")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(pred_map, cmap="viridis")
        axes[1].set_title("pred")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        im2 = axes[2].imshow(err, cmap="coolwarm")
        axes[2].set_title("error")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(label)
        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"qual_{label}.png"), dpi=150)
        plt.close(fig)


def write_test_metrics(out_dir: str, metrics: Dict[str, float]) -> None:
    path = os.path.join(out_dir, "test_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
