from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import matplotlib.pyplot as plt

ARCH_NAME = "arch_v1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_output_dirs(root: str) -> Dict[str, str]:
    out = {
        "metrics": os.path.join(root, "metrics", ARCH_NAME),
        "figures": os.path.join(root, "figures", ARCH_NAME),
        "logs": os.path.join(root, "logs", ARCH_NAME),
        "models": os.path.join(root, "models", ARCH_NAME),
    }
    for path in out.values():
        os.makedirs(path, exist_ok=True)
    return out


def setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger(ARCH_NAME)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def normalize_batch_global(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, mask_idx: List[int]) -> torch.Tensor:
    x0 = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu_t = mu[None, :, None, None].to(x0.device, dtype=x0.dtype)
    sigma_t = sigma[None, :, None, None].to(x0.device, dtype=x0.dtype)
    out = (x0 - mu_t) / sigma_t
    if mask_idx:
        out[:, mask_idx, :, :] = x0[:, mask_idx, :, :]
    return out


def compute_input_stats(dataset, n_samples: int, mask_idx: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    sum_x = None
    sum_sq = None
    count = None
    for i in range(n_samples):
        sample = dataset[i]
        x = sample["x"].float()
        finite = torch.isfinite(x)
        if sum_x is None:
            c = x.shape[0]
            sum_x = torch.zeros(c, dtype=torch.float64)
            sum_sq = torch.zeros(c, dtype=torch.float64)
            count = torch.zeros(c, dtype=torch.float64)
        for ch in range(x.shape[0]):
            if ch in mask_idx:
                continue
            vals = x[ch][finite[ch]]
            if vals.numel() == 0:
                continue
            sum_x[ch] += vals.double().sum()
            sum_sq[ch] += (vals.double() ** 2).sum()
            count[ch] += vals.numel()
    mu = torch.zeros_like(sum_x, dtype=torch.float32)
    sigma = torch.ones_like(sum_x, dtype=torch.float32)
    for ch in range(mu.numel()):
        if ch in mask_idx:
            continue
        if count[ch] > 0:
            mu[ch] = float(sum_x[ch] / count[ch])
            var = max(0.0, float(sum_sq[ch] / count[ch] - mu[ch] * mu[ch]))
            sigma[ch] = float(var**0.5) if var > 0 else 1.0
    return mu, sigma

def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    mask = mask.float()
    finite = torch.isfinite(pred) & torch.isfinite(target)
    mask = mask * finite.float()
    diff = pred - target
    abs_diff = diff.abs()
    huber = torch.where(abs_diff < delta, 0.5 * diff**2, delta * (abs_diff - 0.5 * delta))
    denom = mask.sum().clamp(min=1.0)
    return (huber * mask).sum() / denom


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    finite = torch.isfinite(pred) & torch.isfinite(target)
    mask = mask * finite.float()
    diff = pred - target
    denom = mask.sum().clamp(min=1.0)
    return torch.sqrt((diff * diff * mask).sum() / denom)


def apply_alpha_forcing(alpha_pred: torch.Tensor, is_landsat: torch.Tensor, stage: str, force_alpha_weak: bool) -> torch.Tensor:
    if stage in ("A", "A2"):
        return torch.ones_like(alpha_pred)
    if stage == "B":
        is_ls = is_landsat.view(-1, 1, 1, 1)
        if force_alpha_weak:
            return is_ls * torch.ones_like(alpha_pred)
        return torch.where(is_ls > 0.5, torch.ones_like(alpha_pred), alpha_pred)
    return alpha_pred


def save_metrics(metrics_path: str, history: List[Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        headers = [
            "epoch",
            "stage",
            "lambda",
            "train_loss",
            "val_loss",
            "train_rmse_all",
            "val_rmse_all",
            "train_rmse_ls",
            "val_rmse_ls",
            "train_rmse_wk",
            "val_rmse_wk",
            "train_alpha_mean",
            "train_alpha_std",
            "train_alpha_p01",
            "train_alpha_p09",
            "val_alpha_mean",
            "val_alpha_std",
            "val_alpha_p01",
            "val_alpha_p09",
            "skipped_batches",
            "lr",
            "time_sec",
        ]
        f.write(",".join(headers) + "\n")
        for row in history:
            f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")

    json_path = os.path.splitext(metrics_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def normalize_for_vis(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None:
        x = x.clone()
        x[mask <= 0.5] = 0.0
    vmin = x.amin(dim=(-2, -1), keepdim=True)
    vmax = x.amax(dim=(-2, -1), keepdim=True)
    scale = (vmax - vmin).clamp(min=1e-6)
    return (x - vmin) / scale


@torch.no_grad()
def save_debug_images(out_dir: str, epoch: int, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
    pred = normalize_for_vis(pred, mask)
    target = normalize_for_vis(target, mask)
    err = normalize_for_vis((pred - target).abs(), mask)
    p = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
    t = target.squeeze(0).squeeze(0).detach().cpu().numpy()
    e = err.squeeze(0).squeeze(0).detach().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(p, cmap="inferno")
    axes[0].set_title("pred")
    axes[1].imshow(t, cmap="inferno")
    axes[1].set_title("target")
    axes[2].imshow(e, cmap="magma")
    axes[2].set_title("err")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"epoch_{epoch:03d}_pred_target_err.png"), dpi=150)
    plt.close(fig)


def _plot_curve(
    epochs: List[int],
    train_vals: List[float],
    val_vals: List[float],
    title: str,
    ylabel: str,
    out_path: str,
    vlines: List[int],
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_vals, label="train")
    plt.plot(epochs, val_vals, label="val")
    for v in vlines:
        plt.axvline(v, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metrics(history: List[Dict[str, float]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    epochs = [int(row["epoch"]) + 1 for row in history]
    vlines = [20, 110]

    _plot_curve(
        epochs,
        [row.get("train_rmse_all", float("nan")) for row in history],
        [row.get("val_rmse_all", float("nan")) for row in history],
        "RMSE (Overall)",
        "rmse",
        os.path.join(out_dir, "rmse_overall.png"),
        vlines,
    )
    _plot_curve(
        epochs,
        [row.get("train_rmse_ls", float("nan")) for row in history],
        [row.get("val_rmse_ls", float("nan")) for row in history],
        "RMSE (Landsat-only)",
        "rmse",
        os.path.join(out_dir, "rmse_landsat.png"),
        vlines,
    )
    _plot_curve(
        epochs,
        [row.get("train_rmse_wk", float("nan")) for row in history],
        [row.get("val_rmse_wk", float("nan")) for row in history],
        "RMSE (Weak-only)",
        "rmse",
        os.path.join(out_dir, "rmse_weak.png"),
        vlines,
    )
    _plot_curve(
        epochs,
        [row.get("train_loss", float("nan")) for row in history],
        [row.get("val_loss", float("nan")) for row in history],
        "Loss",
        "loss",
        os.path.join(out_dir, "loss.png"),
        vlines,
    )
