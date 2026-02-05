from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils_data import (
    PatchSequenceDataset,
    ensure_dir,
    masked_losses_metrics,
    make_splits,
    monthly_composite,
    open_zarr_find_lst_var,
    save_figures,
    save_run_config,
    save_training_logs,
    set_seed,
    write_test_metrics,
)


class Conv3DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN3D(nn.Module):
    def __init__(self, base_ch: int = 16) -> None:
        super().__init__()
        self.enc1 = Conv3DBlock(1, base_ch)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.enc2 = Conv3DBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.bottleneck = Conv3DBlock(base_ch * 2, base_ch * 4)
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = Conv3DBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = Conv3DBlock(base_ch * 2, base_ch)
        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.pool_time = nn.AdaptiveAvgPool3d((1, None, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        pooled = self.pool_time(d1).squeeze(2)
        return self.head(pooled)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    loss_type: str,
    smooth_weight: float,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_count = 0.0

    for x, y, m in loader:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred = model(x)
            loss, mae, rmse, count = masked_losses_metrics(
                pred, y, m, loss_type=loss_type, smooth_weight=smooth_weight
            )

        if is_train:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * count.item()
        total_mae += mae.item() * count.item()
        total_rmse += rmse.item() * count.item()
        total_count += count.item()

    denom = max(total_count, 1.0)
    return {
        "loss": total_loss / denom,
        "mae": total_mae / denom,
        "rmse": total_rmse / denom,
    }


def predict_full_month(
    model: nn.Module,
    da,
    target_idx: int,
    k: int,
    patch_size: int,
    device: torch.device,
    batch_size: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    da = da.transpose("time", "y", "x")
    height = int(da.sizes["y"])
    width = int(da.sizes["x"])
    y_positions = list(range(0, height, patch_size))
    x_positions = list(range(0, width, patch_size))
    coords = set()
    for y in y_positions:
        y0 = min(y, height - patch_size)
        for x in x_positions:
            x0 = min(x, width - patch_size)
            coords.add((y0, x0))
    coords_list = list(coords)

    pred_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(coords_list), batch_size):
            batch_coords = coords_list[start : start + batch_size]
            batch_seq = []
            for y0, x0 in batch_coords:
                seq = da.isel(
                    time=slice(target_idx - k, target_idx),
                    y=slice(y0, y0 + patch_size),
                    x=slice(x0, x0 + patch_size),
                )
                seq_np = np.asarray(seq.values, dtype=np.float32)
                seq_np = np.nan_to_num(seq_np, nan=0.0, posinf=0.0, neginf=0.0)
                batch_seq.append(seq_np)
            batch_np = np.stack(batch_seq, axis=0)
            batch_tensor = torch.from_numpy(batch_np[:, :, None, :, :]).to(device)
            preds = model(batch_tensor).cpu().numpy()
            for (y0, x0), pred in zip(batch_coords, preds):
                pred_map[y0 : y0 + patch_size, x0 : x0 + patch_size] += pred[0]
                count_map[y0 : y0 + patch_size, x0 : x0 + patch_size] += 1.0

    count_map[count_map == 0] = 1.0
    pred_map = pred_map / count_map

    true_map = np.asarray(da.isel(time=target_idx).values, dtype=np.float32)
    true_map = np.nan_to_num(true_map, nan=0.0, posinf=0.0, neginf=0.0)
    return true_map, pred_map


def main() -> None:
    parser = argparse.ArgumentParser("3D CNN baseline")
    parser.add_argument("--zarr", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--patches-per-time", type=int, default=8)
    parser.add_argument("--val-patches-per-time", type=int, default=8)
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "l1"])
    parser.add_argument("--smooth", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    da, var_name = open_zarr_find_lst_var(args.zarr)
    da = monthly_composite(da)
    splits = make_splits(da)

    train_targets = [i for i in splits.train_idx if i >= args.K]
    val_targets = [i for i in splits.val_idx if i >= args.K]
    test_targets = [i for i in splits.test_idx if i >= args.K]

    model_name = "cnn3d"
    out_dir = os.path.join(args.out, model_name)
    figures_dir = os.path.join(out_dir, "figures")
    ensure_dir(out_dir)
    ensure_dir(figures_dir)

    train_ds = PatchSequenceDataset(
        da,
        train_targets,
        k=args.K,
        patch_size=args.patch,
        patches_per_time=args.patches_per_time,
        mode="random",
        seed=args.seed,
    )
    val_ds = PatchSequenceDataset(
        da,
        val_targets,
        k=args.K,
        patch_size=args.patch,
        patches_per_time=args.val_patches_per_time,
        mode="fixed",
        seed=args.seed + 1,
        valid_fractions=train_ds.valid_fractions,
    )
    test_ds = PatchSequenceDataset(
        da,
        test_targets,
        k=args.K,
        patch_size=args.patch,
        patches_per_time=args.val_patches_per_time,
        mode="fixed",
        seed=args.seed + 2,
        valid_fractions=train_ds.valid_fractions,
    )

    print(f"Skipped train samples: {train_ds.skipped}")
    print(f"Skipped val samples: {val_ds.skipped}")
    print(f"Skipped test samples: {test_ds.skipped}")

    if len(train_ds.target_indices) == 0:
        raise RuntimeError("No valid training samples after filtering.")
    if len(val_ds.target_indices) == 0 or len(test_ds.target_indices) == 0:
        raise RuntimeError("No valid validation/test samples after filtering.")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = CNN3D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_rmse = float("inf")
    best_epoch = -1
    patience_left = args.patience
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_stats = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            loss_type=args.loss,
            smooth_weight=args.smooth,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            device,
            None,
            None,
            loss_type=args.loss,
            smooth_weight=args.smooth,
        )
        elapsed = time.time() - start

        lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["loss"],
                "train_mae": train_stats["mae"],
                "val_mae": val_stats["mae"],
                "train_rmse": train_stats["rmse"],
                "val_rmse": val_stats["rmse"],
                "lr": lr,
                "time_sec": elapsed,
            }
        )

        if val_stats["rmse"] < best_rmse:
            best_rmse = val_stats["rmse"]
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_rmse": best_rmse,
                },
                os.path.join(out_dir, "best_checkpoint.pt"),
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    save_training_logs(out_dir, history)
    save_run_config(
        out_dir,
        {
            "zarr_path": args.zarr,
            "var_name": var_name,
            "K": args.K,
            "patch_size": args.patch,
            "batch_size": args.batch,
            "epochs": args.epochs,
            "split_dates": splits.split_dates,
            "seed": args.seed,
            "device": str(device),
            "best_epoch": best_epoch,
        },
    )

    checkpoint = torch.load(os.path.join(out_dir, "best_checkpoint.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_stats = run_epoch(
        model,
        test_loader,
        device,
        None,
        None,
        loss_type=args.loss,
        smooth_weight=args.smooth,
    )

    scatter_true: List[np.ndarray] = []
    scatter_pred: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y, m in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            y_np = y.numpy()
            m_np = m.numpy().astype(bool)
            pred_flat = pred[m_np]
            true_flat = y_np[m_np]
            if pred_flat.size > 0:
                scatter_pred.append(pred_flat)
                scatter_true.append(true_flat)

    if scatter_pred:
        scatter_pred_np = np.concatenate(scatter_pred)
        scatter_true_np = np.concatenate(scatter_true)
        if scatter_pred_np.size > 20000:
            idx = np.random.choice(scatter_pred_np.size, 20000, replace=False)
            scatter_pred_np = scatter_pred_np[idx]
            scatter_true_np = scatter_true_np[idx]
    else:
        scatter_pred_np = np.array([])
        scatter_true_np = np.array([])

    maps = []
    for t in test_targets[:3]:
        label = str(da["time"].values[t])[:10]
        true_map, pred_map = predict_full_month(
            model, da, t, args.K, args.patch, device
        )
        maps.append((label, true_map, pred_map))

    save_figures(figures_dir, history, scatter_pred_np, scatter_true_np, maps)
    write_test_metrics(out_dir, {"mae": test_stats["mae"], "rmse": test_stats["rmse"]})


if __name__ == "__main__":
    main()
