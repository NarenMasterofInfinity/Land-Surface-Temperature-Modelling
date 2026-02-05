from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from arch_v1_utils import apply_alpha_forcing, masked_huber, normalize_batch_global, save_debug_images


@dataclass
class StageConfig:
    stage_a_epochs: int = 20
    stage_b_epochs: int = 90
    stage_a2_epochs: int = 40
    ramp_epochs: int = 30
    lambda_start: float = 0.02
    lambda_end: float = 0.10
    force_alpha_weak: bool = True
    consistency_b: float = 0.01
    consistency_c: float = 0.03

    def stage_at(self, epoch: int) -> Tuple[str, float, float]:
        if epoch < self.stage_a_epochs:
            return "A", 0.0, 0.0
        if epoch < self.stage_a_epochs + self.stage_b_epochs:
            ramp_start = self.stage_a_epochs
            ramp_end = self.stage_a_epochs + self.ramp_epochs - 1
            if epoch <= ramp_end:
                denom = max(1, ramp_end - ramp_start)
                t = (epoch - ramp_start) / denom
                lam = self.lambda_start + t * (self.lambda_end - self.lambda_start)
            else:
                lam = self.lambda_end
            return "B", lam, self.consistency_b
        if epoch < self.stage_a_epochs + self.stage_b_epochs + self.stage_a2_epochs:
            return "A2", 0.0, 0.0
        return "A2", 0.0, 0.0


@dataclass
class TrainConfig:
    batch_size: int = 4
    lr: float = 5e-4
    weight_decay: float = 5e-4
    grad_clip: float = 1.0
    beta: float = 0.5
    gamma: float = 0.5
    delta_huber: float = 1.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    stage: str,
    lam: float,
    cons_weight: float,
    cfg: TrainConfig,
    force_alpha_weak: bool,
    logger: Optional[object] = None,
    mu_x: Optional[torch.Tensor] = None,
    sigma_x: Optional[torch.Tensor] = None,
    mask_idx: Optional[List[int]] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    sse_ls = torch.tensor(0.0, device=device)
    cnt_ls = torch.tensor(0.0, device=device)
    sse_wk = torch.tensor(0.0, device=device)
    cnt_wk = torch.tensor(0.0, device=device)
    alpha_sum = torch.tensor(0.0, device=device)
    alpha_sumsq = torch.tensor(0.0, device=device)
    alpha_count = torch.tensor(0.0, device=device)
    alpha_p01 = torch.tensor(0.0, device=device)
    alpha_p09 = torch.tensor(0.0, device=device)

    skipped = 0
    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device)
        y_ls = batch["y_ls"].to(device)
        m_ls = batch["m_ls"].to(device)
        y_wk = batch["y_wk"].to(device)
        m_wk = batch["m_wk"].to(device)
        is_ls = batch["is_landsat"].to(device)
        doy = batch.get("doy")
        doy = doy.to(device) if doy is not None else None

        if mu_x is not None and sigma_x is not None:
            x = normalize_batch_global(x, mu_x, sigma_x, mask_idx or [])

        if stage == "A" and float((m_ls > 0.5).sum().item()) == 0.0:
            skipped += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=scaler.is_enabled()):
            out = model(x, doy=doy)
            alpha = apply_alpha_forcing(out["alpha"], is_ls, stage, force_alpha_weak)
            y_hat = out["base"] + alpha * out["r_strong"] + (1.0 - alpha) * out["r_weak"]
            y_strong = out["y_strong"]
            y_weak = out["y_weak"]

            loss = masked_huber(y_hat, y_ls, m_ls, delta=cfg.delta_huber)
            loss = loss + cfg.beta * masked_huber(y_strong, y_ls, m_ls, delta=cfg.delta_huber)

            if stage != "A":
                loss = loss + lam * (
                    masked_huber(y_hat, y_wk, m_wk, delta=cfg.delta_huber)
                    + cfg.gamma * masked_huber(y_weak, y_wk, m_wk, delta=cfg.delta_huber)
                )

            if cons_weight > 0.0:
                m_union = ((m_ls > 0.5) | (m_wk > 0.5)).float()
                loss = loss + cons_weight * masked_huber(out["r_strong"], out["r_weak"], m_union, delta=cfg.delta_huber)

        if not torch.isfinite(loss):
            if logger is not None:
                def _stat(t: torch.Tensor) -> str:
                    return f"finite={torch.isfinite(t).float().mean().item():.4f} min={t.nan_to_num().min().item():.3f} max={t.nan_to_num().max().item():.3f}"

                logger.warning(
                    "nonfinite loss at batch=%d stage=%s lam=%.3f cons=%.3f",
                    batch_idx,
                    stage,
                    lam,
                    cons_weight,
                )
                logger.warning("x: %s", _stat(x))
                logger.warning("y_ls: %s m_ls=%.4f", _stat(y_ls), m_ls.mean().item())
                logger.warning("y_wk: %s m_wk=%.4f", _stat(y_wk), m_wk.mean().item())
                logger.warning("base: %s", _stat(out["base"]))
                logger.warning("r_strong: %s r_weak: %s", _stat(out["r_strong"]), _stat(out["r_weak"]))
                logger.warning("alpha: %s y_hat: %s", _stat(alpha), _stat(y_hat))
            raise RuntimeError("Non-finite loss encountered. See logs for batch diagnostics.")

        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * x.shape[0]
        total_count += x.shape[0]

        ls_sel = is_ls.view(-1, 1, 1, 1).float()
        wk_sel = 1.0 - ls_sel
        diff_ls = (y_hat - y_ls) * m_ls * ls_sel
        diff_wk = (y_hat - y_wk) * m_wk * wk_sel
        sse_ls += (diff_ls * diff_ls).sum()
        cnt_ls += (m_ls * ls_sel).sum()
        sse_wk += (diff_wk * diff_wk).sum()
        cnt_wk += (m_wk * wk_sel).sum()
        alpha_sum += alpha.sum()
        alpha_sumsq += (alpha * alpha).sum()
        alpha_count += alpha.numel()
        alpha_p01 += (alpha < 0.1).float().sum()
        alpha_p09 += (alpha > 0.9).float().sum()

    train_loss = total_loss / max(1, total_count)
    train_rmse_ls = float(torch.sqrt(sse_ls / cnt_ls.clamp(min=1.0)).item()) if cnt_ls > 0 else float("nan")
    train_rmse_wk = float(torch.sqrt(sse_wk / cnt_wk.clamp(min=1.0)).item()) if cnt_wk > 0 else float("nan")
    sse_all = sse_ls + sse_wk
    cnt_all = cnt_ls + cnt_wk
    train_rmse_all = float(torch.sqrt(sse_all / cnt_all.clamp(min=1.0)).item()) if cnt_all > 0 else float("nan")
    alpha_mean = float((alpha_sum / alpha_count.clamp(min=1.0)).item()) if alpha_count > 0 else float("nan")
    alpha_var = (alpha_sumsq / alpha_count.clamp(min=1.0)) - (alpha_mean**2)
    alpha_std = float(torch.sqrt(torch.tensor(max(alpha_var, 0.0), device=device)).item()) if alpha_count > 0 else float("nan")
    alpha_p01 = float((alpha_p01 / alpha_count.clamp(min=1.0)).item()) if alpha_count > 0 else float("nan")
    alpha_p09 = float((alpha_p09 / alpha_count.clamp(min=1.0)).item()) if alpha_count > 0 else float("nan")
    return {
        "train_loss": train_loss,
        "train_rmse_all": train_rmse_all,
        "train_rmse_ls": train_rmse_ls,
        "train_rmse_wk": train_rmse_wk,
        "skipped_batches": skipped,
        "train_alpha_mean": alpha_mean,
        "train_alpha_std": alpha_std,
        "train_alpha_p01": alpha_p01,
        "train_alpha_p09": alpha_p09,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_fig_dir: Optional[str],
    epoch: int,
    stage: str,
    lam: float,
    cons_weight: float,
    cfg: TrainConfig,
    force_alpha_weak: bool,
    mu_x: Optional[torch.Tensor] = None,
    sigma_x: Optional[torch.Tensor] = None,
    mask_idx: Optional[List[int]] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    sse_ls = torch.tensor(0.0, device=device)
    cnt_ls = torch.tensor(0.0, device=device)
    sse_wk = torch.tensor(0.0, device=device)
    cnt_wk = torch.tensor(0.0, device=device)
    alpha_sum = torch.tensor(0.0, device=device)
    alpha_sumsq = torch.tensor(0.0, device=device)
    alpha_count = torch.tensor(0.0, device=device)
    alpha_p01 = torch.tensor(0.0, device=device)
    alpha_p09 = torch.tensor(0.0, device=device)
    saved = False
    for batch in loader:
        x = batch["x"].to(device)
        y_ls = batch["y_ls"].to(device)
        m_ls = batch["m_ls"].to(device)
        is_ls = batch["is_landsat"].to(device)
        y_wk = batch["y_wk"].to(device)
        m_wk = batch["m_wk"].to(device)
        doy = batch.get("doy")
        doy = doy.to(device) if doy is not None else None

        if mu_x is not None and sigma_x is not None:
            x = normalize_batch_global(x, mu_x, sigma_x, mask_idx or [])

        out = model(x, doy=doy)
        alpha = apply_alpha_forcing(out["alpha"], is_ls, stage, force_alpha_weak)
        y_hat = out["base"] + alpha * out["r_strong"] + (1.0 - alpha) * out["r_weak"]
        y_strong = out["y_strong"]
        y_weak = out["y_weak"]

        loss = masked_huber(y_hat, y_ls, m_ls, delta=cfg.delta_huber)
        loss = loss + cfg.beta * masked_huber(y_strong, y_ls, m_ls, delta=cfg.delta_huber)
        if stage != "A":
            loss = loss + lam * (
                masked_huber(y_hat, y_wk, m_wk, delta=cfg.delta_huber)
                + cfg.gamma * masked_huber(y_weak, y_wk, m_wk, delta=cfg.delta_huber)
            )
        if cons_weight > 0.0:
            m_union = ((m_ls > 0.5) | (m_wk > 0.5)).float()
            loss = loss + cons_weight * masked_huber(out["r_strong"], out["r_weak"], m_union, delta=cfg.delta_huber)
        total_loss += float(loss.item()) * x.shape[0]
        total_count += x.shape[0]

        ls_sel = is_ls.view(-1, 1, 1, 1).float()
        wk_sel = 1.0 - ls_sel
        diff_ls = (y_hat - y_ls) * m_ls * ls_sel
        diff_wk = (y_hat - y_wk) * m_wk * wk_sel
        sse_ls += (diff_ls * diff_ls).sum()
        cnt_ls += (m_ls * ls_sel).sum()
        sse_wk += (diff_wk * diff_wk).sum()
        cnt_wk += (m_wk * wk_sel).sum()
        alpha_sum += alpha.sum()
        alpha_sumsq += (alpha * alpha).sum()
        alpha_count += alpha.numel()
        alpha_p01 += (alpha < 0.1).float().sum()
        alpha_p09 += (alpha > 0.9).float().sum()

        if save_fig_dir is not None and not saved:
            save_debug_images(save_fig_dir, epoch, y_hat[:1], y_ls[:1], m_ls[:1])
            saved = True

    val_loss = total_loss / max(1, total_count)
    val_rmse_ls = float(torch.sqrt(sse_ls / cnt_ls.clamp(min=1.0)).item()) if cnt_ls > 0 else float("nan")
    val_rmse_wk = float(torch.sqrt(sse_wk / cnt_wk.clamp(min=1.0)).item()) if cnt_wk > 0 else float("nan")
    sse_all = sse_ls + sse_wk
    cnt_all = cnt_ls + cnt_wk
    val_rmse_all = float(torch.sqrt(sse_all / cnt_all.clamp(min=1.0)).item()) if cnt_all > 0 else float("nan")
    alpha_mean = float((alpha_sum / alpha_count.clamp(min=1.0)).item()) if alpha_count > 0 else float("nan")
    alpha_var = (alpha_sumsq / alpha_count.clamp(min=1.0)) - (alpha_mean**2)
    alpha_std = float(torch.sqrt(torch.tensor(max(alpha_var, 0.0), device=device)).item()) if alpha_count > 0 else float("nan")
    alpha_p01 = float((alpha_p01 / alpha_count.clamp(min=1.0)).item()) if alpha_count > 0 else float("nan")
    alpha_p09 = float((alpha_p09 / alpha_count.clamp(min=1.0)).item()) if alpha_count > 0 else float("nan")
    return {
        "val_loss": val_loss,
        "val_rmse_all": val_rmse_all,
        "val_rmse_ls": val_rmse_ls,
        "val_rmse_wk": val_rmse_wk,
        "val_alpha_mean": alpha_mean,
        "val_alpha_std": alpha_std,
        "val_alpha_p01": alpha_p01,
        "val_alpha_p09": alpha_p09,
    }


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_rmse: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_rmse": best_rmse,
        },
        path,
    )
