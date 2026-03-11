from __future__ import annotations

import torch
import torch.nn.functional as F


def _weighted_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(w.sum(), min=1.0e-6)
    return (x * w).sum() / denom


def weighted_huber_with_delta_penalty(
    *,
    yhat: torch.Tensor,
    y: torch.Tensor,
    sample_weight: torch.Tensor,
    usable_mask: torch.Tensor,
    delta_term: torch.Tensor,
    huber_delta: float = 1.0,
    lambda_delta: float = 1e-3,
    lambda_bias: float = 0.0,
) -> torch.Tensor:
    loss = F.huber_loss(yhat, y, delta=huber_delta, reduction="none")
    w = sample_weight * usable_mask
    base = _weighted_mean(loss, w)
    penalty = lambda_delta * torch.mean(delta_term * delta_term)
    resid = yhat - y
    bias = _weighted_mean(resid, w)
    bias_penalty = lambda_bias * (bias * bias)
    return base + penalty + bias_penalty


def weighted_gaussian_nll_with_delta_penalty(
    *,
    yhat: torch.Tensor,
    y: torch.Tensor,
    log_sigma: torch.Tensor,
    sample_weight: torch.Tensor,
    usable_mask: torch.Tensor,
    delta_term: torch.Tensor,
    lambda_delta: float = 1e-3,
    clamp_log_sigma: tuple[float, float] = (-3.0, 2.0),
    lambda_bias: float = 0.0,
) -> torch.Tensor:
    ls = torch.clamp(log_sigma, min=float(clamp_log_sigma[0]), max=float(clamp_log_sigma[1]))
    inv_var = torch.exp(-2.0 * ls)
    nll = 0.5 * (((y - yhat) * (y - yhat) * inv_var) + (2.0 * ls))
    w = sample_weight * usable_mask
    base = _weighted_mean(nll, w)
    penalty = lambda_delta * torch.mean(delta_term * delta_term)
    resid = yhat - y
    bias = _weighted_mean(resid, w)
    bias_penalty = lambda_bias * (bias * bias)
    return base + penalty + bias_penalty


def weighted_huber(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    usable_mask: torch.Tensor,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    h = F.huber_loss(pred, target, delta=huber_delta, reduction="none")
    return _weighted_mean(h, sample_weight * usable_mask)
