from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F

LST_MIN_C = 10.0
LST_MAX_C = 70.0
LANDSAT_FILL_VALUE = 149.0


@dataclass
class FilteringConfig:
    landsat_scale: float = 1.0
    landsat_offset: float = 0.0
    landsat_fill: float = LANDSAT_FILL_VALUE
    min_c: float = LST_MIN_C
    max_c: float = LST_MAX_C


def landsat_to_celsius(
    y: torch.Tensor,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    fill_value: float = LANDSAT_FILL_VALUE,
) -> torch.Tensor:
    y = y.float()
    if fill_value is not None:
        y = torch.where(y == fill_value, torch.tensor(float("nan"), device=y.device), y)
    if scale != 1.0 or offset != 0.0:
        y = y * scale + offset
    finite = torch.isfinite(y)
    if finite.any():
        vals = y[finite]
        if vals.numel() > 0 and vals.median() > 200:
            y = y - 273.15
    return y


def apply_range_mask(
    y: torch.Tensor,
    *,
    min_c: float = LST_MIN_C,
    max_c: float = LST_MAX_C,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(y) & (y >= min_c) & (y <= max_c)
    y = torch.where(valid, y, torch.zeros_like(y))
    return y, valid.float()


def extract_modis(modis_lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if modis_lr.shape[0] >= 6:
        lst = modis_lr[0].float()
        qc = modis_lr[4].float()
    elif modis_lr.shape[0] >= 2:
        lst = modis_lr[0].float()
        qc = modis_lr[1].float()
    else:
        lst = modis_lr[0].float()
        qc = torch.zeros_like(lst)
    valid_qc = qc == 1
    valid_lst = torch.isfinite(lst) & (lst != -9999.0) & (lst > 0)
    mask = valid_qc & valid_lst
    lst = torch.where(mask, lst, torch.tensor(float("nan"), device=lst.device))
    finite = torch.isfinite(lst)
    if finite.any():
        vals = lst[finite]
        if vals.numel() > 0 and vals.median() > 200:
            lst = lst - 273.15
    return lst, mask.float()


def extract_viirs(viirs_lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if viirs_lr.shape[0] >= 4:
        lst = viirs_lr[0].float()
        qc = viirs_lr[2].float()
    elif viirs_lr.shape[0] >= 2:
        lst = viirs_lr[0].float()
        qc = viirs_lr[1].float()
    else:
        lst = viirs_lr[0].float()
        qc = torch.zeros_like(lst)
    valid_qc = qc <= 1
    valid_lst = torch.isfinite(lst) & (lst != -9999.0) & (lst >= 273.0)
    mask = valid_qc & valid_lst
    lst = torch.where(mask, lst, torch.tensor(float("nan"), device=lst.device))
    lst = lst - 273.15
    return lst, mask.float()


def upsample_ignore_nan(x: torch.Tensor, target_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError("upsample_ignore_nan expects a 2D tensor")
    mask = torch.isfinite(x).float()
    x = torch.nan_to_num(x, nan=0.0)
    xw = x * mask
    xw = xw.unsqueeze(0).unsqueeze(0)
    mask = mask.unsqueeze(0).unsqueeze(0)
    xw_up = F.interpolate(xw, size=target_hw, mode="bilinear", align_corners=False)
    mask_up = F.interpolate(mask, size=target_hw, mode="bilinear", align_corners=False)
    out = xw_up / mask_up.clamp(min=1e-6)
    return out.squeeze(0).squeeze(0), mask_up.squeeze(0).squeeze(0)


def build_weak_label_from_modis_viirs(
    modis_lr: Optional[torch.Tensor],
    viirs_lr: Optional[torch.Tensor],
    target_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    weak_vals: List[torch.Tensor] = []
    weak_wts: List[torch.Tensor] = []

    if modis_lr is not None:
        modis_lst, _ = extract_modis(modis_lr)
        modis_up, modis_w = upsample_ignore_nan(modis_lst, target_hw)
        weak_vals.append(modis_up)
        weak_wts.append(modis_w)

    if viirs_lr is not None:
        viirs_lst, _ = extract_viirs(viirs_lr)
        viirs_up, viirs_w = upsample_ignore_nan(viirs_lst, target_hw)
        weak_vals.append(viirs_up)
        weak_wts.append(viirs_w)

    if not weak_vals:
        y = torch.zeros(target_hw, dtype=torch.float32)
        m = torch.zeros_like(y)
        return y, m

    vals = torch.stack(weak_vals, dim=0)
    wts = torch.stack(weak_wts, dim=0)
    denom = wts.sum(dim=0).clamp(min=1e-6)
    y = (vals * wts).sum(dim=0) / denom
    y, m = apply_range_mask(y)
    return y, m
