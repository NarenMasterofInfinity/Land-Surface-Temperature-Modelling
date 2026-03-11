from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _groups(ch: int) -> int:
    if ch % 8 == 0:
        return 8
    if ch % 4 == 0:
        return 4
    if ch % 2 == 0:
        return 2
    return 1


def _with_mask(x: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(x).to(dtype=x.dtype)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.cat([x, finite], dim=1)


def _to_hw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if x.dim() == 2:
        x = x[:, :, None, None]
    if x.dim() == 3:
        x = x[:, :, None, :]
    if x.dim() != 4:
        raise ValueError("Expected tensor with shape (B,C), (B,C,1,1), or (B,C,H,W).")
    if x.shape[-2:] != (h, w):
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualLSTNet30m(nn.Module):
    def __init__(
        self,
        *,
        s2_ch: int,
        s1_ch: int = 0,
        dem_ch: int = 1,
        lc_ch: int = 2,
        era5_ch: int = 4,
        widths: Sequence[int] = (64, 96, 128, 192),
        residual_clip_c: Optional[float] = 12.0,
    ) -> None:
        super().__init__()
        if len(widths) != 4:
            raise ValueError("widths must contain exactly 4 values.")
        w1, w2, w3, w4 = [int(v) for v in widths]

        in_ch = (2 * s2_ch) + (2 * s1_ch) + (2 * dem_ch) + (2 * lc_ch) + (2 * era5_ch) + 2
        self.enc1 = ConvBlock(in_ch, w1)
        self.down1 = nn.Conv2d(w1, w2, 3, stride=2, padding=1, bias=False)
        self.enc2 = ConvBlock(w2, w2)
        self.down2 = nn.Conv2d(w2, w3, 3, stride=2, padding=1, bias=False)
        self.enc3 = ConvBlock(w3, w3)
        self.down3 = nn.Conv2d(w3, w4, 3, stride=2, padding=1, bias=False)
        self.enc4 = ConvBlock(w4, w4)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(w4 + w3, w3)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(w3 + w2, w2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(w2 + w1, w1)

        self.head = nn.Sequential(
            nn.Conv2d(w1, w1 // 2, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(max(2, w1 // 2)), max(2, w1 // 2)),
            nn.SiLU(),
            nn.Conv2d(max(2, w1 // 2), 1, 1),
        )
        self.residual_clip_c = residual_clip_c

    def forward(
        self,
        *,
        basenet_hr: torch.Tensor,
        s2: torch.Tensor,
        s1: Optional[torch.Tensor] = None,
        dem: Optional[torch.Tensor] = None,
        lc: Optional[torch.Tensor] = None,
        era5: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        b, _, h, w = s2.shape
        base = _to_hw(basenet_hr, h, w)

        parts = [_with_mask(s2), _with_mask(base)]
        if s1 is not None and s1.numel() > 0:
            parts.append(_with_mask(s1))
        if dem is not None and dem.numel() > 0:
            parts.append(_with_mask(_to_hw(dem, h, w)))
        if lc is not None and lc.numel() > 0:
            parts.append(_with_mask(_to_hw(lc, h, w)))
        if era5 is not None and era5.numel() > 0:
            parts.append(_with_mask(_to_hw(era5, h, w)))

        x = torch.cat(parts, dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        d3 = self.up3(e4)
        if d3.shape[-2:] != e3.shape[-2:]:
            e3 = F.interpolate(e3, size=d3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            e2 = F.interpolate(e2, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            e1 = F.interpolate(e1, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        residual = self.head(d1)
        if self.residual_clip_c is not None:
            residual = torch.tanh(residual) * float(self.residual_clip_c)
        yhat = base + residual
        return {"yhat": yhat, "residual": residual, "base": base}
