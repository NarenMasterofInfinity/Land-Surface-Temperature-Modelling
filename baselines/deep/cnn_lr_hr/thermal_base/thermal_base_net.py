from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _gn_groups(ch: int) -> int:
    if ch % 8 == 0:
        return 8
    if ch % 4 == 0:
        return 4
    if ch % 2 == 0:
        return 2
    return 1


class BasicBlock(nn.Module):
    def __init__(self, ch: int, groups: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = out + identity
        return self.act(out)


class ContextBlock(nn.Module):
    def __init__(self, ch: int, groups: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=2, dilation=2, bias=False)
        self.conv3 = nn.Conv2d(ch, ch, kernel_size=3, padding=4, dilation=4, bias=False)
        self.gn = nn.GroupNorm(groups, ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x) + self.conv2(x) + self.conv3(x)
        out = self.gn(out)
        return self.act(out)


class SpatialStem(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 16, groups: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU()
        self.block1 = BasicBlock(out_ch, groups=groups)
        self.block2 = BasicBlock(out_ch, groups=groups)
        self.context = ContextBlock(out_ch, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.context(x)


class TinyMamba(nn.Module):
    def __init__(self, d_model: int = 16, ffn_ratio: int = 2, use_ffn: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.use_ffn = use_ffn
        if use_ffn:
            hidden = d_model * ffn_ratio
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.SiLU(),
                nn.Linear(hidden, d_model),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x_norm = self.norm(x)
        x_t = x_norm.transpose(1, 2)  # (B, C, T)
        x_t = F.pad(x_t, (2, 0))  # causal padding for kernel=3
        x_t = self.dw_conv(x_t)
        x_t = x_t.transpose(1, 2)
        out = x + x_t
        if self.use_ffn:
            out = out + self.ffn(self.norm2(out))
        return out


class TemporalMixer(nn.Module):
    def __init__(self, d_model: int = 16, use_ffn: bool = True):
        super().__init__()
        self.mamba = TinyMamba(d_model=d_model, use_ffn=use_ffn)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, C, H, W)
        b, t, c, h, w = frames.shape
        seq = frames.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        seq = self.mamba(seq)
        last = seq[:, -1, :]
        return last.reshape(b, h, w, c).permute(0, 3, 1, 2)


class MetFiLM(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 32, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch * 2, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.gn(x)
        x = self.act(x)
        x = self.conv2(x)
        gamma, beta = torch.chunk(x, 2, dim=1)
        return gamma, beta


class BaseHead(nn.Module):
    def __init__(self, ch: int = 32, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv2d(ch, 16, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(4, 16)
        self.dropout = nn.Dropout2d(p=0.1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=1, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act(x)
        return self.conv3(x)


class UpsampleHead(nn.Module):
    def __init__(self, in_ch: int = 32, schedule: Sequence[int] = (32, 24, 16, 8)):
        super().__init__()
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(in_ch, schedule[0], kernel_size=3, padding=1, bias=False)
        self.proj_gn = nn.GroupNorm(_gn_groups(schedule[0]), schedule[0])

        blocks = []
        for c_in, c_out in zip(schedule[:-1], schedule[1:]):
            blocks.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            blocks.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False))
            blocks.append(nn.GroupNorm(_gn_groups(c_out), c_out))
            blocks.append(nn.SiLU())
            blocks.append(nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False))
            blocks.append(nn.GroupNorm(_gn_groups(c_out), c_out))
            blocks.append(nn.SiLU())
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv2d(schedule[-1], 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.proj_gn(x)
        x = self.act(x)
        x = self.blocks(x)
        return self.out(x)


class ThermalBaseNet(nn.Module):
    def __init__(self, era5_k: int = 4, static_ch: int = 3, use_ffn: bool = True):
        super().__init__()
        self.era5_k = era5_k
        self.static_ch = static_ch
        self.spatial_modis = SpatialStem(in_ch=2, out_ch=16, groups=4)
        self.spatial_viirs = SpatialStem(in_ch=2, out_ch=16, groups=4)
        self.temporal_modis = TemporalMixer(d_model=16, use_ffn=use_ffn)
        self.temporal_viirs = TemporalMixer(d_model=16, use_ffn=use_ffn)
        self.gate = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        self.film = MetFiLM(in_ch=era5_k + 2 + static_ch, out_ch=32, groups=8)
        self.head = BaseHead(ch=32, groups=8)

    def _standardize_frames(
        self,
        frames: torch.Tensor | Sequence[torch.Tensor],
        masks: Optional[torch.Tensor | Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(frames, (list, tuple)):
            frames = torch.stack(frames, dim=1)
        if frames.dim() == 4:
            frames = frames.unsqueeze(2)
        if frames.dim() != 5:
            raise ValueError("frames must be (B,3,1,H,W) or (B,3,H,W)")

        orig = frames
        finite = torch.isfinite(orig)
        frames = torch.nan_to_num(orig, nan=0.0, posinf=0.0, neginf=0.0)

        if masks is None:
            masks = finite
        else:
            if isinstance(masks, (list, tuple)):
                masks = torch.stack(masks, dim=1)
            if masks.dim() == 4:
                masks = masks.unsqueeze(2)
            masks = masks.to(dtype=frames.dtype)
            masks = masks > 0
            masks = masks & finite
        return frames, masks.to(dtype=frames.dtype)

    def _broadcast_aux(self, aux: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if aux.dim() == 2:
            aux = aux[:, :, None, None]
        if aux.dim() == 3:
            aux = aux[:, :, None, :]
        if aux.dim() == 4:
            if aux.shape[-2:] == (h, w):
                return aux
            return aux.expand(-1, -1, h, w)
        raise ValueError("aux must be (B,C), (B,C,1,1), or (B,C,H,W)")

    def forward(
        self,
        modis_frames: torch.Tensor | Sequence[torch.Tensor],
        viirs_frames: torch.Tensor | Sequence[torch.Tensor],
        era5: torch.Tensor,
        doy: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        modis_masks: Optional[torch.Tensor | Sequence[torch.Tensor]] = None,
        viirs_masks: Optional[torch.Tensor | Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        modis_frames, modis_masks = self._standardize_frames(modis_frames, modis_masks)
        viirs_frames, viirs_masks = self._standardize_frames(viirs_frames, viirs_masks)

        b, t, _, h, w = modis_frames.shape

        modis_feats = []
        viirs_feats = []
        for idx in range(t):
            modis_in = torch.cat([modis_frames[:, idx], modis_masks[:, idx]], dim=1)
            viirs_in = torch.cat([viirs_frames[:, idx], viirs_masks[:, idx]], dim=1)
            modis_feats.append(self.spatial_modis(modis_in))
            viirs_feats.append(self.spatial_viirs(viirs_in))
        modis_feats = torch.stack(modis_feats, dim=1)
        viirs_feats = torch.stack(viirs_feats, dim=1)

        f_modis = self.temporal_modis(modis_feats)
        f_viirs = self.temporal_viirs(viirs_feats)

        a_m = modis_masks.mean(dim=1)
        a_v = viirs_masks.mean(dim=1)
        g = torch.sigmoid(self.gate(torch.cat([a_m, a_v], dim=1)))

        modis_present = a_m > 0
        viirs_present = a_v > 0
        g = torch.where(modis_present & ~viirs_present, torch.ones_like(g), g)
        g = torch.where(viirs_present & ~modis_present, torch.zeros_like(g), g)

        both_missing = ~(modis_present | viirs_present)
        f_modis = torch.where(both_missing, torch.zeros_like(f_modis), f_modis)
        f_viirs = torch.where(both_missing, torch.zeros_like(f_viirs), f_viirs)

        f_th = torch.cat([g * f_modis, (1.0 - g) * f_viirs], dim=1)

        era5 = self._broadcast_aux(era5, h, w)
        doy = self._broadcast_aux(doy, h, w)
        if static is None:
            static = torch.zeros((b, self.static_ch, h, w), device=era5.device, dtype=era5.dtype)
        else:
            static = self._broadcast_aux(static, h, w)
        if era5.shape[1] != self.era5_k:
            raise ValueError(f"expected era5 with {self.era5_k} channels, got {era5.shape[1]}")
        if doy.shape[1] != 2:
            raise ValueError(f"expected doy with 2 channels, got {doy.shape[1]}")
        if static.shape[1] != self.static_ch:
            raise ValueError(f"expected static with {self.static_ch} channels, got {static.shape[1]}")

        film_in = torch.cat([era5, doy, static], dim=1)
        gamma, beta = self.film(film_in)
        f_th = gamma * f_th + beta

        return self.head(f_th)
