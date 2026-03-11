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


def _norm_layer(num_channels: int) -> nn.Module:
    groups = 8
    if num_channels < groups or num_channels % groups != 0:
        groups = 1
    return nn.GroupNorm(groups, num_channels)


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


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm = _norm_layer(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class BasicBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvGNAct(ch, ch, kernel_size=3)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = _norm_layer(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm2(self.conv2(out))
        out = out + x
        return self.act(out)


class HRModule(nn.Module):
    def __init__(self, num_branches: int, channels: Sequence[int], num_blocks: int):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            blocks = [BasicBlock(channels[i]) for _ in range(num_blocks)]
            self.branches.append(nn.Sequential(*blocks))

        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse = nn.ModuleList()
            for j in range(num_branches):
                if i == j:
                    fuse.append(nn.Identity())
                elif j > i:
                    fuse.append(
                        nn.Sequential(
                            nn.Conv2d(channels[j], channels[i], kernel_size=1, bias=False),
                            _norm_layer(channels[i]),
                        )
                    )
                else:
                    ops = []
                    in_ch = channels[j]
                    for k in range(i - j):
                        out_ch = channels[i] if k == i - j - 1 else in_ch
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                                _norm_layer(out_ch),
                                nn.ReLU(inplace=True),
                            )
                        )
                        in_ch = out_ch
                    fuse.append(nn.Sequential(*ops))
            self.fuse_layers.append(fuse)
        self.act = nn.ReLU(inplace=True)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        xs = [b(x) for b, x in zip(self.branches, xs)]
        fused = []
        for i in range(self.num_branches):
            y = None
            for j in range(self.num_branches):
                xj = xs[j]
                if i == j:
                    out = xj
                elif j > i:
                    scale = 2 ** (j - i)
                    out = self.fuse_layers[i][j](xj)
                    out = F.interpolate(out, scale_factor=scale, mode="bilinear", align_corners=False)
                else:
                    out = self.fuse_layers[i][j](xj)
                y = out if y is None else y + out
            fused.append(self.act(y))
        return fused


class ResidualHRNet(nn.Module):
    def __init__(self, in_ch: int, widths: Sequence[int] = (24, 32, 48, 64), blocks: Sequence[int] = (1, 1, 1, 1)):
        super().__init__()
        if len(widths) != 4 or len(blocks) != 4:
            raise ValueError("widths and blocks must each have 4 values.")

        self.stem = nn.Sequential(
            ConvGNAct(in_ch, widths[0], kernel_size=3),
            ConvGNAct(widths[0], widths[0], kernel_size=3),
        )
        self.stage1 = nn.Sequential(*[BasicBlock(widths[0]) for _ in range(blocks[0])])

        self.transition1 = nn.ModuleList(
            [
                nn.Identity(),
                nn.Sequential(
                    nn.Conv2d(widths[0], widths[1], kernel_size=3, stride=2, padding=1, bias=False),
                    _norm_layer(widths[1]),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.stage2 = HRModule(num_branches=2, channels=widths[:2], num_blocks=blocks[1])

        self.transition2 = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
                nn.Sequential(
                    nn.Conv2d(widths[1], widths[2], kernel_size=3, stride=2, padding=1, bias=False),
                    _norm_layer(widths[2]),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.stage3 = HRModule(num_branches=3, channels=widths[:3], num_blocks=blocks[2])

        self.transition3 = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
                nn.Sequential(
                    nn.Conv2d(widths[2], widths[3], kernel_size=3, stride=2, padding=1, bias=False),
                    _norm_layer(widths[3]),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.stage4 = HRModule(num_branches=4, channels=widths[:4], num_blocks=blocks[3])

        self.fuse_out = nn.Sequential(
            nn.Conv2d(sum(widths), widths[0], kernel_size=1, bias=False),
            _norm_layer(widths[0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        xs = [self.transition1[0](x), self.transition1[1](x)]
        xs = self.stage2(xs)
        xs = [
            self.transition2[0](xs[0]),
            self.transition2[1](xs[1]),
            self.transition2[2](xs[1]),
        ]
        xs = self.stage3(xs)
        xs = [
            self.transition3[0](xs[0]),
            self.transition3[1](xs[1]),
            self.transition3[2](xs[2]),
            self.transition3[3](xs[2]),
        ]
        xs = self.stage4(xs)

        high = xs[0]
        ups = [high]
        for i in range(1, len(xs)):
            scale = 2 ** i
            ups.append(F.interpolate(xs[i], scale_factor=scale, mode="bilinear", align_corners=False))
        out = torch.cat(ups, dim=1)
        return self.fuse_out(out)


class GateNet(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_ch, in_ch, kernel_size=3),
            ConvGNAct(in_ch, max(8, in_ch // 2), kernel_size=3),
            nn.Conv2d(max(8, in_ch // 2), 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class ResidualLSTNet30m(nn.Module):
    def __init__(
        self,
        *,
        s2_ch: int,
        s1_ch: int = 0,
        dem_ch: int = 1,
        lc_ch: int = 2,
        era5_ch: int = 4,
        widths: Sequence[int] = (24, 32, 48, 64),
        blocks: Sequence[int] = (1, 1, 1, 1),
        residual_clip_c: Optional[float] = 12.0,
        gate_use_baseline: bool = True,
    ) -> None:
        super().__init__()
        self.gate_use_baseline = bool(gate_use_baseline)
        in_ch = (2 * s2_ch) + (2 * s1_ch) + (2 * dem_ch) + (2 * lc_ch) + (2 * era5_ch) + 2

        self.hrnet = ResidualHRNet(in_ch=in_ch, widths=widths, blocks=blocks)
        feat_ch = int(widths[0])
        self.head_strong = nn.Conv2d(feat_ch, 1, kernel_size=1)
        self.head_weak = nn.Conv2d(feat_ch, 1, kernel_size=1)
        gate_in = feat_ch + (1 if self.gate_use_baseline else 0)
        self.gate = GateNet(gate_in)
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
        feat = self.hrnet(x)

        r_strong = self.head_strong(feat)
        r_weak = self.head_weak(feat)

        gate_in = feat
        if self.gate_use_baseline:
            gate_in = torch.cat([feat, base], dim=1)
        alpha = self.gate(gate_in)

        residual = alpha * r_strong + (1.0 - alpha) * r_weak
        if self.residual_clip_c is not None:
            residual = torch.tanh(residual) * float(self.residual_clip_c)
        yhat = base + residual
        return {
            "yhat": yhat,
            "residual": residual,
            "base": base,
            "alpha": alpha,
            "r_strong": r_strong,
            "r_weak": r_weak,
        }
