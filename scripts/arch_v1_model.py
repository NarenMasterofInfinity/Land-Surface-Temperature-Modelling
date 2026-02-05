from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import nn


def _norm_layer(num_channels: int) -> nn.Module:
    groups = 8
    if num_channels < groups or num_channels % groups != 0:
        groups = 1
    return nn.GroupNorm(groups, num_channels)


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


class BaseNet(nn.Module):
    """
    Meteorology-driven baseline using low-capacity, dilated convs for smooth fields.
    """

    def __init__(self, in_ch: int, width: int = 16, use_doy: bool = False):
        super().__init__()
        self.use_doy = use_doy
        total_in = in_ch + (1 if use_doy else 0)
        self.stem = ConvGNAct(total_in, width, kernel_size=5)
        self.d1 = ConvGNAct(width, width, kernel_size=3, dilation=2)
        self.d2 = ConvGNAct(width, width, kernel_size=3, dilation=4)
        self.head = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, x_era5: torch.Tensor, doy: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_doy:
            if doy is None:
                raise ValueError("doy must be provided when use_doy=True")
            if doy.ndim <= 2:
                doy = doy.view(-1, 1, 1, 1)
            doy_map = doy.expand(-1, 1, x_era5.shape[-2], x_era5.shape[-1])
            x = torch.cat([x_era5, doy_map], dim=1)
        else:
            x = x_era5
        x = self.stem(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.head(x)


class GroupEmbedding(nn.Module):
    """
    Group-wise embeddings that allocate capacity by source importance.
    """

    def __init__(self, group_defs: Sequence[tuple[str, List[int], int]]):
        super().__init__()
        self.group_defs = list(group_defs)
        self.embeds = nn.ModuleDict()
        for name, idxs, out_ch in self.group_defs:
            if not idxs:
                continue
            self.embeds[name] = nn.Sequential(
                nn.Conv2d(len(idxs), out_ch, kernel_size=1, bias=False),
                _norm_layer(out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        for name, idxs, _ in self.group_defs:
            if not idxs:
                continue
            group_x = x.index_select(1, torch.as_tensor(idxs, device=x.device))
            feats.append(self.embeds[name](group_x))
        if not feats:
            raise ValueError("No groups provided to GroupEmbedding.")
        return torch.cat(feats, dim=1)


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
                    ops: List[nn.Module] = []
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

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        xs = [b(x) for b, x in zip(self.branches, xs)]
        fused: List[torch.Tensor] = []
        for i in range(self.num_branches):
            y = None
            for j in range(self.num_branches):
                xj = xs[j]
                if i == j:
                    out = xj
                elif j > i:
                    scale = 2 ** (j - i)
                    out = self.fuse_layers[i][j](xj)
                    out = torch.nn.functional.interpolate(out, scale_factor=scale, mode="bilinear", align_corners=False)
                else:
                    out = self.fuse_layers[i][j](xj)
                y = out if y is None else y + out
            fused.append(self.act(y))
        return fused


class ResidualHRNet(nn.Module):
    def __init__(self, in_ch: int, widths: Sequence[int] = (24, 32, 48, 64), blocks: Sequence[int] = (1, 1, 1, 1)):
        super().__init__()
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
            ups.append(torch.nn.functional.interpolate(xs[i], scale_factor=scale, mode="bilinear", align_corners=False))
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


class LSTFusionModel(nn.Module):
    def __init__(
        self,
        *,
        era5_idx: List[int],
        s2_idx: List[int],
        s1_idx: List[int],
        dem_idx: List[int],
        world_idx: List[int],
        dyn_idx: List[int],
        use_doy: bool = False,
        gate_use_baseline: bool = True,
    ):
        super().__init__()
        self.era5_idx = era5_idx
        self.use_doy = use_doy
        self.gate_use_baseline = gate_use_baseline

        self.base = BaseNet(in_ch=len(era5_idx), use_doy=use_doy)

        group_defs = [
            ("s2", s2_idx, 24),
            ("s1", s1_idx, 6),
            ("dem", dem_idx, 4),
            ("world", world_idx, 6),
            ("dyn", dyn_idx, 6),
        ]
        self.group_embed = GroupEmbedding(group_defs)
        embed_ch = sum(out_ch for _, _, out_ch in group_defs)
        residual_in_ch = embed_ch + (1 if gate_use_baseline else 0)

        self.hrnet = ResidualHRNet(in_ch=residual_in_ch)
        self.head_strong = nn.Conv2d(24, 1, kernel_size=1)
        self.head_weak = nn.Conv2d(24, 1, kernel_size=1)

        gate_in = 24 + (1 if gate_use_baseline else 0)
        self.gate = GateNet(gate_in)

    def forward(self, x: torch.Tensor, doy: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        era5 = x.index_select(1, torch.as_tensor(self.era5_idx, device=x.device))
        base = self.base(era5, doy=doy)

        high = self.group_embed(x)
        if self.gate_use_baseline:
            residual_in = torch.cat([high, base], dim=1)
        else:
            residual_in = high
        feat = self.hrnet(residual_in)

        r_strong = self.head_strong(feat)
        r_weak = self.head_weak(feat)

        gate_in = feat
        if self.gate_use_baseline:
            gate_in = torch.cat([feat, base], dim=1)
        alpha = self.gate(gate_in)

        y_hat = base + alpha * r_strong + (1.0 - alpha) * r_weak
        return {
            "base": base,
            "r_strong": r_strong,
            "r_weak": r_weak,
            "alpha": alpha,
            "y_hat": y_hat,
            "y_strong": base + r_strong,
            "y_weak": base + r_weak,
        }


def build_default_channel_indices() -> Dict[str, List[int]]:
    return {
        "era5": list(range(0, 8)),
        "s2": list(range(8, 25)),
        "s1": list(range(25, 31)),
        "dem": list(range(31, 34)),
        "world": [34],
        "dyn": [35],
    }
