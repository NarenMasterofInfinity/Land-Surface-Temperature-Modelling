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


def _nan_to_num_with_mask(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(x)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, finite.to(dtype=x.dtype)


def _broadcast_to_hw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if x.dim() == 2:
        x = x[:, :, None, None]
    if x.dim() == 3:
        x = x[:, :, None, :]
    if x.dim() != 4:
        raise ValueError("conditioning input must be (B,C), (B,C,1,1), or (B,C,H,W)")
    if x.shape[-2:] != (h, w):
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class BasicBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_gn_groups(ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(_gn_groups(ch), ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + identity
        return self.act(out)


class AvailabilityGate(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.gate = nn.Conv2d(ch, ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(mask))
        return x * gate


class AdapterS2(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 32):
        super().__init__()
        self.stem = ConvBlock(in_ch * 2, out_ch)
        self.block1 = BasicBlock(out_ch)
        self.block2 = BasicBlock(out_ch)
        self.block3 = BasicBlock(out_ch)
        self.gate = AvailabilityGate(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, mask = _nan_to_num_with_mask(x)
        feats = self.stem(torch.cat([x, mask], dim=1))
        feats = self.block1(self.block2(self.block3(feats)))
        mask_feat = mask.mean(dim=1, keepdim=True).expand_as(feats)
        return self.gate(feats, mask_feat)


class AdapterDEM(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 12):
        super().__init__()
        self.block1 = ConvBlock(in_ch * 2, out_ch)
        self.block2 = ConvBlock(out_ch, out_ch)
        self.gate = AvailabilityGate(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, mask = _nan_to_num_with_mask(x)
        feats = self.block1(torch.cat([x, mask], dim=1))
        feats = self.block2(feats)
        mask_feat = mask.mean(dim=1, keepdim=True).expand_as(feats)
        return self.gate(feats, mask_feat)


class AdapterS1(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 12):
        super().__init__()
        self.block1 = ConvBlock(in_ch * 2, out_ch)
        self.block2 = ConvBlock(out_ch, out_ch)
        self.gate = AvailabilityGate(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, mask = _nan_to_num_with_mask(x)
        feats = self.block1(torch.cat([x, mask], dim=1))
        feats = self.block2(feats)
        mask_feat = mask.mean(dim=1, keepdim=True).expand_as(feats)
        return self.gate(feats, mask_feat)


class AdapterLC(nn.Module):
    def __init__(self, num_classes: Optional[int], out_ch: int = 16, one_hot: bool = False):
        super().__init__()
        self.one_hot = one_hot
        self.num_classes = num_classes
        if one_hot:
            if num_classes is None:
                raise ValueError("num_classes required for one-hot landcover")
            self.embed = None
            self.proj = nn.Conv2d(num_classes, out_ch, kernel_size=1, bias=False)
        else:
            if num_classes is None:
                raise ValueError("num_classes required for categorical landcover")
            self.embed = nn.Embedding(num_classes, 8)
            self.proj = nn.Conv2d(8, out_ch, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.one_hot:
            feats = self.proj(x)
        else:
            if x.dim() != 4 or x.shape[1] != 1:
                raise ValueError("categorical landcover must be (B,1,H,W)")
            x = x.squeeze(1).long().clamp(min=0, max=self.num_classes - 1)
            emb = self.embed(x).permute(0, 3, 1, 2)
            feats = self.proj(emb)
        return self.act(self.gn(feats))


class FiLM(nn.Module):
    def __init__(self, in_ch: int, feat_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, feat_ch, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(_gn_groups(feat_ch), feat_ch)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(feat_ch, feat_ch * 2, kernel_size=1, bias=True)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.act(self.gn(self.conv1(cond)))
        gamma, beta = torch.chunk(self.conv2(x), 2, dim=1)
        return gamma, beta


class UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class ResidualUNet(nn.Module):
    def __init__(self, in_ch: int, channels: Sequence[int], use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        c1, c2, c3, c4 = channels
        self.enc1 = UNetBlock(in_ch, c1)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc2 = UNetBlock(c2, c2)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc3 = UNetBlock(c3, c3)
        self.down3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc4 = UNetBlock(c4, c4)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = UNetBlock(c4 + c3, c3)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = UNetBlock(c3 + c2, c2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = UNetBlock(c2 + c1, c1)

    def _cp(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.use_checkpoint:
            return module(x)
        from torch.utils.checkpoint import checkpoint
        return checkpoint(module, x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self._cp(self.enc1, x)
        e2 = self._cp(self.enc2, self.down1(e1))
        e3 = self._cp(self.enc3, self.down2(e2))
        e4 = self._cp(self.enc4, self.down3(e3))
        up3 = self.up3(e4)
        if up3.shape[-2:] != e3.shape[-2:]:
            e3 = F.interpolate(e3, size=up3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self._cp(self.dec3, torch.cat([up3, e3], dim=1))
        up2 = self.up2(d3)
        if up2.shape[-2:] != e2.shape[-2:]:
            e2 = F.interpolate(e2, size=up2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self._cp(self.dec2, torch.cat([up2, e2], dim=1))
        up1 = self.up1(d2)
        if up1.shape[-2:] != e1.shape[-2:]:
            e1 = F.interpolate(e1, size=up1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self._cp(self.dec1, torch.cat([up1, e1], dim=1))
        return d1, d2, d3, e4


class ResidualNet(nn.Module):
    def __init__(
        self,
        s2_ch: int,
        s1_ch: Optional[int] = None,
        dem_ch: Optional[int] = None,
        lc_num_classes: Optional[int] = None,
        lc_one_hot: bool = False,
        era5_ch: int = 0,
        base_ch: int = 1,
        unet_channels: Sequence[int] = (64, 96, 128, 192),
        residual_tanh: Optional[float] = 15.0,
        s2_out: int = 32,
        s1_out: int = 12,
        dem_out: int = 12,
        head_dropout: float = 0.1,
        head_ch: int = 32,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.s2_adapter = AdapterS2(s2_ch, out_ch=s2_out)
        self.s1_adapter = AdapterS1(s1_ch, out_ch=s1_out) if s1_ch else None
        self.dem_adapter = AdapterDEM(dem_ch or 1, out_ch=dem_out) if dem_ch else None
        self.lc_adapter = AdapterLC(lc_num_classes, out_ch=16, one_hot=lc_one_hot) if lc_num_classes else None

        in_ch = s2_out
        if self.lc_adapter:
            in_ch += 16
        if self.dem_adapter:
            in_ch += dem_out
        if self.s1_adapter:
            in_ch += s1_out

        self.trunk = ResidualUNet(in_ch, unet_channels, use_checkpoint=use_checkpoint)
        self.film1 = FiLM(base_ch + era5_ch, unet_channels[0])
        self.film2 = FiLM(base_ch + era5_ch, unet_channels[1])
        self.film3 = FiLM(base_ch + era5_ch, unet_channels[2])
        self.film4 = FiLM(base_ch + era5_ch, unet_channels[3])

        self.head = nn.Sequential(
            nn.Conv2d(unet_channels[0], head_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(head_ch), head_ch),
            nn.SiLU(),
            nn.Dropout2d(p=head_dropout),
            nn.Conv2d(head_ch, 1, kernel_size=1, bias=True),
        )
        self.residual_tanh = residual_tanh

    def _film_apply(self, feats: torch.Tensor, film: FiLM, base: torch.Tensor, era5: Optional[torch.Tensor]) -> torch.Tensor:
        b, _, h, w = feats.shape
        cond = [_broadcast_to_hw(base, h, w)]
        if era5 is not None and era5.numel() > 0:
            cond.append(_broadcast_to_hw(era5, h, w))
        cond = torch.cat(cond, dim=1)
        gamma, beta = film(cond)
        return gamma * feats + beta

    def forward(
        self,
        base_b: torch.Tensor,
        s2: torch.Tensor,
        s1: Optional[torch.Tensor] = None,
        dem: Optional[torch.Tensor] = None,
        lc: Optional[torch.Tensor] = None,
        era5: Optional[torch.Tensor] = None,
        return_residual: bool = False,
    ) -> torch.Tensor:
        b, _, h, w = s2.shape
        base_b = _broadcast_to_hw(base_b, h, w)

        feats = [self.s2_adapter(s2)]
        if self.lc_adapter:
            if lc is None:
                lc = torch.zeros((b, 1, h, w), device=s2.device, dtype=s2.dtype)
            feats.append(self.lc_adapter(lc))
        if self.dem_adapter:
            if dem is None:
                dem = torch.zeros((b, 1, h, w), device=s2.device, dtype=s2.dtype)
            feats.append(self.dem_adapter(dem))
        if self.s1_adapter:
            if s1 is None:
                s1 = torch.zeros((b, self.s1_adapter.block1.conv.in_channels // 2, h, w), device=s2.device, dtype=s2.dtype)
            feats.append(self.s1_adapter(s1))

        x = torch.cat(feats, dim=1)
        d1, d2, d3, d4 = self.trunk(x)

        d1 = self._film_apply(d1, self.film1, base_b, era5)
        d2 = self._film_apply(d2, self.film2, base_b, era5)
        d3 = self._film_apply(d3, self.film3, base_b, era5)
        d4 = self._film_apply(d4, self.film4, base_b, era5)

        r = self.head(d1)
        if self.residual_tanh is not None:
            r = torch.tanh(r) * float(self.residual_tanh)
        if r.shape[-2:] != base_b.shape[-2:]:
            r = F.interpolate(r, size=base_b.shape[-2:], mode="bilinear", align_corners=False)
        y = base_b + r
        return r if return_residual else y
