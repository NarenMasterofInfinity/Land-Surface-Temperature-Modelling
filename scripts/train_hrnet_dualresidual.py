from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    def forward(self, x_era5: torch.Tensor, doy: Optional[torch.Tensor] = None) -> torch.Tensor:
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

    def __init__(self, group_defs: Sequence[Tuple[str, List[int], int]]):
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
    def __init__(self, in_ch: int, widths: Sequence[int] = (32, 48, 64, 80), blocks: Sequence[int] = (2, 2, 2, 2)):
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
            ConvGNAct(in_ch, in_ch // 2, kernel_size=3),
            nn.Conv2d(in_ch // 2, 1, kernel_size=1),
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
            ("s2", s2_idx, 32),
            ("s1", s1_idx, 8),
            ("dem", dem_idx, 4),
            ("world", world_idx, 8),
            ("dyn", dyn_idx, 8),
        ]
        self.group_embed = GroupEmbedding(group_defs)
        embed_ch = sum(out_ch for _, _, out_ch in group_defs)
        residual_in_ch = embed_ch + (1 if gate_use_baseline else 0)

        self.hrnet = ResidualHRNet(in_ch=residual_in_ch)
        self.head_strong = nn.Conv2d(32, 1, kernel_size=1)
        self.head_weak = nn.Conv2d(32, 1, kernel_size=1)

        gate_in = 32 + (1 if gate_use_baseline else 0)
        self.gate = GateNet(gate_in)

    def forward(
        self,
        x: torch.Tensor,
        doy: Optional[torch.Tensor] = None,
        *,
        alpha_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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
        if alpha_override is not None:
            alpha = alpha_override

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


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    mask = mask.float()
    diff = pred - target
    abs_diff = diff.abs()
    huber = torch.where(abs_diff < delta, 0.5 * diff**2, delta * (abs_diff - 0.5 * delta))
    denom = mask.sum().clamp(min=1.0)
    return (huber * mask).sum() / denom


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    diff = pred - target
    denom = mask.sum().clamp(min=1.0)
    return torch.sqrt((diff * diff * mask).sum() / denom)


def ssim_stub(*_: torch.Tensor) -> float:
    return float("nan")


@dataclass
class StageConfig:
    stage_a_epochs: int = 10
    stage_b_epochs: int = 30
    stage_c_epochs: int = 20
    lambda_start: float = 0.02
    lambda_end: float = 0.15
    force_alpha_weak: bool = True
    consistency_b: float = 0.01
    consistency_c: float = 0.05

    def total_epochs(self) -> int:
        return self.stage_a_epochs + self.stage_b_epochs + self.stage_c_epochs

    def stage_at(self, epoch: int) -> Tuple[str, float, float]:
        if epoch < self.stage_a_epochs:
            return "A", 0.0, 0.0
        if epoch < self.stage_a_epochs + self.stage_b_epochs:
            t = (epoch - self.stage_a_epochs) / max(1, self.stage_b_epochs)
            lam = self.lambda_start + t * (self.lambda_end - self.lambda_start)
            return "B", lam, self.consistency_b
        return "C", self.lambda_end, self.consistency_c


@dataclass
class TrainConfig:
    batch_size: int = 4
    lr: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    beta: float = 0.5
    gamma: float = 0.5
    delta_huber: float = 1.0


class LSTSampleDataset(Dataset):
    """
    Dataset stub. Provide a list of dicts with keys:
    x, y_ls, m_ls, y_wk, m_wk, is_landsat, doy (optional).
    """

    def __init__(self, samples: Sequence[Dict[str, torch.Tensor]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class DummyLSTDataset(Dataset):
    def __init__(self, n: int, in_ch: int, h: int, w: int):
        self.n = n
        self.in_ch = in_ch
        self.h = h
        self.w = w

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.randn(self.in_ch, self.h, self.w)
        is_ls = torch.tensor(1 if idx % 5 == 0 else 0, dtype=torch.float32)
        y_ls = torch.randn(1, self.h, self.w)
        y_wk = torch.randn(1, self.h, self.w)
        m_ls = (torch.rand(1, self.h, self.w) > 0.3).float() * is_ls
        m_wk = (torch.rand(1, self.h, self.w) > 0.3).float() * (1.0 - is_ls)
        doy = torch.rand(1)
        return {
            "x": x,
            "y_ls": y_ls,
            "m_ls": m_ls,
            "y_wk": y_wk,
            "m_wk": m_wk,
            "is_landsat": is_ls,
            "doy": doy,
        }


def collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if vals[0].ndim == 0:
            out[k] = torch.stack([v.view(1) for v in vals], dim=0)
        else:
            out[k] = torch.stack(vals, dim=0)
    return out


def apply_alpha_forcing(alpha_pred: torch.Tensor, is_landsat: torch.Tensor, stage: str, force_alpha_weak: bool) -> torch.Tensor:
    if stage == "A":
        return torch.ones_like(alpha_pred)
    if stage == "B":
        is_ls = is_landsat.view(-1, 1, 1, 1)
        if force_alpha_weak:
            return is_ls * torch.ones_like(alpha_pred)
        return torch.where(is_ls > 0.5, torch.ones_like(alpha_pred), alpha_pred)
    return alpha_pred


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
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_ls = batch["y_ls"].to(device)
        m_ls = batch["m_ls"].to(device)
        y_wk = batch["y_wk"].to(device)
        m_wk = batch["m_wk"].to(device)
        is_ls = batch["is_landsat"].to(device)
        doy = batch.get("doy")
        doy = doy.to(device) if doy is not None else None

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            out = model(x, doy=doy)
            alpha = apply_alpha_forcing(out["alpha"], is_ls, stage, force_alpha_weak)
            y_hat = out["base"] + alpha * out["r_strong"] + (1.0 - alpha) * out["r_weak"]
            y_strong = out["y_strong"]
            y_weak = out["y_weak"]

            loss = torch.tensor(0.0, device=device)
            if stage != "A":
                loss = loss + lam * (
                    masked_huber(y_hat, y_wk, m_wk, delta=cfg.delta_huber)
                    + cfg.gamma * masked_huber(y_weak, y_wk, m_wk, delta=cfg.delta_huber)
                )
            loss = loss + masked_huber(y_hat, y_ls, m_ls, delta=cfg.delta_huber)
            loss = loss + cfg.beta * masked_huber(y_strong, y_ls, m_ls, delta=cfg.delta_huber)

            if cons_weight > 0.0:
                m_union = ((m_ls > 0.5) | (m_wk > 0.5)).float()
                loss = loss + cons_weight * masked_huber(out["r_strong"], out["r_weak"], m_union, delta=cfg.delta_huber)

        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * x.shape[0]
        total_count += x.shape[0]

    return total_loss / max(1, total_count)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> float:
    model.eval()
    rmses = []
    for batch in loader:
        x = batch["x"].to(device)
        y_ls = batch["y_ls"].to(device)
        m_ls = batch["m_ls"].to(device)
        is_ls = batch["is_landsat"].to(device)
        doy = batch.get("doy")
        doy = doy.to(device) if doy is not None else None

        out = model(x, doy=doy)
        if is_ls.sum() == 0:
            continue
        alpha = torch.ones_like(out["alpha"])
        y_hat = out["base"] + alpha * out["r_strong"] + (1.0 - alpha) * out["r_weak"]
        rmse_val = masked_rmse(y_hat, y_ls, m_ls)
        rmses.append(float(rmse_val.item()))
    if not rmses:
        return float("nan")
    return float(sum(rmses) / len(rmses))


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


def build_default_channel_indices() -> Dict[str, List[int]]:
    return {
        "era5": list(range(0, 8)),
        "s2": list(range(8, 25)),
        "s1": list(range(25, 31)),
        "dem": list(range(31, 34)),
        "world": [34],
        "dyn": [35],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-doy", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="models/hrnet_dualresidual")
    parser.add_argument("--use-dummy-data", action="store_true")
    parser.add_argument("--dummy-samples", type=int, default=64)
    parser.add_argument("--dummy-h", type=int, default=128)
    parser.add_argument("--dummy-w", type=int, default=128)
    args = parser.parse_args()

    set_seed(args.seed)

    ch = build_default_channel_indices()
    model = LSTFusionModel(
        era5_idx=ch["era5"],
        s2_idx=ch["s2"],
        s1_idx=ch["s1"],
        dem_idx=ch["dem"],
        world_idx=ch["world"],
        dyn_idx=ch["dyn"],
        use_doy=args.use_doy,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_cfg = TrainConfig(batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip)
    stage_cfg = StageConfig()

    if args.use_dummy_data:
        in_ch = 36
        dataset = DummyLSTDataset(args.dummy_samples, in_ch, args.dummy_h, args.dummy_w)
        n_train = int(0.8 * len(dataset))
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    else:
        raise RuntimeError("Provide your dataset and dataloaders here (set --use-dummy-data to run a smoke test).")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_rmse = float("inf")
    total_epochs = args.epochs

    for epoch in range(total_epochs):
        stage, lam, cons_weight = stage_cfg.stage_at(epoch)
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            stage,
            lam,
            cons_weight,
            train_cfg,
            stage_cfg.force_alpha_weak,
        )
        val_rmse = validate(model, val_loader, device, train_cfg)
        elapsed = time.time() - t0
        print(
            f"epoch={epoch:03d} stage={stage} lambda={lam:.3f} train_loss={train_loss:.4f} val_rmse={val_rmse:.4f} time={elapsed:.1f}s"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            save_checkpoint(os.path.join(args.checkpoint_dir, "best.pt"), model, optimizer, epoch, best_rmse)

    save_checkpoint(os.path.join(args.checkpoint_dir, "last.pt"), model, optimizer, total_epochs - 1, best_rmse)


if __name__ == "__main__":
    main()
