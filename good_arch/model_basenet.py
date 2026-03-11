from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from ft_transformer_basenet import FTTransformerRegressor
from mlp_blocks_basenet import ResMLP
from moe_basenet import MoECorrectionHead


def _build_mlp(in_dim: int, hidden: List[int], dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers.extend([nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(dropout)])
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def _zero_last_linear(module: nn.Module) -> None:
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
            return


class BaseNetModel(nn.Module):
    def __init__(
        self,
        *,
        feature_names: List[str],
        gate_input_keys: List[str],
        gate_hidden: List[int],
        corr_hidden: List[int],
        dropout: float,
        layer_norm: bool = True,
        thermal_pair_mode: str = "mean",
        use_calibration_head: bool = False,
        arch: str = "mlp",
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        attn_dropout: float = 0.1,
        ffn_mult: int = 4,
        moe_num_experts: int = 4,
        moe_hidden: List[int] | None = None,
        moe_gate_keys: List[str] | None = None,
        use_heteroscedastic: bool = False,
        cal_a_range: float = 0.3,
        cal_b_range: float = 5.0,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_names)
        self.name_to_idx = {n: i for i, n in enumerate(self.feature_names)}
        self.thermal_pair_mode = str(thermal_pair_mode)
        self.use_calibration_head = bool(use_calibration_head)
        self.arch = str(arch).lower()
        self.use_heteroscedastic = bool(use_heteroscedastic)
        self.cal_a_range = float(cal_a_range)
        self.cal_b_range = float(cal_b_range)

        self.gate_idx = [self.name_to_idx[k] for k in gate_input_keys if k in self.name_to_idx]
        if not self.gate_idx:
            raise ValueError("No gate_input_keys found in features.")
        self.gate = _build_mlp(len(self.gate_idx), gate_hidden, dropout)

        if self.use_calibration_head:
            self.calib = _build_mlp(len(self.gate_idx), [32], dropout)
            self.calib_b = _build_mlp(len(self.gate_idx), [32], dropout)
            _zero_last_linear(self.calib)
            _zero_last_linear(self.calib_b)
        else:
            self.calib = None
            self.calib_b = None

        in_dim = len(self.feature_names) + 1
        if self.arch == "mlp":
            self.corr = ResMLP(
                in_dim=in_dim,
                hidden=corr_hidden,
                dropout=dropout,
                layer_norm=layer_norm,
            )
            self.log_sigma_head = (
                nn.Sequential(
                    nn.Linear(in_dim, max(32, in_dim // 2)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(max(32, in_dim // 2), 1),
                )
                if self.use_heteroscedastic
                else None
            )
            self.ft = None
            self.moe = None
            self.moe_gate_idx = []
        elif self.arch == "ft_transformer":
            self.ft = FTTransformerRegressor(
                num_features=in_dim,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                attn_dropout=attn_dropout,
                ffn_mult=ffn_mult,
                dropout=dropout,
                use_heteroscedastic=self.use_heteroscedastic,
            )
            self.corr = None
            self.log_sigma_head = None
            self.moe = None
            self.moe_gate_idx = []
        elif self.arch == "moe":
            keys = list(moe_gate_keys or [])
            if not keys:
                keys = [
                    k
                    for k in self.feature_names
                    if (
                        k.startswith(("modis_valid_", "viirs_valid_", "modis_qc_score_", "viirs_qc_score_"))
                        or k in ("doy_sin", "doy_cos")
                        or k.startswith(("worldcover_frac_", "dynamic_world_frac_"))
                    )
                ]
            self.moe_gate_idx = [self.name_to_idx[k] for k in keys if k in self.name_to_idx]
            if not self.moe_gate_idx:
                self.moe_gate_idx = self.gate_idx[:]
            self.moe = MoECorrectionHead(
                in_dim=in_dim,
                gate_in_dim=len(self.moe_gate_idx),
                num_experts=int(moe_num_experts),
                expert_hidden=list(moe_hidden or corr_hidden),
                dropout=dropout,
                layer_norm=layer_norm,
                use_heteroscedastic=self.use_heteroscedastic,
            )
            self.corr = None
            self.log_sigma_head = None
            self.ft = None
        else:
            raise ValueError(f"Unsupported model arch: {arch}")

    def _col(self, x: torch.Tensor, key: str) -> torch.Tensor:
        return x[:, self.name_to_idx[key]]

    def _thermal_mean(self, day: torch.Tensor, night: torch.Tensor) -> torch.Tensor:
        if self.thermal_pair_mode == "keep_day":
            return day
        if self.thermal_pair_mode == "keep_night":
            return night
        return 0.5 * (day + night)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        gate_in = x[:, self.gate_idx]
        w = torch.sigmoid(self.gate(gate_in).squeeze(-1))

        modis_day = self._col(x, "modis_day")
        modis_night = self._col(x, "modis_night")
        viirs_day = self._col(x, "viirs_day")
        viirs_night = self._col(x, "viirs_night")
        modis_valid = ((self._col(x, "modis_valid_day") > 0.5) | (self._col(x, "modis_valid_night") > 0.5))
        viirs_valid = ((self._col(x, "viirs_valid_day") > 0.5) | (self._col(x, "viirs_valid_night") > 0.5))

        m = self._thermal_mean(modis_day, modis_night)
        v = self._thermal_mean(viirs_day, viirs_night)

        w = torch.where(~modis_valid & viirs_valid, torch.zeros_like(w), w)
        w = torch.where(modis_valid & ~viirs_valid, torch.ones_like(w), w)
        usable = modis_valid | viirs_valid

        fused = w * m + (1.0 - w) * v
        if self.use_calibration_head and self.calib is not None and self.calib_b is not None:
            a_raw = self.calib(gate_in).squeeze(-1)
            b_raw = self.calib_b(gate_in).squeeze(-1)
            a = 1.0 + self.cal_a_range * torch.tanh(a_raw)
            b = self.cal_b_range * torch.tanh(b_raw)
            fused_cal = (a * fused) + b
        else:
            a = torch.ones_like(fused)
            b = torch.zeros_like(fused)
            fused_cal = fused

        corr_in = torch.cat([x, fused_cal.unsqueeze(1)], dim=1)
        gate_probs = None
        gate_entropy = None
        expert_usage = None
        log_sigma = None

        if self.arch == "mlp":
            delta = self.corr(corr_in)
            if self.log_sigma_head is not None:
                log_sigma = self.log_sigma_head(corr_in).squeeze(-1)
        elif self.arch == "ft_transformer":
            out = self.ft(corr_in)
            delta = out["delta"]
            log_sigma = out.get("log_sigma")
        elif self.arch == "moe":
            moe_in = x[:, self.moe_gate_idx]
            out = self.moe(corr_in, moe_in)
            delta = out["delta"]
            log_sigma = out.get("log_sigma")
            gate_probs = out.get("gate_probs")
            gate_entropy = out.get("gate_entropy")
            expert_usage = out.get("expert_usage")
        else:
            raise RuntimeError(f"Unknown arch at forward: {self.arch}")

        yhat = fused_cal + delta
        return {
            "yhat": yhat,
            "w_gate": w,
            "delta": delta,
            "fused": fused,
            "fused_cal": fused_cal,
            "calib_a": a,
            "calib_b": b,
            "usable": usable.float(),
            "log_sigma": log_sigma,
            "gate_probs": gate_probs,
            "gate_entropy": gate_entropy,
            "expert_usage": expert_usage,
        }
