from __future__ import annotations

from typing import List

import torch
from torch import nn

from mlp_blocks_basenet import ResMLP


class MoECorrectionHead(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        gate_in_dim: int,
        num_experts: int,
        expert_hidden: List[int],
        dropout: float,
        layer_norm: bool,
        use_heteroscedastic: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.use_heteroscedastic = bool(use_heteroscedastic)
        if self.num_experts < 2:
            raise ValueError("num_experts must be >= 2 for MoE.")

        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, max(32, gate_in_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(32, gate_in_dim), self.num_experts),
        )
        self.experts = nn.ModuleList(
            [
                ResMLP(
                    in_dim=in_dim,
                    hidden=list(expert_hidden),
                    dropout=dropout,
                    layer_norm=layer_norm,
                )
                for _ in range(self.num_experts)
            ]
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

    def forward(self, x: torch.Tensor, gate_x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.gate(gate_x)
        probs = torch.softmax(logits, dim=-1)
        exp_out = [e(x).unsqueeze(-1) for e in self.experts]
        exp_mat = torch.cat(exp_out, dim=-1)
        delta = torch.sum(exp_mat * probs, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1.0e-8))).sum(dim=-1).mean()
        usage = probs.mean(dim=0)

        out = {
            "delta": delta,
            "gate_probs": probs,
            "gate_entropy": entropy,
            "expert_usage": usage,
        }
        if self.log_sigma_head is not None:
            out["log_sigma"] = self.log_sigma_head(x).squeeze(-1)
        return out

