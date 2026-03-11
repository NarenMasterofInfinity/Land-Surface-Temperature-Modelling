from __future__ import annotations

from typing import List

import torch
from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float, layer_norm: bool) -> None:
        super().__init__()
        norm = nn.LayerNorm(width) if layer_norm else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            norm,
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ResMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float, layer_norm: bool) -> None:
        super().__init__()
        if not hidden:
            raise ValueError("hidden cannot be empty")
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.LayerNorm(hidden[0]) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for i in range(len(hidden) - 1):
            blocks.append(nn.Linear(hidden[i], hidden[i + 1]))
            if layer_norm:
                blocks.append(nn.LayerNorm(hidden[i + 1]))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Dropout(dropout))
        self.mid = nn.Sequential(*blocks)
        self.res = ResidualMLPBlock(hidden[-1], dropout=dropout, layer_norm=layer_norm)
        self.out = nn.Linear(hidden[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.mid(x)
        x = self.res(x)
        return self.out(x).squeeze(-1)

