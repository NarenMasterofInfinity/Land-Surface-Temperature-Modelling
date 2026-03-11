from __future__ import annotations

import torch
from torch import nn


class FTTransformerRegressor(nn.Module):
    def __init__(
        self,
        *,
        num_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        attn_dropout: float = 0.1,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        use_heteroscedastic: bool = False,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be > 0")
        self.num_features = int(num_features)
        self.d_model = int(d_model)
        self.use_heteroscedastic = bool(use_heteroscedastic)

        self.num_weight = nn.Parameter(torch.empty(self.num_features, self.d_model))
        self.num_bias = nn.Parameter(torch.zeros(self.num_features, self.d_model))
        nn.init.xavier_uniform_(self.num_weight)

        self.cls = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.num_features + 1, self.d_model))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

        enc = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(self.d_model * int(ffn_mult)),
            dropout=float(attn_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=int(n_layers))
        self.pre_head = nn.Sequential(nn.LayerNorm(self.d_model), nn.Dropout(float(dropout)))
        self.delta_head = nn.Linear(self.d_model, 1)
        self.log_sigma_head = nn.Linear(self.d_model, 1) if self.use_heteroscedastic else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Numeric token embedding per feature: e_j = W_j * x_j + b_j
        tok = x.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
        cls = self.cls.expand(x.shape[0], -1, -1)
        tok = torch.cat([cls, tok], dim=1) + self.pos
        enc = self.encoder(tok)
        pooled = self.pre_head(enc[:, 0, :])
        delta = self.delta_head(pooled).squeeze(-1)
        out = {"delta": delta}
        if self.log_sigma_head is not None:
            out["log_sigma"] = self.log_sigma_head(pooled).squeeze(-1)
        return out

