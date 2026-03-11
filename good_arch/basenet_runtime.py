from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from basenet_features_from_zarr import BaseNetFeatureTable
from model_basenet import BaseNetModel


@dataclass
class BaseNetRuntime:
    model: BaseNetModel
    ckpt: dict
    device: torch.device
    batch_cells: int
    feature_names: list[str]


def _resolve_device(device_text: str) -> torch.device:
    d = str(device_text).lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(d)


def load_basenet_from_ckpt(
    *,
    ckpt_path: str | Path,
    device: str = "auto",
    batch_cells: int = 65536,
) -> BaseNetRuntime:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = ckpt["config"]
    model = BaseNetModel(
        feature_names=list(ckpt["feature_names"]),
        gate_input_keys=list(cfg["model"]["gate_input_keys"]),
        gate_hidden=list(cfg["model"]["gate_hidden"]),
        corr_hidden=list(cfg["model"]["corr_hidden"]),
        dropout=float(cfg["model"].get("dropout", 0.15)),
        layer_norm=bool(cfg["model"].get("layer_norm", True)),
        thermal_pair_mode=str(cfg["features"].get("thermal_pair_mode", "mean")),
        use_calibration_head=bool(cfg["model"].get("use_calibration_head", False)),
        arch=str(cfg["model"].get("arch", "mlp")),
        d_model=int(cfg["model"].get("d_model", 64)),
        n_heads=int(cfg["model"].get("n_heads", 4)),
        n_layers=int(cfg["model"].get("n_layers", 3)),
        attn_dropout=float(cfg["model"].get("attn_dropout", 0.1)),
        ffn_mult=int(cfg["model"].get("ffn_mult", 4)),
        moe_num_experts=int(cfg["model"].get("moe_num_experts", 4)),
        moe_hidden=list(cfg["model"].get("moe_hidden", cfg["model"]["corr_hidden"])),
        moe_gate_keys=list(cfg["model"].get("moe_gate_keys", [])),
        use_heteroscedastic=bool(cfg["model"].get("heteroscedastic", cfg["model"].get("use_heteroscedastic", False))),
        cal_a_range=float(cfg["model"].get("cal_a_range", 0.3)),
        cal_b_range=float(cfg["model"].get("cal_b_range", 5.0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.requires_grad_(False)
    resolved = _resolve_device(device)
    model = model.to(resolved)
    return BaseNetRuntime(
        model=model,
        ckpt=ckpt,
        device=resolved,
        batch_cells=max(1, int(batch_cells)),
        feature_names=list(ckpt["feature_names"]),
    )


def predict_basenet_1km(runtime: BaseNetRuntime, table: BaseNetFeatureTable, strict_feature_match: bool = True) -> np.ndarray:
    if strict_feature_match and list(table.feature_names) != list(runtime.feature_names):
        raise RuntimeError(
            f"BaseNet feature order mismatch.\nexpected[:10]={runtime.feature_names[:10]}\n"
            f"got[:10]={table.feature_names[:10]}"
        )
    x = table.x.astype(np.float32, copy=False)
    preds = np.zeros((x.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, x.shape[0], runtime.batch_cells):
            i1 = min(x.shape[0], i0 + runtime.batch_cells)
            xb = torch.from_numpy(x[i0:i1]).to(runtime.device)
            out = runtime.model(xb)
            preds[i0:i1] = out["yhat"].detach().cpu().numpy().astype(np.float32)
    h, w = table.grid_shape
    return preds.reshape(h, w)


def upsample_1km_to_30m(base_1km: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    x = np.asarray(base_1km, dtype=np.float32)
    valid = np.isfinite(x).astype(np.float32)
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    xv = torch.from_numpy(x)[None, None]
    mv = torch.from_numpy(valid)[None, None]
    xv_up = F.interpolate(xv, size=out_hw, mode="bilinear", align_corners=False)
    mv_up = F.interpolate(mv, size=out_hw, mode="bilinear", align_corners=False)
    out = (xv_up / mv_up.clamp(min=1.0e-6))[0, 0].cpu().numpy().astype(np.float32)
    out[mv_up[0, 0].cpu().numpy() <= 1.0e-6] = np.nan
    return out


def predict_basenet_30m(
    *,
    runtime: BaseNetRuntime,
    table: BaseNetFeatureTable,
    out_hw: Tuple[int, int],
    strict_feature_match: bool = True,
) -> np.ndarray:
    pred_1km = predict_basenet_1km(runtime=runtime, table=table, strict_feature_match=strict_feature_match)
    return upsample_1km_to_30m(pred_1km, out_hw=out_hw)

