from __future__ import annotations

import argparse
import json
from pathlib import Path
import glob
from typing import List
import subprocess
import sys
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ubuntu/portable import resolution:
# base modules may exist in sibling folder `good_archi`.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _REPO_ROOT / "good_archi", _REPO_ROOT):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from dataset_basenet import BaseNetTable, build_basenet_table
from model_basenet import BaseNetModel
from io_basenet import load_yaml, save_json
from logger_basenet import setup_logging
from metrics_basenet import bias, mae, rmse


def _loader(table, batch_size: int = 4096) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(table.x).float(),
        torch.from_numpy(table.y).float(),
        torch.from_numpy(table.w).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _apply_eval_imputation(table: BaseNetTable, ckpt: dict) -> BaseNetTable:
    imp = ckpt.get("impute", {})
    base_names = imp.get("base_feature_names", [])
    med = np.asarray(imp.get("medians", []), dtype=np.float32)
    final_names = imp.get("final_feature_names", list(table.feature_names))
    mask_names = imp.get("mask_feature_names", [])

    if base_names and list(table.feature_names) != list(base_names):
        name_to_idx = {n: i for i, n in enumerate(table.feature_names)}
        idx = [name_to_idx[n] for n in base_names if n in name_to_idx]
        if len(idx) != len(base_names):
            raise RuntimeError("Evaluation feature names do not match training base features.")
        x = table.x[:, idx]
    else:
        x = table.x
        base_names = list(table.feature_names)

    if med.size == 0:
        med = np.nanmedian(x, axis=0).astype(np.float32)
        med = np.where(np.isfinite(med), med, 0.0)
    if med.shape[0] != x.shape[1]:
        raise RuntimeError("Imputation medians shape mismatch.")

    nan_mask = ~np.isfinite(x)
    x_f = x.copy()
    for j in range(x_f.shape[1]):
        x_f[nan_mask[:, j], j] = med[j]

    if mask_names:
        idx = [base_names.index(n.replace("_isnan", "")) for n in mask_names]
        x_f = np.concatenate([x_f, nan_mask[:, idx].astype(np.float32)], axis=1)

    return BaseNetTable(
        x=x_f.astype(np.float32),
        y=table.y,
        w=table.w,
        vf=table.vf,
        qc_weight=table.qc_weight,
        date_idx=table.date_idx,
        cell_idx=table.cell_idx,
        feature_names=list(final_names),
        dates=table.dates,
        grid_shape=table.grid_shape,
        debug_info=table.debug_info,
    )


def _derive_qc_weight(table: BaseNetTable, cfg: dict) -> np.ndarray:
    idx = {n: i for i, n in enumerate(table.feature_names)}
    req = [
        "modis_valid_day",
        "modis_valid_night",
        "viirs_valid_day",
        "viirs_valid_night",
        "modis_qc_score_day",
        "modis_qc_score_night",
        "viirs_qc_score_day",
        "viirs_qc_score_night",
    ]
    if not all(k in idx for k in req):
        return np.ones((table.x.shape[0],), dtype=np.float32)
    x = table.x
    tcfg = cfg.get("training", {})
    qcfg = cfg.get("qc", {})
    qmin = float(tcfg.get("qc_weight_min", 0.2))
    qpow = max(0.1, float(tcfg.get("qc_weight_power", 1.0)))
    unknown = float(qcfg.get("unknown_qc_score", 0.5))
    num = np.zeros((x.shape[0],), dtype=np.float32)
    den = np.zeros((x.shape[0],), dtype=np.float32)
    pairs = [
        ("modis_qc_score_day", "modis_valid_day"),
        ("modis_qc_score_night", "modis_valid_night"),
        ("viirs_qc_score_day", "viirs_valid_day"),
        ("viirs_qc_score_night", "viirs_valid_night"),
    ]
    for qn, vn in pairs:
        q = x[:, idx[qn]].astype(np.float32)
        v = x[:, idx[vn]] > 0.5
        m = v & np.isfinite(q)
        num[m] += np.clip(q[m], 0.0, 1.0)
        den[m] += 1.0
    out = np.full((x.shape[0],), fill_value=np.clip(unknown, 0.0, 1.0), dtype=np.float32)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    out = np.clip(out, qmin, 1.0).astype(np.float32)
    out = np.power(out, qpow, dtype=np.float32)
    return out


def _resolve_dataset_paths(
    cfg: dict,
    *,
    cli_zarr_path: str | None,
    cli_zarr_30m_path: str | None,
) -> dict:
    cfg = dict(cfg)
    cfg.setdefault("dataset", {})
    ds = cfg["dataset"]

    if cli_zarr_path:
        ds["zarr_path"] = str(cli_zarr_path)
    if cli_zarr_30m_path:
        ds["zarr_30m_path"] = str(cli_zarr_30m_path)

    p_daily = Path(str(ds.get("zarr_path", ""))) if ds.get("zarr_path") else None
    p_30m = Path(str(ds.get("zarr_30m_path", ""))) if ds.get("zarr_30m_path") else None

    if p_daily and p_daily.exists() and p_30m and p_30m.exists():
        return cfg

    env_daily = os.environ.get("DATASET_ZARR") or os.environ.get("MADURAI_ZARR")
    env_30m = os.environ.get("DATASET_30M_ZARR") or os.environ.get("MADURAI_30M_ZARR")
    if env_daily and env_30m and Path(env_daily).exists() and Path(env_30m).exists():
        ds["zarr_path"] = str(Path(env_daily))
        ds["zarr_30m_path"] = str(Path(env_30m))
        return cfg

    candidates = [
        _REPO_ROOT / "madurai.zarr",
        Path.cwd() / "madurai.zarr",
    ]
    candidates_30m = [
        _REPO_ROOT / "madurai_30m.zarr",
        Path.cwd() / "madurai_30m.zarr",
    ]
    found_daily = next((p for p in candidates if p.exists()), None)
    found_30m = next((p for p in candidates_30m if p.exists()), None)
    if found_daily and found_30m:
        ds["zarr_path"] = str(found_daily)
        ds["zarr_30m_path"] = str(found_30m)
        return cfg

    raise RuntimeError(
        "Could not resolve dataset zarr paths for evaluation. "
        "Use --zarr_path and --zarr_30m_path or set DATASET_ZARR/DATASET_30M_ZARR."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BaseNet 1km run.")
    parser.add_argument("--base_dir", type=str, default="basenet_1km")
    parser.add_argument("--run_dir", type=str, default=None, help="Backward compatibility for old runs/<name> layout.")
    parser.add_argument("--zarr_path", type=str, default=None, help="Override daily zarr path (Linux/portable).")
    parser.add_argument("--zarr_30m_path", type=str, default=None, help="Override 30m zarr path (Linux/portable).")
    parser.add_argument(
        "--ensemble_runs",
        type=str,
        default=None,
        help="Comma-separated run directories and/or glob patterns for ensemble checkpoints.",
    )
    parser.add_argument(
        "--ensemble_run_dirs",
        type=str,
        default=None,
        help="Alias of --ensemble_runs (comma-separated run dirs).",
    )
    parser.add_argument(
        "--ensemble_manifest",
        type=str,
        default=None,
        help="Path to ensemble manifest json generated by train_basenet.py --seeds.",
    )
    parser.add_argument("--render_maps", action="store_true", help="Render Madurai district maps after evaluation.")
    parser.add_argument("--render_date_idx", type=int, default=None, help="Optional date_idx for map rendering.")
    parser.add_argument("--render_date", type=str, default=None, help="Optional YYYY-MM-DD for map rendering.")
    args = parser.parse_args()

    base_dir = Path(args.run_dir) if args.run_dir else Path(args.base_dir)
    cfg = load_yaml(base_dir / "results" / "config_resolved.yaml")
    cfg = _resolve_dataset_paths(
        cfg,
        cli_zarr_path=args.zarr_path,
        cli_zarr_30m_path=args.zarr_30m_path,
    )
    logger = setup_logging(base_dir / "logs" / "evaluate_basenet.log", logger_name=f"basenet_1km_eval_{base_dir.name}")

    splits_path = base_dir / "results" / "splits.json"
    if not splits_path.exists():
        splits_path = base_dir / "splits.json"
    split = json.loads(splits_path.read_text(encoding="utf-8"))
    test_idx = np.asarray(split["test_indices"], dtype=np.int64)
    if test_idx.size == 0:
        raise RuntimeError("No test split indices found.")

    table_raw = build_basenet_table(cfg=cfg, split_date_indices=test_idx, logger=logger, split_role="eval")
    ckpt = torch.load(base_dir / "checkpoints" / "best.pt", map_location="cpu")
    table = _apply_eval_imputation(table_raw, ckpt)
    loader = _loader(table, batch_size=int(cfg["training"]["batch_size"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaseNetModel(
        feature_names=ckpt["feature_names"],
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
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    yt_all = []
    yp_all = []
    sigma_vals = []
    gate_entropies = []
    expert_usage = []
    with torch.no_grad():
        for x, y, _w in loader:
            x = x.to(device)
            out = model(x)
            usable = out["usable"] > 0.5
            yp = out["yhat"].detach().cpu().numpy()
            yt = y.numpy()
            m = usable.detach().cpu().numpy() & np.isfinite(yp) & np.isfinite(yt)
            if np.any(m):
                yt_all.append(yt[m])
                yp_all.append(yp[m])
            if out.get("log_sigma") is not None:
                ls_min = float(cfg.get("training", {}).get("nll_log_sigma_min", -2.0))
                ls_max = float(cfg.get("training", {}).get("nll_log_sigma_max", 1.0))
                s = torch.exp(torch.clamp(out["log_sigma"], min=ls_min, max=ls_max))
                sigma_vals.append(float(s.mean().detach().cpu().item()))
            if out.get("gate_entropy") is not None:
                gate_entropies.append(float(out["gate_entropy"].detach().cpu().item()))
            if out.get("expert_usage") is not None:
                expert_usage.append(out["expert_usage"].detach().cpu().numpy().astype(np.float32))

    yt = np.concatenate(yt_all) if yt_all else np.array([], dtype=np.float32)
    yp = np.concatenate(yp_all) if yp_all else np.array([], dtype=np.float32)
    met = {
        "rmse": rmse(yt, yp),
        "mae": mae(yt, yp),
        "bias": bias(yt, yp),
        "n_eval": int(yt.size),
    }
    logger.info("Evaluation metrics: %s", met)
    extra = {
        "model_arch": str(cfg["model"].get("arch", "mlp")).lower(),
        "sigma_mean": float(np.mean(sigma_vals)) if sigma_vals else None,
        "gate_entropy_mean": float(np.mean(gate_entropies)) if gate_entropies else None,
        "expert_usage_mean": np.mean(np.stack(expert_usage, axis=0), axis=0).tolist() if expert_usage else None,
    }

    save_json(base_dir / "results" / "metrics.json", met)
    save_json(base_dir / "results" / "metrics_eval_overall.json", met)
    save_json(base_dir / "results" / "metrics_eval_extra.json", extra)
    pd.DataFrame([met]).to_csv(base_dir / "results" / "metrics_eval.csv", index=False)
    h, w = table.grid_shape
    pred_dir = base_dir / "results" / "predictions_sample"
    pred_dir.mkdir(parents=True, exist_ok=True)
    all_pred = np.zeros((table.x.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, table.x.shape[0], int(cfg["training"]["batch_size"])):
            i1 = min(table.x.shape[0], i0 + int(cfg["training"]["batch_size"]))
            x = torch.from_numpy(table.x[i0:i1]).float().to(device)
            all_pred[i0:i1] = model(x)["yhat"].detach().cpu().numpy()
    qc_weight = _derive_qc_weight(table, cfg)
    vf = table.vf.astype(np.float32) if table.vf is not None else table.w.astype(np.float32)
    bins = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.000001)]
    rows = []
    y_full = table.y.astype(np.float32)
    yp_full = all_pred.astype(np.float32)
    for lo, hi in bins:
        m = (qc_weight >= lo) & (qc_weight < hi) & np.isfinite(y_full) & np.isfinite(yp_full)
        yt_b = y_full[m]
        yp_b = yp_full[m]
        rows.append(
            {
                "qc_bin": f"{lo:.1f}-{min(hi,1.0):.1f}",
                "n": int(np.sum(m)),
                "rmse": rmse(yt_b, yp_b),
                "mae": mae(yt_b, yp_b),
                "bias": bias(yt_b, yp_b),
            }
        )
    pd.DataFrame(rows).to_csv(base_dir / "results" / "metrics_eval_by_qc.csv", index=False)
    vf_bins = [(0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.000001)]
    vf_rows = []
    for lo, hi in vf_bins:
        m = (vf >= lo) & (vf < hi) & np.isfinite(y_full) & np.isfinite(yp_full)
        yt_b = y_full[m]
        yp_b = yp_full[m]
        vf_rows.append(
            {
                "valid_frac_bin": f"{lo:.2f}-{min(hi,1.0):.2f}",
                "n": int(np.sum(m)),
                "rmse": rmse(yt_b, yp_b),
                "mae": mae(yt_b, yp_b),
                "bias": bias(yt_b, yp_b),
            }
        )
    pd.DataFrame(vf_rows).to_csv(base_dir / "results" / "metrics_eval_by_valid_frac.csv", index=False)
    qv_rows = []
    for qlo, qhi in bins:
        for vlo, vhi in vf_bins:
            m = (
                (qc_weight >= qlo)
                & (qc_weight < qhi)
                & (vf >= vlo)
                & (vf < vhi)
                & np.isfinite(y_full)
                & np.isfinite(yp_full)
            )
            yt_b = y_full[m]
            yp_b = yp_full[m]
            qv_rows.append(
                {
                    "qc_bin": f"{qlo:.1f}-{min(qhi,1.0):.1f}",
                    "valid_frac_bin": f"{vlo:.2f}-{min(vhi,1.0):.2f}",
                    "n": int(np.sum(m)),
                    "rmse": rmse(yt_b, yp_b),
                    "mae": mae(yt_b, yp_b),
                    "bias": bias(yt_b, yp_b),
                }
            )
    pd.DataFrame(qv_rows).to_csv(base_dir / "results" / "metrics_eval_by_qc_and_vf.csv", index=False)
    for d in np.unique(table.date_idx):
        m = table.date_idx == d
        y = table.y[m]
        cell = table.cell_idx[m]
        yp_d = all_pred[m]
        pm = np.full((h * w,), np.nan, dtype=np.float32)
        tm = np.full((h * w,), np.nan, dtype=np.float32)
        pm[cell] = yp_d
        tm[cell] = y
        date_str = str(pd.Timestamp(table.dates[int(d)]).date())
        np.save(pred_dir / f"pred_eval_{date_str}.npy", pm.reshape(h, w))
        np.save(pred_dir / f"true_eval_{date_str}.npy", tm.reshape(h, w))

    debug = pd.DataFrame(
        {
            "date_idx": table.date_idx[:200],
            "cell_id": table.cell_idx[:200],
            "y_true": table.y[:200],
            "y_pred": all_pred[:200],
            "landsat_valid_frac": vf[:200],
            "sample_weight": table.w[:200],
            "qc_weight": qc_weight[:200],
        }
    )
    debug.to_csv(base_dir / "results" / "debug_sample.csv", index=False)

    print(json.dumps({"base_dir": base_dir.as_posix(), "metrics": met}, indent=2))

    ens_spec = args.ensemble_runs or args.ensemble_run_dirs
    run_dirs: List[Path] = []
    if args.ensemble_manifest:
        mani = json.loads(Path(args.ensemble_manifest).read_text(encoding="utf-8"))
        for row in mani.get("runs", []):
            p = row.get("run_dir")
            if p:
                run_dirs.append(Path(str(p)))
    elif ens_spec:
        parts = [p.strip() for p in str(ens_spec).split(",") if p.strip()]
        for p in parts:
            if any(ch in p for ch in ("*", "?", "[")):
                run_dirs.extend([Path(x) for x in glob.glob(p)])
            else:
                run_dirs.append(Path(p))
    if run_dirs:
        run_dirs = [p for p in run_dirs if (p / "checkpoints" / "best.pt").exists()]
        if not run_dirs:
            raise RuntimeError("No valid ensemble run dirs found with checkpoints/best.pt.")

        preds = []
        per_model_rows = []
        for rdir in run_dirs:
            e_ckpt = torch.load(rdir / "checkpoints" / "best.pt", map_location="cpu")
            e_table = _apply_eval_imputation(table_raw, e_ckpt)
            e_model = BaseNetModel(
                feature_names=e_ckpt["feature_names"],
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
            ).to(device)
            e_model.load_state_dict(e_ckpt["model_state_dict"])
            e_model.eval()
            e_pred = np.zeros((e_table.x.shape[0],), dtype=np.float32)
            with torch.no_grad():
                for i0 in range(0, e_table.x.shape[0], int(cfg["training"]["batch_size"])):
                    i1 = min(e_table.x.shape[0], i0 + int(cfg["training"]["batch_size"]))
                    x = torch.from_numpy(e_table.x[i0:i1]).float().to(device)
                    e_pred[i0:i1] = e_model(x)["yhat"].detach().cpu().numpy()
            preds.append(e_pred)
            ym = e_table.y.astype(np.float32)
            mm = np.isfinite(ym) & np.isfinite(e_pred)
            per_model_rows.append(
                {
                    "run_dir": rdir.as_posix(),
                    "rmse": rmse(ym[mm], e_pred[mm]),
                    "mae": mae(ym[mm], e_pred[mm]),
                    "bias": bias(ym[mm], e_pred[mm]),
                    "n_eval": int(np.sum(mm)),
                }
            )
        y_ens = np.mean(np.stack(preds, axis=0), axis=0)
        y_true = table.y.astype(np.float32)
        m = np.isfinite(y_true) & np.isfinite(y_ens)
        ens_met = {
            "rmse": rmse(y_true[m], y_ens[m]),
            "mae": mae(y_true[m], y_ens[m]),
            "bias": bias(y_true[m], y_ens[m]),
            "n_eval": int(np.sum(m)),
            "n_models": len(run_dirs),
            "models": [p.as_posix() for p in run_dirs],
        }
        save_json(base_dir / "results" / "metrics_eval_ensemble.json", ens_met)
        pd.DataFrame([ens_met]).to_csv(base_dir / "results" / "metrics_eval_ensemble.csv", index=False)
        pd.DataFrame(per_model_rows).to_csv(base_dir / "results" / "metrics_eval_per_model.csv", index=False)
        logger.info("Ensemble metrics: %s", ens_met)

    if bool(args.render_maps):
        cmd = [
            sys.executable,
            str((Path(__file__).resolve().parent / "render_madurai_maps.py").as_posix()),
            "--run_dir",
            base_dir.as_posix(),
            "--split",
            "test",
            "--save_geotiff",
            "true",
        ]
        if args.render_date_idx is not None:
            cmd.extend(["--date_idx", str(int(args.render_date_idx))])
        elif args.render_date:
            cmd.extend(["--date", str(args.render_date)])
        logger.info("Running render maps command: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stdout:
            logger.info("render_madurai_maps stdout: %s", proc.stdout.strip())
        if proc.returncode != 0:
            logger.error("render_madurai_maps failed rc=%s stderr=%s", proc.returncode, (proc.stderr or "").strip())
        else:
            logger.info("render_madurai_maps completed successfully.")


if __name__ == "__main__":
    main()





