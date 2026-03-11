from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader, TensorDataset

from dataset_basenet import BaseNetTable, build_basenet_table
from splits_basenet import build_time_splits
from losses_basenet import (
    weighted_gaussian_nll_with_delta_penalty,
    weighted_huber,
    weighted_huber_with_delta_penalty,
)
from model_basenet import BaseNetModel
from io_basenet import load_yaml, save_json, save_yaml
from seed_basenet import seed_everything
from logger_basenet import setup_logging
from metrics_basenet import bias, mae, rmse
from registry_basenet import discover_dataset_paths


def _decode_times(root_daily: zarr.Group, group_name: str) -> pd.DatetimeIndex:
    vals = root_daily[group_name][:]
    raw = [v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v) for v in vals.tolist()]
    return pd.to_datetime(raw, format="%Y_%m_%d", errors="coerce")


def _make_loader(table: BaseNetTable, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(table.x).float(),
        torch.from_numpy(table.y).float(),
        torch.from_numpy(table.w).float(),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _run_epoch(
    *,
    model: BaseNetModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    use_amp: bool,
    huber_delta: float,
    lambda_delta: float,
    grad_clip_norm: float,
    loss_type: str,
    nll_clamp_log_sigma: Tuple[float, float],
    aux_delta_loss: bool,
    aux_delta_weight: float,
    lambda_bias: float,
    lambda_cal: float,
    cal_b_range: float,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []
    losses: List[float] = []
    sigmas: List[float] = []
    gate_entropy_vals: List[float] = []
    expert_usage_vals: List[np.ndarray] = []
    a_vals: List[float] = []
    b_vals: List[float] = []
    cal_penalties: List[float] = []

    for x, y, w in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        w = torch.clamp(torch.nan_to_num(w, nan=1.0e-3, posinf=1.0, neginf=1.0e-3), min=1.0e-3)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x)
            if str(loss_type).lower() == "nll":
                log_sigma = out.get("log_sigma")
                if log_sigma is None:
                    log_sigma = torch.zeros_like(out["yhat"])
                loss = weighted_gaussian_nll_with_delta_penalty(
                    yhat=out["yhat"],
                    y=y,
                    log_sigma=log_sigma,
                    sample_weight=w,
                    usable_mask=out["usable"],
                    delta_term=out["delta"],
                    lambda_delta=lambda_delta,
                    clamp_log_sigma=nll_clamp_log_sigma,
                    lambda_bias=lambda_bias,
                )
            else:
                loss = weighted_huber_with_delta_penalty(
                    yhat=out["yhat"],
                    y=y,
                    sample_weight=w,
                    usable_mask=out["usable"],
                    delta_term=out["delta"],
                    huber_delta=huber_delta,
                    lambda_delta=lambda_delta,
                    lambda_bias=lambda_bias,
                )
            if aux_delta_loss:
                target_delta = y - out["fused_cal"]
                dloss = weighted_huber(
                    pred=out["delta"],
                    target=target_delta,
                    sample_weight=w,
                    usable_mask=out["usable"],
                    huber_delta=huber_delta,
                )
                loss = loss + (aux_delta_weight * dloss)
            calib_a = out.get("calib_a")
            calib_b = out.get("calib_b")
            if lambda_cal > 0.0 and calib_a is not None and calib_b is not None:
                denom = max(float(cal_b_range), 1.0e-6)
                cal_pen = ((calib_a - 1.0) * (calib_a - 1.0)) + ((calib_b / denom) * (calib_b / denom))
                cal_pen = torch.mean(cal_pen)
                loss = loss + (lambda_cal * cal_pen)
                cal_penalties.append(float(cal_pen.detach().cpu().item()))
        if (not torch.isfinite(loss)) or (not torch.isfinite(out["yhat"]).all()):
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
            continue

        if train_mode:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        losses.append(float(loss.item()))
        if out.get("log_sigma") is not None:
            sigma = torch.exp(torch.clamp(out["log_sigma"], min=nll_clamp_log_sigma[0], max=nll_clamp_log_sigma[1]))
            sigmas.append(float(torch.mean(sigma).detach().cpu().item()))
        if out.get("gate_entropy") is not None:
            gate_entropy_vals.append(float(out["gate_entropy"].detach().cpu().item()))
        if out.get("expert_usage") is not None:
            expert_usage_vals.append(out["expert_usage"].detach().cpu().numpy().astype(np.float32))
        if out.get("calib_a") is not None:
            a_vals.append(float(out["calib_a"].detach().mean().cpu().item()))
        if out.get("calib_b") is not None:
            b_vals.append(float(out["calib_b"].detach().mean().cpu().item()))
        usable = (out["usable"] > 0.5).detach().cpu().numpy()
        yp = out["yhat"].detach().cpu().numpy()
        yt = y.detach().cpu().numpy()
        finite = np.isfinite(yp) & np.isfinite(yt)
        m = usable & finite
        if np.any(m):
            y_true_list.append(yt[m])
            y_pred_list.append(yp[m])

    y_true = np.concatenate(y_true_list) if y_true_list else np.array([], dtype=np.float32)
    y_pred = np.concatenate(y_pred_list) if y_pred_list else np.array([], dtype=np.float32)
    out_row = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }
    if sigmas:
        out_row["sigma_mean"] = float(np.mean(sigmas))
    if gate_entropy_vals:
        out_row["gate_entropy"] = float(np.mean(gate_entropy_vals))
    if expert_usage_vals:
        out_row["expert_usage"] = np.mean(np.stack(expert_usage_vals, axis=0), axis=0).tolist()
    if a_vals:
        out_row["a_mean"] = float(np.mean(a_vals))
    if b_vals:
        out_row["b_mean"] = float(np.mean(b_vals))
    if cal_penalties:
        out_row["cal_penalty"] = float(np.mean(cal_penalties))
    return out_row


def _build_run_summary(
    *,
    base_dir: Path,
    best_epoch: int,
    best_val_rmse: float,
    feature_names: List[str],
    split_info: Dict[str, List[str]],
) -> None:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1])
            .strip()
        )
    except Exception:
        commit = "unavailable"
    txt = [
        "# BaseNet 1km Run Summary",
        "",
        f"- commit: `{commit}`",
        f"- base_dir: `{base_dir.as_posix()}`",
        f"- best_epoch: `{best_epoch}`",
        f"- best_val_rmse: `{best_val_rmse:.6f}`",
        f"- features ({len(feature_names)}): `{', '.join(feature_names)}`",
        f"- train_date_range: `{split_info.get('train', ['n/a', 'n/a'])}`",
        f"- val_date_range: `{split_info.get('val', ['n/a', 'n/a'])}`",
        f"- test_date_range: `{split_info.get('test', ['n/a', 'n/a'])}`",
    ]
    (base_dir / "results" / "run_summary.md").write_text("\n".join(txt), encoding="utf-8")


def _date_range_from_idx(times: pd.DatetimeIndex, idx: np.ndarray) -> List[str]:
    if idx.size == 0:
        return ["n/a", "n/a"]
    tt = times[idx]
    return [str(pd.Timestamp(tt.min()).date()), str(pd.Timestamp(tt.max()).date())]


def _filter_dates_with_gt(cfg: Dict, date_indices: np.ndarray, min_frac: float = 0.005) -> np.ndarray:
    if date_indices.size == 0:
        return date_indices
    root_30m = zarr.open_group(cfg["dataset"]["zarr_30m_path"], mode="r")
    valid_arr = root_30m[cfg["dataset"]["landsat_group"]]["valid"]
    keep = []
    for t in date_indices.tolist():
        v = np.asarray(valid_arr[int(t), 0], dtype=np.float32)
        frac = float(np.mean(v > 0))
        if frac >= min_frac:
            keep.append(int(t))
    return np.asarray(keep, dtype=np.int64)


def _stats(arr: np.ndarray) -> Dict[str, float]:
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return {k: float("nan") for k in ("min", "mean", "max", "median", "p01", "p99")}
    v = x[finite]
    return {
        "min": float(np.nanmin(v)),
        "mean": float(np.nanmean(v)),
        "max": float(np.nanmax(v)),
        "median": float(np.nanmedian(v)),
        "p01": float(np.nanpercentile(v, 1)),
        "p99": float(np.nanpercentile(v, 99)),
    }


def _log_split_sanity(name: str, table: BaseNetTable, logger) -> None:
    logger.info("[%s] y stats: %s", name, _stats(table.y))
    key_feats = ["modis_day", "viirs_day", "era5_band_1", "era5_band_2"]
    for fn in key_feats:
        if fn in table.feature_names:
            i = table.feature_names.index(fn)
            vals = table.x[:, i]
            nan_pct = float(np.mean(~np.isfinite(vals)) * 100.0)
            logger.info("[%s] %s stats=%s nan%%=%.3f", name, fn, _stats(vals), nan_pct)
    if table.debug_info:
        logger.info("[%s] raw_149_count=%s post_149_count=%s", name, table.debug_info.get("raw_149_count"), table.debug_info.get("post_149_count"))


def _hard_sanity_checks(cfg: Dict, table_val: BaseNetTable, table_train: BaseNetTable) -> None:
    s = cfg.get("sanity", {})
    yv = _stats(table_val.y)
    y_min = float(s.get("y_c_min", -50))
    y_max = float(s.get("y_c_max", 80))
    if not (y_min <= yv["median"] <= y_max):
        raise RuntimeError(f"Val target median out of bounds: {yv['median']} not in [{y_min}, {y_max}]")
    if yv["p99"] > float(s.get("y_p99_max", 100)):
        raise RuntimeError(f"Val target p99 too high: {yv['p99']}")
    if yv["p01"] < float(s.get("y_p01_min", -80)):
        raise RuntimeError(f"Val target p01 too low: {yv['p01']}")

    for name in ("modis_day", "viirs_day"):
        if name not in table_train.feature_names:
            continue
        i = table_train.feature_names.index(name)
        st = _stats(table_train.x[:, i])
        if st["median"] < y_min or st["median"] > y_max:
            raise RuntimeError(f"{name} median out of range after conversion: {st['median']}")

    if table_val.debug_info:
        post = table_val.debug_info.get("post_149_count", {})
        if any(float(v) > 0 for v in post.values()):
            raise RuntimeError(f"149 sentinel remains after sanitization: {post}")


def _apply_imputation(train: BaseNetTable, val: BaseNetTable, test: BaseNetTable, cfg: Dict, logger) -> Dict[str, Any]:
    im_cfg = cfg.get("impute", {})
    strategy = str(im_cfg.get("strategy", "median")).lower()
    add_nan_masks = bool(im_cfg.get("add_nan_masks", True))
    if strategy != "median":
        raise RuntimeError(f"Unsupported imputation strategy: {strategy}")

    x_train = train.x.copy()
    x_val = val.x.copy()
    x_test = test.x.copy()
    feat_names_in = list(train.feature_names)
    med = np.zeros((x_train.shape[1],), dtype=np.float32)
    for j in range(x_train.shape[1]):
        col = x_train[:, j]
        finite = np.isfinite(col)
        med[j] = float(np.nanmedian(col[finite])) if np.any(finite) else 0.0

    def _fill(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nan_mask = ~np.isfinite(x)
        out = x.copy()
        for j in range(out.shape[1]):
            out[nan_mask[:, j], j] = med[j]
        return out.astype(np.float32), nan_mask.astype(np.float32)

    x_train_f, nan_train = _fill(x_train)
    x_val_f, nan_val = _fill(x_val)
    x_test_f, nan_test = _fill(x_test)
    feature_names = list(train.feature_names)
    mask_names: List[str] = []

    if add_nan_masks:
        selected_idx = []
        for i, n in enumerate(feature_names):
            if n.startswith(("modis_", "viirs_", "era5_")):
                selected_idx.append(i)
        selected_idx = sorted(set(selected_idx))
        if selected_idx:
            x_train_f = np.concatenate([x_train_f, nan_train[:, selected_idx]], axis=1)
            x_val_f = np.concatenate([x_val_f, nan_val[:, selected_idx]], axis=1)
            x_test_f = np.concatenate([x_test_f, nan_test[:, selected_idx]], axis=1)
            mask_names = [f"{feature_names[i]}_isnan" for i in selected_idx]
            feature_names = feature_names + mask_names

    def _update_table(t: BaseNetTable, x: np.ndarray, names: List[str]) -> BaseNetTable:
        return BaseNetTable(
            x=x,
            y=t.y,
            w=t.w,
            vf=t.vf,
            qc_weight=t.qc_weight,
            date_idx=t.date_idx,
            cell_idx=t.cell_idx,
            feature_names=names,
            dates=t.dates,
            grid_shape=t.grid_shape,
            debug_info=t.debug_info,
        )

    train2 = _update_table(train, x_train_f, feature_names)
    val2 = _update_table(val, x_val_f, feature_names)
    test2 = _update_table(test, x_test_f, feature_names)

    if not np.isfinite(train2.x).all() or not np.isfinite(val2.x).all() or not np.isfinite(test2.x).all():
        raise RuntimeError("NaNs remained after imputation.")
    lag_nb_idx = [i for i, n in enumerate(feat_names_in) if ("_lag" in n or "_nb_" in n)]
    if lag_nb_idx:
        tr_before = float(np.mean(~np.isfinite(x_train[:, lag_nb_idx])) * 100.0)
        va_before = float(np.mean(~np.isfinite(x_val[:, lag_nb_idx])) * 100.0)
        te_before = float(np.mean(~np.isfinite(x_test[:, lag_nb_idx])) * 100.0)
        tr_after = float(np.mean(~np.isfinite(train2.x[:, lag_nb_idx])) * 100.0)
        va_after = float(np.mean(~np.isfinite(val2.x[:, lag_nb_idx])) * 100.0)
        te_after = float(np.mean(~np.isfinite(test2.x[:, lag_nb_idx])) * 100.0)
        logger.info(
            "Lag/NB missingness%% before_impute train=%.3f val=%.3f test=%.3f after_impute train=%.3f val=%.3f test=%.3f",
            tr_before,
            va_before,
            te_before,
            tr_after,
            va_after,
            te_after,
        )
    logger.info("Imputation done strategy=%s add_nan_masks=%s final_feature_count=%d", strategy, add_nan_masks, len(feature_names))
    return {
        "train": train2,
        "val": val2,
        "test": test2,
        "medians": med.tolist(),
        "base_feature_names": train.feature_names,
        "final_feature_names": feature_names,
        "mask_feature_names": mask_names,
    }


def _write_debug_sample(path: Path, table: BaseNetTable, preds: np.ndarray, n_rows: int = 200) -> None:
    cols = {n: table.x[:, i] for i, n in enumerate(table.feature_names)}
    df = pd.DataFrame(
        {
            "date_idx": table.date_idx,
            "cell_id": table.cell_idx,
            "y_true": table.y,
            "y_pred": preds,
            "landsat_valid_frac": table.vf if table.vf is not None else table.w,
            "sample_weight": table.w,
            "modis_day": cols.get("modis_day", np.nan),
            "viirs_day": cols.get("viirs_day", np.nan),
            "era5_band_1": cols.get("era5_band_1", np.nan),
        }
    )
    df = df.head(n_rows)
    df.to_csv(path, index=False)


def _parse_seed_list(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BaseNet 1km tabular model.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--zarr_path", type=str, default=None)
    parser.add_argument("--zarr_30m_path", type=str, default=None)
    parser.add_argument("--min_epochs", type=int, default=None)
    parser.add_argument("--max_bad_epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for multi-run training.")
    parser.add_argument("--seed_override", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--base_dir_override", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(args.config)
    if args.seeds and args.seed_override is None:
        seeds = _parse_seed_list(args.seeds)
        if not seeds:
            raise RuntimeError("No valid seeds provided in --seeds.")
        out_cfg0 = cfg.setdefault("output", cfg.get("outputs", {}))
        base0 = Path(out_cfg0.get("base_dir", "basenet_1km"))
        if not base0.is_absolute():
            base0 = repo_root / base0
        manifest_runs: List[Dict[str, Any]] = []
        for s in seeds:
            run_base = Path(f"{base0.as_posix()}_seed{s}")
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(args.config),
                "--seed_override",
                str(int(s)),
                "--base_dir_override",
                run_base.as_posix(),
            ]
            if args.zarr_path:
                cmd += ["--zarr_path", str(args.zarr_path)]
            if args.zarr_30m_path:
                cmd += ["--zarr_30m_path", str(args.zarr_30m_path)]
            if args.min_epochs is not None:
                cmd += ["--min_epochs", str(int(args.min_epochs))]
            if args.max_bad_epochs is not None:
                cmd += ["--max_bad_epochs", str(int(args.max_bad_epochs))]
            if args.smoke:
                cmd += ["--smoke"]
            print(f"[ensemble-train] launching seed={s} base_dir={run_base.as_posix()}")
            subprocess.run(cmd, check=True, cwd=repo_root)
            manifest_runs.append(
                {
                    "seed": int(s),
                    "run_dir": run_base.as_posix(),
                    "checkpoint": (run_base / "checkpoints" / "best.pt").as_posix(),
                }
            )
        manifest = {
            "config": str(Path(args.config).as_posix()),
            "seeds": [int(s) for s in seeds],
            "runs": manifest_runs,
        }
        save_json(base0 / "results" / "ensemble_manifest.json", manifest)
        print(json.dumps({"ensemble_manifest": (base0 / "results" / "ensemble_manifest.json").as_posix(), "runs": manifest_runs}, indent=2))
        return

    # Normalize important output/cache paths to repository root to avoid nested layouts.
    cache_dir_cfg = cfg.setdefault("dataset", {}).get("cache_dir", "good_archi/results/cache")
    cache_dir = Path(cache_dir_cfg)
    if not cache_dir.is_absolute():
        cache_dir = repo_root / cache_dir
    cfg["dataset"]["cache_dir"] = str(cache_dir)

    out_cfg = cfg.setdefault("output", cfg.get("outputs", {}))
    base_dir = Path(args.base_dir_override) if args.base_dir_override else Path(out_cfg.get("base_dir", "basenet_1km"))
    if not base_dir.is_absolute():
        base_dir = repo_root / base_dir
    if args.base_dir_override:
        logs_dir = base_dir / "logs"
        results_dir = base_dir / "results"
        checkpoints_dir = base_dir / "checkpoints"
        tensorboard_dir = base_dir / "results" / "tensorboard"
    else:
        logs_dir = Path(out_cfg.get("logs_dir", base_dir / "logs"))
        results_dir = Path(out_cfg.get("results_dir", base_dir / "results"))
        checkpoints_dir = Path(out_cfg.get("checkpoints_dir", base_dir / "checkpoints"))
        tensorboard_dir = Path(out_cfg.get("tensorboard_dir", base_dir / "results" / "tensorboard"))
    if not logs_dir.is_absolute():
        logs_dir = repo_root / logs_dir
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    if not checkpoints_dir.is_absolute():
        checkpoints_dir = repo_root / checkpoints_dir
    if not tensorboard_dir.is_absolute():
        tensorboard_dir = repo_root / tensorboard_dir
    pred_dir = results_dir / "predictions_sample"
    for p in (base_dir, logs_dir, results_dir, checkpoints_dir, tensorboard_dir, pred_dir):
        p.mkdir(parents=True, exist_ok=True)
    cfg["output"] = {
        "base_dir": str(base_dir),
        "logs_dir": str(logs_dir),
        "results_dir": str(results_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "tensorboard_dir": str(tensorboard_dir),
        "predictions_dir": str(pred_dir),
    }

    ds_resolved = discover_dataset_paths(
        repo_root=repo_root,
        cli_daily=args.zarr_path,
        cli_30m=args.zarr_30m_path,
    )
    cfg.setdefault("dataset", {})
    cfg["dataset"]["zarr_path"] = ds_resolved["daily"]
    cfg["dataset"]["zarr_30m_path"] = ds_resolved["zarr_30m"]

    if args.seed_override is not None:
        cfg["seed"] = int(args.seed_override)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    paths = {
        "base_dir": base_dir,
        "logs": logs_dir,
        "checkpoints": checkpoints_dir,
        "results": results_dir,
        "predictions_sample": pred_dir,
        "tensorboard": tensorboard_dir,
    }
    logger = setup_logging(paths["logs"] / "train_basenet.log", logger_name="basenet_1km_train")
    logger.info("Dataset path resolution branch: %s", ds_resolved["source"])
    logger.info("Resolved daily zarr: %s", cfg["dataset"]["zarr_path"])
    logger.info("Resolved 30m zarr: %s", cfg["dataset"]["zarr_30m_path"])
    logger.info("Output base directory: %s", base_dir.as_posix())

    if not torch.cuda.is_available() and bool(cfg["training"].get("require_cuda", True)):
        raise RuntimeError("CUDA is required by config but not available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    root_daily = zarr.open_group(cfg["dataset"]["zarr_path"], mode="r")
    daily_times = _decode_times(root_daily, cfg["dataset"]["time_daily_group"])
    split = build_time_splits(daily_times, cfg["splits"], seed)
    split.train_idx = _filter_dates_with_gt(cfg, split.train_idx)
    split.val_idx = _filter_dates_with_gt(cfg, split.val_idx)
    split.test_idx = _filter_dates_with_gt(cfg, split.test_idx)
    logger.info("Dates with GT coverage train=%d val=%d test=%d", len(split.train_idx), len(split.val_idx), len(split.test_idx))

    smoke_cfg = cfg["training"].get("smoke", {})
    do_smoke = args.smoke or bool(smoke_cfg.get("enabled", False))
    if do_smoke:
        cap = int(smoke_cfg.get("max_dates_per_split", 14))
        split.train_idx = split.train_idx[:cap]
        split.val_idx = split.val_idx[: max(4, cap // 2)]
        split.test_idx = split.test_idx[: max(4, cap // 2)]
        cfg["training"]["epochs"] = int(smoke_cfg.get("epochs", 2))
        logger.info("Smoke mode enabled: train_dates=%d val_dates=%d", len(split.train_idx), len(split.val_idx))

    if split.train_idx.size == 0 or split.val_idx.size == 0 or split.test_idx.size == 0:
        raise RuntimeError(
            f"Split has empty date set after GT filtering: train={split.train_idx.size}, "
            f"val={split.val_idx.size}, test={split.test_idx.size}"
        )

    split_json = {
        "train_indices": split.train_idx.tolist(),
        "val_indices": split.val_idx.tolist(),
        "test_indices": split.test_idx.tolist(),
        "train_dates": [str(pd.Timestamp(daily_times[i]).date()) for i in split.train_idx.tolist()],
        "val_dates": [str(pd.Timestamp(daily_times[i]).date()) for i in split.val_idx.tolist()],
        "test_dates": [str(pd.Timestamp(daily_times[i]).date()) for i in split.test_idx.tolist()],
    }
    save_json(paths["results"] / "splits.json", split_json)

    train_table = build_basenet_table(cfg=cfg, split_date_indices=split.train_idx, logger=logger, split_role="train")
    val_table = build_basenet_table(cfg=cfg, split_date_indices=split.val_idx, logger=logger, split_role="val")
    test_table = build_basenet_table(cfg=cfg, split_date_indices=split.test_idx, logger=logger, split_role="eval")

    _log_split_sanity("train", train_table, logger)
    _log_split_sanity("val", val_table, logger)
    _log_split_sanity("test", test_table, logger)
    _hard_sanity_checks(cfg, val_table, train_table)

    imp = _apply_imputation(train_table, val_table, test_table, cfg, logger)
    train_table = imp["train"]
    val_table = imp["val"]
    test_table = imp["test"]

    if do_smoke:
        tr_cap = int(smoke_cfg.get("max_train_samples", 50000))
        va_cap = int(smoke_cfg.get("max_val_samples", 20000))
        train_table = BaseNetTable(**{**train_table.__dict__, "x": train_table.x[:tr_cap], "y": train_table.y[:tr_cap], "w": train_table.w[:tr_cap], "date_idx": train_table.date_idx[:tr_cap], "cell_idx": train_table.cell_idx[:tr_cap]})
        val_table = BaseNetTable(**{**val_table.__dict__, "x": val_table.x[:va_cap], "y": val_table.y[:va_cap], "w": val_table.w[:va_cap], "date_idx": val_table.date_idx[:va_cap], "cell_idx": val_table.cell_idx[:va_cap]})

    logger.info(
        "Samples train=%d val=%d test=%d min_valid_frac_train=%.3f min_valid_frac_eval=%.3f",
        train_table.x.shape[0],
        val_table.x.shape[0],
        test_table.x.shape[0],
        float(cfg["dataset"].get("min_valid_frac_train", cfg["training"].get("min_valid_frac", cfg["dataset"].get("min_valid_frac", 0.6)))),
        float(cfg["dataset"].get("min_valid_frac_eval", cfg["training"].get("min_valid_frac", cfg["dataset"].get("min_valid_frac", 0.6)))),
    )
    fc_before = train_table.debug_info.get("feature_count_before_lag")
    fc_after = train_table.debug_info.get("feature_count_after_lag")
    if fc_before is not None and fc_after is not None:
        logger.info(
            "Feature count base_plus_engineered=%s after_lag=%s neighborhood_created=%s",
            fc_before,
            fc_after,
            train_table.debug_info.get("neighborhood_created"),
        )
    if train_table.debug_info.get("neighborhood_enabled"):
        logger.info(
            "Neighborhood summary added_count=%s cache_hits=%s compute_sec=%.2f",
            train_table.debug_info.get("neighborhood_added_count"),
            train_table.debug_info.get("neighborhood_cache_hits"),
            float(train_table.debug_info.get("neighborhood_build_sec", 0.0)),
        )
    if "qc_weight_mean" in train_table.debug_info:
        logger.info(
            "Train qc_weight mean=%.4f p10=%.4f p50=%.4f p90=%.4f clamped_frac=%.4f",
            float(train_table.debug_info.get("qc_weight_mean", float("nan"))),
            float(train_table.debug_info.get("qc_weight_p10", float("nan"))),
            float(train_table.debug_info.get("qc_weight_p50", float("nan"))),
            float(train_table.debug_info.get("qc_weight_p90", float("nan"))),
            float(train_table.debug_info.get("qc_weight_clamped_frac", float("nan"))),
        )
    if "qc_weight_mean" in val_table.debug_info:
        logger.info(
            "Val qc_weight mean=%.4f p10=%.4f p50=%.4f p90=%.4f clamped_frac=%.4f",
            float(val_table.debug_info.get("qc_weight_mean", float("nan"))),
            float(val_table.debug_info.get("qc_weight_p10", float("nan"))),
            float(val_table.debug_info.get("qc_weight_p50", float("nan"))),
            float(val_table.debug_info.get("qc_weight_p90", float("nan"))),
            float(val_table.debug_info.get("qc_weight_clamped_frac", float("nan"))),
        )
    logger.info("Feature keys: %s", ", ".join(train_table.feature_names))

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"].get("num_workers", 0))
    train_loader = _make_loader(train_table, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = _make_loader(val_table, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = _make_loader(test_table, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = BaseNetModel(
        feature_names=train_table.feature_names,
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
    n_params = sum(int(p.numel()) for p in model.parameters())
    logger.info("Model arch=%s params=%d", str(cfg['model'].get('arch', 'mlp')).lower(), n_params)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    sched_name = str(cfg["training"].get("lr_scheduler", "none")).lower()
    scheduler = None
    if sched_name == "plateau":
        plateau_factor = float(cfg["training"].get("plateau_factor", 0.5))
        plateau_patience = int(cfg["training"].get("plateau_patience", 3))
        min_lr = float(cfg["training"].get("min_lr", 1.0e-5))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=min_lr,
        )
    elif sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(cfg["training"]["epochs"])),
        )

    with torch.no_grad():
        xb = torch.from_numpy(train_table.x[: min(1024, train_table.x.shape[0])]).float().to(device)
        out0 = model(xb)["yhat"].detach().cpu().numpy()
        st0 = _stats(out0)
        yabs = float(cfg.get("sanity", {}).get("y_pred_abs_max", 150.0))
        if (not np.isfinite(st0["median"])) or abs(st0["median"]) > yabs or st0["max"] > yabs or st0["min"] < -yabs:
            raise RuntimeError(f"First forward y_pred scale is absurd: {st0}")
        logger.info("First forward y_pred stats: %s", st0)

    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(log_dir=str(paths["tensorboard"]))
    except Exception:
        logger.info("TensorBoard not available; created tensorboard dir without writer.")

    epochs = int(cfg["training"]["epochs"])
    min_epochs = int(args.min_epochs if args.min_epochs is not None else cfg["training"].get("min_epochs", 5))
    patience = int(cfg["training"].get("patience", 6))
    max_bad_epochs = int(args.max_bad_epochs if args.max_bad_epochs is not None else cfg["training"].get("max_bad_epochs", 6))
    huber_delta = float(cfg["training"].get("huber_delta", 1.0))
    lambda_delta = float(cfg["training"].get("lambda_delta", 1e-3))
    grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 1.0))
    use_amp = bool(cfg["training"].get("use_amp", True)) and device.type == "cuda"
    loss_type = str(cfg["training"].get("loss_type", "huber")).lower()
    if "nll_log_sigma_min" in cfg["training"] or "nll_log_sigma_max" in cfg["training"]:
        nll_clamp_log_sigma = (
            float(cfg["training"].get("nll_log_sigma_min", -2.0)),
            float(cfg["training"].get("nll_log_sigma_max", 1.0)),
        )
    else:
        nll_clamp_cfg = cfg["training"].get("nll_clamp_log_sigma", [-3.0, 2.0])
        if isinstance(nll_clamp_cfg, (list, tuple)) and len(nll_clamp_cfg) == 2:
            nll_clamp_log_sigma = (float(nll_clamp_cfg[0]), float(nll_clamp_cfg[1]))
        else:
            nll_clamp_log_sigma = (-3.0, 2.0)
    aux_delta_loss = bool(cfg["training"].get("aux_delta_loss", False))
    aux_delta_weight = float(cfg["training"].get("aux_delta_weight", 0.2))
    lambda_bias_cfg = float(cfg["training"].get("lambda_bias", 0.0))
    lambda_bias = min(max(lambda_bias_cfg, 0.0), 0.01)
    if lambda_bias_cfg != lambda_bias:
        logger.warning("Clamped training.lambda_bias from %.6f to %.6f for stability.", lambda_bias_cfg, lambda_bias)
    lambda_cal = max(float(cfg["training"].get("lambda_cal", 0.01)), 0.0)
    cal_b_range = max(float(cfg["model"].get("cal_b_range", 5.0)), 1.0e-6)

    history: List[Dict[str, float]] = []
    best_val_rmse = float("inf")
    best_epoch = -1
    bad_epochs = 0
    ckpt_best = paths["checkpoints"] / "best.pt"
    ckpt_last = paths["checkpoints"] / "last.pt"

    for epoch in range(1, epochs + 1):
        tr = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            use_amp=use_amp,
            huber_delta=huber_delta,
            lambda_delta=lambda_delta,
            grad_clip_norm=grad_clip_norm,
            loss_type=loss_type,
            nll_clamp_log_sigma=nll_clamp_log_sigma,
            aux_delta_loss=aux_delta_loss,
            aux_delta_weight=aux_delta_weight,
            lambda_bias=lambda_bias,
            lambda_cal=lambda_cal,
            cal_b_range=cal_b_range,
        )
        va = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            use_amp=use_amp,
            huber_delta=huber_delta,
            lambda_delta=lambda_delta,
            grad_clip_norm=0.0,
            loss_type=loss_type,
            nll_clamp_log_sigma=nll_clamp_log_sigma,
            aux_delta_loss=aux_delta_loss,
            aux_delta_weight=aux_delta_weight,
            lambda_bias=lambda_bias,
            lambda_cal=lambda_cal,
            cal_b_range=cal_b_range,
        )
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "train_rmse": tr["rmse"],
            "val_rmse": va["rmse"],
            "train_mae": tr["mae"],
            "val_mae": va["mae"],
            "train_bias": tr["bias"],
            "val_bias": va["bias"],
        }
        if "a_mean" in tr:
            row["train_a_mean"] = tr["a_mean"]
        if "a_mean" in va:
            row["val_a_mean"] = va["a_mean"]
        if "b_mean" in tr:
            row["train_b_mean"] = tr["b_mean"]
        if "b_mean" in va:
            row["val_b_mean"] = va["b_mean"]
        if "cal_penalty" in tr:
            row["train_cal_penalty"] = tr["cal_penalty"]
        if "cal_penalty" in va:
            row["val_cal_penalty"] = va["cal_penalty"]
        history.append(row)
        if scheduler is not None:
            if sched_name == "plateau":
                scheduler.step(float(va["rmse"]) if np.isfinite(va["rmse"]) else 1.0e9)
            else:
                scheduler.step()
        cur_lr = float(optimizer.param_groups[0]["lr"])
        logger.info(
            "epoch=%d train_rmse=%.5f val_rmse=%.5f train_mae=%.5f val_mae=%.5f train_bias=%.5f val_bias=%.5f lr=%.6g",
            epoch,
            tr["rmse"],
            va["rmse"],
            tr["mae"],
            va["mae"],
            tr["bias"],
            va["bias"],
            cur_lr,
        )
        if "a_mean" in va or "b_mean" in va:
            logger.info(
                "epoch=%d calibration train_a_mean=%.5f train_b_mean=%.5f val_a_mean=%.5f val_b_mean=%.5f",
                epoch,
                float(tr.get("a_mean", float("nan"))),
                float(tr.get("b_mean", float("nan"))),
                float(va.get("a_mean", float("nan"))),
                float(va.get("b_mean", float("nan"))),
            )
        if "cal_penalty" in va:
            logger.info("epoch=%d val_cal_penalty=%.6f", epoch, float(va["cal_penalty"]))
        if "sigma_mean" in va:
            logger.info("epoch=%d val_sigma_mean=%.5f", epoch, float(va["sigma_mean"]))
        if "gate_entropy" in va:
            logger.info("epoch=%d val_gate_entropy=%.5f", epoch, float(va["gate_entropy"]))
        if "expert_usage" in va:
            usage = [round(float(v), 4) for v in va["expert_usage"]]
            logger.info("epoch=%d val_expert_usage=%s", epoch, usage)

        ckpt_obj = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "feature_names": train_table.feature_names,
            "config": cfg,
            "impute": {
                "medians": imp["medians"],
                "base_feature_names": imp["base_feature_names"],
                "final_feature_names": imp["final_feature_names"],
                "mask_feature_names": imp["mask_feature_names"],
            },
        }
        torch.save(ckpt_obj, ckpt_last)

        if tb_writer is not None:
            tb_writer.add_scalar("train/rmse", tr["rmse"], epoch)
            tb_writer.add_scalar("val/rmse", va["rmse"], epoch)
            tb_writer.add_scalar("train/loss", tr["loss"], epoch)
            tb_writer.add_scalar("val/loss", va["loss"], epoch)

        if np.isfinite(va["rmse"]) and va["rmse"] < best_val_rmse:
            best_val_rmse = va["rmse"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(ckpt_obj, ckpt_best)
        else:
            bad_epochs += 1

        if epoch >= min_epochs and (bad_epochs >= max_bad_epochs or bad_epochs >= patience):
            logger.info("Early stopping at epoch=%d bad_epochs=%d min_epochs=%d", epoch, bad_epochs, min_epochs)
            break

    if not ckpt_best.exists():
        torch.save(torch.load(ckpt_last, map_location="cpu"), ckpt_best)
    best_obj = torch.load(ckpt_best, map_location=device)
    model.load_state_dict(best_obj["model_state_dict"])
    model.eval()

    te = _run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        optimizer=None,
        use_amp=use_amp,
        huber_delta=huber_delta,
        lambda_delta=lambda_delta,
        grad_clip_norm=0.0,
        loss_type=loss_type,
        nll_clamp_log_sigma=nll_clamp_log_sigma,
        aux_delta_loss=aux_delta_loss,
        aux_delta_weight=aux_delta_weight,
        lambda_bias=lambda_bias,
        lambda_cal=lambda_cal,
        cal_b_range=cal_b_range,
    )
    final_metrics = {
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "test_rmse": te["rmse"],
        "test_mae": te["mae"],
        "test_bias": te["bias"],
        "n_train": int(train_table.x.shape[0]),
        "n_val": int(val_table.x.shape[0]),
        "n_test": int(test_table.x.shape[0]),
    }
    save_json(paths["results"] / "metrics.json", final_metrics)
    pd.DataFrame(history).to_csv(paths["results"] / "metrics.csv", index=False)

    split_ranges = {
        "train": _date_range_from_idx(daily_times, split.train_idx),
        "val": _date_range_from_idx(daily_times, split.val_idx),
        "test": _date_range_from_idx(daily_times, split.test_idx),
    }
    save_yaml(paths["results"] / "config_resolved.yaml", cfg)
    _build_run_summary(
        base_dir=base_dir,
        best_epoch=best_epoch,
        best_val_rmse=best_val_rmse,
        feature_names=train_table.feature_names,
        split_info=split_ranges,
    )

    h_lr, w_lr = test_table.grid_shape
    pred_dir = paths["predictions_sample"]
    with torch.no_grad():
        model.eval()
        all_pred = np.zeros((test_table.x.shape[0],), dtype=np.float32)
        for i0 in range(0, test_table.x.shape[0], batch_size):
            i1 = min(test_table.x.shape[0], i0 + batch_size)
            x = torch.from_numpy(test_table.x[i0:i1]).float().to(device)
            all_pred[i0:i1] = model(x)["yhat"].detach().cpu().numpy()

        for d in np.unique(test_table.date_idx)[:3]:
            m = test_table.date_idx == d
            if not np.any(m):
                continue
            y = test_table.y[m]
            cell_idx = test_table.cell_idx[m]
            yp = all_pred[m]
            pred_map = np.full((h_lr * w_lr,), np.nan, dtype=np.float32)
            true_map = np.full((h_lr * w_lr,), np.nan, dtype=np.float32)
            pred_map[cell_idx] = yp
            true_map[cell_idx] = y
            date_str = str(pd.Timestamp(daily_times[int(d)]).date())
            np.save(pred_dir / f"pred_{date_str}.npy", pred_map.reshape(h_lr, w_lr))
            np.save(pred_dir / f"true_{date_str}.npy", true_map.reshape(h_lr, w_lr))

    _write_debug_sample(paths["results"] / "debug_sample.csv", test_table, all_pred, n_rows=200)
    logger.info("Best epoch=%d best val RMSE=%.6f", best_epoch, best_val_rmse)
    logger.info("Output base directory: %s", base_dir.as_posix())
    if tb_writer is not None:
        tb_writer.close()
    print(json.dumps({"base_dir": base_dir.as_posix(), "metrics": final_metrics}, indent=2))


if __name__ == "__main__":
    main()







