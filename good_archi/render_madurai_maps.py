from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import zarr
from matplotlib.path import Path as MplPath
from rasterio.transform import Affine, array_bounds

from dataset_basenet import (
    _aggregate_target_1km,
    _build_neighborhood_features,
    _compute_qc_weight,
    _decode_time_values,
    _prepare_static_features,
    _sanitize_and_convert,
)
from features_basenet import bilinear_resample_2d, build_hr_to_lr_index, infer_nodata_values, sanitize_array
from io_basenet import load_yaml, save_json
from logger_basenet import setup_logging
from model_basenet import BaseNetModel
from qc_basenet import map_qc
from registry_basenet import discover_dataset_paths


ROI_COORDS_LONLAT = [[
    [78.19018990852709, 9.878220339041174],
    [78.18528420047912, 9.882890316485547],
    [78.09810549679275, 9.894348798281609],
    [78.06157864432538, 9.932892703854442],
    [78.02656656110504, 9.94245002321148],
    [78.00260828167308, 9.965541799781803],
    [77.98788548795139, 9.97202781239784],
    [77.98759869039448, 9.972273355756196],
    [77.98926344201777, 9.974277652950137],
    [78.17034512986598, 10.09203200706642],
    [78.33409611393074, 10.299897884952559],
    [78.33446180425422, 10.301538866748283],
    [78.3424560260113, 10.300913203358672],
    [78.3620537189306, 10.301140183092196],
    [78.37500586343197, 10.310994022414183],
    [78.38801590170993, 10.315994815585963],
    [78.39017505117465, 10.315419769552259],
    [78.38542553295602, 10.314175493832316],
    [78.38810333700448, 10.269213359046619],
    [78.41273304358974, 10.212526305038399],
    [78.39622781746723, 10.177726046483423],
    [78.45775072031367, 10.100259541710393],
    [78.45817143938241, 9.967811808760526],
    [78.19018990852709, 9.878220339041174],
]]


def _parse_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _resolve_run_artifacts(run_dir: Path) -> tuple[Path, Path, Path]:
    cfg_candidates = [
        run_dir / "results" / "config_resolved.yaml",
        run_dir / "config_resolved.yaml",
        run_dir / "config.yaml",
    ]
    cfg_path = next((p for p in cfg_candidates if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError(f"Could not find config file in {run_dir.as_posix()}.")

    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path.as_posix()}")

    split_candidates = [run_dir / "results" / "splits.json", run_dir / "splits.json"]
    split_path = next((p for p in split_candidates if p.exists()), None)
    if split_path is None:
        raise FileNotFoundError(f"Could not find splits.json in {run_dir.as_posix()}.")
    return cfg_path, ckpt_path, split_path


def _resolve_dataset_paths(cfg: Dict[str, Any], repo_root: Path, zarr_path: str | None, zarr_30m_path: str | None, logger) -> None:
    ds_cfg = cfg.setdefault("dataset", {})
    p_daily = ds_cfg.get("zarr_path") or ds_cfg.get("madurai_zarr")
    p_30m = ds_cfg.get("zarr_30m_path") or ds_cfg.get("madurai_30m_zarr")
    if p_daily and p_30m and not zarr_path and not zarr_30m_path:
        ds_cfg["zarr_path"] = str(p_daily)
        ds_cfg["zarr_30m_path"] = str(p_30m)
        logger.info("Dataset path resolution branch: config")
        return

    resolved = discover_dataset_paths(repo_root=repo_root, cli_daily=zarr_path, cli_30m=zarr_30m_path)
    ds_cfg["zarr_path"] = resolved["daily"]
    ds_cfg["zarr_30m_path"] = resolved["zarr_30m"]
    logger.info("Dataset path resolution branch: %s", resolved.get("source", "unknown"))


def _split_key_from_name(split_name: str) -> str:
    s = str(split_name).strip().lower()
    if s in {"train", "tr"}:
        return "train_indices"
    if s in {"val", "valid", "validation"}:
        return "val_indices"
    return "test_indices"


def _resolve_date_index(
    *,
    daily_times: pd.DatetimeIndex,
    split_indices: np.ndarray,
    date_idx: Optional[int],
    date_str: Optional[str],
) -> int:
    if date_idx is not None:
        t = int(date_idx)
        if t < 0 or t >= len(daily_times):
            raise ValueError(f"date_idx out of range: {t} (valid: 0..{len(daily_times)-1})")
        return t

    if date_str:
        ts = pd.Timestamp(date_str).normalize()
        dnorm = daily_times.tz_localize(None).normalize() if daily_times.tz is not None else daily_times.normalize()
        match = np.where(dnorm == ts)[0]
        if match.size == 0:
            raise ValueError(f"Date {date_str} not found in daily timeline.")
        return int(match[0])

    if split_indices.size == 0:
        raise ValueError("Selected split has no dates; provide --date_idx or --date.")
    return int(split_indices[0])


def _infer_era5_kind(arr: np.ndarray) -> str:
    finite = np.isfinite(arr)
    if not np.any(finite):
        return "era5_other"
    med = float(np.nanmedian(arr[finite]))
    if 150.0 < med < 400.0:
        return "era5_temp"
    if -80.0 < med < 80.0:
        return "era5_temp"
    return "era5_other"


def _extract_thermal_cols_for_t(
    *,
    t: int,
    modis_arr,
    viirs_arr,
    qc_cfg: Dict[str, Any],
    units_cfg: Dict[str, Any],
    extra_nodata: List[float],
    logger,
) -> Dict[str, np.ndarray]:
    mod_raw = np.asarray(modis_arr[t], dtype=np.float32)
    vii_raw = np.asarray(viirs_arr[t], dtype=np.float32)

    mod_day, _ = _sanitize_and_convert(
        mod_raw[0], zarr_obj=modis_arr, extra_nodata=extra_nodata, name="modis_day", kind="modis_lst", logger=logger, units_cfg=units_cfg
    )
    mod_night_src = mod_raw[1] if mod_raw.shape[0] > 1 else mod_raw[0]
    mod_night, _ = _sanitize_and_convert(
        mod_night_src,
        zarr_obj=modis_arr,
        extra_nodata=extra_nodata,
        name="modis_night",
        kind="modis_lst",
        logger=logger,
        units_cfg=units_cfg,
    )
    mod_qc_day, _ = sanitize_array(
        mod_raw[4] if mod_raw.shape[0] > 4 else np.zeros_like(mod_day),
        infer_nodata_values(modis_arr),
        extra_nodata,
        "modis_qc_day",
    )
    mod_qc_night, _ = sanitize_array(
        mod_raw[5] if mod_raw.shape[0] > 5 else mod_qc_day,
        infer_nodata_values(modis_arr),
        extra_nodata,
        "modis_qc_night",
    )

    vii_day, _ = _sanitize_and_convert(
        vii_raw[0], zarr_obj=viirs_arr, extra_nodata=extra_nodata, name="viirs_day", kind="viirs_lst", logger=logger, units_cfg=units_cfg
    )
    vii_night_src = vii_raw[1] if vii_raw.shape[0] > 1 else vii_raw[0]
    vii_night, _ = _sanitize_and_convert(
        vii_night_src,
        zarr_obj=viirs_arr,
        extra_nodata=extra_nodata,
        name="viirs_night",
        kind="viirs_lst",
        logger=logger,
        units_cfg=units_cfg,
    )
    vii_qc_day, _ = sanitize_array(
        vii_raw[2] if vii_raw.shape[0] > 2 else np.zeros_like(vii_day),
        infer_nodata_values(viirs_arr),
        extra_nodata,
        "viirs_qc_day",
    )
    vii_qc_night, _ = sanitize_array(
        vii_raw[3] if vii_raw.shape[0] > 3 else vii_qc_day,
        infer_nodata_values(viirs_arr),
        extra_nodata,
        "viirs_qc_night",
    )

    qc = map_qc(
        mod_day,
        mod_night,
        vii_day,
        vii_night,
        mod_qc_day,
        mod_qc_night,
        vii_qc_day,
        vii_qc_night,
        unknown_qc_score=float(qc_cfg.get("unknown_qc_score", 0.5)),
    )

    return {
        "modis_day": mod_day.reshape(-1).astype(np.float32),
        "modis_night": mod_night.reshape(-1).astype(np.float32),
        "viirs_day": vii_day.reshape(-1).astype(np.float32),
        "viirs_night": vii_night.reshape(-1).astype(np.float32),
        "modis_valid_day": qc.modis_valid_day.reshape(-1).astype(np.float32),
        "modis_valid_night": qc.modis_valid_night.reshape(-1).astype(np.float32),
        "viirs_valid_day": qc.viirs_valid_day.reshape(-1).astype(np.float32),
        "viirs_valid_night": qc.viirs_valid_night.reshape(-1).astype(np.float32),
        "modis_qc_score_day": qc.modis_qc_score_day.reshape(-1).astype(np.float32),
        "modis_qc_score_night": qc.modis_qc_score_night.reshape(-1).astype(np.float32),
        "viirs_qc_score_day": qc.viirs_qc_score_day.reshape(-1).astype(np.float32),
        "viirs_qc_score_night": qc.viirs_qc_score_night.reshape(-1).astype(np.float32),
    }


def _align_and_impute_features(*, cols: Dict[str, np.ndarray], ckpt: Dict[str, Any], n_cells: int, logger) -> tuple[np.ndarray, List[str]]:
    imp = ckpt.get("impute", {})
    ckpt_feature_names = list(ckpt.get("feature_names", []))

    base_names = list(imp.get("base_feature_names", []))
    if not base_names:
        base_names = [n for n in ckpt_feature_names if not n.endswith("_isnan")]
        seen = set()
        dedup: List[str] = []
        for n in base_names:
            if n not in seen:
                seen.add(n)
                dedup.append(n)
        base_names = dedup

    med = np.asarray(imp.get("medians", []), dtype=np.float32)
    if med.size == 0 or med.shape[0] != len(base_names):
        med = np.zeros((len(base_names),), dtype=np.float32)

    miss = [n for n in base_names if n not in cols]
    if miss:
        logger.warning("Missing %d base features for map inference; filling with NaN then median.", len(miss))

    x_base = np.stack(
        [np.asarray(cols.get(n, np.full((n_cells,), np.nan, dtype=np.float32)), dtype=np.float32) for n in base_names],
        axis=1,
    )
    nan_mask = ~np.isfinite(x_base)
    x_f = x_base.copy()
    for j in range(x_f.shape[1]):
        x_f[nan_mask[:, j], j] = med[j]

    mask_names = list(imp.get("mask_feature_names", []))
    final_names = list(imp.get("final_feature_names", []))
    if mask_names:
        bidx = {n: i for i, n in enumerate(base_names)}
        mask_cols: List[np.ndarray] = []
        for mn in mask_names:
            src = mn.replace("_isnan", "")
            if src in bidx:
                mask_cols.append(nan_mask[:, bidx[src]].astype(np.float32))
            else:
                mask_cols.append(np.ones((n_cells,), dtype=np.float32))
        x_f = np.concatenate([x_f, np.stack(mask_cols, axis=1)], axis=1)

    if not final_names:
        final_names = base_names + mask_names

    if ckpt_feature_names:
        if final_names != ckpt_feature_names:
            idx = {n: i for i, n in enumerate(final_names)}
            if not all(n in idx for n in ckpt_feature_names):
                missing = [n for n in ckpt_feature_names if n not in idx]
                raise RuntimeError(f"Feature alignment failed; missing in built matrix: {missing[:10]}")
            x_f = x_f[:, [idx[n] for n in ckpt_feature_names]]
            final_names = ckpt_feature_names

    if not np.isfinite(x_f).all():
        raise RuntimeError("Non-finite values remain after imputation.")
    return x_f.astype(np.float32), final_names


def _stats(arr: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float32)
    m = np.isfinite(a)
    if not np.any(m):
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {"min": float(np.nanmin(a[m])), "mean": float(np.nanmean(a[m])), "max": float(np.nanmax(a[m]))}


def _transform_list_to_xy(polys_lonlat: Sequence[Sequence[Sequence[float]]], dst_crs: str) -> List[np.ndarray]:
    dst_label = str(dst_crs or "EPSG:4326")
    if dst_label.upper() in {"EPSG:4326", "WGS84"}:
        return [np.asarray(poly, dtype=np.float64) for poly in polys_lonlat]

    tr = None
    try:
        from pyproj import CRS, Transformer  # type: ignore

        src = CRS.from_epsg(4326)
        dst = CRS.from_user_input(dst_label)
        if dst != src:
            tr = Transformer.from_crs(src, dst, always_xy=True)
    except Exception:
        tr = None

    out: List[np.ndarray] = []
    for poly in polys_lonlat:
        arr = np.asarray(poly, dtype=np.float64)
        if tr is not None:
            x, y = tr.transform(arr[:, 0], arr[:, 1])
        else:
            try:
                from rasterio.warp import transform as rio_transform

                x, y = rio_transform("EPSG:4326", dst_label, arr[:, 0].tolist(), arr[:, 1].tolist())
            except Exception:
                x, y = arr[:, 0], arr[:, 1]
        out.append(np.column_stack([x, y]).astype(np.float64))
    return out


def _build_mask_from_polygon_xy(h: int, w: int, aff: Affine, polys_xy: Sequence[np.ndarray]) -> np.ndarray:
    cols = np.arange(w, dtype=np.float64) + 0.5
    rows = np.arange(h, dtype=np.float64) + 0.5
    cc, rr = np.meshgrid(cols, rows)
    x = aff.a * cc + aff.b * rr + aff.c
    y = aff.d * cc + aff.e * rr + aff.f
    pts = np.column_stack([x.ravel(), y.ravel()])
    mask = np.zeros((h * w,), dtype=bool)
    for poly in polys_xy:
        path = MplPath(np.asarray(poly, dtype=np.float64))
        mask |= path.contains_points(pts)
    return mask.reshape(h, w)


def _save_geotiff(path: Path, arr: np.ndarray, crs: str, aff: Affine) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(arr, dtype=np.float32)
    with rasterio.open(
        str(path),
        mode="w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=aff,
        nodata=np.nan,
        compress="deflate",
    ) as ds:
        ds.write(data, 1)


def _plot_map(
    *,
    arr: np.ndarray,
    out_png: Path,
    title: str,
    cmap: str,
    cbar_label: str,
    extent: Tuple[float, float, float, float],
    boundary_xy: Optional[List[np.ndarray]],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, origin="upper")
    if boundary_xy:
        for poly in boundary_xy:
            ax.plot(poly[:, 0], poly[:, 1], color="black", linewidth=1.1)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _as_date_tag(ts: pd.Timestamp, idx: int) -> str:
    if pd.isna(ts):
        return f"date_idx_{idx:05d}"
    return f"{str(ts.date())}_idx_{idx:05d}"


def _build_model(cfg: Dict[str, Any], ckpt: Dict[str, Any], device: torch.device) -> BaseNetModel:
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
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Madurai district maps from BaseNet checkpoint predictions.")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing checkpoints/ and results/")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split used when date is not explicitly provided.")
    parser.add_argument("--date_idx", type=int, default=None, help="Absolute daily time index.")
    parser.add_argument("--date", type=str, default=None, help="Date string YYYY-MM-DD.")
    parser.add_argument("--cell_grid", type=str, default="1km")
    parser.add_argument("--save_geotiff", type=str, default="true")
    parser.add_argument("--zarr_path", type=str, default=None)
    parser.add_argument("--zarr_30m_path", type=str, default=None)
    parser.add_argument("--vmin", type=float, default=20.0, help="LST color minimum for pred/gt.")
    parser.add_argument("--vmax", type=float, default=50.0, help="LST color maximum for pred/gt.")
    parser.add_argument("--scale_mode", type=str, default="fixed", choices=["fixed", "percentile"])
    parser.add_argument("--percentile_low", type=float, default=2.0)
    parser.add_argument("--percentile_high", type=float, default=98.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path, ckpt_path, split_path = _resolve_run_artifacts(run_dir)
    log_path = run_dir / "logs" / "render_madurai_maps.log"
    logger = setup_logging(log_path, logger_name=f"basenet_1km_render_{run_dir.name}")

    cfg = load_yaml(cfg_path)
    repo_root = Path(__file__).resolve().parents[1]
    _resolve_dataset_paths(cfg, repo_root, args.zarr_path, args.zarr_30m_path, logger)

    ds_cfg = cfg["dataset"]
    feature_cfg = cfg.get("features", {})
    qc_cfg = cfg.get("qc", {})
    units_cfg = cfg.get("units", {})
    nodata_cfg = cfg.get("nodata", {})
    extra_nodata = [float(v) for v in nodata_cfg.get("extra_values", [149, -9999, 0, 65535, 32767, -32768, 9999, 1e20])]

    splits = json.loads(split_path.read_text(encoding="utf-8"))
    split_key = _split_key_from_name(args.split)
    split_indices = np.asarray(splits.get(split_key, []), dtype=np.int64)

    root_daily = zarr.open_group(ds_cfg["zarr_path"], mode="r")
    root_30m = zarr.open_group(ds_cfg["zarr_30m_path"], mode="r")
    daily_times = _decode_time_values(root_daily[ds_cfg["time_daily_group"]][:])
    t = _resolve_date_index(daily_times=daily_times, split_indices=split_indices, date_idx=args.date_idx, date_str=args.date)
    if split_indices.size > 0 and t not in set(split_indices.tolist()):
        logger.warning("Selected date_idx=%d is not in split=%s.", t, args.split)

    modis_group_path = ds_cfg["modis_group"]
    modis_g = root_daily[modis_group_path]
    modis_arr = modis_g["data"]
    viirs_arr = root_daily[ds_cfg["viirs_group"]]["data"]
    h_lr, w_lr = int(modis_arr.shape[-2]), int(modis_arr.shape[-1])
    n_cells = h_lr * w_lr

    landsat_arr = root_30m[ds_cfg["landsat_group"]]["data"]
    landsat_valid_arr = root_30m[ds_cfg["landsat_group"]]["valid"]
    h_hr, w_hr = int(landsat_arr.shape[-2]), int(landsat_arr.shape[-1])
    cell_index = build_hr_to_lr_index(h_hr, w_hr, h_lr, w_lr)

    static = _prepare_static_features(root_30m, ds_cfg, feature_cfg, cell_index, n_cells, logger, extra_nodata)
    cols = _extract_thermal_cols_for_t(
        t=t,
        modis_arr=modis_arr,
        viirs_arr=viirs_arr,
        qc_cfg=qc_cfg,
        units_cfg=units_cfg,
        extra_nodata=extra_nodata,
        logger=logger,
    )

    use_era5_daily = ds_cfg.get("era5_group_daily") in root_daily
    if use_era5_daily:
        era5_arr = root_daily[ds_cfg["era5_group_daily"]]["data"]
        era5_raw = np.asarray(era5_arr[t], dtype=np.float32)
        for ch in range(era5_raw.shape[0]):
            kind = _infer_era5_kind(era5_raw[ch])
            s, _ = _sanitize_and_convert(
                era5_raw[ch],
                zarr_obj=era5_arr,
                extra_nodata=extra_nodata,
                name=f"era5_band_{ch+1}",
                kind=kind,
                logger=logger,
                units_cfg=units_cfg,
            )
            cols[f"era5_band_{ch+1}"] = bilinear_resample_2d(s, h_lr, w_lr).reshape(-1).astype(np.float32)
    else:
        era5_arr = root_30m[ds_cfg["era5_group_30m"]]["data"]
        era5_valid = root_30m[ds_cfg["era5_group_30m"]]["valid"]
        era5_raw = np.asarray(era5_arr[t], dtype=np.float32)
        era5v = np.asarray(era5_valid[t, 0], dtype=np.float32) > 0
        for ch in range(era5_raw.shape[0]):
            kind = _infer_era5_kind(era5_raw[ch])
            s, _ = _sanitize_and_convert(
                era5_raw[ch],
                zarr_obj=era5_arr,
                extra_nodata=extra_nodata,
                name=f"era5_band_{ch+1}",
                kind=kind,
                logger=logger,
                units_cfg=units_cfg,
            )
            v = np.where(era5v, s, np.nan)
            idx = cell_index.ravel()
            finite = np.isfinite(v).ravel()
            vals = np.where(finite, v.ravel(), 0.0)
            cnt = np.bincount(idx, weights=finite.astype(np.float32), minlength=n_cells).astype(np.float32)
            sm = np.bincount(idx, weights=vals.astype(np.float32), minlength=n_cells).astype(np.float32)
            out = np.full((n_cells,), np.nan, dtype=np.float32)
            np.divide(sm, cnt, out=out, where=cnt > 0)
            cols[f"era5_band_{ch+1}"] = out

    dt = daily_times[t]
    if pd.isna(dt):
        raise RuntimeError(f"Invalid timestamp at date_idx={t}")
    doy = float(pd.Timestamp(dt).dayofyear)
    doy_rad = 2.0 * np.pi * (doy / 365.0)
    cols["doy_sin"] = np.full((n_cells,), np.sin(doy_rad), dtype=np.float32)
    cols["doy_cos"] = np.full((n_cells,), np.cos(doy_rad), dtype=np.float32)

    for k, v in static.items():
        cols[k] = np.asarray(v, dtype=np.float32)

    if bool(feature_cfg.get("add_engineered", False)):
        cols["modis_dnd"] = (cols["modis_day"] - cols["modis_night"]).astype(np.float32)
        cols["viirs_dnd"] = (cols["viirs_day"] - cols["viirs_night"]).astype(np.float32)
        cols["m_minus_v_day"] = (cols["modis_day"] - cols["viirs_day"]).astype(np.float32)
        cols["m_minus_v_night"] = (cols["modis_night"] - cols["viirs_night"]).astype(np.float32)
        if "era5_band_1" in cols and "era5_band_2" in cols:
            cols["era5_delta_1_2"] = (cols["era5_band_1"] - cols["era5_band_2"]).astype(np.float32)
        if "era5_band_3" in cols and "era5_band_4" in cols:
            cols["era5_delta_3_4"] = (cols["era5_band_3"] - cols["era5_band_4"]).astype(np.float32)
        if "era5_band_5" in cols and "era5_band_6" in cols:
            cols["era5_delta_5_6"] = (cols["era5_band_5"] - cols["era5_band_6"]).astype(np.float32)

    if bool(feature_cfg.get("use_neighborhood_stats", False)):
        cache_root = Path(feature_cfg.get("neighborhood_cache", "good_archi/cache/neighborhood"))
        if not cache_root.is_absolute():
            cache_root = Path.cwd() / cache_root
        nb_cols, from_cache, created_any, dt_sec = _build_neighborhood_features(
            t=t,
            cols=cols,
            feature_cfg=feature_cfg,
            cache_root=cache_root,
            h_lr=h_lr,
            w_lr=w_lr,
            logger=logger,
        )
        if created_any:
            cols.update(nb_cols)
            logger.info("Neighborhood features loaded for date_idx=%d (cache=%s, sec=%.2f).", t, from_cache, dt_sec)
        else:
            logger.warning("Neighborhood features requested but none created for date_idx=%d.", t)

    if bool(feature_cfg.get("add_lags", False)):
        lag_days = [int(v) for v in feature_cfg.get("lag_days", [1]) if int(v) > 0]
        lag_columns = [str(v) for v in feature_cfg.get("lag_columns", ["modis_day", "modis_night", "viirs_day", "viirs_night"])]
        add_lag_valid_masks = bool(feature_cfg.get("add_lag_valid_masks", True))
        for lag in sorted(set(lag_days)):
            tt = int(t - lag)
            lag_src: Dict[str, np.ndarray] = {}
            if 0 <= tt < int(modis_arr.shape[0]):
                lag_src = _extract_thermal_cols_for_t(
                    t=tt,
                    modis_arr=modis_arr,
                    viirs_arr=viirs_arr,
                    qc_cfg=qc_cfg,
                    units_cfg=units_cfg,
                    extra_nodata=extra_nodata,
                    logger=logger,
                )
            for c in lag_columns:
                lname = f"{c}_lag{lag}"
                if c in lag_src:
                    cols[lname] = lag_src[c].astype(np.float32)
                    vmask = np.isfinite(cols[lname]).astype(np.float32)
                else:
                    cols[lname] = np.full((n_cells,), np.nan, dtype=np.float32)
                    vmask = np.zeros((n_cells,), dtype=np.float32)
                if add_lag_valid_masks:
                    cols[f"{lname}_valid"] = vmask

    ls_raw = np.asarray(landsat_arr[t, 0], dtype=np.float32)
    ls_s, _ = _sanitize_and_convert(
        ls_raw,
        zarr_obj=landsat_arr,
        extra_nodata=extra_nodata,
        name="landsat_target",
        kind="landsat_lst",
        logger=logger,
        units_cfg=units_cfg,
    )
    ls_valid = np.asarray(landsat_valid_arr[t, 0], dtype=np.float32) > 0
    ls_valid &= np.isfinite(ls_s)
    y, vf = _aggregate_target_1km(ls_s, ls_valid, cell_index, n_cells)
    y = y.astype(np.float32)
    vf = np.clip(vf.astype(np.float32), 0.0, 1.0)

    qc_weight = _compute_qc_weight(cols, cfg) if bool(cfg.get("training", {}).get("qc_weighting", False)) else np.ones((n_cells,), dtype=np.float32)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    x, final_names = _align_and_impute_features(cols=cols, ckpt=ckpt, n_cells=n_cells, logger=logger)
    logger.info("Inference feature matrix built: n_cells=%d n_features=%d", n_cells, x.shape[1])
    if final_names != list(ckpt["feature_names"]):
        raise RuntimeError("Final inference feature order mismatch with checkpoint feature_names.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg, ckpt, device)

    bs = int(cfg.get("training", {}).get("batch_size", 4096))
    pred = np.full((n_cells,), np.nan, dtype=np.float32)
    usable = np.zeros((n_cells,), dtype=bool)
    with torch.no_grad():
        for i0 in range(0, n_cells, bs):
            i1 = min(n_cells, i0 + bs)
            xb = torch.from_numpy(x[i0:i1]).float().to(device)
            out = model(xb)
            yp = out["yhat"].detach().cpu().numpy().astype(np.float32)
            ub = out["usable"].detach().cpu().numpy() > 0.5
            pred[i0:i1] = yp
            usable[i0:i1] = ub
    pred[~usable] = np.nan

    pred_map = pred.reshape(h_lr, w_lr)
    gt_map = y.reshape(h_lr, w_lr)
    vf_map = vf.reshape(h_lr, w_lr)
    qcw_map = qc_weight.reshape(h_lr, w_lr)
    err_map = pred_map - gt_map

    modis_meta = dict(modis_g.attrs)
    transform_vals = modis_meta.get("transform", None)
    crs_str = modis_meta.get("crs", None)
    if not transform_vals or len(transform_vals) < 6:
        raise RuntimeError("Could not read 1km transform from modis group attrs.")
    aff = Affine(*[float(v) for v in transform_vals[:6]])
    west, south, east, north = array_bounds(h_lr, w_lr, aff)
    extent = (west, east, south, north)

    boundary_xy = _transform_list_to_xy(ROI_COORDS_LONLAT, str(crs_str) if crs_str else "EPSG:4326")
    roi_mask = _build_mask_from_polygon_xy(h_lr, w_lr, aff, boundary_xy)
    if np.any(roi_mask):
        pred_map = np.where(roi_mask, pred_map, np.nan)
        gt_map = np.where(roi_mask, gt_map, np.nan)
        err_map = np.where(roi_mask, err_map, np.nan)
        vf_map = np.where(roi_mask, vf_map, np.nan)
        qcw_map = np.where(roi_mask, qcw_map, np.nan)
    else:
        logger.warning("ROI mask empty on 1km grid; maps will not be masked to district.")

    date_tag = _as_date_tag(pd.Timestamp(dt), t)
    out_dir = run_dir / "results" / "maps" / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scale_mode == "percentile":
        stack = np.concatenate([pred_map[np.isfinite(pred_map)], gt_map[np.isfinite(gt_map)]])
        if stack.size > 0:
            vmin = float(np.nanpercentile(stack, float(args.percentile_low)))
            vmax = float(np.nanpercentile(stack, float(args.percentile_high)))
        else:
            vmin = float(args.vmin)
            vmax = float(args.vmax)
    else:
        vmin = float(args.vmin)
        vmax = float(args.vmax)

    ev = err_map[np.isfinite(err_map)]
    emax = max(1.0, float(np.nanpercentile(np.abs(ev), 98))) if ev.size > 0 else 5.0

    date_label = str(pd.Timestamp(dt).date())
    _plot_map(
        arr=pred_map,
        out_png=out_dir / "pred.png",
        title=f"Madurai District - Predicted LST - {date_label}",
        cmap="inferno",
        cbar_label="LST (°C)",
        extent=extent,
        boundary_xy=boundary_xy,
        vmin=vmin,
        vmax=vmax,
    )
    _plot_map(
        arr=gt_map,
        out_png=out_dir / "gt.png",
        title=f"Madurai District - Landsat 1km LST - {date_label}",
        cmap="inferno",
        cbar_label="LST (°C)",
        extent=extent,
        boundary_xy=boundary_xy,
        vmin=vmin,
        vmax=vmax,
    )
    _plot_map(
        arr=err_map,
        out_png=out_dir / "err.png",
        title=f"Madurai District - Error (Pred - GT) - {date_label}",
        cmap="coolwarm",
        cbar_label="Error (°C)",
        extent=extent,
        boundary_xy=boundary_xy,
        vmin=-emax,
        vmax=emax,
    )
    _plot_map(
        arr=vf_map,
        out_png=out_dir / "valid_frac.png",
        title=f"Madurai District - Landsat Valid Fraction - {date_label}",
        cmap="viridis",
        cbar_label="Valid Fraction",
        extent=extent,
        boundary_xy=boundary_xy,
        vmin=0.0,
        vmax=1.0,
    )
    _plot_map(
        arr=qcw_map,
        out_png=out_dir / "qc_weight.png",
        title=f"Madurai District - QC Weight - {date_label}",
        cmap="viridis",
        cbar_label="QC Weight",
        extent=extent,
        boundary_xy=boundary_xy,
        vmin=0.0,
        vmax=1.0,
    )

    if _parse_bool(args.save_geotiff):
        _save_geotiff(out_dir / "pred.tif", pred_map, str(crs_str), aff)
        _save_geotiff(out_dir / "gt.tif", gt_map, str(crs_str), aff)
        _save_geotiff(out_dir / "err.tif", err_map, str(crs_str), aff)
        _save_geotiff(out_dir / "valid_frac.tif", vf_map, str(crs_str), aff)
        _save_geotiff(out_dir / "qc_weight.tif", qcw_map, str(crs_str), aff)

    valid_eval = np.isfinite(pred_map) & np.isfinite(gt_map)
    if np.any(valid_eval):
        rmse_day = float(np.sqrt(np.nanmean((pred_map[valid_eval] - gt_map[valid_eval]) ** 2)))
        mae_day = float(np.nanmean(np.abs(pred_map[valid_eval] - gt_map[valid_eval])))
        bias_day = float(np.nanmean(pred_map[valid_eval] - gt_map[valid_eval]))
    else:
        rmse_day = float("nan")
        mae_day = float("nan")
        bias_day = float("nan")

    stats = {
        "date_idx": int(t),
        "date": date_label,
        "split": str(args.split),
        "n_cells": int(n_cells),
        "n_pred_finite": int(np.sum(np.isfinite(pred_map))),
        "n_gt_finite": int(np.sum(np.isfinite(gt_map))),
        "n_eval": int(np.sum(valid_eval)),
        "rmse": rmse_day,
        "mae": mae_day,
        "bias": bias_day,
        "pred_stats": _stats(pred_map),
        "gt_stats": _stats(gt_map),
        "err_stats": _stats(err_map),
        "valid_frac_stats": _stats(vf_map),
        "qc_weight_stats": _stats(qcw_map),
        "paths": {
            "pred_png": (out_dir / "pred.png").as_posix(),
            "gt_png": (out_dir / "gt.png").as_posix(),
            "err_png": (out_dir / "err.png").as_posix(),
            "valid_frac_png": (out_dir / "valid_frac.png").as_posix(),
            "qc_weight_png": (out_dir / "qc_weight.png").as_posix(),
            "pred_tif": (out_dir / "pred.tif").as_posix(),
            "gt_tif": (out_dir / "gt.tif").as_posix(),
            "err_tif": (out_dir / "err.tif").as_posix(),
            "valid_frac_tif": (out_dir / "valid_frac.tif").as_posix(),
            "qc_weight_tif": (out_dir / "qc_weight.tif").as_posix(),
        },
    }
    save_json(out_dir / "stats.json", stats)
    logger.info("Saved Madurai maps to: %s", out_dir.as_posix())
    print(json.dumps({"run_dir": run_dir.as_posix(), "maps_dir": out_dir.as_posix(), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
