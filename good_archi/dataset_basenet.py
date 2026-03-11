from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time

import numpy as np
import pandas as pd
import zarr

from features_basenet import (
    aggregate_mean_by_index,
    bilinear_resample_2d,
    build_hr_to_lr_index,
    ensure_celsius,
    infer_nodata_values,
    sanitize_array,
)
from qc_basenet import map_qc

TARGET_CACHE_VERSION = 2


@dataclass
class BaseNetTable:
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    date_idx: np.ndarray
    cell_idx: np.ndarray
    feature_names: List[str]
    dates: pd.DatetimeIndex
    grid_shape: Tuple[int, int]
    vf: np.ndarray | None = None
    qc_weight: np.ndarray | None = None
    debug_info: Dict[str, Any] = field(default_factory=dict)


def _decode_time_values(vals: np.ndarray) -> pd.DatetimeIndex:
    raw = [v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v) for v in vals.tolist()]
    return pd.to_datetime(raw, format="%Y_%m_%d", errors="coerce")


def _agg_stats(vals: np.ndarray) -> Dict[str, float]:
    v = np.asarray(vals, dtype=np.float32)
    finite = np.isfinite(v)
    if not np.any(finite):
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan"), "median": float("nan")}
    x = v[finite]
    return {
        "min": float(np.nanmin(x)),
        "mean": float(np.nanmean(x)),
        "max": float(np.nanmax(x)),
        "median": float(np.nanmedian(x)),
    }


def _sanitize_and_convert(
    arr2d: np.ndarray,
    *,
    zarr_obj: Any,
    extra_nodata: List[float],
    name: str,
    kind: str,
    logger,
    units_cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    nodata = infer_nodata_values(zarr_obj)
    s, valid = sanitize_array(arr2d, nodata, extra_nodata, name)
    if np.any(np.isfinite(s)):
        s = ensure_celsius(
            s,
            name=name,
            kind=kind,
            log_fn=logger.info,
            attrs=dict(getattr(zarr_obj, "attrs", {})),
            modis_scale_auto=bool(units_cfg.get("modis_scale_auto", True)),
            viirs_scale_auto=bool(units_cfg.get("viirs_scale_auto", True)),
            assume_era5_kelvin=str(units_cfg.get("assume_era5_kelvin", "auto")),
        )
    return s, valid


def _infer_era5_kind(arr: np.ndarray) -> str:
    finite = np.isfinite(arr)
    if not np.any(finite):
        return "era5_other"
    med = float(np.nanmedian(arr[finite]))
    if med > 150.0 and med < 400.0:
        return "era5_temp"
    if med > -80.0 and med < 80.0:
        return "era5_temp"
    return "era5_other"


def _prepare_static_features(
    root_30m: zarr.Group,
    ds_cfg: Dict,
    feature_cfg: Dict,
    cell_index: np.ndarray,
    n_cells: int,
    logger,
    extra_nodata: List[float],
) -> Dict[str, np.ndarray]:
    static: Dict[str, np.ndarray] = {}

    if feature_cfg.get("use_dem", True):
        dem_g = root_30m[ds_cfg["dem_group"]]
        dem = np.asarray(dem_g["data"][0], dtype=np.float32)
        dem_valid = np.asarray(dem_g["valid"][0, 0], dtype=np.float32) > 0
        dem_means = []
        for ch in range(dem.shape[0]):
            dch, _ = sanitize_array(dem[ch], infer_nodata_values(dem_g["data"]), extra_nodata, f"dem_band_{ch+1}")
            avg, _ = aggregate_mean_by_index(dch, cell_index, n_cells, valid_mask=dem_valid)
            dem_means.append(avg.astype(np.float32))
        if dem.shape[0] >= 3:
            static["elev_mean"] = dem_means[0]
            static["slope_mean"] = dem_means[1]
            asp = np.deg2rad(dem[2].astype(np.float32))
            a_sin, _ = aggregate_mean_by_index(np.sin(asp), cell_index, n_cells, valid_mask=dem_valid)
            a_cos, _ = aggregate_mean_by_index(np.cos(asp), cell_index, n_cells, valid_mask=dem_valid)
            static["aspect_sin_mean"] = a_sin.astype(np.float32)
            static["aspect_cos_mean"] = a_cos.astype(np.float32)
        else:
            for i, arr in enumerate(dem_means):
                static[f"dem_band_{i+1}_mean"] = arr

    if feature_cfg.get("use_landcover", True):
        max_classes = int(feature_cfg.get("max_landcover_classes", 12))
        for group_key, prefix in (
            (ds_cfg["worldcover_group"], "worldcover"),
            (ds_cfg["dynamic_world_group"], "dynamic_world"),
        ):
            if group_key not in root_30m:
                logger.warning("Static group missing: %s", group_key)
                continue
            g = root_30m[group_key]
            arr = np.asarray(g["data"][0, 0], dtype=np.float32)
            arr, _ = sanitize_array(arr, infer_nodata_values(g["data"]), extra_nodata, f"{prefix}_classes")
            valid = np.asarray(g["valid"][0, 0], dtype=np.float32) > 0
            cls_safe = np.where(np.isfinite(arr), np.rint(arr), -9999.0)
            cls = np.where(valid, cls_safe, -9999.0).astype(np.int32)
            finite_cls = cls[cls >= 0]
            if finite_cls.size == 0:
                continue
            uniq = np.unique(finite_cls)[:max_classes]
            total_per_cell = np.bincount(cell_index.ravel(), weights=valid.ravel().astype(np.float32), minlength=n_cells)
            for c in uniq.tolist():
                mask = (cls == int(c)).astype(np.float32)
                cnt = np.bincount(cell_index.ravel(), weights=mask.ravel(), minlength=n_cells).astype(np.float32)
                frac = np.zeros((n_cells,), dtype=np.float32)
                np.divide(cnt, total_per_cell, out=frac, where=total_per_cell > 0)
                static[f"{prefix}_frac_{int(c)}"] = np.clip(frac, 0.0, 1.0).astype(np.float32)

    return static


def _aggregate_target_1km(
    landsat_data: np.ndarray,
    landsat_valid: np.ndarray,
    cell_index: np.ndarray,
    n_cells: int,
) -> Tuple[np.ndarray, np.ndarray]:
    y, cnt = aggregate_mean_by_index(landsat_data, cell_index, n_cells, valid_mask=landsat_valid)
    total = np.bincount(cell_index.ravel(), minlength=n_cells).astype(np.float32)
    frac = np.zeros((n_cells,), dtype=np.float32)
    np.divide(cnt, total, out=frac, where=total > 0)
    y = np.where(cnt > 0, y, np.nan).astype(np.float32)
    return y, frac


def _neighbor_mean_std_valid(
    vec: np.ndarray,
    h: int,
    w: int,
    kernel: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = int(kernel)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    pad = k // 2
    arr = np.asarray(vec, dtype=np.float32).reshape(h, w)
    val = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    msk = np.isfinite(arr).astype(np.float32)
    pv = np.pad(val, ((pad, pad), (pad, pad)), mode="reflect")
    pm = np.pad(msk, ((pad, pad), (pad, pad)), mode="reflect")
    s = np.zeros((h, w), dtype=np.float32)
    c = np.zeros((h, w), dtype=np.float32)
    s2 = np.zeros((h, w), dtype=np.float32)
    for di in range(k):
        for dj in range(k):
            vv = pv[di : di + h, dj : dj + w]
            mm = pm[di : di + h, dj : dj + w]
            s += vv * mm
            s2 += (vv * vv) * mm
            c += mm
    mean = np.full((h, w), np.nan, dtype=np.float32)
    var = np.full((h, w), np.nan, dtype=np.float32)
    ok = c > 0
    mean[ok] = s[ok] / c[ok]
    var[ok] = np.maximum(0.0, (s2[ok] / c[ok]) - (mean[ok] * mean[ok]))
    std = np.sqrt(np.where(np.isfinite(var), var, np.nan)).astype(np.float32)
    valid_frac = (c / float(k * k)).astype(np.float32)
    return (
        mean.reshape(-1).astype(np.float32),
        std.reshape(-1).astype(np.float32),
        valid_frac.reshape(-1).astype(np.float32),
    )


def _build_neighborhood_features(
    *,
    t: int,
    cols: Dict[str, np.ndarray],
    feature_cfg: Dict,
    cache_root: Path,
    h_lr: int,
    w_lr: int,
    logger,
) -> tuple[Dict[str, np.ndarray], bool, bool, float]:
    if not bool(feature_cfg.get("use_neighborhood_stats", False)):
        return {}, False, False, 0.0
    nb_cols = [str(v) for v in feature_cfg.get("neighborhood_columns", ["modis_day", "viirs_day"])]
    nb_stats = [str(v).lower() for v in feature_cfg.get("neighborhood_stats", ["mean", "std"])]
    kernel = int(feature_cfg.get("neighborhood_kernel", 3))
    if not nb_cols:
        return {}, False, False, 0.0
    cache_root.mkdir(parents=True, exist_ok=True)
    tag_cols = "_".join(nb_cols)
    tag_stats = "_".join(nb_stats)
    cache_path = cache_root / f"t{int(t):05d}_k{kernel}_{tag_cols}_{tag_stats}.npz"

    t0 = time.perf_counter()
    if cache_path.exists():
        obj = np.load(cache_path)
        out = {k: obj[k].astype(np.float32) for k in obj.files}
        dt = time.perf_counter() - t0
        return out, True, bool(out), dt

    out: Dict[str, np.ndarray] = {}
    for c in nb_cols:
        if c not in cols:
            logger.warning("Neighborhood source column missing: %s", c)
            continue
        m, s, vf = _neighbor_mean_std_valid(cols[c], h_lr, w_lr, kernel=kernel)
        suffix = f"{kernel}"
        if "mean" in nb_stats:
            out[f"{c}_nb_mean{suffix}"] = m
        if "std" in nb_stats:
            out[f"{c}_nb_std{suffix}"] = s
        out[f"{c}_nb_valid"] = vf

    if out:
        np.savez_compressed(cache_path, **{k: v.astype(np.float32) for k, v in out.items()})
    dt = time.perf_counter() - t0
    return out, False, bool(out), dt


def _apply_lag_features(table: BaseNetTable, cfg: Dict, logger) -> BaseNetTable:
    fcfg = cfg.get("features", {})
    if not bool(fcfg.get("add_lags", False)):
        return table
    lag_days = [int(v) for v in fcfg.get("lag_days", [1])]
    lag_days = [v for v in sorted(set(lag_days)) if v > 0]
    if not lag_days:
        return table

    base_cols = [str(v) for v in fcfg.get("lag_columns", ["modis_day", "modis_night", "viirs_day", "viirs_night"])]
    add_lag_valid_masks = bool(fcfg.get("add_lag_valid_masks", True))
    missing = [c for c in base_cols if c not in table.feature_names]
    if missing:
        logger.warning("Lag feature skipped; missing thermal columns: %s", ", ".join(missing))
        return table

    df = pd.DataFrame(
        {
            "row_idx": np.arange(table.x.shape[0], dtype=np.int64),
            "cell_id": table.cell_idx.astype(np.int64),
            "date_idx": table.date_idx.astype(np.int64),
        }
    )
    for c in base_cols:
        df[c] = table.x[:, table.feature_names.index(c)]
    df = df.sort_values(["cell_id", "date_idx"], kind="mergesort")

    lag_names: List[str] = []
    lag_arrays: List[np.ndarray] = []
    lag_valid_names: List[str] = []
    lag_valid_arrays: List[np.ndarray] = []
    for lag in lag_days:
        for c in base_cols:
            lname = f"{c}_lag{lag}"
            shifted = df.groupby("cell_id", sort=False)[c].shift(lag).to_numpy(dtype=np.float32)
            lag_names.append(lname)
            lag_arrays.append(shifted)
            if add_lag_valid_masks:
                vname = f"{lname}_valid"
                lag_valid_names.append(vname)
                lag_valid_arrays.append(np.isfinite(shifted).astype(np.float32))
    for name, arr in zip(lag_names, lag_arrays):
        df[name] = arr

    inv = np.empty(df.shape[0], dtype=np.int64)
    inv[df["row_idx"].to_numpy(dtype=np.int64)] = np.arange(df.shape[0], dtype=np.int64)
    lag_mat = np.stack([df[n].to_numpy(dtype=np.float32)[inv] for n in lag_names], axis=1)
    add_blocks = [lag_mat]
    add_names = lag_names[:]
    if lag_valid_arrays:
        lag_valid_mat = np.stack([np.asarray(a, dtype=np.float32)[inv] for a in lag_valid_arrays], axis=1)
        add_blocks.append(lag_valid_mat)
        add_names.extend(lag_valid_names)
    logger.info(
        "Lag features added lag_days=%s columns=%s count=%d valid_mask_count=%d",
        lag_days,
        ",".join(base_cols),
        len(lag_names),
        len(lag_valid_names),
    )

    return BaseNetTable(
        x=np.concatenate([table.x] + add_blocks, axis=1).astype(np.float32),
        y=table.y,
        w=table.w,
        vf=table.vf,
        qc_weight=table.qc_weight,
        date_idx=table.date_idx,
        cell_idx=table.cell_idx,
        feature_names=table.feature_names + add_names,
        dates=table.dates,
        grid_shape=table.grid_shape,
        debug_info=table.debug_info,
    )


def _compute_qc_weight(cols: Dict[str, np.ndarray], cfg: Dict) -> np.ndarray:
    tcfg = cfg.get("training", {})
    qcfg = cfg.get("qc", {})
    qmin = float(tcfg.get("qc_weight_min", 0.2))
    qpow = float(tcfg.get("qc_weight_power", 1.0))
    qpow = max(0.1, qpow)
    unknown = float(qcfg.get("unknown_qc_score", 0.5))
    mod_vd = cols.get("modis_valid_day", np.zeros_like(cols["modis_day"]))
    mod_vn = cols.get("modis_valid_night", np.zeros_like(cols["modis_day"]))
    vii_vd = cols.get("viirs_valid_day", np.zeros_like(cols["modis_day"]))
    vii_vn = cols.get("viirs_valid_night", np.zeros_like(cols["modis_day"]))
    cand = [
        (cols.get("modis_qc_score_day"), mod_vd > 0),
        (cols.get("modis_qc_score_night"), mod_vn > 0),
        (cols.get("viirs_qc_score_day"), vii_vd > 0),
        (cols.get("viirs_qc_score_night"), vii_vn > 0),
    ]
    num = np.zeros_like(cols["modis_day"], dtype=np.float32)
    den = np.zeros_like(cols["modis_day"], dtype=np.float32)
    for q, m in cand:
        if q is None:
            continue
        qf = np.asarray(q, dtype=np.float32)
        mm = np.asarray(m, dtype=bool) & np.isfinite(qf)
        num[mm] += np.clip(qf[mm], 0.0, 1.0)
        den[mm] += 1.0
    out = np.full_like(num, fill_value=np.clip(unknown, 0.0, 1.0), dtype=np.float32)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    out = np.clip(out, qmin, 1.0).astype(np.float32)
    out = np.power(out, qpow, dtype=np.float32)
    return out


def build_basenet_table(
    *,
    cfg: Dict,
    split_date_indices: np.ndarray,
    logger,
    split_role: str = "train",
) -> BaseNetTable:
    ds_cfg = cfg["dataset"]
    feature_cfg = cfg["features"]
    qc_cfg = cfg["qc"]
    units_cfg = cfg.get("units", {})
    nodata_cfg = cfg.get("nodata", {})
    extra_nodata = [float(v) for v in nodata_cfg.get("extra_values", [149, -9999, 0, 65535, 32767, -32768, 9999])]

    fallback_min_frac = float(cfg["training"].get("min_valid_frac", ds_cfg.get("min_valid_frac", 0.6)))
    min_valid_frac_train = float(ds_cfg.get("min_valid_frac_train", fallback_min_frac))
    min_valid_frac_eval = float(ds_cfg.get("min_valid_frac_eval", fallback_min_frac))
    split_role_l = str(split_role).lower()
    if split_role_l in {"train", "val", "validation"}:
        min_valid_frac = min_valid_frac_train
    else:
        min_valid_frac = min_valid_frac_eval
    cache_dir = Path(ds_cfg.get("cache_dir", "good_archi/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    nb_cache_dir_cfg = feature_cfg.get("neighborhood_cache", str(cache_dir / "neighborhood"))
    nb_cache_dir = Path(nb_cache_dir_cfg)
    if not nb_cache_dir.is_absolute():
        nb_cache_dir = Path.cwd() / nb_cache_dir
    nb_cache_dir.mkdir(parents=True, exist_ok=True)

    root_daily = zarr.open_group(ds_cfg["zarr_path"], mode="r")
    root_30m = zarr.open_group(ds_cfg["zarr_30m_path"], mode="r")

    modis_g = root_daily[ds_cfg["modis_group"]]
    viirs_g = root_daily[ds_cfg["viirs_group"]]
    modis_arr = modis_g["data"]
    viirs_arr = viirs_g["data"]
    daily_times = _decode_time_values(root_daily[ds_cfg["time_daily_group"]][:])
    h_lr, w_lr = int(modis_arr.shape[-2]), int(modis_arr.shape[-1])
    n_cells = h_lr * w_lr

    landsat_g = root_30m[ds_cfg["landsat_group"]]
    landsat_arr = landsat_g["data"]
    landsat_valid_arr = landsat_g["valid"]
    h_hr, w_hr = int(landsat_arr.shape[-2]), int(landsat_arr.shape[-1])
    cell_index = build_hr_to_lr_index(h_hr, w_hr, h_lr, w_lr)

    static = _prepare_static_features(root_30m, ds_cfg, feature_cfg, cell_index, n_cells, logger, extra_nodata)

    use_era5_daily = ds_cfg.get("era5_group_daily") in root_daily
    if use_era5_daily:
        era5_g = root_daily[ds_cfg["era5_group_daily"]]
        era5_daily_arr = era5_g["data"]
    else:
        era5_g = root_30m[ds_cfg["era5_group_30m"]]
        era5_30m_arr = era5_g["data"]
        era5_30m_valid = era5_g["valid"]
        logger.warning("Daily ERA5 group missing; using 30m ERA5 aggregation.")

    rows_x: List[np.ndarray] = []
    rows_y: List[np.ndarray] = []
    rows_w: List[np.ndarray] = []
    rows_date: List[np.ndarray] = []
    rows_cell: List[np.ndarray] = []
    feature_names: List[str] | None = None
    n_total_cells = 0
    n_kept_cells = 0
    both_invalid_count = 0
    modis_only_count = 0
    viirs_only_count = 0
    both_valid_count = 0
    raw_149_count = {"modis_day": 0, "viirs_day": 0, "landsat": 0}
    post_149_count = {"modis_day": 0, "viirs_day": 0, "landsat": 0}
    y_all = []
    vf_all = []
    qcw_all = []
    engineered_feature_names: List[str] = []
    era_delta_names: List[str] = []
    neighborhood_created = False
    neighborhood_added_count = 0
    neighborhood_cache_hits = 0
    neighborhood_build_sec = 0.0
    dropped_by_valid_frac = 0
    dropped_nonfinite_or_unusable = 0
    rows_vf: List[np.ndarray] = []
    rows_qcw: List[np.ndarray] = []
    vf_power = float(cfg.get("training", {}).get("valid_frac_power", 1.0))
    vf_power = max(0.1, vf_power)

    logger.info("Split role=%s using min_valid_frac=%.3f", split_role_l, min_valid_frac)

    for t in split_date_indices.tolist():
        t = int(t)
        mod_raw = np.asarray(modis_arr[t], dtype=np.float32)
        vii_raw = np.asarray(viirs_arr[t], dtype=np.float32)
        raw_149_count["modis_day"] += int(np.isclose(mod_raw[0], 149.0).sum())
        raw_149_count["viirs_day"] += int(np.isclose(vii_raw[0], 149.0).sum())

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

        post_149_count["modis_day"] += int(np.isclose(np.nan_to_num(mod_day, nan=-9999.0), 149.0).sum())
        post_149_count["viirs_day"] += int(np.isclose(np.nan_to_num(vii_day, nan=-9999.0), 149.0).sum())

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

        mod_day_f = mod_day.reshape(-1).astype(np.float32)
        mod_night_f = mod_night.reshape(-1).astype(np.float32)
        vii_day_f = vii_day.reshape(-1).astype(np.float32)
        vii_night_f = vii_night.reshape(-1).astype(np.float32)

        if use_era5_daily:
            era5 = np.asarray(era5_daily_arr[t], dtype=np.float32)
            era5_feats = []
            for ch in range(era5.shape[0]):
                s0, _ = sanitize_array(era5[ch], infer_nodata_values(era5_daily_arr), extra_nodata, f"era5_band_{ch+1}")
                kind = _infer_era5_kind(s0)
                s, _ = _sanitize_and_convert(
                    era5[ch],
                    zarr_obj=era5_daily_arr,
                    extra_nodata=extra_nodata,
                    name=f"era5_band_{ch+1}",
                    kind=kind,
                    logger=logger,
                    units_cfg=units_cfg,
                )
                era5_feats.append(bilinear_resample_2d(s, h_lr, w_lr).reshape(-1))
        else:
            era5 = np.asarray(era5_30m_arr[t], dtype=np.float32)
            era5v = np.asarray(era5_30m_valid[t, 0], dtype=np.float32) > 0
            era5_feats = []
            for ch in range(era5.shape[0]):
                s0, _ = sanitize_array(era5[ch], infer_nodata_values(era5_30m_arr), extra_nodata, f"era5_band_{ch+1}")
                kind = _infer_era5_kind(s0)
                s, _ = _sanitize_and_convert(
                    era5[ch],
                    zarr_obj=era5_30m_arr,
                    extra_nodata=extra_nodata,
                    name=f"era5_band_{ch+1}",
                    kind=kind,
                    logger=logger,
                    units_cfg=units_cfg,
                )
                avg, _ = aggregate_mean_by_index(s, cell_index, n_cells, valid_mask=era5v)
                era5_feats.append(avg.astype(np.float32))

        target_cache_path = cache_dir / f"landsat_1km_t{t:05d}.npz"
        use_cache = False
        if target_cache_path.exists():
            obj = np.load(target_cache_path)
            cache_ver = int(np.ravel(obj["version"])[0]) if "version" in obj.files else -1
            if cache_ver == TARGET_CACHE_VERSION:
                y = obj["y"].astype(np.float32)
                w = obj["w"].astype(np.float32)
                use_cache = True
        if not use_cache:
            ls_raw = np.asarray(landsat_arr[t, 0], dtype=np.float32)
            raw_149_count["landsat"] += int(np.isclose(ls_raw, 149.0).sum())
            ls_s, _ = _sanitize_and_convert(
                ls_raw,
                zarr_obj=landsat_arr,
                extra_nodata=extra_nodata,
                name="landsat_target",
                kind="landsat_lst",
                logger=logger,
                units_cfg=units_cfg,
            )
            post_149_count["landsat"] += int(np.isclose(np.nan_to_num(ls_s, nan=-9999.0), 149.0).sum())
            ls_valid = np.asarray(landsat_valid_arr[t, 0], dtype=np.float32) > 0
            ls_valid &= np.isfinite(ls_s)
            y, w = _aggregate_target_1km(ls_s, ls_valid, cell_index, n_cells)
            np.savez_compressed(
                target_cache_path,
                version=np.array([TARGET_CACHE_VERSION], dtype=np.int32),
                y=y.astype(np.float32),
                w=w.astype(np.float32),
            )

        dt = daily_times[t]
        if pd.isna(dt):
            continue
        doy = float(pd.Timestamp(dt).dayofyear)
        doy_rad = 2.0 * np.pi * (doy / 365.0)
        doy_sin = np.full((n_cells,), np.sin(doy_rad), dtype=np.float32)
        doy_cos = np.full((n_cells,), np.cos(doy_rad), dtype=np.float32)

        cols: Dict[str, np.ndarray] = {
            "modis_day": mod_day_f,
            "modis_night": mod_night_f,
            "viirs_day": vii_day_f,
            "viirs_night": vii_night_f,
            "modis_valid_day": qc.modis_valid_day.reshape(-1),
            "modis_valid_night": qc.modis_valid_night.reshape(-1),
            "viirs_valid_day": qc.viirs_valid_day.reshape(-1),
            "viirs_valid_night": qc.viirs_valid_night.reshape(-1),
            "modis_qc_score_day": qc.modis_qc_score_day.reshape(-1),
            "modis_qc_score_night": qc.modis_qc_score_night.reshape(-1),
            "viirs_qc_score_day": qc.viirs_qc_score_day.reshape(-1),
            "viirs_qc_score_night": qc.viirs_qc_score_night.reshape(-1),
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
        }
        for i, feat in enumerate(era5_feats):
            cols[f"era5_band_{i+1}"] = feat.astype(np.float32)
        for k, v in static.items():
            cols[k] = v.astype(np.float32)

        if bool(feature_cfg.get("add_engineered", False)):
            cols["modis_dnd"] = (cols["modis_day"] - cols["modis_night"]).astype(np.float32)
            cols["viirs_dnd"] = (cols["viirs_day"] - cols["viirs_night"]).astype(np.float32)
            cols["m_minus_v_day"] = (cols["modis_day"] - cols["viirs_day"]).astype(np.float32)
            cols["m_minus_v_night"] = (cols["modis_night"] - cols["viirs_night"]).astype(np.float32)
            if not engineered_feature_names:
                engineered_feature_names.extend(["modis_dnd", "viirs_dnd", "m_minus_v_day", "m_minus_v_night"])
            if "era5_band_1" in cols and "era5_band_2" in cols:
                cols["era5_delta_1_2"] = (cols["era5_band_1"] - cols["era5_band_2"]).astype(np.float32)
                if "era5_delta_1_2" not in era_delta_names:
                    era_delta_names.append("era5_delta_1_2")
            if "era5_band_3" in cols and "era5_band_4" in cols:
                cols["era5_delta_3_4"] = (cols["era5_band_3"] - cols["era5_band_4"]).astype(np.float32)
                if "era5_delta_3_4" not in era_delta_names:
                    era_delta_names.append("era5_delta_3_4")
            if "era5_band_5" in cols and "era5_band_6" in cols:
                cols["era5_delta_5_6"] = (cols["era5_band_5"] - cols["era5_band_6"]).astype(np.float32)
                if "era5_delta_5_6" not in era_delta_names:
                    era_delta_names.append("era5_delta_5_6")

        if bool(feature_cfg.get("use_neighborhood_stats", False)):
            try:
                nb_cols, from_cache, created_any, dt_sec = _build_neighborhood_features(
                    t=t,
                    cols=cols,
                    feature_cfg=feature_cfg,
                    cache_root=nb_cache_dir,
                    h_lr=h_lr,
                    w_lr=w_lr,
                    logger=logger,
                )
                neighborhood_build_sec += float(dt_sec)
                if from_cache:
                    neighborhood_cache_hits += 1
                if created_any:
                    cols.update(nb_cols)
                    neighborhood_created = True
                    neighborhood_added_count = max(neighborhood_added_count, len(nb_cols))
            except Exception as exc:
                logger.warning("Neighborhood feature creation skipped at t=%d due to: %s", t, exc)

        qc_weight = _compute_qc_weight(cols, cfg) if bool(cfg.get("training", {}).get("qc_weighting", False)) else np.ones((n_cells,), dtype=np.float32)

        if feature_names is None:
            feature_names = list(cols.keys())
        else:
            for k in feature_names:
                if k not in cols:
                    cols[k] = np.full((n_cells,), np.nan, dtype=np.float32)
        x = np.stack([cols[k] for k in feature_names], axis=1).astype(np.float32)

        mod_ok = (cols["modis_valid_day"] > 0) | (cols["modis_valid_night"] > 0)
        vii_ok = (cols["viirs_valid_day"] > 0) | (cols["viirs_valid_night"] > 0)
        both_invalid = (~mod_ok) & (~vii_ok)
        mod_only = mod_ok & (~vii_ok)
        vii_only = (~mod_ok) & vii_ok
        both_valid = mod_ok & vii_ok
        both_invalid_count += int(np.sum(both_invalid))
        modis_only_count += int(np.sum(mod_only))
        viirs_only_count += int(np.sum(vii_only))
        both_valid_count += int(np.sum(both_valid))

        drop_both_invalid = bool(ds_cfg.get("drop_if_both_thermal_invalid", True))
        usable = (~both_invalid) if drop_both_invalid else (mod_ok | vii_ok | both_invalid)
        keep_base = usable & np.isfinite(y)
        keep_valid = w >= min_valid_frac
        keep = keep_base & keep_valid
        dropped_by_valid_frac += int(np.sum(keep_base & (~keep_valid)))
        dropped_nonfinite_or_unusable += int(np.sum(~keep_base))
        n_total_cells += int(keep.size)
        n_kept_cells += int(np.sum(keep))
        if not np.any(keep):
            continue

        vf_kept = np.clip(w[keep].astype(np.float32), 0.0, 1.0)
        vf_weight = np.power(vf_kept, vf_power, dtype=np.float32)
        qcw_kept = np.asarray(qc_weight[keep], dtype=np.float32)
        sample_w = vf_weight * qcw_kept
        sample_w = np.where(np.isfinite(sample_w), sample_w, 0.0).astype(np.float32)
        sample_w = np.clip(sample_w, 1.0e-3, None)
        rows_x.append(x[keep])
        rows_y.append(y[keep].astype(np.float32))
        rows_w.append(sample_w)
        rows_vf.append(vf_kept)
        rows_qcw.append(qcw_kept)
        rows_date.append(np.full(np.sum(keep), t, dtype=np.int64))
        rows_cell.append(np.flatnonzero(keep).astype(np.int64))
        y_all.append(y[keep].astype(np.float32))
        vf_all.append(vf_kept)
        qcw_all.append(qcw_kept)

    if not rows_x:
        raise RuntimeError("No usable samples after filtering by valid fraction and QC.")

    y_cat = np.concatenate(y_all, axis=0) if y_all else np.array([], dtype=np.float32)
    vf_cat = np.concatenate(vf_all, axis=0) if vf_all else np.array([], dtype=np.float32)
    logger.info("Target stats after sanitize: %s", _agg_stats(y_cat))
    if vf_cat.size > 0:
        logger.info(
            "landsat_valid_frac quantiles p01=%.4f p50=%.4f p99=%.4f",
            float(np.nanpercentile(vf_cat, 1)),
            float(np.nanpercentile(vf_cat, 50)),
            float(np.nanpercentile(vf_cat, 99)),
        )

    if engineered_feature_names:
        logger.info("Engineered features enabled: %s", ", ".join(engineered_feature_names + era_delta_names))
    if bool(feature_cfg.get("use_neighborhood_stats", False)):
        if neighborhood_created:
            logger.info(
                "Neighborhood stats enabled added_features=%d cache_hits=%d total_compute_sec=%.2f",
                neighborhood_added_count,
                neighborhood_cache_hits,
                neighborhood_build_sec,
            )
        else:
            logger.warning("Neighborhood stats requested but none were created.")

    table = BaseNetTable(
        x=np.concatenate(rows_x, axis=0),
        y=np.concatenate(rows_y, axis=0),
        w=np.concatenate(rows_w, axis=0),
        vf=np.concatenate(rows_vf, axis=0) if rows_vf else None,
        qc_weight=np.concatenate(rows_qcw, axis=0) if rows_qcw else None,
        date_idx=np.concatenate(rows_date, axis=0),
        cell_idx=np.concatenate(rows_cell, axis=0),
        feature_names=feature_names or [],
        dates=daily_times,
        grid_shape=(h_lr, w_lr),
        debug_info={
            "raw_149_count": raw_149_count,
            "post_149_count": post_149_count,
            "both_invalid_count": both_invalid_count,
            "modis_only_count": modis_only_count,
            "viirs_only_count": viirs_only_count,
            "both_valid_count": both_valid_count,
            "target_stats": _agg_stats(y_cat),
            "valid_frac_stats": {
                "min": float(np.nanmin(vf_cat)) if vf_cat.size else float("nan"),
                "median": float(np.nanmedian(vf_cat)) if vf_cat.size else float("nan"),
                "max": float(np.nanmax(vf_cat)) if vf_cat.size else float("nan"),
            },
            "feature_count_before_lag": int(len(feature_names or [])),
            "engineered_enabled": bool(feature_cfg.get("add_engineered", False)),
            "neighborhood_enabled": bool(feature_cfg.get("use_neighborhood_stats", False)),
            "neighborhood_created": bool(neighborhood_created),
            "neighborhood_added_count": int(neighborhood_added_count),
            "neighborhood_cache_hits": int(neighborhood_cache_hits),
            "neighborhood_build_sec": float(neighborhood_build_sec),
            "split_role": split_role_l,
            "min_valid_frac_used": float(min_valid_frac),
            "dropped_by_valid_frac": int(dropped_by_valid_frac),
            "dropped_nonfinite_or_unusable": int(dropped_nonfinite_or_unusable),
        },
    )
    table = _apply_lag_features(table, cfg, logger)
    table.debug_info["feature_count_after_lag"] = int(len(table.feature_names))
    if qcw_all:
        qcw_cat = np.concatenate(qcw_all, axis=0).astype(np.float32)
        qmin = float(cfg.get("training", {}).get("qc_weight_min", 0.2))
        qp10 = float(np.nanpercentile(qcw_cat, 10))
        qp50 = float(np.nanpercentile(qcw_cat, 50))
        qp90 = float(np.nanpercentile(qcw_cat, 90))
        table.debug_info["qc_weight_mean"] = float(np.nanmean(qcw_cat))
        table.debug_info["qc_weight_clamped_frac"] = float(np.mean(qcw_cat <= (qmin + 1.0e-6)))
        table.debug_info["qc_weight_p10"] = qp10
        table.debug_info["qc_weight_p50"] = qp50
        table.debug_info["qc_weight_p90"] = qp90
        logger.info(
            "QC weighting stats mean=%.4f p10=%.4f p50=%.4f p90=%.4f clamped_frac=%.4f min=%.3f",
            float(np.nanmean(qcw_cat)),
            qp10,
            qp50,
            qp90,
            float(np.mean(qcw_cat <= (qmin + 1.0e-6))),
            qmin,
        )
    logger.info(
        "Built table samples=%d features=%d unique_dates=%d kept=%d/%d dropped_valid_frac=%d dropped_nonfinite_or_unusable=%d",
        table.x.shape[0],
        table.x.shape[1],
        np.unique(table.date_idx).size,
        n_kept_cells,
        n_total_cells,
        dropped_by_valid_frac,
        dropped_nonfinite_or_unusable,
    )
    logger.info(
        "Gate override fractions mod_only=%.4f viirs_only=%.4f both_valid=%.4f both_invalid=%.4f",
        modis_only_count / max(1, n_total_cells),
        viirs_only_count / max(1, n_total_cells),
        both_valid_count / max(1, n_total_cells),
        both_invalid_count / max(1, n_total_cells),
    )
    return table


