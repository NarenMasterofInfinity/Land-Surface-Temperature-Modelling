from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import zarr


@dataclass
class BaseNetFeatureTable:
    x: np.ndarray
    feature_names: List[str]
    base_feature_names: List[str]
    grid_shape: Tuple[int, int]
    date_idx: int


def _to_celsius_if_needed(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32, copy=False)
    if np.isfinite(y).any() and float(np.nanmedian(y)) > 200.0:
        y = y - 273.15
    return y


def _to_celsius_per_band_if_needed(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    if y.ndim != 3:
        return _to_celsius_if_needed(y)
    for b in range(y.shape[0]):
        band = y[b]
        finite = np.isfinite(band)
        if not np.any(finite):
            continue
        med = float(np.nanmedian(band[finite]))
        # Convert only bands that look like Kelvin temperature fields.
        if 200.0 <= med <= 350.0:
            y[b] = band - 273.15
    return y


def _safe_float(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float32)
    x[~np.isfinite(x)] = np.nan
    return x


def _qc_modis_to_score(qc: np.ndarray) -> np.ndarray:
    q = _safe_float(qc)
    score = np.full_like(q, np.nan, dtype=np.float32)
    score[q <= 0] = 1.0
    score[q == 1] = 0.7
    score[q == 2] = 0.4
    score[q >= 3] = 0.2
    return np.clip(score, 0.0, 1.0)


def _qc_viirs_to_score(cloud: np.ndarray) -> np.ndarray:
    c = _safe_float(cloud)
    score = np.full_like(c, np.nan, dtype=np.float32)
    score[c <= 0] = 1.0
    score[c == 1] = 0.7
    score[c == 2] = 0.3
    score[c >= 3] = 0.1
    return np.clip(score, 0.0, 1.0)


def _nanmean_blocks(arr: np.ndarray, y_edges: np.ndarray, x_edges: np.ndarray) -> np.ndarray:
    out_h = len(y_edges) - 1
    out_w = len(x_edges) - 1
    out = np.full((out_h, out_w), np.nan, dtype=np.float32)
    for iy in range(out_h):
        y0, y1 = int(y_edges[iy]), int(y_edges[iy + 1])
        for ix in range(out_w):
            x0, x1 = int(x_edges[ix]), int(x_edges[ix + 1])
            block = arr[y0:y1, x0:x1]
            if block.size == 0:
                continue
            finite = np.isfinite(block)
            if np.any(finite):
                out[iy, ix] = float(np.nanmean(block[finite]))
    return out


def _resize_2d(arr: np.ndarray, out_hw: Tuple[int, int], mode: str = "bilinear") -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    t = torch.from_numpy(x)[None, None]
    if mode == "nearest":
        o = F.interpolate(t, size=out_hw, mode="nearest")
    else:
        o = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return o[0, 0].cpu().numpy().astype(np.float32)


def _ensure_hw(arr: np.ndarray | float, h: int, w: int, mode: str = "bilinear") -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 0:
        return np.full((h, w), float(x), dtype=np.float32)
    if x.ndim == 1:
        if x.shape[0] == (h * w):
            return x.reshape(h, w).astype(np.float32)
        raise RuntimeError(f"Cannot map 1D feature of shape={x.shape} to grid {(h, w)}")
    if x.ndim != 2:
        raise RuntimeError(f"Expected 2D feature grid, got shape={x.shape}")
    if x.shape == (h, w):
        return x.astype(np.float32, copy=False)
    return _resize_2d(x, (h, w), mode=mode)


def _fraction_blocks(mask: np.ndarray, y_edges: np.ndarray, x_edges: np.ndarray) -> np.ndarray:
    out_h = len(y_edges) - 1
    out_w = len(x_edges) - 1
    out = np.zeros((out_h, out_w), dtype=np.float32)
    m = mask.astype(np.float32, copy=False)
    for iy in range(out_h):
        y0, y1 = int(y_edges[iy]), int(y_edges[iy + 1])
        for ix in range(out_w):
            x0, x1 = int(x_edges[ix]), int(x_edges[ix + 1])
            block = m[y0:y1, x0:x1]
            if block.size == 0:
                continue
            out[iy, ix] = float(np.mean(block))
    return out


def _nb_stats(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = _safe_float(arr)
    m = np.isfinite(v).astype(np.float32)
    vv = np.where(np.isfinite(v), v, 0.0)
    vv2 = vv * vv
    pad_v = np.pad(vv, 1, mode="constant", constant_values=0.0)
    pad_m = np.pad(m, 1, mode="constant", constant_values=0.0)
    pad_v2 = np.pad(vv2, 1, mode="constant", constant_values=0.0)
    s = np.zeros_like(vv, dtype=np.float32)
    s2 = np.zeros_like(vv, dtype=np.float32)
    c = np.zeros_like(vv, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            s += pad_v[dy : dy + vv.shape[0], dx : dx + vv.shape[1]]
            s2 += pad_v2[dy : dy + vv.shape[0], dx : dx + vv.shape[1]]
            c += pad_m[dy : dy + vv.shape[0], dx : dx + vv.shape[1]]
    mean = np.full_like(vv, np.nan, dtype=np.float32)
    np.divide(s, c, out=mean, where=c > 0)
    s2_over_c = np.full_like(vv, np.nan, dtype=np.float32)
    np.divide(s2, c, out=s2_over_c, where=c > 0)
    var = s2_over_c - (mean * mean)
    var = np.where(var < 0, 0, var)
    std = np.sqrt(var, dtype=np.float32)
    valid_frac = (c / 9.0).astype(np.float32)
    return mean, std, valid_frac


def _extract_modis_fields(modis_t: np.ndarray) -> Dict[str, np.ndarray]:
    day = _safe_float(modis_t[0])
    night = _safe_float(modis_t[1])
    day = _to_celsius_if_needed(day)
    night = _to_celsius_if_needed(night)
    qc_day = _qc_modis_to_score(modis_t[2])
    qc_night = _qc_modis_to_score(modis_t[3])
    valid_day = _safe_float(modis_t[4]) > 0.5
    valid_night = _safe_float(modis_t[5]) > 0.5
    day = np.where(valid_day, day, np.nan)
    night = np.where(valid_night, night, np.nan)
    qc_day = np.where(valid_day, qc_day, np.nan)
    qc_night = np.where(valid_night, qc_night, np.nan)
    return {
        "day": day.astype(np.float32),
        "night": night.astype(np.float32),
        "valid_day": valid_day.astype(np.float32),
        "valid_night": valid_night.astype(np.float32),
        "qc_day": qc_day.astype(np.float32),
        "qc_night": qc_night.astype(np.float32),
    }


def _extract_viirs_fields(viirs_t: np.ndarray) -> Dict[str, np.ndarray]:
    day = _safe_float(viirs_t[0])
    night = _safe_float(viirs_t[1])
    day = _to_celsius_if_needed(day)
    night = _to_celsius_if_needed(night)
    cloud_day = _safe_float(viirs_t[2])
    cloud_night = _safe_float(viirs_t[3])
    valid_day = np.isfinite(day) & (cloud_day <= 1.0)
    valid_night = np.isfinite(night) & (cloud_night <= 1.0)
    qc_day = _qc_viirs_to_score(cloud_day)
    qc_night = _qc_viirs_to_score(cloud_night)
    day = np.where(valid_day, day, np.nan)
    night = np.where(valid_night, night, np.nan)
    qc_day = np.where(valid_day, qc_day, np.nan)
    qc_night = np.where(valid_night, qc_night, np.nan)
    return {
        "day": day.astype(np.float32),
        "night": night.astype(np.float32),
        "valid_day": valid_day.astype(np.float32),
        "valid_night": valid_night.astype(np.float32),
        "qc_day": qc_day.astype(np.float32),
        "qc_night": qc_night.astype(np.float32),
    }


def _build_static_lr(
    *,
    dem_30m: np.ndarray,
    world_30m: np.ndarray,
    dyn_30m: np.ndarray,
    lr_shape: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    h_lr, w_lr = lr_shape
    h_hr, w_hr = dem_30m.shape
    y_edges = np.linspace(0, h_hr, h_lr + 1, dtype=np.int64)
    x_edges = np.linspace(0, w_hr, w_lr + 1, dtype=np.int64)

    dem = _safe_float(dem_30m)
    gy, gx = np.gradient(np.where(np.isfinite(dem), dem, np.nanmedian(dem)))
    slope = np.sqrt((gx * gx) + (gy * gy), dtype=np.float32)
    aspect = np.arctan2(gy, gx).astype(np.float32)

    out: Dict[str, np.ndarray] = {
        "elev_mean": _nanmean_blocks(dem, y_edges, x_edges),
        "slope_mean": _nanmean_blocks(slope, y_edges, x_edges),
        "aspect_sin_mean": _nanmean_blocks(np.sin(aspect, dtype=np.float32), y_edges, x_edges),
        "aspect_cos_mean": _nanmean_blocks(np.cos(aspect, dtype=np.float32), y_edges, x_edges),
    }

    world = _safe_float(world_30m)
    dyn = _safe_float(dyn_30m)
    for cls in (10, 20, 30, 40, 50, 60, 80, 90):
        out[f"worldcover_frac_{cls}"] = _fraction_blocks(world == float(cls), y_edges, x_edges)
    for cls in (1, 2, 3, 4, 5, 6, 7):
        out[f"dynamic_world_frac_{cls}"] = _fraction_blocks(dyn == float(cls), y_edges, x_edges)
    return out


def _doy_features(date_ts) -> Tuple[float, float]:
    doy = float(date_ts.dayofyear)
    phase = (2.0 * np.pi * doy) / 365.25
    return float(np.sin(phase)), float(np.cos(phase))


def build_1km_feature_table_for_date(
    *,
    date_idx: int,
    root_daily: zarr.Group,
    root_30m: zarr.Group,
    daily_times,
    ckpt: dict,
    static_cache: Dict[str, np.ndarray] | None = None,
) -> BaseNetFeatureTable:
    cfg = ckpt["config"]
    ds_cfg = cfg["dataset"]
    impute = ckpt["impute"]
    base_names: List[str] = list(impute["base_feature_names"])
    final_names: List[str] = list(impute["final_feature_names"])
    mask_names: List[str] = list(impute["mask_feature_names"])
    medians = np.asarray(impute["medians"], dtype=np.float32)

    modis = np.asarray(root_daily[ds_cfg["modis_group"]]["data"][int(date_idx)], dtype=np.float32)
    viirs = np.asarray(root_daily[ds_cfg["viirs_group"]]["data"][int(date_idx)], dtype=np.float32)
    era5 = np.asarray(root_daily[ds_cfg["era5_group_daily"]]["data"][int(date_idx)], dtype=np.float32)
    h_lr, w_lr = modis.shape[-2], modis.shape[-1]

    if static_cache is None:
        static_cache = {}
    if not static_cache:
        dem_30m = np.asarray(root_30m[ds_cfg["dem_group"]]["data"][0, 0], dtype=np.float32)
        world_30m = np.asarray(root_30m[ds_cfg["worldcover_group"]]["data"][0, 0], dtype=np.float32)
        dyn_30m = np.asarray(root_30m[ds_cfg["dynamic_world_group"]]["data"][0, 0], dtype=np.float32)
        static_cache.update(_build_static_lr(dem_30m=dem_30m, world_30m=world_30m, dyn_30m=dyn_30m, lr_shape=(h_lr, w_lr)))

    m0 = _extract_modis_fields(modis)
    v0 = _extract_viirs_fields(viirs)

    prev_idx = max(0, int(date_idx) - 1)
    m1 = _extract_modis_fields(np.asarray(root_daily[ds_cfg["modis_group"]]["data"][prev_idx], dtype=np.float32))
    v1 = _extract_viirs_fields(np.asarray(root_daily[ds_cfg["viirs_group"]]["data"][prev_idx], dtype=np.float32))

    era5 = _safe_float(era5)
    era5 = _to_celsius_per_band_if_needed(era5)
    if era5.shape[0] < 8:
        raise RuntimeError(f"Expected at least 8 daily ERA5 bands, got shape={era5.shape}")

    doy_sin, doy_cos = _doy_features(daily_times[int(date_idx)])
    modis_nb_mean, modis_nb_std, modis_nb_valid = _nb_stats(m0["day"])
    viirs_nb_mean, viirs_nb_std, viirs_nb_valid = _nb_stats(v0["day"])

    fields: Dict[str, np.ndarray] = {
        "modis_day": m0["day"],
        "modis_night": m0["night"],
        "viirs_day": v0["day"],
        "viirs_night": v0["night"],
        "modis_valid_day": m0["valid_day"],
        "modis_valid_night": m0["valid_night"],
        "viirs_valid_day": v0["valid_day"],
        "viirs_valid_night": v0["valid_night"],
        "modis_qc_score_day": m0["qc_day"],
        "modis_qc_score_night": m0["qc_night"],
        "viirs_qc_score_day": v0["qc_day"],
        "viirs_qc_score_night": v0["qc_night"],
        "doy_sin": np.full((h_lr, w_lr), doy_sin, dtype=np.float32),
        "doy_cos": np.full((h_lr, w_lr), doy_cos, dtype=np.float32),
        "era5_band_1": era5[0],
        "era5_band_2": era5[1],
        "era5_band_3": era5[2],
        "era5_band_4": era5[3],
        "era5_band_5": era5[4],
        "era5_band_6": era5[5],
        "era5_band_7": era5[6],
        "era5_band_8": era5[7],
        "modis_dnd": m0["day"] - m0["night"],
        "viirs_dnd": v0["day"] - v0["night"],
        "m_minus_v_day": m0["day"] - v0["day"],
        "m_minus_v_night": m0["night"] - v0["night"],
        "era5_delta_1_2": era5[0] - era5[1],
        "era5_delta_3_4": era5[2] - era5[3],
        "era5_delta_5_6": era5[4] - era5[5],
        "modis_day_nb_mean3": modis_nb_mean,
        "modis_day_nb_std3": modis_nb_std,
        "modis_day_nb_valid": modis_nb_valid,
        "viirs_day_nb_mean3": viirs_nb_mean,
        "viirs_day_nb_std3": viirs_nb_std,
        "viirs_day_nb_valid": viirs_nb_valid,
        "modis_day_lag1": m1["day"],
        "modis_night_lag1": m1["night"],
        "viirs_day_lag1": v1["day"],
        "viirs_night_lag1": v1["night"],
        "modis_day_lag1_valid": m1["valid_day"],
        "modis_night_lag1_valid": m1["valid_night"],
        "viirs_day_lag1_valid": v1["valid_day"],
        "viirs_night_lag1_valid": v1["valid_night"],
    }
    fields.update(static_cache)

    # Enforce a single canonical grid shape for every feature before stacking.
    # This guards against slight resolution mismatches across source groups.
    for k, v in list(fields.items()):
        mode = "nearest" if ("_valid" in k or "_frac_" in k or "_isnan" in k) else "bilinear"
        fields[k] = _ensure_hw(v, h_lr, w_lr, mode=mode)

    missing = [n for n in base_names if n not in fields]
    if missing:
        raise RuntimeError(f"Missing base features for BaseNet: {missing[:20]}")

    base_mat = np.stack([fields[n].reshape(-1).astype(np.float32) for n in base_names], axis=1)
    if base_mat.shape[1] != medians.shape[0]:
        raise RuntimeError(f"Median shape mismatch: base_mat={base_mat.shape}, medians={medians.shape}")

    nan_mask = ~np.isfinite(base_mat)
    x_filled = base_mat.copy()
    for j in range(x_filled.shape[1]):
        x_filled[nan_mask[:, j], j] = medians[j]

    if mask_names:
        idx = [base_names.index(n.replace("_isnan", "")) for n in mask_names]
        x_final = np.concatenate([x_filled, nan_mask[:, idx].astype(np.float32)], axis=1)
    else:
        x_final = x_filled

    if list(final_names) != list(ckpt["feature_names"]):
        raise RuntimeError("Checkpoint final_feature_names mismatch with ckpt feature_names.")

    return BaseNetFeatureTable(
        x=x_final.astype(np.float32),
        feature_names=list(final_names),
        base_feature_names=list(base_names),
        grid_shape=(h_lr, w_lr),
        date_idx=int(date_idx),
    )


def build_1km_feature_table_for_dates(
    *,
    date_indices: List[int],
    root_daily: zarr.Group,
    root_30m: zarr.Group,
    daily_times,
    ckpt: dict,
) -> Dict[int, BaseNetFeatureTable]:
    out: Dict[int, BaseNetFeatureTable] = {}
    static_cache: Dict[str, np.ndarray] = {}
    for d in date_indices:
        out[int(d)] = build_1km_feature_table_for_date(
            date_idx=int(d),
            root_daily=root_daily,
            root_30m=root_30m,
            daily_times=daily_times,
            ckpt=ckpt,
            static_cache=static_cache,
        )
    return out
