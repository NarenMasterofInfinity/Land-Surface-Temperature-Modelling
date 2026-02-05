from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from helper import make_madurai_data, load_subset


PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project").resolve()
OUT_DIR = PROJECT_ROOT / "metrics" / "linear_baselines_simple"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


@dataclass
class SampleSpec:
    max_samples_per_time: int = 4000
    max_total_samples: int = 40000
    tile_size: int = 256
    max_tiles_per_time: int = 8


def _infer_xy_dims(da: xr.DataArray) -> Tuple[str, str]:
    if "y" in da.dims and "x" in da.dims:
        return "y", "x"
    return da.dims[-2], da.dims[-1]


def _infer_time_dim_for_var(da: xr.DataArray) -> Optional[str]:
    y_dim, x_dim = _infer_xy_dims(da)
    for d in da.dims:
        if d not in (y_dim, x_dim) and d not in ("band", "bands", "channel", "c"):
            return d
    return None


def _valid_var_for(data_var: str, ds: xr.Dataset) -> Optional[str]:
    if data_var.endswith("/data"):
        candidate = f"{data_var.rsplit('/', 1)[0]}/valid"
        if candidate in ds.data_vars:
            return candidate
    if "/band_" in data_var:
        candidate = f"{data_var.split('/band_')[0]}/valid"
        if candidate in ds.data_vars:
            return candidate
    return None


def _nanify_nodata(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    out = np.array(a, copy=True)
    out[np.isclose(out, -9999.0)] = np.nan
    return out


def _is_landsat_var(var_name: str) -> bool:
    return "landsat" in var_name.lower()


def _is_viirs_var(var_name: str) -> bool:
    return "viirs" in var_name.lower()


def _is_modis_var(var_name: str) -> bool:
    return "modis" in var_name.lower()


def _is_lst_source(var_name: str) -> bool:
    name = var_name.lower()
    if "era5" in name or "ecmwf" in name:
        return False
    return (
        _is_landsat_var(name)
        or _is_viirs_var(name)
        or _is_modis_var(name)
        or "lst" in name
        or "temperature" in name
        or "temp" in name
        or name.endswith("band_01")
    )


def _mask_lst_values(data: np.ndarray, var_name: str) -> np.ndarray:
    if not _is_lst_source(var_name):
        return data
    out = np.where(np.isfinite(data) & (data > 0), data, np.nan)
    if _is_landsat_var(var_name):
        out = np.where(np.isclose(out, 149.0), np.nan, out)
    return out


def _convert_lst_units(data: np.ndarray, var_name: str) -> np.ndarray:
    if data.size == 0:
        return data
    finite = np.isfinite(data)
    if not np.any(finite):
        return data
    if _is_modis_var(var_name):
        return data  # MODIS already in °C
    if _is_landsat_var(var_name) or _is_viirs_var(var_name):
        med = float(np.nanmedian(data[finite]))
        if med > 200:
            return data - 273.15
    return data


def _apply_valid_mask_data(data: np.ndarray, valid: Optional[np.ndarray]) -> np.ndarray:
    if valid is None:
        return data
    v = np.asarray(valid)
    v = np.squeeze(v)
    mask = np.isfinite(v) & (v > 0)
    if mask.ndim == data.ndim - 1:
        mask = np.broadcast_to(mask, data.shape)
    if mask.shape != data.shape:
        return data
    return np.where(mask, data, np.nan)


def _select_time_dataset(
    ds: xr.Dataset,
    td: str,
    t_val,
    time_index_map: Dict[pd.Timestamp, int],
) -> xr.Dataset:
    if td not in ds.dims:
        td = detect_time_dim(ds) or td
        if td not in ds.dims:
            return ds
    t_stamp = pd.Timestamp(t_val)
    if td in ds.coords:
        try:
            out = ds.sel({td: t_stamp})
            if td in out.dims:
                return out.squeeze(dim=td, drop=True)
            return out
        except Exception:
            pass
    idx = time_index_map.get(t_stamp, 0)
    idx = int(np.clip(idx, 0, ds.sizes[td] - 1))
    out = ds.isel({td: idx})
    if td in out.dims:
        return out.squeeze(dim=td, drop=True)
    return out


def _select_time_for_var(
    da: xr.DataArray,
    t_val,
    time_index_map: Optional[Dict[pd.Timestamp, int]] = None,
) -> xr.DataArray:
    tdim = _infer_time_dim_for_var(da)
    if tdim is None or tdim not in da.dims:
        return da
    t_stamp = pd.Timestamp(t_val)
    if tdim in da.coords:
        try:
            out = da.sel({tdim: t_stamp})
            return out.squeeze(tdim, drop=True)
        except Exception:
            pass
        times = pd.DatetimeIndex(pd.to_datetime(da[tdim].values))
        if len(times) > 0:
            idx = int(np.argmin(np.abs(times - t_stamp)))
            out = da.isel({tdim: idx})
            return out.squeeze(tdim, drop=True)
    if time_index_map:
        idx = time_index_map.get(t_stamp, 0)
        idx = int(np.clip(idx, 0, da.sizes[tdim] - 1))
        out = da.isel({tdim: idx})
        return out.squeeze(tdim, drop=True)
    return da.isel({tdim: 0}).squeeze(tdim, drop=True)


def detect_time_dim(ds: xr.Dataset) -> Optional[str]:
    for td in ("time", "date", "t", "datetime"):
        if td in ds.coords or td in ds.dims:
            return td
    for c in ds.coords:
        try:
            if np.issubdtype(ds[c].dtype, np.datetime64):
                return c
        except Exception:
            pass
    return None


def default_feature_vars(ds: xr.Dataset, target: str) -> List[str]:
    feats = []
    for v in ds.data_vars:
        if v == target:
            continue
        name = v.lower()
        if any(k in name for k in ("qc", "quality", "mask", "flag", "cloud", "valid")):
            continue
        if v in {
            "labels_30m/landsat/band_02",
            "static_30m/worldcover/band_01",
            "static_30m/dynamic_world/band_01",
        }:
            continue
        feats.append(v)
    if not feats:
        raise ValueError("No feature variables found. Provide --features explicitly.")
    return feats


def sample_xy_for_time(
    ds: xr.Dataset,
    td: str,
    t_val,
    target: str,
    features: Sequence[str],
    spec: SampleSpec,
    rng: np.random.Generator,
    *,
    valid_map: Optional[Dict[str, str]] = None,
    target_valid: Optional[str] = None,
    time_index_map: Optional[Dict[pd.Timestamp, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if time_index_map is None:
        time_index_map = {}
    dts = _select_time_dataset(ds, td, t_val, time_index_map)
    y_da = _select_time_for_var(dts[target], t_val, time_index_map)
    y_dim, x_dim = _infer_xy_dims(y_da)
    H = y_da.sizes[y_dim]
    W = y_da.sizes[x_dim]

    X_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    remaining = int(spec.max_samples_per_time)

    for _ in range(spec.max_tiles_per_time):
        if remaining <= 0:
            break
        r0 = int(rng.integers(0, max(1, H - spec.tile_size + 1)))
        c0 = int(rng.integers(0, max(1, W - spec.tile_size + 1)))
        r1 = min(H, r0 + spec.tile_size)
        c1 = min(W, c0 + spec.tile_size)

        y_tile = np.asarray(
            y_da.isel({y_dim: slice(r0, r1), x_dim: slice(c0, c1)}).data,
            dtype=np.float32,
        )
        y_tile = _nanify_nodata(y_tile)
        if target_valid and target_valid in dts.data_vars:
            v_da = _select_time_for_var(dts[target_valid], t_val, time_index_map)
            v_tile = np.asarray(
                v_da.isel({y_dim: slice(r0, r1), x_dim: slice(c0, c1)}).data
            )
            y_tile = _apply_valid_mask_data(y_tile, v_tile)
        y_tile = _mask_lst_values(y_tile, target)
        y_tile = _convert_lst_units(y_tile, target)
        m_tile = np.isfinite(y_tile)
        idx_all = np.flatnonzero(m_tile.reshape(-1))
        if idx_all.size == 0:
            continue
        n = int(min(remaining, idx_all.size))
        idx = rng.choice(idx_all, size=n, replace=False)

        X_tile = np.zeros((n, len(features)), dtype=np.float32)
        for j, v in enumerate(features):
            a_da = _select_time_for_var(dts[v], t_val, time_index_map)
            a = np.asarray(
                a_da.isel({y_dim: slice(r0, r1), x_dim: slice(c0, c1)}).data,
                dtype=np.float32,
            )
            a = _nanify_nodata(a)
            if valid_map:
                valid_var = valid_map.get(v)
                if valid_var and valid_var in dts.data_vars:
                    v_da = _select_time_for_var(dts[valid_var], t_val, time_index_map)
                    v_tile = np.asarray(
                        v_da.isel({y_dim: slice(r0, r1), x_dim: slice(c0, c1)}).data
                    )
                    a = _apply_valid_mask_data(a, v_tile)
            a = _mask_lst_values(a, v)
            a = _convert_lst_units(a, v)
            col = a.reshape(-1)[idx].astype(np.float32, copy=False)
            col[~np.isfinite(col)] = np.nan
            X_tile[:, j] = col

        y_s = y_tile.reshape(-1)[idx].astype(np.float32, copy=False)
        X_chunks.append(X_tile)
        y_chunks.append(y_s)
        remaining -= n

    if not y_chunks:
        return np.empty((0, len(features)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    return X, y


def impute_with_train_medians(X_train: np.ndarray, X_any: np.ndarray) -> np.ndarray:
    meds = np.nanmedian(X_train, axis=0)
    meds = np.where(np.isfinite(meds), meds, 0.0)
    X = np.array(X_any, copy=True)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(meds, np.where(bad)[1])
    return X


def _tile_to_numpy(da: xr.DataArray, r0: int, r1: int) -> np.ndarray:
    data = da.data
    tile = data[r0:r1, :]
    return np.asarray(tile, dtype=np.float32)


def predict_full_map_simple(
    model: Pipeline,
    dts: xr.Dataset,
    td: str,
    t_val,
    target: str,
    features: Sequence[str],
    *,
    tile_rows: int,
    valid_map: Optional[Dict[str, str]] = None,
    target_valid: Optional[str] = None,
    time_index_map: Optional[Dict[pd.Timestamp, int]] = None,
    train_medians: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if time_index_map is None:
        time_index_map = {}
    y_da = _select_time_for_var(dts[target], t_val, time_index_map)
    y_true = np.asarray(y_da.data, dtype=np.float32)
    y_true = _nanify_nodata(y_true)
    if target_valid and target_valid in dts.data_vars:
        v_da = _select_time_for_var(dts[target_valid], t_val, time_index_map)
        v = np.asarray(v_da.data)
        y_true = _apply_valid_mask_data(y_true, v)
    y_true = _mask_lst_values(y_true, target)
    y_true = _convert_lst_units(y_true, target)

    H, W = y_true.shape[-2], y_true.shape[-1]
    y_pred = np.full((H, W), np.nan, dtype=np.float32)
    feature_arrays = [(_select_time_for_var(dts[v], t_val, time_index_map)) for v in features]

    for r0 in range(0, H, tile_rows):
        r1 = min(H, r0 + tile_rows)
        cols = []
        for da, v in zip(feature_arrays, features):
            tile = _tile_to_numpy(da, r0, r1)
            tile = _nanify_nodata(tile)
            if valid_map:
                valid_var = valid_map.get(da.name)
                if valid_var and valid_var in dts.data_vars:
                    v_da = _select_time_for_var(dts[valid_var], t_val, time_index_map)
                    v_tile = _tile_to_numpy(v_da, r0, r1)
                    tile = _apply_valid_mask_data(tile, v_tile)
            tile = _mask_lst_values(tile, v)
            tile = _convert_lst_units(tile, v)
            cols.append(tile.reshape(-1))
        X_tile = np.stack(cols, axis=1)
        X_tile[~np.isfinite(X_tile)] = np.nan
        if train_medians is not None:
            bad = ~np.isfinite(X_tile)
            if bad.any():
                X_tile[bad] = np.take(train_medians, np.where(bad)[1])
        pred_tile = model.predict(X_tile).astype(np.float32, copy=False)
        y_pred[r0:r1, :] = pred_tile.reshape((r1 - r0, W))

    return y_true, y_pred


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("linear_baseline_simple")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)sZ | %(levelname)s | %(message)s", "%Y-%m-%dT%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def main(
    *,
    dataset_key: str = "madurai_30m",
    start: Optional[str] = None,
    end: Optional[str] = None,
    target: Optional[str] = None,
    features: Optional[List[str]] = None,
    test_frac: float = 0.2,
    max_samples_per_time: int = 4000,
    max_total_samples: int = 40000,
    tile_size: int = 256,
    max_tiles_per_time: int = 8,
) -> None:
    logger = setup_logging()
    md = make_madurai_data(consolidated=False)

    ds_meta = md.get_dataset(dataset_key)
    td = detect_time_dim(ds_meta)
    if td is None:
        raise ValueError(f"{dataset_key} has no time dimension; need time to split train/test.")
    ds_meta = ds_meta.assign_coords({td: pd.to_datetime(ds_meta[td].values)})
    times = pd.DatetimeIndex(ds_meta[td].values).sort_values()
    if start:
        times = times[times >= pd.Timestamp(start)]
    if end:
        times = times[times <= pd.Timestamp(end)]

    common_dates_path = PROJECT_ROOT / "common_dates.csv"
    if common_dates_path.exists():
        common_df = pd.read_csv(common_dates_path)
        common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna()
        common_dates = pd.DatetimeIndex(common_dates).sort_values()
        times = times[times.isin(common_dates)]
        logger.info("common_dates_filtered=%d", len(times))

    if len(times) == 0:
        raise RuntimeError("No dates selected for training/testing.")

    time_index_map = {pd.Timestamp(t): i for i, t in enumerate(pd.DatetimeIndex(ds_meta[td].values))}

    if target and "band_01" not in target.lower():
        raise ValueError("Only Landsat band_01 is supported as target.")
    tgt = target or "labels_30m/landsat/band_01"
    if tgt not in ds_meta.data_vars:
        raise KeyError(f"target not found: {tgt}")

    feats = features or default_feature_vars(ds_meta, tgt)
    logger.info("target=%s n_features=%d", tgt, len(feats))

    valid_map = {}
    for v in feats:
        valid_var = _valid_var_for(v, ds_meta)
        if valid_var:
            valid_map[v] = valid_var
    target_valid = _valid_var_for(tgt, ds_meta)

    # Split by time
    n_test = max(1, int(round(len(times) * test_frac)))
    test_times = times[-n_test:]
    train_times = times[:-n_test]
    logger.info("train_times=%d test_times=%d", len(train_times), len(test_times))

    include_vars = {tgt, *feats}
    include_vars.update(valid_map.values())
    if target_valid:
        include_vars.add(target_valid)

    rng = np.random.default_rng(RANDOM_SEED)
    spec = SampleSpec(
        max_samples_per_time=max_samples_per_time,
        max_total_samples=max_total_samples,
        tile_size=tile_size,
        max_tiles_per_time=max_tiles_per_time,
    )

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    total = 0
    for t in train_times:
        dts = load_subset(
            md,
            dataset_key,
            vars_include=sorted(include_vars),
            start=pd.Timestamp(t),
            end=pd.Timestamp(t),
        )
        Xs, ys = sample_xy_for_time(
            dts,
            td,
            t,
            tgt,
            feats,
            spec,
            rng,
            valid_map=valid_map,
            target_valid=target_valid,
            time_index_map=time_index_map,
        )
        del dts
        gc.collect()
        if ys.size == 0:
            continue
        X_list.append(Xs)
        y_list.append(ys)
        total = int(sum(len(y) for y in y_list))
        if total >= spec.max_total_samples:
            break

    if not y_list:
        raise RuntimeError("No training samples found.")
    X_train = np.concatenate(X_list, axis=0)[: spec.max_total_samples]
    y_train = np.concatenate(y_list, axis=0)[: spec.max_total_samples]
    train_medians = np.nanmedian(X_train, axis=0)
    train_medians = np.where(np.isfinite(train_medians), train_medians, 0.0)
    X_train = impute_with_train_medians(X_train, X_train)

    model = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    model.fit(X_train, y_train)
    y_tr_pred = model.predict(X_train).astype(np.float32, copy=False)
    tr_err = y_tr_pred - y_train
    train_rmse = float(np.sqrt(np.mean(tr_err ** 2))) if tr_err.size else float("nan")
    logger.info("fit_done train_samples=%d train_rmse=%.6f", X_train.shape[0], train_rmse)

    # Evaluation (full-map + sampled for comparability)
    rows = []
    eval_spec = SampleSpec(
        max_samples_per_time=max(1000, max_samples_per_time // 2),
        max_total_samples=10000,
        tile_size=tile_size,
        max_tiles_per_time=max_tiles_per_time,
    )
    for t in test_times:
        dts = load_subset(
            md,
            dataset_key,
            vars_include=sorted(include_vars),
            start=pd.Timestamp(t),
            end=pd.Timestamp(t),
        )
        y_true, y_pred = predict_full_map_simple(
            model,
            dts,
            td,
            t,
            tgt,
            feats,
            tile_rows=tile_size,
            valid_map=valid_map,
            target_valid=target_valid,
            time_index_map=time_index_map,
            train_medians=train_medians,
        )
        m = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(m):
            continue
        err = y_pred[m] - y_true[m]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        rmse_sum = float(np.sqrt(np.sum(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        n_full = int(np.sum(m))

        Xs, ys = sample_xy_for_time(
            dts,
            td,
            t,
            tgt,
            feats,
            eval_spec,
            rng,
            valid_map=valid_map,
            target_valid=target_valid,
            time_index_map=time_index_map,
        )
        if ys.size > 0:
            Xs = impute_with_train_medians(X_train, Xs)
            y_pred_s = model.predict(Xs).astype(np.float32, copy=False)
            m_s = np.isfinite(ys) & np.isfinite(y_pred_s)
            if np.any(m_s):
                err_s = y_pred_s[m_s] - ys[m_s]
                rmse_sampled = float(np.sqrt(np.mean(err_s ** 2)))
            else:
                rmse_sampled = float("nan")
            n_sampled = int(np.sum(m_s))
        else:
            rmse_sampled = float("nan")
            n_sampled = 0
        rows.append(
            {
                "time": str(pd.Timestamp(t).date()),
                "rmse": rmse,
                "rmse_sum": rmse_sum,
                "rmse_sampled": rmse_sampled,
                "mae": mae,
                "n": n_full,
                "n_sampled": n_sampled,
            }
        )
        del y_true, y_pred, dts
        gc.collect()

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "linear_simple_metrics.csv"
    df.to_csv(out_csv, index=False)
    logger.info("saved_metrics=%s", out_csv)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("Simple linear baseline (fast)")
    p.add_argument("--dataset", default="madurai_30m", choices=["madurai", "madurai_30m", "madurai_alphaearth_30m"])
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--features", default=None, nargs="+")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--max-samples-per-time", type=int, default=4000)
    p.add_argument("--max-total-samples", type=int, default=40000)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--max-tiles-per-time", type=int, default=8)
    args = p.parse_args()

    main(
        dataset_key=args.dataset,
        start=args.start,
        end=args.end,
        target=args.target,
        features=args.features,
        test_frac=args.test_frac,
        max_samples_per_time=args.max_samples_per_time,
        max_total_samples=args.max_total_samples,
        tile_size=args.tile_size,
        max_tiles_per_time=args.max_tiles_per_time,
    )
