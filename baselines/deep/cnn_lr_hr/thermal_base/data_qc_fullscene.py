import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


def _to_str(arr):
    return np.array([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr])


def _qc_mask(qc: np.ndarray) -> np.ndarray:
    qc = qc.astype(np.float32, copy=False)
    finite = np.isfinite(qc)
    if not finite.any():
        return finite
    qc_max = float(np.nanmax(qc))
    if qc_max <= 1:
        return finite & (qc <= 1)
    if qc_max <= 2:
        return finite & (qc <= 2)
    if qc_max <= 3:
        return finite & (qc <= 2)
    if qc_max <= 10:
        return finite & (qc <= 3)
    return finite


def _extract_modis(modis_lr, lst_min, lst_max):
    if modis_lr.shape[0] >= 6:
        lst = modis_lr[0].astype(np.float32)
        qc = modis_lr[4].astype(np.float32)
    elif modis_lr.shape[0] >= 2:
        lst = modis_lr[0].astype(np.float32)
        qc = modis_lr[1].astype(np.float32)
    else:
        lst = modis_lr[0].astype(np.float32)
        qc = np.zeros_like(lst, dtype=np.float32)
    valid_qc = _qc_mask(qc)
    if float(np.mean(valid_qc)) < 0.01:
        valid_qc = np.isfinite(qc)
    lst = np.where(np.isfinite(lst) & (lst != -9999.0), lst, np.nan)
    if np.isfinite(lst).any() and np.nanmedian(lst) > 200:
        lst = lst - 273.15
    valid_lst = np.isfinite(lst) & (lst >= lst_min) & (lst <= lst_max)
    valid = valid_qc & valid_lst
    return valid


def _extract_viirs(viirs_lr, lst_min, lst_max):
    if viirs_lr.shape[0] >= 4:
        lst = viirs_lr[0].astype(np.float32)
        qc = viirs_lr[2].astype(np.float32)
    elif viirs_lr.shape[0] >= 2:
        lst = viirs_lr[0].astype(np.float32)
        qc = viirs_lr[1].astype(np.float32)
    else:
        lst = viirs_lr[0].astype(np.float32)
        qc = np.zeros_like(lst, dtype=np.float32)
    valid_qc = _qc_mask(qc)
    if float(np.mean(valid_qc)) < 0.01:
        valid_qc = np.isfinite(qc)
    lst = np.where(np.isfinite(lst) & (lst != -9999.0), lst, np.nan)
    if np.isfinite(lst).any() and np.nanmedian(lst) > 200:
        lst = lst - 273.15
    valid_lst = np.isfinite(lst) & (lst >= lst_min) & (lst <= lst_max)
    valid = valid_qc & valid_lst
    return valid


def _landsat_to_celsius(arr, attrs):
    arr = arr.astype(np.float32, copy=False)
    arr = np.where(arr == 149, np.nan, arr)
    scale = attrs.get("scale_factor", attrs.get("scale", 1.0))
    offset = attrs.get("add_offset", attrs.get("offset", 0.0))
    if scale is None:
        scale = 1.0
    if offset is None:
        offset = 0.0
    if scale != 1.0 or offset != 0.0:
        arr = arr * float(scale) + float(offset)
    if np.isfinite(arr).any() and np.nanmedian(arr) > 200:
        arr = arr - 273.15
    return arr


def _apply_range_mask(arr, lst_min, lst_max):
    valid = np.isfinite(arr) & (arr >= lst_min) & (arr <= lst_max)
    return valid


def _read_time(root_daily):
    if "time" in root_daily:
        time_group = root_daily["time"]
        if "daily" in time_group and hasattr(time_group["daily"], "shape"):
            raw = time_group["daily"][:]
            return _parse_time_raw(raw)
    for key in ("time", "dates", "date"):
        if key in root_daily:
            obj = root_daily[key]
            if hasattr(obj, "shape"):
                raw = obj[:]
                return _parse_time_raw(raw)
    for key in ("time", "dates", "date"):
        if key in root_daily.attrs:
            raw = root_daily.attrs[key]
            return _parse_time_raw(raw)
    # fallback: find any 1D string-like array at root
    for key in root_daily.keys():
        obj = root_daily[key]
        if hasattr(obj, "shape") and len(obj.shape) == 1:
            try:
                raw = obj[:]
                ts = _parse_time_raw(raw)
                if ts.notna().any():
                    return ts
            except Exception:
                continue
    raise RuntimeError("No time/dates array found in daily zarr")


def _parse_time_raw(raw):
    arr = np.asarray(raw)
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.to_datetime(arr, errors="coerce")
    if np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return pd.to_datetime([], errors="coerce")
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmin > 1e7 and vmax < 3e9 and np.all(np.floor(finite) == finite):
            return pd.to_datetime(finite.astype(np.int64).astype(str), format="%Y%m%d", errors="coerce")
        if vmax > 1e12:
            return pd.to_datetime(arr, unit="ms", errors="coerce")
        if vmax > 1e9:
            return pd.to_datetime(arr, unit="s", errors="coerce")
        if vmax < 100000:
            return pd.to_datetime(arr, unit="D", origin="unix", errors="coerce")
    s = _to_str(arr)
    if s.size > 0:
        first = str(s.flat[0])
        if len(first) == 8 and first.isdigit():
            return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        if len(first) == 10 and first[4] == "-" and first[7] == "-":
            return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        if len(first) == 10 and first[4] == "_" and first[7] == "_":
            return pd.to_datetime(s, format="%Y_%m_%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-30m", type=str, default="/home/naren-root/Documents/FYP2/Project/madurai_30m.zarr")
    ap.add_argument("--root-daily", type=str, default="/home/naren-root/Documents/FYP2/Project/madurai.zarr")
    ap.add_argument("--common-dates", type=str, default="/home/naren-root/Documents/FYP2/Project/common_dates.csv")
    ap.add_argument("--out", type=str, default="/home/naren-root/Documents/FYP2/Project/baselines/deep/cnn_lr_hr/thermal_base/data_qc_fullscene.csv")
    ap.add_argument("--time-key", type=str, default="")
    ap.add_argument("--lst-min", type=float, default=10.0)
    ap.add_argument("--lst-max", type=float, default=70.0)
    ap.add_argument("--min-target", type=float, default=0.30)
    ap.add_argument("--min-modis", type=float, default=0.05)
    ap.add_argument("--min-viirs", type=float, default=0.05)
    ap.add_argument("--min-static", type=float, default=0.90)
    args = ap.parse_args()

    root_30m = zarr.open_group(str(Path(args.root_30m)), mode="r")
    root_daily = zarr.open_group(str(Path(args.root_daily)), mode="r")

    if args.time_key:
        obj = root_daily[args.time_key]
        raw = obj[:] if hasattr(obj, "shape") else obj
        daily_times = _parse_time_raw(raw).to_period("D").to_timestamp()
    else:
        daily_times = _read_time(root_daily).to_period("D").to_timestamp()
    if daily_times.isna().all():
        raise RuntimeError("daily_times parse failed; use --time-key to specify")
    print(f"daily_times total={len(daily_times)} valid={int(daily_times.notna().sum())}")
    daily_map = {pd.Timestamp(t).date(): i for i, t in enumerate(daily_times) if pd.notna(t)}

    common = pd.read_csv(args.common_dates)
    if "date" in common.columns:
        dates = pd.to_datetime(common["date"], errors="coerce")
    else:
        dates = pd.to_datetime(common.iloc[:, 0], errors="coerce")
    dates = dates.dropna().dt.to_period("D").dt.to_timestamp()
    print(f"common_dates rows={len(dates)}")

    landsat = root_30m["labels_30m"]["landsat"]["data"]
    landsat_attrs = dict(root_30m["labels_30m"]["landsat"].attrs)
    modis = root_daily["products"]["modis"]["data"]
    viirs = root_daily["products"]["viirs"]["data"]
    dem = root_30m["static_30m"]["dem"]["data"][0, 0]
    world = root_30m["static_30m"]["worldcover"]["data"][0, 0]
    dyn = root_30m["static_30m"]["dynamic_world"]["data"][0, 0]

    static_valid = np.isfinite(dem) & np.isfinite(world) & np.isfinite(dyn)
    static_valid_frac = float(np.mean(static_valid))

    rows = []
    for d in dates:
        t = daily_map.get(pd.Timestamp(d).date(), None)
        if t is None:
            continue
        t = int(t)
        try:
            y = landsat[t, 0]
            y = _landsat_to_celsius(y, landsat_attrs)
            target_valid = _apply_range_mask(y, args.lst_min, args.lst_max)
            target_frac = float(np.mean(target_valid))
        except Exception:
            target_frac = 0.0
        try:
            modis_valid = _extract_modis(modis[t], args.lst_min, args.lst_max)
            modis_frac = float(np.mean(modis_valid))
        except Exception:
            modis_frac = 0.0
        try:
            viirs_valid = _extract_viirs(viirs[t], args.lst_min, args.lst_max)
            viirs_frac = float(np.mean(viirs_valid))
        except Exception:
            viirs_frac = 0.0

        ok = (target_frac >= args.min_target) and (
            (modis_frac >= args.min_modis) or (viirs_frac >= args.min_viirs) or (static_valid_frac >= args.min_static)
        )

        rows.append(
            {
                "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                "t_index": t,
                "landsat_valid_frac": target_frac,
                "modis_valid_frac": modis_frac,
                "viirs_valid_frac": viirs_frac,
                "static_valid_frac": static_valid_frac,
                "ok": bool(ok),
            }
        )

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    total = len(df)
    ok_count = int(df["ok"].sum()) if total else 0
    print(f"saved {total} rows to {out_path}")
    print(f"ok={ok_count} ({ok_count}/{total})")


if __name__ == "__main__":
    main()
