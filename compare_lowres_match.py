import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from helper.metrics_image import compute_all

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"
COMMON_DATES = PROJECT_ROOT / "common_dates.csv"

LANDSAT_NODATA = 149
LANDSAT_MIN_VALID_K = 273.0
MODIS_C_TO_K = 273.15
MODIS_NODATA = -9999.0
VIIRS_LST_NODATA = 0.0


def _to_str(arr):
    return np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
    )


def aggregate_to_lowres(hi_res: np.ndarray, H_lr: int, W_lr: int) -> np.ndarray:
    H_hr, W_hr = hi_res.shape
    y_map = np.floor(np.linspace(0, H_lr - 1, H_hr)).astype(np.int64)
    x_map = np.floor(np.linspace(0, W_lr - 1, W_hr)).astype(np.int64)
    yy = np.repeat(y_map[:, None], W_hr, axis=1)
    xx = np.repeat(x_map[None, :], H_hr, axis=0)

    flat_vals = hi_res.reshape(-1)
    flat_yy = yy.reshape(-1)
    flat_xx = xx.reshape(-1)

    valid = np.isfinite(flat_vals)
    flat_vals = flat_vals[valid]
    flat_yy = flat_yy[valid]
    flat_xx = flat_xx[valid]

    idx = flat_yy * W_lr + flat_xx
    sums = np.bincount(idx, weights=flat_vals, minlength=H_lr * W_lr)
    counts = np.bincount(idx, minlength=H_lr * W_lr)
    out = np.full((H_lr * W_lr), np.nan, dtype=np.float64)
    valid_cells = counts > 0
    out[valid_cells] = sums[valid_cells] / counts[valid_cells]
    return out.reshape(H_lr, W_lr)


def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> None:
    metrics = compute_all(y_true, y_pred, mask=mask)
    metrics = {k: metrics[k] for k in ("rmse", "ssim", "psnr", "sam", "cc")}
    print(f"{name} metrics: {metrics} valid_px={int(mask.sum())}")


def modis_lst_mask(modis: np.ndarray, *, to_k: bool) -> tuple[np.ndarray, np.ndarray]:
    modis_lst = modis[0].astype(np.float32)
    if modis.shape[0] >= 6:
        mask = modis[4].astype(np.float32)
        valid = mask == 1
    else:
        mask = modis[1].astype(np.float32)
        valid = mask == 0
    modis_lst = np.where(valid, modis_lst, np.nan)
    modis_lst = np.where(modis_lst == MODIS_NODATA, np.nan, modis_lst)
    if to_k:
        modis_lst = modis_lst + MODIS_C_TO_K
    return modis_lst, mask


def viirs_lst_mask(viirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    viirs_lst = viirs[0].astype(np.float32)
    if viirs.shape[0] >= 6:
        mask = viirs[4].astype(np.float32)
        valid = mask == 1
    elif viirs.shape[0] >= 4:
        mask = viirs[2].astype(np.float32)
        valid = mask <= 1
    else:
        mask = viirs[1].astype(np.float32)
        valid = mask <= 1
    viirs_lst = np.where(valid, viirs_lst, np.nan)
    viirs_lst = np.where(viirs_lst == VIIRS_LST_NODATA, np.nan, viirs_lst)
    return viirs_lst, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="YYYY_MM_DD or YYYY-MM-DD")
    ap.add_argument("--modis-unit", choices=["c", "k"], default="c",
                    help="MODIS LST unit in zarr (c=default, k=convert to Kelvin)")
    ap.add_argument("--loop", action="store_true", help="scan all common_dates and report valid counts")
    args = ap.parse_args()

    root_30m = zarr.open_group(str(ROOT_30M), mode="r")
    root_daily = zarr.open_group(str(ROOT_DAILY), mode="r")

    common_df = pd.read_csv(COMMON_DATES)
    common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna()
    common_dates = pd.DatetimeIndex(common_dates).sort_values()

    daily_raw = _to_str(root_30m["time"]["daily"][:])
    daily_times = pd.to_datetime(daily_raw, format="%Y_%m_%d", errors="coerce").dropna()
    daily_idx = np.flatnonzero(daily_times.isin(common_dates))

    if args.loop:
        valid_rows = []
        for date in common_dates:
            if date not in daily_times:
                continue
            t = int(np.where(daily_times == date)[0][0])

            landsat = root_30m["labels_30m"]["landsat"]["data"][t, 0].astype(np.float32)
            landsat = np.where(landsat == LANDSAT_NODATA, np.nan, landsat)
            landsat = np.where(landsat < LANDSAT_MIN_VALID_K, np.nan, landsat)

            modis = root_daily["products"]["modis"]["data"][t]
            viirs = root_daily["products"]["viirs"]["data"][t]

            modis_lst, _ = modis_lst_mask(modis, to_k=args.modis_unit == "k")
            viirs_lst, _ = viirs_lst_mask(viirs)

            H_lr_modis, W_lr_modis = modis_lst.shape[-2], modis_lst.shape[-1]
            H_lr_viirs, W_lr_viirs = viirs_lst.shape[-2], viirs_lst.shape[-1]

            landsat_modis = aggregate_to_lowres(landsat, H_lr_modis, W_lr_modis)
            landsat_viirs = aggregate_to_lowres(landsat, H_lr_viirs, W_lr_viirs)

            modis_mask = np.isfinite(landsat_modis) & np.isfinite(modis_lst)
            viirs_mask = np.isfinite(landsat_viirs) & np.isfinite(viirs_lst)

            valid_rows.append(
                {
                    "date": str(date.date()),
                    "modis_valid_px": int(modis_mask.sum()),
                    "viirs_valid_px": int(viirs_mask.sum()),
                }
            )
        df = pd.DataFrame(valid_rows)
        out_path = PROJECT_ROOT / "metrics" / "deep_baselines" / "lowres_match_valid_px.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"saved valid-px report: {out_path}")
        return

    date = pd.to_datetime(args.date.replace("_", "-"), errors="coerce")
    if pd.isna(date):
        raise SystemExit("Invalid date format.")
    if date not in daily_times:
        raise SystemExit("Date not in daily timeline.")
    t = int(np.where(daily_times == date)[0][0])

    landsat = root_30m["labels_30m"]["landsat"]["data"][t, 0].astype(np.float32)
    landsat = np.where(landsat == LANDSAT_NODATA, np.nan, landsat)
    landsat = np.where(landsat < LANDSAT_MIN_VALID_K, np.nan, landsat)

    modis = root_daily["products"]["modis"]["data"][t]
    viirs = root_daily["products"]["viirs"]["data"][t]

    modis_lst, _ = modis_lst_mask(modis, to_k=args.modis_unit == "k")
    viirs_lst, _ = viirs_lst_mask(viirs)

    H_lr_modis, W_lr_modis = modis_lst.shape[-2], modis_lst.shape[-1]
    H_lr_viirs, W_lr_viirs = viirs_lst.shape[-2], viirs_lst.shape[-1]

    landsat_modis = aggregate_to_lowres(landsat, H_lr_modis, W_lr_modis)
    landsat_viirs = aggregate_to_lowres(landsat, H_lr_viirs, W_lr_viirs)

    modis_mask = np.isfinite(landsat_modis) & np.isfinite(modis_lst)
    viirs_mask = np.isfinite(landsat_viirs) & np.isfinite(viirs_lst)

    print(f"date={date.date()} modis shape={modis_lst.shape} viirs shape={viirs_lst.shape}")
    print_metrics("modis vs landsat", landsat_modis, modis_lst, modis_mask)
    print_metrics("viirs vs landsat", landsat_viirs, viirs_lst, viirs_mask)

    modis_diff = np.where(modis_mask, modis_lst - landsat_modis, np.nan)
    viirs_diff = np.where(viirs_mask, viirs_lst - landsat_viirs, np.nan)
    print(
        f"modis diff: min={np.nanmin(modis_diff):.3f} max={np.nanmax(modis_diff):.3f} "
        f"mean={np.nanmean(modis_diff):.3f}"
    )
    print(
        f"viirs diff: min={np.nanmin(viirs_diff):.3f} max={np.nanmax(viirs_diff):.3f} "
        f"mean={np.nanmean(viirs_diff):.3f}"
    )


if __name__ == "__main__":
    main()
