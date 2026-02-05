from pathlib import Path

import numpy as np
import pandas as pd
import zarr

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"
COMMON_DATES = PROJECT_ROOT / "common_dates.csv"

LANDSAT_NODATA = 149
LANDSAT_MIN_VALID_K = 273.0
MODIS_NODATA = -9999.0
VIIRS_LST_NODATA = 0.0


def _to_str(arr):
    return np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
    )


def _landsat_valid(arr):
    arr = np.where(arr == LANDSAT_NODATA, np.nan, arr)
    arr = np.where(arr < LANDSAT_MIN_VALID_K, np.nan, arr)
    return np.isfinite(arr)


def _modis_valid(arr):
    lst = arr[0]
    if arr.shape[0] >= 6:
        mask = arr[4]
        lst = np.where(mask == 1, lst, np.nan)
    else:
        mask = arr[1]
        lst = np.where(mask == 0, lst, np.nan)
    lst = np.where(lst == MODIS_NODATA, np.nan, lst)
    return np.isfinite(lst)


def _viirs_valid(arr):
    lst = arr[0]
    if arr.shape[0] >= 6:
        mask = arr[4]
        lst = np.where(mask == 1, lst, np.nan)
    elif arr.shape[0] >= 4:
        mask = arr[2]
        lst = np.where(mask <= 1, lst, np.nan)
    else:
        mask = arr[1]
        lst = np.where(mask <= 1, lst, np.nan)
    lst = np.where(lst == VIIRS_LST_NODATA, np.nan, lst)
    return np.isfinite(lst)


def main():
    root_30m = zarr.open_group(str(ROOT_30M), mode="r")
    root_daily = zarr.open_group(str(ROOT_DAILY), mode="r")

    common_df = pd.read_csv(COMMON_DATES)
    common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna()
    common_dates = pd.DatetimeIndex(common_dates).sort_values()

    daily_raw_30m = _to_str(root_30m["time"]["daily"][:])
    daily_times_30m = pd.to_datetime(daily_raw_30m, format="%Y_%m_%d", errors="coerce").dropna()

    daily_raw_daily = _to_str(root_daily["time"]["daily"][:])
    daily_times_daily = pd.to_datetime(daily_raw_daily, format="%Y_%m_%d", errors="coerce").dropna()

    missing_in_30m = common_dates.difference(daily_times_30m)
    missing_in_daily = common_dates.difference(daily_times_daily)

    print(f"common_dates={len(common_dates)}")
    print(f"daily_times_30m={len(daily_times_30m)} missing_from_30m={len(missing_in_30m)}")
    print(f"daily_times_daily={len(daily_times_daily)} missing_from_daily={len(missing_in_daily)}")
    if len(missing_in_30m) > 0:
        print("missing_in_30m sample:", [str(d.date()) for d in missing_in_30m[:5]])
    if len(missing_in_daily) > 0:
        print("missing_in_daily sample:", [str(d.date()) for d in missing_in_daily[:5]])

    daily_idx = np.flatnonzero(daily_times_30m.isin(common_dates))

    g_landsat = root_30m["labels_30m"]["landsat"]["data"]
    g_modis = root_daily["products"]["modis"]["data"]
    g_viirs = root_daily["products"]["viirs"]["data"]

    rows = []
    for t in daily_idx:
        t = int(t)
        date_str = daily_times_30m[t].strftime("%Y-%m-%d")
        landsat = g_landsat[t, 0]
        modis = g_modis[t]
        viirs = g_viirs[t]
        rows.append(
            {
                "date": date_str,
                "landsat_valid_px": int(_landsat_valid(landsat).sum()),
                "modis_valid_px": int(_modis_valid(modis).sum()),
                "viirs_valid_px": int(_viirs_valid(viirs).sum()),
            }
        )

    out_path = PROJECT_ROOT / "metrics" / "deep_baselines" / "lst_valid_px_report.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"saved lst-only valid report: {out_path}")


if __name__ == "__main__":
    main()
