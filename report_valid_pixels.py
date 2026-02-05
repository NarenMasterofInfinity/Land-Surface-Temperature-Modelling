from pathlib import Path
import re

import numpy as np
import pandas as pd
import zarr

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"
COMMON_DATES = PROJECT_ROOT / "common_dates.csv"
MODIS_TIF_DIR = Path("/home/naren-root/Documents/FYP2/data/modis")
VIIRS_TIF_DIR = Path("/home/naren-root/Documents/FYP2/data/viirs")

LANDSAT_NODATA = 149
LANDSAT_MIN_VALID_K = 273.0
MODIS_NODATA = -9999.0
VIIRS_LST_NODATA = 0.0
MODIS_MIN_VALID_C = 0.0


def _to_str(arr):
    return np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
    )


def _build_date_map(folder: Path, pattern: str) -> dict:
    date_map = {}
    regex = re.compile(pattern, re.IGNORECASE)
    for path in folder.rglob("*.tif"):
        m = regex.search(path.name)
        if not m:
            continue
        date_map[m.group(1)] = path
    return date_map


def _read_tif(path: Path) -> np.ndarray:
    try:
        import rasterio

        with rasterio.open(path) as ds:
            data = ds.read()
        return data
    except Exception:
        try:
            from osgeo import gdal
        except Exception as exc:
            raise RuntimeError("Missing raster reader: install rasterio or GDAL Python bindings") from exc

        ds = gdal.Open(str(path))
        if ds is None:
            raise RuntimeError(f"Failed to open {path}")
        data = ds.ReadAsArray()
        if data.ndim == 2:
            data = data[None, ...]
        return data


def _valid_all_channels(arr):
    if arr.ndim < 3:
        return np.isfinite(arr)
    return np.isfinite(arr).all(axis=0)


def _landsat_valid(arr):
    arr = np.where(arr == LANDSAT_NODATA, np.nan, arr)
    arr = np.where(arr < LANDSAT_MIN_VALID_K, np.nan, arr)
    return np.isfinite(arr)


def _modis_valid(arr):
    if arr.shape[0] < 2:
        return _valid_all_channels(arr)
    lst = arr[0]
    if arr.shape[0] >= 6:
        # MODIS 6-band layout: valid_day at band 4 (1=valid)
        mask = arr[4]
        lst = np.where(mask == 1, lst, np.nan)
    else:
        # 2-band layout: mask is channel 1 (0=valid)
        mask = arr[1]
        lst = np.where(mask == 0, lst, np.nan)
    lst = np.where(lst == MODIS_NODATA, np.nan, lst)
    lst = np.where(lst <= MODIS_MIN_VALID_C, np.nan, lst)
    return np.isfinite(lst)


def _viirs_valid(arr):
    if arr.shape[0] < 2:
        return _valid_all_channels(arr)
    lst = arr[0]
    if arr.shape[0] >= 6:
        # VIIRS 6-band layout: cloudfree_day at band 4 (1=valid)
        mask = arr[4]
        lst = np.where(mask == 1, lst, np.nan)
    elif arr.shape[0] >= 4:
        # 4-band layout: cloud day at band 2 (0/1 valid, 2/3 invalid)
        mask = arr[2]
        lst = np.where(mask <= 1, lst, np.nan)
    else:
        # 2-band layout: mask is channel 1 (0/1 valid)
        mask = arr[1]
        lst = np.where(mask <= 1, lst, np.nan)
    lst = np.where(lst == VIIRS_LST_NODATA, np.nan, lst)
    return np.isfinite(lst)


def _modis_tif_valid(path: Path) -> int:
    data = _read_tif(path).astype(np.float32)
    if data.shape[0] < 5:
        return 0
    lst = data[0]
    valid = data[4] == 1
    lst = np.where(valid, lst, np.nan)
    lst = np.where(lst == MODIS_NODATA, np.nan, lst)
    return int(np.isfinite(lst).sum())


def _viirs_tif_valid(path: Path) -> int:
    data = _read_tif(path).astype(np.float32)
    if data.shape[0] < 3:
        return 0
    lst = data[0]
    cloud = data[2]
    valid = cloud <= 1
    lst = np.where(valid, lst, np.nan)
    lst = np.where(lst == VIIRS_LST_NODATA, np.nan, lst)
    return int(np.isfinite(lst).sum())


def main():
    root_30m = zarr.open_group(str(ROOT_30M), mode="r")
    root_daily = zarr.open_group(str(ROOT_DAILY), mode="r")

    common_df = pd.read_csv(COMMON_DATES)
    common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna()
    common_dates = pd.DatetimeIndex(common_dates).sort_values()

    daily_raw = _to_str(root_30m["time"]["daily"][:])
    monthly_raw = _to_str(root_30m["time"]["monthly"][:])

    daily_times = pd.to_datetime(daily_raw, format="%Y_%m_%d", errors="coerce").dropna()
    monthly_times = pd.to_datetime(monthly_raw, format="%Y_%m", errors="coerce").dropna()

    daily_idx = np.flatnonzero(daily_times.isin(common_dates))
    month_index = pd.DatetimeIndex(common_dates.to_period("M").to_timestamp())
    monthly_idx = np.flatnonzero(monthly_times.isin(month_index))

    monthly_map = {t: i for i, t in enumerate(monthly_times)}
    daily_to_month = []
    for t in daily_times[daily_idx]:
        m = t.to_period("M").to_timestamp()
        daily_to_month.append(monthly_map.get(m, -1))
    daily_to_month = np.array(daily_to_month)
    valid = daily_to_month >= 0
    daily_idx = daily_idx[valid]
    daily_to_month = daily_to_month[valid]
    daily_to_month_map = {int(t): int(m) for t, m in zip(daily_idx, daily_to_month)}

    modis_tif_map = _build_date_map(
        MODIS_TIF_DIR, r"(?:MODIS|MOD11A1)_(\d{4}_\d{2}_\d{2})_Madurai"
    )
    viirs_tif_map = _build_date_map(
        VIIRS_TIF_DIR, r"viirs_raw_(\d{4}_\d{2}_\d{2})_4band"
    )

    g_era5 = root_30m["products_30m"]["era5"]["data"]
    g_landsat = root_30m["labels_30m"]["landsat"]["data"]
    g_s1 = root_30m["products_30m"]["sentinel1"]["data"]
    g_s2 = root_30m["products_30m"]["sentinel2"]["data"]
    g_modis = root_daily["products"]["modis"]["data"]
    g_viirs = root_daily["products"]["viirs"]["data"]

    g_dem = root_30m["static_30m"]["dem"]["data"][0]
    g_world = root_30m["static_30m"]["worldcover"]["data"][0]
    g_dyn = root_30m["static_30m"]["dynamic_world"]["data"][0]

    static_dem_valid = int(_valid_all_channels(g_dem).sum())
    static_world_valid = int(_valid_all_channels(g_world).sum())
    static_dyn_valid = int(_valid_all_channels(g_dyn).sum())

    rows = []
    for t in daily_idx:
        t = int(t)
        m = int(daily_to_month_map[t])
        date_str = daily_times[t].strftime("%Y-%m-%d")
        date_key = date_str.replace("-", "_")

        landsat = g_landsat[t, 0]
        era5 = g_era5[t]
        s1 = g_s1[m]
        s2 = g_s2[m]
        modis = g_modis[t]
        viirs = g_viirs[t]

        row = {
            "date": date_str,
            "landsat_valid_px": int(_landsat_valid(landsat).sum()),
            "era5_valid_px": int(_valid_all_channels(era5).sum()),
            "sentinel1_valid_px": int(_valid_all_channels(s1).sum()),
            "sentinel2_valid_px": int(_valid_all_channels(s2).sum()),
            "modis_valid_px": int(_modis_valid(modis).sum()),
            "viirs_valid_px": int(_viirs_valid(viirs).sum()),
            "modis_tif_valid_px": _modis_tif_valid(modis_tif_map[date_key])
            if date_key in modis_tif_map
            else 0,
            "viirs_tif_valid_px": _viirs_tif_valid(viirs_tif_map[date_key])
            if date_key in viirs_tif_map
            else 0,
            "dem_valid_px": static_dem_valid,
            "world_valid_px": static_world_valid,
            "dynamic_world_valid_px": static_dyn_valid,
        }
        rows.append(row)

    out_path = PROJECT_ROOT / "metrics" / "deep_baselines" / "valid_px_report.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"saved valid pixel report: {out_path}")


if __name__ == "__main__":
    main()
