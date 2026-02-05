from pathlib import Path
import re

import numpy as np
import pandas as pd
import zarr

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ZARR_PATH = PROJECT_ROOT / "madurai.zarr"

MODIS_DIR = Path("/home/naren-root/Documents/FYP2/data/modis")
VIIRS_DIR = Path("/home/naren-root/Documents/FYP2/data/viirs")

# Band layout (0-based indices)
# MODIS: [LST_Day_C, LST_Night_C, QC_Day, QC_Night, valid_day, valid_night]
# VIIRS: [LST_DAY, LST_NIGHT, CLOUD_DAY, CLOUD_NIGHT]
MODIS_NODATA = -9999.0
VIIRS_LST_NODATA = 0.0
LST_MIN_VALID_K = 273.0
MODIS_MIN_VALID_C = 0.0


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


def _resize_nearest(data: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    _, in_h, in_w = data.shape
    if (in_h, in_w) == (out_h, out_w):
        return data
    y_idx = np.floor(np.linspace(0, in_h - 1, out_h)).astype(np.int64)
    x_idx = np.floor(np.linspace(0, in_w - 1, out_w)).astype(np.int64)
    return data[:, y_idx][:, :, x_idx]


def _build_date_map(folder: Path, pattern: str) -> dict:
    date_map = {}
    regex = re.compile(pattern, re.IGNORECASE)
    for path in folder.rglob("*.tif"):
        m = regex.search(path.name)
        if not m:
            continue
        date_map[m.group(1)] = path
    return date_map


def _clean_modis_lst(arr: np.ndarray) -> np.ndarray:
    # MODIS LST is in Celsius
    arr = np.where(arr == MODIS_NODATA, np.nan, arr)
    arr = np.where(arr <= MODIS_MIN_VALID_C, np.nan, arr)
    return arr


def _clean_viirs_lst(arr: np.ndarray) -> np.ndarray:
    arr = np.where(arr == VIIRS_LST_NODATA, np.nan, arr)
    arr = np.where(arr < LST_MIN_VALID_K, np.nan, arr)
    return arr


def main():
    root = zarr.open_group(str(ZARR_PATH), mode="r+")
    modis_arr = root["products"]["modis"]["data"]
    viirs_arr = root["products"]["viirs"]["data"]

    daily_raw = root["time"]["daily"][:]
    daily_dates = np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in daily_raw]
    )
    daily_dates = np.array([d.replace("-", "_") for d in daily_dates])
    print(f"daily_dates count={len(daily_dates)} sample={daily_dates[:5].tolist()}")

    modis_files = _build_date_map(
        MODIS_DIR, r"(?:MODIS|MOD11A1)_(\d{4}_\d{2}_\d{2})_Madurai"
    )
    viirs_files = _build_date_map(
        VIIRS_DIR, r"viirs_raw_(\d{4}_\d{2}_\d{2})_4band"
    )
    print(f"modis files found={len(modis_files)} sample={list(modis_files)[:5]}")
    print(f"viirs files found={len(viirs_files)} sample={list(viirs_files)[:5]}")

    modis_written = 0
    viirs_written = 0
    modis_missing = 0
    viirs_missing = 0
    modis_bad = 0
    viirs_bad = 0

    for t, date_str in enumerate(daily_dates):
        modis_path = modis_files.get(date_str)
        if modis_path is not None:
            try:
                data = _read_tif(modis_path).astype(np.float32)
                data = _resize_nearest(data, modis_arr.shape[-2], modis_arr.shape[-1])
                # write all bands; clean LST bands only
                if data.shape[0] >= 1:
                    data[0] = _clean_modis_lst(data[0])
                if data.shape[0] >= 2:
                    data[1] = _clean_modis_lst(data[1])
                modis_arr[t, : data.shape[0], :, :] = data
                modis_written += 1
            except Exception as exc:
                modis_bad += 1
                print(f"modis read fail date={date_str} path={modis_path} err={exc}")
        else:
            modis_missing += 1

        viirs_path = viirs_files.get(date_str)
        if viirs_path is not None:
            try:
                data = _read_tif(viirs_path).astype(np.float32)
                data = _resize_nearest(data, viirs_arr.shape[-2], viirs_arr.shape[-1])
                # write all bands; clean LST bands only
                if data.shape[0] >= 1:
                    data[0] = _clean_viirs_lst(data[0])
                if data.shape[0] >= 2:
                    data[1] = _clean_viirs_lst(data[1])
                viirs_arr[t, : data.shape[0], :, :] = data
                viirs_written += 1
            except Exception as exc:
                viirs_bad += 1
                print(f"viirs read fail date={date_str} path={viirs_path} err={exc}")
        else:
            viirs_missing += 1

    print(f"MODIS written: {modis_written}")
    print(f"MODIS missing: {modis_missing} bad: {modis_bad}")
    print(f"VIIRS written: {viirs_written}")
    print(f"VIIRS missing: {viirs_missing} bad: {viirs_bad}")


if __name__ == "__main__":
    main()
