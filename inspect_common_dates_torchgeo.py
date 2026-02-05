from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from rasterio.crs import CRS
from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import BoundingBox
    from torchgeo.samplers import RandomGeoSampler
    from rtree.index import Index

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"
COMMON_DATES = PROJECT_ROOT / "common_dates.csv"
OUT_CSV = PROJECT_ROOT / "metrics" / "deep_baselines" / "common_dates_quality_torchgeo.csv"

PATCH_SIZE = 128
N_PATCHES = 50


def _to_str(arr):
    return np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
    )


class ZarrPatchDataset(GeoDataset):
    def __init__(self, arrays, height, width):
        super().__init__()
        self.arrays = arrays
        self.height = int(height)
        self.width = int(width)

        self.index = Index()
        self.index.crs = CRS.from_epsg(4326)
        self.index.insert(0, (0, 0, self.width, self.height))
        self.crs = CRS.from_epsg(4326)
        self.res = (1.0, 1.0)
        self.bounds = BoundingBox(0, self.width, 0, self.height, 0, 1)

    def __getitem__(self, query):
        if isinstance(query, BoundingBox):
            x0 = int(max(0, np.floor(query.minx)))
            x1 = int(min(self.width, np.ceil(query.maxx)))
            y0 = int(max(0, np.floor(query.miny)))
            y1 = int(min(self.height, np.ceil(query.maxy)))
        else:
            x0, x1, y0, y1 = 0, self.width, 0, self.height
        sample = {}
        for name, arr in self.arrays.items():
            if arr.ndim == 2:
                sample[name] = arr[y0:y1, x0:x1]
            else:
                sample[name] = arr[:, y0:y1, x0:x1]
        return sample


def _valid_ratio(arr, mask):
    if arr.size == 0:
        return float("nan"), 0
    if mask.size == 0:
        return float("nan"), 0
    total = int(mask.size)
    valid = int(mask.sum())
    ratio = float(valid / total) if total > 0 else float("nan")
    return ratio, valid


def _landsat_valid(y):
    y = y.astype(np.float32, copy=False)
    valid = np.isfinite(y) & (y != 149.0) & (y >= 273.0) & (y != 0.0)
    return valid


def _modis_valid(modis_lr):
    if modis_lr.shape[0] >= 6:
        lst = modis_lr[0].astype(np.float32)
        qc = modis_lr[4].astype(np.float32)
    elif modis_lr.shape[0] >= 2:
        lst = modis_lr[0].astype(np.float32)
        qc = modis_lr[1].astype(np.float32)
    else:
        lst = modis_lr[0].astype(np.float32)
        qc = np.zeros_like(lst, dtype=np.float32)
    valid_qc = np.isfinite(qc) & (qc == 1)
    valid_lst = np.isfinite(lst) & (lst != -9999.0) & (lst > 0)
    return valid_qc & valid_lst


def _viirs_valid(viirs_lr):
    if viirs_lr.shape[0] >= 4:
        lst = viirs_lr[0].astype(np.float32)
        qc = viirs_lr[2].astype(np.float32)
    elif viirs_lr.shape[0] >= 2:
        lst = viirs_lr[0].astype(np.float32)
        qc = viirs_lr[1].astype(np.float32)
    else:
        lst = viirs_lr[0].astype(np.float32)
        qc = np.zeros_like(lst, dtype=np.float32)
    valid_qc = np.isfinite(qc) & (qc <= 1)
    valid_lst = np.isfinite(lst) & (lst != -9999.0) & (lst >= 273.0)
    return valid_qc & valid_lst


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

    landsat_shape = root_30m["labels_30m"]["landsat"]["data"].shape
    H_hr, W_hr = landsat_shape[-2], landsat_shape[-1]

    rows = []
    for t in daily_idx:
        t = int(t)
        date_str = daily_times[t].strftime("%Y-%m-%d")

        era5 = root_30m["products_30m"]["era5"]["data"][t]
        m = daily_to_month_map.get(t, None)
        if m is None:
            print(f"{date_str} skipped (no monthly mapping)")
            continue
        s1 = root_30m["products_30m"]["sentinel1"]["data"][m]
        s2 = root_30m["products_30m"]["sentinel2"]["data"][m]
        dem = root_30m["static_30m"]["dem"]["data"][0]
        world = root_30m["static_30m"]["worldcover"]["data"][0]
        dyn = root_30m["static_30m"]["dynamic_world"]["data"][0]
        y = root_30m["labels_30m"]["landsat"]["data"][t, 0]

        ds = ZarrPatchDataset(
            {
                "era5": era5,
                "s1": s1,
                "s2": s2,
                "dem": dem,
                "world": world,
                "dyn": dyn,
                "landsat": y,
            },
            H_hr,
            W_hr,
        )
        sampler = RandomGeoSampler(ds, size=PATCH_SIZE, length=N_PATCHES)

        ratios = {k: [] for k in ds.arrays.keys()}
        valids = {k: [] for k in ds.arrays.keys()}
        for bbox in sampler:
            sample = ds[bbox]
            for name, arr in sample.items():
                if name == "landsat":
                    m = _landsat_valid(arr)
                else:
                    m = np.isfinite(arr)
                ratio, valid = _valid_ratio(arr, m)
                ratios[name].append(ratio)
                valids[name].append(valid)

        modis_lr = root_daily["products"]["modis"]["data"][t]
        viirs_lr = root_daily["products"]["viirs"]["data"][t]
        modis_mask = _modis_valid(modis_lr)
        viirs_mask = _viirs_valid(viirs_lr)
        modis_ratio, modis_valid = _valid_ratio(modis_mask, modis_mask)
        viirs_ratio, viirs_valid = _valid_ratio(viirs_mask, viirs_mask)

        row = {
            "date": date_str,
            "patch_size": PATCH_SIZE,
            "n_patches": N_PATCHES,
            "modis_valid_ratio": modis_ratio,
            "modis_valid_px": modis_valid,
            "viirs_valid_ratio": viirs_ratio,
            "viirs_valid_px": viirs_valid,
        }
        for name in ratios:
            row[f"{name}_valid_ratio_mean"] = float(np.nanmean(ratios[name]))
            row[f"{name}_valid_ratio_std"] = float(np.nanstd(ratios[name]))
            row[f"{name}_valid_px_mean"] = float(np.nanmean(valids[name]))
        rows.append(row)
        print(f"{date_str} done")

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
