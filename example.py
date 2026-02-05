from __future__ import annotations

import gc
import numpy as np
import pandas as pd

from helper import make_madurai_data, load_subset, list_vars_from_store
from helper_module.helper.utils import open_zarr_tree


def main() -> None:
    md = make_madurai_data()

    ds = load_subset(
        md,
        "madurai_30m",
        vars_include=[
            "labels_30m/landsat/band_01",
            "products_30m/sentinel1/band_01",
            "products_30m/sentinel2/band_01",
            "static_30m/dem/band_01",
            "static_30m/dem/band_02",
            "static_30m/dem/band_03",
        ],
        start="2019-01-01",
        end="2019-03-31",
    )

    ds_dem = load_subset(
        md,
        "madurai",
        vars_include=[
            "static/dem/band_01",
            "static/dem/band_02",
            "static/dem/band_03",
        ],
    )

    td = "time" if "time" in ds.coords else list(ds.coords)[0]
    t0 = pd.Timestamp(ds[td].values[2])

    dts = ds.sel({td: t0})
    y = np.asarray(dts["labels_30m/landsat/band_01"].data, dtype=np.float32)
    s1 = np.asarray(dts["products_30m/sentinel1/band_01"].data, dtype=np.float32)
    s2 = np.asarray(dts["products_30m/sentinel2/band_01"].data, dtype=np.float32)
    def _dem_array(da):
        if "time" in da.dims:
            return np.asarray(da.isel(time=0).data, dtype=np.float32)
        return np.asarray(da.data, dtype=np.float32)

    # Load DEM from a separate subset to avoid time slicing issues.
    ds_dem30 = load_subset(
        md,
        "madurai_30m",
        vars_include=[
            "static_30m/dem/band_01",
            "static_30m/dem/band_02",
            "static_30m/dem/band_03",
        ],
    )
    dem1 = _dem_array(ds_dem30["static_30m/dem/band_01"])
    dem2 = _dem_array(ds_dem30["static_30m/dem/band_02"])
    dem3 = _dem_array(ds_dem30["static_30m/dem/band_03"])
    dem_stack = np.stack([dem1, dem2, dem3], axis=0)
    dem_valid = np.isfinite(dem_stack)
    best_idx = int(np.argmax(dem_valid.sum(axis=(1, 2))))
    dem = [dem1, dem2, dem3][best_idx]

    X = np.stack([s1, s2, dem], axis=-1)

    print("time:", str(t0.date()))
    print("target (y) shape:", y.shape, "dtype:", y.dtype)
    print("features (X) shape:", X.shape, "dtype:", X.dtype)
    print("sample y[0,0]:", y[12, 324])
    print("sample X[0,0,:]:", X[0, 0, :])
    print("dem valid fraction per band:", dem_valid.reshape(3, -1).mean(axis=1))
    print("dem chosen band:", f"static_30m/dem/band_0{best_idx+1}")
    dem_madurai = []
    for i in range(1, 4):
        da = ds_dem[f"static/dem/band_0{i}"]
        if "time" in da.dims:
            arr = np.asarray(da.isel(time=0).data, dtype=np.float32)
        else:
            arr = np.asarray(da.data, dtype=np.float32)
        dem_madurai.append(np.isfinite(arr).mean())
    print("dem (madurai.zarr) valid fraction per band:", dem_madurai)

    def summarize(name: str, arr: np.ndarray) -> None:
        vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            print(f"{name}: no finite values")
            return
        pcts = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99])
        print(
            f"{name}: n={vals.size} min={vals.min():.3f} max={vals.max():.3f} "
            f"mean={vals.mean():.3f} std={vals.std():.3f}"
        )
        print(
            f"{name}: p01={pcts[0]:.3f} p05={pcts[1]:.3f} p25={pcts[2]:.3f} "
            f"p50={pcts[3]:.3f} p75={pcts[4]:.3f} p95={pcts[5]:.3f} p99={pcts[6]:.3f}"
        )
        hist, edges = np.histogram(vals, bins=20)
        print(f"{name}: histogram bins={len(hist)}")
        for i in range(len(hist)):
            print(f"  {edges[i]:.3f} to {edges[i+1]:.3f}: {hist[i]}")

    summarize("y (target)", y)
    for i, feat in enumerate(["sentinel1_b01", "sentinel2_b01", f"dem_band_0{best_idx+1}"]):
        summarize(f"X[{i}] ({feat})", X[..., i])

    # ----------------------------
    # Scan all variables in madurai_30m.zarr (memory-safe)
    # ----------------------------
    print("\n[Scan] madurai_30m variable distributions (sampled)")
    vars_all = list_vars_from_store("/home/naren-root/Documents/FYP2/Project/madurai_30m.zarr")
    rng = np.random.default_rng(0)
    max_samples = 200_000
    band_counts = {}
    for v in vars_all:
        parts = v.split("/")
        if len(parts) >= 2:
            key = "/".join(parts[:2])
        else:
            key = parts[0]
        band_counts[key] = band_counts.get(key, 0) + 1

    def _pick_time(da, target_time: pd.Timestamp):
        if "time" not in da.dims:
            return da
        times = pd.to_datetime(da["time"].values, errors="coerce")
        if np.all(pd.isna(times)):
            return da.isel(time=0)
        idx = int(np.argmin(np.abs(times - target_time)))
        return da.isel(time=idx)

    for v in vars_all:
        ds_v = open_zarr_tree("/home/naren-root/Documents/FYP2/Project/madurai_30m.zarr", include_vars=[v])
        da = _pick_time(ds_v[v], t0)
        data = np.asarray(da.data, dtype=np.float32)
        vals = data[np.isfinite(data)]
        finite_frac = float(vals.size / data.size) if data.size else 0.0
        if vals.size == 0:
            print(f"{v}: finite_frac={finite_frac:.3f} (no finite values)")
        else:
            if vals.size > max_samples:
                idx = rng.choice(vals.size, size=max_samples, replace=False)
                vals = vals[idx]
            pcts = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99])
            print(
                f"{v}: finite_frac={finite_frac:.3f} "
                f"min={vals.min():.3f} max={vals.max():.3f} "
                f"mean={vals.mean():.3f} std={vals.std():.3f} "
                f"p01={pcts[0]:.3f} p50={pcts[3]:.3f} p99={pcts[6]:.3f}"
            )
        del ds_v, da, data, vals
        gc.collect()

    print("\n[Summary] band counts by source (madurai_30m)")
    for k in sorted(band_counts):
        print(f"{k}: {band_counts[k]}")

    # ----------------------------
    # MODIS/VIIRS scan (madurai.zarr)
    # ----------------------------
    print("\n[Scan] madurai variable distributions (MODIS/VIIRS)")
    vars_m = list_vars_from_store("/home/naren-root/Documents/FYP2/Project/madurai.zarr")
    mv = [v for v in vars_m if v.startswith("products/modis/") or v.startswith("products/viirs/")]
    rng = np.random.default_rng(0)
    max_samples = 200_000
    for v in mv:
        ds_v = open_zarr_tree("/home/naren-root/Documents/FYP2/Project/madurai.zarr", include_vars=[v])
        da = ds_v[v]
        if "time" in da.dims:
            da = da.isel(time=0)
        data = np.asarray(da.data, dtype=np.float32)
        vals = data[np.isfinite(data)]
        finite_frac = float(vals.size / data.size) if data.size else 0.0
        if vals.size == 0:
            print(f"{v}: finite_frac={finite_frac:.3f} (no finite values)")
        else:
            if vals.size > max_samples:
                idx = rng.choice(vals.size, size=max_samples, replace=False)
                vals = vals[idx]
            pcts = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99])
            print(
                f"{v}: finite_frac={finite_frac:.3f} "
                f"min={vals.min():.3f} max={vals.max():.3f} "
                f"mean={vals.mean():.3f} std={vals.std():.3f} "
                f"p01={pcts[0]:.3f} p50={pcts[3]:.3f} p99={pcts[6]:.3f}"
            )
        del ds_v, da, data, vals
        gc.collect()

    # Quick check for VIIRS day LST + cloud mask
    viirs_day = "products/viirs/LST_1KM_DAY_RAW"
    viirs_cloud = "products/viirs/CLOUD_DAY"
    if viirs_day in mv and viirs_cloud in mv:
        ds_viirs = open_zarr_tree(
            "/home/naren-root/Documents/FYP2/Project/madurai.zarr",
            include_vars=[viirs_day, viirs_cloud],
        )
        da_lst = ds_viirs[viirs_day]
        da_cloud = ds_viirs[viirs_cloud]
        if "time" in da_lst.dims:
            da_lst = da_lst.isel(time=0)
        if "time" in da_cloud.dims:
            da_cloud = da_cloud.isel(time=0)
        lst = np.asarray(da_lst.data, dtype=np.float32)
        cloud = np.asarray(da_cloud.data, dtype=np.float32)
        mask = np.isfinite(cloud) & (cloud == 0) & np.isfinite(lst) & (lst > 0)
        lst_masked = np.where(mask, lst, np.nan)
        vals = lst_masked[np.isfinite(lst_masked)]
        print("\n[VIIRS] masked day LST stats")
        if vals.size:
            pcts = np.percentile(vals, [1, 5, 50, 95, 99])
            print(
                f"finite_frac={vals.size / lst.size:.3f} "
                f"min={vals.min():.3f} max={vals.max():.3f} "
                f"p01={pcts[0]:.3f} p50={pcts[2]:.3f} p99={pcts[4]:.3f}"
            )
        else:
            print("no finite values after cloud mask")
        del ds_viirs, da_lst, da_cloud, lst, cloud, lst_masked, vals
        gc.collect()


if __name__ == "__main__":
    main()
