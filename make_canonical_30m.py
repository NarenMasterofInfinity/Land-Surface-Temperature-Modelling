#!/usr/bin/env python3
"""
make_canonical_30m.py (single-file)

Build a canonical 30 m Zarr (Landsat grid) from a multi-grid Zarr (madurai.zarr),
with a RAM-safe AlphaEarth path.

What it does:
- Canonical grid = products/landsat grid (30 m).
- Resamples onto canonical grid:
    - products/sentinel2 (bilinear)
    - products/sentinel1 (bilinear)
    - products/era5 (bilinear; explicit ERA5 grid)
    - static/dem (bilinear)
    - static/worldcover (nearest)
    - static/dynamic_world (nearest)
    - products/alphaearth (bilinear)  <-- OPTIMIZED blockwise (low RAM) if enabled
- Copies time axes.
- Copies Landsat into labels_30m/landsat (already canonical; resampling still performed but grid matches).

What it does NOT do:
- Does not resample MODIS/VIIRS here (keep them in raw madurai.zarr).

Run:
  python make_canonical_30m.py \
    --in-zarr  /path/to/madurai.zarr \
    --out-zarr /path/to/madurai_30m.zarr \
    --chunk-hw 256 --compress-level 9 --dtype float32 \
    --include-alphaearth --alphaearth-optimized \
    --tile 512 --band-batch 8

If AlphaEarth still stresses RAM:
  --tile 256 --band-batch 4
"""

import argparse
import time
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr import dtype as zarr_dtype

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds


TEXT_DTYPE = zarr_dtype.VariableLengthUTF8()
ERA5_CRS_STR = "EPSG:4326"
ERA5_ORIGIN_X = 77.9737666615744587
ERA5_ORIGIN_Y = 10.3306257673744977
ERA5_PIXEL_X = 0.08983152841195214677
ERA5_PIXEL_Y = -0.08983152841195214677


# ----------------------------
# Generic helpers
# ----------------------------
def ensure_group(root: zarr.Group, path: str) -> zarr.Group:
    g = root
    for part in path.strip("/").split("/"):
        g = g.require_group(part)
    return g
bil

def create_text_array(group: zarr.Group, name: str, values: List[str]) -> zarr.Array:
    arr = group.create_array(name, shape=(len(values),), dtype=TEXT_DTYPE, overwrite=True)
    if values:
        arr[:] = np.asarray(values, dtype=object)
    return arr


def copy_time_axes(zin: zarr.Group, zout: zarr.Group):
    for name in ["daily", "monthly", "annual"]:
        if "time" in zin and name in zin["time"]:
            src = ensure_group(zin, "time")[name]
        else:
            continue
        g = ensure_group(zout, "time")
        arr = g.create_array(name, shape=src.shape, dtype=TEXT_DTYPE, overwrite=True)
        arr[:] = src[:]


def _parse_crs(crs_str: Optional[str]) -> Optional[CRS]:
    if not crs_str:
        return None
    s = str(crs_str)
    try:
        return CRS.from_string(s)
    except Exception:
        try:
            return CRS.from_wkt(s)
        except Exception:
            return None


def _parse_transform(tup) -> Affine:
    return Affine(*tuple(tup))


def era5_override_grid() -> Dict[str, Any]:
    crs = CRS.from_string(ERA5_CRS_STR)
    transform = Affine(ERA5_PIXEL_X, 0.0, ERA5_ORIGIN_X, 0.0, ERA5_PIXEL_Y, ERA5_ORIGIN_Y)
    return {
        "crs": crs,
        "crs_str": ERA5_CRS_STR,
        "transform": transform,
        "transform_tuple": tuple(transform),
    }


def choose_chunks(shape: Tuple[int, int, int, int], chunk_hw: int) -> Tuple[int, int, int, int]:
    T, C, H, W = shape
    ch = min(chunk_hw, H)
    cw = min(chunk_hw, W)
    cchunk = C if C <= 32 else 8
    return (1, cchunk, ch, cw)


def copy_group_attrs(srcg: zarr.Group, dstg: zarr.Group, extra: Dict[str, Any] = None):
    for k, v in dict(srcg.attrs).items():
        try:
            dstg.attrs[k] = v
        except Exception:
            pass
    if extra:
        for k, v in extra.items():
            dstg.attrs[k] = v


def get_grid_from_group(zroot: zarr.Group, group_path: str) -> Dict[str, Any]:
    g = ensure_group(zroot, group_path)
    crs_obj = _parse_crs(g.attrs.get("crs"))
    if crs_obj is None:
        raise RuntimeError(f"Failed to parse CRS for {group_path}")

    tf = _parse_transform(g.attrs.get("transform"))
    return {
        "crs_obj": crs_obj,
        "crs_str": g.attrs.get("crs"),
        "transform_obj": tf,
        "transform_tuple": tuple(g.attrs.get("transform")),
        "height": int(g.attrs.get("height")),
        "width": int(g.attrs.get("width")),
        "pixel_size_x": g.attrs.get("pixel_size_x"),
        "pixel_size_y": g.attrs.get("pixel_size_y"),
        "reference_file": g.attrs.get("reference_file"),
    }


# ----------------------------
# Simple (safe) resampler for small-ish products
# ----------------------------
def warp_stack_to_dst(
    src: np.ndarray,                 # (C,H,W)
    src_crs: CRS,
    src_transform: Affine,
    dst_crs: CRS,
    dst_transform: Affine,
    dst_hw: Tuple[int, int],
    resampling: Resampling,
    dst_dtype=np.float32
) -> np.ndarray:
    C, _, _ = src.shape
    Hdst, Wdst = dst_hw
    out = np.full((C, Hdst, Wdst), np.nan, dtype=dst_dtype)
    for b in range(C):
        reproject(
            source=src[b, :, :],
            destination=out[b, :, :],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    return out


def warp_mask_to_dst(
    src_mask: np.ndarray,            # (H,W) uint8 0/1
    src_crs: CRS,
    src_transform: Affine,
    dst_crs: CRS,
    dst_transform: Affine,
    dst_hw: Tuple[int, int],
) -> np.ndarray:
    Hdst, Wdst = dst_hw
    out = np.zeros((Hdst, Wdst), dtype=np.uint8)
    reproject(
        source=src_mask.astype(np.uint8),
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    return out


def resample_product_to_30m_simple(
    zin: zarr.Group,
    zout: zarr.Group,
    src_group_path: str,
    dst_group_path: str,
    dst_grid: Dict[str, Any],
    compressor: BloscCodec,
    chunk_hw: int,
    data_dtype: np.dtype,
    is_categorical: bool = False,
    override_src_crs: Optional[CRS] = None,
    override_src_transform: Optional[Affine] = None,
    override_src_crs_str: Optional[str] = None,
    override_src_transform_tuple: Optional[Tuple[float, float, float, float, float, float]] = None,
):
    srcg = ensure_group(zin, src_group_path)
    if "data" not in srcg or "valid" not in srcg:
        raise RuntimeError(f"{src_group_path}: missing data/valid arrays")

    src_data = srcg["data"]     # (T,C,H,W)
    src_valid = srcg["valid"]   # (T,1,H,W)

    labels = list(srcg["labels"][:]) if "labels" in srcg else [str(i) for i in range(src_data.shape[0])]
    band_names = list(srcg["band_names"][:]) if "band_names" in srcg else [f"band_{i+1:02d}" for i in range(src_data.shape[1])]

    src_crs_str = override_src_crs_str or srcg.attrs.get("crs")
    src_transform_tuple = override_src_transform_tuple or srcg.attrs.get("transform")
    src_crs = override_src_crs or _parse_crs(src_crs_str)
    src_transform = override_src_transform or _parse_transform(src_transform_tuple)
    if src_crs is None:
        raise RuntimeError(f"{src_group_path}: CRS missing/unparseable")

    dst_crs = dst_grid["crs_obj"]
    dst_transform = dst_grid["transform_obj"]
    Hdst, Wdst = dst_grid["height"], dst_grid["width"]

    dstg = ensure_group(zout, dst_group_path)
    copy_group_attrs(
        srcg, dstg,
        extra={
            "canonical_grid": "landsat_30m",
            "source_group": src_group_path,
            "source_crs": src_crs_str,
            "source_transform": src_transform_tuple,
            "resampling": "nearest" if is_categorical else "bilinear",
        }
    )
    create_text_array(dstg, "labels", [str(x) for x in labels])
    create_text_array(dstg, "band_names", [str(x) for x in band_names])

    T, C, _, _ = src_data.shape
    shape = (T, C, Hdst, Wdst)
    chunks = choose_chunks(shape, chunk_hw)

    dst_arr = dstg.create_array(
        "data", shape=shape, chunks=chunks, dtype=data_dtype,
        compressors=[compressor], fill_value=np.nan, overwrite=True
    )
    dst_val = dstg.create_array(
        "valid", shape=(T, 1, Hdst, Wdst),
        chunks=(1, 1, chunks[2], chunks[3]),
        dtype=np.uint8, compressors=[compressor], fill_value=0, overwrite=True
    )

    resamp = Resampling.nearest if is_categorical else Resampling.bilinear

    t0 = time.time()
    for t in range(T):
        x = src_data[t, :, :, :].astype(np.float32, copy=False)
        v = src_valid[t, 0, :, :].astype(np.uint8, copy=False)
        if src_group_path == "static/dem" and np.all(v == 0):
            # DEM may not have a meaningful valid mask; derive from finite data.
            v = np.isfinite(x).any(axis=0).astype(np.uint8)

        # prevent NaN bleed
        x_masked = x.copy()
        x_masked[:, v == 0] = np.nan

        xw = warp_stack_to_dst(
            src=x_masked,
            src_crs=src_crs,
            src_transform=src_transform,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            dst_hw=(Hdst, Wdst),
            resampling=resamp,
            dst_dtype=np.float32,
        )
        vw = warp_mask_to_dst(
            src_mask=v,
            src_crs=src_crs,
            src_transform=src_transform,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            dst_hw=(Hdst, Wdst),
        )

        dst_arr[t, :, :, :] = xw.astype(data_dtype, copy=False)
        dst_val[t, 0, :, :] = vw

        if (t + 1) % 25 == 0 or (t + 1) == T:
            dt = time.time() - t0
            print(f"{dst_group_path}: {t+1}/{T} ({dt:.1f}s)", flush=True)


# ----------------------------
# Optimized AlphaEarth resampler (tile + band-batch)
# ----------------------------
def window_bounds(transform: Affine, window: Window) -> Tuple[float, float, float, float]:
    return rasterio.windows.bounds(window, transform)


def clamp_window(win: Window, H: int, W: int) -> Window:
    # rasterio Window is (col_off,row_off,width,height)
    c0 = max(int(win.col_off), 0)
    r0 = max(int(win.row_off), 0)
    c1 = min(int(win.col_off + win.width), W)
    r1 = min(int(win.row_off + win.height), H)
    if r1 <= r0 or c1 <= c0:
        return Window(0, 0, 0, 0)
    return Window(c0, r0, c1 - c0, r1 - r0)


def resample_alphaearth_to_30m_optimized(
    zin: zarr.Group,
    zout: zarr.Group,
    src_group_path: str,
    dst_group_path: str,
    dst_grid: Dict[str, Any],
    compressor: BloscCodec,
    chunk_hw: int,
    data_dtype: np.dtype,
    tile: int = 512,
    band_batch: int = 8,
):
    srcg = ensure_group(zin, src_group_path)
    if "data" not in srcg or "valid" not in srcg:
        raise RuntimeError(f"{src_group_path}: missing data/valid arrays")

    src_data = srcg["data"]      # (T,C,H,W)
    src_valid = srcg["valid"]    # (T,1,H,W)

    labels = list(srcg["labels"][:]) if "labels" in srcg else [str(i) for i in range(src_data.shape[0])]
    band_names = list(srcg["band_names"][:]) if "band_names" in srcg else [f"band_{i+1:02d}" for i in range(src_data.shape[1])]

    src_crs = _parse_crs(srcg.attrs.get("crs"))
    src_transform = _parse_transform(srcg.attrs.get("transform"))
    if src_crs is None:
        raise RuntimeError(f"{src_group_path}: CRS missing/unparseable")

    Hs = int(srcg.attrs.get("height"))
    Ws = int(srcg.attrs.get("width"))

    dst_crs = dst_grid["crs_obj"]
    dst_transform = dst_grid["transform_obj"]
    Hd = dst_grid["height"]
    Wd = dst_grid["width"]

    # Create destination group + arrays
    dstg = ensure_group(zout, dst_group_path)
    copy_group_attrs(
        srcg, dstg,
        extra={
            "canonical_grid": "landsat_30m",
            "source_group": src_group_path,
            "source_crs": srcg.attrs.get("crs"),
            "source_transform": srcg.attrs.get("transform"),
            "resampling": "bilinear",
            "alphaearth_optimized": True,
            "tile": int(tile),
            "band_batch": int(band_batch),
        }
    )
    create_text_array(dstg, "labels", [str(x) for x in labels])
    create_text_array(dstg, "band_names", [str(x) for x in band_names])

    T, C, _, _ = src_data.shape
    shape = (T, C, Hd, Wd)
    chunks = choose_chunks(shape, chunk_hw)

    dst_arr = dstg.create_array(
        "data", shape=shape, chunks=chunks, dtype=data_dtype,
        compressors=[compressor], fill_value=np.nan, overwrite=True
    )
    dst_val = dstg.create_array(
        "valid", shape=(T, 1, Hd, Wd),
        chunks=(1, 1, chunks[2], chunks[3]),
        dtype=np.uint8, compressors=[compressor], fill_value=0, overwrite=True
    )

    for t in range(T):
        t0 = time.time()
        for y0 in range(0, Hd, tile):
            y1 = min(y0 + tile, Hd)
            for x0 in range(0, Wd, tile):
                x1 = min(x0 + tile, Wd)
                dst_win = Window(x0, y0, x1 - x0, y1 - y0)

                # Bounds in dst CRS -> transform to src CRS -> source window
                left, bottom, right, top = window_bounds(dst_transform, dst_win)
                try:
                    s_left, s_bottom, s_right, s_top = transform_bounds(
                        dst_crs, src_crs, left, bottom, right, top, densify_pts=21
                    )
                except Exception:
                    s_left, s_bottom, s_right, s_top = left, bottom, right, top

                src_win_f = from_bounds(s_left, s_bottom, s_right, s_top, transform=src_transform)
                src_win = clamp_window(src_win_f.round_offsets().round_lengths(), Hs, Ws)
                if src_win.width == 0 or src_win.height == 0:
                    continue

                rs0, cs0 = int(src_win.row_off), int(src_win.col_off)
                rs1 = rs0 + int(src_win.height)
                cs1 = cs0 + int(src_win.width)
                if t == 0 and y0 == 0 and x0 == 0:
                    h_src = int(src_win.height)
                    w_src = int(src_win.width)
                    h_dst = int(dst_win.height)
                    w_dst = int(dst_win.width)
                    bytes_v_src = h_src * w_src * np.dtype(np.uint8).itemsize
                    bytes_v_dst = h_dst * w_dst * np.dtype(np.uint8).itemsize
                    bytes_x_src = h_src * w_src * np.dtype(np.float32).itemsize
                    bytes_x_dst = h_dst * w_dst * np.dtype(np.float32).itemsize
                    est_total = bytes_v_src + bytes_v_dst + bytes_x_src + bytes_x_dst
                    print(
                        "AlphaEarth optimized: estimated per-tile buffers "
                        f"(single-band) ~ {est_total / (1024**2):.2f} MiB "
                        f"[src={h_src}x{w_src}, dst={h_dst}x{w_dst}]",
                        flush=True
                    )

                # Read source valid for this window
                v_src = src_valid[t, 0, rs0:rs1, cs0:cs1].astype(np.uint8, copy=False)

                # Warp valid -> dst tile (nearest)
                v_dst = np.zeros((int(dst_win.height), int(dst_win.width)), dtype=np.uint8)
                reproject(
                    source=v_src,
                    destination=v_dst,
                    src_transform=src_transform * Affine.translation(cs0, rs0),
                    src_crs=src_crs,
                    dst_transform=dst_transform * Affine.translation(x0, y0),
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    src_nodata=0,
                    dst_nodata=0,
                )
                dst_val[t, 0, y0:y1, x0:x1] = v_dst

                # Bands in small batches, written one band at a time to keep memory low
                for b0 in range(0, C, band_batch):
                    b1 = min(b0 + band_batch, C)
                    for bi in range(b0, b1):
                        x_src = src_data[t, bi, rs0:rs1, cs0:cs1].astype(np.float32, copy=False)

                        # Prevent NaN bleed
                        x_src = x_src.copy()
                        x_src[v_src == 0] = np.nan

                        x_dst = np.full((int(dst_win.height), int(dst_win.width)), np.nan, dtype=np.float32)
                        reproject(
                            source=x_src,
                            destination=x_dst,
                            src_transform=src_transform * Affine.translation(cs0, rs0),
                            src_crs=src_crs,
                            dst_transform=dst_transform * Affine.translation(x0, y0),
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear,
                            src_nodata=np.nan,
                            dst_nodata=np.nan,
                        )
                        dst_arr[t, bi, y0:y1, x0:x1] = x_dst.astype(data_dtype, copy=False)

        dt = time.time() - t0
        print(f"{dst_group_path}: t={t+1}/{T} done in {dt:.1f}s", flush=True)


# ----------------------------
# Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser("Create canonical Landsat-30m grid Zarr from multi-grid Zarr (AlphaEarth optimized).")
    p.add_argument("--in-zarr", required=True, help="Input Zarr folder (madurai.zarr)")
    p.add_argument("--out-zarr", required=True, help="Output Zarr folder (madurai_30m.zarr)")
    p.add_argument("--compress-level", type=int, default=9)
    p.add_argument("--chunk-hw", type=int, default=256)
    p.add_argument("--dtype", type=str, default="float32")

    p.add_argument("--include-alphaearth", action="store_true")
    p.add_argument("--alphaearth-optimized", action="store_true", help="Use low-RAM AlphaEarth resampling.")
    p.add_argument("--tile", type=int, default=512, help="AlphaEarth optimized: dst tile size")
    p.add_argument("--band-batch", type=int, default=8, help="AlphaEarth optimized: bands per batch")

    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.in_zarr).resolve()
    out_path = Path(args.out_zarr).resolve()

    zin = zarr.open_group(str(in_path), mode="r")
    zout = zarr.open_group(str(out_path), mode="w")

    compressor = BloscCodec(cname="zstd", clevel=int(args.compress_level), shuffle="bitshuffle")
    data_dtype = np.dtype(args.dtype)

    # Copy global attrs + time axes
    zout.attrs.update(dict(zin.attrs))
    zout.attrs["canonical_grid"] = "landsat_30m"
    copy_time_axes(zin, zout)

    # Canonical grid from Landsat
    grid = get_grid_from_group(zin, "products/landsat")

    # Store canonical grid metadata
    ggrid = ensure_group(zout, "grid")
    ggrid.attrs["name"] = "landsat_30m"
    ggrid.attrs["crs"] = grid["crs_str"]
    ggrid.attrs["transform"] = grid["transform_tuple"]
    ggrid.attrs["height"] = grid["height"]
    ggrid.attrs["width"] = grid["width"]
    ggrid.attrs["pixel_size_x"] = grid["pixel_size_x"]
    ggrid.attrs["pixel_size_y"] = grid["pixel_size_y"]
    ggrid.attrs["landsat_reference_file"] = grid["reference_file"]

    # AlphaEarth (optional) - run first
    if args.include_alphaearth:
        src_gp = "products/alphaearth"
        dst_gp = "products_30m/alphaearth"
        print(f"=== Resampling {src_gp} -> {dst_gp} (AlphaEarth) ===", flush=True)
        if args.alphaearth_optimized:
            resample_alphaearth_to_30m_optimized(
                zin=zin,
                zout=zout,
                src_group_path=src_gp,
                dst_group_path=dst_gp,
                dst_grid=grid,
                compressor=compressor,
                chunk_hw=int(args.chunk_hw),
                data_dtype=data_dtype,
                tile=int(args.tile),
                band_batch=int(args.band_batch),
            )
        else:
            # Warning: this may use a lot of RAM depending on AlphaEarth size
            resample_product_to_30m_simple(
                zin=zin,
                zout=zout,
                src_group_path=src_gp,
                dst_group_path=dst_gp,
                dst_grid=grid,
                compressor=compressor,
                chunk_hw=int(args.chunk_hw),
                data_dtype=data_dtype,
                is_categorical=False,
            )

    # Plan: fine drivers + statics
    plan = [
        ("products/sentinel2", "products_30m/sentinel2", False, "simple"),
        ("products/sentinel1", "products_30m/sentinel1", False, "simple"),
        ("products/era5", "products_30m/era5", False, "simple"),
        ("static/dem", "static_30m/dem", False, "simple"),
        ("static/worldcover", "static_30m/worldcover", True, "simple"),
        ("static/dynamic_world", "static_30m/dynamic_world", True, "simple"),
    ]

    # Execute plan
    era5_override = era5_override_grid()
    for src_gp, dst_gp, is_cat, mode in plan:
        print(f"=== Resampling {src_gp} -> {dst_gp} (categorical={is_cat}) ===", flush=True)
        override_kwargs = {}
        if src_gp == "products/era5":
            override_kwargs = {
                "override_src_crs": era5_override["crs"],
                "override_src_transform": era5_override["transform"],
                "override_src_crs_str": era5_override["crs_str"],
                "override_src_transform_tuple": era5_override["transform_tuple"],
            }
        resample_product_to_30m_simple(
            zin=zin,
            zout=zout,
            src_group_path=src_gp,
            dst_group_path=dst_gp,
            dst_grid=grid,
            compressor=compressor,
            chunk_hw=int(args.chunk_hw),
            data_dtype=data_dtype,
            is_categorical=is_cat,
            **override_kwargs,
        )

    # Landsat copied into output labels
    print("=== Copying Landsat into labels_30m/landsat ===", flush=True)
    resample_product_to_30m_simple(
        zin=zin,
        zout=zout,
        src_group_path="products/landsat",
        dst_group_path="labels_30m/landsat",
        dst_grid=grid,
        compressor=compressor,
        chunk_hw=int(args.chunk_hw),
        data_dtype=data_dtype,
        is_categorical=False,
    )

    # Manifest
    manifest = {
        "input": str(in_path),
        "output": str(out_path),
        "canonical_grid": "products/landsat",
        "included_alphaearth": bool(args.include_alphaearth),
        "alphaearth_optimized": bool(args.alphaearth_optimized),
        "alphaearth_tile": int(args.tile),
        "alphaearth_band_batch": int(args.band_batch),
        "dtype": str(data_dtype),
        "compress": {"codec": "blosc", "cname": "zstd", "level": int(args.compress_level), "shuffle": "bitshuffle"},
        "chunk_hw": int(args.chunk_hw),
        "timestamp_utc": str(np.datetime64("now")),
    }
    zout.attrs["canonical_build_manifest"] = manifest
    meta = ensure_group(zout, "meta")
    create_text_array(meta, "canonical_manifest_json", [json.dumps(manifest, indent=2)])

    print(f"DONE: wrote canonical 30m Zarr to {out_path}", flush=True)


if __name__ == "__main__":
    main()
