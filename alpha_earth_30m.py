#!/usr/bin/env python3
"""
AlphaEarth -> canonical 30 m (Landsat grid) WITHOUT blowing RAM.

Key idea:
- Process destination in tiles (e.g., 512x512)
- For each tile, compute its bounds in dst CRS
- Transform bounds to src CRS
- Read ONLY the required src window from Zarr
- Reproject tile-by-tile, band-batch-by-band-batch

This avoids loading full (C,H,W) into memory.

Run:
  python alphaearth_to_30m_optimized.py \
    --in-zarr  /path/to/madurai.zarr \
    --out-zarr /path/to/madurai_30m_alphaearth.zarr \
    --dst-from products/landsat \
    --src-from products/alphaearth \
    --tile 512 --band-batch 8 \
    --compress-level 9 --chunk-hw 256
"""

import argparse
import time
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr import dtype as zarr_dtype

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds, Window


TEXT_DTYPE = zarr_dtype.VariableLengthUTF8()


def ensure_group(root: zarr.Group, path: str) -> zarr.Group:
    g = root
    for part in path.strip("/").split("/"):
        g = g.require_group(part)
    return g


def create_text_array(group: zarr.Group, name: str, values: List[str]) -> zarr.Array:
    arr = group.create_array(name, shape=(len(values),), dtype=TEXT_DTYPE, overwrite=True)
    if values:
        arr[:] = np.asarray(values, dtype=object)
    return arr


def _parse_crs(crs_str: Optional[str]) -> CRS:
    if not crs_str:
        raise ValueError("Missing CRS string")
    s = str(crs_str)
    try:
        return CRS.from_string(s)
    except Exception:
        return CRS.from_wkt(s)


def _parse_transform(tup) -> Affine:
    return Affine(*tuple(tup))


def choose_chunks(shape: Tuple[int, int, int, int], chunk_hw: int) -> Tuple[int, int, int, int]:
    T, C, H, W = shape
    ch = min(chunk_hw, H)
    cw = min(chunk_hw, W)
    cchunk = C if C <= 32 else 8
    return (1, cchunk, ch, cw)


def get_grid_from_group(zroot: zarr.Group, group_path: str) -> Dict[str, Any]:
    g = ensure_group(zroot, group_path)
    return {
        "crs_str": g.attrs.get("crs"),
        "crs": _parse_crs(g.attrs.get("crs")),
        "transform_tuple": tuple(g.attrs.get("transform")),
        "transform": _parse_transform(g.attrs.get("transform")),
        "height": int(g.attrs.get("height")),
        "width": int(g.attrs.get("width")),
        "reference_file": g.attrs.get("reference_file"),
        "pixel_size_x": g.attrs.get("pixel_size_x"),
        "pixel_size_y": g.attrs.get("pixel_size_y"),
    }


def window_bounds(transform: Affine, window: Window) -> Tuple[float, float, float, float]:
    # Returns bounds in the CRS of the transform
    return rasterio.windows.bounds(window, transform)


def clamp_window(win: Window, H: int, W: int) -> Window:
    r0 = max(int(win.row_off), 0)
    c0 = max(int(win.col_off), 0)
    r1 = min(int(win.row_off + win.height), H)
    c1 = min(int(win.col_off + win.width), W)
    if r1 <= r0 or c1 <= c0:
        return Window(0, 0, 0, 0)
    return Window(c0, r0, c1 - c0, r1 - r0)  # Window(col_off,row_off,width,height)


def alphaearth_to_30m_blockwise(
    zin: zarr.Group,
    zout: zarr.Group,
    src_path: str,
    dst_path: str,
    dst_grid_from: str,
    compressor: BloscCodec,
    out_dtype: np.dtype,
    chunk_hw: int,
    tile: int,
    band_batch: int,
):
    srcg = ensure_group(zin, src_path)
    if "data" not in srcg or "valid" not in srcg:
        raise RuntimeError(f"{src_path}: missing data/valid arrays")

    # Source arrays
    src_data = srcg["data"]      # (T,C,H,W)
    src_valid = srcg["valid"]    # (T,1,H,W)

    # Source grid
    src_crs = _parse_crs(srcg.attrs.get("crs"))
    src_transform = _parse_transform(srcg.attrs.get("transform"))
    Hs = int(srcg.attrs.get("height"))
    Ws = int(srcg.attrs.get("width"))

    # Destination grid (canonical 30m) from Landsat group
    dst_grid = get_grid_from_group(zin, dst_grid_from)
    dst_crs = dst_grid["crs"]
    dst_transform = dst_grid["transform"]
    Hd = dst_grid["height"]
    Wd = dst_grid["width"]

    # Labels / band names
    labels = list(srcg["labels"][:]) if "labels" in srcg else [str(i) for i in range(src_data.shape[0])]
    band_names = list(srcg["band_names"][:]) if "band_names" in srcg else [f"band_{i+1:02d}" for i in range(src_data.shape[1])]

    # Create dst group
    dg = ensure_group(zout, dst_path)
    dg.attrs.update({k: v for k, v in dict(srcg.attrs).items() if isinstance(k, str)})
    dg.attrs["canonical_grid"] = "landsat_30m"
    dg.attrs["source_group"] = src_path
    dg.attrs["source_crs"] = srcg.attrs.get("crs")
    dg.attrs["source_transform"] = srcg.attrs.get("transform")
    dg.attrs["resampling"] = "bilinear"
    dg.attrs["dst_crs"] = dst_grid["crs_str"]
    dg.attrs["dst_transform"] = dst_grid["transform_tuple"]
    dg.attrs["dst_height"] = Hd
    dg.attrs["dst_width"] = Wd

    create_text_array(dg, "labels", [str(x) for x in labels])
    create_text_array(dg, "band_names", [str(x) for x in band_names])

    T, C, _, _ = src_data.shape
    shape = (T, C, Hd, Wd)
    chunks = choose_chunks(shape, chunk_hw)

    dst_arr = dg.create_array(
        "data", shape=shape, chunks=chunks, dtype=out_dtype,
        compressors=[compressor], fill_value=np.nan, overwrite=True
    )
    dst_val = dg.create_array(
        "valid", shape=(T, 1, Hd, Wd),
        chunks=(1, 1, chunks[2], chunks[3]),
        dtype=np.uint8, compressors=[compressor], fill_value=0, overwrite=True
    )

    # Process each timestep (annual usually small T, but still blockwise)
    for t in range(T):
        t0 = time.time()

        # Process dst in tiles
        for y0 in range(0, Hd, tile):
            y1 = min(y0 + tile, Hd)
            for x0 in range(0, Wd, tile):
                x1 = min(x0 + tile, Wd)

                dst_win = Window(x0, y0, x1 - x0, y1 - y0)

                # Bounds in dst CRS
                left, bottom, right, top = window_bounds(dst_transform, dst_win)

                # Transform those bounds to src CRS to find a minimal src window to read
                try:
                    s_left, s_bottom, s_right, s_top = transform_bounds(
                        dst_crs, src_crs, left, bottom, right, top, densify_pts=21
                    )
                except Exception:
                    # If CRS same, fallback
                    s_left, s_bottom, s_right, s_top = left, bottom, right, top

                # Source window from transformed bounds
                src_win_f = from_bounds(s_left, s_bottom, s_right, s_top, transform=src_transform)
                src_win = clamp_window(src_win_f.round_offsets().round_lengths(), Hs, Ws)
                if src_win.width == 0 or src_win.height == 0:
                    continue

                rs0, cs0 = int(src_win.row_off), int(src_win.col_off)
                rs1 = rs0 + int(src_win.height)
                cs1 = cs0 + int(src_win.width)

                # Read src valid window (tiny) and expand to mark finite regions
                v_src = src_valid[t, 0, rs0:rs1, cs0:cs1].astype(np.uint8, copy=False)

                # Warp valid mask into dst tile
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

                # Band-batch to reduce RAM
                for b0 in range(0, C, band_batch):
                    b1 = min(b0 + band_batch, C)

                    # Read only the needed src window for this band batch
                    x_src = src_data[t, b0:b1, rs0:rs1, cs0:cs1].astype(np.float32, copy=False)

                    # Apply src valid mask before reprojection (prevents NaN bleed)
                    if v_src is not None:
                        x_src = x_src.copy()
                        x_src[:, v_src == 0] = np.nan

                    # Destination tile buffer for this band batch
                    x_dst = np.full((b1 - b0, int(dst_win.height), int(dst_win.width)), np.nan, dtype=np.float32)

                    # Reproject band-batch
                    for bi in range(b1 - b0):
                        reproject(
                            source=x_src[bi],
                            destination=x_dst[bi],
                            src_transform=src_transform * Affine.translation(cs0, rs0),
                            src_crs=src_crs,
                            dst_transform=dst_transform * Affine.translation(x0, y0),
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear,
                            src_nodata=np.nan,
                            dst_nodata=np.nan,
                        )

                    dst_arr[t, b0:b1, y0:y1, x0:x1] = x_dst.astype(out_dtype, copy=False)

        dt = time.time() - t0
        print(f"{dst_path}: t={t+1}/{T} done in {dt:.1f}s", flush=True)


def parse_args():
    p = argparse.ArgumentParser("Optimized AlphaEarth -> canonical 30m Zarr (low RAM).")
    p.add_argument("--in-zarr", default="madurai.zarr")
    p.add_argument("--out-zarr", default = "madurai_alphaearth_30m.zarr")
    p.add_argument("--src-from", default="products/alphaearth")
    p.add_argument("--dst-from", default="products/landsat", help="Canonical grid source group (30m)")
    p.add_argument("--dst-path", default="products_30m/alphaearth")
    p.add_argument("--tile", type=int, default=512, help="Destination tile size (pixels)")
    p.add_argument("--band-batch", type=int, default=8, help="Bands per batch to reproject")
    p.add_argument("--compress-level", type=int, default=9)
    p.add_argument("--chunk-hw", type=int, default=256)
    p.add_argument("--dtype", default="float32")
    return p.parse_args()


def main():
    args = parse_args()
    zin = zarr.open_group(str(Path(args.in_zarr).resolve()), mode="r")
    zout = zarr.open_group(str(Path(args.out_zarr).resolve()), mode="w")

    # copy global attrs minimally
    zout.attrs.update(dict(zin.attrs))
    zout.attrs["canonical_grid"] = "landsat_30m"
    zout.attrs["note"] = "alphaearth-only canonical build (blockwise low-RAM)"

    compressor = BloscCodec(cname="zstd", clevel=int(args.compress_level), shuffle="bitshuffle")
    out_dtype = np.dtype(args.dtype)

    alphaearth_to_30m_blockwise(
        zin=zin,
        zout=zout,
        src_path=args.src_from,
        dst_path=args.dst_path,
        dst_grid_from=args.dst_from,
        compressor=compressor,
        out_dtype=out_dtype,
        chunk_hw=int(args.chunk_hw),
        tile=int(args.tile),
        band_batch=int(args.band_batch),
    )

    # manifest
    meta = ensure_group(zout, "meta")
    manifest = {
        "input": str(Path(args.in_zarr).resolve()),
        "output": str(Path(args.out_zarr).resolve()),
        "src": args.src_from,
        "dst": args.dst_path,
        "dst_grid_from": args.dst_from,
        "tile": int(args.tile),
        "band_batch": int(args.band_batch),
        "dtype": str(out_dtype),
    }
    create_text_array(meta, "alphaearth_manifest_json", [json.dumps(manifest, indent=2)])
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
