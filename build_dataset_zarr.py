#!/usr/bin/env python3
import os
import re
import json
import time
import argparse
import traceback
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
import zarr
from zarr import dtype as zarr_dtype
from zarr.codecs import BloscCodec
try:
    from zarr.storage import DirectoryStore
except Exception:
    from zarr.storage import LocalStore as DirectoryStore


# ----------------------------
# ROI stored only as metadata
# ----------------------------
ROI_COORDS = [[
    [78.19018990852709, 9.878220339041174],
    [78.18528420047912, 9.882890316485547],
    [78.09810549679275, 9.894348798281609],
    [78.06157864432538, 9.932892703854442],
    [78.02656656110504, 9.94245002321148],
    [78.00260828167308, 9.965541799781803],
    [77.98788548795139, 9.97202781239784],
    [77.98759869039448, 9.972273355756196],
    [77.98926344201777, 9.974277652950137],
    [78.17034512986598, 10.09203200706642],
    [78.33409611393074, 10.299897884952559],
    [78.33446180425422, 10.301538866748283],
    [78.3424560260113, 10.300913203358672],
    [78.3620537189306, 10.301140183092196],
    [78.37500586343197, 10.310994022414183],
    [78.38801590170993, 10.315994815585963],
    [78.39017505117465, 10.315419769552259],
    [78.38542553295602, 10.314175493832316],
    [78.38810333700448, 10.269213359046619],
    [78.41273304358974, 10.212526305038399],
    [78.39622781746723, 10.177726046483423],
    [78.45775072031367, 10.100259541710393],
    [78.45817143938241, 9.967811808760526],
    [78.19018990852709, 9.878220339041174],
]]


@dataclass
class ProductSpec:
    name: str
    cadence: str  # "daily" | "monthly" | "annual" | "static"
    group: str    # zarr group path
    file_glob: Optional[str] = None
    is_alphaearth: bool = False


# ----------------------------
# Your folder layout
# ----------------------------
SPECS = [
        ProductSpec(name="era5",      cadence="daily",   group="products/era5",      file_glob="era5_daily/*.tif"),

    ProductSpec(name="viirs",     cadence="daily",   group="products/viirs",     file_glob="viirs/viirs_raw_*_4band.tif"),

    ProductSpec(name="sentinel2", cadence="monthly", group="products/sentinel2", file_glob="sentinel2/*.tif"),
    ProductSpec(name="sentinel1", cadence="monthly", group="products/sentinel1", file_glob="sentinel1/*.tif"),
    ProductSpec(name="landsat",   cadence="monthly", group="products/landsat",   file_glob="landsat/*.tif"),

    ProductSpec(name="modis",     cadence="daily",   group="products/modis",     file_glob="modis/*.tif"),

    # viirs_dn/<files>

    # alphaearth/<year_folder>/... (chunks + final)
    ProductSpec(name="alphaearth", cadence="annual", group="products/alphaearth", is_alphaearth=True),

    ProductSpec(name="dem",           cadence="static", group="static/dem",           file_glob="static_data/dem/*.tif"),
    ProductSpec(name="dynamic_world", cadence="static", group="static/dynamic_world", file_glob="static_data/dynamic_world/*.tif"),
    ProductSpec(name="worldcover",    cadence="static", group="static/worldcover",    file_glob="static_data/worldcover/*.tif"),
]


# ----------------------------
# Logging utilities
# ----------------------------
class Logger:
    def __init__(self, logfile: Path):
        self.logfile = logfile
        logfile.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(logfile, "a", encoding="utf-8")

    def _write(self, level: str, msg: str):
        ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts} UTC] [{level}] {msg}"
        print(line, flush=True)
        self.fp.write(line + "\n")
        self.fp.flush()

    def info(self, msg: str): self._write("INFO", msg)
    def warn(self, msg: str): self._write("WARN", msg)
    def error(self, msg: str): self._write("ERROR", msg)

    def close(self):
        try:
            self.fp.close()
        except Exception:
            pass


def progress_line(i: int, n: int, prefix: str = "", width: int = 24) -> str:
    if n <= 0:
        return f"{prefix} 0/0"
    frac = i / n
    filled = int(frac * width)
    bar = "#" * filled + "." * (width - filled)
    return f"{prefix} [{bar}] {i}/{n} ({frac*100:5.1f}%)"


# ----------------------------
# Zarr helpers
# ----------------------------
def ensure_group(root: zarr.Group, path: str) -> zarr.Group:
    g = root
    for part in path.strip("/").split("/"):
        g = g.require_group(part)
    return g


TEXT_DTYPE = zarr_dtype.VariableLengthUTF8()


def create_text_array(group: zarr.Group, name: str, values: List[str]) -> zarr.Array:
    arr = group.create_array(name, shape=(len(values),), dtype=TEXT_DTYPE, overwrite=True)
    if values:
        arr[:] = np.asarray(values, dtype=object)
    return arr


def safe_open_raster(path: Path):
    try:
        ds = rasterio.open(path)
        return ds, None
    except Exception as e:
        return None, repr(e)


def raster_signature(ds: rasterio.DatasetReader):
    return {
        "width": ds.width,
        "height": ds.height,
        "count": ds.count,
        "crs": ds.crs.to_string() if ds.crs else None,
        "transform": tuple(ds.transform) if ds.transform else None,
        "nodata": ds.nodata
    }


def infer_band_names(ds: rasterio.DatasetReader) -> List[str]:
    out = []
    for i in range(1, ds.count + 1):
        d = ds.descriptions[i - 1]
        out.append(d if d else f"band_{i:02d}")
    return out


def choose_chunks(shape: Tuple[int, int, int, int], chunk_hw: int) -> Tuple[int, int, int, int]:
    T, C, H, W = shape
    ch = min(chunk_hw, H)
    cw = min(chunk_hw, W)
    cchunk = C if C <= 32 else 8
    return (1, cchunk, ch, cw)


def warp_to_reference(src_ds: rasterio.DatasetReader, ref_meta: dict, out_dtype=np.float32) -> np.ndarray:
    H = ref_meta["height"]
    W = ref_meta["width"]
    dst = np.full((src_ds.count, H, W), np.nan, dtype=out_dtype)

    for b in range(1, src_ds.count + 1):
        src = src_ds.read(b).astype(out_dtype, copy=False)
        dst_band = dst[b - 1]
        reproject(
            source=src,
            destination=dst_band,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            resampling=Resampling.bilinear,
            dst_nodata=np.nan
        )
    return dst


def iter_blocks(height: int, width: int, block_hw: int):
    for y0 in range(0, height, block_hw):
        y1 = min(y0 + block_hw, height)
        for x0 in range(0, width, block_hw):
            x1 = min(x0 + block_hw, width)
            yield y0, y1, x0, x1


# ----------------------------
# Date label extraction
# ----------------------------
RE_DAILY = re.compile(r"(\d{4})[_-](\d{2})[_-](\d{2})")
RE_MONTHLY = re.compile(r"(\d{4})[_-](\d{2})(?![_-]\d{2})")
RE_YEAR = re.compile(r"\b(20\d{2}|19\d{2})\b")


def extract_label_from_path(p: Path, cadence: str) -> Optional[str]:
    parent = p.parent.name

    if cadence == "daily":
        m = RE_DAILY.search(p.name) or RE_DAILY.search(parent)
        if m:
            return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
        return None

    if cadence == "monthly":
        m = RE_MONTHLY.search(p.name) or RE_MONTHLY.search(parent)
        if m:
            return f"{m.group(1)}_{m.group(2)}"
        return None

    if cadence == "annual":
        m = RE_YEAR.search(p.name) or RE_YEAR.search(parent)
        if m:
            return m.group(1)
        return None
    
    if cadence == "static":
        return p.stem

    return None


def date_range_daily(start: date, end: date) -> List[str]:
    idx = pd.date_range(start=start, end=end, freq="D")
    return [d.strftime("%Y_%m_%d") for d in idx]


def date_range_monthly(start: date, end: date) -> List[str]:
    idx = pd.date_range(
        start=pd.Timestamp(start).replace(day=1),
        end=pd.Timestamp(end).replace(day=1),
        freq="MS"
    )
    return [d.strftime("%Y_%m") for d in idx]


def date_range_annual(start: date, end: date) -> List[str]:
    return [f"{y:04d}" for y in range(start.year, end.year + 1)]


# ----------------------------
# Index builders
# ----------------------------
def build_index_glob(root: Path, glob_pat: str, cadence: str, log: Logger) -> Dict[str, Path]:
    files = sorted(root.glob(glob_pat))
    idx: Dict[str, Path] = {}
    skipped = 0

    for f in files:
        lab = extract_label_from_path(f, cadence)
        if lab is None:
            skipped += 1
            continue
        if lab in idx:
            # keep larger (typically merged)
            if f.stat().st_size > idx[lab].stat().st_size:
                idx[lab] = f
        else:
            idx[lab] = f

    log.info(f"Index scan: glob='{glob_pat}' -> files={len(files)} indexed={len(idx)} skipped_no_label={skipped}")
    return idx


def pick_alphaearth_final_tif(year_dir: Path) -> Optional[Path]:
    tifs = list(year_dir.rglob("*.tif"))
    if not tifs:
        return None

    tifs_no_tiles = [p for p in tifs if "tiles" not in p.parts and "tile" not in p.name.lower()]
    tifs_64b = [p for p in tifs_no_tiles if "64b" in p.name.lower()]
    if tifs_64b:
        return max(tifs_64b, key=lambda p: p.stat().st_size)

    tifs_finalish = [p for p in tifs_no_tiles if "chunk" not in p.name.lower() and "c00" not in p.name.lower()]
    if tifs_finalish:
        return max(tifs_finalish, key=lambda p: p.stat().st_size)

    return max(tifs_no_tiles if tifs_no_tiles else tifs, key=lambda p: p.stat().st_size)


def build_index_alphaearth(root: Path, log: Logger) -> Dict[str, Path]:
    base = root / "alphaearth"
    idx: Dict[str, Path] = {}
    if not base.exists():
        log.warn("alphaearth/ not found; alphaearth index will be empty.")
        return idx

    for year_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        y = extract_label_from_path(year_dir, "annual")
        if y is None:
            continue
        tif = pick_alphaearth_final_tif(year_dir)
        if tif is None:
            log.warn(f"AlphaEarth year folder {year_dir.name}: no tif found")
            continue
        idx[y] = tif

    log.info(f"AlphaEarth index: years_indexed={len(idx)}")
    return idx


def first_valid_raster(paths: List[Path], log: Logger, spec_name: str):
    for p in paths:
        ds, err = safe_open_raster(p)
        if ds is not None:
            return ds, p
        else:
            log.warn(f"{spec_name}: reference candidate failed open: {p} err={err}")
    return None, None


# ----------------------------
# Product writer with detailed logging/audit
# ----------------------------
def write_product(zroot: zarr.Group, spec: ProductSpec, labels: List[str], file_index: Dict[str, Path],
                  compressor, data_dtype: np.dtype, chunk_hw: int, warp_to_first: bool, log: Logger) -> Tuple[Dict, Dict]:

    g = ensure_group(zroot, spec.group)
    g.attrs["name"] = spec.name
    g.attrs["cadence"] = spec.cadence
    g.attrs["source_count_found"] = int(len(file_index))
    create_text_array(g, "labels", labels)

    audit = {
        "missing": [],
        "corrupt": [],
        "grid_mismatch": [],
        "warped": [],
        "warp_failed": [],
        "label_parse_warnings": [],
        "open_warnings": [],
        "exceptions": []
    }

    existing_paths = [file_index[l] for l in labels if l in file_index]
    ref_ds, ref_path = first_valid_raster(existing_paths, log, spec.name)

    if ref_ds is None:
        g.attrs["status"] = "all_missing_or_corrupt"
        audit["missing"] = [l for l in labels if l not in file_index]
        audit["corrupt"] = [l for l in labels if l in file_index]
        log.error(f"{spec.name}: no valid reference raster; all missing/corrupt?")
        return (
            {"status": "all_missing_or_corrupt", "expected": len(labels), "found": len(file_index)},
            audit
        )

    ref_meta = raster_signature(ref_ds)
    ref_band_names = infer_band_names(ref_ds)

    g.attrs["status"] = "ok"
    g.attrs["reference_file"] = str(ref_path)
    g.attrs["crs"] = ref_meta["crs"]
    g.attrs["transform"] = ref_meta["transform"]
    g.attrs["width"] = ref_meta["width"]
    g.attrs["height"] = ref_meta["height"]
    g.attrs["band_count"] = ref_meta["count"]
    g.attrs["nodata"] = ref_meta["nodata"]
    try:
        g.attrs["pixel_size_x"] = float(abs(ref_ds.transform.a))
        g.attrs["pixel_size_y"] = float(abs(ref_ds.transform.e))
    except Exception:
        pass
    create_text_array(g, "band_names", ref_band_names)

    T = len(labels)
    C = ref_meta["count"]
    H = ref_meta["height"]
    W = ref_meta["width"]
    shape = (T, C, H, W)
    chunks = choose_chunks(shape, chunk_hw)

    log.info(f"{spec.name}: creating zarr arrays data shape={shape} chunks={chunks} dtype={data_dtype}")
    data = g.create_array(
        "data", shape=shape, chunks=chunks, dtype=data_dtype,
        compressors=[compressor], fill_value=np.nan, overwrite=True
    )
    valid = g.create_array(
        "valid", shape=(T, 1, H, W),
        chunks=(1, 1, chunks[2], chunks[3]),
        dtype=np.uint8, compressors=[compressor], fill_value=0, overwrite=True
    )

    t0 = time.time()
    last_print = time.time()

    for ti, lab in enumerate(labels, start=1):
        p = file_index.get(lab, None)
        if p is None:
            audit["missing"].append(lab)
            continue

        ds, err = safe_open_raster(p)
        if ds is None:
            audit["corrupt"].append({"label": lab, "path": str(p), "err": err})
            continue

        try:
            arr = None
            sig = raster_signature(ds)
            same_grid = (
                sig["width"] == ref_meta["width"] and
                sig["height"] == ref_meta["height"] and
                sig["crs"] == ref_meta["crs"] and
                sig["transform"] == ref_meta["transform"] and
                sig["count"] == ref_meta["count"]
            )

            if not same_grid:
                audit["grid_mismatch"].append({"label": lab, "path": str(p), "sig": sig})
                if warp_to_first and sig["count"] == ref_meta["count"] and ds.crs is not None:
                    try:
                        arr = warp_to_reference(ds, ref_meta, out_dtype=np.float32).astype(data_dtype, copy=False)
                        audit["warped"].append(lab)
                    except Exception as we:
                        audit["warp_failed"].append({"label": lab, "path": str(p), "err": repr(we)})
                        audit["corrupt"].append({"label": lab, "path": str(p), "err": "warp_failed"})
                        continue
                else:
                    # cannot align -> treat as corrupt for the package
                    audit["corrupt"].append({"label": lab, "path": str(p), "err": "grid_mismatch_no_warp"})
                    continue
            if same_grid and spec.is_alphaearth:
                block_hw = min(chunk_hw, H, W)
                for y0, y1, x0, x1 in iter_blocks(H, W, block_hw):
                    window = Window(x0, y0, x1 - x0, y1 - y0)
                    block = ds.read(out_dtype=data_dtype, window=window)
                    if np.issubdtype(block.dtype, np.floating):
                        v = np.isfinite(block).any(axis=0).astype(np.uint8)
                    else:
                        v = np.ones((y1 - y0, x1 - x0), dtype=np.uint8)
                    data[ti - 1, :, y0:y1, x0:x1] = block
                    valid[ti - 1, 0, y0:y1, x0:x1] = v
            else:
                if arr is None:
                    arr = ds.read(out_dtype=data_dtype)  # (C,H,W)

                if np.issubdtype(arr.dtype, np.floating):
                    v = np.isfinite(arr).any(axis=0).astype(np.uint8)
                else:
                    v = np.ones((H, W), dtype=np.uint8)

                data[ti - 1, :, :, :] = arr
                valid[ti - 1, 0, :, :] = v

        except Exception as e:
            audit["exceptions"].append({
                "label": lab,
                "path": str(p),
                "err": repr(e),
                "trace": traceback.format_exc(limit=3)
            })
            audit["corrupt"].append({"label": lab, "path": str(p), "err": repr(e)})
        finally:
            try:
                ds.close()
            except Exception:
                pass

        # periodic progress output
        now = time.time()
        if now - last_print > 2.0:
            log.info(progress_line(ti, T, prefix=f"{spec.name}"))
            last_print = now

    # attrs + final product summary
    g.attrs["missing_labels"] = audit["missing"]
    g.attrs["corrupt_labels"] = [x["label"] if isinstance(x, dict) else x for x in audit["corrupt"]]
    g.attrs["grid_mismatch_labels"] = [x["label"] for x in audit["grid_mismatch"]]
    g.attrs["warped_labels"] = audit["warped"]
    g.attrs["warp_failed_labels"] = [x["label"] for x in audit["warp_failed"]]

    info = {
        "status": "ok",
        "cadence": spec.cadence,
        "expected": T,
        "found": len(file_index),
        "missing": len(audit["missing"]),
        "corrupt": len(audit["corrupt"]),
        "grid_mismatch": len(audit["grid_mismatch"]),
        "warped": len(audit["warped"]),
        "warp_failed": len(audit["warp_failed"]),
        "shape": shape,
        "chunks": chunks,
        "reference_file": str(ref_path),
        "seconds": round(time.time() - t0, 2)
    }

    log.info(f"{spec.name}: DONE in {info['seconds']}s "
             f"(missing={info['missing']} corrupt={info['corrupt']} "
             f"grid_mismatch={info['grid_mismatch']} warped={info['warped']} warp_failed={info['warp_failed']})")
    return info, audit


# ----------------------------
# Index building utilities
# ----------------------------
def build_index_for_spec(root: Path, spec: ProductSpec, log: Logger) -> Dict[str, Path]:
    if spec.is_alphaearth:
        return build_index_alphaearth(root, log)
    if spec.file_glob is None:
        return {}
    return build_index_glob(root, spec.file_glob, spec.cadence, log)


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser("Build a complete multi-grid Zarr package with verbose logs.")
    p.add_argument("--root", type=str, default="../data", help="Root folder containing alphaearth/, era5_daily/, modis/, ...")
    p.add_argument("--out", type=str, default = "madurai.zarr", help="Output Zarr folder path, e.g., ./madurai_complete.zarr")
    p.add_argument("--start", type=str, default="2015-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--warp-to-first", action="store_true",
                   help="Warp mismatched grids within a product to the first valid grid.")
    p.add_argument("--compress-level", type=int, default=5, help="Zstd compression level (1-9)")
    p.add_argument("--chunk-hw", type=int, default=256, help="Chunk size for H/W")
    p.add_argument("--dtype", type=str, default="float32", help="Zarr data dtype (float32 recommended)")
    p.add_argument("--log-file", type=str, default="zarr.log", help="Path to log file (default: <out>/_build_logs/build.log)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve()

    # log file inside zarr folder (even before zarr exists)
    log_path = Path(args.log_file).resolve() if args.log_file else out_path / "_build_logs" / "build.log"
    log = Logger(log_path)
    log.info("===== START ZARR BUILD =====")
    log.info(f"root={root}")
    log.info(f"out={out_path}")
    log.info(f"date_start={args.start} date_end={args.end or 'today'}")
    log.info(f"warp_to_first={bool(args.warp_to_first)}")
    log.info(f"compress=zstd(level={args.compress_level}) bitshuffle")
    log.info(f"dtype={args.dtype} chunk_hw={args.chunk_hw}")

    start_d = date.fromisoformat(args.start)
    end_d = date.today() if args.end is None else date.fromisoformat(args.end)

    compressor = BloscCodec(cname="zstd", clevel=int(args.compress_level), shuffle="bitshuffle")
    data_dtype = np.dtype(args.dtype)

    # Create Zarr
    store = DirectoryStore(str(out_path))
    zroot = zarr.group(store=store, overwrite=True)

    # Global metadata
    zroot.attrs["package_name"] = "Madurai complete multi-sensor dataset"
    zroot.attrs["created_utc"] = pd.Timestamp.utcnow().isoformat()
    zroot.attrs["roi_coords_lonlat"] = ROI_COORDS
    zroot.attrs["date_start"] = str(start_d)
    zroot.attrs["date_end"] = str(end_d)
    zroot.attrs["notes"] = (
        "Multi-grid Zarr. Each product keeps native grid. Missing/corrupt -> NaN + valid=0. "
        "Grid mismatches optionally warped to first valid."
    )

    # Time axes
    time_daily = date_range_daily(start_d, end_d)
    time_monthly = date_range_monthly(start_d, end_d)
    time_annual = date_range_annual(start_d, end_d)

    create_text_array(zroot, "time/daily", [str(t) for t in time_daily])
    create_text_array(zroot, "time/monthly", [str(t) for t in time_monthly])
    create_text_array(zroot, "time/annual", [str(t) for t in time_annual])

    summary = {}
    audits = {}

    # Write each product
    for spec in SPECS:
        print("new dataset builder")
        log.info(f"--- PRODUCT: {spec.name} ({spec.cadence}) ---")

        # expected labels
        if spec.cadence == "daily":
            labels = time_daily
        elif spec.cadence == "monthly":
            labels = time_monthly
        elif spec.cadence == "annual":
            labels = time_annual
        elif spec.cadence == "static":
            labels = []  # will be from index keys
        else:
            raise ValueError(f"Unknown cadence: {spec.cadence}")

        # build file index
        file_index = build_index_for_spec(root, spec, log)

        # static labels come from index keys
        if spec.cadence == "static":
            labels = sorted(list(file_index.keys()))
            log.info(f"{spec.name}: static files={len(labels)}")

        # write product
        try:
            info, audit = write_product(
                zroot=zroot,
                spec=spec,
                labels=labels,
                file_index=file_index,
                compressor=compressor,
                data_dtype=data_dtype,
                chunk_hw=int(args.chunk_hw),
                warp_to_first=bool(args.warp_to_first),
                log=log
            )
            summary[spec.name] = info
            audits[spec.name] = audit
        except Exception as e:
            log.error(f"{spec.name}: FATAL error: {repr(e)}")
            log.error(traceback.format_exc(limit=5))
            summary[spec.name] = {"status": "fatal_error", "err": repr(e)}
            audits[spec.name] = {"fatal": repr(e), "trace": traceback.format_exc(limit=10)}

    # Store summaries inside Zarr
    zroot.attrs["build_summary"] = summary
    meta = ensure_group(zroot, "meta")
    create_text_array(meta, "summary_json", [json.dumps(summary, indent=2)])
    create_text_array(meta, "audit_json", [json.dumps(audits, indent=2)])

    # Final console/log summary
    log.info("===== BUILD COMPLETE =====")
    for k, v in summary.items():
        if v.get("status") == "ok":
            log.info(f"{k}: ok  expected={v['expected']} missing={v['missing']} corrupt={v['corrupt']} "
                     f"grid_mismatch={v['grid_mismatch']} warped={v['warped']} secs={v['seconds']}")
        else:
            log.warn(f"{k}: status={v.get('status')} info={v}")

    log.info(f"Zarr written to: {out_path}")
    log.info(f"Build log: {out_path / '_build_logs' / 'build.log'}")
    log.close()


if __name__ == "__main__":
    main()
