#!/usr/bin/env python3
import argparse
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rasterio


RE_DAILY = re.compile(r"(20\d{2})[_-](\d{2})[_-](\d{2})")


def extract_date(text: str) -> Optional[str]:
    m = RE_DAILY.search(text)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"


def classify_day_night(name: str) -> Optional[str]:
    low = name.lower()
    if "night" in low:
        return "night"
    if "day" in low:
        return "day"
    return None


def collect_tif_paths(input_dir: Path, keep_extracted: bool) -> Tuple[List[Path], List[tempfile.TemporaryDirectory]]:
    tif_paths: List[Path] = []
    temp_dirs: List[tempfile.TemporaryDirectory] = []

    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue

        if zipfile.is_zipfile(p):
            if keep_extracted:
                with zipfile.ZipFile(p) as zf:
                    zf.extractall(input_dir)
                    for name in zf.namelist():
                        if name.lower().endswith(".tif"):
                            tif_paths.append((input_dir / name).resolve())
            else:
                td = tempfile.TemporaryDirectory()
                temp_dirs.append(td)
                with zipfile.ZipFile(p) as zf:
                    zf.extractall(td.name)
                for root, _, files in os.walk(td.name):
                    for f in files:
                        if f.lower().endswith(".tif"):
                            tif_paths.append(Path(root) / f)
        elif p.suffix.lower() == ".tif":
            tif_paths.append(p)

    return tif_paths, temp_dirs


def build_pairs(tif_paths: List[Path]) -> Dict[str, Dict[str, Path]]:
    pairs: Dict[str, Dict[str, Path]] = {}

    for p in tif_paths:
        date = extract_date(p.name)
        if not date:
            continue
        kind = classify_day_night(p.name)
        if not kind:
            continue
        pairs.setdefault(date, {})[kind] = p

    return pairs


def _band_descriptions(ds: rasterio.DatasetReader, prefix: str) -> List[str]:
    descs: List[str] = []
    for i in range(1, ds.count + 1):
        d = ds.descriptions[i - 1]
        if d:
            descs.append(d)
        else:
            descs.append(f"{prefix}_band_{i:02d}")
    return descs


def write_multiband(day_path: Path, night_path: Path, out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        print(f"SKIP {out_path} (exists)")
        return

    with rasterio.open(day_path) as ds_day, rasterio.open(night_path) as ds_night:
        if (ds_day.width != ds_night.width or ds_day.height != ds_night.height or
                ds_day.transform != ds_night.transform or ds_day.crs != ds_night.crs):
            raise ValueError(f"Grid mismatch: {day_path} vs {night_path}")

        day_count = ds_day.count
        night_count = ds_night.count
        profile = ds_day.profile.copy()
        profile.update(count=day_count + night_count)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(ds_day.read(), list(range(1, day_count + 1)))
            dst.write(ds_night.read(), list(range(day_count + 1, day_count + night_count + 1)))
            descriptions = _band_descriptions(ds_day, "day") + _band_descriptions(ds_night, "night")
            for i, d in enumerate(descriptions, start=1):
                dst.set_band_description(i, d)

    print(f"WROTE {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unzip VIIRS archives and merge day/night into a single multi-band GeoTIFF per date.")
    ap.add_argument("--input-dir", required=True, type=Path, help="Folder containing VIIRS zip/tif files (e.g. data/viirs_dn)")
    ap.add_argument("--output-dir", type=Path, default=None, help="Output folder for merged tifs (default: input-dir)")
    ap.add_argument("--keep-extracted", action="store_true", help="Extract zip contents into input-dir instead of a temp folder")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing merged files")
    args = ap.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or input_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_paths, temp_dirs = collect_tif_paths(input_dir, keep_extracted=args.keep_extracted)
    pairs = build_pairs(tif_paths)

    if not pairs:
        print("No day/night pairs found.")
        return

    for date, v in sorted(pairs.items()):
        day_path = v.get("day")
        night_path = v.get("night")
        if not day_path or not night_path:
            print(f"SKIP {date}: missing day or night")
            continue
        out_path = output_dir / f"viirs_dn_{date}.tif"
        write_multiband(day_path, night_path, out_path, overwrite=args.overwrite)

    for td in temp_dirs:
        td.cleanup()


if __name__ == "__main__":
    main()
