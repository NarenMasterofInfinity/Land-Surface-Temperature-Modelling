#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import rasterio

from build_dataset_zarr import SPECS, _to_jsonable, raster_verbose_metadata

INCLUDED_FOLDERS = [
    "sentinel1",
    "sentinel2",
    "landsat",
    "era5",
    "modis",
    "viirs",
    "alphaearth",
    "dem",
    "dynamic_world",
    "worldcover",
]


def _folder_from_spec(spec) -> str:
    if spec.file_glob:
        return spec.file_glob.split("/")[0]
    return spec.name


def _find_example_file(data_root: Path, spec) -> Optional[Path]:
    if spec.file_glob:
        matches = sorted(data_root.glob(spec.file_glob))
        return matches[0] if matches else None
    folder = data_root / spec.name
    matches = sorted(folder.rglob("*.tif"))
    return matches[0] if matches else None


def _read_metadata(path: Path) -> Dict[str, Any]:
    with rasterio.open(path) as ds:
        return raster_verbose_metadata(ds)


def _pretty_print_report(report: List[Dict[str, Any]]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def main() -> int:
    data_root = (Path(__file__).resolve().parent / ".." / "data").resolve()
    report: List[Dict[str, Any]] = []

    for spec in SPECS:
        folder = _folder_from_spec(spec)
        if folder not in INCLUDED_FOLDERS:
            continue
        example = _find_example_file(data_root, spec)
        entry: Dict[str, Any] = {
            "folder": folder,
            "example_file": str(example) if example else None,
        }
        if example:
            try:
                entry["metadata"] = _to_jsonable(_read_metadata(example))
            except Exception as exc:
                entry["error"] = f"failed to read metadata: {exc}"
        else:
            entry["error"] = "no .tif files found"
        report.append(entry)

    pretty = _pretty_print_report(report)
    print(pretty)

    out_path = Path(__file__).resolve().parent / "metadata_report.json"
    out_path.write_text(pretty + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
