from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from pyproj import Transformer


def _load_grid_meta(path: Path) -> Tuple[str, List[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    attrs = data["attributes"]
    return str(attrs["crs"]), [float(v) for v in attrs["transform"]]


def _pixel_center_xy(transform: List[float], row: float, col: float) -> Tuple[float, float]:
    a, b, c, d, e, f, _, _, _ = transform
    x = c + a * (col + 0.5) + b * (row + 0.5)
    y = f + d * (col + 0.5) + e * (row + 0.5)
    return x, y


def _pixel_corner_xy(transform: List[float], row: float, col: float) -> Tuple[float, float]:
    a, b, c, d, e, f, _, _, _ = transform
    x = c + a * col + b * row
    y = f + d * col + e * row
    return x, y


def _bbox_lonlat(
    transform: List[float],
    transformer: Transformer,
    min_row: int,
    min_col: int,
    max_row: int,
    max_col: int,
) -> Dict[str, float]:
    corners = [
        _pixel_corner_xy(transform, min_row, min_col),
        _pixel_corner_xy(transform, min_row, max_col + 1),
        _pixel_corner_xy(transform, max_row + 1, min_col),
        _pixel_corner_xy(transform, max_row + 1, max_col + 1),
    ]
    lonlat = [transformer.transform(x, y) for x, y in corners]
    lons = [pt[0] for pt in lonlat]
    lats = [pt[1] for pt in lonlat]
    return {
        "bbox_min_lon": min(lons),
        "bbox_max_lon": max(lons),
        "bbox_min_lat": min(lats),
        "bbox_max_lat": max(lats),
    }


def _read_region_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _region_extents(label_map: np.ndarray) -> Dict[int, Dict[str, int]]:
    extents: Dict[int, Dict[str, int]] = {}
    region_ids = np.unique(label_map)
    region_ids = region_ids[region_ids > 0]
    for region_id in region_ids:
        ys, xs = np.where(label_map == region_id)
        extents[int(region_id)] = {
            "min_row": int(ys.min()),
            "max_row": int(ys.max()),
            "min_col": int(xs.min()),
            "max_col": int(xs.max()),
        }
    return extents


def _load_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_cache(path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=True), encoding="utf-8")


def _reverse_geocode(lat: float, lon: float, user_agent: str) -> Dict[str, str]:
    params = urllib.parse.urlencode(
        {
            "format": "jsonv2",
            "lat": f"{lat:.7f}",
            "lon": f"{lon:.7f}",
            "zoom": 18,
            "addressdetails": 1,
        }
    )
    url = f"https://nominatim.openstreetmap.org/reverse?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    address = payload.get("address", {})
    return {
        "display_name": payload.get("display_name", ""),
        "place_name": address.get("neighbourhood")
        or address.get("suburb")
        or address.get("quarter")
        or address.get("hamlet")
        or address.get("village")
        or address.get("town")
        or address.get("city_district")
        or address.get("city")
        or address.get("county")
        or "",
        "road": address.get("road", ""),
        "neighbourhood": address.get("neighbourhood", ""),
        "suburb": address.get("suburb", ""),
        "city_district": address.get("city_district", ""),
        "village": address.get("village", ""),
        "town": address.get("town", ""),
        "city": address.get("city", ""),
        "county": address.get("county", ""),
        "state_district": address.get("state_district", ""),
        "state": address.get("state", ""),
        "postcode": address.get("postcode", ""),
        "country": address.get("country", ""),
        "country_code": address.get("country_code", ""),
    }


def export_locations(
    run_root: Path,
    grid_meta_path: Path,
    out_csv: Path,
    *,
    do_reverse_geocode: bool,
    cache_path: Path,
    user_agent: str,
    sleep_seconds: float,
) -> Path:
    region_csv = run_root / "step4_regions" / "hotspot_regions.csv"
    label_map_path = run_root / "step4_regions" / "region_id_map.npy"
    rows = _read_region_rows(region_csv)
    label_map = np.load(label_map_path)
    extents = _region_extents(label_map)
    src_crs, transform = _load_grid_meta(grid_meta_path)
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    cache = _load_cache(cache_path)

    out_rows: List[Dict[str, object]] = []
    for row in rows:
        region_id = int(row["region_id"])
        extent = extents[region_id]
        centroid_row = float(row["centroid_y"])
        centroid_col = float(row["centroid_x"])
        centroid_x_utm, centroid_y_utm = _pixel_center_xy(transform, centroid_row, centroid_col)
        centroid_lon, centroid_lat = transformer.transform(centroid_x_utm, centroid_y_utm)
        bbox = _bbox_lonlat(
            transform,
            transformer,
            extent["min_row"],
            extent["min_col"],
            extent["max_row"],
            extent["max_col"],
        )

        geo = {
            "display_name": "",
            "place_name": "",
            "road": "",
            "neighbourhood": "",
            "suburb": "",
            "city_district": "",
            "village": "",
            "town": "",
            "city": "",
            "county": "",
            "state_district": "",
            "state": "",
            "postcode": "",
            "country": "",
            "country_code": "",
        }
        if do_reverse_geocode:
            key = f"{centroid_lat:.7f},{centroid_lon:.7f}"
            if key not in cache:
                cache[key] = _reverse_geocode(centroid_lat, centroid_lon, user_agent)
                _save_cache(cache_path, cache)
                time.sleep(sleep_seconds)
            geo = cache[key]

        out_rows.append(
            {
                **row,
                "centroid_row": centroid_row,
                "centroid_col": centroid_col,
                "centroid_easting_m": centroid_x_utm,
                "centroid_northing_m": centroid_y_utm,
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
                **extent,
                **bbox,
                **geo,
            }
        )

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Export hotspot region coordinates and optional place names.")
    parser.add_argument("--run-root", required=True, help="Actionable insights run root")
    parser.add_argument("--grid-meta", default="madurai_30m.zarr/grid/zarr.json", help="Grid metadata JSON path")
    parser.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path. Defaults to <run-root>/step4_regions/hotspot_region_locations.csv",
    )
    parser.add_argument("--reverse-geocode", action="store_true", help="Attach centroid place names using Nominatim")
    parser.add_argument(
        "--cache-path",
        default="",
        help="Cache path for reverse geocoding responses. Defaults to <run-root>/step4_regions/reverse_geocode_cache.json",
    )
    parser.add_argument(
        "--user-agent",
        default="codex-hotspot-region-export/1.0 (research use)",
        help="User agent for reverse geocoding requests",
    )
    parser.add_argument("--sleep-seconds", type=float, default=1.1, help="Delay between reverse-geocode requests")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_csv = Path(args.out_csv) if args.out_csv else run_root / "step4_regions" / "hotspot_region_locations.csv"
    cache_path = (
        Path(args.cache_path)
        if args.cache_path
        else run_root / "step4_regions" / "reverse_geocode_cache.json"
    )
    out = export_locations(
        run_root=run_root,
        grid_meta_path=Path(args.grid_meta),
        out_csv=out_csv,
        do_reverse_geocode=args.reverse_geocode,
        cache_path=cache_path,
        user_agent=args.user_agent,
        sleep_seconds=args.sleep_seconds,
    )
    print(out)


if __name__ == "__main__":
    main()
