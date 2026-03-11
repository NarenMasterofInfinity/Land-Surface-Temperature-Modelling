from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from pyproj import Transformer


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _load_grid_meta(path: Path) -> Tuple[str, List[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    attrs = data["attributes"]
    return str(attrs["crs"]), [float(v) for v in attrs["transform"]]


def _load_region_locations(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _overall_bbox(rows: Sequence[Dict[str, str]]) -> Tuple[float, float, float, float]:
    min_lat = min(float(r["bbox_min_lat"]) for r in rows)
    max_lat = max(float(r["bbox_max_lat"]) for r in rows)
    min_lon = min(float(r["bbox_min_lon"]) for r in rows)
    max_lon = max(float(r["bbox_max_lon"]) for r in rows)
    return min_lat, min_lon, max_lat, max_lon


def _tile_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    rows: int,
    cols: int,
) -> List[Tuple[float, float, float, float]]:
    tiles: List[Tuple[float, float, float, float]] = []
    lat_edges = np.linspace(min_lat, max_lat, rows + 1)
    lon_edges = np.linspace(min_lon, max_lon, cols + 1)
    for i in range(rows):
        for j in range(cols):
            tiles.append((float(lat_edges[i]), float(lon_edges[j]), float(lat_edges[i + 1]), float(lon_edges[j + 1])))
    return tiles


def _overpass_query(tile: Tuple[float, float, float, float], timeout_s: int) -> str:
    south, west, north, east = tile
    return f"""
[out:json][timeout:{timeout_s}];
(
  way["building"]({south},{west},{north},{east});
);
out geom;
"""


def _fetch_overpass_json(query: str, user_agent: str, retries: int, retry_sleep_s: float) -> Dict[str, object]:
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(
            OVERPASS_URL,
            data=body,
            headers={
                "User-Agent": user_agent,
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            if attempt + 1 >= retries:
                raise
            time.sleep(retry_sleep_s * (attempt + 1))
    raise RuntimeError("Unreachable")


def _polygon_area_centroid_xy(points_xy: Sequence[Tuple[float, float]]) -> Tuple[float, float, float]:
    if len(points_xy) < 3:
        return 0.0, math.nan, math.nan
    if points_xy[0] != points_xy[-1]:
        points_xy = list(points_xy) + [points_xy[0]]
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for (x0, y0), (x1, y1) in zip(points_xy[:-1], points_xy[1:]):
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area2) < 1e-9:
        xs = [p[0] for p in points_xy[:-1]]
        ys = [p[1] for p in points_xy[:-1]]
        return 0.0, float(np.mean(xs)), float(np.mean(ys))
    area = area2 / 2.0
    cx /= 3.0 * area2
    cy /= 3.0 * area2
    return abs(area), cx, cy


def _xy_to_rowcol(transform: List[float], x: float, y: float) -> Tuple[int, int]:
    a, b, c, d, e, f, _, _, _ = transform
    det = a * e - b * d
    if abs(det) < 1e-12:
        raise RuntimeError("Non-invertible transform")
    dx = x - c
    dy = y - f
    col = (e * dx - b * dy) / det
    row = (-d * dx + a * dy) / det
    return int(math.floor(row)), int(math.floor(col))


def export_overpass_density(
    run_root: Path,
    grid_meta_path: Path,
    out_csv: Path,
    *,
    tile_rows: int,
    tile_cols: int,
    user_agent: str,
    sleep_seconds: float,
    timeout_s: int,
    retries: int,
    retry_sleep_s: float,
    state_path: Path,
) -> Path:
    region_locations = _load_region_locations(run_root / "step4_regions" / "hotspot_region_locations.csv")
    label_map = np.load(run_root / "step4_regions" / "region_id_map.npy")
    src_crs, transform = _load_grid_meta(grid_meta_path)
    ll_to_grid = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)

    min_lat, min_lon, max_lat, max_lon = _overall_bbox(region_locations)
    tiles = _tile_bbox(min_lat, min_lon, max_lat, max_lon, tile_rows, tile_cols)

    per_region: Dict[int, Dict[str, float]] = {
        int(row["region_id"]): {
            "osm_building_count": 0.0,
            "osm_building_footprint_area_m2": 0.0,
        }
        for row in region_locations
    }

    done_tiles: set[str] = set()
    seen: set[Tuple[str, int]] = set()
    if state_path.exists():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        done_tiles = set(saved.get("done_tiles", []))
        seen = {tuple(item) for item in saved.get("seen_ids", [])}
        for region_id_str, metrics in saved.get("per_region", {}).items():
            region_id = int(region_id_str)
            if region_id in per_region:
                per_region[region_id]["osm_building_count"] = float(metrics["osm_building_count"])
                per_region[region_id]["osm_building_footprint_area_m2"] = float(metrics["osm_building_footprint_area_m2"])

    def save_state() -> None:
        state = {
            "done_tiles": sorted(done_tiles),
            "seen_ids": [list(item) for item in sorted(seen)],
            "per_region": per_region,
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state), encoding="utf-8")

    for tile in tiles:
        tile_key = ",".join(f"{v:.6f}" for v in tile)
        if tile_key in done_tiles:
            continue
        payload = _fetch_overpass_json(_overpass_query(tile, timeout_s), user_agent, retries, retry_sleep_s)
        for elem in payload.get("elements", []):
            if elem.get("type") != "way":
                continue
            elem_id = ("way", int(elem["id"]))
            if elem_id in seen:
                continue
            seen.add(elem_id)
            geom = elem.get("geometry", [])
            if len(geom) < 3:
                continue
            lonlat = [(float(p["lon"]), float(p["lat"])) for p in geom]
            if lonlat[0] != lonlat[-1]:
                lonlat.append(lonlat[0])
            pts_xy = [ll_to_grid.transform(lon, lat) for lon, lat in lonlat]
            area_m2, cx, cy = _polygon_area_centroid_xy(pts_xy)
            if not np.isfinite(cx) or not np.isfinite(cy):
                continue
            row_idx, col_idx = _xy_to_rowcol(transform, cx, cy)
            if row_idx < 0 or col_idx < 0 or row_idx >= label_map.shape[0] or col_idx >= label_map.shape[1]:
                continue
            region_id = int(label_map[row_idx, col_idx])
            if region_id <= 0:
                continue
            per_region[region_id]["osm_building_count"] += 1.0
            per_region[region_id]["osm_building_footprint_area_m2"] += float(area_m2)
        done_tiles.add(tile_key)
        save_state()
        time.sleep(sleep_seconds)

    out_rows: List[Dict[str, object]] = []
    for row in region_locations:
        region_id = int(row["region_id"])
        area_m2 = float(row["area_ha"]) * 10000.0
        count = per_region[region_id]["osm_building_count"]
        footprint_area = per_region[region_id]["osm_building_footprint_area_m2"]
        density = footprint_area / area_m2 if area_m2 > 0 else math.nan
        out_rows.append(
            {
                **row,
                "osm_building_count": int(count),
                "osm_building_footprint_area_m2": footprint_area,
                "osm_building_density_fraction": density,
                "osm_building_density_percent": density * 100.0 if np.isfinite(density) else math.nan,
                "osm_building_count_per_ha": count / float(row["area_ha"]) if float(row["area_ha"]) > 0 else math.nan,
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
    parser = argparse.ArgumentParser(description="Verify hotspot building density using Overpass building footprints.")
    parser.add_argument("--run-root", required=True, help="Actionable insights run root")
    parser.add_argument("--grid-meta", default="madurai_30m.zarr/grid/zarr.json")
    parser.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path. Defaults to <run-root>/step4_regions/hotspot_region_building_density_overpass.csv",
    )
    parser.add_argument("--tile-rows", type=int, default=5)
    parser.add_argument("--tile-cols", type=int, default=5)
    parser.add_argument("--user-agent", default="codex-hotspot-density-overpass/1.0 research")
    parser.add_argument("--sleep-seconds", type=float, default=1.5)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument(
        "--state-path",
        default="",
        help="Progress checkpoint path. Defaults to <run-root>/step4_regions/overpass_density_progress.json",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else run_root / "step4_regions" / "hotspot_region_building_density_overpass.csv"
    )
    state_path = (
        Path(args.state_path)
        if args.state_path
        else run_root / "step4_regions" / "overpass_density_progress.json"
    )
    out = export_overpass_density(
        run_root=run_root,
        grid_meta_path=Path(args.grid_meta),
        out_csv=out_csv,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        user_agent=args.user_agent,
        sleep_seconds=args.sleep_seconds,
        timeout_s=args.timeout_seconds,
        retries=args.retries,
        retry_sleep_s=args.retry_sleep_seconds,
        state_path=state_path,
    )
    print(out)


if __name__ == "__main__":
    main()
