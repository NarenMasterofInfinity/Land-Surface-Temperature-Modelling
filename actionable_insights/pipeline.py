from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import date as Date
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


PIXEL_SIZE_M = 30.0
PIXEL_AREA_HA = (PIXEL_SIZE_M * PIXEL_SIZE_M) / 10000.0
INTERVENTIONS = ("cool_roof", "urban_greening", "cool_pavement")


@dataclass
class PipelineConfig:
    pred_dir: str
    night_pred_dir: str
    out_root: str
    roi_mask: str
    root_30m: Optional[str] = None
    tag: str = "default"
    year: int = 2025
    threshold_mode: str = "percentile"
    hotspot_percentile: float = 90.0
    hotspot_zscore: float = 1.5
    chronic_frequency_threshold: float = 0.30
    min_valid_observations: int = 20
    min_cooling_observations: int = 20
    min_region_pixels: int = 25
    connectivity: int = 8
    sample_png_every: int = 30
    vmin: float = 10.0
    vmax: float = 70.0


@dataclass
class RegionRecord:
    region_id: int
    area_pixels: int
    area_ha: float
    hotspot_frequency_mean: float
    mean_excess_heat_c: float
    hotspot_score_mean: float
    mean_night_cooling_c: float
    cooling_deficit_c: float
    centroid_y: float
    centroid_x: float


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_run_dirs(root: Path, tag: str) -> Dict[str, Path]:
    base = root / tag
    out = {
        "root": base,
        "step1": base / "step1_baseline",
        "step2": base / "step2_daily_hotspots",
        "step3": base / "step3_persistence",
        "step4": base / "step4_regions",
        "step5": base / "step5_interventions",
    }
    for path in out.values():
        _ensure_dir(path)
    return out


def _validate_config(config: PipelineConfig) -> None:
    if config.year < 1900 or config.year > 2100:
        raise RuntimeError(f"Unsupported year: {config.year}")
    if config.threshold_mode not in {"percentile", "zscore"}:
        raise RuntimeError(f"Unsupported threshold_mode: {config.threshold_mode}")
    if not (0.0 < config.hotspot_percentile <= 100.0):
        raise RuntimeError(f"hotspot_percentile must be in (0, 100], got {config.hotspot_percentile}")
    if config.hotspot_zscore <= 0.0:
        raise RuntimeError(f"hotspot_zscore must be > 0, got {config.hotspot_zscore}")
    if not (0.0 <= config.chronic_frequency_threshold <= 1.0):
        raise RuntimeError(
            f"chronic_frequency_threshold must be in [0, 1], got {config.chronic_frequency_threshold}"
        )
    if config.min_valid_observations <= 0:
        raise RuntimeError(f"min_valid_observations must be > 0, got {config.min_valid_observations}")
    if config.min_cooling_observations <= 0:
        raise RuntimeError(f"min_cooling_observations must be > 0, got {config.min_cooling_observations}")
    if config.min_region_pixels <= 0:
        raise RuntimeError(f"min_region_pixels must be > 0, got {config.min_region_pixels}")
    if config.connectivity not in {4, 8}:
        raise RuntimeError(f"connectivity must be 4 or 8, got {config.connectivity}")
    if config.sample_png_every < 0:
        raise RuntimeError(f"sample_png_every must be >= 0, got {config.sample_png_every}")
    if config.vmin >= config.vmax:
        raise RuntimeError(f"vmin must be < vmax, got vmin={config.vmin}, vmax={config.vmax}")


def _discover_prediction_files(pred_dir: Path, *, year: Optional[int] = None) -> List[Tuple[Date, Path]]:
    items: List[Tuple[Date, Path]] = []
    for path in sorted(pred_dir.glob("lst_*.npy")):
        stem = path.stem
        suffix = stem.replace("lst_", "")
        if len(suffix) != 8 or not suffix.isdigit():
            continue
        ts = datetime.strptime(suffix, "%Y%m%d").date()
        if year is not None and ts.year != year:
            continue
        items.append((ts, path))
    if not items:
        year_msg = f" for year {year}" if year is not None else ""
        raise RuntimeError(f"No prediction .npy files found in {pred_dir}{year_msg}")
    return items


def _discover_prediction_map(pred_dir: Path, *, year: Optional[int] = None) -> Dict[str, Path]:
    return {ts.strftime("%Y-%m-%d"): path for ts, path in _discover_prediction_files(pred_dir, year=year)}


def _load_roi_mask(mask_path: Path) -> np.ndarray:
    mask = np.load(mask_path)
    if mask.ndim != 2:
        raise RuntimeError(f"ROI mask must be 2D, got shape={mask.shape}")
    return mask.astype(bool, copy=False)


def _save_array(path: Path, arr: np.ndarray) -> None:
    np.save(path, arr)


def _write_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cooked: Dict[str, object] = {}
            for key, value in row.items():
                if isinstance(value, Date):
                    cooked[key] = value.isoformat()
                else:
                    cooked[key] = value
            writer.writerow(cooked)


def _save_heatmap(
    path: Path,
    arr: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_label: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    if cbar_label is not None:
        cbar = fig.colorbar(im, ax=ax, shrink=0.82)
        cbar.set_label(cbar_label)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_binary_map(path: Path, arr: np.ndarray, *, title: str) -> None:
    _save_heatmap(path, arr.astype(np.float32), title=title, cmap="gray", vmin=0.0, vmax=1.0, cbar_label=None)


def _mad(x: np.ndarray) -> float:
    med = float(np.nanmedian(x))
    return float(np.nanmedian(np.abs(x - med)))


def _daily_threshold(anomaly_vals: np.ndarray, mode: str, percentile: float, zscore: float) -> float:
    if mode == "percentile":
        return float(np.nanpercentile(anomaly_vals, percentile))
    med = float(np.nanmedian(anomaly_vals))
    mad = _mad(anomaly_vals)
    robust_sigma = 1.4826 * mad
    return med + zscore * robust_sigma


def _plot_baseline_series(rows: List[Dict[str, object]], out_path: Path) -> None:
    dates = [row["date"] for row in rows]
    baseline = np.asarray([row["baseline_median_c"] for row in rows], dtype=np.float32)
    mad = np.asarray([row["mad_c"] for row in rows], dtype=np.float32)
    fig, ax1 = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax1.plot(dates, baseline, label="baseline_median", color="tab:red")
    ax1.fill_between(dates, baseline - mad, baseline + mad, alpha=0.2, color="tab:red")
    ax1.set_ylabel("Temperature (C)")
    ax1.set_title("Daily City Baseline")
    ax1.grid(alpha=0.25)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_hotspot_fraction(rows: List[Dict[str, object]], out_path: Path) -> None:
    dates = [row["date"] for row in rows]
    fractions = [row["hotspot_fraction"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(dates, fractions, color="tab:blue")
    ax.set_title("Daily Hotspot Fraction")
    ax.set_ylabel("Fraction of ROI")
    ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_cooling_series(rows: List[Dict[str, object]], out_path: Path) -> None:
    finite_rows = [row for row in rows if np.isfinite(row["mean_night_cooling_c"])]
    if not finite_rows:
        return
    dates = [row["date"] for row in finite_rows]
    vals = [row["mean_night_cooling_c"] for row in finite_rows]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(dates, vals, color="tab:purple")
    ax.set_title("Daily Day-to-Night Cooling")
    ax.set_ylabel("Cooling (C)")
    ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_static_layers(
    root_30m: Optional[Path],
    expected_shape: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if root_30m is None:
        return None, None
    if not root_30m.exists():
        raise FileNotFoundError(f"Static 30m data not found: {root_30m}")
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("zarr is required to load static 30m layers") from exc

    root = zarr.open_group(str(root_30m), mode="r")
    world = np.asarray(root["static_30m"]["worldcover"]["data"][0], dtype=np.float32).squeeze()
    dyn = np.asarray(root["static_30m"]["dynamic_world"]["data"][0], dtype=np.float32).squeeze()
    if world.ndim != 2 or dyn.ndim != 2:
        raise RuntimeError(
            f"Static layers must be 2D, got worldcover={world.shape}, dynamic_world={dyn.shape}"
        )
    if world.shape != expected_shape or dyn.shape != expected_shape:
        raise RuntimeError(
            f"Static layer shape mismatch: worldcover={world.shape}, dynamic_world={dyn.shape}, expected={expected_shape}"
        )
    return world, dyn


def _suitability_masks(world: Optional[np.ndarray], dyn: Optional[np.ndarray], chronic_mask: np.ndarray) -> Dict[str, np.ndarray]:
    if world is None or dyn is None:
        base = chronic_mask.astype(bool, copy=False)
        return {name: base.copy() for name in INTERVENTIONS}

    built = (world == 50) | (dyn == 6)
    bare = (world == 60) | (dyn == 7)
    water = (world == 80) | (dyn == 0)

    return {
        "cool_roof": chronic_mask & built,
        "urban_greening": chronic_mask & (built | bare) & (~water),
        "cool_pavement": chronic_mask & (built | bare) & (~water),
    }


def _cooling_maps(score: np.ndarray, chronic_mask: np.ndarray, suitability: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    valid_score = score[chronic_mask & np.isfinite(score)]
    if valid_score.size == 0:
        scale = np.zeros_like(score, dtype=np.float32)
    else:
        denom = max(float(np.nanpercentile(valid_score, 95)), 1e-6)
        scale = np.clip(score / denom, 0.0, 1.5).astype(np.float32, copy=False)

    specs = {
        "cool_roof": (1.1, 0.9),
        "urban_greening": (1.3, 1.1),
        "cool_pavement": (0.8, 0.7),
    }
    out: Dict[str, np.ndarray] = {}
    for name, (base_c, extra_c) in specs.items():
        delta = (base_c + extra_c * scale) * suitability[name].astype(np.float32)
        delta[~np.isfinite(delta)] = 0.0
        out[name] = delta.astype(np.float32, copy=False)
    return out


def _label_connected(mask: np.ndarray, min_pixels: int, connectivity: int) -> np.ndarray:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    region_id = 0
    if connectivity == 4:
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        neighbors = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            queue = deque([(y, x)])
            pixels: List[Tuple[int, int]] = []
            labels[y, x] = -1
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or nx < 0 or ny >= h or nx >= w:
                        continue
                    if not mask[ny, nx] or labels[ny, nx] != 0:
                        continue
                    labels[ny, nx] = -1
                    queue.append((ny, nx))
            if len(pixels) < min_pixels:
                for py, px in pixels:
                    labels[py, px] = 0
                continue
            region_id += 1
            for py, px in pixels:
                labels[py, px] = region_id
    return labels


def _region_table(
    labels: np.ndarray,
    frequency: np.ndarray,
    mean_excess: np.ndarray,
    score: np.ndarray,
    mean_night_cooling: np.ndarray,
    cooling_deficit: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[RegionRecord] = []
    for region_id in range(1, int(labels.max()) + 1):
        mask = labels == region_id
        if not np.any(mask):
            continue
        ys, xs = np.where(mask)
        rows.append(
            RegionRecord(
                region_id=region_id,
                area_pixels=int(mask.sum()),
                area_ha=float(mask.sum() * PIXEL_AREA_HA),
                hotspot_frequency_mean=float(np.nanmean(frequency[mask])),
                mean_excess_heat_c=float(np.nanmean(mean_excess[mask])),
                hotspot_score_mean=float(np.nanmean(score[mask])),
                mean_night_cooling_c=float(np.nanmean(mean_night_cooling[mask])),
                cooling_deficit_c=float(np.nanmean(cooling_deficit[mask])),
                centroid_y=float(np.mean(ys)),
                centroid_x=float(np.mean(xs)),
            )
        )
    return [asdict(r) for r in rows]


def _plot_region_scores(region_rows: List[Dict[str, object]], out_path: Path) -> None:
    if not region_rows:
        return
    top = sorted(region_rows, key=lambda row: row["hotspot_score_mean"], reverse=True)[:15]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar([str(row["region_id"]) for row in top], [row["hotspot_score_mean"] for row in top], color="tab:red")
    ax.set_title("Top Hotspot Regions by Score")
    ax.set_xlabel("Region")
    ax.set_ylabel("Hotspot score")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _intervention_table(
    region_rows: List[Dict[str, object]],
    labels: np.ndarray,
    cooling_maps: Dict[str, np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for region in region_rows:
        region_id = int(region["region_id"])
        mask = labels == region_id
        mean_excess = float(region["mean_excess_heat_c"])
        hotspot_score_mean = float(region["hotspot_score_mean"])
        cooling_deficit = float(region["cooling_deficit_c"])
        mean_night_cooling = float(region["mean_night_cooling_c"])
        if not np.isfinite(mean_excess):
            mean_excess = 0.0
        if not np.isfinite(hotspot_score_mean):
            hotspot_score_mean = 0.0
        if not np.isfinite(cooling_deficit):
            cooling_deficit = 0.0
        cooling_leverage = 1.0 + max(cooling_deficit, 0.0)
        for name, delta_map in cooling_maps.items():
            vals = delta_map[mask]
            proxy_mean_cooling = float(np.nanmean(vals)) if vals.size else 0.0
            proxy_max_cooling = float(np.nanmax(vals)) if vals.size else 0.0
            feasible_frac = float(np.mean(vals > 0)) if vals.size else 0.0
            proxy_reduction_pct = float(
                np.clip(100.0 * proxy_mean_cooling / max(mean_excess, 0.25), 0.0, 100.0)
            )
            priority_score = (
                float(region["area_ha"])
                * proxy_mean_cooling
                * max(hotspot_score_mean, 0.0)
                * max(feasible_frac, 0.0)
                * cooling_leverage
            )
            rows.append(
                {
                    "region_id": region_id,
                    "intervention": name,
                    "area_ha": float(region["area_ha"]),
                    "hotspot_score_mean": hotspot_score_mean,
                    "mean_excess_heat_c": mean_excess,
                    "mean_night_cooling_c": mean_night_cooling,
                    "cooling_deficit_c": cooling_deficit,
                    "cooling_leverage_factor": cooling_leverage,
                    "feasible_fraction": feasible_frac,
                    "proxy_mean_cooling_c": proxy_mean_cooling,
                    "proxy_max_cooling_c": proxy_max_cooling,
                    "proxy_hotspot_freq_reduction_pct": proxy_reduction_pct,
                    "priority_score": priority_score,
                }
            )
    return rows


def _best_intervention_map(
    labels: np.ndarray,
    intervention_rows: List[Dict[str, object]],
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    if not intervention_rows:
        return np.zeros_like(labels, dtype=np.int16), intervention_rows
    best_by_region: Dict[int, Dict[str, object]] = {}
    for row in sorted(intervention_rows, key=lambda item: item["priority_score"], reverse=True):
        region_id = int(row["region_id"])
        if region_id not in best_by_region:
            best_by_region[region_id] = dict(row)
    best = list(best_by_region.values())
    code_map = {"cool_roof": 1, "urban_greening": 2, "cool_pavement": 3}
    out = np.zeros_like(labels, dtype=np.int16)
    for row in best:
        out[labels == int(row["region_id"])] = code_map[str(row["intervention"])]
    for row in best:
        row["intervention_code"] = code_map[str(row["intervention"])]
    return out, best


def _plot_best_intervention_map(path: Path, arr: np.ndarray) -> None:
    cmap = plt.get_cmap("tab10", 4)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=3)
    ax.set_title("Recommended Intervention by Region")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["none", "roof", "green", "pavement"])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(config: PipelineConfig) -> Dict[str, str]:
    _validate_config(config)
    pred_dir = Path(config.pred_dir)
    night_pred_dir = Path(config.night_pred_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Day prediction directory not found: {pred_dir}")
    if not night_pred_dir.exists():
        raise FileNotFoundError(f"Night prediction directory not found: {night_pred_dir}")
    roi_mask = _load_roi_mask(Path(config.roi_mask))
    files = _discover_prediction_files(pred_dir, year=config.year)
    night_map = _discover_prediction_map(night_pred_dir, year=config.year)
    sample_arr = np.load(files[0][1])
    if sample_arr.ndim != 2:
        raise RuntimeError(f"Prediction arrays must be 2D, got shape={sample_arr.shape} for {files[0][1]}")
    h, w = sample_arr.shape
    if roi_mask.shape != (h, w):
        raise RuntimeError(f"ROI mask shape {roi_mask.shape} does not match prediction shape {(h, w)}")

    out_dirs = _make_run_dirs(Path(config.out_root), config.tag)
    (out_dirs["root"] / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    freq_count = np.zeros((h, w), dtype=np.uint32)
    valid_count = np.zeros((h, w), dtype=np.uint32)
    excess_sum = np.zeros((h, w), dtype=np.float32)
    cooling_sum = np.zeros((h, w), dtype=np.float32)
    cooling_count = np.zeros((h, w), dtype=np.uint32)
    daily_rows: List[Dict[str, object]] = []
    matched_night_dates = 0
    missing_night_dates = 0

    for i, (date, path) in enumerate(files):
        arr = np.load(path).astype(np.float32, copy=False)
        if arr.shape != (h, w):
            raise RuntimeError(f"Prediction shape mismatch for {path}: {arr.shape} vs {(h, w)}")
        valid = roi_mask & np.isfinite(arr)
        if not np.any(valid):
            daily_rows.append(
                {
                    "date": date,
                    "valid_pixels": 0,
                    "baseline_median_c": np.nan,
                    "mad_c": np.nan,
                    "threshold_c": np.nan,
                    "hotspot_pixels": 0,
                    "hotspot_fraction": 0.0,
                    "mean_anomaly_hot_pixels": np.nan,
                    "has_night_pair": False,
                    "mean_night_cooling_c": np.nan,
                }
            )
            continue

        vals = arr[valid]
        baseline = float(np.nanmedian(vals))
        mad = _mad(vals)
        anomaly = arr - baseline
        threshold = _daily_threshold(anomaly[valid], config.threshold_mode, config.hotspot_percentile, config.hotspot_zscore)
        hot = valid & (anomaly >= threshold)

        date_key = date.isoformat()
        night_path = night_map.get(date_key)
        mean_night_cooling_day = np.nan
        has_night_pair = night_path is not None
        if night_path is not None:
            night_arr = np.load(night_path).astype(np.float32, copy=False)
            if night_arr.shape != arr.shape:
                raise RuntimeError(f"Night prediction shape mismatch for {date_key}: {night_arr.shape} vs {arr.shape}")
            cooling_valid = valid & np.isfinite(night_arr)
            if np.any(cooling_valid):
                cooling = arr - night_arr
                cooling_sum += np.where(cooling_valid, cooling, 0.0).astype(np.float32, copy=False)
                cooling_count += cooling_valid.astype(np.uint32)
                mean_night_cooling_day = float(np.nanmean(cooling[cooling_valid]))
            matched_night_dates += 1
        else:
            missing_night_dates += 1

        valid_count += valid.astype(np.uint32)
        freq_count += hot.astype(np.uint32)
        excess_sum += np.where(hot, anomaly, 0.0).astype(np.float32, copy=False)

        hotspot_fraction = float(hot.sum() / max(int(valid.sum()), 1))
        hot_mean = float(np.nanmean(anomaly[hot])) if np.any(hot) else np.nan
        daily_rows.append(
            {
                "date": date,
                "valid_pixels": int(valid.sum()),
                "baseline_median_c": baseline,
                "mad_c": mad,
                "threshold_c": float(threshold),
                "hotspot_pixels": int(hot.sum()),
                "hotspot_fraction": hotspot_fraction,
                "mean_anomaly_hot_pixels": hot_mean,
                "has_night_pair": has_night_pair,
                "mean_night_cooling_c": mean_night_cooling_day,
            }
        )

        if config.sample_png_every > 0 and (i % config.sample_png_every == 0):
            _save_heatmap(
                out_dirs["step2"] / f"sample_lst_{date.strftime('%Y%m%d')}.png",
                np.where(roi_mask, arr, np.nan),
                title=f"LST {date}",
                cmap="inferno",
                vmin=config.vmin,
                vmax=config.vmax,
                cbar_label="LST (C)",
            )
            _save_binary_map(
                out_dirs["step2"] / f"sample_hotspot_{date.strftime('%Y%m%d')}.png",
                hot.astype(np.uint8),
                title=f"Hotspots {date}",
            )

    _write_rows_csv(out_dirs["step1"] / "daily_baseline.csv", daily_rows)
    _write_rows_csv(out_dirs["step2"] / "daily_hotspot_summary.csv", daily_rows)
    _plot_baseline_series(daily_rows, out_dirs["step1"] / "baseline_series.png")
    _plot_hotspot_fraction(daily_rows, out_dirs["step2"] / "hotspot_fraction_series.png")
    _plot_cooling_series(daily_rows, out_dirs["step2"] / "day_night_cooling_series.png")

    observation_ok = valid_count >= config.min_valid_observations
    cooling_ok = cooling_count >= config.min_cooling_observations

    frequency = np.divide(freq_count, np.maximum(valid_count, 1), dtype=np.float32)
    frequency[~observation_ok] = np.nan
    mean_excess = np.divide(excess_sum, np.maximum(freq_count, 1), dtype=np.float32)
    mean_excess[(freq_count == 0) | (~observation_ok)] = np.nan
    score = (frequency * np.nan_to_num(mean_excess, nan=0.0)).astype(np.float32, copy=False)
    score[~observation_ok] = np.nan
    mean_night_cooling = np.divide(cooling_sum, np.maximum(cooling_count, 1), dtype=np.float32)
    mean_night_cooling[~cooling_ok] = np.nan
    cooling_vals = mean_night_cooling[roi_mask & np.isfinite(mean_night_cooling)]
    cooling_baseline = float(np.nanmedian(cooling_vals)) if cooling_vals.size else 0.0
    cooling_deficit = np.full((h, w), np.nan, dtype=np.float32)
    if np.any(cooling_ok):
        cooling_deficit[cooling_ok] = np.clip(
            cooling_baseline - mean_night_cooling[cooling_ok],
            0.0,
            None,
        ).astype(np.float32, copy=False)
    chronic_mask = (
        observation_ok
        & (frequency >= config.chronic_frequency_threshold)
        & roi_mask
        & np.isfinite(score)
        & (score > 0)
    )

    _save_array(out_dirs["step3"] / "hotspot_frequency.npy", frequency)
    _save_array(out_dirs["step3"] / "mean_excess_heat.npy", mean_excess)
    _save_array(out_dirs["step3"] / "hotspot_score.npy", score)
    _save_array(out_dirs["step3"] / "mean_night_cooling.npy", mean_night_cooling)
    _save_array(out_dirs["step3"] / "cooling_deficit.npy", cooling_deficit)
    _save_array(out_dirs["step3"] / "chronic_hotspot_mask.npy", chronic_mask.astype(np.uint8))

    _save_heatmap(out_dirs["step3"] / "hotspot_frequency.png", np.where(roi_mask, frequency, np.nan), title="Hotspot Frequency", cmap="magma", vmin=0.0, vmax=1.0, cbar_label="fraction")
    _save_heatmap(out_dirs["step3"] / "mean_excess_heat.png", np.where(roi_mask, mean_excess, np.nan), title="Mean Excess Heat", cmap="inferno", vmin=0.0, vmax=float(np.nanpercentile(mean_excess[np.isfinite(mean_excess)], 99)) if np.isfinite(mean_excess).any() else 1.0, cbar_label="C")
    _save_heatmap(out_dirs["step3"] / "hotspot_score.png", np.where(roi_mask, score, np.nan), title="Hotspot Score", cmap="plasma", vmin=0.0, vmax=float(np.nanpercentile(score[np.isfinite(score)], 99)) if np.isfinite(score).any() else 1.0, cbar_label="score")
    _save_heatmap(out_dirs["step3"] / "mean_night_cooling.png", np.where(roi_mask, mean_night_cooling, np.nan), title="Mean Day-to-Night Cooling", cmap="viridis", vmin=float(np.nanpercentile(mean_night_cooling[np.isfinite(mean_night_cooling)], 1)) if np.isfinite(mean_night_cooling).any() else 0.0, vmax=float(np.nanpercentile(mean_night_cooling[np.isfinite(mean_night_cooling)], 99)) if np.isfinite(mean_night_cooling).any() else 1.0, cbar_label="C")
    _save_heatmap(out_dirs["step3"] / "cooling_deficit.png", np.where(roi_mask, cooling_deficit, np.nan), title="Cooling Deficit", cmap="cividis", vmin=0.0, vmax=float(np.nanpercentile(cooling_deficit[np.isfinite(cooling_deficit)], 99)) if np.isfinite(cooling_deficit).any() else 1.0, cbar_label="C")
    _save_binary_map(out_dirs["step3"] / "chronic_hotspot_mask.png", chronic_mask.astype(np.uint8), title="Chronic Hotspot Mask")

    labels = _label_connected(chronic_mask, config.min_region_pixels, config.connectivity)
    region_rows = _region_table(labels, frequency, mean_excess, score, mean_night_cooling, cooling_deficit)
    _write_rows_csv(out_dirs["step4"] / "hotspot_regions.csv", region_rows)
    _save_array(out_dirs["step4"] / "region_id_map.npy", labels)
    _save_heatmap(
        out_dirs["step4"] / "region_id_map.png",
        np.where(labels > 0, labels, np.nan),
        title="Hotspot Regions",
        cmap="tab20",
        vmin=1.0,
        vmax=float(max(1, labels.max())),
        cbar_label="region id",
    )
    _plot_region_scores(region_rows, out_dirs["step4"] / "top_hotspot_regions.png")

    world, dyn = _load_static_layers(Path(config.root_30m) if config.root_30m else None, (h, w))
    suitability = _suitability_masks(world, dyn, chronic_mask)
    cooling_maps = _cooling_maps(score, chronic_mask, suitability)
    intervention_rows = _intervention_table(region_rows, labels, cooling_maps)
    _write_rows_csv(out_dirs["step5"] / "intervention_region_scores.csv", intervention_rows)

    for name, arr in cooling_maps.items():
        _save_array(out_dirs["step5"] / f"{name}_cooling.npy", arr)
        vmax = float(np.nanpercentile(arr[arr > 0], 99)) if np.any(arr > 0) else 1.0
        _save_heatmap(
            out_dirs["step5"] / f"{name}_cooling.png",
            np.where(arr > 0, arr, np.nan),
            title=f"{name.replace('_', ' ').title()} Cooling",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            cbar_label="proxy cooling (C)",
        )

    best_map, best_rows = _best_intervention_map(labels, intervention_rows)
    best_rows = sorted(best_rows, key=lambda row: row["priority_score"], reverse=True)
    _write_rows_csv(out_dirs["step5"] / "actionable_insights.csv", best_rows)
    _save_array(out_dirs["step5"] / "recommended_intervention_map.npy", best_map)
    _plot_best_intervention_map(out_dirs["step5"] / "recommended_intervention_map.png", best_map)

    run_summary = {
        "year": config.year,
        "day_dates_considered": len(files),
        "night_dates_available": len(night_map),
        "matched_night_dates": matched_night_dates,
        "missing_night_dates": missing_night_dates,
        "min_valid_observations": config.min_valid_observations,
        "min_cooling_observations": config.min_cooling_observations,
        "connectivity": config.connectivity,
        "n_regions": int(labels.max()),
        "n_ranked_interventions": int(len(best_rows)),
    }
    (out_dirs["root"] / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    return {k: str(v) for k, v in out_dirs.items()}
