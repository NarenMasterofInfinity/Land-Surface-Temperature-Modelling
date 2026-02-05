#!/usr/bin/env python3
"""
Create a multi-panel comparison figure like the reference image:
Land cover + multiple products for a single day.

This script auto-discovers products in madurai.zarr and plots one panel per product.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

try:
    import zarr  # type: ignore
except Exception as exc:
    raise RuntimeError("zarr is required for this script") from exc


# ---------------------------
# Land-cover palette (Dynamic World default)
# 0: Water, 1: Trees, 2: Grass, 3: Flooded vegetation, 4: Crops,
# 5: Shrub & scrub, 6: Built area, 7: Bare ground, 8: Snow/ice
# ---------------------------
DW_LABELS = [
    "Water",
    "Forest",
    "Grassland",
    "Wetland",
    "Cropland",
    "Shrubland",
    "Urban",
    "Barren land",
    "Snow/Ice",
]
DW_COLORS = [
    "#419BDF",  # Water
    "#397D49",  # Forest
    "#88B053",  # Grassland
    "#7A87C6",  # Wetland
    "#E49635",  # Cropland
    "#DFC35A",  # Shrubland
    "#C4281B",  # Urban
    "#A59B8F",  # Barren land
    "#B39FE1",  # Snow/Ice
]


def _load_zarr_array(path: str | Path) -> np.ndarray:
    arr = zarr.open(str(path), mode="r")
    return np.asarray(arr)


def _load_time_index(time_zarr: str | Path) -> List[str]:
    arr = zarr.open(str(time_zarr), mode="r")
    values = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in arr[:]]
    return values


def _load_landcover(landcover_zarr: str | Path) -> np.ndarray:
    arr = _load_zarr_array(landcover_zarr)[0, 0, :, :].astype("float32")
    lc_path = Path(landcover_zarr)
    valid_path = lc_path.parent / "valid"
    if valid_path.exists():
        v = _load_zarr_array(valid_path)
        try:
            v = v[0, 0, :, :]
        except Exception:
            v = v[0, :, :]
        arr = np.where(v > 0.0, arr, np.nan)
    return arr


def _normalize_date_str(value: str) -> str:
    value = value.strip()
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else digits


def _find_date_index(values: Sequence[str], date: str) -> Optional[int]:
    if date in values:
        return values.index(date)
    target = _normalize_date_str(date)
    if not target:
        return None
    norm_map = [_normalize_date_str(v) for v in values]
    try:
        return norm_map.index(target)
    except ValueError:
        return None


def _list_products(products_root: str | Path) -> List[str]:
    products_root = Path(products_root)
    if not products_root.exists():
        return []
    names = []
    for p in sorted(products_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "data").exists():
            names.append(p.name)
    return names


def _load_time_optional(path: str | Path) -> Optional[List[str]]:
    if Path(path).exists():
        return _load_time_index(path)
    return None


def _pick_time_index(
    time_values: Optional[Sequence[str]],
    date: str,
    date_index: Optional[int],
    *,
    month_only: bool = False,
) -> Optional[int]:
    if time_values is None:
        return None
    if date_index is not None:
        idx = int(date_index)
        if 0 <= idx < len(time_values):
            return idx
    if not date:
        return None
    if month_only:
        target = _normalize_date_str(date)[:6]
        norm_map = [_normalize_date_str(v)[:6] for v in time_values]
        try:
            return norm_map.index(target)
        except ValueError:
            return None
    return _find_date_index(time_values, date)


def _load_product_array(
    product_name: str,
    product_root: str | Path,
    date: str,
    *,
    date_index: Optional[int],
    band_map: Dict[str, int],
    time_daily: Optional[Sequence[str]],
    time_monthly: Optional[Sequence[str]],
) -> Tuple[np.ndarray, Optional[float]]:
    arr_path = Path(product_root) / product_name / "data"
    arr = zarr.open(str(arr_path), mode="r")
    shape = arr.shape
    ndim = arr.ndim

    t_idx = None
    if ndim >= 3 and time_daily and shape[0] == len(time_daily):
        t_idx = _pick_time_index(time_daily, date, date_index)
    elif ndim >= 3 and time_monthly and shape[0] == len(time_monthly):
        t_idx = _pick_time_index(time_monthly, date, date_index, month_only=True)

    if t_idx is None and ndim >= 3:
        t_idx = 0

    band = int(band_map.get(product_name, 0))

    if ndim == 4:
        data = np.asarray(arr[t_idx, band, :, :]).astype("float32")
    elif ndim == 3:
        data = np.asarray(arr[t_idx, :, :]).astype("float32")
    elif ndim == 2:
        data = np.asarray(arr[:, :]).astype("float32")
    else:
        raise ValueError(f"Unsupported array shape for {product_name}: {shape}")

    valid_frac: Optional[float] = None
    # Apply valid mask if present and shape-compatible
    valid_path = Path(product_root) / product_name / "valid"
    if valid_path.exists() and ndim >= 3:
        v_arr = zarr.open(str(valid_path), mode="r")
        try:
            v = np.asarray(v_arr[t_idx, :, :]).astype("float32")
        except Exception:
            v = np.asarray(v_arr[t_idx, 0, :, :]).astype("float32")

        # Ensure 2D
        if v.ndim == 3:
            v = v[0, :, :]

        if v.shape == data.shape:
            valid_mask = v > 0.0
            valid_frac = float(np.mean(valid_mask))
            data = np.where(valid_mask, data, np.nan)
        else:
            valid_frac = None

    return data, valid_frac


def _valid_fraction_series(valid_path: str | Path) -> np.ndarray:
    v_arr = zarr.open(str(valid_path), mode="r")
    if v_arr.ndim == 4:
        time_len = v_arr.shape[0]
        band_axis = True
    elif v_arr.ndim == 3:
        time_len = v_arr.shape[0]
        band_axis = False
    else:
        raise ValueError(f"Unexpected valid array shape: {v_arr.shape}")

    # Stream per-timestep to avoid loading full time series into RAM.
    out = np.zeros((time_len,), dtype="float32")
    for t in range(time_len):
        if band_axis:
            v = np.asarray(v_arr[t, 0, :, :], dtype="float32")
        else:
            v = np.asarray(v_arr[t, :, :], dtype="float32")
        out[t] = float(np.mean(v > 0.0))
    return out


def _best_coverage_date(
    products_root: str | Path,
    time_daily: Sequence[str],
    coverage_products: Sequence[str],
) -> Tuple[int, str]:
    scores = None
    for name in coverage_products:
        valid_path = Path(products_root) / name / "valid"
        if not valid_path.exists():
            continue
        frac = _valid_fraction_series(valid_path)
        scores = frac if scores is None else scores + frac
    if scores is None:
        raise RuntimeError("No valid coverage arrays found for coverage products.")
    best_idx = int(np.argmax(scores))
    return best_idx, time_daily[best_idx]


def _topk_coverage_dates(
    products_root: str | Path,
    time_daily: Sequence[str],
    coverage_products: Sequence[str],
    k: int,
) -> List[Tuple[int, str]]:
    scores = None
    for name in coverage_products:
        valid_path = Path(products_root) / name / "valid"
        if not valid_path.exists():
            continue
        frac = _valid_fraction_series(valid_path)
        scores = frac if scores is None else scores + frac
    if scores is None:
        raise RuntimeError("No valid coverage arrays found for coverage products.")
    k = max(1, min(k, len(scores)))
    top_idx = np.argsort(scores)[-k:][::-1]
    return [(int(i), time_daily[int(i)]) for i in top_idx]


def _resize_to(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    try:
        from skimage.transform import resize  # type: ignore

        return resize(arr, shape, order=1, preserve_range=True, anti_aliasing=False).astype(arr.dtype)
    except Exception:
        try:
            from scipy.ndimage import zoom  # type: ignore

            zoom_factors = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
            return zoom(arr, zoom=zoom_factors, order=1)
        except Exception as exc:
            raise RuntimeError(
                "Resampling requires scikit-image or scipy. "
                "Install one or provide pre-resampled inputs."
            ) from exc


def _parse_boxes(boxes: Optional[str]) -> List[Tuple[int, int, int, int]]:
    if boxes is None:
        return []
    if Path(boxes).exists():
        data = json.loads(Path(boxes).read_text())
    else:
        data = json.loads(boxes)
    return [tuple(map(int, b)) for b in data]


def _crop(arr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = box
    return arr[y0:y1, x0:x1]


def _downsample_maxdim(arr: np.ndarray, max_dim: Optional[int]) -> np.ndarray:
    if max_dim is None:
        return arr
    if arr.ndim == 3:
        # If accidental extra channel dimension survives, take first.
        arr = arr[0, :, :]
    if arr.ndim != 2:
        return arr
    h, w = arr.shape
    m = max(h, w)
    if m <= max_dim:
        return arr
    stride = int(np.ceil(m / max_dim))
    return arr[::stride, ::stride]


def _percentile_range(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> Tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


def plot_grid(
    panels: Sequence[Tuple[str, np.ndarray, str, Optional[float]]],
    *,
    crops: Sequence[Tuple[int, int, int, int]],
    highlight_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    per_panel_scale: bool = True,
    max_dim: Optional[int] = None,
    out_path: str | Path = "figures/multimodel_grid.png",
) -> None:
    if highlight_boxes is None:
        highlight_boxes = []

    nrows = max(1, len(crops))
    ncols = len(panels)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(2.6 * ncols, 2.6 * nrows),
        constrained_layout=False,
    )
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r in range(nrows):
        box = crops[r] if crops else (0, panels[0][1].shape[0], 0, panels[0][1].shape[1])
        for c, (label, arr_full, kind, _) in enumerate(panels):
            ax = axes[r, c]
            ax.set_facecolor("white")
            arr = _crop(arr_full, box)
            arr = _downsample_maxdim(arr, max_dim)
            if kind == "landcover":
                cmap = ListedColormap(DW_COLORS)
                cmap.set_bad("white")
                arr_masked = np.ma.masked_invalid(arr)
                ax.imshow(arr_masked, interpolation="nearest", cmap=cmap,
                          vmin=-0.5, vmax=len(DW_COLORS) - 0.5)
            else:
                if per_panel_scale or vmin is None or vmax is None:
                    pvmin, pvmax = _percentile_range(arr)
                else:
                    pvmin, pvmax = float(vmin), float(vmax)
                cmap = plt.get_cmap("turbo").copy()
                cmap.set_bad("white")
                ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=pvmin, vmax=pvmax)
            ax.set_xticks([])
            ax.set_yticks([])

            if highlight_boxes and r < len(highlight_boxes) and c == ncols - 1:
                y0, y1, x0, x1 = highlight_boxes[r]
                rect = Rectangle((x0 - box[2], y0 - box[0]), x1 - x0, y1 - y0,
                                 linewidth=1.2, edgecolor="black", facecolor="none")
                ax.add_patch(rect)

    for c, (label, _, _, vfrac) in enumerate(panels):
        if vfrac is not None:
            title = f"{label} ({vfrac*100:.0f}%)"
        else:
            title = label
        axes[0, c].set_title(title, fontsize=10)

    # Colorbar for continuous panels (use first non-landcover panel)
    mappable = None
    for c, (_, _, kind, _) in enumerate(panels):
        if kind != "landcover":
            mappable = axes[0, c].images[0]
            break
    if mappable is not None:
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label("(K)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-panel single-day product grid.")
    parser.add_argument("--date", required=False, help="Date string in time index, e.g. 2023-03-25")
    parser.add_argument("--date-index", type=int, default=None,
                        help="Optional explicit index into the time array (overrides --date)")
    parser.add_argument("--products-root", default="madurai.zarr/products")
    parser.add_argument("--products", default="all",
                        help="Comma-separated product list or 'all' (auto-discover data arrays)")
    parser.add_argument("--band-map", default=None,
                        help="JSON dict for band selection per product, e.g. '{\"landsat\":0,\"modis\":0}'")
    parser.add_argument("--landcover-zarr", default="madurai_30m.zarr/static_30m/dynamic_world/data")
    parser.add_argument("--no-landcover", action="store_true",
                        help="Skip landcover panel.")
    parser.add_argument("--crops", default=None, help="JSON list or file of boxes: [[y0,y1,x0,x1], ...]")
    parser.add_argument("--highlight", default=None, help="JSON list or file of boxes for rectangles (same format)")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--per-panel-scale", action="store_true",
                        help="Auto-scale each panel by percentiles (default).")
    parser.add_argument("--global-scale", action="store_true",
                        help="Use global vmin/vmax across all panels (set --vmin/--vmax).")
    parser.add_argument("--out", default=None,
                        help="Optional explicit output path (otherwise auto-named).")
    parser.add_argument("--resample-to-landcover", action="store_true",
                        help="Resample all continuous products to landcover grid")
    parser.add_argument("--max-dim", type=int, default=None,
                        help="Downsample panels for display to this max dimension (e.g. 512).")
    parser.add_argument("--auto-date", action="store_true",
                        help="Select date with maximum coverage for MODIS+VIIRS.")
    parser.add_argument("--coverage-products", default="modis,viirs,landsat",
                        help="Comma-separated products for auto-date coverage scoring.")
    parser.add_argument("--topk", type=int, default=1,
                        help="Number of best-coverage dates to render when --auto-date is set.")
    parser.add_argument("--out-dir", default=None,
                        help="Directory for multiple outputs (used with --topk > 1).")
    parser.add_argument("--min-valid-frac", type=float, default=None,
                        help="Skip products with valid fraction below this threshold (0-1).")
    parser.add_argument("--resample-products", default="sentinel1,sentinel2",
                        help="Comma-separated product names to resample to landcover grid.")
    args = parser.parse_args()

    if args.date is None and args.date_index is None and not args.auto_date:
        raise ValueError("Provide either --date/--date-index or use --auto-date")

    band_map: Dict[str, int] = {}
    if args.band_map:
        band_map = json.loads(args.band_map)

    products = []
    if args.products.strip().lower() == "all":
        products = _list_products(args.products_root)
    else:
        products = [p.strip() for p in args.products.split(",") if p.strip()]

    time_daily = _load_time_optional("madurai.zarr/time/daily")
    time_monthly = _load_time_optional("madurai.zarr/time/monthly")

    if args.auto_date:
        if not time_daily:
            raise RuntimeError("Daily time index not found for auto-date.")
        cov_products = [p.strip() for p in args.coverage_products.split(",") if p.strip()]
        topk = max(1, int(args.topk))
        picks = _topk_coverage_dates(args.products_root, time_daily, cov_products, topk)
        if topk == 1:
            best_idx, best_date = picks[0]
            print(f"[info] auto-date picked {best_date} (index {best_idx}) using {cov_products}")
            args.date_index = best_idx
            args.date = best_date
        else:
            out_dir = Path(args.out_dir or "figures/best_days")
            out_dir.mkdir(parents=True, exist_ok=True)
            for rank, (idx, date_str) in enumerate(picks, start=1):
                print(f"[info] auto-date picked {date_str} (index {idx}) rank={rank}")
                args.date_index = idx
                args.date = date_str

                panels: List[Tuple[str, np.ndarray, str, Optional[float]]] = []
                landcover = None
                if not args.no_landcover:
                    landcover = _load_landcover(args.landcover_zarr)
                    panels.append(("Land cover", landcover, "landcover", None))

                for name in products:
                    try:
                        data, vfrac = _load_product_array(
                            name,
                            args.products_root,
                            args.date or "",
                            date_index=args.date_index,
                            band_map=band_map,
                            time_daily=time_daily,
                            time_monthly=time_monthly,
                        )
                    except Exception as exc:
                        print(f"[warn] Skipping {name}: {exc}")
                        continue
                    if args.min_valid_frac is not None and vfrac is not None:
                        if vfrac < args.min_valid_frac:
                            print(f"[info] Skipping {name}: valid {vfrac:.3f} < {args.min_valid_frac}")
                            continue
                    panels.append((name.upper(), data, "continuous", vfrac))

                if landcover is not None:
                    target_shape = landcover.shape
                    resampled = []
                    resample_names = {s.strip().upper() for s in args.resample_products.split(",") if s.strip()}
                    for label, arr, kind, vfrac in panels:
                        if kind == "landcover":
                            resampled.append((label, arr, kind, vfrac))
                            continue
                        if args.resample_to_landcover or label.upper() in resample_names:
                            resampled.append((label, _resize_to(arr, target_shape), kind, vfrac))
                        else:
                            resampled.append((label, arr, kind, vfrac))
                    panels = resampled

                crops = _parse_boxes(args.crops)
                highlight = _parse_boxes(args.highlight)
                out_path = out_dir / f"all_products_{date_str}.png"

                plot_grid(
                    panels=panels,
                    crops=crops,
                    highlight_boxes=highlight,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    per_panel_scale=not args.global_scale,
                    max_dim=args.max_dim,
                    out_path=out_path,
                )
            return

    panels: List[Tuple[str, np.ndarray, str, Optional[float]]] = []
    landcover = None
    if not args.no_landcover:
        landcover = _load_landcover(args.landcover_zarr)
        panels.append(("Land cover", landcover, "landcover", None))

    for name in products:
        try:
            data, vfrac = _load_product_array(
                name,
                args.products_root,
                args.date or "",
                date_index=args.date_index,
                band_map=band_map,
                time_daily=time_daily,
                time_monthly=time_monthly,
            )
        except Exception as exc:
            print(f"[warn] Skipping {name}: {exc}")
            continue
        if args.min_valid_frac is not None and vfrac is not None:
            if vfrac < args.min_valid_frac:
                print(f"[info] Skipping {name}: valid {vfrac:.3f} < {args.min_valid_frac}")
                continue
        panels.append((name.upper(), data, "continuous", vfrac))

    if not panels:
        raise RuntimeError("No panels to plot. Check product paths or names.")

    if landcover is not None:
        target_shape = landcover.shape
        resampled = []
        resample_names = {s.strip().upper() for s in args.resample_products.split(",") if s.strip()}
        for label, arr, kind, vfrac in panels:
            if kind == "landcover":
                resampled.append((label, arr, kind, vfrac))
                continue
            if args.resample_to_landcover or label.upper() in resample_names:
                resampled.append((label, _resize_to(arr, target_shape), kind, vfrac))
            else:
                resampled.append((label, arr, kind, vfrac))
        panels = resampled

    crops = _parse_boxes(args.crops)
    highlight = _parse_boxes(args.highlight)

    # Auto-name output if not provided
    if args.out is None:
        date_tag = (args.date or "unknown").replace("/", "_")
        out_path = Path("figures") / f"all_products_{date_tag}.png"
    else:
        out_path = Path(args.out)

    plot_grid(
        panels=panels,
        crops=crops,
        highlight_boxes=highlight,
        vmin=args.vmin,
        vmax=args.vmax,
        per_panel_scale=not args.global_scale,
        max_dim=args.max_dim,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
