from __future__ import annotations

from typing import Any, Callable, Iterable, Set, Tuple

import numpy as np


def build_hr_to_lr_index(h_hr: int, w_hr: int, h_lr: int, w_lr: int) -> np.ndarray:
    row_map = np.floor(np.arange(h_hr, dtype=np.float64) * h_lr / h_hr).astype(np.int32)
    col_map = np.floor(np.arange(w_hr, dtype=np.float64) * w_lr / w_hr).astype(np.int32)
    row_map = np.clip(row_map, 0, h_lr - 1)
    col_map = np.clip(col_map, 0, w_lr - 1)
    return row_map[:, None] * w_lr + col_map[None, :]


def aggregate_mean_by_index(
    arr2d: np.ndarray,
    cell_index: np.ndarray,
    n_cells: int,
    *,
    valid_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(arr2d, dtype=np.float32)
    finite = np.isfinite(v)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)

    idx = cell_index.ravel()
    vals = np.where(finite, v, 0.0).ravel()
    cnt = np.bincount(idx, weights=finite.ravel().astype(np.float32), minlength=n_cells).astype(np.float32)
    s = np.bincount(idx, weights=vals.astype(np.float32), minlength=n_cells).astype(np.float32)
    out = np.full((n_cells,), np.nan, dtype=np.float32)
    np.divide(s, cnt, out=out, where=cnt > 0)
    return out, cnt


def bilinear_resample_2d(src: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src = np.asarray(src, dtype=np.float32)
    in_h, in_w = src.shape
    row_f = np.linspace(0, max(0, in_h - 1), out_h, dtype=np.float64)
    col_f = np.linspace(0, max(0, in_w - 1), out_w, dtype=np.float64)
    r0 = np.floor(row_f).astype(np.int32)
    c0 = np.floor(col_f).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, in_h - 1)
    c1 = np.clip(c0 + 1, 0, in_w - 1)
    fr = (row_f - r0)[:, None]
    fc = (col_f - c0)[None, :]
    v00 = src[r0[:, None], c0[None, :]]
    v01 = src[r0[:, None], c1[None, :]]
    v10 = src[r1[:, None], c0[None, :]]
    v11 = src[r1[:, None], c1[None, :]]
    w00 = (1.0 - fr) * (1.0 - fc)
    w01 = (1.0 - fr) * fc
    w10 = fr * (1.0 - fc)
    w11 = fr * fc
    vals = np.stack([v00, v01, v10, v11], axis=0)
    wts = np.stack([w00, w01, w10, w11], axis=0)
    finite = np.isfinite(vals)
    wts = np.where(finite, wts, 0.0)
    num = np.sum(np.where(finite, vals, 0.0) * wts, axis=0)
    den = np.sum(wts, axis=0)
    out = np.full_like(num, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=den > 0)
    return out.astype(np.float32)


def infer_nodata_values(zarr_array: Any) -> Set[float]:
    vals: Set[float] = set()
    attrs = {}
    try:
        attrs = dict(zarr_array.attrs)
    except Exception:
        attrs = {}
    for key in ("_FillValue", "missing_value", "nodata", "fill_value"):
        if key not in attrs:
            continue
        v = attrs[key]
        if isinstance(v, (list, tuple, np.ndarray)):
            for item in v:
                try:
                    vals.add(float(item))
                except Exception:
                    pass
        else:
            try:
                vals.add(float(v))
            except Exception:
                pass
    return vals


def sanitize_array(
    x: np.ndarray,
    nodata_values: Set[float],
    extra_nodata: Iterable[float] | None,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float32).copy()
    invalid = ~np.isfinite(arr)
    for v in nodata_values:
        invalid |= np.isclose(arr, float(v), equal_nan=False)
    if extra_nodata is not None:
        for v in extra_nodata:
            invalid |= np.isclose(arr, float(v), equal_nan=False)
    invalid |= np.abs(arr) >= 1.0e19
    arr[invalid] = np.nan
    return arr.astype(np.float32), (~invalid).astype(np.uint8)


def _stats(x: np.ndarray) -> tuple[float, float, float, float]:
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan"), float("nan"), float("nan"), float("nan")
    v = x[finite]
    return float(np.nanmin(v)), float(np.nanmedian(v)), float(np.nanmax(v)), float(np.nanmean(v))


def _scale_and_offset_from_attrs(attrs: dict[str, Any]) -> tuple[float, float]:
    sf = attrs.get("scale_factor", attrs.get("scale", 1.0))
    ao = attrs.get("add_offset", attrs.get("offset", 0.0))
    try:
        sf = float(sf)
    except Exception:
        sf = 1.0
    try:
        ao = float(ao)
    except Exception:
        ao = 0.0
    return sf, ao


def ensure_celsius(
    x: np.ndarray,
    name: str,
    kind: str,
    log_fn: Callable[[str], None],
    *,
    attrs: dict[str, Any] | None = None,
    modis_scale_auto: bool = True,
    viirs_scale_auto: bool = True,
    assume_era5_kelvin: str = "auto",
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).copy()
    mn0, md0, mx0, me0 = _stats(arr)
    attrs = attrs or {}
    sf, ao = _scale_and_offset_from_attrs(attrs)
    action = "none"

    if kind in {"landsat_lst", "modis_lst", "viirs_lst", "era5_temp"} and (
        abs(sf - 1.0) > 1e-9 or abs(ao) > 1e-9
    ):
        arr = arr * sf + ao
        action = f"scale_offset({sf},{ao})"

    finite = np.isfinite(arr)
    med = float(np.nanmedian(arr[finite])) if np.any(finite) else float("nan")

    if kind in {"modis_lst", "viirs_lst"}:
        if kind == "modis_lst" and modis_scale_auto and np.isfinite(med) and (med > 1000.0 and med < 20000.0):
            arr = arr * 0.02
            action = f"{action}+auto_scale_0.02" if action != "none" else "auto_scale_0.02"
            finite = np.isfinite(arr)
            med = float(np.nanmedian(arr[finite])) if np.any(finite) else float("nan")
        if kind == "viirs_lst" and viirs_scale_auto and np.isfinite(med) and (med > 1000.0 and med < 20000.0):
            arr = arr * 0.02
            action = f"{action}+auto_scale_0.02" if action != "none" else "auto_scale_0.02"
            finite = np.isfinite(arr)
            med = float(np.nanmedian(arr[finite])) if np.any(finite) else float("nan")

    if kind in {"landsat_lst", "modis_lst", "viirs_lst", "era5_temp"} and np.isfinite(med):
        convert_to_c = False
        if kind == "era5_temp":
            era5_flag = str(assume_era5_kelvin).lower()
            if era5_flag == "yes":
                convert_to_c = True
            elif era5_flag == "auto":
                convert_to_c = med > 150.0 and med < 400.0
        else:
            convert_to_c = med > 150.0 and med < 400.0
        if convert_to_c:
            arr = arr - 273.15
            action = f"{action}+K_to_C" if action != "none" else "K_to_C"

    mn1, md1, mx1, me1 = _stats(arr)
    log_fn(
        f"[unit] {name} kind={kind} action={action} "
        f"before(min={mn0:.3f},med={md0:.3f},max={mx0:.3f},mean={me0:.3f}) "
        f"after(min={mn1:.3f},med={md1:.3f},max={mx1:.3f},mean={me1:.3f})"
    )
    return arr.astype(np.float32)

