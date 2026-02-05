# ======================================================================
# File: /home/naren-root/Documents/FYP2/Project/baselines/linear/metrics_image.py
# ======================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
    from skimage.metrics import peak_signal_noise_ratio as _psnr
except Exception:  # keep import-time robust; script will error with a clear message later if used
    _ssim = None
    _psnr = None


def _finite_mask(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a)


def _masked_flat(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.asarray(a[m], dtype=np.float64)


def rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    m = _finite_mask(y_true) & _finite_mask(y_pred)
    if mask is not None:
        m &= mask
    if not np.any(m):
        return float("nan")
    d = _masked_flat(y_pred, m) - _masked_flat(y_true, m)
    return float(np.sqrt(np.mean(d * d)))


def cc(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    m = _finite_mask(y_true) & _finite_mask(y_pred)
    if mask is not None:
        m &= mask
    if np.sum(m) < 2:
        return float("nan")
    a = _masked_flat(y_true, m)
    b = _masked_flat(y_pred, m)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt(np.mean(a * a)) * np.sqrt(np.mean(b * b)))
    if denom == 0:
        return float("nan")
    return float(np.mean(a * b) / denom)


def psnr(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None, data_range: Optional[float] = None) -> float:
    if _psnr is None:
        raise ImportError("scikit-image not installed. Install with: pip install scikit-image")
    m = _finite_mask(y_true) & _finite_mask(y_pred)
    if mask is not None:
        m &= mask
    if not np.any(m):
        return float("nan")

    # PSNR expects full image; fill invalid pixels with a finite value.
    yt = np.array(y_true, dtype=np.float64, copy=True)
    yp = np.array(y_pred, dtype=np.float64, copy=True)
    fill_val = float(np.nanmedian(yt[m])) if np.any(m) else 0.0
    yt[~m] = fill_val
    yp[~m] = fill_val

    if data_range is None:
        vmin = np.nanmin(yt[m])
        vmax = np.nanmax(yt[m])
        data_range = float(vmax - vmin) if vmax > vmin else 1.0

    return float(_psnr(yt, yp, data_range=data_range))


def ssim(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None, data_range: Optional[float] = None) -> float:
    if _ssim is None:
        raise ImportError("scikit-image not installed. Install with: pip install scikit-image")
    m = _finite_mask(y_true) & _finite_mask(y_pred)
    if mask is not None:
        m &= mask
    if not np.any(m):
        return float("nan")

    yt = np.array(y_true, dtype=np.float64, copy=True)
    yp = np.array(y_pred, dtype=np.float64, copy=True)
    fill_val = float(np.nanmedian(yt[m])) if np.any(m) else 0.0
    yt[~m] = fill_val
    yp[~m] = fill_val

    if data_range is None:
        vmin = np.nanmin(yt[m])
        vmax = np.nanmax(yt[m])
        data_range = float(vmax - vmin) if vmax > vmin else 1.0

    return float(_ssim(yt, yp, data_range=data_range))


def sam(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None, channel_axis: Optional[int] = None) -> float:
    """
    Spectral Angle Mapper.
    - If you have multi-channel targets (e.g., [Day, Night]) put channels on channel_axis.
    - If single-channel, SAM degenerates; we still return a number but interpret cautiously.
    Returns mean angle in radians over valid pixels.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if channel_axis is None:
        # treat as single channel -> add channel dim at end
        yt = yt[..., None]
        yp = yp[..., None]
        channel_axis = -1

    # Move channels to last
    yt = np.moveaxis(yt, channel_axis, -1)
    yp = np.moveaxis(yp, channel_axis, -1)

    # Valid where all channels are finite
    m = np.isfinite(yt).all(axis=-1) & np.isfinite(yp).all(axis=-1)
    if mask is not None:
        m &= mask
    if not np.any(m):
        return float("nan")

    a = yt[m]  # [N, C]
    b = yp[m]
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    den = np.where(den == 0, np.nan, den)
    cosang = np.clip(num / den, -1.0, 1.0)
    ang = np.arccos(cosang)
    return float(np.nanmean(ang))


def ergas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    ratio: float = 33.3333333333,  # coarse_res/fine_res default ~ 1000m/30m
    channel_axis: Optional[int] = None,
) -> float:
    """
    ERGAS = 100/ratio * sqrt( mean_i( RMSE_i^2 / mean_i^2 ) )
    Works for 1+ channels.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if channel_axis is None:
        yt = yt[..., None]
        yp = yp[..., None]
        channel_axis = -1

    yt = np.moveaxis(yt, channel_axis, -1)
    yp = np.moveaxis(yp, channel_axis, -1)

    m = np.isfinite(yt).all(axis=-1) & np.isfinite(yp).all(axis=-1)
    if mask is not None:
        m &= mask
    if not np.any(m):
        return float("nan")

    a = yt[m]  # [N, C]
    b = yp[m]
    # per-channel
    rmses = np.sqrt(np.mean((b - a) ** 2, axis=0))
    means = np.mean(a, axis=0)
    means = np.where(means == 0, np.nan, means)
    val = (rmses ** 2) / (means ** 2)
    return float((100.0 / ratio) * np.sqrt(np.nanmean(val)))


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    ratio: float = 33.3333333333,
    channel_axis: Optional[int] = None,
) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred, mask),
        "ssim": ssim(y_true, y_pred, mask),
        "psnr": psnr(y_true, y_pred, mask),
        "sam": sam(y_true, y_pred, mask, channel_axis=channel_axis),
        "cc": cc(y_true, y_pred, mask),
        "ergas": ergas(y_true, y_pred, mask, ratio=ratio, channel_axis=channel_axis),
    }
