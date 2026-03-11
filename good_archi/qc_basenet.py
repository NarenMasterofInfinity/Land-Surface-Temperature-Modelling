from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QCResult:
    modis_valid_day: np.ndarray
    modis_valid_night: np.ndarray
    viirs_valid_day: np.ndarray
    viirs_valid_night: np.ndarray
    modis_qc_score_day: np.ndarray
    modis_qc_score_night: np.ndarray
    viirs_qc_score_day: np.ndarray
    viirs_qc_score_night: np.ndarray


def _score_from_qc(values: np.ndarray, kind: str, unknown_score: float = 0.5) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    out = np.full_like(v, fill_value=float(unknown_score), dtype=np.float32)
    finite = np.isfinite(v)
    if kind == "modis":
        # Existing repo convention: qc==1 is best, then degrade with distance.
        out[finite] = np.clip(1.0 - np.abs(v[finite] - 1.0) / 8.0, 0.0, 1.0)
    elif kind == "viirs":
        # Cloud-like channels in current datasets are usually low-is-good.
        max_v = np.nanmax(v[finite]) if np.any(finite) else 1.0
        scale = max(1.0, float(max_v))
        out[finite] = np.clip(1.0 - (v[finite] / scale), 0.0, 1.0)
    else:
        out[finite] = float(unknown_score)
    return out.astype(np.float32)


def map_qc(
    modis_day: np.ndarray,
    modis_night: np.ndarray,
    viirs_day: np.ndarray,
    viirs_night: np.ndarray,
    modis_qc_day: np.ndarray,
    modis_qc_night: np.ndarray,
    viirs_qc_day: np.ndarray,
    viirs_qc_night: np.ndarray,
    *,
    unknown_qc_score: float = 0.5,
) -> QCResult:
    modis_valid_day = (
        np.isfinite(modis_day) & np.isfinite(modis_qc_day) & (modis_qc_day == 1) & (modis_day > 0)
    ).astype(np.float32)
    modis_valid_night = (
        np.isfinite(modis_night) & np.isfinite(modis_qc_night) & (modis_qc_night == 1) & (modis_night > 0)
    ).astype(np.float32)

    # VIIRS LST is typically in Kelvin in this repo.
    viirs_valid_day = (
        np.isfinite(viirs_day) & np.isfinite(viirs_qc_day) & (viirs_qc_day <= 1) & (viirs_day >= 273.0)
    ).astype(np.float32)
    viirs_valid_night = (
        np.isfinite(viirs_night) & np.isfinite(viirs_qc_night) & (viirs_qc_night <= 1) & (viirs_night >= 273.0)
    ).astype(np.float32)

    # Heuristic fallback for unknown QC encodings.
    if float(np.mean(modis_valid_day)) < 0.01 and float(np.mean(modis_valid_night)) < 0.01:
        modis_valid_day = (np.isfinite(modis_day) & (modis_day > -100.0)).astype(np.float32)
        modis_valid_night = (np.isfinite(modis_night) & (modis_night > -100.0)).astype(np.float32)
    if float(np.mean(viirs_valid_day)) < 0.01 and float(np.mean(viirs_valid_night)) < 0.01:
        viirs_valid_day = (np.isfinite(viirs_day) & (viirs_day > -100.0)).astype(np.float32)
        viirs_valid_night = (np.isfinite(viirs_night) & (viirs_night > -100.0)).astype(np.float32)

    return QCResult(
        modis_valid_day=modis_valid_day,
        modis_valid_night=modis_valid_night,
        viirs_valid_day=viirs_valid_day,
        viirs_valid_night=viirs_valid_night,
        modis_qc_score_day=_score_from_qc(modis_qc_day, "modis", unknown_qc_score),
        modis_qc_score_night=_score_from_qc(modis_qc_night, "modis", unknown_qc_score),
        viirs_qc_score_day=_score_from_qc(viirs_qc_day, "viirs", unknown_qc_score),
        viirs_qc_score_night=_score_from_qc(viirs_qc_night, "viirs", unknown_qc_score),
    )
