from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def build_time_splits(times: pd.DatetimeIndex, cfg: Dict, seed: int) -> SplitResult:
    mode = str(cfg.get("mode", "date_ranges")).lower()
    all_idx = np.arange(len(times), dtype=np.int64)
    valid = ~times.isna()
    all_idx = all_idx[valid]
    tvals = times[valid]

    if mode == "date_ranges":
        def _sel(start: str, end: str) -> np.ndarray:
            s = pd.Timestamp(start)
            e = pd.Timestamp(end)
            m = (tvals >= s) & (tvals <= e)
            return all_idx[m]

        return SplitResult(
            train_idx=_sel(cfg["train_start"], cfg["train_end"]),
            val_idx=_sel(cfg["val_start"], cfg["val_end"]),
            test_idx=_sel(cfg["test_start"], cfg["test_end"]),
        )

    rng = np.random.default_rng(seed)
    idx = np.array(all_idx, copy=True)
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * float(cfg.get("train_frac", 0.7)))
    n_val = int(n * float(cfg.get("val_frac", 0.15)))
    train = np.sort(idx[:n_train])
    val = np.sort(idx[n_train : n_train + n_val])
    test = np.sort(idx[n_train + n_val :])
    return SplitResult(train_idx=train, val_idx=val, test_idx=test)

