from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _read_common_dates(common_dates_path: Path) -> pd.DatetimeIndex:
    df = pd.read_csv(common_dates_path)
    if "landsat_date" in df.columns:
        col = "landsat_date"
    elif "date" in df.columns:
        col = "date"
    else:
        col = df.columns[0]
    dates = pd.to_datetime(df[col], errors="coerce").dropna()
    return pd.DatetimeIndex(dates).sort_values()


def load_or_create_splits(
    common_dates_path: Path,
    splits_path: Path,
    *,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> Dict[str, pd.DatetimeIndex]:
    splits_path = Path(splits_path)
    if splits_path.exists():
        df = pd.read_csv(splits_path)
        if not {"split", "date"}.issubset(df.columns):
            raise ValueError(f"split file missing columns: {splits_path}")
        out: Dict[str, pd.DatetimeIndex] = {}
        for split in ("train", "val", "test"):
            dates = pd.to_datetime(df.loc[df["split"] == split, "date"], errors="coerce").dropna()
            out[split] = pd.DatetimeIndex(dates).sort_values()
        return out

    dates = _read_common_dates(common_dates_path)
    if len(dates) == 0:
        raise ValueError(f"No dates in common_dates: {common_dates_path}")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(dates))
    rng.shuffle(idx)
    n_train = int(len(dates) * train_frac)
    n_val = int(len(dates) * val_frac)
    n_test = len(dates) - n_train - n_val
    if n_test <= 0:
        raise ValueError("Invalid split fractions; no test dates left.")

    train_dates = dates[idx[:n_train]]
    val_dates = dates[idx[n_train : n_train + n_val]]
    test_dates = dates[idx[n_train + n_val :]]

    rows = []
    for split, split_dates in (("train", train_dates), ("val", val_dates), ("test", test_dates)):
        for d in split_dates:
            rows.append({"split": split, "date": pd.Timestamp(d).strftime("%Y-%m-%d")})
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(splits_path, index=False)
    return {"train": train_dates, "val": val_dates, "test": test_dates}
