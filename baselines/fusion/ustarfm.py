"""
uSTARFM-like baseline adapted to current Madurai data access patterns.

Uses:
- HR fine (Landsat LST) from madurai_30m.zarr labels_30m/landsat/data (band 0)
- HR features for class map from static layers
- LR daily anchors from madurai.zarr products/{modis|viirs}/data

Output:
- per-date predictions (.npy)
- eval metrics CSV
- logs in Logs/
"""

from pathlib import Path
from datetime import datetime
import argparse
import time
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import zarr
from sklearn.cluster import MiniBatchKMeans

from helper.split_utils import load_or_create_splits
from helper.eval_utils import build_roi_mask, save_roi_figure, compute_metrics

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"
COMMON_DATES = PROJECT_ROOT / "common_dates.csv"
SPLITS_PATH = PROJECT_ROOT / "metrics" / "common_date_splits.csv"
LOG_DIR = PROJECT_ROOT / "logs" / "new"

# Defaults (edit here if needed)
LOWRES_SOURCE = "modis"
BASE_DATE = None
MAX_TARGETS = -1
N_CLASSES = 20
SAMPLE_PIXELS = 250_000
SEED = 42
RIDGE_LAMBDA = 1e-3
PRIOR_LAMBDA = 1.0


LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_f = None


def log(msg: str) -> None:
    print(msg)
    if _log_f is not None:
        _log_f.write(str(msg) + "\n")


def _to_str(arr):
    return np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
    )


def _extract_landsat(y):
    y = y.astype(np.float32)
    y = np.where(y == 149, np.nan, y)
    if np.isfinite(y).any() and np.nanmedian(y) > 200:
        y = y - 273.15
    return y


def _extract_modis(modis_lr):
    if modis_lr.shape[0] >= 6:
        lst = modis_lr[0].astype(np.float32)
        qc = modis_lr[4].astype(np.float32)
    elif modis_lr.shape[0] >= 2:
        lst = modis_lr[0].astype(np.float32)
        qc = modis_lr[1].astype(np.float32)
    else:
        lst = modis_lr[0].astype(np.float32)
        qc = np.zeros_like(lst, dtype=np.float32)
    valid_qc = np.isfinite(qc) & (qc == 1)
    valid_lst = np.isfinite(lst) & (lst != -9999.0) & (lst > 0)
    valid = valid_qc & valid_lst
    lst = np.where(valid, lst, np.nan)
    return lst.astype(np.float32)


def _extract_viirs(viirs_lr):
    if viirs_lr.shape[0] >= 4:
        lst = viirs_lr[0].astype(np.float32)
        qc = viirs_lr[2].astype(np.float32)
    elif viirs_lr.shape[0] >= 2:
        lst = viirs_lr[0].astype(np.float32)
        qc = viirs_lr[1].astype(np.float32)
    else:
        lst = viirs_lr[0].astype(np.float32)
        qc = np.zeros_like(lst, dtype=np.float32)
    valid_qc = np.isfinite(qc) & (qc <= 1)
    valid_lst = np.isfinite(lst) & (lst != -9999.0) & (lst >= 273.0)
    valid = valid_qc & valid_lst
    lst = np.where(valid, lst, np.nan)
    lst = lst - 273.15
    return lst.astype(np.float32)


def _has_valid(arr: np.ndarray) -> bool:
    return np.isfinite(arr).any()


def sample_values(name: str, arr: np.ndarray, k: int = 5) -> None:
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        log(f"{name}: all NaN (skipped)")
        return
    vals = finite.ravel()[:k]
    log(f"{name}: {vals.tolist()}")


def save_prediction_figure(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, tag: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    vmin = float(np.nanmin(y_true)) if np.isfinite(y_true).any() else 0.0
    vmax = float(np.nanmax(y_true)) if np.isfinite(y_true).any() else 1.0
    axes[0].imshow(y_true, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("truth")
    axes[1].imshow(y_pred, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("pred")
    err = np.abs(y_pred - y_true)
    axes[2].imshow(err, cmap="magma")
    axes[2].set_title("abs_error")
    for ax in axes:
        ax.axis("off")
    fig.savefig(out_dir / f"ustarfm_pred_{tag}.png", dpi=150)
    plt.close(fig)


def _to_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[0]
    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")


def build_class_map(X: np.ndarray, n_classes: int, sample_pixels: int, seed: int) -> np.ndarray:
    H, W, C = X.shape
    Xf = X.reshape(-1, C)
    finite = np.isfinite(Xf).all(axis=1)
    idxs = np.where(finite)[0]
    if idxs.size == 0:
        raise ValueError("No finite pixels in HR feature stack.")
    rng = np.random.default_rng(seed)
    take = min(sample_pixels, idxs.size)
    sample = rng.choice(idxs, size=take, replace=False)
    kmeans = MiniBatchKMeans(n_clusters=n_classes, random_state=seed, batch_size=4096, n_init="auto")
    kmeans.fit(Xf[sample])
    labels = np.full((H * W,), -1, dtype=np.int32)
    labels[finite] = kmeans.predict(Xf[finite])
    labels[labels < 0] = 0
    return labels.reshape(H, W)


def compute_fraction_matrix(class_map: np.ndarray, lr_shape: tuple[int, int], row_map: np.ndarray, col_map: np.ndarray, n_classes: int) -> np.ndarray:
    H, W = class_map.shape
    h, w = lr_shape
    lr_rows = np.repeat(row_map, W)
    lr_cols = np.tile(col_map, H)
    lr_idx = lr_rows * w + lr_cols
    class_flat = class_map.ravel()
    counts = np.zeros((h * w, n_classes), dtype=np.float32)
    np.add.at(counts, (lr_idx, class_flat), 1.0)
    denom = counts.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    return counts / denom


def class_means_from_fine(F_tb: np.ndarray, class_map: np.ndarray, K: int) -> np.ndarray:
    means = np.zeros((K,), dtype=np.float32)
    for k in range(K):
        m = F_tb[class_map == k]
        means[k] = np.nanmean(m) if np.any(np.isfinite(m)) else np.nan
    g = np.nanmean(F_tb)
    means = np.where(np.isfinite(means), means, g).astype(np.float32)
    return means


def render_hr_from_class_temps(class_map: np.ndarray, class_t: np.ndarray) -> np.ndarray:
    return class_t[class_map].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lowres", choices=["modis", "viirs"], default=LOWRES_SOURCE)
    ap.add_argument("--base_date", type=str, default=None)
    ap.add_argument("--max_targets", type=int, default=MAX_TARGETS)
    ap.add_argument("--splits", type=str, default=str(SPLITS_PATH))
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / "metrics" / "fusion_baselines" / f"ustarfm_{args.lowres}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    global _log_f
    log_path = LOG_DIR / f"ustarfm_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.lowres}.log"
    _log_f = open(log_path, "w", buffering=1)

    root_30m = zarr.open_group(str(ROOT_30M), mode="r")
    root_daily = zarr.open_group(str(ROOT_DAILY), mode="r")

    common_df = pd.read_csv(COMMON_DATES)
    common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna()
    common_dates = pd.DatetimeIndex(common_dates).sort_values()

    daily_raw = _to_str(root_30m["time"]["daily"][:])
    daily_times = pd.to_datetime(daily_raw, format="%Y_%m_%d", errors="coerce").dropna()
    daily_idx = np.flatnonzero(daily_times.isin(common_dates))
    if daily_idx.size == 0:
        raise RuntimeError("No overlapping common dates found in daily time axis.")

    splits = load_or_create_splits(COMMON_DATES, Path(args.splits))
    train_dates = pd.DatetimeIndex(splits["train"]).normalize()
    test_dates = pd.DatetimeIndex(splits["test"]).normalize()
    daily_norm = pd.DatetimeIndex(daily_times).normalize()
    train_idx = [int(t) for t in daily_idx if daily_norm[int(t)] in train_dates]
    test_idx = [int(t) for t in daily_idx if daily_norm[int(t)] in test_dates]
    if not test_idx:
        raise RuntimeError("No test dates matched daily time axis.")

    landsat = root_30m["labels_30m"]["landsat"]["data"]
    lowres = root_daily["products"][args.lowres]["data"]

    base_date = pd.to_datetime(args.base_date) if args.base_date else None
    base_idx = None
    if base_date is not None:
        base_idx_arr = np.flatnonzero(daily_times == base_date)
        if base_idx_arr.size == 0:
            raise RuntimeError(f"Base date {base_date.date()} not found in daily time axis.")
        base_idx = int(base_idx_arr[0])
    else:
        for t in train_idx:
            t = int(t)
            y0 = _extract_landsat(landsat[t, 0])
            if not _has_valid(y0):
                continue
            if args.lowres == "modis":
                c0 = _extract_modis(lowres[t])
            else:
                c0 = _extract_viirs(lowres[t])
            if not _has_valid(c0):
                continue
            base_idx = t
            base_date = pd.Timestamp(daily_times[t])
            break
        if base_idx is None:
            raise RuntimeError("No base date with valid Landsat and low-res data found.")

    F_tb = _extract_landsat(landsat[base_idx, 0])
    if args.lowres == "modis":
        C_tb = _extract_modis(lowres[base_idx])
    else:
        C_tb = _extract_viirs(lowres[base_idx])

    H_hr, W_hr = F_tb.shape
    H_lr, W_lr = C_tb.shape
    row_float = np.linspace(0, H_lr - 1, H_hr, dtype=np.float64)
    col_float = np.linspace(0, W_lr - 1, W_hr, dtype=np.float64)
    row_map = np.clip(np.rint(row_float).astype(np.int64), 0, H_lr - 1)
    col_map = np.clip(np.rint(col_float).astype(np.int64), 0, W_lr - 1)

    log(f"LOWRES_SOURCE={args.lowres} base_date={base_date.date()}")
    log(f"base_valid landsat={int(np.isfinite(F_tb).sum())} lowres={int(np.isfinite(C_tb).sum())}")

    # HR feature stack (static)
    dem = _to_2d(root_30m["static_30m"]["dem"]["data"][0])
    world = _to_2d(root_30m["static_30m"]["worldcover"]["data"][0])
    dyn = _to_2d(root_30m["static_30m"]["dynamic_world"]["data"][0])
    X = np.stack([dem, world, dyn], axis=-1).astype(np.float32)

    log(f"building class map: K={N_CLASSES} samples={SAMPLE_PIXELS}")
    class_map = build_class_map(X, N_CLASSES, SAMPLE_PIXELS, SEED)
    A = compute_fraction_matrix(class_map, (H_lr, W_lr), row_map, col_map, N_CLASSES)
    mu_tb = class_means_from_fine(F_tb, class_map, N_CLASSES)

    AtA = (A.T @ A).astype(np.float32)
    I = np.eye(N_CLASSES, dtype=np.float32)

    target_dates = []
    skipped = 0
    for t in test_idx:
        t = int(t)
        if t == base_idx:
            continue
        y_t = _extract_landsat(landsat[t, 0])
        if not _has_valid(y_t):
            skipped += 1
            continue
        if args.lowres == "modis":
            C_t = _extract_modis(lowres[t])
        else:
            C_t = _extract_viirs(lowres[t])
        if not _has_valid(C_t):
            skipped += 1
            continue
        target_dates.append(t)
    if args.max_targets > 0:
        target_dates = target_dates[: args.max_targets]
    log(f"targets={len(target_dates)} skipped={skipped}")

    roi_mask = build_roi_mask(ROOT_30M, (H_hr, W_hr))
    if roi_mask is not None:
        save_roi_figure(roi_mask, fig_dir / "roi_mask.png")

    eval_rows = []
    figure_date = str(pd.Timestamp(daily_times[int(target_dates[-1])]).date()) if target_dates else None
    for t_idx in target_dates:
        t0 = time.time()
        date_str = str(pd.Timestamp(daily_times[int(t_idx)]).date())
        if args.lowres == "modis":
            C_t = _extract_modis(lowres[int(t_idx)])
        else:
            C_t = _extract_viirs(lowres[int(t_idx)])

        c = C_t.reshape(-1).astype(np.float32)
        valid = np.isfinite(c)
        if valid.sum() < max(10, int(0.05 * c.size)):
            log(f"date={date_str} skipped (low valid LR pixels)")
            continue

        A_v = A[valid]
        c_v = c[valid]
        AtA_v = (A_v.T @ A_v).astype(np.float32)
        rhs = (A_v.T @ c_v).astype(np.float32) + PRIOR_LAMBDA * mu_tb
        x = np.linalg.solve(AtA_v + (RIDGE_LAMBDA + PRIOR_LAMBDA) * I, rhs).astype(np.float32)

        pred_hr = render_hr_from_class_temps(class_map, x)
        y_true = _extract_landsat(landsat[int(t_idx), 0])
        met = compute_metrics(y_true, pred_hr, roi_mask=roi_mask)

        sample_values("pred", pred_hr)
        sample_values("y_true", y_true)

        out_path = out_dir / f"ustarfm_pred_{date_str}.npy"
        np.save(out_path, pred_hr.astype(np.float32))
        log(f"Saved: {out_path} shape={pred_hr.shape} time={time.time() - t0:.2f}s")
        eval_rows.append(
            {
                "time": date_str,
                **{k: met[k] for k in ("rmse", "ssim", "psnr", "sam", "cc")},
                "rmse_sum": met["rmse_sum"],
                "n_valid": met["n_valid"],
            }
        )
        if figure_date and date_str == figure_date:
            save_prediction_figure(y_true, pred_hr, fig_dir, date_str)

    if eval_rows:
        metrics_path = out_dir / "ustarfm_eval_metrics.csv"
        pd.DataFrame(eval_rows).to_csv(metrics_path, index=False)
        log(f"saved eval metrics csv: {metrics_path}")

    np.save(out_dir / "ustarfm_class_map.npy", class_map.astype(np.int32))
    np.save(out_dir / "ustarfm_A_fractions.npy", A.astype(np.float32))
    np.save(out_dir / "ustarfm_mu_tb.npy", mu_tb.astype(np.float32))
    log("Done.")


if __name__ == "__main__":
    main()
