from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

from basenet_features_from_zarr import build_1km_feature_table_for_date
from basenet_runtime import load_basenet_from_ckpt, predict_basenet_30m
from residual_net_30m import ResidualLSTNet30m


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent


def _to_str(arr: np.ndarray) -> np.ndarray:
    return np.array([x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr])


def _parse_time_raw(raw: np.ndarray) -> pd.DatetimeIndex:
    arr = np.asarray(raw)
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.to_datetime(arr, errors="coerce")
    if np.issubdtype(arr.dtype, np.number):
        v = arr.astype(np.float64)
        finite = v[np.isfinite(v)]
        if finite.size > 0:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
            if vmin > 1e7 and vmax < 3e9 and np.all(np.floor(finite) == finite):
                return pd.to_datetime(finite.astype(np.int64).astype(str), format="%Y%m%d", errors="coerce")
            if vmax > 1e12:
                return pd.to_datetime(v, unit="ms", errors="coerce")
            if vmax > 1e9:
                return pd.to_datetime(v, unit="s", errors="coerce")
    s = _to_str(arr)
    if s.size > 0:
        first = str(s.flat[0])
        if len(first) == 10 and first[4] == "_" and first[7] == "_":
            return pd.to_datetime(s, format="%Y_%m_%d", errors="coerce")
        if len(first) == 10 and first[4] == "-" and first[7] == "-":
            return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        if len(first) == 8 and first.isdigit():
            return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _read_daily_time(root_daily: zarr.Group) -> pd.DatetimeIndex:
    if "time" in root_daily and "daily" in root_daily["time"]:
        return _parse_time_raw(root_daily["time"]["daily"][:])
    raise RuntimeError("Could not find daily time array in daily zarr.")


def _read_monthly_time(root_30m: zarr.Group) -> pd.DatetimeIndex:
    if "time" in root_30m and "monthly" in root_30m["time"]:
        return _parse_time_raw(root_30m["time"]["monthly"][:])
    return pd.to_datetime([], errors="coerce")


def _landsat_to_celsius(arr: np.ndarray, attrs: Dict) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    x = np.where(x == 149, np.nan, x)
    scale = float(attrs.get("scale_factor", attrs.get("scale", 1.0)) or 1.0)
    offset = float(attrs.get("add_offset", attrs.get("offset", 0.0)) or 0.0)
    if scale != 1.0 or offset != 0.0:
        x = (x * scale) + offset
    if np.isfinite(x).any() and np.nanmedian(x) > 200:
        x = x - 273.15
    return x


def _valid_temp_mask(x: np.ndarray, min_c: float, max_c: float) -> np.ndarray:
    return np.isfinite(x) & (x >= min_c) & (x <= max_c)


def _normalize_date_text(text: str) -> Optional[str]:
    s = str(text).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y_%m_%d", "%Y%m%d"):
        try:
            return str(pd.Timestamp(pd.to_datetime(s, format=fmt, errors="raise")).date())
        except Exception:
            pass
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return None
    return str(pd.Timestamp(ts).date())


def _date_to_basenet_file_map(base_pred_dir: Path) -> Dict[str, Path]:
    date_rx = re.compile(r"(\d{4}[-_]\d{2}[-_]\d{2}|\d{8})")
    out: Dict[str, Path] = {}
    if not base_pred_dir.exists():
        return out
    for p in sorted(base_pred_dir.glob("*.npy")):
        m = date_rx.search(p.name)
        if not m:
            continue
        k = _normalize_date_text(m.group(1))
        if k is not None:
            out[k] = p
    return out


def _masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    sel = mask > 0.5
    if not torch.any(sel):
        return torch.tensor(float("nan"), device=pred.device)
    p = pred[sel]
    t = target[sel]
    e = torch.abs(p - t)
    q = torch.minimum(e, torch.tensor(delta, device=pred.device))
    l = e - q
    return torch.mean(0.5 * q * q + delta * l)


def _masked_huber_sum_count(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    sel = mask > 0.5
    n = torch.sum(sel)
    if int(n.item()) == 0:
        z = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return z, torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    p = pred[sel]
    t = target[sel]
    e = torch.abs(p - t)
    q = torch.minimum(e, torch.tensor(delta, device=pred.device, dtype=pred.dtype))
    l = e - q
    s = torch.sum(0.5 * q * q + delta * l)
    return s, n.to(dtype=pred.dtype)


def _tile_slices(h: int, w: int, tile: int, stride: int):
    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if not ys or ys[-1] != max(0, h - tile):
        ys.append(max(0, h - tile))
    if not xs or xs[-1] != max(0, w - tile):
        xs.append(max(0, w - tile))
    for y0 in ys:
        for x0 in xs:
            y1 = min(h, y0 + tile)
            x1 = min(w, x0 + tile)
            yield slice(y0, y1), slice(x0, x1)


class Residual30mDataset(Dataset):
    def __init__(
        self,
        *,
        t_indices: List[int],
        root_30m: zarr.Group,
        daily_times: pd.DatetimeIndex,
        lst_min: float,
        lst_max: float,
        era5_idx: List[int],
    ) -> None:
        self.t_indices = list(t_indices)
        self.root_30m = root_30m
        self.daily_times = daily_times
        self.lst_min = float(lst_min)
        self.lst_max = float(lst_max)
        self.era5_idx = list(era5_idx)

        self.s2 = root_30m["products_30m"]["sentinel2"]["data"]
        self.s1 = root_30m["products_30m"]["sentinel1"]["data"] if "sentinel1" in root_30m["products_30m"] else None
        self.era5 = root_30m["products_30m"]["era5"]["data"]
        self.dem = np.asarray(root_30m["static_30m"]["dem"]["data"][0, 0], dtype=np.float32)
        self.world = np.asarray(root_30m["static_30m"]["worldcover"]["data"][0, 0], dtype=np.float32)
        self.dynamic = np.asarray(root_30m["static_30m"]["dynamic_world"]["data"][0, 0], dtype=np.float32)
        self.landsat = root_30m["labels_30m"]["landsat"]["data"]
        self.landsat_attrs = dict(root_30m["labels_30m"]["landsat"].attrs)

        monthly_times = _read_monthly_time(root_30m).to_period("M").to_timestamp()
        self.month_map = {pd.Timestamp(t): i for i, t in enumerate(monthly_times) if pd.notna(t)}

    def __len__(self) -> int:
        return len(self.t_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = int(self.t_indices[idx])
        date = pd.Timestamp(self.daily_times[t]).to_period("D").to_timestamp()
        date_str = str(date.date())
        m_key = date.to_period("M").to_timestamp()
        m_idx = self.month_map.get(m_key, -1)

        y = _landsat_to_celsius(np.asarray(self.landsat[t, 0]), self.landsat_attrs)
        m = _valid_temp_mask(y, self.lst_min, self.lst_max).astype(np.float32)
        y = np.where(m > 0.5, y, 0.0).astype(np.float32)

        if m_idx < 0:
            s2 = np.full((self.s2.shape[1], y.shape[0], y.shape[1]), np.nan, dtype=np.float32)
            s1 = np.full((self.s1.shape[1], y.shape[0], y.shape[1]), np.nan, dtype=np.float32) if self.s1 is not None else None
        else:
            s2 = np.asarray(self.s2[m_idx], dtype=np.float32)
            s1 = np.asarray(self.s1[m_idx], dtype=np.float32) if self.s1 is not None else None

        era5 = np.asarray(self.era5[t], dtype=np.float32)[self.era5_idx]
        dem = np.where(np.isfinite(self.dem), self.dem, 0.0)[None, ...].astype(np.float32)
        lc = np.stack(
            [
                np.where(np.isfinite(self.world), self.world, 0.0),
                np.where(np.isfinite(self.dynamic), self.dynamic, 0.0),
            ],
            axis=0,
        ).astype(np.float32)

        return {
            "date": date_str,
            "date_idx": torch.tensor(t, dtype=torch.int64),
            "s2": torch.from_numpy(s2),
            "s1": torch.from_numpy(s1) if s1 is not None else torch.empty((0, y.shape[0], y.shape[1])),
            "dem": torch.from_numpy(dem),
            "lc": torch.from_numpy(lc),
            "era5": torch.from_numpy(era5),
            "y": torch.from_numpy(y[None, ...]),
            "mask": torch.from_numpy(m[None, ...]),
        }


def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in ["date_idx", "s2", "s1", "dem", "lc", "era5", "y", "mask"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["date"] = [b["date"] for b in batch]
    return out


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _get_base_map_for_batch(
    *,
    batch_dates: List[str],
    batch_date_idx: np.ndarray,
    y_hw: Tuple[int, int],
    root_daily: zarr.Group,
    root_30m: zarr.Group,
    daily_times: pd.DatetimeIndex,
    basenet_runtime,
    strict_feature_match: bool,
    use_precomputed: Dict[str, Path],
    cache_preds: bool,
    cache_store: Dict[str, np.ndarray],
    show_tqdm: bool,
    tqdm_desc: str,
) -> np.ndarray:
    out = np.zeros((len(batch_dates), y_hw[0], y_hw[1]), dtype=np.float32)
    date_iter = enumerate(batch_dates)
    if show_tqdm and len(batch_dates) > 1:
        date_iter = enumerate(
            tqdm(batch_dates, desc=tqdm_desc, leave=False, dynamic_ncols=True)
        )
    for i, date_str in date_iter:
        if date_str in cache_store:
            out[i] = cache_store[date_str]
            continue
        if date_str in use_precomputed:
            arr = np.load(use_precomputed[date_str]).astype(np.float32)
            if arr.shape != y_hw:
                # Nearest is used only for legacy precomputed compatibility path.
                yy = np.linspace(0, arr.shape[0] - 1, y_hw[0]).astype(np.int64)
                xx = np.linspace(0, arr.shape[1] - 1, y_hw[1]).astype(np.int64)
                arr = arr[yy][:, xx]
            arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
            out[i] = arr
            if cache_preds:
                cache_store[date_str] = arr
            continue

        table = build_1km_feature_table_for_date(
            date_idx=int(batch_date_idx[i]),
            root_daily=root_daily,
            root_30m=root_30m,
            daily_times=daily_times,
            ckpt=basenet_runtime.ckpt,
        )
        arr = predict_basenet_30m(
            runtime=basenet_runtime,
            table=table,
            out_hw=y_hw,
            strict_feature_match=strict_feature_match,
        )
        arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
        out[i] = arr
        if cache_preds:
            cache_store[date_str] = arr
    return out


def _run_epoch(
    *,
    model: ResidualLSTNet30m,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    use_amp: bool,
    residual_l2: float,
    root_daily: zarr.Group,
    root_30m: zarr.Group,
    daily_times: pd.DatetimeIndex,
    basenet_runtime,
    strict_feature_match: bool,
    precomputed_map: Dict[str, Path],
    cache_basenet_pred: bool,
    basenet_cache: Dict[str, np.ndarray],
    tile_size: int,
    tile_stride: int,
    phase: str,
    epoch_idx: int,
    total_epochs: int,
    show_tqdm: bool,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses: List[float] = []
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []

    iter_loader = loader
    if show_tqdm:
        iter_loader = tqdm(
            loader,
            desc=f"{phase} {epoch_idx}/{total_epochs}",
            leave=False,
            dynamic_ncols=True,
        )
    running_pred: List[np.ndarray] = []
    running_true: List[np.ndarray] = []
    for bidx, batch in enumerate(iter_loader):
        s2 = batch["s2"].to(device).float()
        s1 = batch["s1"].to(device).float()
        dem = batch["dem"].to(device).float()
        lc = batch["lc"].to(device).float()
        era5 = batch["era5"].to(device).float()
        y = batch["y"].to(device).float()
        m = batch["mask"].to(device).float()

        y_hw = (int(y.shape[-2]), int(y.shape[-1]))
        base_np = _get_base_map_for_batch(
            batch_dates=batch["date"],
            batch_date_idx=batch["date_idx"].cpu().numpy(),
            y_hw=y_hw,
            root_daily=root_daily,
            root_30m=root_30m,
            daily_times=daily_times,
            basenet_runtime=basenet_runtime,
            strict_feature_match=strict_feature_match,
            use_precomputed=precomputed_map,
            cache_preds=cache_basenet_pred,
            cache_store=basenet_cache,
            show_tqdm=show_tqdm,
            tqdm_desc=f"BaseNet {phase} {epoch_idx}/{total_epochs} b{bidx}",
        )
        base = torch.from_numpy(base_np[:, None, :, :]).to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        batch_loss_num = 0.0
        batch_loss_den = 0.0
        did_backward = False
        for bi in range(int(s2.shape[0])):
            s2b = s2[bi : bi + 1]
            s1b = s1[bi : bi + 1] if s1.shape[1] > 0 else None
            demb = dem[bi : bi + 1]
            lcb = lc[bi : bi + 1]
            era5b = era5[bi : bi + 1]
            yb = y[bi : bi + 1]
            mb = m[bi : bi + 1]
            baseb = base[bi : bi + 1]
            h, w = int(yb.shape[-2]), int(yb.shape[-1])
            den = float(torch.sum(mb > 0.5).item())
            if den <= 0:
                continue

            for ys, xs in _tile_slices(h, w, tile_size, tile_stride):
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(
                        basenet_hr=baseb[:, :, ys, xs],
                        s2=s2b[:, :, ys, xs],
                        s1=s1b[:, :, ys, xs] if s1b is not None else None,
                        dem=demb[:, :, ys, xs],
                        lc=lcb[:, :, ys, xs],
                        era5=era5b[:, :, ys, xs],
                    )
                    lsum, lcount = _masked_huber_sum_count(out["yhat"], yb[:, :, ys, xs], mb[:, :, ys, xs], delta=1.0)
                    if residual_l2 > 0.0 and torch.isfinite(lsum):
                        lsum = lsum + (residual_l2 * torch.mean(out["residual"] * out["residual"]) * lcount)
                    ltile = lsum / max(den, 1.0)

                if not torch.isfinite(ltile):
                    continue

                if train_mode:
                    scaler.scale(ltile).backward()
                    did_backward = True

                batch_loss_num += float(lsum.detach().item())
                batch_loss_den += float(lcount.detach().item())

                yp = out["yhat"].detach().cpu().numpy()
                yt = yb[:, :, ys, xs].detach().cpu().numpy()
                mm = mb[:, :, ys, xs].detach().cpu().numpy() > 0.5
                finite = np.isfinite(yp) & np.isfinite(yt)
                keep = mm & finite
                if np.any(keep):
                    kept_pred = yp[keep]
                    kept_true = yt[keep]
                    y_pred.append(kept_pred)
                    y_true.append(kept_true)
                    running_pred.append(kept_pred)
                    running_true.append(kept_true)

        if train_mode and did_backward:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        if batch_loss_den > 0:
            losses.append(float(batch_loss_num / batch_loss_den))
        if show_tqdm and hasattr(iter_loader, "set_postfix"):
            lp = float(np.mean(losses)) if losses else float("nan")
            if running_pred and running_true:
                rp = np.concatenate(running_pred)
                rt = np.concatenate(running_true)
                rr = _rmse(rp, rt)
                rn = int(rt.size)
            else:
                rr = float("nan")
                rn = 0
            iter_loader.set_postfix({"loss": f"{lp:.4f}", "rmse": f"{rr:.4f}", "n": rn}, refresh=False)

    yp_all = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.float32)
    yt_all = np.concatenate(y_true) if y_true else np.array([], dtype=np.float32)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "rmse": _rmse(yp_all, yt_all),
        "n_eval": int(yt_all.size),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train 30m residual LST model with on-the-fly BaseNet inference from zarr.")
    ap.add_argument("--root_30m", type=str, default=str((_REPO_ROOT / "madurai_30m.zarr").resolve()))
    ap.add_argument("--root_daily", type=str, default=str((_REPO_ROOT / "madurai.zarr").resolve()))
    ap.add_argument("--common_dates_csv", type=str, default=str((_REPO_ROOT / "common_dates.csv").resolve()))
    ap.add_argument("--out_dir", type=str, default=str((_THIS_DIR / "residual_30m_run").resolve()))

    ap.add_argument("--basenet_ckpt", type=str, default=str((_THIS_DIR / "checkpoints" / "best.pt").resolve()))
    ap.add_argument("--basenet_device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--basenet_batch_cells", type=int, default=65536)
    ap.add_argument("--cache_basenet_pred", action="store_true")
    ap.add_argument("--strict_basenet_feature_match", action="store_true", default=True)
    ap.add_argument("--no_strict_basenet_feature_match", action="store_true")

    ap.add_argument("--base_pred_dir", type=str, default="", help="Optional legacy precomputed BaseNet prediction dir.")
    ap.add_argument("--era5_idx", type=str, default="0,1,3,4")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--tile_size", type=int, default=256)
    ap.add_argument("--tile_stride", type=int, default=256)
    ap.add_argument("--widths", type=str, default="24,32,48,64")
    ap.add_argument("--no_tqdm", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--residual_l2", type=float, default=1e-4)
    ap.add_argument("--lst_min_c", type=float, default=10.0)
    ap.add_argument("--lst_max_c", type=float, default=70.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    strict_feature_match = bool(args.strict_basenet_feature_match) and not bool(args.no_strict_basenet_feature_match)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    root_30m = zarr.open_group(args.root_30m, mode="r")
    root_daily = zarr.open_group(args.root_daily, mode="r")
    daily_times = _read_daily_time(root_daily).to_period("D").to_timestamp()
    daily_map = {str(pd.Timestamp(t).date()): i for i, t in enumerate(daily_times) if pd.notna(t)}

    common = pd.read_csv(args.common_dates_csv)
    if "date" in common.columns:
        d = pd.to_datetime(common["date"], errors="coerce")
    elif "landsat_date" in common.columns:
        d = pd.to_datetime(common["landsat_date"], errors="coerce")
    else:
        d = pd.to_datetime(common.iloc[:, 0], errors="coerce")
    common_dates = sorted({str(pd.Timestamp(x).date()) for x in d.dropna()})

    legacy_pred = _date_to_basenet_file_map(Path(args.base_pred_dir)) if args.base_pred_dir else {}

    usable_dates = [x for x in common_dates if x in daily_map]
    if not usable_dates:
        raise RuntimeError(
            json.dumps(
                {
                    "error": "No overlapping dates across common_dates and daily zarr time.",
                    "n_common_dates": len(common_dates),
                    "n_daily_dates": len(daily_map),
                    "common_sample": common_dates[:10],
                    "daily_sample": sorted(list(daily_map.keys()))[:10],
                },
                indent=2,
            )
        )

    t_indices = sorted([int(daily_map[d]) for d in usable_dates])
    n = len(t_indices)
    n_val = max(1, int(0.15 * n))
    n_test = max(1, int(0.15 * n))
    n_train = max(1, n - n_val - n_test)
    train_idx = t_indices[:n_train]
    val_idx = t_indices[n_train : n_train + n_val]
    test_idx = t_indices[n_train + n_val :]
    if not test_idx:
        test_idx = val_idx[:]

    era5_idx = [int(x.strip()) for x in str(args.era5_idx).split(",") if x.strip()]
    train_ds = Residual30mDataset(
        t_indices=train_idx,
        root_30m=root_30m,
        daily_times=daily_times,
        lst_min=args.lst_min_c,
        lst_max=args.lst_max_c,
        era5_idx=era5_idx,
    )
    val_ds = Residual30mDataset(
        t_indices=val_idx,
        root_30m=root_30m,
        daily_times=daily_times,
        lst_min=args.lst_min_c,
        lst_max=args.lst_max_c,
        era5_idx=era5_idx,
    )
    test_ds = Residual30mDataset(
        t_indices=test_idx,
        root_30m=root_30m,
        daily_times=daily_times,
        lst_min=args.lst_min_c,
        lst_max=args.lst_max_c,
        era5_idx=era5_idx,
    )

    sample = train_ds[0]
    widths = [int(x.strip()) for x in str(args.widths).split(",") if x.strip()]
    model = ResidualLSTNet30m(
        s2_ch=int(sample["s2"].shape[0]),
        s1_ch=int(sample["s1"].shape[0]),
        dem_ch=int(sample["dem"].shape[0]),
        lc_ch=int(sample["lc"].shape[0]),
        era5_ch=int(sample["era5"].shape[0]),
        widths=widths,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    basenet_runtime = load_basenet_from_ckpt(
        ckpt_path=args.basenet_ckpt,
        device=args.basenet_device,
        batch_cells=int(args.basenet_batch_cells),
    )

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_collate)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    use_amp = torch.cuda.is_available()

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    bad_epochs = 0
    basenet_cache: Dict[str, np.ndarray] = {}
    show_tqdm = not bool(args.no_tqdm)

    date_diag = {
        "n_common_dates": len(common_dates),
        "n_daily_dates": len(daily_map),
        "n_usable_dates": len(usable_dates),
        "n_legacy_precomputed_dates": len(legacy_pred),
        "missing_in_daily": sorted(list(set(common_dates) - set(daily_map.keys())))[:20],
    }
    print(json.dumps({"date_diagnostics": date_diag}, indent=2))

    epoch_iter = range(1, int(args.epochs) + 1)
    if show_tqdm:
        epoch_iter = tqdm(epoch_iter, desc="Epochs", leave=True, dynamic_ncols=True)
    for epoch in epoch_iter:
        tr = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            residual_l2=float(args.residual_l2),
            root_daily=root_daily,
            root_30m=root_30m,
            daily_times=daily_times,
            basenet_runtime=basenet_runtime,
            strict_feature_match=strict_feature_match,
            precomputed_map=legacy_pred,
            cache_basenet_pred=bool(args.cache_basenet_pred),
            basenet_cache=basenet_cache,
            tile_size=int(args.tile_size),
            tile_stride=int(args.tile_stride),
            phase="train",
            epoch_idx=epoch,
            total_epochs=int(args.epochs),
            show_tqdm=show_tqdm,
        )
        va = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            use_amp=use_amp,
            residual_l2=float(args.residual_l2),
            root_daily=root_daily,
            root_30m=root_30m,
            daily_times=daily_times,
            basenet_runtime=basenet_runtime,
            strict_feature_match=strict_feature_match,
            precomputed_map=legacy_pred,
            cache_basenet_pred=bool(args.cache_basenet_pred),
            basenet_cache=basenet_cache,
            tile_size=int(args.tile_size),
            tile_stride=int(args.tile_stride),
            phase="val",
            epoch_idx=epoch,
            total_epochs=int(args.epochs),
            show_tqdm=show_tqdm,
        )
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "train_rmse": tr["rmse"],
            "val_rmse": va["rmse"],
            "train_n_eval": tr["n_eval"],
            "val_n_eval": va["n_eval"],
        }
        history.append(row)
        print(json.dumps(row))
        tr_loss_txt = f"{tr['loss']:.4f}" if np.isfinite(tr["loss"]) else "nan"
        va_loss_txt = f"{va['loss']:.4f}" if np.isfinite(va["loss"]) else "nan"
        tr_rmse_txt = f"{tr['rmse']:.4f}" if np.isfinite(tr["rmse"]) else "nan"
        va_rmse_txt = f"{va['rmse']:.4f}" if np.isfinite(va["rmse"]) else "nan"
        print(
            f"Epoch {epoch}/{int(args.epochs)} | "
            f"train_loss={tr_loss_txt} val_loss={va_loss_txt} | "
            f"train_rmse={tr_rmse_txt} val_rmse={va_rmse_txt}"
        )
        if show_tqdm and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                {
                    "tr_loss": f"{tr['loss']:.4f}" if np.isfinite(tr["loss"]) else "nan",
                    "va_loss": f"{va['loss']:.4f}" if np.isfinite(va["loss"]) else "nan",
                    "tr_rmse": f"{tr['rmse']:.4f}" if np.isfinite(tr["rmse"]) else "nan",
                    "va_rmse": f"{va['rmse']:.4f}" if np.isfinite(va["rmse"]) else "nan",
                    "va_n": int(va["n_eval"]),
                },
                refresh=False,
            )

        if np.isfinite(va["rmse"]) and va["rmse"] < best_val:
            best_val = va["rmse"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_rmse": float(best_val),
                    "era5_idx": era5_idx,
                    "config": vars(args),
                },
                ckpt_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                break

    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    te = _run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        device=device,
        use_amp=use_amp,
        residual_l2=float(args.residual_l2),
        root_daily=root_daily,
        root_30m=root_30m,
        daily_times=daily_times,
        basenet_runtime=basenet_runtime,
        strict_feature_match=strict_feature_match,
        precomputed_map=legacy_pred,
        cache_basenet_pred=bool(args.cache_basenet_pred),
        basenet_cache=basenet_cache,
        tile_size=int(args.tile_size),
        tile_stride=int(args.tile_stride),
        phase="test",
        epoch_idx=int(ckpt.get("epoch", 0)),
        total_epochs=int(args.epochs),
        show_tqdm=show_tqdm,
    )
    final_metrics = {
        "best_val_rmse": float(ckpt.get("best_val_rmse", float("nan"))),
        "test_rmse": te["rmse"],
        "test_loss": te["loss"],
        "test_n_eval": te["n_eval"],
        "n_train_dates": len(train_idx),
        "n_val_dates": len(val_idx),
        "n_test_dates": len(test_idx),
    }

    pd.DataFrame(history).to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "splits.json").write_text(
        json.dumps(
            {
                "train_dates": [str(pd.Timestamp(daily_times[i]).date()) for i in train_idx],
                "val_dates": [str(pd.Timestamp(daily_times[i]).date()) for i in val_idx],
                "test_dates": [str(pd.Timestamp(daily_times[i]).date()) for i in test_idx],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "metrics_final.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": out_dir.as_posix(), "final_metrics": final_metrics}, indent=2))


if __name__ == "__main__":
    main()
