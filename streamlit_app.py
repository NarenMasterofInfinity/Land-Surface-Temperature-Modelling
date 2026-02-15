import re
import math
from datetime import date as Date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import zarr
import matplotlib.pyplot as plt

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    TORCH_ERR = ""
except Exception as _exc:  # pragma: no cover
    TORCH_AVAILABLE = False
    TORCH_ERR = str(_exc)


ROOT = Path(__file__).resolve().parent
MADURAI_ZARR = ROOT / "madurai.zarr"
MADURAI_30M_ZARR = ROOT / "madurai_30m.zarr"
GOOD_DATES_CSV = ROOT / "good_landsat_dates.csv"
COMMON_DATES_CSV = ROOT / "common_dates.csv"


# ROI polygon (lon, lat) for Madurai
ROI_COORDS = [[
    [78.19018990852709, 9.878220339041174],
    [78.18528420047912, 9.882890316485547],
    [78.09810549679275, 9.894348798281609],
    [78.06157864432538, 9.932892703854442],
    [78.02656656110504, 9.94245002321148],
    [78.00260828167308, 9.965541799781803],
    [77.98788548795139, 9.97202781239784],
    [77.98759869039448, 9.972273355756196],
    [77.98926344201777, 9.974277652950137],
    [78.17034512986598, 10.09203200706642],
    [78.33409611393074, 10.299897884952559],
    [78.33446180425422, 10.301538866748283],
    [78.3424560260113, 10.300913203358672],
    [78.3620537189306, 10.301140183092196],
    [78.37500586343197, 10.310994022414183],
    [78.38801590170993, 10.315994815585963],
    [78.39017505117465, 10.315419769552259],
    [78.38542553295602, 10.314175493832316],
    [78.38810333700448, 10.269213359046619],
    [78.41273304358974, 10.212526305038399],
    [78.39622781746723, 10.177726046483423],
    [78.45775072031367, 10.100259541710393],
    [78.45817143938241, 9.967811808760526],
    [78.19018990852709, 9.878220339041174],
]]


def _decode_strings(vals: np.ndarray) -> List[str]:
    out = []
    for v in vals.tolist():
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def _parse_time_values(vals: np.ndarray) -> pd.DatetimeIndex:
    if vals.dtype.kind in {"S", "U", "O"}:
        raw = [s.strip() for s in _decode_strings(vals)]
        if raw and all(re.fullmatch(r"\d{4}_\d{2}_\d{2}", s) for s in raw):
            return pd.to_datetime(raw, format="%Y_%m_%d", errors="coerce")
        if raw and all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) for s in raw):
            return pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
        if raw and all(re.fullmatch(r"\d{4}_\d{2}", s) for s in raw):
            return pd.to_datetime(raw, format="%Y_%m", errors="coerce")
        if raw and all(re.fullmatch(r"\d{4}-\d{2}", s) for s in raw):
            return pd.to_datetime(raw, format="%Y-%m", errors="coerce")
        if raw and all(re.fullmatch(r"\d{4}", s) for s in raw):
            return pd.to_datetime(raw, format="%Y", errors="coerce")
        return pd.to_datetime(raw, format="mixed", errors="coerce")
    return pd.to_datetime(vals, errors="coerce")


@st.cache_data(show_spinner=False)
def load_good_dates(path: Path) -> List[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    col = df.columns[0]
    vals = [str(v).strip() for v in df[col].tolist() if isinstance(v, str) or not pd.isna(v)]
    return [v for v in vals if v]


@st.cache_data(show_spinner=False)
def load_common_dates(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "landsat_date" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["landsat_date"] = pd.to_datetime(df["landsat_date"], errors="coerce")
    df = df.dropna(subset=["landsat_date"])
    df["landsat_date"] = df["landsat_date"].dt.normalize()
    return df


@st.cache_resource(show_spinner=False)
def open_zarr_group(path: Path) -> zarr.Group:
    return zarr.open_group(str(path), mode="r")


@st.cache_data(show_spinner=False)
def load_time_map(root_path: Path) -> Dict[int, pd.DatetimeIndex]:
    out: Dict[int, pd.DatetimeIndex] = {}
    time_root = root_path / "time"
    if not time_root.exists():
        return out
    for key in ("daily", "monthly", "annual"):
        arr_path = time_root / key
        if not arr_path.exists():
            continue
        arr = zarr.open_array(str(arr_path), mode="r")
        vals = arr[:]
        parsed = _parse_time_values(vals)
        if len(parsed) > 0:
            out[len(parsed)] = parsed
    return out


@st.cache_data(show_spinner=False)
def load_grid_meta(root_30m_path: Path) -> Tuple[Optional[List[float]], Optional[str]]:
    grid_path = root_30m_path / "grid"
    if not grid_path.exists():
        return None, None
    group = zarr.open_group(str(grid_path), mode="r")
    transform = group.attrs.get("transform")
    crs_str = group.attrs.get("crs")
    if isinstance(transform, np.ndarray):
        transform = transform.tolist()
    return transform, crs_str


def _find_time_index(time_vals: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[int]:
    if time_vals is None or len(time_vals) == 0:
        return None
    if time_vals.isna().all():
        return None
    t_norm = target.normalize()
    eq = time_vals.normalize() == t_norm
    if np.any(eq):
        return int(np.argmax(eq))
    same_month = (time_vals.year == t_norm.year) & (time_vals.month == t_norm.month)
    if np.any(same_month):
        return int(np.argmax(same_month))
    return None


def _get_band_names(group_path: Path) -> Optional[List[str]]:
    bn_path = group_path / "band_names"
    if not bn_path.exists():
        return None
    try:
        arr = zarr.open_array(str(bn_path), mode="r")
        return _decode_strings(arr[:])
    except Exception:
        return None


def _choose_band(band_names: Optional[List[str]], prefer: Iterable[str]) -> int:
    if not band_names:
        return 0
    pref = [p.lower() for p in prefer]
    for i, name in enumerate(band_names):
        low = str(name).lower()
        for p in pref:
            if p in low:
                return i
    return 0


def _load_data_slice(
    root_path: Path,
    group_path: str,
    target_date: pd.Timestamp,
    time_map: Dict[int, pd.DatetimeIndex],
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    data_path = root_path / group_path / "data"
    if not data_path.exists():
        return None, "missing data array"
    arr = zarr.open_array(str(data_path), mode="r")
    shape = arr.shape
    if len(shape) == 4:
        time_len = shape[0]
        time_vals = time_map.get(time_len)
        if time_len == 1 and time_vals is None:
            return np.asarray(arr[0]), None
        idx = _find_time_index(time_vals, target_date) if time_vals is not None else None
        if idx is None:
            return None, "date not available"
        return np.asarray(arr[idx]), None
    if len(shape) == 3 and shape[0] in time_map:
        time_vals = time_map.get(shape[0])
        idx = _find_time_index(time_vals, target_date) if time_vals is not None else None
        if idx is None:
            return None, "date not available"
        return np.asarray(arr[idx]), None
    if len(shape) == 3 and shape[0] == 1:
        return np.asarray(arr[0]), None
    return np.asarray(arr[:]), None


def _build_roi_mask(
    shape: Tuple[int, int],
    transform: Optional[List[float]],
    crs_str: Optional[str],
) -> Optional[np.ndarray]:
    if not transform or len(transform) < 6:
        return None
    try:
        from matplotlib.path import Path as MplPath
    except Exception:
        return None
    base_roi = ROI_COORDS[0]
    roi_coords = base_roi
    transformed = False
    if crs_str and str(crs_str).upper() not in ("EPSG:4326", "WGS84"):
        try:
            from pyproj import CRS, Transformer
        except Exception:
            return None
        try:
            src = CRS.from_epsg(4326)
            dst = CRS.from_string(str(crs_str))
            transformer = Transformer.from_crs(src, dst, always_xy=True)
            roi_coords = [transformer.transform(lon, lat) for lon, lat in roi_coords]
            transformed = True
        except Exception:
            return None
    a, b, c, d, e, f = transform[:6]
    h, w = shape
    cols = np.arange(w, dtype=np.float64) + 0.5
    rows = np.arange(h, dtype=np.float64) + 0.5
    cc, rr = np.meshgrid(cols, rows)
    lon = a * cc + b * rr + c
    lat = d * cc + e * rr + f
    points = np.column_stack([lon.ravel(), lat.ravel()])
    poly = np.array(roi_coords, dtype=np.float64)
    path = MplPath(poly)
    mask = path.contains_points(points).reshape(h, w)
    if not np.any(mask) and transformed:
        poly = np.array(base_roi, dtype=np.float64)
        path = MplPath(poly)
        mask = path.contains_points(points).reshape(h, w)
    if not np.any(mask):
        return None
    return mask


def _roi_crop(arr2d: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if mask is None or mask.shape != arr2d.shape:
        return arr2d, None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return arr2d, None
    cropped = arr2d[np.ix_(rows, cols)]
    cropped_mask = mask[np.ix_(rows, cols)]
    cropped = np.where(cropped_mask, cropped, np.nan)
    return cropped, cropped_mask


def _normalize_for_display(arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    vmin, vmax = np.nanpercentile(arr[finite], [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(arr[finite]))
        vmax = float(np.nanmax(arr[finite]))
        if vmin == vmax:
            vmax = vmin + 1.0
    out = (arr - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _display_range(arr: np.ndarray) -> Tuple[float, float]:
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(arr[finite], [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(arr[finite]))
        vmax = float(np.nanmax(arr[finite]))
        if vmin == vmax:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _plot_heatmap(arr: np.ndarray, cmap_name: str, title: str) -> None:
    vmin, vmax = _display_range(arr)
    data = np.ma.masked_invalid(arr)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="black", alpha=0.0)
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _resize_with_mask(lr: np.ndarray, target_shape: Tuple[int, int], mode: str) -> np.ndarray:
    valid = np.isfinite(lr)
    if not np.any(valid):
        return np.full(target_shape, np.nan, dtype=np.float32)
    fill_val = float(np.nanmedian(lr[valid]))
    lr_filled = np.where(valid, lr, fill_val).astype(np.float32)

    def _interp(x: np.ndarray, m: str) -> np.ndarray:
        try:
            import torch

            tx = torch.from_numpy(x)[None, None, :, :]
            align = False if m in ("bilinear", "bicubic") else None
            ty = torch.nn.functional.interpolate(tx, size=target_shape, mode=m, align_corners=align)
            return ty[0, 0].cpu().numpy()
        except Exception:
            pass
        try:
            from skimage.transform import resize

            order = 0 if m == "nearest" else 1
            return resize(x, target_shape, order=order, preserve_range=True, anti_aliasing=False).astype(np.float32)
        except Exception:
            pass
        raise RuntimeError("No backend available for resizing. Install torch or scikit-image.")

    pred = _interp(lr_filled, mode)
    mask_up = _interp(valid.astype(np.float32), "nearest") > 0.5
    return np.where(mask_up, pred, np.nan).astype(np.float32)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray]) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None and mask.shape == y_true.shape:
        m &= mask
    if not np.any(m):
        return float("nan")
    err = y_pred[m] - y_true[m]
    return float(np.sqrt(np.mean(err * err)))


CNN_LRHR_INPUT_ORDER = ["era5", "s1", "s2", "dem", "world", "dyn"]
ARCH_V1_INPUT_ORDER = ["era5", "s2", "s1", "dem", "world", "dyn"]


@st.cache_data(show_spinner=False)
def load_time_vectors_30m(root_path: Path) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, np.ndarray]:
    root = zarr.open_group(str(root_path), mode="r")
    daily_raw = _decode_strings(root["time"]["daily"][:])
    monthly_raw = _decode_strings(root["time"]["monthly"][:])
    daily = pd.to_datetime(daily_raw, format="%Y_%m_%d", errors="coerce")
    monthly = pd.to_datetime(monthly_raw, format="%Y_%m", errors="coerce")
    if pd.isna(daily).all():
        daily = pd.to_datetime(daily_raw, format="mixed", errors="coerce")
    if pd.isna(monthly).all():
        monthly = pd.to_datetime(monthly_raw, format="mixed", errors="coerce")
    daily = pd.DatetimeIndex(daily).dropna()
    monthly = pd.DatetimeIndex(monthly).dropna()
    monthly_map = {t: i for i, t in enumerate(monthly)}
    daily_to_month = []
    for t in daily:
        m = t.to_period("M").to_timestamp()
        daily_to_month.append(monthly_map.get(m, -1))
    return daily, monthly, np.array(daily_to_month, dtype=np.int64)


def _tile_starts(full_size: int, tile: int) -> List[int]:
    if full_size <= tile:
        return [0]
    starts = list(range(0, full_size - tile + 1, tile))
    last = full_size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _landsat_scale_offset(root_30m: zarr.Group) -> Tuple[float, float]:
    try:
        g = root_30m["labels_30m"]["landsat"]
        attrs = dict(g.attrs)
        scale = attrs.get("scale_factor", attrs.get("scale", 1.0))
        offset = attrs.get("add_offset", attrs.get("offset", 0.0))
        return float(scale or 1.0), float(offset or 0.0)
    except Exception:
        return 1.0, 0.0


def _landsat_to_celsius(arr: np.ndarray, scale: float, offset: float) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    arr = np.where(arr == 149, np.nan, arr)
    if scale != 1.0 or offset != 0.0:
        arr = arr * scale + offset
    if np.isfinite(arr).any() and np.nanmedian(arr) > 200:
        arr = arr - 273.15
    return arr


def _apply_range_mask(arr: np.ndarray, min_c: float = 10.0, max_c: float = 70.0) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(arr) & (arr >= min_c) & (arr <= max_c)
    return np.where(valid, arr, np.nan).astype(np.float32), valid.astype(bool)


def _channel_sizes(root_30m: zarr.Group) -> Dict[str, int]:
    return {
        "era5": int(root_30m["products_30m"]["era5"]["data"].shape[1]),
        "s1": int(root_30m["products_30m"]["sentinel1"]["data"].shape[1]),
        "s2": int(root_30m["products_30m"]["sentinel2"]["data"].shape[1]),
        "dem": int(root_30m["static_30m"]["dem"]["data"].shape[1]),
        "world": int(root_30m["static_30m"]["worldcover"]["data"].shape[1]),
        "dyn": int(root_30m["static_30m"]["dynamic_world"]["data"].shape[1]),
    }


def _mask_channel_indices(order: List[str], sizes: Dict[str, int]) -> List[int]:
    offsets: Dict[str, int] = {}
    idx = 0
    for name in order:
        offsets[name] = idx
        idx += sizes[name]
    mask_idx: List[int] = []
    if "world" in offsets:
        w0 = offsets["world"]
        for i in range(sizes["world"]):
            mask_idx.append(w0 + i)
    if "dyn" in offsets:
        d0 = offsets["dyn"]
        for i in range(sizes["dyn"]):
            mask_idx.append(d0 + i)
    return sorted(set(mask_idx))


def _stack_inputs(comp: Dict[str, np.ndarray], order: List[str]) -> np.ndarray:
    parts = []
    for key in order:
        arr = comp[key]
        if arr.ndim == 2:
            arr = arr[None, ...]
        parts.append(arr)
    return np.concatenate(parts, axis=0)


def _build_inputs_patch(
    root_30m: zarr.Group,
    t_idx: int,
    m_idx: int,
    y0: int,
    x0: int,
    patch_size: int,
) -> Dict[str, np.ndarray]:
    y1 = y0 + patch_size
    x1 = x0 + patch_size
    g_era5 = root_30m["products_30m"]["era5"]["data"]
    g_s1 = root_30m["products_30m"]["sentinel1"]["data"]
    g_s2 = root_30m["products_30m"]["sentinel2"]["data"]
    g_dem = root_30m["static_30m"]["dem"]["data"]
    g_world = root_30m["static_30m"]["worldcover"]["data"]
    g_dyn = root_30m["static_30m"]["dynamic_world"]["data"]

    if m_idx < 0:
        s1 = np.full((g_s1.shape[1], patch_size, patch_size), np.nan, dtype=np.float32)
        s2 = np.full((g_s2.shape[1], patch_size, patch_size), np.nan, dtype=np.float32)
    else:
        s1 = g_s1[m_idx, :, y0:y1, x0:x1]
        s2 = g_s2[m_idx, :, y0:y1, x0:x1]

    return {
        "era5": g_era5[t_idx, :, y0:y1, x0:x1],
        "s1": s1,
        "s2": s2,
        "dem": g_dem[0, :, y0:y1, x0:x1],
        "world": g_world[0, :, y0:y1, x0:x1],
        "dyn": g_dyn[0, :, y0:y1, x0:x1],
    }


def _normalize_batch_global(x: torch.Tensor, mu: np.ndarray, sigma: np.ndarray, mask_idx: List[int]) -> torch.Tensor:
    x0 = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu_t = torch.as_tensor(mu, device=x0.device, dtype=x0.dtype)[None, :, None, None]
    sigma_t = torch.as_tensor(sigma, device=x0.device, dtype=x0.dtype)[None, :, None, None]
    out = (x0 - mu_t) / sigma_t
    if mask_idx:
        out[:, mask_idx, :, :] = x0[:, mask_idx, :, :]
    return out


@st.cache_data(show_spinner=False)
def compute_input_stats(
    order_key: str,
    root_path: str,
    daily_to_month: Tuple[int, ...],
    allowed_t: Tuple[int, ...],
    patch_size: int,
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    root_30m = zarr.open_group(root_path, mode="r")
    sizes = _channel_sizes(root_30m)
    order = CNN_LRHR_INPUT_ORDER if order_key == "cnn_lr_hr" else ARCH_V1_INPUT_ORDER
    mask_idx = _mask_channel_indices(order, sizes)
    rng = np.random.default_rng(seed)

    sum_x = np.zeros(sum(sizes[k] for k in order), dtype=np.float64)
    sum_sq = np.zeros_like(sum_x)
    count = np.zeros_like(sum_x)

    H = int(root_30m["labels_30m"]["landsat"]["data"].shape[-2])
    W = int(root_30m["labels_30m"]["landsat"]["data"].shape[-1])

    daily_to_month_arr = np.array(daily_to_month, dtype=np.int64)
    allowed = np.array(allowed_t, dtype=np.int64)
    for _ in range(max(1, n_samples)):
        t_idx = int(rng.choice(allowed))
        y0 = int(rng.integers(0, max(1, H - patch_size + 1)))
        x0 = int(rng.integers(0, max(1, W - patch_size + 1)))
        m_idx = int(daily_to_month_arr[t_idx]) if t_idx < len(daily_to_month_arr) else -1
        comp = _build_inputs_patch(root_30m, t_idx, m_idx, y0, x0, patch_size)
        x = _stack_inputs(comp, order)
        for ch in range(x.shape[0]):
            if ch in mask_idx:
                continue
            vals = x[ch]
            finite = np.isfinite(vals)
            if not np.any(finite):
                continue
            v = vals[finite].astype(np.float64, copy=False)
            sum_x[ch] += v.sum()
            sum_sq[ch] += (v * v).sum()
            count[ch] += v.size

    mu = np.zeros_like(sum_x, dtype=np.float32)
    sigma = np.ones_like(sum_x, dtype=np.float32)
    for ch in range(mu.size):
        if ch in mask_idx:
            continue
        if count[ch] > 0:
            mu[ch] = float(sum_x[ch] / count[ch])
            var = max(0.0, float(sum_sq[ch] / count[ch] - mu[ch] * mu[ch]))
            sigma[ch] = float(var**0.5) if var > 0 else 1.0
    return mu, sigma


class SimpleCNN(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.act(out)


class ResNetSmall(nn.Module):
    def __init__(self, in_ch: int, width: int = 32, depth: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(depth)])
        self.head = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class ConvNeXtBlock(nn.Module):
    def __init__(self, ch: int, expansion: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(ch, ch, kernel_size=7, padding=3, groups=ch)
        self.norm = nn.LayerNorm(ch)
        self.pw1 = nn.Linear(ch, expansion * ch)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(expansion * ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwconv(x)
        y = y.permute(0, 2, 3, 1)
        y = self.norm(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        y = y.permute(0, 3, 1, 2)
        return x + y


class ConvNeXtSmall(nn.Module):
    def __init__(self, in_ch: int, width: int = 32, depth: int = 4):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, width, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(width) for _ in range(depth)])
        self.head = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class HRNetBasicBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.act(out)


def _make_branch(ch: int, blocks: int) -> nn.Sequential:
    return nn.Sequential(*[HRNetBasicBlock(ch) for _ in range(blocks)])


def _downsample(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )


def _upsample(in_ch: int, out_ch: int, scale: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_ch),
    )


class HRNetSmall(nn.Module):
    def __init__(self, in_ch: int, blocks: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.stage1 = _make_branch(32, blocks)
        self.transition1 = _downsample(32, 64)
        self.stage2_b1 = _make_branch(32, blocks)
        self.stage2_b2 = _make_branch(64, blocks)
        self.fuse2_1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, bias=False), nn.BatchNorm2d(32))
        self.fuse2_2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64))
        self.transition2 = _downsample(64, 128)
        self.stage3_b1 = _make_branch(32, blocks)
        self.stage3_b2 = _make_branch(64, blocks)
        self.stage3_b3 = _make_branch(128, blocks)
        self.fuse3_1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, bias=False), nn.BatchNorm2d(32))
        self.fuse3_1b = nn.Sequential(nn.Conv2d(128, 32, kernel_size=1, bias=False), nn.BatchNorm2d(32))
        self.fuse3_2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64))
        self.fuse3_2b = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        self.fuse3_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.fuse3_3b = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128))
        self.transition3 = _downsample(128, 256)
        self.stage4_b1 = _make_branch(32, blocks)
        self.stage4_b2 = _make_branch(64, blocks)
        self.stage4_b3 = _make_branch(128, blocks)
        self.stage4_b4 = _make_branch(256, blocks)
        self.up_2_to_1 = _upsample(64, 32, scale=2)
        self.up_3_to_1 = _upsample(128, 32, scale=4)
        self.up_4_to_1 = _upsample(256, 32, scale=8)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.transition1(x1)
        y1 = self.stage2_b1(x1)
        y2 = self.stage2_b2(x2)
        y1 = y1 + F.interpolate(self.fuse2_1(y2), size=y1.shape[-2:], mode="bilinear", align_corners=False)
        y2 = y2 + self.fuse2_2(y1)
        x3 = self.transition2(y2)
        z1 = self.stage3_b1(y1)
        z2 = self.stage3_b2(y2)
        z3 = self.stage3_b3(x3)
        z1 = z1 + F.interpolate(self.fuse3_1(z2), size=z1.shape[-2:], mode="bilinear", align_corners=False)
        z1 = z1 + F.interpolate(self.fuse3_1b(z3), size=z1.shape[-2:], mode="bilinear", align_corners=False)
        z2 = z2 + self.fuse3_2(z1)
        z2 = z2 + F.interpolate(self.fuse3_2b(z3), size=z2.shape[-2:], mode="bilinear", align_corners=False)
        z3 = z3 + self.fuse3_3(z1)
        z3 = z3 + self.fuse3_3b(z2)
        x4 = self.transition3(z3)
        w1 = self.stage4_b1(z1)
        w2 = self.stage4_b2(z2)
        w3 = self.stage4_b3(z3)
        w4 = self.stage4_b4(x4)
        w = w1
        w = w + self.up_2_to_1(w2)
        w = w + self.up_3_to_1(w3)
        w = w + self.up_4_to_1(w4)
        return self.head(w)


class ResDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, blocks: int = 2):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        ]
        for _ in range(blocks):
            layers.append(ResBlock(out_ch))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            ResBlock(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(),
            ResBlock(base),
        )
        self.enc1 = ResDown(base, base * 2, blocks=2)
        self.enc2 = ResDown(base * 2, base * 4, blocks=2)
        self.enc3 = ResDown(base * 4, base * 8, blocks=2)
        self.bottleneck = nn.Sequential(ResBlock(base * 8), ResBlock(base * 8))
        self.up2 = ResUp(base * 8 + base * 4, base * 4)
        self.up1 = ResUp(base * 4 + base * 2, base * 2)
        self.up0 = ResUp(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x3 = self.bottleneck(x3)
        x = self.up2(x3, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        return self.head(x)


def _discover_cnn_lr_hr_models() -> Dict[str, Dict[str, str]]:
    base_dir = ROOT / "models" / "deep_baselines" / "cnn_lr_hr"
    specs: Dict[str, Dict[str, str]] = {}
    if not base_dir.exists():
        return specs
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        best = None
        for cand in run_dir.glob("*_best.pt"):
            best = cand
            break
        if best is None:
            continue
        name = run_dir.name
        lname = name.lower()
        if "hrnet" in lname:
            arch = "hrnet"
        elif "unet" in lname:
            arch = "unet"
        elif "convnext" in lname:
            arch = "convnext"
        elif "resnet" in lname:
            arch = "resnet"
        else:
            arch = "cnn"
        specs[name] = {"arch": arch, "ckpt": str(best)}
    return specs


def _label_cnn_lr_hr_run(run_name: str) -> Optional[str]:
    lname = run_name.lower()
    if "hr_lr" in lname or "cnn_hr_lr" in lname:
        return None
    if lname == "cnn_lr_hr":
        return "cnn model"
    if lname.startswith("cnn_lr_hr_"):
        return lname.replace("cnn_lr_hr_", "")
    return run_name


def _find_fusion_pred(date_str: str, method: str, source: str) -> Optional[Path]:
    base = ROOT / "metrics" / "fusion_baselines"
    if method == "starfm":
        if source == "modis":
            cand = base / "starfm" / "modis" / f"starfm_pred_{date_str}.npy"
            return cand if cand.exists() else None
        if source == "viirs":
            cand = base / "starfm" / "viirs" / f"starfm_pred_{date_str}.npy"
            return cand if cand.exists() else None
        cand = base / "starfm" / f"starfm_pred_{date_str}.npy"
        return cand if cand.exists() else None
    if method == "ustarfm":
        if source == "modis":
            cand = base / "ustarfm_modis" / f"ustarfm_pred_{date_str}.npy"
            return cand if cand.exists() else None
        if source == "viirs":
            cand = base / "ustarfm_viirs" / f"ustarfm_pred_{date_str}.npy"
            return cand if cand.exists() else None
    if method == "fsdaf":
        if source == "modis":
            cand = base / "fsdaf" / "modis" / f"fsdaf_pred_{date_str}.npy"
            return cand if cand.exists() else None
        if source == "viirs":
            cand = base / "fsdaf" / "viirs" / f"fsdaf_pred_{date_str}.npy"
            return cand if cand.exists() else None
    return None


def _load_starfm_module():
    import importlib.util

    path = ROOT / "baselines" / "fusion" / "starfm.py"
    spec = importlib.util.spec_from_file_location("starfm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load baselines/fusion/starfm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_ustarfm_module():
    import importlib.util

    path = ROOT / "baselines" / "fusion" / "ustarfm.py"
    spec = importlib.util.spec_from_file_location("ustarfm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load baselines/fusion/ustarfm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fsdaf_module():
    import importlib.util

    path = ROOT / "baselines" / "fusion" / "fsdaf.py"
    spec = importlib.util.spec_from_file_location("fsdaf", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load baselines/fusion/fsdaf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lowres_available(source: str, modis_files: str, viirs_files: str) -> bool:
    if source == "modis":
        return bool(str(modis_files or "").strip())
    if source == "viirs":
        return bool(str(viirs_files or "").strip())
    return False


def _select_base_idx(
    allowed_t: Iterable[int],
    target_idx: int,
    landsat_group: zarr.Array,
    lowres_group: zarr.Array,
    source: str,
    mod,
) -> Tuple[int, np.ndarray, np.ndarray]:
    candidates = [int(t) for t in allowed_t if int(t) <= int(target_idx)]
    if not candidates:
        candidates = [int(t) for t in allowed_t]
    candidates = sorted(set(candidates), reverse=True)
    for t in candidates:
        y0 = mod._extract_landsat(landsat_group[int(t), 0])
        if not mod._has_valid(y0):
            continue
        if source == "modis":
            c0 = mod._extract_modis(lowres_group[int(t)])
        else:
            c0 = mod._extract_viirs(lowres_group[int(t)])
        if not mod._has_valid(c0):
            continue
        return int(t), y0, c0
    raise RuntimeError("No base date with valid Landsat and low-res data found.")


def _starfm_predict_with_progress(
    mod,
    F_tb: np.ndarray,
    C_tb_lr: np.ndarray,
    C_t_lr: np.ndarray,
    row_float: np.ndarray,
    col_float: np.ndarray,
    y_slices: list,
    x_slices: list,
    r_lr: int,
    sigma_d: float,
    sigma_f: float,
    sigma_c: float,
    sigma_t: float,
    min_weight_sum: float,
    progress: Optional[st.delta_generator.DeltaGenerator],
) -> np.ndarray:
    H, W = F_tb.shape
    h, w = C_tb_lr.shape
    C_tb_hr = mod._bilinear_full(C_tb_lr, row_float, col_float)
    C_t_hr = mod._bilinear_full(C_t_lr, row_float, col_float)
    F_hat = np.full((H, W), np.nan, dtype=np.float32)

    for i in range(h):
        if progress is not None and h > 1:
            progress.progress(min(0.99, i / (h - 1)))
        ysl = y_slices[i]
        if ysl.stop <= ysl.start:
            continue
        for j in range(w):
            xsl = x_slices[j]
            if xsl.stop <= xsl.start:
                continue

            F_block = F_tb[ysl, xsl]
            if np.all(np.isnan(F_block)):
                continue
            Bh, Bw = F_block.shape

            Ctb_block = C_tb_hr[ysl, xsl]
            Ct_block = C_t_hr[ysl, xsl]

            i0 = max(0, i - r_lr)
            i1 = min(h - 1, i + r_lr)
            j0 = max(0, j - r_lr)
            j1 = min(w - 1, j + r_lr)

            candidates_F = []
            candidates_Ctb = []
            candidates_Ct = []
            candidates_dist = []

            for ni in range(i0, i1 + 1):
                ysl2 = y_slices[ni]
                if ysl2.stop <= ysl2.start:
                    continue
                for nj in range(j0, j1 + 1):
                    xsl2 = x_slices[nj]
                    if xsl2.stop <= xsl2.start:
                        continue
                    Fb2 = F_tb[ysl2, xsl2]
                    if np.all(np.isnan(Fb2)):
                        continue
                    Ctb2 = C_tb_hr[ysl2, xsl2]
                    Ct2 = C_t_hr[ysl2, xsl2]
                    Fb2 = mod._resize_bilinear_block(Fb2, Bh, Bw)
                    Ctb2 = mod._resize_bilinear_block(Ctb2, Bh, Bw)
                    Ct2 = mod._resize_bilinear_block(Ct2, Bh, Bw)
                    d = math.sqrt((ni - i) ** 2 + (nj - j) ** 2)
                    candidates_F.append(Fb2)
                    candidates_Ctb.append(Ctb2)
                    candidates_Ct.append(Ct2)
                    candidates_dist.append(d)

            if len(candidates_F) == 0:
                continue

            cand_F = np.stack(candidates_F, axis=0)
            cand_Ctb = np.stack(candidates_Ctb, axis=0)
            cand_Ct = np.stack(candidates_Ct, axis=0)
            cand_d = np.array(candidates_dist, dtype=np.float32)[:, None, None]

            cand_val = cand_F + (cand_Ct - cand_Ctb)

            w_d = np.exp(-cand_d / max(sigma_d, 1e-6))
            F_ref = mod.safe_nanmean(F_block)
            w_f = np.exp(-np.abs(cand_F - F_ref) / max(sigma_f, 1e-6))
            C_ref = mod.safe_nanmean(Ctb_block)
            w_c = np.exp(-np.abs(cand_Ctb - C_ref) / max(sigma_c, 1e-6))
            w_t = np.exp(-np.abs(cand_Ct - cand_Ctb) / max(sigma_t, 1e-6))

            w_all = w_d * w_f * w_c * w_t
            w_all = np.where(np.isnan(cand_val), 0.0, w_all)
            w_sum = np.sum(w_all, axis=0)
            num = np.sum(w_all * cand_val, axis=0)
            pred_block = np.where(w_sum > min_weight_sum, num / w_sum, np.nan).astype(np.float32)
            F_hat[ysl, xsl] = pred_block

    if progress is not None:
        progress.progress(1.0)
    return F_hat


@st.cache_resource(show_spinner=False)
def _ustarfm_class_map(root_30m_path: str, n_classes: int, sample_pixels: int, seed: int) -> np.ndarray:
    mod = _load_ustarfm_module()
    root_30m = open_zarr_group(Path(root_30m_path))
    dem = mod._to_2d(root_30m["static_30m"]["dem"]["data"][0])
    world = mod._to_2d(root_30m["static_30m"]["worldcover"]["data"][0])
    dyn = mod._to_2d(root_30m["static_30m"]["dynamic_world"]["data"][0])
    X = np.stack([dem, world, dyn], axis=-1).astype(np.float32)
    return mod.build_class_map(X, n_classes, sample_pixels, seed)


@st.cache_data(show_spinner=False)
def _load_linear_rmse(model_name: str) -> Dict[str, float]:
    path = ROOT / "metrics" / "linear_baselines" / f"{model_name}_metrics.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: Dict[str, float] = {}
    if "time" not in df.columns or "rmse" not in df.columns:
        return out
    for _, row in df.iterrows():
        date_str = str(row["time"])
        try:
            rmse = float(row["rmse"])
        except Exception:
            rmse = float("nan")
        out[date_str] = rmse
    return out


def _find_linear_pred_png(model_name: str, date_str: str) -> Optional[Path]:
    fig = ROOT / "metrics" / "linear_baselines" / "figures" / f"{model_name}_{date_str}_map.png"
    return fig if fig.exists() else None


def _load_arch_v1_module():
    import importlib.util

    path = ROOT / "scripts" / "arch_v1_model.py"
    spec = importlib.util.spec_from_file_location("arch_v1_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load arch_v1_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_model_from_checkpoint(arch: str, ckpt_path: Path, in_ch: int) -> Tuple[nn.Module, Dict[str, float], int]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    in_ch_ckpt = int(ckpt.get("in_ch", in_ch))
    if in_ch_ckpt != in_ch:
        raise ValueError(f"Checkpoint expects in_ch={in_ch_ckpt}, but inputs provide in_ch={in_ch}.")
    if arch == "cnn":
        model = SimpleCNN(in_ch=in_ch_ckpt)
    elif arch == "resnet":
        model = ResNetSmall(in_ch=in_ch_ckpt)
    elif arch == "convnext":
        model = ConvNeXtSmall(in_ch=in_ch_ckpt)
    elif arch == "unet":
        model = UNetResNet(in_ch=in_ch_ckpt)
    elif arch == "hrnet":
        model = HRNetSmall(in_ch=in_ch_ckpt)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    state = ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    target_mu = float(ckpt.get("target_mu", 0.0))
    target_sigma = float(ckpt.get("target_sigma", 1.0))
    return model, {"target_mu": target_mu, "target_sigma": target_sigma}, in_ch_ckpt


def _predict_full_map(
    *,
    model: nn.Module,
    root_30m: zarr.Group,
    t_idx: int,
    m_idx: int,
    patch_size: int,
    order: List[str],
    mu_x: np.ndarray,
    sigma_x: np.ndarray,
    mask_idx: List[int],
    target_mu: float,
    target_sigma: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    landsat_arr = root_30m["labels_30m"]["landsat"]["data"]
    H, W = landsat_arr.shape[-2], landsat_arr.shape[-1]
    y_pred = np.full((H, W), np.nan, dtype=np.float32)
    y_true = np.full((H, W), np.nan, dtype=np.float32)

    scale, offset = _landsat_scale_offset(root_30m)
    ys = _tile_starts(H, patch_size)
    xs = _tile_starts(W, patch_size)

    model = model.to(device)
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                comp = _build_inputs_patch(root_30m, t_idx, m_idx, y0, x0, patch_size)
                x_np = _stack_inputs(comp, order)
                xb = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
                xb = _normalize_batch_global(xb, mu_x, sigma_x, mask_idx)
                pred = model(xb).squeeze(0).squeeze(0).cpu().numpy()
                pred = pred * (target_sigma if target_sigma else 1.0) + target_mu

                y_patch = landsat_arr[t_idx, 0, y0 : y0 + patch_size, x0 : x0 + patch_size]
                y_patch = _landsat_to_celsius(np.asarray(y_patch), scale, offset)
                y_patch, _ = _apply_range_mask(y_patch)

                y_true[y0 : y0 + patch_size, x0 : x0 + patch_size] = y_patch
                y_pred[y0 : y0 + patch_size, x0 : x0 + patch_size] = pred.astype(np.float32, copy=False)
    return y_true, y_pred


def _predict_full_map_arch_v1(
    *,
    model,
    root_30m: zarr.Group,
    t_idx: int,
    m_idx: int,
    patch_size: int,
    mu_x: np.ndarray,
    sigma_x: np.ndarray,
    mask_idx: List[int],
    device: torch.device,
    use_doy: bool,
    variant: str,
    doy_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    landsat_arr = root_30m["labels_30m"]["landsat"]["data"]
    H, W = landsat_arr.shape[-2], landsat_arr.shape[-1]
    y_pred = np.full((H, W), np.nan, dtype=np.float32)
    y_true = np.full((H, W), np.nan, dtype=np.float32)

    scale, offset = _landsat_scale_offset(root_30m)
    ys = _tile_starts(H, patch_size)
    xs = _tile_starts(W, patch_size)
    order = ARCH_V1_INPUT_ORDER

    model = model.to(device)
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                comp = _build_inputs_patch(root_30m, t_idx, m_idx, y0, x0, patch_size)
                x_np = _stack_inputs(comp, order)
                xb = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
                xb = _normalize_batch_global(xb, mu_x, sigma_x, mask_idx)
                if use_doy:
                    doy = torch.tensor([float(doy_value)], device=device)
                    out = model(xb, doy=doy)
                else:
                    out = model(xb)
                pred = out["base"] if variant == "base" else out["y_hat"]
                pred = pred.squeeze(0).squeeze(0).cpu().numpy()

                y_patch = landsat_arr[t_idx, 0, y0 : y0 + patch_size, x0 : x0 + patch_size]
                y_patch = _landsat_to_celsius(np.asarray(y_patch), scale, offset)
                y_patch, _ = _apply_range_mask(y_patch)

                y_true[y0 : y0 + patch_size, x0 : x0 + patch_size] = y_patch
                y_pred[y0 : y0 + patch_size, x0 : x0 + patch_size] = pred.astype(np.float32, copy=False)
    return y_true, y_pred


def _load_product_2d(
    root_path: Path,
    group_path: str,
    target_date: pd.Timestamp,
    time_map: Dict[int, pd.DatetimeIndex],
    prefer_bands: Iterable[str],
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[List[str]]]:
    data, err = _load_data_slice(root_path, group_path, target_date, time_map)
    if data is None:
        return None, err, None
    if data.ndim == 2:
        return data, None, None
    band_names = _get_band_names(root_path / group_path)
    band_idx = _choose_band(band_names, prefer_bands)
    if data.ndim == 3:
        return data[band_idx], None, band_names
    return None, "unexpected data shape", band_names


def _load_daily_product_2d_by_index(
    root_path: Path,
    group_path: str,
    t_idx: Optional[int],
    prefer_bands: Iterable[str],
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[List[str]]]:
    if t_idx is None or t_idx < 0:
        return None, "date not available", None
    data_path = root_path / group_path / "data"
    if not data_path.exists():
        return None, "missing data array", None
    arr = zarr.open_array(str(data_path), mode="r")
    shape = arr.shape
    if len(shape) == 4:
        if t_idx >= shape[0]:
            return None, "date index out of range", None
        data = np.asarray(arr[t_idx])
    elif len(shape) == 3:
        if shape[0] == 1:
            data = np.asarray(arr[0])
        else:
            if t_idx >= shape[0]:
                return None, "date index out of range", None
            data = np.asarray(arr[t_idx])
    else:
        data = np.asarray(arr[:])
    if data.ndim == 2:
        return data, None, None
    band_names = _get_band_names(root_path / group_path)
    band_idx = _choose_band(band_names, prefer_bands)
    if data.ndim == 3:
        return data[band_idx], None, band_names
    return None, "unexpected data shape", band_names


def _find_month_index(monthly_times: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[int]:
    if monthly_times is None or len(monthly_times) == 0:
        return None
    t = target.to_period("M").to_timestamp().normalize()
    m_norm = pd.DatetimeIndex(monthly_times).normalize()
    idx = np.where(m_norm == t)[0]
    if idx.size:
        return int(idx[0])
    return None


def _load_monthly_product_2d(
    root_path: Path,
    group_path: str,
    month_idx: Optional[int],
    prefer_bands: Iterable[str],
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[List[str]]]:
    if month_idx is None or month_idx < 0:
        return None, "month not available", None
    data_path = root_path / group_path / "data"
    if not data_path.exists():
        return None, "missing data array", None
    arr = zarr.open_array(str(data_path), mode="r")
    shape = arr.shape
    if len(shape) == 4:
        if month_idx >= shape[0]:
            return None, "month index out of range", None
        data = np.asarray(arr[month_idx])
    elif len(shape) == 3:
        if shape[0] == 1:
            data = np.asarray(arr[0])
        else:
            if month_idx >= shape[0]:
                return None, "month index out of range", None
            data = np.asarray(arr[month_idx])
    else:
        data = np.asarray(arr[:])
    if data.ndim == 2:
        return data, None, None
    band_names = _get_band_names(root_path / group_path)
    band_idx = _choose_band(band_names, prefer_bands)
    if data.ndim == 3:
        return data[band_idx], None, band_names
    return None, "unexpected data shape", band_names


def _make_source_specs() -> List[Dict[str, str]]:
    return [
        {"label": "Landsat LST (30m, 16days)", "root": "madurai_30m", "group": "labels_30m/landsat", "cadence": "daily", "cmap": "inferno"},
        {"label": "Sentinel-2 (monthly, 30m)", "root": "madurai_30m", "group": "products_30m/sentinel2", "cadence": "monthly", "cmap": "viridis"},
        {"label": "Sentinel-1 (monthly, 30m)", "root": "madurai_30m", "group": "products_30m/sentinel1", "cadence": "monthly", "cmap": "cividis"},
        {"label": "ERA5 (daily, 30m)", "root": "madurai_30m", "group": "products_30m/era5", "cadence": "daily", "cmap": "plasma"},
        {"label": "DEM (static, 30m)", "root": "madurai_30m", "group": "static_30m/dem", "cadence": "static", "cmap": "terrain"},
        {"label": "WorldCover (static, 30m)", "root": "madurai_30m", "group": "static_30m/worldcover", "cadence": "static", "cmap": "tab20"},
        {"label": "Dynamic World (static, 30m)", "root": "madurai_30m", "group": "static_30m/dynamic_world", "cadence": "static", "cmap": "tab20b"},
        {"label": "MODIS (daily, 1km)", "root": "madurai", "group": "products/modis", "cadence": "daily", "cmap": "magma"},
        {"label": "VIIRS (daily, 1km)", "root": "madurai", "group": "products/viirs", "cadence": "daily", "cmap": "hot"},
        {"label": "ERA5 (daily, coarse)", "root": "madurai", "group": "products/era5", "cadence": "daily", "cmap": "plasma"},
    ]


st.set_page_config(page_title="Madurai LST Super-Resolution", layout="wide")
st.title("Madurai LST Super-Resolution")

if not MADURAI_ZARR.exists() or not MADURAI_30M_ZARR.exists():
    st.error("Required Zarr stores not found. Expected madurai.zarr and madurai_30m.zarr in the project root.")
    st.stop()

good_dates = load_good_dates(GOOD_DATES_CSV)
common_df = load_common_dates(COMMON_DATES_CSV)
if common_df.empty:
    st.error("common_dates.csv is missing or empty.")
    st.stop()

daily_times_30m, monthly_times_30m, daily_to_month = load_time_vectors_30m(MADURAI_30M_ZARR)
daily_norm = pd.DatetimeIndex(daily_times_30m).normalize()
daily_set = set(daily_norm)

common_df = common_df.copy()
common_df["landsat_date"] = pd.to_datetime(common_df["landsat_date"], errors="coerce")
common_df = common_df.dropna(subset=["landsat_date"])
common_dates = pd.DatetimeIndex(common_df["landsat_date"])
daily_idx = np.flatnonzero(daily_times_30m.isin(common_dates))
available_dates = pd.DatetimeIndex(daily_times_30m[daily_idx]).strftime("%Y-%m-%d")

def _pref_score(row: pd.Series) -> int:
    score = 0
    if str(row.get("modis_files", "") or "").strip():
        score += 2
    if str(row.get("viirs_files", "") or "").strip():
        score += 2
    if str(row.get("era5_files", "") or "").strip():
        score += 1
    outlier = str(row.get("outlier", "") or "").strip().lower()
    if outlier == "no":
        score += 1
    return score

common_df["pref_score"] = common_df.apply(_pref_score, axis=1)
common_df = common_df.sort_values(["pref_score", "landsat_date"], ascending=[False, True])
available_dates = (
    pd.DatetimeIndex(common_df["landsat_date"])
    .normalize()
    .isin(daily_times_30m)
)
available_dates = (
    common_df.loc[available_dates, "landsat_date"]
    .dt.strftime("%Y-%m-%d")
    .drop_duplicates()
    .tolist()
)
if not available_dates:
    st.error("No usable dates found after intersecting common_dates.csv with madurai_30m daily timeline.")
    st.stop()

if good_dates:
    examples = [d for d in good_dates[:5] if d in available_dates]
    example_text = ", ".join(examples if examples else good_dates[:5])
    st.caption("Top 5 best dates (examples): " + example_text)

selected_date_str = st.selectbox("Select a date", options=available_dates, index=0)
selected_date = pd.Timestamp(Date.fromisoformat(selected_date_str))

date_row = common_df[common_df["landsat_date"] == selected_date.normalize()]
modis_files = ""
viirs_files = ""
if not date_row.empty:
    modis_files = str(date_row.iloc[0].get("modis_files", "") or "")
    viirs_files = str(date_row.iloc[0].get("viirs_files", "") or "")
    col_a, col_b = st.columns(2)
    col_a.write(f"MODIS files: `{modis_files if modis_files else 'none'}`")
    col_b.write(f"VIIRS files: `{viirs_files if viirs_files else 'none'}`")

daily_norm = pd.DatetimeIndex(daily_times_30m).normalize()
sel_idx = np.where(daily_norm == selected_date.normalize())[0]
selected_t_idx = int(sel_idx[0]) if sel_idx.size else None

time_map_daily = load_time_map(MADURAI_ZARR)
time_map_30m = load_time_map(MADURAI_30M_ZARR)
grid_transform_30m, grid_crs_30m = load_grid_meta(MADURAI_30M_ZARR)

st.subheader("Data Sources (Madurai ROI Cut-Out)")
sources = _make_source_specs()
cols = st.columns(3)

for i, src in enumerate(sources):
    root_key = src["root"]
    group_path = src["group"]
    label = src["label"]
    cadence = src.get("cadence", "daily")
    cmap_name = src.get("cmap", "viridis")
    root_path = MADURAI_30M_ZARR if root_key == "madurai_30m" else MADURAI_ZARR
    time_map = time_map_30m if root_key == "madurai_30m" else time_map_daily

    try:
        group = open_zarr_group(root_path)[group_path]
    except Exception:
        cols[i % 3].warning(f"{label}: missing group")
        continue

    if cadence == "static":
        data2d, err, band_names = _load_monthly_product_2d(
            root_path,
            group_path,
            month_idx=0,
            prefer_bands=("lst", "temperature", "day"),
        )
    elif cadence == "monthly":
        month_idx = _find_month_index(monthly_times_30m, selected_date)
        data2d, err, band_names = _load_monthly_product_2d(
            root_path,
            group_path,
            month_idx=month_idx,
            prefer_bands=("lst", "temperature", "day"),
        )
    else:
        if "MODIS" in label and not modis_files:
            cols[i % 3].warning(f"{label}: not available for this date")
            continue
        if "VIIRS" in label and not viirs_files:
            cols[i % 3].warning(f"{label}: not available for this date")
            continue
        data2d, err, band_names = _load_daily_product_2d_by_index(
            root_path,
            group_path,
            selected_t_idx,
            prefer_bands=("lst", "temperature", "day"),
        )
    if data2d is None:
        cols[i % 3].warning(f"{label}: {err}")
        continue

    if root_key == "madurai_30m" and grid_transform_30m is not None:
        mask = _build_roi_mask(data2d.shape, grid_transform_30m, grid_crs_30m)
    else:
        transform = group.attrs.get("transform")
        crs_str = group.attrs.get("crs")
        mask = _build_roi_mask(data2d.shape, transform, crs_str)
    data_roi, _ = _roi_crop(data2d, mask)
    with cols[i % 3]:
        _plot_heatmap(data_roi, cmap_name, label)


st.subheader("Super-Resolution")
if not TORCH_AVAILABLE:
    st.warning(f"Torch not available: {TORCH_ERR}")

sidebar = st.sidebar
patch_size = int(sidebar.selectbox("Patch size", options=[128, 192, 256], index=2))
n_stats_samples = int(sidebar.number_input("Input stats samples", min_value=20, max_value=400, value=120, step=20))

cnn_specs = _discover_cnn_lr_hr_models()

model_groups: Dict[str, Dict[str, Dict[str, str]]] = {
    "Fusion": {},
    "Linear": {},
    "CNN": {},
    "ResNet": {},
    "HRNet": {},
    "ConvNeXt": {},
    "Architecture": {},
}

for name, spec in cnn_specs.items():
    label = _label_cnn_lr_hr_run(name)
    if not label:
        continue
    if spec["arch"] == "cnn":
        pretty = label if label != "cnn model" else "cnn model"
        model_groups["CNN"][pretty] = {"family": "cnn_lr_hr", "name": name, **spec}
    elif spec["arch"] == "resnet":
        model_groups["ResNet"][label] = {"family": "cnn_lr_hr", "name": name, **spec}
    elif spec["arch"] == "hrnet":
        model_groups["HRNet"][label] = {"family": "cnn_lr_hr", "name": name, **spec}
    elif spec["arch"] == "convnext":
        model_groups["ConvNeXt"][label] = {"family": "cnn_lr_hr", "name": name, **spec}

arch_ckpt = ROOT / "models" / "arch_v1" / "best.pt"
cnn_ckpt = ROOT / "models" / "deep_baselines" / "cnn_lr_hr" / "cnn_lr_hr" / "cnn_lr_hr_best.pt"
arch_ckpt_final = cnn_ckpt if cnn_ckpt.exists() else arch_ckpt
if arch_ckpt_final.exists():
    model_groups["Architecture"]["Architecture"] = {"family": "arch_v1", "name": "arch_v1", "ckpt": str(arch_ckpt_final)}

model_groups["Fusion"]["STARFM (MODIS)"] = {"family": "fusion", "method": "starfm", "source": "modis"}
model_groups["Fusion"]["STARFM (VIIRS)"] = {"family": "fusion", "method": "starfm", "source": "viirs"}
model_groups["Fusion"]["USTARFM (MODIS)"] = {"family": "fusion", "method": "ustarfm", "source": "modis"}
model_groups["Fusion"]["USTARFM (VIIRS)"] = {"family": "fusion", "method": "ustarfm", "source": "viirs"}
model_groups["Fusion"]["FSDAF (MODIS)"] = {"family": "fusion", "method": "fsdaf", "source": "modis"}
model_groups["Fusion"]["FSDAF (VIIRS)"] = {"family": "fusion", "method": "fsdaf", "source": "viirs"}

model_groups["Linear"]["ols"] = {"family": "linear", "model": "ols"}
model_groups["Linear"]["ridge"] = {"family": "linear", "model": "ridge"}
model_groups["Linear"]["lasso"] = {"family": "linear", "model": "lasso"}
model_groups["Linear"]["elasticnet"] = {"family": "linear", "model": "elasticnet"}

available_groups = [k for k, v in model_groups.items() if v]
if not available_groups:
    st.error("No models found for inference.")
    st.stop()

group_choice = st.selectbox("Model Group", options=available_groups)
model_label = st.selectbox("Model", options=list(model_groups[group_choice].keys()))
model_spec = model_groups[group_choice][model_label]

run_clicked = st.button("Run Super-Resolution")

pred_roi = None
pred_image_path = None
hr_roi = None
rmse_val = float("nan")
model_name = model_label
hr_band_names = None

if run_clicked:
    if not TORCH_AVAILABLE:
        st.error(f"Torch not available: {TORCH_ERR}")
        st.stop()

    root_30m = open_zarr_group(MADURAI_30M_ZARR)
    if selected_t_idx is None:
        st.error("Selected date not found in madurai_30m daily timeline.")
        st.stop()
    t_idx = int(selected_t_idx)
    m_idx = int(daily_to_month[t_idx]) if t_idx < len(daily_to_month) else -1

    common_set = set(common_df["landsat_date"].dt.normalize())
    allowed_t = [int(i) for i, t in enumerate(daily_norm) if t in common_set]
    if not allowed_t:
        st.error("No allowed dates found for model stats.")
        st.stop()

    sizes = _channel_sizes(root_30m)

    if model_spec["family"] == "cnn_lr_hr":
        order = CNN_LRHR_INPUT_ORDER
        mask_idx = _mask_channel_indices(order, sizes)
        mu_x, sigma_x = compute_input_stats(
            "cnn_lr_hr",
            str(MADURAI_30M_ZARR),
            tuple(daily_to_month.tolist()),
            tuple(allowed_t),
            patch_size,
            n_stats_samples,
            seed=42,
        )
        in_ch = sum(sizes[k] for k in order)
        model, meta, _ = _load_model_from_checkpoint(model_spec["arch"], Path(model_spec["ckpt"]), in_ch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with st.spinner("Running model inference..."):
            y_true, y_pred = _predict_full_map(
                model=model,
                root_30m=root_30m,
                t_idx=t_idx,
                m_idx=m_idx,
                patch_size=patch_size,
                order=order,
                mu_x=mu_x,
                sigma_x=sigma_x,
                mask_idx=mask_idx,
                target_mu=meta["target_mu"],
                target_sigma=meta["target_sigma"],
                device=device,
            )
    elif model_spec["family"] == "arch_v1":
        order = ARCH_V1_INPUT_ORDER
        mask_idx = _mask_channel_indices(order, sizes)
        mu_x, sigma_x = compute_input_stats(
            "arch_v1",
            str(MADURAI_30M_ZARR),
            tuple(daily_to_month.tolist()),
            tuple(allowed_t),
            patch_size,
            n_stats_samples,
            seed=42,
        )
        in_ch = sum(sizes[k] for k in order)
        model, meta, _ = _load_model_from_checkpoint("cnn", Path(model_spec["ckpt"]), in_ch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with st.spinner("Running model inference..."):
            y_true, y_pred = _predict_full_map(
                model=model,
                root_30m=root_30m,
                t_idx=t_idx,
                m_idx=m_idx,
                patch_size=patch_size,
                order=order,
                mu_x=mu_x,
                sigma_x=sigma_x,
                mask_idx=mask_idx,
                device=device,
                target_mu=meta["target_mu"],
                target_sigma=meta["target_sigma"],
            )
    elif model_spec["family"] == "fusion":
        date_str = selected_date.strftime("%Y-%m-%d")
        if not _lowres_available(model_spec["source"], modis_files, viirs_files):
            st.error(f"Data not available for {model_label} on {date_str}.")
            st.stop()
        pred_path = _find_fusion_pred(date_str, model_spec["method"], model_spec["source"])
        if model_spec["method"] == "starfm":
            fusion_mod = _load_starfm_module()
        elif model_spec["method"] == "ustarfm":
            fusion_mod = _load_ustarfm_module()
        elif model_spec["method"] == "fsdaf":
            fusion_mod = _load_fsdaf_module()
        else:
            st.error("Unsupported fusion method.")
            st.stop()
        landsat_group = root_30m["labels_30m"]["landsat"]["data"]
        if pred_path is not None:
            pred_full = np.load(pred_path)
            y_pred = np.asarray(pred_full, dtype=np.float32)
        else:
            root_daily = open_zarr_group(MADURAI_ZARR)
            lowres_group = root_daily["products"][model_spec["source"]]["data"]
            try:
                base_idx, F_tb, C_tb = _select_base_idx(
                    allowed_t,
                    t_idx,
                    landsat_group,
                    lowres_group,
                    model_spec["source"],
                    fusion_mod,
                )
            except Exception as exc:
                st.error(str(exc))
                st.stop()
            if model_spec["source"] == "modis":
                C_t = fusion_mod._extract_modis(lowres_group[t_idx])
            else:
                C_t = fusion_mod._extract_viirs(lowres_group[t_idx])
            if not fusion_mod._has_valid(C_t):
                st.error(f"Data not available for {model_label} on {date_str}.")
                st.stop()
            H_hr, W_hr = F_tb.shape
            H_lr, W_lr = C_tb.shape
            row_float = np.linspace(0, H_lr - 1, H_hr, dtype=np.float64)
            col_float = np.linspace(0, W_lr - 1, W_hr, dtype=np.float64)
            row_map = np.clip(np.rint(row_float).astype(np.int64), 0, H_lr - 1)
            col_map = np.clip(np.rint(col_float).astype(np.int64), 0, W_lr - 1)
            if model_spec["method"] == "starfm":
                y_slices = fusion_mod.build_hr_slices(row_map, H_lr)
                x_slices = fusion_mod.build_hr_slices(col_map, W_lr)
                progress = st.progress(0.0)
                with st.spinner("Running STARFM..."):
                    y_pred = _starfm_predict_with_progress(
                        fusion_mod,
                        F_tb=F_tb,
                        C_tb_lr=C_tb,
                        C_t_lr=C_t,
                        row_float=row_float,
                        col_float=col_float,
                        y_slices=y_slices,
                        x_slices=x_slices,
                        r_lr=fusion_mod.R_LR,
                        sigma_d=fusion_mod.SIGMA_D,
                        sigma_f=fusion_mod.SIGMA_F,
                        sigma_c=fusion_mod.SIGMA_C,
                        sigma_t=fusion_mod.SIGMA_T,
                        min_weight_sum=fusion_mod.MIN_WEIGHT_SUM,
                        progress=progress,
                    )
            elif model_spec["method"] == "ustarfm":
                progress = st.progress(0.0)
                with st.spinner("Running USTARFM..."):
                    progress.progress(0.2)
                    class_map = _ustarfm_class_map(
                        str(MADURAI_30M_ZARR),
                        fusion_mod.N_CLASSES,
                        fusion_mod.SAMPLE_PIXELS,
                        fusion_mod.SEED,
                    )
                    progress.progress(0.5)
                    A = fusion_mod.compute_fraction_matrix(
                        class_map,
                        (H_lr, W_lr),
                        row_map,
                        col_map,
                        fusion_mod.N_CLASSES,
                    )
                    mu_tb = fusion_mod.class_means_from_fine(F_tb, class_map, fusion_mod.N_CLASSES)
                    c = C_t.reshape(-1).astype(np.float32)
                    valid = np.isfinite(c)
                    if valid.sum() < max(10, int(0.05 * c.size)):
                        st.error(f"Data not available for {model_label} on {date_str}.")
                        st.stop()
                    A_v = A[valid]
                    c_v = c[valid]
                    AtA_v = (A_v.T @ A_v).astype(np.float32)
                    I = np.eye(fusion_mod.N_CLASSES, dtype=np.float32)
                    rhs = (A_v.T @ c_v).astype(np.float32) + fusion_mod.PRIOR_LAMBDA * mu_tb
                    x = np.linalg.solve(
                        AtA_v + (fusion_mod.RIDGE_LAMBDA + fusion_mod.PRIOR_LAMBDA) * I,
                        rhs,
                    ).astype(np.float32)
                    y_pred = fusion_mod.render_hr_from_class_temps(class_map, x)
                    progress.progress(1.0)
            else:
                progress = st.progress(0.0)
                with st.spinner("Running FSDAF..."):
                    progress.progress(0.3)
                    C_tb_hr = fusion_mod._bilinear_full(C_tb, row_float, col_float)
                    C_t_hr = fusion_mod._bilinear_full(C_t, row_float, col_float)
                    y_pred = (F_tb + (C_t_hr - C_tb_hr)).astype(np.float32)
                    progress.progress(1.0)
        y_true = fusion_mod._extract_landsat(landsat_group[t_idx, 0, :, :])
        if y_pred.shape != y_true.shape:
            st.error("Fusion prediction shape does not match Landsat grid.")
            st.stop()
    elif model_spec["family"] == "linear":
        date_str = selected_date.strftime("%Y-%m-%d")
        rmse_map = _load_linear_rmse(model_spec["model"])
        rmse_val = float(rmse_map.get(date_str, float("nan")))
        pred_image_path = _find_linear_pred_png(model_spec["model"], date_str)
        landsat_group = root_30m["labels_30m"]["landsat"]["data"]
        y_true = landsat_group[t_idx, 0, :, :]
        scale, offset = _landsat_scale_offset(root_30m)
        y_true = _landsat_to_celsius(np.asarray(y_true), scale, offset)
        y_true, _ = _apply_range_mask(y_true)
    else:
        st.error("Selected model family not supported.")
        st.stop()

    if grid_transform_30m is not None:
        roi_mask_hr = _build_roi_mask(y_true.shape, grid_transform_30m, grid_crs_30m)
    else:
        landsat_group = open_zarr_group(MADURAI_30M_ZARR)["labels_30m/landsat"]
        roi_mask_hr = _build_roi_mask(y_true.shape, landsat_group.attrs.get("transform"), landsat_group.attrs.get("crs"))
    if model_spec["family"] != "linear":
        rmse_val = _rmse(y_true, y_pred, roi_mask_hr)
        pred_roi, _ = _roi_crop(y_pred, roi_mask_hr)
    hr_roi, _ = _roi_crop(y_true, roi_mask_hr)

col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse_val:.3f}" if np.isfinite(rmse_val) else "nan")
col2.write(f"Selected model: `{model_label}`")
col3.write("Landsat band: index 0")

st.subheader("Predicted LST (Madurai ROI)")
if pred_image_path is not None and pred_image_path.exists():
    st.image(str(pred_image_path), caption="Predicted LST (linear baseline)", width="stretch")
elif pred_roi is None:
    st.info("Run super-resolution to see predictions.")
else:
    _plot_heatmap(pred_roi, "inferno", "Predicted LST")

st.subheader("Landsat LST (Madurai ROI)")
if hr_roi is None:
    st.info("Run super-resolution to see target LST.")
else:
    _plot_heatmap(hr_roi, "inferno", "Landsat LST")
