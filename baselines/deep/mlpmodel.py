from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import zarr
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

from helper.split_utils import load_or_create_splits
from helper import eval_utils

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"
COMMON_DATES = PROJECT_ROOT / "common_dates.csv"
SPLITS_PATH = PROJECT_ROOT / "metrics" / "common_date_splits.csv"
OUT_DIR = PROJECT_ROOT / "metrics" / "deep_baselines" / "mlp"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models" / "deep_baselines" / "mlp"

TARGET_SCALE = 10000.0

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

root_30m = zarr.open_group(str(ROOT_30M), mode="r")
root_daily = zarr.open_group(str(ROOT_DAILY), mode="r")

# ROI polygon (lon, lat)
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


def _to_str(arr):
    return np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
    )


common_df = pd.read_csv(COMMON_DATES)
common_dates = pd.to_datetime(common_df["landsat_date"], errors="coerce").dropna()
common_dates = pd.DatetimeIndex(common_dates).sort_values()

daily_raw = _to_str(root_30m["time"]["daily"][:])
monthly_raw = _to_str(root_30m["time"]["monthly"][:])

daily_times = pd.to_datetime(daily_raw, format="%Y_%m_%d", errors="coerce").dropna()
monthly_times = pd.to_datetime(monthly_raw, format="%Y_%m", errors="coerce").dropna()

daily_idx = np.flatnonzero(daily_times.isin(common_dates))
month_index = pd.DatetimeIndex(common_dates.to_period("M").to_timestamp())
monthly_idx = np.flatnonzero(monthly_times.isin(month_index))

monthly_map = {t: i for i, t in enumerate(monthly_times)}
daily_to_month = []
for t in daily_times[daily_idx]:
    m = t.to_period("M").to_timestamp()
    daily_to_month.append(monthly_map.get(m, -1))
daily_to_month = np.array(daily_to_month)
valid = daily_to_month >= 0
daily_idx = daily_idx[valid]
daily_to_month = daily_to_month[valid]
daily_to_month_map = {int(t): int(m) for t, m in zip(daily_idx, daily_to_month)}

print(f"daily_idx={len(daily_idx)} monthly_idx={len(monthly_idx)}")

landsat_shape = root_30m["labels_30m"]["landsat"]["data"].shape
H_hr, W_hr = landsat_shape[-2], landsat_shape[-1]

modis_shape = root_daily["products"]["modis"]["data"].shape
viirs_shape = root_daily["products"]["viirs"]["data"].shape
H_lr_modis, W_lr_modis = modis_shape[-2], modis_shape[-1]
H_lr_viirs, W_lr_viirs = viirs_shape[-2], viirs_shape[-1]

y_map_modis = np.floor(np.linspace(0, H_lr_modis - 1, H_hr)).astype(np.int64)
x_map_modis = np.floor(np.linspace(0, W_lr_modis - 1, W_hr)).astype(np.int64)
y_map_viirs = np.floor(np.linspace(0, H_lr_viirs - 1, H_hr)).astype(np.int64)
x_map_viirs = np.floor(np.linspace(0, W_lr_viirs - 1, W_hr)).astype(np.int64)

patch_size = 256


class TileDataset(Dataset):
    def __init__(self, n_samples, seed=0, allowed_t=None):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.allowed_t = np.array(allowed_t) if allowed_t is not None else np.array(daily_idx)

        self.g_era5 = root_30m["products_30m"]["era5"]["data"]
        self.g_landsat = root_30m["labels_30m"]["landsat"]["data"]
        self.g_s1 = root_30m["products_30m"]["sentinel1"]["data"]
        self.g_s2 = root_30m["products_30m"]["sentinel2"]["data"]
        self.g_modis = root_daily["products"]["modis"]["data"]
        self.g_viirs = root_daily["products"]["viirs"]["data"]

        self.g_dem = root_30m["static_30m"]["dem"]["data"]
        self.g_world = root_30m["static_30m"]["worldcover"]["data"]
        self.g_dyn = root_30m["static_30m"]["dynamic_world"]["data"]

    def __len__(self):
        return self.n_samples

    def _build_components(self, t, y0, x0):
        t = int(t)
        y0 = int(y0)
        x0 = int(x0)
        m = int(daily_to_month_map[t])

        y1 = y0 + patch_size
        x1 = x0 + patch_size

        era5 = self.g_era5[t, :, y0:y1, x0:x1]
        s1 = self.g_s1[m, :, y0:y1, x0:x1]
        s2 = self.g_s2[m, :, y0:y1, x0:x1]
        dem = self.g_dem[0, :, y0:y1, x0:x1]
        world = self.g_world[0, :, y0:y1, x0:x1]
        dyn = self.g_dyn[0, :, y0:y1, x0:x1]

        y = self.g_landsat[t, 0, y0:y1, x0:x1]

        y_idx_m = y_map_modis[y0:y1]
        x_idx_m = x_map_modis[x0:x1]
        y_idx_v = y_map_viirs[y0:y1]
        x_idx_v = x_map_viirs[x0:x1]

        modis_lr = self.g_modis[t, :, :, :]
        viirs_lr = self.g_viirs[t, :, :, :]
        modis_lr = self._apply_modis_mask(modis_lr)
        viirs_lr = self._apply_viirs_mask(viirs_lr)
        modis = modis_lr[:, y_idx_m][:, :, x_idx_m]
        viirs = viirs_lr[:, y_idx_v][:, :, x_idx_v]

        return {
            "era5": era5,
            "modis": modis,
            "viirs": viirs,
            "s1": s1,
            "s2": s2,
            "dem": dem,
            "world": world,
            "dyn": dyn,
            "y": y,
        }

    def _sample_components(self):
        i = self.rng.integers(0, len(self.allowed_t))
        t = int(self.allowed_t[i])
        y0 = self.rng.integers(0, H_hr - patch_size + 1)
        x0 = self.rng.integers(0, W_hr - patch_size + 1)
        return self._build_components(t, y0, x0)

    def __getitem__(self, idx):
        comp = self._sample_components()
        x = np.concatenate(
            [
                comp["era5"],
                comp["modis"],
                comp["viirs"],
                comp["s1"],
                comp["s2"],
                comp["dem"],
                comp["world"],
                comp["dyn"],
            ],
            axis=0,
        )
        return torch.from_numpy(x).float(), torch.from_numpy(comp["y"]).float()

    @staticmethod
    def _apply_modis_mask(modis_lr):
        if modis_lr.shape[0] < 2:
            return modis_lr
        lst = modis_lr[0].astype(np.float32) + 273.15
        mask = modis_lr[1].astype(np.float32)
        valid = mask == 0
        lst = np.where(valid, lst, np.nan)
        return np.stack([lst, mask], axis=0)

    @staticmethod
    def _apply_viirs_mask(viirs_lr):
        if viirs_lr.shape[0] < 2:
            return viirs_lr
        lst = viirs_lr[0].astype(np.float32)
        mask = viirs_lr[1].astype(np.float32)
        valid = mask <= 1
        lst = np.where(valid, lst, np.nan)
        return np.stack([lst, mask], axis=0)

    def sample_components(self):
        return self._sample_components()

    def get_components_at(self, t, y0, x0):
        return self._build_components(t, y0, x0)


def fill_nan_nearest(x):
    if torch.isfinite(x).all():
        return x
    x0 = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    h, w = x.shape[-2:]
    x_low = F.interpolate(x0, scale_factor=0.5, mode="nearest")
    x_up = F.interpolate(x_low, size=(h, w), mode="nearest")
    return torch.where(torch.isfinite(x), x, x_up)


def ensure_nchw(x, in_ch=None):
    if x.dim() != 4:
        return x
    if in_ch is None:
        if x.shape[1] > x.shape[-1]:
            return x.permute(0, 3, 1, 2).contiguous()
        return x
    if x.shape[1] == in_ch:
        return x
    if x.shape[-1] == in_ch:
        return x.permute(0, 3, 1, 2).contiguous()
    if x.shape[1] < x.shape[-1]:
        return x.permute(0, 3, 1, 2).contiguous()
    return x


def normalize_batch(x, in_ch=None):
    x = ensure_nchw(x, in_ch=in_ch)
    finite = torch.isfinite(x)
    x0 = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    denom = finite.sum(dim=(0, 2, 3)).clamp(min=1)
    mean = (x0 * finite).sum(dim=(0, 2, 3)) / denom
    var = ((x0 - mean[None, :, None, None]) ** 2 * finite).sum(dim=(0, 2, 3)) / denom
    std = var.sqrt().clamp(min=1e-6)
    return (x0 - mean[None, :, None, None]) / std[None, :, None, None]


def sample_pixels(xb, yb, max_pixels):
    b, c, h, w = xb.shape
    mask = torch.isfinite(yb)
    if not mask.any():
        return None, None
    x_flat = xb.permute(0, 2, 3, 1).reshape(-1, c)
    y_flat = yb.reshape(-1)
    m_flat = mask.reshape(-1)
    idx = torch.nonzero(m_flat, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return None, None
    if idx.numel() > max_pixels:
        perm = torch.randperm(idx.numel(), device=idx.device)[:max_pixels]
        idx = idx[perm]
    return x_flat[idx], y_flat[idx]


class PixelMLP(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def save_loss_plot(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(df["epoch"], df["train_loss"], label="train")
    ax.plot(df["epoch"], df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("mlp loss")
    ax.legend()
    fig.savefig(out_dir / "mlp_loss.png", dpi=150)
    plt.close(fig)


def save_prediction_figure(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> None:
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
    fig.savefig(out_dir / "mlp_pred_sample.png", dpi=150)
    plt.close(fig)


def save_metric_bar(metrics: dict, out_dir: Path) -> None:
    keys = list(metrics.keys())
    vals = [float(metrics[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    ax.bar(keys, vals)
    ax.set_title("mlp metrics")
    ax.set_ylabel("value")
    fig.savefig(out_dir / "mlp_metrics_bar.png", dpi=150)
    plt.close(fig)


def _tile_starts(full_size: int, tile: int) -> list:
    if full_size <= tile:
        return [0]
    starts = list(range(0, full_size - tile + 1, tile))
    last = full_size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def predict_full_map_mlp(model_in, dataset: TileDataset, t_idx: int, device, in_ch: int):
    model_in.eval()
    H, W = H_hr, W_hr
    y_pred = np.full((H, W), np.nan, dtype=np.float32)
    y_true = np.full((H, W), np.nan, dtype=np.float32)
    ys = _tile_starts(H, patch_size)
    xs = _tile_starts(W, patch_size)
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                comp = dataset.get_components_at(t_idx, y0, x0)
                x_raw = np.concatenate(
                    [
                        comp["era5"],
                        comp["modis"],
                        comp["viirs"],
                        comp["s1"],
                        comp["s2"],
                        comp["dem"],
                        comp["world"],
                        comp["dyn"],
                    ],
                    axis=0,
                )
                xb = torch.from_numpy(x_raw).float().unsqueeze(0).to(device)
                xb = ensure_nchw(xb, in_ch=in_ch)
                if not torch.isfinite(xb).all():
                    xb = fill_nan_nearest(xb)
                xb = normalize_batch(xb, in_ch=in_ch)
                x_flat = xb.permute(0, 2, 3, 1).reshape(-1, in_ch)
                pred = model_in(x_flat).reshape(patch_size, patch_size).cpu().numpy() * TARGET_SCALE
                y_patch = comp["y"]
                y_true[y0 : y0 + patch_size, x0 : x0 + patch_size] = y_patch
                y_pred[y0 : y0 + patch_size, x0 : x0 + patch_size] = pred
    return y_true, y_pred


def build_roi_mask(shape):
    try:
        root = zarr.open_group(str(ROOT_30M), mode="r")
        if "grid" not in root:
            return None
        g = root["grid"]
        transform = g.attrs.get("transform")
        if not transform or len(transform) < 6:
            return None
        crs_str = g.attrs.get("crs")
        base_roi = ROI_COORDS[0]
        roi_coords = base_roi
        transformed = False
        if crs_str and str(crs_str).upper() not in ("EPSG:4326", "WGS84"):
            try:
                from pyproj import CRS, Transformer
            except Exception as exc:
                print(f"ROI mask skipped: pyproj unavailable ({exc})")
                return None
            try:
                src = CRS.from_epsg(4326)
                dst = CRS.from_string(str(crs_str))
                transformer = Transformer.from_crs(src, dst, always_xy=True)
                roi_coords = [transformer.transform(lon, lat) for lon, lat in roi_coords]
                transformed = True
            except Exception as exc:
                print(f"ROI mask skipped: failed CRS transform ({exc})")
                return None
        a, b, c, d, e, f = transform[:6]
        H, W = shape
        cols = np.arange(W, dtype=np.float64) + 0.5
        rows = np.arange(H, dtype=np.float64) + 0.5
        cc, rr = np.meshgrid(cols, rows)
        lon = a * cc + b * rr + c
        lat = d * cc + e * rr + f
        points = np.column_stack([lon.ravel(), lat.ravel()])
        poly = np.array(roi_coords, dtype=np.float64)
        path = MplPath(poly)
        mask = path.contains_points(points).reshape(H, W)
        if not np.any(mask) and transformed:
            poly = np.array(base_roi, dtype=np.float64)
            path = MplPath(poly)
            mask = path.contains_points(points).reshape(H, W)
        if not np.any(mask):
            print("ROI mask empty; skipping ROI plot.")
            return None
        return mask
    except Exception as exc:
        print(f"ROI mask generation failed: {exc}")
        return None


def save_roi_figure(mask: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.imshow(mask.astype(np.float32), cmap="viridis")
    ax.set_title("Madurai ROI mask")
    ax.axis("off")
    fig.savefig(out_dir / "mlp_roi_mask.png", dpi=150)
    plt.close(fig)


splits = load_or_create_splits(COMMON_DATES, SPLITS_PATH)
train_dates = pd.DatetimeIndex(splits["train"]).normalize()
val_dates = pd.DatetimeIndex(splits["val"]).normalize()
test_dates = pd.DatetimeIndex(splits["test"]).normalize()
daily_norm = pd.DatetimeIndex(daily_times).normalize()
train_dates = [int(t) for t in daily_idx if daily_norm[int(t)] in train_dates]
val_dates = [int(t) for t in daily_idx if daily_norm[int(t)] in val_dates]
test_dates = [int(t) for t in daily_idx if daily_norm[int(t)] in test_dates]

split_rows = []
for split_name, dates in (("train", train_dates), ("val", val_dates), ("test", test_dates)):
    for t in dates:
        split_rows.append({"split": split_name, "date": daily_times[int(t)].strftime("%Y-%m-%d")})
split_path = OUT_DIR / "mlp_date_splits.csv"
pd.DataFrame(split_rows).to_csv(split_path, index=False)
print(f"saved date splits: {split_path}")

samples_per_epoch = 200
n_train = int(samples_per_epoch * 0.7)
n_val = int(samples_per_epoch * 0.1)
n_test = samples_per_epoch - n_train - n_val

train_ds = TileDataset(n_train, seed=1, allowed_t=train_dates)
val_ds = TileDataset(n_val, seed=2, allowed_t=val_dates)
test_ds = TileDataset(n_test, seed=3, allowed_t=test_dates)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_x = ensure_nchw(next(iter(train_loader))[0])
in_ch = sample_x.shape[1]

model = PixelMLP(in_ch=in_ch).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

max_epochs = 100
pixels_per_tile = 1024

history = []
best_val = float("inf")
best_epoch = 0
best_state = None

for epoch in range(1, max_epochs + 1):
    model.train()
    train_losses = []
    train_sq = 0.0
    train_n = 0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        xb = ensure_nchw(xb, in_ch=in_ch)
        if not torch.isfinite(xb).all():
            xb = fill_nan_nearest(xb)
        xb = normalize_batch(xb, in_ch=in_ch)

        finite_tgt = torch.isfinite(yb)
        if not finite_tgt.any():
            continue
        yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
        yb = yb / TARGET_SCALE

        x_pix, y_pix = sample_pixels(xb, yb, pixels_per_tile)
        if x_pix is None:
            continue
        pred = model(x_pix).squeeze(1)
        loss = torch.mean((pred - y_pix) ** 2)
        if not torch.isfinite(loss):
            continue

        err = pred - y_pix
        train_sq += float((err * err).sum().item())
        train_n += int(err.numel())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    val_sq = 0.0
    val_n = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb = ensure_nchw(xb, in_ch=in_ch)
            if not torch.isfinite(xb).all():
                xb = fill_nan_nearest(xb)
            xb = normalize_batch(xb, in_ch=in_ch)

            finite_tgt = torch.isfinite(yb)
            if not finite_tgt.any():
                continue
            yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
            yb = yb / TARGET_SCALE

            x_pix, y_pix = sample_pixels(xb, yb, pixels_per_tile)
            if x_pix is None:
                continue
            pred = model(x_pix).squeeze(1)
            loss = torch.mean((pred - y_pix) ** 2)
            if not torch.isfinite(loss):
                continue
            err = pred - y_pix
            val_sq += float((err * err).sum().item())
            val_n += int(err.numel())
            val_losses.append(loss.item())

    train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
    val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
    train_rmse = float(np.sqrt(train_sq / train_n)) if train_n > 0 else float("nan")
    val_rmse = float(np.sqrt(val_sq / val_n)) if val_n > 0 else float("nan")
    train_rmse_renorm = train_rmse * TARGET_SCALE if np.isfinite(train_rmse) else float("nan")
    val_rmse_renorm = val_rmse * TARGET_SCALE if np.isfinite(val_rmse) else float("nan")
    print(
        "epoch="
        f"{epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
        f"train_rmse={train_rmse_renorm:.6f} val_rmse={val_rmse_renorm:.6f}"
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_rmse": train_rmse_renorm,
            "val_rmse": val_rmse_renorm,
        }
    )

    if np.isfinite(val_loss) and val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        model_path = MODEL_DIR / "mlp_best.pt"
        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": best_state,
                "in_ch": in_ch,
                "target_scale": TARGET_SCALE,
            },
            model_path,
        )
        print(f"saved best model: {model_path}")

df_hist = pd.DataFrame(history)
csv_path = OUT_DIR / "mlp_metrics.csv"
df_hist.to_csv(csv_path, index=False)
print(f"saved metrics csv: {csv_path}")
save_loss_plot(df_hist, FIG_DIR)

if best_state is not None:
    model.load_state_dict(best_state)

roi_mask = eval_utils.build_roi_mask(ROOT_30M, (H_hr, W_hr))
if roi_mask is not None:
    eval_utils.save_roi_figure(roi_mask, FIG_DIR / "roi_mask.png")

eval_rows = []
figure_date = str(pd.Timestamp(daily_times[int(test_dates[-1])]).date()) if test_dates else None
for t in test_dates:
    y_true, y_pred = predict_full_map_mlp(model, test_ds, int(t), device, in_ch)
    met = eval_utils.compute_metrics(y_true, y_pred, roi_mask=roi_mask)
    date_str = str(pd.Timestamp(daily_times[int(t)]).date())
    eval_rows.append(
        {
            "time": date_str,
            **{k: met[k] for k in ("rmse", "ssim", "psnr", "sam", "cc")},
            "rmse_sum": met["rmse_sum"],
            "n_valid": met["n_valid"],
        }
    )
    if figure_date and date_str == figure_date:
        save_prediction_figure(y_true, y_pred, FIG_DIR)

metrics_path = OUT_DIR / "mlp_eval_metrics.csv"
pd.DataFrame(eval_rows).to_csv(metrics_path, index=False)
if eval_rows:
    save_metric_bar({k: eval_rows[0][k] for k in ("rmse", "ssim", "psnr", "sam", "cc")}, FIG_DIR)
print(f"saved eval metrics csv: {metrics_path}")
