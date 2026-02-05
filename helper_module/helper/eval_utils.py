from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import zarr
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

from helper.metrics_image import compute_all

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


def build_roi_mask(root_30m: Path, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    try:
        root = zarr.open_group(str(root_30m), mode="r")
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
            return None
        return mask
    except Exception:
        return None


def save_roi_figure(mask: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.imshow(mask.astype(np.uint8), cmap="gray")
    ax.set_title("ROI mask")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if roi_mask is not None and roi_mask.shape == y_true.shape:
        m = np.isfinite(y_true) & np.isfinite(y_pred) & roi_mask
    else:
        m = np.isfinite(y_true) & np.isfinite(y_pred)
    met = compute_all(y_true, y_pred, mask=m) if np.any(m) else {
        k: float("nan") for k in ("rmse", "ssim", "psnr", "sam", "cc")
    }
    if np.any(m):
        err = y_pred[m] - y_true[m]
        rmse_sum = float(np.sqrt(np.sum(err ** 2)))
    else:
        rmse_sum = float("nan")
    met_out = {k: float(met[k]) for k in ("rmse", "ssim", "psnr", "sam", "cc")}
    met_out["rmse_sum"] = rmse_sum
    met_out["n_valid"] = int(np.sum(m))
    return met_out
