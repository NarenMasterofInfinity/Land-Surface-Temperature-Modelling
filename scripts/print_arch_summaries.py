from __future__ import annotations

from pathlib import Path
import sys

try:
    from torchinfo import summary
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "torchinfo is required. Install with: python3 -m pip install --user torchinfo"
    ) from exc

import torch
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"
ROOT_DAILY = PROJECT_ROOT / "madurai.zarr"

# Defaults (used when zarr is missing or incomplete)
H_HR_DEFAULT = 256
W_HR_DEFAULT = 256
S2_CH_DEFAULT = 10
S1_CH_DEFAULT = 2
ERA5_CH_DEFAULT = 4
LC_CLASSES_DEFAULT = 17


def _get_shapes():
    if not ROOT_30M.exists() or not ROOT_DAILY.exists():
        return {
            "H_hr": H_HR_DEFAULT,
            "W_hr": W_HR_DEFAULT,
            "H_lr": H_HR_DEFAULT // 8,
            "W_lr": W_HR_DEFAULT // 8,
            "s2_ch": S2_CH_DEFAULT,
            "s1_ch": S1_CH_DEFAULT,
            "era5_ch": ERA5_CH_DEFAULT,
            "lc_classes": LC_CLASSES_DEFAULT,
        }

    root_30m = zarr.open_group(str(ROOT_30M), mode="r")
    root_daily = zarr.open_group(str(ROOT_DAILY), mode="r")

    landsat_shape = root_30m["labels_30m"]["landsat"]["data"].shape
    H_hr, W_hr = landsat_shape[-2], landsat_shape[-1]

    modis_shape = root_daily["products"]["modis"]["data"].shape
    H_lr, W_lr = modis_shape[-2], modis_shape[-1]

    s2_ch = root_30m["products_30m"]["sentinel2"]["data"].shape[1]
    s1_ch = root_30m["products_30m"]["sentinel1"]["data"].shape[1]

    world_labels = root_30m["static_30m"]["worldcover"]["labels"][:]
    lc_classes = int(len(world_labels))

    return {
        "H_hr": H_hr,
        "W_hr": W_hr,
        "H_lr": H_lr,
        "W_lr": W_lr,
        "s2_ch": s2_ch,
        "s1_ch": s1_ch,
        "era5_ch": ERA5_CH_DEFAULT,
        "lc_classes": lc_classes,
    }


def main() -> int:
    shapes = _get_shapes()
    H_hr = shapes["H_hr"]
    W_hr = shapes["W_hr"]
    H_lr = shapes["H_lr"]
    W_lr = shapes["W_lr"]

    from baselines.deep.cnn_lr_hr.thermal_base.thermal_base_net import ThermalBaseNet, UpsampleHead
    from baselines.deep.cnn_lr_hr.thermal_base.thermal_residual_net import ResidualNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BaseNet inputs
    B = 1
    T = 3
    modis_frames = torch.zeros((B, T, 1, H_lr, W_lr), device=device)
    viirs_frames = torch.zeros((B, T, 1, H_lr, W_lr), device=device)
    modis_masks = torch.zeros((B, T, 1, H_lr, W_lr), device=device)
    viirs_masks = torch.zeros((B, T, 1, H_lr, W_lr), device=device)
    era5 = torch.zeros((B, shapes["era5_ch"], H_lr, W_lr), device=device)
    doy = torch.zeros((B, 2), device=device)
    static = torch.zeros((B, 3, H_lr, W_lr), device=device)

    base_net = ThermalBaseNet(era5_k=shapes["era5_ch"]).to(device)
    up_head = UpsampleHead(in_ch=1).to(device)

    print("\n=== ThermalBaseNet (LR) ===")
    summary(
        base_net,
        input_data=(modis_frames, viirs_frames, era5, doy, static, modis_masks, viirs_masks),
        verbose=0,
        device=device,
    )

    print("\n=== UpsampleHead (LR -> HR) ===")
    summary(
        up_head,
        input_data=(torch.zeros((B, 1, H_lr, W_lr), device=device),),
        verbose=0,
        device=device,
    )

    # ResidualNet inputs
    s2 = torch.zeros((B, shapes["s2_ch"], H_hr, W_hr), device=device)
    s1 = torch.zeros((B, shapes["s1_ch"], H_hr, W_hr), device=device)
    dem = torch.zeros((B, 1, H_hr, W_hr), device=device)
    lc = torch.zeros((B, 1, H_hr, W_hr), device=device)
    era5_hr = torch.zeros((B, shapes["era5_ch"], H_hr, W_hr), device=device)
    base_hr = torch.zeros((B, 1, H_hr, W_hr), device=device)

    res_net = ResidualNet(
        s2_ch=shapes["s2_ch"],
        s1_ch=shapes["s1_ch"],
        dem_ch=1,
        lc_num_classes=shapes["lc_classes"],
        lc_one_hot=False,
        era5_ch=shapes["era5_ch"],
        base_ch=1,
    ).to(device)

    print("\n=== ResidualNet (HR) ===")
    summary(
        res_net,
        input_data=(base_hr, s2, s1, dem, lc, era5_hr, False),
        verbose=0,
        device=device,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
