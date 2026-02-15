from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

from helper import eval_utils

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
ROOT_30M = PROJECT_ROOT / "madurai_30m.zarr"


def _tight_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return y0, y1, x0, x1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-npy", required=True, help="Input .npy file")
    ap.add_argument("--out-npy", default=None, help="Output .npy file (default: add _roi)")
    ap.add_argument("--tight", action="store_true", help="Crop to tight ROI bounding box")
    ap.add_argument("--out-mask", default=None, help="Output .npy for ROI mask")
    ap.add_argument("--mask-only", action="store_true", help="Only save ROI mask, skip masked array")
    args = ap.parse_args()

    in_path = Path(args.in_npy)
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    arr = np.load(in_path)
    if arr.ndim != 2:
        raise SystemExit(f"expected 2D array, got shape={arr.shape}")

    roi_mask = eval_utils.build_roi_mask(ROOT_30M, arr.shape)
    if roi_mask is None:
        raise SystemExit("ROI mask not available for given shape")

    masked = np.where(roi_mask, arr, np.nan)

    out_path = Path(args.out_npy) if args.out_npy else in_path.with_name(f"{in_path.stem}_roi{in_path.suffix}")
    if args.tight:
        bbox = _tight_bbox(roi_mask)
        if bbox is None:
            raise SystemExit("ROI mask is empty; cannot crop")
        y0, y1, x0, x1 = bbox
        masked = masked[y0:y1, x0:x1]
        roi_mask = roi_mask[y0:y1, x0:x1]

    if not args.mask_only:
        np.save(out_path, masked)
    out_mask = Path(args.out_mask) if args.out_mask else out_path.with_name(f"{out_path.stem}_mask.npy")
    np.save(out_mask, roi_mask.astype(np.uint8))
    if not args.mask_only:
        print(f"saved {out_path} shape={masked.shape}")
    print(f"saved {out_mask} shape={roi_mask.shape}")


if __name__ == "__main__":
    main()
