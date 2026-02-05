from pathlib import Path
import csv
import math

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
BASE_DIR = PROJECT_ROOT / "metrics" / "deep_baselines" / "cnn"
OUT_PATH = PROJECT_ROOT / "metrics" / "cnn_group_summary.csv"

GROUPS = {
    "era5_meteorology_modis": "ERA5 meteorology + MODIS LST",
    "era5_meteorology_viirs": "ERA5 meteorology + VIIRS LST",
    "modis_lst": "MODIS LST",
    "viirs_lst": "VIIRS LST",
    "vegetation_indices_modis": "Vegetation indices (S2) + MODIS LST",
    "vegetation_indices_viirs": "Vegetation indices (S2) + VIIRS LST",
    "builtup_proxies_modis": "Built-up proxies (S1 + world + dyn) + MODIS LST",
    "builtup_proxies_viirs": "Built-up proxies (S1 + world + dyn) + VIIRS LST",
}


def mean(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


rows = []
for run_name, label in GROUPS.items():
    metrics_path = BASE_DIR / run_name / "cnn_eval_metrics.csv"
    if not metrics_path.exists():
        rows.append(
            {
                "group": label,
                "run_name": run_name,
                "rmse_mean": "",
                "rmse_std": "",
                "n_rows": "",
                "file": str(metrics_path),
                "note": "missing",
            }
        )
        continue
    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rmses = []
        for r in reader:
            try:
                rmses.append(float(r.get("rmse", "")))
            except Exception:
                pass
    rmse_mean = mean(rmses)
    rmse_std = float("nan")
    if rmses:
        mu = rmse_mean
        rmse_std = math.sqrt(sum((x - mu) ** 2 for x in rmses) / len(rmses))
    rows.append(
        {
            "group": label,
            "run_name": run_name,
            "rmse_mean": rmse_mean if math.isfinite(rmse_mean) else "",
            "rmse_std": rmse_std if math.isfinite(rmse_std) else "",
            "n_rows": len(rmses),
            "file": str(metrics_path),
            "note": "",
        }
    )

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["group", "run_name", "rmse_mean", "rmse_std", "n_rows", "file", "note"],
    )
    w.writeheader()
    w.writerows(rows)

print(OUT_PATH)
