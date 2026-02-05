from pathlib import Path

from helper.split_utils import load_or_create_splits

PROJECT_ROOT = Path("/home/naren-root/Documents/FYP2/Project")
COMMON_DATES = PROJECT_ROOT / "common_dates.csv"
SPLITS_PATH = PROJECT_ROOT / "metrics" / "common_date_splits.csv"


def main() -> None:
    splits = load_or_create_splits(COMMON_DATES, SPLITS_PATH)
    counts = {k: len(v) for k, v in splits.items()}
    print(f"saved splits: {SPLITS_PATH}")
    print(f"counts: {counts}")


if __name__ == "__main__":
    main()
