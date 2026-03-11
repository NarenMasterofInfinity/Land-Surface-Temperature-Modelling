from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config: {path}")
    return data


def save_yaml(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def ensure_run_dirs(run_dir: Path) -> Dict[str, Path]:
    out = {
        "run_dir": run_dir,
        "logs": run_dir / "logs",
        "checkpoints": run_dir / "checkpoints",
        "results": run_dir / "results",
        "predictions_sample": run_dir / "results" / "predictions_sample",
        "tensorboard": run_dir / "tensorboard",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out

