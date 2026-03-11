from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_dataset_paths(cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    ds = cfg.get("dataset", {})
    if not isinstance(ds, dict):
        return None, None
    p_daily = ds.get("madurai_zarr") or ds.get("zarr_path")
    p_30m = ds.get("madurai_30m_zarr") or ds.get("zarr_30m_path")
    return (str(p_daily) if p_daily else None, str(p_30m) if p_30m else None)


def _exists_local(path_text: Optional[str]) -> bool:
    if not path_text:
        return False
    try:
        p = Path(str(path_text))
        return p.exists()
    except Exception:
        return False


def _to_abs_if_exists(path_text: str) -> str:
    p = Path(str(path_text))
    try:
        if p.exists():
            return str(p.resolve())
    except Exception:
        pass
    return str(p)


def discover_dataset_paths(
    *,
    repo_root: Path,
    cli_daily: Optional[str] = None,
    cli_30m: Optional[str] = None,
) -> Dict[str, str]:
    # 1) Explicit CLI override (if both are provided and valid)
    if cli_daily and cli_30m and _exists_local(cli_daily) and _exists_local(cli_30m):
        return {
            "daily": _to_abs_if_exists(cli_daily),
            "zarr_30m": _to_abs_if_exists(cli_30m),
            "source": "cli_override",
        }

    # 2) Built-in local default (Ubuntu-friendly, no user input needed)
    local_daily = repo_root / "madurai.zarr"
    local_30m = repo_root / "madurai_30m.zarr"
    if local_daily.exists() and local_30m.exists():
        return {
            "daily": str(local_daily.resolve()),
            "zarr_30m": str(local_30m.resolve()),
            "source": "repo_local_default",
        }

    # 3) Config discovery, but only accept paths that exist on this machine.
    candidates: List[Path] = [
        repo_root / "patch_restnet" / "config.yaml",
        repo_root / "patch_cnn" / "config.yaml",
    ]
    candidates.extend(sorted(repo_root.glob("config*.y*ml")))
    candidates.extend(sorted(repo_root.glob("dataset*.y*ml")))
    candidates.extend(sorted(repo_root.glob("settings*.y*ml")))

    for c in candidates:
        cfg = _read_yaml(c)
        if not cfg:
            continue
        p_daily, p_30m = _extract_dataset_paths(cfg)
        if p_daily and p_30m and _exists_local(p_daily) and _exists_local(p_30m):
            return {
                "daily": _to_abs_if_exists(p_daily),
                "zarr_30m": _to_abs_if_exists(p_30m),
                "source": f"existing_config:{c.as_posix()}",
            }

    # 4) Environment variables (if set and valid).
    env_daily = os.environ.get("DATASET_ZARR") or os.environ.get("MADURAI_ZARR")
    env_30m = os.environ.get("DATASET_30M_ZARR") or os.environ.get("MADURAI_30M_ZARR")
    if env_daily and env_30m and _exists_local(env_daily) and _exists_local(env_30m):
        return {
            "daily": _to_abs_if_exists(env_daily),
            "zarr_30m": _to_abs_if_exists(env_30m),
            "source": "env_vars",
        }

    raise RuntimeError(
        "Could not resolve dataset paths. Expected local files "
        f"`{(repo_root / 'madurai.zarr').as_posix()}` and "
        f"`{(repo_root / 'madurai_30m.zarr').as_posix()}` or valid CLI/env overrides."
    )
