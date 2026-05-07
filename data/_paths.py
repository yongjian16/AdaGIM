"""Resolve a dataset root, honouring the ``DATASETS_ROOT`` environment variable."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

def resolve(name: str, cfg: Dict[str, Any]) -> str:
    cfg_root = cfg.get("root")
    if cfg_root and Path(cfg_root).is_dir():
        return str(cfg_root)

    env = os.environ.get("DATASETS_ROOT")
    if env:
        return str(Path(env) / name)

    return str(_repo_root() / "datasets" / name)
