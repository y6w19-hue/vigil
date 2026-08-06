from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "mlflow" in cfg and "tracking_uri" in cfg["mlflow"]:
        uri = cfg["mlflow"]["tracking_uri"]
        if uri.startswith("file:") and not Path(uri[5:]).is_absolute():
            cfg["mlflow"]["tracking_uri"] = f"file:{REPO_ROOT / uri[5:]}"
    return cfg


def get_path(config: dict[str, Any], key: str) -> Path:
    raw = config["paths"][key]
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p
