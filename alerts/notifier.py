from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from typing import Any

from src.config import get_path, load_config

logger = logging.getLogger("alerts")
logger.setLevel(logging.INFO)

_cfg = load_config()
_log_path = get_path(_cfg, "logs_dir") / "alerts.log"
_log_path.parent.mkdir(parents=True, exist_ok=True)

_handler = RotatingFileHandler(_log_path, maxBytes=10 * 1024 * 1024, backupCount=5)
_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
logger.addHandler(_handler)


def log_alert(alert: dict[str, Any]) -> None:
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "probability": alert.get("probability"),
        "threshold": alert.get("threshold"),
        "amount": alert.get("transaction", {}).get("Amount"),
        "top_features": alert.get("top_features", []),
    }
    logger.info(json.dumps(entry))
