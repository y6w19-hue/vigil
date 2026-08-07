from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_path, load_config

logger = logging.getLogger(__name__)

_df: pd.DataFrame | None = None
_fraud_df: pd.DataFrame | None = None
_legit_df: pd.DataFrame | None = None

V_COLS = [f"V{i}" for i in range(1, 29)]

MERCHANT_PROFILES: dict[str, dict[str, float]] = {
    "grocery": {"base_risk": 0.05, "amount_typical": 30, "amount_spread": 25},
    "restaurant": {"base_risk": 0.08, "amount_typical": 45, "amount_spread": 30},
    "gas": {"base_risk": 0.10, "amount_typical": 50, "amount_spread": 20},
    "electronics": {"base_risk": 0.45, "amount_typical": 400, "amount_spread": 300},
    "online": {"base_risk": 0.55, "amount_typical": 120, "amount_spread": 200},
    "atm": {"base_risk": 0.30, "amount_typical": 200, "amount_spread": 150},
}


def load_dataset() -> None:
    global _df, _fraud_df, _legit_df
    cfg = load_config()
    raw_path = get_path(cfg, "raw_data")
    if not raw_path.exists() and raw_path.suffix == ".csv":
        gz_path = raw_path.with_suffix(".csv.gz")
        if gz_path.exists():
            raw_path = gz_path
    if not raw_path.exists():
        processed = get_path(cfg, "processed_dir") / "train.parquet"
        if processed.exists():
            raw_path = processed
        else:
            logger.warning("No dataset found for scenario mapping")
            return

    _df = pd.read_csv(raw_path) if raw_path.suffix in (".csv", ".gz") else pd.read_parquet(raw_path)
    _fraud_df = _df[_df["Class"] == 1].copy()
    _legit_df = _df[_df["Class"] == 0].copy()
    logger.info(
        "Scenario dataset loaded: %d rows (%d fraud, %d legit)",
        len(_df),
        len(_fraud_df),
        len(_legit_df),
    )


def _compute_risk(
    merchant: str,
    amount: float,
    hour: int,
    card_present: bool,
    new_device: bool,
    foreign: bool,
) -> float:
    profile = MERCHANT_PROFILES.get(merchant, MERCHANT_PROFILES["online"])
    risk = profile["base_risk"]

    if not card_present:
        risk += 0.15
    if new_device:
        risk += 0.20
    if foreign:
        risk += 0.15

    if hour < 6 or hour >= 23:
        risk += 0.15
    elif hour < 8 or hour >= 21:
        risk += 0.05

    if amount > 500:
        risk += 0.15
    elif amount > 200:
        risk += 0.08

    return min(risk, 0.95)


def _find_closest_row(
    pool: pd.DataFrame,
    target_amount: float,
    target_hour: int,
) -> pd.Series:
    if "Time" in pool.columns:
        pool_seconds = pool["Time"].values
        target_seconds = target_hour * 3600
        time_diff = np.abs(pool_seconds - target_seconds)
        time_diff = np.minimum(time_diff, 86400 - time_diff)
        time_norm = time_diff / 43200
    else:
        time_norm = np.zeros(len(pool))

    amount_norm = np.abs(pool["Amount"].values - target_amount) / (target_amount + 1)
    distance = time_norm + amount_norm * 0.5

    idx = np.argmin(distance)
    return pool.iloc[idx]


def build_transaction(
    merchant: str,
    amount: float,
    hour: int,
    card_present: bool,
    new_device: bool,
    foreign: bool,
) -> dict[str, Any] | None:
    if _df is None:
        return None

    risk = _compute_risk(merchant, amount, hour, card_present, new_device, foreign)

    use_fraud = np.random.random() < risk
    pool = _fraud_df if use_fraud and len(_fraud_df) > 0 else _legit_df

    if len(pool) == 0:
        pool = _df

    row = _find_closest_row(pool, amount, hour)

    tx: dict[str, Any] = {}
    for v in V_COLS:
        tx[v] = float(row[v])
    tx["Time"] = float(hour * 3600)
    tx["Amount"] = float(amount)

    return {
        "transaction": tx,
        "risk_score": round(risk, 3),
        "sampled_from": "fraud" if use_fraud else "legitimate",
        "scenario": {
            "merchant": merchant,
            "amount": amount,
            "hour": hour,
            "card_present": card_present,
            "new_device": new_device,
            "foreign": foreign,
        },
    }
