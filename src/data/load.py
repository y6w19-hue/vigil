from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import get_path, load_config

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
]


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    csv_path = Path(path) if path else get_path(cfg, "raw_data")
    if not csv_path.exists() and csv_path.suffix == ".csv":
        gz_path = csv_path.with_suffix(".csv.gz")
        if gz_path.exists():
            csv_path = gz_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place at data/raw/creditcard.csv"
        )
    df = pd.read_csv(csv_path)
    validate_schema(df)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra:
        raise ValueError(f"Unexpected columns: {extra}")
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Null values in columns: {null_cols}")
    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError(f"Class column has unexpected values: {df['Class'].unique()}")


def class_balance(df: pd.DataFrame) -> dict[str, int | float]:
    counts = df["Class"].value_counts().to_dict()
    total = len(df)
    fraud = counts.get(1, 0)
    return {
        "total": total,
        "legit": counts.get(0, 0),
        "fraud": fraud,
        "fraud_pct": round(fraud / total * 100, 4),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = load_raw()
    balance = class_balance(df)
    logger.info("Rows:           %s", f"{balance['total']:,}")
    logger.info("Legitimate:     %s", f"{balance['legit']:,}")
    logger.info("Fraud:          %s", f"{balance['fraud']:,}")
    logger.info("Fraud %%:        %s", balance["fraud_pct"])
    logger.info("Columns:        %d", len(df.columns))
    logger.info("Null values:    %d", df.isnull().sum().sum())
    logger.info("Memory (MB):    %.1f", df.memory_usage(deep=True).sum() / 1e6)


if __name__ == "__main__":
    main()
