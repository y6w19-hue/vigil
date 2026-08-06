from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.config import get_path, load_config
from src.data.load import load_raw
from src.features.engineering import engineer

logger = logging.getLogger(__name__)


def stratified_split(
    df: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    remaining = 1.0 - test_size
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Class"],
        random_state=seed,
    )
    adj_val = val_size / remaining
    train, val = train_test_split(
        train,
        test_size=adj_val,
        stratify=train["Class"],
        random_state=seed,
    )
    return train, val, test


def fit_scaler(train: pd.DataFrame) -> RobustScaler:
    cols = ["Amount", "log_amount"]
    scaler = RobustScaler()
    scaler.fit(train[cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    out = df.copy()
    cols = ["Amount", "log_amount"]
    out[cols] = scaler.transform(out[cols])
    return out


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.parquet"
        split.to_parquet(path, index=False)
        logger.info("Saved %s (%d rows) to %s", name, len(split), path)


def run(config_path: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    val_size = cfg["split"]["val_size"]
    test_size = cfg["split"]["test_size"]
    out_dir = get_path(cfg, "processed_dir")

    df = load_raw()
    df = engineer(df)

    train, val, test = stratified_split(df, val_size, test_size, seed)

    scaler = fit_scaler(train)
    train = apply_scaler(train, scaler)
    val = apply_scaler(val, scaler)
    test = apply_scaler(test, scaler)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        fraud = split["Class"].sum()
        logger.info(
            "%s: %d rows, %d fraud (%.4f%%)",
            name,
            len(split),
            fraud,
            fraud / len(split) * 100,
        )

    save_splits(train, val, test, out_dir)
