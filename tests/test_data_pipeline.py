from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.imbalance import class_weight_dict, get_strategy, smote_sampler
from src.data.load import class_balance, validate_schema
from src.data.preprocess import apply_scaler, fit_scaler, stratified_split
from src.features.engineering import engineer


def _make_synthetic(n: int = 1000, fraud_frac: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_fraud = int(n * fraud_frac)
    n_legit = n - n_fraud
    data = {
        "Time": rng.uniform(0, 172800, n),
        "Amount": rng.lognormal(3, 1.5, n),
        "Class": np.array([0] * n_legit + [1] * n_fraud),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n)
    return pd.DataFrame(data)


def test_validate_schema_passes():
    df = _make_synthetic()
    validate_schema(df)


def test_validate_schema_missing_column():
    df = _make_synthetic().drop(columns=["V1"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(df)


def test_class_balance():
    df = _make_synthetic(n=1000, fraud_frac=0.05)
    bal = class_balance(df)
    assert bal["total"] == 1000
    assert bal["fraud"] == 50
    assert abs(bal["fraud_pct"] - 5.0) < 0.1


def test_stratified_split_preserves_ratio():
    df = _make_synthetic(n=2000, fraud_frac=0.05)
    train, val, test = stratified_split(df, val_size=0.15, test_size=0.15, seed=42)
    assert len(train) + len(val) + len(test) == 2000
    for split in [train, val, test]:
        ratio = split["Class"].mean()
        assert abs(ratio - 0.05) < 0.02


def test_fit_apply_scaler_preserves_shape():
    df = engineer(_make_synthetic())
    scaler = fit_scaler(df)
    scaled = apply_scaler(df, scaler)
    assert scaled.shape == df.shape
    assert "Amount" in scaled.columns


def test_engineer_adds_features_drops_time():
    df = _make_synthetic()
    out = engineer(df)
    assert "Time" not in out.columns
    assert "hour_sin" in out.columns
    assert "hour_cos" in out.columns
    assert "log_amount" in out.columns
    assert out["hour_sin"].between(-1, 1).all()
    assert out["hour_cos"].between(-1, 1).all()


def test_smote_sampler():
    df = _make_synthetic(n=500, fraud_frac=0.05)
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    sampler = smote_sampler(sampling_strategy=0.5, seed=42)
    X_res, y_res = sampler.fit_resample(X, y)
    assert y_res.sum() > y.sum()


def test_class_weight_dict():
    y = np.array([0] * 950 + [1] * 50)
    weights = class_weight_dict(y)
    assert weights[1] > weights[0]


def test_get_strategy_unknown():
    with pytest.raises(ValueError, match="Unknown imbalance strategy"):
        get_strategy("bogus", {}, seed=42)


def test_get_strategy_returns_none_for_class_weight():
    assert get_strategy("class_weight", {}, seed=42) is None
    assert get_strategy("none", {}, seed=42) is None
