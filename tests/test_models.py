from __future__ import annotations

import numpy as np

from src.evaluation.metrics import compute_metrics, format_metrics
from src.models.registry import build_lightgbm, build_logistic, build_random_forest, build_xgboost


def _make_classification_data(n=500, fraud_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_frac)
    n_legit = n - n_fraud
    X = rng.standard_normal((n, 5))
    X[:n_legit, 0] += 2
    X[n_legit:, 0] -= 2
    y = np.array([0] * n_legit + [1] * n_fraud)
    return X, y


def test_compute_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_proba = np.array([0.1, 0.6, 0.9, 0.8, 0.2, 0.4])
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 2
    assert 0 <= m["pr_auc"] <= 1
    assert 0 <= m["roc_auc"] <= 1
    assert 0 <= m["f1"] <= 1


def test_format_metrics():
    m = compute_metrics([0, 1], [0, 1], [0.1, 0.9])
    s = format_metrics(m)
    assert "PR-AUC" in s
    assert "TP:" in s


def test_build_logistic():
    model = build_logistic({"C": 1.0, "class_weight": "balanced"})
    X, y = _make_classification_data()
    model.fit(X, y)
    assert model.predict(X).shape == (len(y),)


def test_build_random_forest():
    model = build_random_forest({"n_estimators": 10})
    X, y = _make_classification_data()
    model.fit(X, y)
    assert model.predict(X).shape == (len(y),)


def test_build_xgboost():
    model = build_xgboost({"n_estimators": 10, "early_stopping_rounds": 5}, scale_pos_weight=9.0)
    X, y = _make_classification_data()
    model.fit(X, y, eval_set=[(X, y)], verbose=False)
    assert model.predict(X).shape == (len(y),)


def test_build_lightgbm():
    model = build_lightgbm({"n_estimators": 10, "is_unbalance": True})
    X, y = _make_classification_data()
    model.fit(X, y)
    assert model.predict(X).shape == (len(y),)
