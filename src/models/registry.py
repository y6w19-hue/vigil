from __future__ import annotations

from typing import Any

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_logistic(config: dict[str, Any]) -> LogisticRegression:
    return LogisticRegression(
        C=config.get("C", 1.0),
        penalty=config.get("penalty", "l2"),
        solver=config.get("solver", "lbfgs"),
        max_iter=config.get("max_iter", 1000),
        class_weight=config.get("class_weight", "balanced"),
        random_state=42,
        n_jobs=-1,
    )


def build_random_forest(config: dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.get("n_estimators", 300),
        max_depth=config.get("max_depth"),
        min_samples_leaf=config.get("min_samples_leaf", 2),
        class_weight=config.get("class_weight", "balanced_subsample"),
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )


def build_xgboost(config: dict[str, Any], scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=config.get("n_estimators", 300),
        max_depth=config.get("max_depth", 6),
        learning_rate=config.get("learning_rate", 0.1),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        min_child_weight=config.get("min_child_weight", 1),
        eval_metric=config.get("eval_metric", "aucpr"),
        early_stopping_rounds=config.get("early_stopping_rounds", 50),
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def build_lightgbm(config: dict[str, Any], scale_pos_weight: float = 1.0) -> lgb.LGBMClassifier:
    params = dict(
        n_estimators=config.get("n_estimators", 300),
        max_depth=config.get("max_depth", -1),
        learning_rate=config.get("learning_rate", 0.1),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if config.get("is_unbalance"):
        params["is_unbalance"] = True
    else:
        params["scale_pos_weight"] = scale_pos_weight
    return lgb.LGBMClassifier(**params)


REGISTRY = {
    "logistic": build_logistic,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
}
