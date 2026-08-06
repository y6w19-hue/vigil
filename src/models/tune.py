from __future__ import annotations

import argparse
import logging
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.config import get_path, load_config
from src.data.imbalance import get_strategy
from src.models.registry import build_lightgbm, build_xgboost
from src.models.train import split_xy

logger = logging.getLogger(__name__)

N_TRIALS = 30
CV_FOLDS = 3


def xgboost_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


def lightgbm_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
    }


def objective(
    trial: optuna.Trial,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict[str, Any],
) -> float:
    if model_name == "xgboost":
        params = xgboost_search_space(trial)
        params["eval_metric"] = "aucpr"
        params["early_stopping_rounds"] = 50
        scale_pos_weight = (y == 0).sum() / max(y.sum(), 1)
        model = build_xgboost(params, scale_pos_weight=scale_pos_weight)
    elif model_name == "lightgbm":
        params = lightgbm_search_space(trial)
        params["is_unbalance"] = True
        params["metric"] = "average_precision"
        model = build_lightgbm(params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    imbalance_cfg = cfg["imbalance"]
    seed = cfg["seed"]
    sampler = get_strategy(imbalance_cfg["strategy"], imbalance_cfg, seed)
    pipeline = ImbPipeline([("sampler", sampler), ("model", model)])

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_name == "xgboost":
            pipeline.fit(X_tr, y_tr, model__eval_set=[(X_val, y_val)], model__verbose=False)
        else:
            pipeline.fit(X_tr, y_tr, model__eval_set=[(X_val, y_val)])

        y_proba = pipeline.predict_proba(X_val)[:, 1]
        scores.append(average_precision_score(y_val, y_proba))

    return float(np.mean(scores))


def tune_model(model_name: str, cfg: dict[str, Any], n_trials: int = N_TRIALS) -> dict[str, Any]:
    processed_dir = get_path(cfg, "processed_dir")
    train = pd.read_parquet(processed_dir / "train.parquet")
    X, y = split_xy(train)

    study = optuna.create_study(direction="maximize", study_name=f"tune_{model_name}")
    study.optimize(
        lambda trial: objective(trial, model_name, X, y, cfg),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    logger.info("%s best PR-AUC (CV): %.4f", model_name, study.best_value)
    logger.info("%s best params: %s", model_name, study.best_params)

    return {
        "model": model_name,
        "best_pr_auc": study.best_value,
        "best_params": study.best_params,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "lightgbm"],
        help="Models to tune",
    )
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    trials = args.trials

    cfg = load_config(args.config)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    results = {}
    for model_name in args.models:
        logger.info("=== Tuning %s (%d trials, %d-fold CV) ===", model_name, trials, CV_FOLDS)
        result = tune_model(model_name, cfg, n_trials=trials)
        results[model_name] = result

        with mlflow.start_run(run_name=f"{model_name}_tuned"):
            mlflow.log_param("model", f"{model_name}_tuned")
            mlflow.log_param("git_commit", "tuned")
            mlflow.log_params(result["best_params"])
            mlflow.log_metric("pr_auc_cv", result["best_pr_auc"])

    logger.info("\n=== Tuning summary ===")
    for model_name, result in results.items():
        logger.info(
            "%s: PR-AUC=%.4f, params=%s",
            model_name,
            result["best_pr_auc"],
            result["best_params"],
        )


if __name__ == "__main__":
    main()
