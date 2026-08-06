from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

from src.config import get_path, load_config
from src.data.imbalance import get_strategy
from src.data.preprocess import run as preprocess_run
from src.evaluation.metrics import compute_metrics, format_metrics
from src.models.nn import build_ann, build_autoencoder
from src.models.registry import REGISTRY

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def load_splits(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    processed_dir = get_path(cfg, "processed_dir")
    splits = {}
    for name in ("train", "val", "test"):
        path = processed_dir / f"{name}.parquet"
        if not path.exists():
            logger.info("Processed splits not found, running preprocessing...")
            preprocess_run()
        splits[name] = pd.read_parquet(processed_dir / f"{name}.parquet")
    return splits


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["Class"]), df["Class"]


def calc_scale_pos_weight(y: pd.Series) -> float:
    counts = y.value_counts()
    return float(counts[0] / counts[1])


def train_sklearn_model(
    model_name: str,
    cfg: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[Any, dict[str, float]]:
    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)

    model_cfg = cfg["models"][model_name]
    imbalance_cfg = cfg["imbalance"]
    seed = cfg["seed"]

    scale_pos_weight = calc_scale_pos_weight(y_train)
    builder = REGISTRY[model_name]

    if model_name in ("xgboost", "lightgbm"):
        model = builder(model_cfg, scale_pos_weight=scale_pos_weight)
    else:
        model = builder(model_cfg)

    strategy_name = imbalance_cfg["strategy"]
    sampler = get_strategy(strategy_name, imbalance_cfg, seed)

    if model_name == "logistic":
        steps = [("scaler", StandardScaler()), ("model", model)]
        pipeline = ImbPipeline(steps)
    elif sampler is not None:
        pipeline = ImbPipeline([("sampler", sampler), ("model", model)])
    else:
        pipeline = model

    if model_name == "xgboost":
        pipeline.fit(
            X_train,
            y_train,
            model__eval_set=[(X_val, y_val)],
            model__verbose=False,
        )
    elif model_name == "lightgbm":
        pipeline.fit(
            X_train,
            y_train,
            model__eval_set=[(X_val, y_val)],
        )
    else:
        pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_val, y_pred, y_proba)
    return pipeline, metrics


def train_ann(
    cfg: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[Any, dict[str, float]]:
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    ann_cfg = cfg["models"]["ann"]
    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    output_bias = np.log([pos / neg])[0]

    model = build_ann(ann_cfg, n_features=X_train_scaled.shape[1], output_bias=output_bias)

    class_weight = {0: 1.0, 1: neg / pos}
    callbacks = [
        keras_callbacks(ann_cfg.get("patience", 10)),
    ]

    model.fit(
        X_train_scaled,
        y_train,
        batch_size=ann_cfg.get("batch_size", 256),
        epochs=ann_cfg.get("epochs", 100),
        validation_data=(X_val_scaled, y_val),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )

    y_proba = model.predict(X_val_scaled, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_val, y_pred, y_proba)

    model.scaler = scaler
    return model, metrics


def keras_callbacks(patience: int):
    from tensorflow import keras

    return keras.callbacks.EarlyStopping(
        monitor="val_prc",
        mode="max",
        patience=patience,
        restore_best_weights=True,
    )


def train_autoencoder(
    cfg: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[Any, dict[str, float]]:
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_legit = X_train_scaled[y_train == 0]

    ae_cfg = cfg["models"].get("autoencoder", {"encoding_dim": 16})
    model = build_autoencoder(ae_cfg, n_features=X_train_scaled.shape[1])

    model.fit(
        X_train_legit,
        X_train_legit,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        shuffle=True,
        verbose=0,
    )

    recon_val = model.predict(X_val_scaled, verbose=0)
    recon_err = np.mean((X_val_scaled - recon_val) ** 2, axis=1)

    threshold = np.percentile(recon_err[y_val == 0], 95)
    y_pred = (recon_err >= threshold).astype(int)
    y_proba = recon_err / recon_err.max()

    metrics = compute_metrics(y_val, y_pred, y_proba)
    model.scaler = scaler
    model.threshold = float(threshold)
    return model, metrics


def save_model(model: Any, model_name: str, cfg: dict[str, Any]) -> Path:
    models_dir = get_path(cfg, "models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    if model_name in ("ann", "autoencoder"):
        path = models_dir / f"{model_name}.keras"
        model.save(path)
    else:
        path = models_dir / f"{model_name}.joblib"
        joblib.dump(model, path)

    logger.info("Saved %s to %s", model_name, path)
    return path


def run(model_name: str, config_path: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    splits = load_splits(cfg)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("git_commit", get_git_commit())
        mlflow.log_param("seed", cfg["seed"])

        if model_name == "ann":
            model, metrics = train_ann(cfg, splits["train"], splits["val"])
        elif model_name == "autoencoder":
            model, metrics = train_autoencoder(cfg, splits["train"], splits["val"])
        else:
            model, metrics = train_sklearn_model(model_name, cfg, splits["train"], splits["val"])

        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        save_model(model, model_name, cfg)

        logger.info("=== %s validation metrics ===", model_name)
        logger.info("\n%s", format_metrics(metrics))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fraud detection model")
    all_models = list(REGISTRY.keys()) + ["ann", "autoencoder"]
    parser.add_argument("--model", required=True, choices=all_models)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.model, args.config)


if __name__ == "__main__":
    main()
