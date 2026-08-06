from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import mlflow

from src.config import get_path, load_config

logger = logging.getLogger(__name__)


def select_best(config_path: str | Path | None = None) -> str:
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    exp = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment_name"])

    if exp is None:
        raise RuntimeError("MLflow experiment not found. Train models first.")

    runs = mlflow.search_runs([exp.experiment_id])
    if runs.empty:
        raise RuntimeError("No MLflow runs found. Train models first.")

    metric_col = "metrics.pr_auc"
    if metric_col not in runs.columns:
        raise RuntimeError(f"{metric_col} not found in MLflow runs.")

    best = runs.loc[runs[metric_col].idxmax()]
    best_name = best["tags.mlflow.runName"]
    pr_auc = best[metric_col]

    models_dir = get_path(cfg, "models_dir")
    src = models_dir / f"{best_name}.joblib"
    if not src.exists():
        src = models_dir / f"{best_name}.keras"
    if not src.exists():
        raise FileNotFoundError(f"Model artifact not found: {src}")

    dst = models_dir / "best.joblib" if src.suffix == ".joblib" else models_dir / "best.keras"
    shutil.copy2(src, dst)
    logger.info("Best model: %s (PR-AUC=%.4f). Copied to %s", best_name, pr_auc, dst)

    return best_name


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Select best model by validation PR-AUC")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    select_best(args.config)


if __name__ == "__main__":
    main()
