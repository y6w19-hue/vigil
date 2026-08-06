from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: Any, X: Any, model_type: str = "tree", feature_names: list[str] | None = None
) -> shap.Explanation:
    if feature_names is None and hasattr(X, "columns"):
        feature_names = list(X.columns)
    X_arr = np.asarray(X)

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_arr)
    elif model_type == "deep":
        explainer = shap.DeepExplainer(model, X_arr[:200])
        shap_values = explainer(X_arr)
    elif model_type == "kernel":
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_arr, 100))
        shap_values = explainer.shap_values(X_arr)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if isinstance(shap_values, shap.Explanation) and feature_names is not None:
        shap_values.feature_names = feature_names

    return shap_values


def summary_plot(
    shap_values: Any,
    X: Any,
    feature_names: list[str],
    out_path: str | Path,
) -> None:
    plt.figure(figsize=(10, 8))
    if isinstance(shap_values, shap.Explanation):
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
    else:
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(sv, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def dependence_plot(
    shap_values: Any,
    X: Any,
    feature_names: list[str],
    feature: str,
    out_path: str | Path,
) -> None:
    idx = feature_names.index(feature)
    plt.figure(figsize=(8, 6))
    if isinstance(shap_values, shap.Explanation):
        shap.plots.scatter(shap_values[:, idx], color=shap_values, show=False)
    else:
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.dependence_plot(idx, sv, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def top_features_for_prediction(
    shap_values: Any, feature_names: list[str], idx: int, n: int = 5
) -> list[dict[str, float | str]]:
    if isinstance(shap_values, shap.Explanation):
        vals = shap_values[idx].values
    else:
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        vals = sv[idx]

    abs_vals = np.abs(vals)
    top_idx = np.argsort(abs_vals)[-n:][::-1]

    return [{"feature": feature_names[i], "shap_value": float(vals[i])} for i in top_idx]


def main() -> None:
    import argparse
    import logging

    import joblib
    import pandas as pd

    from src.config import get_path, load_config

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="SHAP analysis on best model")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default="best")
    args = parser.parse_args()

    cfg = load_config(args.config)
    models_dir = get_path(cfg, "models_dir")
    processed_dir = get_path(cfg, "processed_dir")
    figures_dir = get_path(cfg, "figures_dir") / "shap"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(models_dir / f"{args.model}.joblib")
    if not (models_dir / f"{args.model}.joblib").exists():
        model = joblib.load(models_dir / "best.joblib")

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator = model.named_steps["model"]
    elif hasattr(model, "steps"):
        estimator = model.steps[-1][1]
    else:
        estimator = model

    test = pd.read_parquet(processed_dir / "test.parquet")
    X_test = test.drop(columns=["Class"])
    feature_names = list(X_test.columns)

    shap_values = compute_shap_values(
        estimator, X_test, model_type="tree", feature_names=feature_names
    )

    summary_plot(shap_values, X_test, feature_names, figures_dir / "summary_plot.png")
    logging.info("Saved SHAP summary plot")

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top6_idx = np.argsort(mean_abs)[-6:][::-1]
    for i in top6_idx:
        feat = feature_names[i]
        dependence_plot(
            shap_values, X_test, feature_names, feat, figures_dir / f"dependence_{feat}.png"
        )
        logging.info("Saved dependence plot for %s", feat)


if __name__ == "__main__":
    main()
