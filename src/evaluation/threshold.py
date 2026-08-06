from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def sweep_thresholds(y_true: Any, y_proba: Any, n_points: int = 200) -> dict[str, np.ndarray]:
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    thresholds = np.linspace(0.01, 0.99, n_points)

    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    return {
        "thresholds": thresholds,
        "precision": np.array(precisions),
        "recall": np.array(recalls),
        "f1": np.array(f1s),
    }


def best_f1_threshold(sweep: dict[str, np.ndarray]) -> float:
    idx = np.argmax(sweep["f1"])
    return float(sweep["thresholds"][idx])


def cost_weighted_threshold(
    y_true: Any,
    y_proba: Any,
    cost_fp: float = 1.0,
    cost_fn: float = 50.0,
    n_points: int = 200,
) -> float:
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    thresholds = np.linspace(0.01, 0.99, n_points)

    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        costs.append(cost_fp * fp + cost_fn * fn)

    idx = np.argmin(costs)
    return float(thresholds[idx])


def save_threshold(threshold: float, policy: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"threshold": threshold, "policy": policy}, f, indent=2)


def load_threshold(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    import argparse
    import logging

    import joblib
    import pandas as pd

    from src.config import get_path, load_config

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Tune classification threshold")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default="best", help="Model name or 'best'")
    args = parser.parse_args()

    cfg = load_config(args.config)
    models_dir = get_path(cfg, "models_dir")
    processed_dir = get_path(cfg, "processed_dir")

    model_path = models_dir / f"{args.model}.joblib"
    if not model_path.exists():
        model_path = models_dir / "best.joblib"
    model = joblib.load(model_path)

    val = pd.read_parquet(processed_dir / "val.parquet")
    X_val = val.drop(columns=["Class"])
    y_val = val["Class"]

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_proba = model.predict(X_val).ravel()

    sweep = sweep_thresholds(y_val, y_proba)
    f1_thresh = best_f1_threshold(sweep)
    cost_thresh = cost_weighted_threshold(y_val, y_proba, cost_fp=1, cost_fn=50)

    logging.info("F1-maximising threshold: %.4f", f1_thresh)
    logging.info("Cost-weighted threshold: %.4f", cost_thresh)

    save_threshold(cost_thresh, "cost_weighted_1_50", models_dir / "threshold.json")
    logging.info("Saved cost-weighted threshold to %s", models_dir / "threshold.json")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(sweep["thresholds"], sweep["f1"], color="#3b82f6")
    axes[0].axvline(f1_thresh, color="#ef4444", linestyle="--", label=f"F1 max @ {f1_thresh:.3f}")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("F1 vs Threshold")
    axes[0].legend()

    costs = []
    for t in sweep["thresholds"]:
        yp = (y_proba >= t).astype(int)
        fp = ((yp == 1) & (y_val == 0)).sum()
        fn = ((yp == 0) & (y_val == 1)).sum()
        costs.append(fp + 50 * fn)
    axes[1].plot(sweep["thresholds"], costs, color="#f59e0b")
    axes[1].axvline(
        cost_thresh, color="#ef4444", linestyle="--", label=f"Cost min @ {cost_thresh:.3f}"
    )
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Total Cost (FP + 50*FN)")
    axes[1].set_title("Cost vs Threshold")
    axes[1].legend()

    plt.tight_layout()
    fig_path = get_path(cfg, "figures_dir") / "threshold" / "threshold_tuning.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Saved threshold plot to %s", fig_path)


if __name__ == "__main__":
    main()
