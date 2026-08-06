from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def error_breakdown(df: pd.DataFrame, y_true: Any, y_pred: Any) -> dict[str, pd.DataFrame]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    tp_mask = (y_pred == 1) & (y_true == 1)

    return {
        "false_positives": df[fp_mask].copy(),
        "false_negatives": df[fn_mask].copy(),
        "true_positives": df[tp_mask].copy(),
    }


def profile_errors(
    errors_df: pd.DataFrame, label: str, cols: list[str] | None = None
) -> dict[str, dict[str, float]]:
    if cols is None:
        cols = ["Amount", "log_amount", "hour_sin", "hour_cos"]
        v_cols = [c for c in errors_df.columns if c.startswith("V")]
        cols = cols + v_cols[:5]

    profile = {}
    for col in cols:
        if col in errors_df.columns:
            profile[col] = {
                "mean": float(errors_df[col].mean()),
                "median": float(errors_df[col].median()),
                "std": float(errors_df[col].std()),
                "min": float(errors_df[col].min()),
                "max": float(errors_df[col].max()),
            }
    return {label: profile}


def plot_error_amounts(
    fp: pd.DataFrame,
    fn: pd.DataFrame,
    tp: pd.DataFrame,
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    data = [
        tp.get("Amount", pd.Series([0])).values,
        fp.get("Amount", pd.Series([0])).values,
        fn.get("Amount", pd.Series([0])).values,
    ]
    labels = [
        f"True Positives\n(n={len(tp)})",
        f"False Positives\n(n={len(fp)})",
        f"False Negatives\n(n={len(fn)})",
    ]

    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_ylabel("Amount")
    ax.set_title("Transaction Amounts by Error Type")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def mcnemar_test(y_true: Any, y_pred_a: Any, y_pred_b: Any) -> dict[str, float]:
    from statsmodels.stats.contingency_tables import mcnemar

    y_true = np.asarray(y_true).ravel()
    a = np.asarray(y_pred_a).ravel()
    b = np.asarray(y_pred_b).ravel()

    correct_a = (a == y_true).astype(int)
    correct_b = (b == y_true).astype(int)

    b01 = ((correct_a == 0) & (correct_b == 1)).sum()
    b10 = ((correct_a == 1) & (correct_b == 0)).sum()

    table = [[0, b10], [b01, 0]]
    result = mcnemar(table, exact=True)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def main() -> None:
    import argparse
    import json
    import logging

    import joblib
    import pandas as pd

    from src.config import get_path, load_config

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Error analysis on best model")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default="best")
    args = parser.parse_args()

    cfg = load_config(args.config)
    models_dir = get_path(cfg, "models_dir")
    processed_dir = get_path(cfg, "processed_dir")
    figures_dir = get_path(cfg, "figures_dir") / "error_analysis"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(models_dir / f"{args.model}.joblib")
    if not (models_dir / f"{args.model}.joblib").exists():
        model = joblib.load(models_dir / "best.joblib")

    test = pd.read_parquet(processed_dir / "test.parquet")
    X_test = test.drop(columns=["Class"])
    y_test = test["Class"]

    with open(models_dir / "threshold.json") as f:
        threshold = json.load(f)["threshold"]

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= threshold).astype(int)

    errors = error_breakdown(test, y_test, y_pred)
    logging.info("False positives: %d", len(errors["false_positives"]))
    logging.info("False negatives: %d", len(errors["false_negatives"]))
    logging.info("True positives:  %d", len(errors["true_positives"]))

    plot_error_amounts(
        errors["true_positives"],
        errors["false_positives"],
        errors["false_negatives"],
        figures_dir / "amounts_by_error_type.png",
    )
    logging.info("Saved error amount plot")

    fp_profile = profile_errors(errors["false_positives"], "false_positives")
    fn_profile = profile_errors(errors["false_negatives"], "false_negatives")
    logging.info("FP profile: %s", json.dumps(fp_profile, indent=2))
    logging.info("FN profile: %s", json.dumps(fn_profile, indent=2))


if __name__ == "__main__":
    main()
