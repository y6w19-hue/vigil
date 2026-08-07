from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

from src.config import get_path, load_config
from src.data.load import class_balance, load_raw

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate_eda_figures() -> None:
    cfg = load_config()
    eda_dir = get_path(cfg, "figures_dir") / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    v_cols = [f"V{i}" for i in range(1, 29)]
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Class"].value_counts()
    ax.bar(["Legitimate", "Fraud"], counts.values, color=["#2ecc71", "#e74c3c"])
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Class Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v * 1.5, f"{v:,}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(eda_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved class_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in zip(axes, ["Amount", "Time"]):
        for cls, label, color in [(0, "Legitimate", "#2ecc71"), (1, "Fraud", "#e74c3c")]:
            data = df.loc[df["Class"] == cls, col]
            ax.hist(data, bins=50, alpha=0.5, label=label, color=color, density=True)
        ax.set_title(f"{col} Distribution by Class")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
    axes[0].set_xlim(0, df["Amount"].quantile(0.99))
    plt.tight_layout()
    plt.savefig(eda_dir / "amount_time_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved amount_time_distributions.png")

    corr = df[v_cols + ["Amount", "Class"]].corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True,
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(eda_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved correlation_heatmap.png")

    ks_results = []
    for col in v_cols + ["Amount", "Time"]:
        legit = df.loc[df["Class"] == 0, col]
        fraud = df.loc[df["Class"] == 1, col]
        stat, _ = ks_2samp(legit, fraud)
        ks_results.append({"feature": col, "ks_stat": stat})
    ks_df = pd.DataFrame(ks_results).sort_values("ks_stat", ascending=False).reset_index(drop=True)

    top6 = ks_df["feature"].head(6).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, col in zip(axes.ravel(), top6):
        for cls, label, color in [(0, "Legitimate", "#2ecc71"), (1, "Fraud", "#e74c3c")]:
            data = df.loc[df["Class"] == cls, col]
            ax.hist(data, bins=50, alpha=0.5, label=label, color=color, density=True)
        ks_val = ks_df.loc[ks_df["feature"] == col, "ks_stat"].values[0]
        ax.set_title(f"{col} (KS={ks_val:.3f})")
        ax.legend(fontsize=8)
    plt.suptitle("Top 6 Features by KS-Statistic", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(eda_dir / "top6_ks_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved top6_ks_distributions.png")


def generate_model_figures() -> None:
    cfg = load_config()
    models_fig_dir = get_path(cfg, "figures_dir") / "models"
    models_fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
        from sklearn.metrics import precision_recall_curve, roc_curve, auc

        processed_dir = get_path(cfg, "processed_dir")
        test_path = processed_dir / "test.parquet"
        if not test_path.exists():
            logger.warning("Test data not found — skipping model figures")
            return

        test_df = pd.read_parquet(test_path)
        X_test = test_df.drop(columns=["Class"])
        y_test = test_df["Class"]

        best_path = get_path(cfg, "models_dir") / "best.joblib"
        if not best_path.exists():
            logger.warning("best.joblib not found — skipping model figures")
            return

        model = joblib.load(best_path)
        y_proba = model.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color="#e74c3c", linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve (Best Model)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(models_fig_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved pr_curve.png")

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="#3498db", linewidth=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (Best Model)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(models_fig_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved roc_curve.png")

    except Exception as e:
        logger.warning("Model figures failed: %s", e)


def generate_shap_figures() -> None:
    cfg = load_config()
    shap_dir = get_path(cfg, "figures_dir") / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    try:
        import joblib
        import shap

        processed_dir = get_path(cfg, "processed_dir")
        test_path = processed_dir / "test.parquet"
        best_path = get_path(cfg, "models_dir") / "best.joblib"
        if not test_path.exists() or not best_path.exists():
            logger.warning("Test data or model not found — skipping SHAP figures")
            return

        test_df = pd.read_parquet(test_path)
        X_test = test_df.drop(columns=["Class"])

        model = joblib.load(best_path)
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            estimator = model.named_steps["model"]
        elif hasattr(model, "steps"):
            estimator = model.steps[-1][1]
        else:
            estimator = model

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_test)

        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(shap_dir / "summary_plot.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved summary_plot.png")

        feature_names = X_test.columns.tolist()
        top_features = np.argsort(np.abs(shap_values).mean(0))[-6:][::-1]
        for idx in top_features:
            feat = feature_names[idx]
            plt.figure()
            shap.dependence_plot(idx, shap_values, X_test, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(shap_dir / f"dependence_{feat}.png", dpi=150, bbox_inches="tight")
            plt.close()
        logger.info("Saved dependence plots")

    except Exception as e:
        logger.warning("SHAP figures failed: %s", e)


def main() -> None:
    logger.info("Generating EDA figures...")
    generate_eda_figures()
    logger.info("Generating model figures...")
    generate_model_figures()
    logger.info("Generating SHAP figures...")
    generate_shap_figures()
    logger.info("All figures generated.")


if __name__ == "__main__":
    main()
