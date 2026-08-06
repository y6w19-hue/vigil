from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import requests
import streamlit as st

from src.config import get_path, load_config
from src.data.load import class_balance, load_raw

_FAVICON = str(Path(__file__).parent / "favicon.png")

st.set_page_config(
    page_title="Vigil — Fraud Detection Analytics",
    layout="wide",
    page_icon=_FAVICON,
)

cfg = load_config()
figures_dir = get_path(cfg, "figures_dir")
models_dir = get_path(cfg, "models_dir")
api_url = os.environ.get("API_URL", "http://localhost:8000")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Pages",
    ["Dataset Overview", "EDA", "Model Performance", "SHAP Analysis", "Transaction Lookup"],
)


@st.cache_data
def load_data():
    return load_raw()


def show_dataset_overview():
    st.header("Dataset Overview")
    try:
        df = load_data()
    except FileNotFoundError:
        st.warning("Dataset not found. Download the ULB dataset to data/raw/creditcard.csv")
        return

    bal = class_balance(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{bal['total']:,}")
    col2.metric("Legitimate", f"{bal['legit']:,}")
    col3.metric("Fraud", f"{bal['fraud']:,}")
    col4.metric("Fraud Rate", f"{bal['fraud_pct']}%")

    st.subheader("Feature Summary")
    st.dataframe(df.describe().T, use_container_width=True)

    st.subheader("Class Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = df["Class"].value_counts()
    ax.bar(["Legitimate", "Fraud"], counts.values, color=["#2ecc71", "#e74c3c"])
    ax.set_yscale("log")
    ax.set_ylabel("Count (log)")
    st.pyplot(fig)


def show_eda():
    st.header("Exploratory Data Analysis")
    eda_dir = figures_dir / "eda"

    plots = {
        "Class Distribution": "class_distribution.png",
        "Amount & Time Distributions": "amount_time_distributions.png",
        "Correlation Heatmap": "correlation_heatmap.png",
        "Top 6 KS Features": "top6_ks_distributions.png",
    }

    for title, filename in plots.items():
        path = eda_dir / filename
        if path.exists():
            st.subheader(title)
            st.image(str(path), use_container_width=True)
        else:
            st.info(f"{title} — run notebooks/01_eda.ipynb to generate")


def show_model_performance():
    st.header("Model Performance")

    models_dir_fig = figures_dir / "models"
    if models_dir_fig.exists():
        for fig_file in sorted(models_dir_fig.glob("*.png")):
            st.subheader(fig_file.stem.replace("_", " ").title())
            st.image(str(fig_file), use_container_width=True)

    try:
        import mlflow

        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        experiment = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
        if experiment:
            runs = mlflow.search_runs([experiment.experiment_id])
            if not runs.empty:
                st.subheader("MLflow Runs")
                cols = ["tags.mlflow.runName"] + [
                    c for c in runs.columns if c.startswith("metrics.") and not c.endswith("_std")
                ]
                sorted_runs = runs[cols].sort_values("metrics.pr_auc", ascending=False)
                st.dataframe(sorted_runs, use_container_width=True)
    except Exception:
        st.info("MLflow tracking not available")


def show_shap():
    st.header("SHAP Explainability")
    shap_dir = figures_dir / "shap"

    if not shap_dir.exists():
        st.info("SHAP plots not generated yet. Run the evaluation pipeline.")
        return

    for fig_file in sorted(shap_dir.glob("*.png")):
        st.subheader(fig_file.stem.replace("_", " ").title())
        st.image(str(fig_file), use_container_width=True)


def show_transaction_lookup():
    st.header("Transaction Lookup")
    st.write("Enter transaction features to get a fraud prediction from the API.")

    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input("Time (seconds)", 0.0, 200000.0, 10000.0)
        amount_val = st.number_input("Amount", 0.0, 30000.0, 100.0)

    st.subheader("V1–V28 Features")
    v_cols = st.columns(4)
    v_values = {}
    for i in range(1, 29):
        with v_cols[(i - 1) % 4]:
            v_values[f"V{i}"] = st.number_input(f"V{i}", -100.0, 100.0, 0.0, key=f"v{i}")

    if st.button("Predict"):
        payload = {"Time": time_val, "Amount": amount_val, **v_values}
        try:
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
            result = resp.json()

            if result["is_fraud"]:
                st.error(f"FRAUD DETECTED — Probability: {result['probability']:.4f}")
            else:
                st.success(f"Legitimate — Probability: {result['probability']:.4f}")

            st.write(f"Threshold: {result['threshold']:.4f}")

            if result["top_features"]:
                st.subheader("Top SHAP Features")
                for feat in result["top_features"]:
                    direction = "+" if feat["shap_value"] > 0 else "-"
                    st.write(f"  {feat['feature']}: {direction}{abs(feat['shap_value']):.4f}")
        except requests.ConnectionError:
            st.warning("API not running. Start it with: docker compose up api")


if page == "Dataset Overview":
    show_dataset_overview()
elif page == "EDA":
    show_eda()
elif page == "Model Performance":
    show_model_performance()
elif page == "SHAP Analysis":
    show_shap()
elif page == "Transaction Lookup":
    show_transaction_lookup()
