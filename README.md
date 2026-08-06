# Vigil — Credit Card Fraud Detection

AI-powered credit card fraud detection and real-time alert system. MSc AI &
Data Science final-year project (CSC-44120).

## Overview

- **6 ML models compared:** Logistic Regression, Random Forest, XGBoost,
  LightGBM, ANN (Keras), Autoencoder
- **Best model:** XGBoost (PR-AUC=0.8742), cost-weighted threshold (0.0937)
- **Class imbalance handling:** SMOTE on training fold only
- **Evaluation:** PR-AUC (primary), ROC-AUC, precision, recall, F1, confusion
  matrix, McNemar's test
- **Explainability:** SHAP (TreeExplainer) — per-prediction feature
  contributions shown in the dashboard
- **Real-time alert dashboard (Vigil):** React 18 + Vite + TS with daisyUI 5,
  Lineicons, sonner toasts, Recharts, Web Audio API alert sound. Ably
  pub/sub for live fraud alerts. Transaction detail modal with SHAP bars.
  Scenario builder with visual pipeline diagram.
- **Analytics dashboard:** Streamlit (EDA, model performance, SHAP,
  transaction lookup)
- **Backend:** FastAPI with SQLite transaction store, 7 REST endpoints,
  Ably publishing

## Quick start

### 1. Environment

The project uses a Docker dev container (Python 3.12) because the system
Python (3.14) is incompatible with TensorFlow. No system packages are
modified.

```bash
docker compose --profile dev build dev
```

Run any Python command in the dev container:

```bash
docker compose --profile dev run --rm dev <command>
```

### 2. Dataset

Download the ULB Kaggle dataset:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

Or download manually from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place
`creditcard.csv` in `data/raw/`.

### 3. Full pipeline (one command)

```bash
docker compose --profile dev run --rm dev bash scripts/reproduce.sh
```

This runs preprocessing, trains all 6 models, tunes thresholds, generates
SHAP + error analysis figures, and runs lint + tests.

### 4. Individual steps

**Preprocessing:**
```bash
docker compose --profile dev run --rm dev python -m src.data.preprocess
```

**Train a model:**
```bash
docker compose --profile dev run --rm dev python -m src.models.train --model xgboost
```

**Run notebooks:**
```bash
docker compose --profile dev run --rm --service-ports dev jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
```

Notebooks in `notebooks/` run in numbered order:
1. `01_eda.ipynb` — exploratory data analysis
2. `02_preprocessing.ipynb` — split, scale, feature engineering, resampling
3. `03_model_comparison.ipynb` — model comparison table + PR curves
4. `04_evaluation.ipynb` — final evaluation, threshold tuning, SHAP, error analysis

### 5. Dashboards

**Analytics (Streamlit):**
```bash
docker compose --profile dev run --rm --service-ports dev streamlit run dashboard/app.py
```

**Real-time alerts (React):**
```bash
cd web && bun install && bun run dev
```

Start the API + simulator:
```bash
docker compose up -d api
docker compose --profile dev run --rm dev python -m alerts.simulator --rate 5
```

### 6. Full Docker stack

```bash
docker compose up --build
```

- API: http://localhost:8000/docs
- Streamlit: http://localhost:8501
- React: http://localhost:3000

### 7. Tests + lint

```bash
docker compose --profile dev run --rm dev pytest -q
docker compose --profile dev run --rm dev ruff check . && black --check . && isort --check-only .
```

## Project structure

```
vigil/
├── configs/config.yaml        central config (seeds, paths, model params)
├── data/
│   ├── raw/                   gitignored — place creditcard.csv here
│   ├── processed/             gitignored — train/val/test parquet splits
│   └── fraud_alerts.db        SQLite transaction store (runtime)
├── notebooks/                 01_eda, 02_preprocessing, 03_comparison, 04_evaluation
├── src/
│   ├── data/                  load, preprocess, imbalance
│   ├── features/              engineering (cyclical time, log_amount)
│   ├── models/                registry, nn, train (MLflow logging)
│   └── evaluation/            metrics, threshold, explainability, error_analysis
├── api/                       FastAPI backend (REST + Ably pub/sub + SQLite)
│   ├── main.py                routes, startup, Ably publishing
│   ├── predict.py             /predict logic
│   ├── scenario.py            /scenario logic (risk score + dataset sampling)
│   └── database.py            SQLite layer
├── dashboard/                 Streamlit analytics app
├── web/                       React 18 (Vite + TS) real-time alert dashboard
│   └── src/
│       ├── App.tsx            dashboard page (stats, feed, alerts, submit)
│       ├── Simulate.tsx       scenario builder page
│       ├── PipelineDiagram.tsx  visual simulation pipeline explanation
│       ├── TxDetailModal.tsx    transaction detail modal (SHAP + PCA)
│       ├── Header.tsx         nav header with live status
│       ├── icons.tsx          Lineicons bulk icon library
│       ├── sound.ts           Web Audio API alert sound
│       ├── store.tsx          global state (Ably, polling, lazy loading)
│       └── presets.ts         fraud/legit transaction presets
├── alerts/                    transaction simulator + notifier
├── scripts/reproduce.sh       one-command full pipeline
├── tests/                     pytest (data pipeline, models, API)
└── reports/figures/           gitignored — generated plots
```
