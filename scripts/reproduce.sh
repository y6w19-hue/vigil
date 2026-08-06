#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== 1. Preprocessing ==="
python -m src.data.preprocess

echo "=== 2. Training all models ==="
for model in logistic random_forest xgboost lightgbm ann autoencoder; do
    echo "--- Training $model ---"
    python -m src.models.train --model "$model"
done

echo "=== 3. Selecting best model by PR-AUC ==="
python -m src.models.select_best

echo "=== 4. Threshold tuning ==="
python -m src.evaluation.threshold

echo "=== 5. SHAP analysis ==="
python -m src.evaluation.explainability

echo "=== 6. Error analysis ==="
python -m src.evaluation.error_analysis

echo "=== 7. Lint + tests ==="
ruff check . && black --check . && isort --check-only . && pytest -q

echo "=== Done. Artifacts in models/, figures in reports/figures/ ==="
