from __future__ import annotations

import json
import logging
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api import database
from api import scenario as scenario_mod
from src.config import get_path, load_config
from src.features.engineering import engineer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(title="Vigil — Fraud Detection API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config = load_config()
models_dir = get_path(config, "models_dir")
threshold_path = models_dir / "threshold.json"

_model = None
_threshold = 0.5
_feature_names: list[str] = []
_ably_channel = None

ABLY_CHANNEL_NAME = "fraud-alerts"


class TransactionInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class PredictionResponse(BaseModel):
    is_fraud: bool
    probability: float
    threshold: float
    top_features: list[dict[str, float | str]]


class ScenarioInput(BaseModel):
    merchant: str
    amount: float
    hour: int
    card_present: bool
    new_device: bool
    foreign: bool


def _init_ably() -> None:
    global _ably_channel
    api_key = os.environ.get("ABLY_API_KEY")
    if not api_key:
        logger.warning("ABLY_API_KEY not set — real-time alerts disabled")
        return
    try:
        from ably import AblyRest

        client = AblyRest(key=api_key)
        _ably_channel = client.channels.get(ABLY_CHANNEL_NAME)
        logger.info("Ably connected on channel '%s'", ABLY_CHANNEL_NAME)
    except Exception as e:
        logger.error("Failed to init Ably: %s", e)


def _publish_alert(alert: dict[str, Any]) -> None:
    if _ably_channel is None:
        return
    try:
        import asyncio

        asyncio.run(_ably_channel.publish("fraud_alert", json.dumps(alert)))
    except Exception as e:
        logger.error("Ably publish failed: %s", e)


def load_model() -> None:
    global _model, _threshold, _feature_names

    best_path = models_dir / "best.joblib"
    if best_path.exists():
        _model = joblib.load(best_path)
        logger.info("Loaded best model from %s", best_path)
    else:
        for name in ("xgboost", "lightgbm", "random_forest", "logistic"):
            path = models_dir / f"{name}.joblib"
            if path.exists():
                _model = joblib.load(path)
                logger.info("Loaded model from %s", path)
                break

    if _model is None and (models_dir / "ann.keras").exists():
        from tensorflow import keras

        _model = keras.models.load_model(models_dir / "ann.keras")
        logger.info("Loaded ANN model")

    if threshold_path.exists():
        with open(threshold_path) as f:
            data = json.load(f)
            _threshold = data["threshold"]
            logger.info("Loaded threshold: %.4f (%s)", _threshold, data.get("policy"))

    sample_path = get_path(config, "processed_dir") / "train.parquet"
    if sample_path.exists():
        import pyarrow.parquet as pq

        schema = pq.read_schema(str(sample_path))
        _feature_names = [c for c in schema.names if c != "Class"]


def _to_dataframe(tx: TransactionInput) -> pd.DataFrame:
    data = tx.model_dump()
    df = pd.DataFrame([data])
    df = engineer(df)
    return df


def _predict(tx: TransactionInput) -> PredictionResponse:
    if _model is None:
        return PredictionResponse(
            is_fraud=False,
            probability=0.0,
            threshold=_threshold,
            top_features=[],
        )

    df = _to_dataframe(tx)

    if hasattr(_model, "predict_proba"):
        proba = float(_model.predict_proba(df)[0, 1])
    else:
        proba = float(_model.predict(df.values, verbose=0)[0, 0])

    is_fraud = proba >= _threshold

    top_features = []
    if _feature_names:
        try:
            import shap

            estimator = _model
            if hasattr(_model, "named_steps") and "model" in _model.named_steps:
                estimator = _model.named_steps["model"]
            elif hasattr(_model, "steps"):
                estimator = _model.steps[-1][1]

            explainer = shap.TreeExplainer(estimator)
            sv = explainer.shap_values(df)
            sv = sv[1] if isinstance(sv, list) else sv
            vals = sv[0]
            abs_vals = np.abs(vals)
            top_idx = np.argsort(abs_vals)[-5:][::-1]
            top_features = [
                {"feature": _feature_names[i], "shap_value": float(vals[i])} for i in top_idx
            ]
        except Exception:
            pass

    return PredictionResponse(
        is_fraud=is_fraud,
        probability=proba,
        threshold=_threshold,
        top_features=top_features,
    )


@app.on_event("startup")
def startup() -> None:
    database.init_db()
    load_model()
    scenario_mod.load_dataset()
    _init_ably()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "threshold": _threshold,
        "ably_connected": _ably_channel is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionInput) -> PredictionResponse:
    result = _predict(tx)

    tx_id = database.record_transaction(
        tx.model_dump(),
        {
            "is_fraud": result.is_fraud,
            "probability": result.probability,
            "top_features": result.top_features,
        },
        _threshold,
    )

    if result.is_fraud:
        alert = {
            "type": "fraud_alert",
            "id": tx_id,
            "probability": result.probability,
            "threshold": result.threshold,
            "top_features": result.top_features,
            "transaction": tx.model_dump(),
        }
        _publish_alert(alert)

    return result


@app.get("/stats")
def get_stats() -> dict[str, int]:
    return database.get_stats()


@app.get("/transactions")
def get_transactions(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    return database.get_recent_transactions(limit, offset)


@app.get("/alerts")
def get_alerts(limit: int = 20) -> list[dict[str, Any]]:
    return database.get_recent_alerts(limit)


@app.get("/transaction/{tx_id}")
def get_transaction(tx_id: int) -> dict[str, Any]:
    tx = database.get_transaction_by_id(tx_id)
    if tx is None:
        return {"error": "not found"}
    return tx


@app.get("/alert-history")
def get_alert_history(buckets: int = 20) -> list[dict[str, Any]]:
    return database.get_alert_history(buckets)


@app.post("/scenario")
def create_scenario(sc: ScenarioInput) -> dict[str, Any]:
    result = scenario_mod.build_transaction(
        merchant=sc.merchant,
        amount=sc.amount,
        hour=sc.hour,
        card_present=sc.card_present,
        new_device=sc.new_device,
        foreign=sc.foreign,
    )
    if result is None:
        return {"error": "Dataset not loaded"}

    tx_data = result["transaction"]
    tx = TransactionInput(**tx_data)
    prediction = _predict(tx)

    tx_id = database.record_transaction(
        tx.model_dump(),
        {
            "is_fraud": prediction.is_fraud,
            "probability": prediction.probability,
            "top_features": prediction.top_features,
        },
        _threshold,
    )

    if prediction.is_fraud:
        alert = {
            "type": "fraud_alert",
            "id": tx_id,
            "probability": prediction.probability,
            "threshold": prediction.threshold,
            "top_features": prediction.top_features,
            "transaction": tx.model_dump(),
        }
        _publish_alert(alert)

    return {
        "transaction": tx_data,
        "risk_score": result["risk_score"],
        "sampled_from": result["sampled_from"],
        "scenario": result["scenario"],
        "prediction": {
            "is_fraud": prediction.is_fraud,
            "probability": prediction.probability,
            "threshold": prediction.threshold,
            "top_features": prediction.top_features,
        },
    }
