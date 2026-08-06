from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ABLY_API_KEY", "")
    from api import database

    database.DB_PATH = database.DB_PATH.parent / "test_vigil.db"
    database.init_db()

    from api.main import app

    return TestClient(app)


def _make_transaction(amount: float = 100.0) -> dict[str, float]:
    data = {"Time": 10000.0, "Amount": amount}
    for i in range(1, 29):
        data[f"V{i}"] = 0.0
    return data


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_predict_returns_response(client: TestClient) -> None:
    tx = _make_transaction(amount=50.0)
    resp = client.post("/predict", json=tx)
    assert resp.status_code == 200
    data = resp.json()
    assert "is_fraud" in data
    assert "probability" in data
    assert "threshold" in data
    assert "top_features" in data
    assert isinstance(data["is_fraud"], bool)
    assert 0 <= data["probability"] <= 1


def test_predict_extreme_amount(client: TestClient) -> None:
    tx = _make_transaction(amount=25000.0)
    tx["V14"] = -10.0
    tx["V4"] = 8.0
    resp = client.post("/predict", json=tx)
    assert resp.status_code == 200
    data = resp.json()
    assert "is_fraud" in data


def test_stats_increments(client: TestClient) -> None:
    client.get("/stats")
    tx = _make_transaction()
    client.post("/predict", json=tx)
    resp = client.get("/stats")
    assert resp.status_code == 200
    assert resp.json()["total"] >= 1


def test_transactions_endpoint(client: TestClient) -> None:
    tx = _make_transaction(amount=75.0)
    client.post("/predict", json=tx)
    resp = client.get("/transactions?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["amount"] == 75.0


def test_alerts_endpoint(client: TestClient) -> None:
    resp = client.get("/alerts?limit=5")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
