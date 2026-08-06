from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any

from src.config import REPO_ROOT

DB_PATH = REPO_ROOT / "data" / "vigil.db"

_lock = threading.Lock()


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock:
        conn = get_connection()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                amount REAL NOT NULL,
                time REAL NOT NULL,
                probability REAL NOT NULL,
                is_fraud INTEGER NOT NULL,
                threshold REAL NOT NULL,
                top_features TEXT,
                v1 REAL, v2 REAL, v3 REAL, v4 REAL, v5 REAL,
                v6 REAL, v7 REAL, v8 REAL, v9 REAL, v10 REAL,
                v11 REAL, v12 REAL, v13 REAL, v14 REAL, v15 REAL,
                v16 REAL, v17 REAL, v18 REAL, v19 REAL, v20 REAL,
                v21 REAL, v22 REAL, v23 REAL, v24 REAL, v25 REAL,
                v26 REAL, v27 REAL, v28 REAL
            );

            CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            );

            INSERT OR IGNORE INTO stats (key, value) VALUES ('total', 0);
            INSERT OR IGNORE INTO stats (key, value) VALUES ('fraud', 0);
            INSERT OR IGNORE INTO stats (key, value) VALUES ('alerts', 0);
            """
        )
        conn.commit()
        conn.close()


def record_transaction(
    tx_data: dict[str, float],
    result: dict[str, Any],
    threshold: float,
) -> int:
    now = datetime.now(UTC).isoformat()
    features_json = json.dumps(result.get("top_features", []))

    v_vals = {f"v{i}": tx_data.get(f"V{i}", 0.0) for i in range(1, 29)}

    with _lock:
        conn = get_connection()
        cursor = conn.execute(
            """
            INSERT INTO transactions (
                timestamp, amount, time, probability, is_fraud, threshold,
                top_features, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                v21, v22, v23, v24, v25, v26, v27, v28
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?
            )
            """,
            (
                now,
                tx_data.get("Amount", 0.0),
                tx_data.get("Time", 0.0),
                result["probability"],
                int(result["is_fraud"]),
                threshold,
                features_json,
                *[v_vals[f"v{i}"] for i in range(1, 29)],
            ),
        )
        tx_id = cursor.lastrowid

        conn.execute("UPDATE stats SET value = value + 1 WHERE key = 'total'")
        if result["is_fraud"]:
            conn.execute("UPDATE stats SET value = value + 1 WHERE key = 'fraud'")
            conn.execute("UPDATE stats SET value = value + 1 WHERE key = 'alerts'")

        conn.commit()
        conn.close()

    return tx_id


def get_stats() -> dict[str, int]:
    with _lock:
        conn = get_connection()
        rows = conn.execute("SELECT key, value FROM stats").fetchall()
        conn.close()
    return {row["key"]: row["value"] for row in rows}


def get_recent_transactions(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    with _lock:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM transactions ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        conn.close()
    return [dict(row) for row in rows]


def get_recent_alerts(limit: int = 20) -> list[dict[str, Any]]:
    with _lock:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM transactions WHERE is_fraud = 1 ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
    return [dict(row) for row in rows]


def get_transaction_by_id(tx_id: int) -> dict[str, Any] | None:
    with _lock:
        conn = get_connection()
        row = conn.execute("SELECT * FROM transactions WHERE id = ?", (tx_id,)).fetchone()
        conn.close()
    return dict(row) if row else None


def get_alert_history(buckets: int = 20) -> list[dict[str, Any]]:
    with _lock:
        conn = get_connection()
        rows = conn.execute(
            "SELECT timestamp FROM transactions WHERE is_fraud = 1 ORDER BY timestamp ASC"
        ).fetchall()
        conn.close()

    if not rows:
        return []

    timestamps = [row["timestamp"] for row in rows]
    from datetime import datetime

    parsed = []
    for ts in timestamps:
        try:
            dt = datetime.fromisoformat(ts)
            parsed.append(dt)
        except Exception:
            pass

    if not parsed:
        return []

    first = parsed[0]
    last = parsed[-1]
    total_seconds = max((last - first).total_seconds(), 1)
    bucket_seconds = total_seconds / buckets

    counts = [0] * buckets
    for dt in parsed:
        idx = int((dt - first).total_seconds() / bucket_seconds)
        if idx >= buckets:
            idx = buckets - 1
        counts[idx] += 1

    return [
        {"time": (first.timestamp() + i * bucket_seconds) * 1000, "count": counts[i]}
        for i in range(buckets)
    ]
