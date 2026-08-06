from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

import httpx
import pandas as pd

from src.config import get_path, load_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def load_test_transactions(config_path: str | Path | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    test_path = get_path(cfg, "processed_dir") / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}. Run preprocessing first.")
    return pd.read_parquet(test_path)


def transaction_to_api_format(row: pd.Series) -> dict:
    v_cols = [f"V{i}" for i in range(1, 29)]
    data = {"Time": 0, "Amount": float(row["Amount"])}
    for v in v_cols:
        data[v] = float(row[v])
    return data


async def simulate(
    rate: int = 5,
    limit: int = 0,
    shuffle: bool = True,
    api_url: str = API_URL,
) -> None:
    df = load_test_transactions()
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if limit > 0:
        df = df.head(limit)

    delay = 1.0 / rate
    async with httpx.AsyncClient() as client:
        for i, row in df.iterrows():
            tx_data = transaction_to_api_format(row)
            try:
                resp = await client.post(f"{api_url}/predict", json=tx_data, timeout=10)
                result = resp.json()
                status = "FRAUD" if result["is_fraud"] else "ok"
                logger.info(
                    "[%d/%d] %s prob=%.4f amount=%.2f",
                    i + 1,
                    len(df),
                    status,
                    result["probability"],
                    tx_data["Amount"],
                )
            except Exception as e:
                logger.error("Request failed: %s", e)

            await asyncio.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transaction stream simulator")
    parser.add_argument("--rate", type=int, default=5, help="Transactions per second")
    parser.add_argument("--limit", type=int, default=0, help="Max transactions (0 = all)")
    parser.add_argument("--no-shuffle", action="store_true", help="Keep original order")
    parser.add_argument("--api-url", default=API_URL, help="API base URL")
    args = parser.parse_args()

    asyncio.run(
        simulate(
            rate=args.rate,
            limit=args.limit,
            shuffle=not args.no_shuffle,
            api_url=args.api_url,
        )
    )


if __name__ == "__main__":
    main()
