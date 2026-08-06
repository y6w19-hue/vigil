from __future__ import annotations

import numpy as np
import pandas as pd


def cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour = (out["Time"] % 86400) / 3600
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out


def log_amount(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amount"] = np.log1p(out["Amount"])
    return out


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = cyclical_time(df)
    out = log_amount(out)
    out = out.drop(columns=["Time"])
    return out
