"""OHLCV loading and feature preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "time_period_start": "timestamp",
    "timestamp": "timestamp",
    "date": "timestamp",
    "price_open": "open",
    "open": "open",
    "price_high": "high",
    "high": "high",
    "price_low": "low",
    "low": "low",
    "price_close": "close",
    "close": "close",
    "volume_traded": "volume",
    "volume": "volume",
}

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load a CoinAPI-style CSV and return validated, chronological OHLCV rows."""
    frame = pd.read_csv(path)
    normalized = {column: column.strip().lower() for column in frame.columns}
    frame = frame.rename(columns=normalized)
    frame = frame.rename(columns={key: value for key, value in COLUMN_ALIASES.items() if key in frame.columns})

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")

    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    for column in REQUIRED_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    if not np.isfinite(frame.loc[:, REQUIRED_COLUMNS[1:]].to_numpy()).all():
        raise ValueError("OHLCV data contains non-finite values")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    if (frame["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")

    return frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create price-relative and standardized-volume features without future leakage."""
    prepared = frame.copy()
    open_price = prepared["open"].replace(0, np.nan)
    prepared["open_rel"] = prepared["open"].pct_change().fillna(0.0)
    prepared["high_rel"] = prepared["high"] / open_price - 1.0
    prepared["low_rel"] = prepared["low"] / open_price - 1.0
    prepared["close_rel"] = prepared["close"] / open_price - 1.0

    volume_mean = prepared["volume"].mean()
    volume_std = prepared["volume"].std(ddof=0)
    prepared["volume_z"] = 0.0 if volume_std == 0 else (prepared["volume"] - volume_mean) / volume_std

    feature_columns = ["open_rel", "high_rel", "low_rel", "close_rel", "volume_z"]
    if not np.isfinite(prepared[feature_columns].to_numpy()).all():
        raise ValueError("Feature preparation produced non-finite values")
    return prepared


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows in time order so future observations never enter training."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if len(frame) < 2:
        raise ValueError("At least two observations are required")
    boundary = min(max(int(len(frame) * train_fraction), 1), len(frame) - 1)
    return frame.iloc[:boundary].copy(), frame.iloc[boundary:].copy()

