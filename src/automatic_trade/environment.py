"""A small long/short cryptocurrency backtesting environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ("open_rel", "high_rel", "low_rel", "close_rel", "volume_z")


@dataclass(frozen=True)
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    portfolio_value: float


class TradingEnvironment:
    """Track an all-long or all-short position using daily closing prices.

    Action 0 targets a fully short position and action 1 targets a fully long
    position. Reward is the fractional change in portfolio value for one step.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
    ) -> None:
        missing = [column for column in (*FEATURE_COLUMNS, "close") if column not in frame.columns]
        if missing:
            raise ValueError(f"Frame is missing columns: {', '.join(missing)}")
        if len(frame) < 2:
            raise ValueError("Trading requires at least two rows")
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not 0 <= transaction_cost < 1:
            raise ValueError("transaction_cost must be in [0, 1)")

        self.frame = frame.reset_index(drop=True)
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self._index = 0
        self._position = 0
        self._value = self.initial_cash

    @property
    def state_size(self) -> int:
        return len(FEATURE_COLUMNS) + 1

    @property
    def portfolio_value(self) -> float:
        return self._value

    def _state(self) -> np.ndarray:
        market = self.frame.loc[self._index, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        return np.concatenate([market, np.array([self._position], dtype=np.float32)])

    def reset(self) -> np.ndarray:
        self._index = 0
        self._position = 0
        self._value = self.initial_cash
        return self._state()

    def step(self, action: int) -> StepResult:
        if action not in (0, 1):
            raise ValueError("action must be 0 (short) or 1 (long)")
        if self._index >= len(self.frame) - 1:
            raise RuntimeError("episode is already complete")

        target_position = -1 if action == 0 else 1
        previous_value = self._value
        if target_position != self._position:
            self._value *= 1.0 - self.transaction_cost

        current_close = float(self.frame.loc[self._index, "close"])
        next_close = float(self.frame.loc[self._index + 1, "close"])
        market_return = next_close / current_close - 1.0
        self._value *= 1.0 + target_position * market_return
        self._value = max(self._value, 0.0)
        self._position = target_position
        self._index += 1

        reward = self._value / previous_value - 1.0
        done = self._index == len(self.frame) - 1 or self._value == 0.0
        return StepResult(self._state(), float(reward), done, self._value)
