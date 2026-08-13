import unittest

import pandas as pd

from automatic_trade.data import prepare_features
from automatic_trade.environment import TradingEnvironment


def market(closes):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(closes), tz="UTC"),
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [100] * len(closes),
        }
    )
    return prepare_features(frame)


class EnvironmentTests(unittest.TestCase):
    def test_long_position_gains_when_price_rises(self):
        environment = TradingEnvironment(market([100, 110]), transaction_cost=0.0)
        environment.reset()
        result = environment.step(1)
        self.assertAlmostEqual(result.reward, 0.10)
        self.assertAlmostEqual(result.portfolio_value, 110_000.0)
        self.assertTrue(result.done)

    def test_short_position_gains_when_price_falls(self):
        environment = TradingEnvironment(market([100, 90]), transaction_cost=0.0)
        environment.reset()
        result = environment.step(0)
        self.assertAlmostEqual(result.reward, 0.10)

    def test_position_change_charges_transaction_cost(self):
        environment = TradingEnvironment(market([100, 100]), transaction_cost=0.01)
        environment.reset()
        result = environment.step(1)
        self.assertAlmostEqual(result.portfolio_value, 99_000.0)


if __name__ == "__main__":
    unittest.main()

