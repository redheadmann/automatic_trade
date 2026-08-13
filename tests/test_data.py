import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from automatic_trade.data import chronological_split, load_ohlcv, prepare_features


class DataTests(unittest.TestCase):
    def test_coinapi_columns_are_loaded_and_sorted(self):
        frame = pd.DataFrame(
            {
                "time_period_start": ["2024-01-02", "2024-01-01"],
                "price_open": [11, 10],
                "price_high": [12, 11],
                "price_low": [10, 9],
                "price_close": [11.5, 10.5],
                "volume_traded": [120, 100],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ohlcv.csv"
            frame.to_csv(path, index=False)
            loaded = load_ohlcv(path)
        self.assertEqual(list(loaded["close"]), [10.5, 11.5])

    def test_features_are_finite_and_split_is_chronological(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, tz="UTC"),
                "open": [10, 11, 12, 13, 14],
                "high": [11, 12, 13, 14, 15],
                "low": [9, 10, 11, 12, 13],
                "close": [10.5, 11.5, 12.5, 13.5, 14.5],
                "volume": [100, 110, 120, 130, 140],
            }
        )
        prepared = prepare_features(frame)
        self.assertTrue(np.isfinite(prepared[["open_rel", "high_rel", "low_rel", "close_rel", "volume_z"]]).all().all())
        train, test = chronological_split(prepared, 0.6)
        self.assertEqual(len(train), 3)
        self.assertLess(train["timestamp"].max(), test["timestamp"].min())


if __name__ == "__main__":
    unittest.main()

