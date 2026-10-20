import unittest

import pandas as pd

from forecasting import (
    build_forecast_candles,
    linear_trend_forecast,
    rolling_origin_backtest,
)


class LinearTrendForecastTests(unittest.TestCase):
    def test_extrapolates_a_linear_close_series(self):
        forecast = linear_trend_forecast([100, 102, 104, 106], steps=2)
        self.assertAlmostEqual(forecast[0], 108.0)
        self.assertAlmostEqual(forecast[1], 110.0)

    def test_repeats_a_single_close_when_slope_cannot_be_estimated(self):
        self.assertEqual(linear_trend_forecast([42], steps=3), [42.0, 42.0, 42.0])


class ForecastCandleTests(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame({
            "Open": [100.0, 101.0],
            "High": [103.0, 104.0],
            "Low": [98.0, 99.0],
            "Close": [102.0, 103.0],
        })

    def test_builds_ohlc_candles_from_predicted_closes_and_typical_wicks(self):
        candles = build_forecast_candles(self.history, [104.0, 105.0])

        self.assertEqual(candles[0], {
            "step": 1, "open": 103.0, "high": 105.0, "low": 101.0, "close": 104.0,
        })
        self.assertEqual(candles[1]["open"], 104.0)
        self.assertEqual(candles[1]["close"], 105.0)
        for candle in candles:
            self.assertLessEqual(candle["low"], min(candle["open"], candle["close"]))
            self.assertGreaterEqual(candle["high"], max(candle["open"], candle["close"]))


class RollingOriginBacktestTests(unittest.TestCase):
    def test_uses_later_candles_as_targets_and_beats_persistence_on_linear_data(self):
        close = [100.0 + index for index in range(60)]
        frame = pd.DataFrame({
            "Open": [close[0]] + close[:-1],
            "High": [value + 2 for value in close],
            "Low": [value - 2 for value in close],
            "Close": close,
        })

        result = rolling_origin_backtest(frame, lookback=30, min_train=20)

        self.assertEqual(result["holdout_candles"], 8)
        self.assertAlmostEqual(result["trend_close_mae"], 0.0, places=10)
        self.assertGreater(result["last_close_mae"], 0.0)
        self.assertEqual(result["directional_accuracy_pct"], 100.0)
        self.assertAlmostEqual(result["ohlc_mae"]["open"], 0.0, places=10)
        self.assertAlmostEqual(result["ohlc_mae"]["high"], 0.0, places=10)
        self.assertAlmostEqual(result["ohlc_mae"]["low"], 0.0, places=10)

    def test_returns_no_result_when_too_few_test_candles_exist(self):
        frame = pd.DataFrame({
            "Open": range(1, 24),
            "High": range(2, 25),
            "Low": range(0, 23),
            "Close": range(1, 24),
        })
        self.assertIsNone(rolling_origin_backtest(frame, min_train=20))


if __name__ == "__main__":
    unittest.main()
