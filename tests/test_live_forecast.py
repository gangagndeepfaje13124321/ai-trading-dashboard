import unittest
from datetime import datetime, timedelta, timezone

from live_forecast import LiveForecastEvaluator, make_live_forecast


START = datetime(2026, 10, 20, 7, 30, tzinfo=timezone.utc)


def candle(second, open_price, high, low, close):
    return {
        "time": START + timedelta(seconds=second),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1.0,
        "trade_count": 1,
    }


class LiveForecastTests(unittest.TestCase):
    def test_flat_market_makes_neutral_ohlc_scenarios(self):
        history = [candle(i, 100, 100, 100, 100) for i in range(20)]

        forecast = make_live_forecast(history, horizon_seconds=5)

        self.assertEqual(forecast["direction"], 0)
        self.assertEqual(forecast["method"], "none")
        self.assertEqual(len(forecast["forecast_candles"]), 5)
        for index, future in enumerate(forecast["forecast_candles"], start=1):
            self.assertEqual(future["time"], history[-1]["time"] + timedelta(seconds=index))
            self.assertLessEqual(future["low"], min(future["open"], future["close"]))
            self.assertGreaterEqual(future["high"], max(future["open"], future["close"]))

    def test_ascending_triangle_only_gets_direction_after_breakout(self):
        history = []
        peak_indices = {2, 8, 14, 20, 26}
        troughs = {5: 99.0, 11: 99.3, 17: 99.6, 23: 99.9}
        for index in range(30):
            high = 101.0 if index in peak_indices else 100.2
            low = troughs.get(index, 99.95)
            history.append(candle(index, 100.0, high, low, 100.0))

        forming = make_live_forecast(history + [candle(30, 100.0, 100.2, 99.95, 100.0)])
        broken = make_live_forecast(history + [candle(30, 100.9, 101.3, 100.8, 101.2)])

        self.assertIn("Ascending triangle", forming["pattern"])
        self.assertEqual(forming["method"], "none")
        self.assertIn("Ascending triangle", broken["pattern"])
        self.assertEqual(broken["method"], "pattern")
        self.assertEqual(broken["direction"], 1)

    def test_double_top_needs_a_neckline_break(self):
        history = []
        for index in range(25):
            high = 101.5 if index in {4, 16} else 100.2
            low = 99.0 if index == 10 else 99.8
            history.append(candle(index, 100.0, high, low, 100.0))

        forming = make_live_forecast(history + [candle(25, 100.0, 100.2, 99.8, 100.0)])
        broken = make_live_forecast(history + [candle(25, 99.0, 99.1, 98.5, 98.6)])

        self.assertIn("Double top", forming["pattern"])
        self.assertEqual(forming["direction"], 0)
        self.assertIn("Double top", broken["pattern"])
        self.assertEqual(broken["method"], "pattern")
        self.assertEqual(broken["direction"], -1)

    def test_head_and_shoulders_needs_a_neckline_break(self):
        history = []
        peaks = {4: 101.5, 11: 103.0, 18: 101.5}
        troughs = {7: 99.0, 14: 99.0}
        for index in range(25):
            history.append(candle(index, 100.0, peaks.get(index, 100.2), troughs.get(index, 99.8), 100.0))

        broken = make_live_forecast(history + [candle(25, 98.9, 99.0, 98.4, 98.5)])

        self.assertIn("Head and shoulders", broken["pattern"])
        self.assertEqual(broken["method"], "pattern")
        self.assertEqual(broken["direction"], -1)

    def test_online_evaluator_scores_against_later_trade_candle(self):
        history = [
            candle(i, 100 + i, 100.1 + i, 99.9 + i, 100 + i)
            for i in range(20)
        ]
        evaluator = LiveForecastEvaluator(horizon_seconds=5)

        first = evaluator.update(history)
        later = history + [
            candle(i, 120 - (i - 20) * 2, 120.1 - (i - 20) * 2,
                   119.9 - (i - 20) * 2, 120 - (i - 20) * 2)
            for i in range(20, 25)
        ]
        result = evaluator.update(later)

        self.assertTrue(first["pending_evaluation"])
        self.assertEqual(result["all"]["samples"], 1)
        self.assertEqual(result["all"]["wins"], 0)
        self.assertEqual(result["all"]["hit_rate_pct"], 0.0)
        self.assertGreater(
            result["all"]["forecast_close_mae_pct"],
            result["all"]["last_close_mae_pct"],
        )

    def test_longer_second_scenario_has_requested_linear_ohlc_horizon(self):
        history = [
            candle(index, 100 + index, 100.2 + index, 99.8 + index, 100 + index)
            for index in range(40)
        ]

        forecast = make_live_forecast(history, horizon_seconds=60)

        self.assertEqual(len(forecast["forecast_candles"]), 60)
        self.assertEqual(
            forecast["forecast_candles"][-1]["time"],
            history[-1]["time"] + timedelta(seconds=60),
        )
        closes = [row["close"] for row in forecast["forecast_candles"]]
        self.assertTrue(all(later >= earlier for earlier, later in zip(closes, closes[1:])))
        steps = [later - earlier for earlier, later in zip([history[-1]["close"]] + closes[:-1], closes)]
        self.assertTrue(all(abs(step - steps[0]) < 1e-9 for step in steps))


if __name__ == "__main__":
    unittest.main()
