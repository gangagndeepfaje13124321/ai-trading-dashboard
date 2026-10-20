import unittest
from datetime import datetime, timedelta, timezone

from minute_forecast import make_minute_forecast, walk_forward_direction_test


START = datetime(2026, 10, 20, 7, 0, tzinfo=timezone.utc)


def candles_from_closes(closes, volumes=None):
    volumes = volumes or [10.0] * len(closes)
    rows = []
    previous = closes[0]
    for index, close in enumerate(closes):
        spread = 0.2
        rows.append({
            "time": START + timedelta(minutes=index),
            "open": previous,
            "high": max(previous, close) + spread,
            "low": min(previous, close) - spread,
            "close": close,
            "volume": volumes[index],
            "trade_count": 10,
        })
        previous = close
    return rows


class MinuteForecastTests(unittest.TestCase):
    def test_flat_history_is_neutral_and_returns_a_range_not_a_path(self):
        rows = candles_from_closes([100.0] * 50)

        forecast = make_minute_forecast(rows, horizon_minutes=3)

        self.assertIsNotNone(forecast)
        self.assertEqual(forecast["direction"], 0)
        self.assertEqual(forecast["bias"], "Mixed / no clear bias")
        self.assertFalse(forecast["is_trained_model"])
        self.assertLess(forecast["scenario_low"], forecast["price"])
        self.assertGreater(forecast["scenario_high"], forecast["price"])
        self.assertEqual(forecast["horizon_minutes"], 3)

    def test_rising_history_blends_multiple_methods(self):
        closes = [100 + index * 0.25 for index in range(80)]
        rows = candles_from_closes(closes)

        forecast = make_minute_forecast(rows, horizon_minutes=1)

        self.assertEqual(forecast["direction"], 1)
        self.assertEqual(forecast["bias"], "Upward bias")
        self.assertGreaterEqual(forecast["active_votes"], 2)
        self.assertIn("EMA trend (9/21)", [signal["name"] for signal in forecast["signals"]])
        self.assertIn("RSI(14) reversal check", [signal["name"] for signal in forecast["signals"]])
        self.assertIn("20-minute range breakout", [signal["name"] for signal in forecast["signals"]])

    def test_insufficient_history_returns_no_forecast(self):
        self.assertIsNone(make_minute_forecast(candles_from_closes([100.0] * 20)))

    def test_walk_forward_uses_later_candles_and_compares_a_baseline(self):
        closes = [100 + index * 0.2 for index in range(150)]
        result = walk_forward_direction_test(candles_from_closes(closes), horizon_minutes=3)

        self.assertGreater(result["calls"], 0)
        self.assertEqual(result["calls"], result["wins"])
        self.assertEqual(result["hit_rate_pct"], 100.0)
        self.assertEqual(result["baseline_hit_rate_pct"], 100.0)
        self.assertFalse(result["validated"])


if __name__ == "__main__":
    unittest.main()
