import unittest
from datetime import datetime, timedelta, timezone
import math

from strategy_lab import MODEL_LABELS, fit_and_evaluate, make_minute_forecast


START = datetime(2026, 10, 20, 7, 30, tzinfo=timezone.utc)


def history_rows(count=520):
    rows = []
    previous_close = 100.0
    for index in range(count):
        close = 100 + index * 0.02 + math.sin(index / 5) * 0.25
        open_price = previous_close
        rows.append({
            "time": START + timedelta(minutes=index),
            "open": open_price,
            "high": max(open_price, close) + 0.12,
            "low": min(open_price, close) - 0.12,
            "close": close,
            "volume": 10.0,
        })
        previous_close = close
    return rows


class StrategyLabTests(unittest.TestCase):
    def test_requires_enough_history_before_selecting_a_method(self):
        result = fit_and_evaluate(history_rows(80), horizon_minutes=1)

        self.assertFalse(result["ready"])
        self.assertGreater(result["minimum_bars"], 80)

    def test_selects_on_earlier_history_and_reports_later_holdout(self):
        rows = history_rows()

        result = fit_and_evaluate(rows, horizon_minutes=3)

        self.assertTrue(result["ready"])
        self.assertIn(result["model"], MODEL_LABELS)
        self.assertLess(result["train_bars"], result["bars"])
        self.assertGreater(result["holdout_bars"], 0)
        self.assertGreater(result["holdout"]["origins"], 0)
        self.assertIsNotNone(result["holdout"]["last_close_mae_pct"])
        self.assertIsNotNone(result["holdout"]["momentum_mae_pct"])

        forecast = make_minute_forecast(rows, result)

        self.assertEqual(len(forecast["forecast_candles"]), 3)
        self.assertEqual(
            forecast["forecast_candles"][-1]["time"],
            rows[-1]["time"] + timedelta(minutes=3),
        )
        for candle in forecast["forecast_candles"]:
            self.assertLessEqual(candle["low"], min(candle["open"], candle["close"]))
            self.assertGreaterEqual(candle["high"], max(candle["open"], candle["close"]))

    def test_holdout_changes_do_not_change_training_rule_selection(self):
        rows = history_rows()
        split = int(len(rows) * 0.70)
        changed = [dict(row) for row in rows]
        for index in range(split, len(changed)):
            original = changed[index]["close"]
            changed[index].update({
                "open": original * 1.10,
                "high": original * 1.12,
                "low": original * 1.08,
                "close": original * 1.10,
            })

        original_fit = fit_and_evaluate(rows, horizon_minutes=1)
        changed_fit = fit_and_evaluate(changed, horizon_minutes=1)

        self.assertEqual(original_fit["model"], changed_fit["model"])
        self.assertEqual(original_fit["expected_returns"], changed_fit["expected_returns"])

    def test_small_or_weak_holdout_does_not_emit_directional_watch(self):
        rows = history_rows()
        fitted = fit_and_evaluate(rows, horizon_minutes=1)
        fitted["model"] = "momentum"
        fitted["model_label"] = MODEL_LABELS["momentum"]
        fitted["expected_returns"] = {-1: -0.001, 1: 0.001}
        fitted["holdout"] = {
            "calls": 8,
            "directional_accuracy_pct": 75.0,
            "forecast_mae_pct": 0.01,
            "last_close_mae_on_calls_pct": 0.02,
        }

        forecast = make_minute_forecast(rows, fitted)

        self.assertFalse(forecast["signal_validated"])
        self.assertEqual(forecast["direction"], 0)


if __name__ == "__main__":
    unittest.main()
