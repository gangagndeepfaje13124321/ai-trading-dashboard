import unittest
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock
from unittest.mock import patch

from market_history import (
    aggregate_trade_candles,
    fetch_kraken_minute_history,
    merge_history_with_live_trades,
    parse_kraken_ohlc,
)


START = datetime(2026, 10, 20, 7, 30, tzinfo=timezone.utc)


class MarketHistoryTests(unittest.TestCase):
    def test_parse_kraken_ohlc_response(self):
        payload = {
            "error": [],
            "result": {
                "BTC/USD": [[int(START.timestamp()), "100", "103", "99", "102", "101", "4.5", 12]],
                "last": 123,
            },
        }

        candles = parse_kraken_ohlc(payload)

        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0]["time"], START)
        self.assertEqual(candles[0]["high"], 103)
        self.assertEqual(candles[0]["volume"], 4.5)
        self.assertEqual(candles[0]["trade_count"], 12)

    def test_kraken_errors_are_not_mistaken_for_empty_history(self):
        with self.assertRaisesRegex(RuntimeError, "Unknown asset pair"):
            parse_kraken_ohlc({"error": ["Unknown asset pair"], "result": {}})

    def test_fetch_uses_public_one_minute_endpoint_without_credentials(self):
        response = Mock()
        response.read.return_value = json.dumps({
            "error": [],
            "result": {"BTC/USD": [[int(START.timestamp()), "100", "103", "99", "102", "101", "4.5", 12]]},
        }).encode("utf-8")
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch("market_history.urlopen", return_value=response) as get:
            candles = fetch_kraken_minute_history("BTC/USD")

        self.assertEqual(len(candles), 1)
        get.assert_called_once()
        self.assertIn("pair=BTC%2FUSD", get.call_args.args[0])
        self.assertIn("interval=1", get.call_args.args[0])
        self.assertIn("assetVersion=1", get.call_args.args[0])
        self.assertEqual(get.call_args.kwargs["timeout"], 8)

    def test_aggregate_trade_seconds_without_inventing_empty_minutes(self):
        seconds = [
            {"time": START + timedelta(seconds=1), "open": 100, "high": 101, "low": 99, "close": 100.5,
             "volume": 2, "trade_count": 3},
            {"time": START + timedelta(seconds=59), "open": 100.5, "high": 102, "low": 100,
             "close": 101.5, "volume": 4, "trade_count": 2},
            {"time": START + timedelta(minutes=2), "open": 103, "high": 104, "low": 102,
             "close": 103.5, "volume": 1, "trade_count": 1},
        ]

        minutes = aggregate_trade_candles(seconds)

        self.assertEqual(len(minutes), 2)
        self.assertEqual(minutes[0]["time"], START)
        self.assertEqual(minutes[0]["open"], 100)
        self.assertEqual(minutes[0]["high"], 102)
        self.assertEqual(minutes[0]["low"], 99)
        self.assertEqual(minutes[0]["close"], 101.5)
        self.assertEqual(minutes[0]["volume"], 6)
        self.assertEqual(minutes[0]["trade_count"], 5)

    def test_merge_uses_rest_for_old_minutes_and_live_for_current_minute(self):
        history = [
            {"time": START, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10, "trade_count": 20},
            {"time": START + timedelta(minutes=1), "open": 101, "high": 103, "low": 100, "close": 102, "volume": 8, "trade_count": 17},
        ]
        live_seconds = [
            {"time": START + timedelta(seconds=20), "open": 101, "high": 102.5, "low": 100.5,
             "close": 102.2, "volume": 1, "trade_count": 2},
            {"time": START + timedelta(minutes=1, seconds=10), "open": 102, "high": 104, "low": 101.5,
             "close": 103.5, "volume": 2, "trade_count": 3},
        ]

        merged = merge_history_with_live_trades(history, live_seconds)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["close"], 101)
        self.assertEqual(merged[1]["open"], 101)
        self.assertEqual(merged[1]["high"], 104)
        self.assertEqual(merged[1]["close"], 103.5)
        self.assertEqual(merged[1]["volume"], 8)


if __name__ == "__main__":
    unittest.main()
