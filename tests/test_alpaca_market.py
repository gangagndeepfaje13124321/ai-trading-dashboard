import unittest
import io
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from alpaca_market import (
    AlpacaTradeFeed,
    alpaca_headers,
    fetch_alpaca_minute_history,
    parse_alpaca_minute_bars,
    parse_alpaca_trade_messages,
)


class AlpacaMarketTests(unittest.TestCase):
    def test_parse_trade_messages_filters_symbol_and_invalid_trades(self):
        payload = [
            {"T": "t", "S": "AAPL", "i": 10, "p": 200.25, "s": 2,
             "t": "2026-10-20T01:02:03.456789Z"},
            {"T": "t", "S": "TSLA", "i": 11, "p": 300, "s": 1,
             "t": "2026-10-20T01:02:03Z"},
            {"T": "t", "S": "AAPL", "i": 12, "p": float("nan"), "s": 1,
             "t": "2026-10-20T01:02:03Z"},
            {"T": "q", "S": "AAPL"},
        ]

        trades = parse_alpaca_trade_messages(payload, "aapl")

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["symbol"], "AAPL")
        self.assertEqual(trades[0]["price"], 200.25)
        self.assertEqual(trades[0]["timestamp"].tzinfo, timezone.utc)

    def test_parse_minute_bars_returns_sorted_utc_ohlc(self):
        payload = {"bars": {"AAPL": [
            {"t": "2026-10-20T01:02:00Z", "o": 200, "h": 202, "l": 199, "c": 201, "v": 80, "n": 6},
            {"t": "2026-10-20T01:01:00Z", "o": 199, "h": 201, "l": 198, "c": 200, "v": 70, "n": 5},
        ]}}

        rows = parse_alpaca_minute_bars(payload, "AAPL")

        self.assertEqual([row["close"] for row in rows], [200, 201])
        self.assertEqual(rows[0]["time"], datetime(2026, 10, 20, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(rows[1]["trade_count"], 6)

    def test_auth_headers_require_both_secrets(self):
        self.assertEqual(
            alpaca_headers("key", "secret"),
            {"APCA-API-KEY-ID": "key", "APCA-API-SECRET-KEY": "secret"},
        )
        with self.assertRaisesRegex(ValueError, "required"):
            alpaca_headers("key", "")

    def test_fetch_history_uses_iex_and_parses_response(self):
        payload = {"bars": {"AAPL": [
            {"t": "2026-10-20T01:02:00Z", "o": 200, "h": 202, "l": 199, "c": 201, "v": 80, "n": 6},
        ]}}
        response = io.BytesIO(json.dumps(payload).encode("utf-8"))
        with patch("alpaca_market.urlopen", return_value=response) as get:
            rows = fetch_alpaca_minute_history("AAPL", "key", "secret")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["close"], 201)
        request = get.call_args.args[0]
        self.assertIn("feed=iex", request.full_url)
        self.assertIn("timeframe=1Min", request.full_url)

    def test_websocket_authenticates_subscribes_and_aggregates_trades(self):
        feed = AlpacaTradeFeed("AAPL", "key", "secret")
        socket = Mock()

        feed._on_open(socket)
        auth = json.loads(socket.send.call_args.args[0])
        self.assertEqual(auth["action"], "auth")
        self.assertEqual(auth["key"], "key")
        self.assertEqual(auth["secret"], "secret")

        feed._on_message(socket, json.dumps([{"T": "success", "msg": "authenticated"}]))
        subscribe = json.loads(socket.send.call_args.args[0])
        self.assertEqual(subscribe, {"action": "subscribe", "trades": ["AAPL"]})
        feed._on_message(socket, json.dumps([{"T": "subscription", "trades": ["AAPL"]}]))
        feed._on_message(socket, json.dumps([{
            "T": "t", "S": "AAPL", "i": 101, "p": 201.2, "s": 3,
            "t": "2026-10-20T01:02:03.456789Z",
        }]))

        snapshot = feed.snapshot()
        self.assertEqual(snapshot["status"], "connected")
        self.assertEqual(snapshot["candles"][0]["close"], 201.2)
        self.assertEqual(snapshot["candles"][0]["trade_count"], 1)


if __name__ == "__main__":
    unittest.main()
