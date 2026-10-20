import json
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock

from trade_candles import (
    KrakenTradeFeed,
    TradeCandleBuffer,
    make_trade_subscription,
    parse_trade_message,
)


def trade(timestamp, price, qty, trade_id, symbol="BTC/USD"):
    return {
        "symbol": symbol,
        "timestamp": datetime.fromisoformat(timestamp.replace("Z", "+00:00")),
        "price": price,
        "qty": qty,
        "trade_id": trade_id,
    }


class ParseTradeMessageTests(unittest.TestCase):
    def test_parses_snapshot_trades_with_utc_exchange_timestamps(self):
        message = json.dumps({
            "channel": "trade",
            "type": "snapshot",
            "data": [{
                "symbol": "BTC/USD",
                "price": 100.5,
                "qty": 0.25,
                "trade_id": 17,
                "timestamp": "2026-10-20T07:30:53.817446Z",
            }],
        })

        parsed = parse_trade_message(message)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["symbol"], "BTC/USD")
        self.assertEqual(parsed[0]["price"], 100.5)
        self.assertEqual(parsed[0]["qty"], 0.25)
        self.assertEqual(parsed[0]["trade_id"], 17)
        self.assertEqual(parsed[0]["timestamp"].tzinfo, timezone.utc)

    def test_ignores_non_trade_messages_and_invalid_events(self):
        self.assertEqual(parse_trade_message({"channel": "status", "type": "update"}), [])
        self.assertEqual(parse_trade_message({"channel": "trade", "type": "update", "data": None}), [])
        parsed = parse_trade_message({
            "channel": "trade",
            "type": "update",
            "data": [
                {"symbol": "BTC/USD", "price": 0, "qty": 1, "trade_id": 1, "timestamp": "2026-10-20T07:30:53Z"},
                {"symbol": "BTC/USD", "price": 100, "qty": 1, "trade_id": 2, "timestamp": "not-a-date"},
            ],
        })
        self.assertEqual(parsed, [])

    def test_builds_trade_subscription_with_recent_snapshot(self):
        self.assertEqual(make_trade_subscription("BTC/USD", request_id=9), {
            "method": "subscribe",
            "params": {"channel": "trade", "symbol": ["BTC/USD"], "snapshot": True},
            "req_id": 9,
        })


class KrakenTradeFeedCallbackTests(unittest.TestCase):
    def test_subscribes_and_adds_received_trade_snapshots(self):
        feed = KrakenTradeFeed("BTC/USD")
        websocket = Mock()

        feed._on_open(websocket)
        subscription = json.loads(websocket.send.call_args.args[0])
        self.assertEqual(subscription["params"]["channel"], "trade")
        self.assertEqual(subscription["params"]["symbol"], ["BTC/USD"])

        feed._on_message(websocket, json.dumps({
            "method": "subscribe",
            "result": {"channel": "trade", "symbol": "BTC/USD"},
            "success": True,
        }))
        feed._on_message(websocket, json.dumps({
            "channel": "trade",
            "type": "snapshot",
            "data": [{
                "symbol": "BTC/USD",
                "price": 100.5,
                "qty": 0.25,
                "trade_id": 17,
                "timestamp": "2026-10-20T07:30:53.817446Z",
            }],
        }))

        snapshot = feed.snapshot()
        self.assertEqual(snapshot["status"], "connected")
        self.assertEqual(len(snapshot["candles"]), 1)
        self.assertEqual(snapshot["candles"][0]["close"], 100.5)

    def test_stops_when_kraken_rejects_the_symbol(self):
        feed = KrakenTradeFeed("NOT/A-PAIR")
        websocket = Mock()

        feed._on_message(websocket, json.dumps({
            "method": "subscribe",
            "success": False,
            "error": "Unknown symbol",
        }))

        self.assertEqual(feed.snapshot()["status"], "unavailable")
        self.assertIn("Unknown symbol", feed.snapshot()["detail"])
        self.assertTrue(feed._stop_event.is_set())
        websocket.close.assert_called_once_with()


class TradeCandleBufferTests(unittest.TestCase):
    def test_aggregates_ohlc_volume_and_trade_count_by_utc_second(self):
        candles = TradeCandleBuffer()
        events = [
            trade("2026-10-20T07:30:53.900000Z", 105, 0.1, 2),
            trade("2026-10-20T07:30:53.100000Z", 100, 0.2, 1),
            trade("2026-10-20T07:30:53.500000Z", 102, 0.3, 3),
            trade("2026-10-20T07:30:53.500000Z", 104, 0.4, 4),
        ]
        for event in events:
            self.assertTrue(candles.add_trade(event))

        candle = candles.snapshot()["candles"][0]
        self.assertEqual(candle["open"], 100)
        self.assertEqual(candle["high"], 105)
        self.assertEqual(candle["low"], 100)
        self.assertEqual(candle["close"], 105)
        self.assertAlmostEqual(candle["volume"], 1.0)
        self.assertEqual(candle["trade_count"], 4)
        self.assertEqual(candle["time"], datetime(2026, 10, 20, 7, 30, 53, tzinfo=timezone.utc))

    def test_does_not_invent_empty_seconds(self):
        candles = TradeCandleBuffer()
        candles.add_trade(trade("2026-10-20T07:30:53.100000Z", 100, 0.2, 1))
        candles.add_trade(trade("2026-10-20T07:30:55.100000Z", 101, 0.3, 2))

        rows = candles.snapshot()["candles"]

        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[1]["time"] - rows[0]["time"]).total_seconds(), 2)

    def test_deduplicates_trade_ids(self):
        candles = TradeCandleBuffer()
        event = trade("2026-10-20T07:30:53.100000Z", 100, 0.2, 1)

        self.assertTrue(candles.add_trade(event))
        self.assertFalse(candles.add_trade(event))
        candle = candles.snapshot()["candles"][0]
        self.assertEqual(candle["trade_count"], 1)
        self.assertAlmostEqual(candle["volume"], 0.2)

    def test_keeps_only_the_newest_candles(self):
        candles = TradeCandleBuffer(max_candles=2)
        for second in (53, 54, 55):
            candles.add_trade(trade(f"2026-10-20T07:30:{second}.100000Z", 100 + second, 0.1, second))

        rows = candles.snapshot()["candles"]

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["time"].second for row in rows], [54, 55])


if __name__ == "__main__":
    unittest.main()
