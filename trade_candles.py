"""Thread-safe aggregation of Kraken public trades into UTC one-second candles."""

import json
import math
import threading
from collections import deque
from datetime import datetime, timezone


KRAKEN_WS_URL = "wss://ws.kraken.com/v2"


def make_trade_subscription(symbol, request_id=1):
    return {
        "method": "subscribe",
        "params": {"channel": "trade", "symbol": [symbol], "snapshot": True},
        "req_id": request_id,
    }


def parse_trade_message(message):
    """Parse a Kraken v2 trade snapshot/update into validated trade records."""
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    payload = json.loads(message) if isinstance(message, str) else message
    if not isinstance(payload, dict):
        return []
    if payload.get("channel") != "trade" or payload.get("type") not in {"snapshot", "update"}:
        return []
    events = payload.get("data")
    if not isinstance(events, list):
        return []

    parsed = []
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            timestamp = datetime.fromisoformat(
                str(event["timestamp"]).replace("Z", "+00:00")
            )
            if timestamp.tzinfo is None:
                continue
            timestamp = timestamp.astimezone(timezone.utc)
            price = float(event["price"])
            quantity = float(event["qty"])
            trade_id = int(event["trade_id"])
            symbol = str(event["symbol"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not symbol or not math.isfinite(price) or price <= 0:
            continue
        if not math.isfinite(quantity) or quantity <= 0:
            continue
        parsed.append({
            "symbol": symbol,
            "timestamp": timestamp,
            "price": price,
            "qty": quantity,
            "trade_id": trade_id,
        })
    return parsed


class TradeCandleBuffer:
    """Aggregate exchange trades without filling seconds that had no trades."""

    def __init__(self, max_candles=300, max_seen_trade_ids=20000):
        self._max_candles = max(1, int(max_candles))
        self._max_seen_trade_ids = max(1, int(max_seen_trade_ids))
        self._candles = {}
        self._seen_trade_ids = set()
        self._seen_order = deque()
        self._last_trade_at = None
        self._lock = threading.Lock()

    def add_trade(self, trade):
        """Add one validated trade; return False for invalid or duplicate trades."""
        try:
            timestamp = trade["timestamp"]
            if timestamp.tzinfo is None:
                return False
            timestamp = timestamp.astimezone(timezone.utc)
            price = float(trade["price"])
            quantity = float(trade["qty"])
            trade_id = int(trade["trade_id"])
            symbol = str(trade["symbol"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not symbol or not math.isfinite(price) or price <= 0:
            return False
        if not math.isfinite(quantity) or quantity <= 0:
            return False

        trade_key = (symbol, trade_id)
        second = int(timestamp.timestamp())
        event_order = (timestamp.timestamp(), trade_id)
        with self._lock:
            if trade_key in self._seen_trade_ids:
                return False
            self._seen_trade_ids.add(trade_key)
            self._seen_order.append(trade_key)
            while len(self._seen_order) > self._max_seen_trade_ids:
                self._seen_trade_ids.discard(self._seen_order.popleft())

            candle = self._candles.get(second)
            if candle is None:
                self._candles[second] = {
                    "time": datetime.fromtimestamp(second, tz=timezone.utc),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": quantity,
                    "trade_count": 1,
                    "_first_order": event_order,
                    "_last_order": event_order,
                }
            else:
                candle["high"] = max(candle["high"], price)
                candle["low"] = min(candle["low"], price)
                candle["volume"] += quantity
                candle["trade_count"] += 1
                if event_order < candle["_first_order"]:
                    candle["open"] = price
                    candle["_first_order"] = event_order
                if event_order > candle["_last_order"]:
                    candle["close"] = price
                    candle["_last_order"] = event_order

            if self._last_trade_at is None or timestamp > self._last_trade_at:
                self._last_trade_at = timestamp
            while len(self._candles) > self._max_candles:
                del self._candles[min(self._candles)]
        return True

    def snapshot(self):
        with self._lock:
            candle_rows = []
            for second in sorted(self._candles):
                candle = self._candles[second]
                candle_rows.append({
                    key: value for key, value in candle.items()
                    if not key.startswith("_")
                })
            return {
                "candles": candle_rows,
                "last_trade_at": self._last_trade_at,
            }


class KrakenTradeFeed:
    """Maintain one reconnecting Kraken trade WebSocket in a daemon thread."""

    def __init__(self, symbol, reconnect_seconds=3, max_candles=300):
        self.symbol = symbol
        self._reconnect_seconds = max(1, float(reconnect_seconds))
        self._buffer = TradeCandleBuffer(max_candles=max_candles)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._websocket = None
        self._status = "not started"
        self._detail = "Waiting to connect to Kraken."

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._status = "connecting"
            self._detail = "Connecting to Kraken trade feed."
            self._thread = threading.Thread(
                target=self._run, name=f"kraken-trades-{self.symbol}", daemon=True
            )
            self._thread.start()

    def _set_status(self, status, detail):
        with self._lock:
            self._status = status
            self._detail = str(detail)[:240]

    def _run(self):
        try:
            import websocket
        except ImportError:
            self._set_status(
                "unavailable", "The websocket-client package is missing. Check app dependencies."
            )
            return

        while not self._stop_event.is_set():
            self._set_status("connecting", "Connecting to Kraken trade feed.")
            app = None
            try:
                app = websocket.WebSocketApp(
                    KRAKEN_WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                with self._lock:
                    self._websocket = app
                app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                self._set_status("reconnecting", f"Kraken connection error: {exc}")
            finally:
                with self._lock:
                    if self._websocket is app:
                        self._websocket = None
            if not self._stop_event.is_set():
                self._set_status("reconnecting", "Connection ended; retrying shortly.")
                self._stop_event.wait(self._reconnect_seconds)

    def _on_open(self, websocket_app):
        self._set_status("subscribing", f"Subscribing to {self.symbol} trades.")
        try:
            websocket_app.send(json.dumps(make_trade_subscription(self.symbol)))
        except Exception as exc:
            self._set_status("reconnecting", f"Could not subscribe to Kraken: {exc}")
            websocket_app.close()

    def _on_message(self, websocket_app, message):
        try:
            payload = json.loads(message)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return

        if payload.get("method") == "subscribe":
            result = payload.get("result") or {}
            if payload.get("success") is True and result.get("channel") == "trade":
                self._set_status("connected", f"Receiving real-time {self.symbol} trades from Kraken.")
            elif payload.get("success") is False:
                detail = payload.get("error") or f"Kraken rejected the {self.symbol} subscription."
                self._set_status("unavailable", detail)
                self._stop_event.set()
                websocket_app.close()
            return

        if payload.get("channel") == "trade":
            accepted = 0
            for trade in parse_trade_message(payload):
                if trade["symbol"] == self.symbol and self._buffer.add_trade(trade):
                    accepted += 1
            if accepted:
                self._set_status("connected", f"Receiving real-time {self.symbol} trades from Kraken.")

    def _on_error(self, _websocket_app, error):
        self._set_status("reconnecting", f"Kraken connection error: {error}")

    def _on_close(self, _websocket_app, _status_code, _message):
        if self._stop_event.is_set():
            with self._lock:
                if self._status != "unavailable":
                    self._status = "stopped"
                    self._detail = "Trade feed stopped."
        else:
            self._set_status("reconnecting", "Kraken connection closed; retrying shortly.")

    def stop(self):
        self._stop_event.set()
        with self._lock:
            websocket_app = self._websocket
            if self._status != "unavailable":
                self._status = "stopped"
                self._detail = "Trade feed stopped."
        if websocket_app is not None:
            websocket_app.close()

    def snapshot(self):
        with self._lock:
            status = self._status
            detail = self._detail
        data = self._buffer.snapshot()
        data.update({"status": status, "detail": detail})
        return data
