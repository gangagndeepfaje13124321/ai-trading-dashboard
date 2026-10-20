"""Alpaca stock trades and minute bars for AAPL/TSLA dashboard views.

This uses Alpaca's authenticated IEX market-data feed. It is not a full-market
SIP feed and the dashboard never uses this module to place orders.
"""

import json
import math
import threading
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from trade_candles import TradeCandleBuffer


DATA_URL = "https://data.alpaca.markets"
STREAM_URL = "wss://stream.data.alpaca.markets/v2/iex"


def alpaca_headers(api_key, secret_key):
    if not api_key or not secret_key:
        raise ValueError("Alpaca API key and secret are required for US stock data.")
    return {
        "APCA-API-KEY-ID": str(api_key),
        "APCA-API-SECRET-KEY": str(secret_key),
    }


def parse_alpaca_trade_messages(payload, symbol):
    """Convert Alpaca websocket trade messages to the common candle format."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    parsed = []
    requested_symbol = str(symbol).upper()
    for event in payload:
        if not isinstance(event, dict) or event.get("T") != "t":
            continue
        try:
            event_symbol = str(event["S"]).upper()
            timestamp = datetime.fromisoformat(str(event["t"]).replace("Z", "+00:00"))
            if timestamp.tzinfo is None or event_symbol != requested_symbol:
                continue
            price = float(event["p"])
            size = float(event["s"])
            trade_id = int(event["i"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(price) or price <= 0 or not math.isfinite(size) or size <= 0:
            continue
        parsed.append({
            "symbol": event_symbol,
            "timestamp": timestamp.astimezone(timezone.utc),
            "price": price,
            "qty": size,
            "trade_id": trade_id,
        })
    return parsed


def parse_alpaca_minute_bars(payload, symbol):
    """Parse a one-symbol Alpaca historical-bars response into OHLC rows."""
    if not isinstance(payload, dict) or not isinstance(payload.get("bars"), dict):
        raise ValueError("Alpaca returned an invalid stock-bars response.")
    raw_bars = payload["bars"].get(str(symbol).upper(), [])
    rows = []
    for bar in raw_bars:
        try:
            timestamp = datetime.fromisoformat(str(bar["t"]).replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                continue
            row = {
                "time": timestamp.astimezone(timezone.utc),
                "open": float(bar["o"]),
                "high": float(bar["h"]),
                "low": float(bar["l"]),
                "close": float(bar["c"]),
                "volume": float(bar.get("v", 0)),
                "trade_count": int(bar.get("n", 0)),
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        values = (row["open"], row["high"], row["low"], row["close"], row["volume"])
        if not all(math.isfinite(value) for value in values):
            continue
        if row["low"] <= 0 or row["low"] > min(row["open"], row["close"]):
            continue
        if row["high"] < max(row["open"], row["close"]):
            continue
        rows.append(row)
    return sorted(rows, key=lambda row: row["time"])


def fetch_alpaca_minute_history(symbol, api_key, secret_key, days=30, timeout=12):
    """Fetch up to 10,000 recent one-minute IEX bars for walk-forward testing."""
    ticker = str(symbol).upper().strip()
    headers = alpaca_headers(api_key, secret_key)
    start = (datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))).isoformat()
    query = urlencode({
            "symbols": ticker,
            "timeframe": "1Min",
            "start": start,
            "limit": 10000,
            "feed": "iex",
            "sort": "asc",
            "adjustment": "split",
        })
    request = Request(f"{DATA_URL}/v2/stocks/bars?{query}", headers=headers)
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = parse_alpaca_minute_bars(payload, ticker)
    if not rows:
        raise ValueError(f"Alpaca returned no one-minute IEX candles for {ticker}.")
    return rows


class AlpacaTradeFeed:
    """Reconnectable IEX trade websocket, aggregated into actual 1s candles."""

    def __init__(self, symbol, api_key, secret_key, reconnect_seconds=3, max_candles=900):
        self.symbol = str(symbol).upper().strip()
        self.api_key = str(api_key)
        self.secret_key = str(secret_key)
        self._reconnect_seconds = max(1, float(reconnect_seconds))
        self._buffer = TradeCandleBuffer(max_candles=max_candles)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._websocket = None
        self._status = "not started"
        self._detail = "Waiting to connect to Alpaca's IEX stock feed."

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._status = "connecting"
            self._detail = "Connecting to Alpaca's IEX stock feed."
            self._thread = threading.Thread(
                target=self._run, name=f"alpaca-iex-trades-{self.symbol}", daemon=True
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
            self._set_status("unavailable", "The websocket-client package is missing.")
            return

        while not self._stop_event.is_set():
            self._set_status("connecting", "Connecting to Alpaca's IEX stock feed.")
            app = None
            try:
                app = websocket.WebSocketApp(
                    STREAM_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                with self._lock:
                    self._websocket = app
                app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                self._set_status("reconnecting", f"Alpaca connection error: {exc}")
            finally:
                with self._lock:
                    if self._websocket is app:
                        self._websocket = None
            if not self._stop_event.is_set():
                self._set_status("reconnecting", "Connection ended; retrying shortly.")
                self._stop_event.wait(self._reconnect_seconds)

    def _on_open(self, websocket_app):
        self._set_status("authenticating", "Authenticating with Alpaca for IEX market data.")
        try:
            websocket_app.send(json.dumps({
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key,
            }))
        except Exception as exc:
            self._set_status("reconnecting", f"Could not authenticate with Alpaca: {exc}")
            websocket_app.close()

    def _on_message(self, websocket_app, message):
        try:
            payload = json.loads(message)
        except (TypeError, ValueError):
            return
        events = payload if isinstance(payload, list) else [payload]
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("T") == "success" and event.get("msg") == "authenticated":
                websocket_app.send(json.dumps({
                    "action": "subscribe",
                    "trades": [self.symbol],
                }))
                continue
            if event.get("T") == "subscription":
                if self.symbol in (event.get("trades") or []):
                    self._set_status("connected", f"Receiving {self.symbol} trades from Alpaca IEX.")
                continue
            if event.get("T") == "error":
                self._set_status("unavailable", event.get("msg", "Alpaca rejected the data request."))
                self._stop_event.set()
                websocket_app.close()
                return

        accepted = 0
        for trade in parse_alpaca_trade_messages(events, self.symbol):
            if self._buffer.add_trade(trade):
                accepted += 1
        if accepted:
            self._set_status("connected", f"Receiving {self.symbol} trades from Alpaca IEX.")

    def _on_error(self, _websocket_app, error):
        self._set_status("reconnecting", f"Alpaca connection error: {error}")

    def _on_close(self, _websocket_app, _status_code, _message):
        if self._stop_event.is_set():
            with self._lock:
                if self._status != "unavailable":
                    self._status = "stopped"
                    self._detail = "Stock trade feed stopped."
        else:
            self._set_status("reconnecting", "Alpaca connection closed; retrying shortly.")

    def stop(self):
        self._stop_event.set()
        with self._lock:
            websocket_app = self._websocket
            if self._status != "unavailable":
                self._status = "stopped"
                self._detail = "Stock trade feed stopped."
        if websocket_app is not None:
            websocket_app.close()

    def snapshot(self):
        with self._lock:
            status = self._status
            detail = self._detail
        data = self._buffer.snapshot()
        data.update({"status": status, "detail": detail})
        return data
