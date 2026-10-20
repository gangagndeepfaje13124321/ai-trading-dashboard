"""Kraken one-minute OHLC history and live trade aggregation helpers."""

from datetime import datetime, timezone
import json
import math
from urllib.parse import urlencode
from urllib.request import urlopen



KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"


def parse_kraken_ohlc(payload):
    """Parse Kraken's public OHLC response into UTC candle dictionaries."""
    if not isinstance(payload, dict):
        raise ValueError("Kraken returned an invalid OHLC response.")
    errors = payload.get("error") or []
    if errors:
        raise RuntimeError("Kraken OHLC error: " + "; ".join(map(str, errors)))
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("Kraken OHLC response has no result data.")

    raw_rows = next((value for value in result.values() if isinstance(value, list)), None)
    if raw_rows is None:
        raise ValueError("Kraken OHLC response contains no candle list.")

    candles = []
    for item in raw_rows:
        if not isinstance(item, (list, tuple)) or len(item) < 8:
            continue
        try:
            timestamp = int(item[0])
            row = {
                "time": datetime.fromtimestamp(timestamp, tz=timezone.utc),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[6]),
                "trade_count": int(item[7]),
            }
        except (TypeError, ValueError, OverflowError, OSError):
            continue
        values = (row["open"], row["high"], row["low"], row["close"], row["volume"])
        if not all(math.isfinite(value) for value in values):
            continue
        if row["low"] > min(row["open"], row["close"]) or row["high"] < max(row["open"], row["close"]):
            continue
        candles.append(row)

    candles.sort(key=lambda candle: candle["time"])
    if not candles:
        raise ValueError("Kraken returned no valid one-minute candles.")
    return candles


def fetch_kraken_minute_history(pair, timeout=8):
    """Fetch Kraken's latest 720 one-minute candles; no API key is required."""
    query = urlencode({"pair": pair, "interval": 1, "assetVersion": 1})
    with urlopen(f"{KRAKEN_OHLC_URL}?{query}", timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return parse_kraken_ohlc(payload)


def aggregate_trade_candles(second_candles):
    """Aggregate trade-based one-second candles into observed UTC minute bars."""
    buckets = {}
    for source in sorted(second_candles, key=lambda row: row.get("time")):
        try:
            timestamp = source["time"]
            if timestamp.tzinfo is None:
                continue
            minute_epoch = int(timestamp.timestamp()) // 60 * 60
            candle = {
                "time": datetime.fromtimestamp(minute_epoch, tz=timezone.utc),
                "open": float(source["open"]),
                "high": float(source["high"]),
                "low": float(source["low"]),
                "close": float(source["close"]),
                "volume": float(source.get("volume", 0.0)),
                "trade_count": int(source.get("trade_count", 0)),
            }
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError, OSError):
            continue
        if not all(math.isfinite(candle[key]) for key in ("open", "high", "low", "close", "volume")):
            continue

        bucket = buckets.get(minute_epoch)
        if bucket is None:
            buckets[minute_epoch] = candle
        else:
            bucket["high"] = max(bucket["high"], candle["high"])
            bucket["low"] = min(bucket["low"], candle["low"])
            bucket["close"] = candle["close"]
            bucket["volume"] += candle["volume"]
            bucket["trade_count"] += candle["trade_count"]
    return [buckets[key] for key in sorted(buckets)]


def merge_history_with_live_trades(history, second_candles, max_candles=720):
    """Keep REST history and overlay only the current/newer streamed minutes.

    REST provides complete historical minute candles. The live trade stream
    updates the current minute each second; older streamed minutes are ignored
    so the 50-trade WebSocket startup snapshot cannot replace complete history
    with a partial bar.
    """
    by_time = {}
    for source in history:
        try:
            timestamp = source["time"]
            if timestamp.tzinfo is None:
                continue
            row = dict(source)
            row["time"] = timestamp.astimezone(timezone.utc).replace(second=0, microsecond=0)
            by_time[row["time"]] = row
        except (AttributeError, KeyError, TypeError, ValueError):
            continue

    latest_history_time = max(by_time) if by_time else None
    for live in aggregate_trade_candles(second_candles):
        timestamp = live["time"]
        if latest_history_time is not None and timestamp < latest_history_time:
            continue
        previous = by_time.get(timestamp)
        if previous:
            # Retain the exchange candle's opening price and volume baseline;
            # the WebSocket extends its range and refreshes its latest close.
            previous["high"] = max(previous["high"], live["high"])
            previous["low"] = min(previous["low"], live["low"])
            previous["close"] = live["close"]
        else:
            by_time[timestamp] = live

    result = [by_time[key] for key in sorted(by_time)]
    return result[-max(1, int(max_candles)):]
