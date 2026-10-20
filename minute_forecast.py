"""Transparent multi-signal watch scenarios for one-minute Kraken candles.

This is a fixed-rule ensemble, not a trained model or probability estimate.
The historical checker uses a chronological walk-forward and non-overlapping
forecast targets so each displayed result has an understandable meaning.
"""

from datetime import timedelta
import math

from live_forecast import _setup


MIN_HISTORY = 30
DEFAULT_HORIZON_MINUTES = 3


def _clean(candles):
    rows = []
    for candle in candles:
        try:
            timestamp = candle["time"]
            if timestamp.tzinfo is None:
                continue
            row = {
                "time": timestamp,
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": float(candle.get("volume", 0.0)),
            }
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(row[key]) for key in ("open", "high", "low", "close", "volume")):
            continue
        if row["close"] <= 0 or row["low"] > min(row["open"], row["close"]):
            continue
        if row["high"] < max(row["open"], row["close"]):
            continue
        rows.append(row)
    rows.sort(key=lambda row: row["time"])
    return rows


def _median(values):
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def _ema(values, span):
    if not values:
        return None
    alpha = 2 / (span + 1)
    value = values[0]
    for current in values[1:]:
        value = alpha * current + (1 - alpha) * value
    return value


def _rsi(closes, period=14):
    changes = [current - previous for previous, current in zip(closes[-period - 1:-1], closes[-period:])]
    if not changes:
        return 50.0
    gains = sum(max(0.0, change) for change in changes) / len(changes)
    losses = sum(max(0.0, -change) for change in changes) / len(changes)
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    relative_strength = gains / losses
    return 100 - (100 / (1 + relative_strength))


def _true_ranges(rows):
    values = []
    previous_close = None
    for row in rows:
        span = row["high"] - row["low"]
        if previous_close is not None:
            span = max(span, abs(row["high"] - previous_close), abs(row["low"] - previous_close))
        values.append(span)
        previous_close = row["close"]
    return values


def _sign(value, threshold=0.0):
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def make_minute_forecast(candles, horizon_minutes=DEFAULT_HORIZON_MINUTES):
    """Blend distinct price-based signals into a direction and scenario range."""
    rows = _clean(candles)
    if len(rows) < MIN_HISTORY:
        return None

    horizon = max(1, int(horizon_minutes))
    closes = [row["close"] for row in rows]
    latest = rows[-1]
    price = latest["close"]
    true_ranges = _true_ranges(rows)
    atr = _median(true_ranges[-14:])
    noise_floor = max(atr * 0.08, price * 0.000001)
    signals = []

    fast_ema = _ema(closes[-60:], 9)
    slow_ema = _ema(closes[-60:], 21)
    ema_direction = _sign(fast_ema - slow_ema, noise_floor * 0.15)
    signals.append({
        "name": "EMA trend (9/21)",
        "direction": ema_direction,
        "detail": "Fast EMA above slow EMA" if ema_direction > 0 else (
            "Fast EMA below slow EMA" if ema_direction < 0 else "EMAs are close together"
        ),
    })

    for window in (3, 10):
        change = price - closes[-window - 1]
        threshold = noise_floor * math.sqrt(window) * 0.15
        direction = _sign(change, threshold)
        signals.append({
            "name": f"{window}-minute momentum",
            "direction": direction,
            "detail": f"Price is {abs(change) / price * 100:.3f}% {'higher' if change > 0 else 'lower'} than {window} minutes ago"
            if direction else "Movement is small relative to recent noise",
        })

    rsi = _rsi(closes)
    rsi_direction = 1 if rsi <= 30 else (-1 if rsi >= 70 else 0)
    signals.append({
        "name": "RSI(14) reversal check",
        "direction": rsi_direction,
        "detail": f"RSI {rsi:.0f}: oversold area" if rsi_direction > 0 else (
            f"RSI {rsi:.0f}: overbought area" if rsi_direction < 0 else f"RSI {rsi:.0f}: neither extreme"
        ),
    })

    previous = rows[-21:-1]
    if len(previous) == 20:
        prior_high = max(row["high"] for row in previous)
        prior_low = min(row["low"] for row in previous)
        breakout_buffer = max(atr * 0.05, price * 0.000001)
        breakout_direction = 1 if price > prior_high + breakout_buffer else (
            -1 if price < prior_low - breakout_buffer else 0
        )
        breakout_detail = "Broke above the previous 20-minute range" if breakout_direction > 0 else (
            "Broke below the previous 20-minute range" if breakout_direction < 0 else "Inside the previous 20-minute range"
        )
    else:
        breakout_direction, breakout_detail = 0, "Not enough candles"
    signals.append({"name": "20-minute range breakout", "direction": breakout_direction, "detail": breakout_detail})

    vwap_rows = rows[-20:]
    total_volume = sum(row["volume"] for row in vwap_rows)
    if total_volume > 0:
        vwap = sum(((row["high"] + row["low"] + row["close"]) / 3) * row["volume"] for row in vwap_rows) / total_volume
        vwap_direction = _sign(price - vwap, noise_floor * 0.05)
        vwap_detail = f"Price is {'above' if vwap_direction > 0 else 'below' if vwap_direction < 0 else 'near'} 20-minute VWAP"
    else:
        vwap_direction, vwap_detail = 0, "Volume unavailable for VWAP"
    signals.append({"name": "20-minute volume-weighted price", "direction": vwap_direction, "detail": vwap_detail})

    pattern = _setup(rows)
    pattern_direction = pattern["direction"] if pattern["method"] == "pattern" else 0
    pattern_label = f"{pattern['name']} — {pattern['state']}" if pattern["name"] else "No classic pattern confirmed"
    signals.append({
        "name": "Confirmed chart-pattern break",
        "direction": pattern_direction,
        "detail": pattern_label,
    })

    votes = [signal["direction"] for signal in signals if signal["direction"]]
    up_votes = sum(direction > 0 for direction in votes)
    down_votes = sum(direction < 0 for direction in votes)
    active_votes = len(votes)
    winning_votes = max(up_votes, down_votes)
    agreement = (winning_votes / active_votes) if active_votes else 0.0
    if active_votes >= 2 and agreement >= 0.60 and up_votes != down_votes:
        direction = 1 if up_votes > down_votes else -1
        bias = "Upward bias" if direction > 0 else "Downward bias"
    else:
        direction = 0
        bias = "Mixed / no clear bias"

    # A visible range is more honest than drawing a deterministic curved price
    # path. This range is based on observed ATR and is not a calibrated interval.
    range_radius = max(atr * math.sqrt(horizon), price * 0.00002)
    center = price + direction * range_radius * 0.12
    return {
        "time": latest["time"],
        "price": price,
        "horizon_minutes": horizon,
        "direction": direction,
        "bias": bias,
        "up_votes": up_votes,
        "down_votes": down_votes,
        "active_votes": active_votes,
        "agreement_pct": agreement * 100,
        "signals": signals,
        "pattern": pattern_label,
        "rsi": rsi,
        "atr": atr,
        "scenario_low": center - range_radius,
        "scenario_high": center + range_radius,
        "range_radius": range_radius,
        "range_note": "Observed-volatility scenario, not a probability interval or exact target.",
        "is_trained_model": False,
    }


def walk_forward_direction_test(candles, horizon_minutes=DEFAULT_HORIZON_MINUTES, warmup=60):
    """Evaluate fixed rules on later candles and compare with momentum persistence.

    Forecasts are issued every `horizon_minutes`, so target windows do not
    overlap. At each origin the rules see only candles at or before that time.
    Neutral model outputs are not counted as directional calls.
    """
    rows = _clean(candles)
    horizon = max(1, int(horizon_minutes))
    warmup = max(MIN_HISTORY, int(warmup))
    calls = wins = baseline_wins = origins = 0
    index = warmup
    while index < len(rows):
        origin = rows[index]
        target_time = origin["time"] + timedelta(minutes=horizon)
        target_index = next(
            (i for i in range(index + 1, len(rows)) if rows[i]["time"] >= target_time),
            None,
        )
        if target_index is None:
            break
        target = rows[target_index]
        if target["time"] > target_time + timedelta(minutes=1):
            index += horizon
            continue

        forecast = make_minute_forecast(rows[:index + 1], horizon)
        if forecast is not None:
            actual_direction = _sign(target["close"] - origin["close"], origin["close"] * 1e-10)
            baseline_origin_time = origin["time"] - timedelta(minutes=horizon)
            baseline_origin = next(
                (row for row in reversed(rows[:index]) if row["time"] <= baseline_origin_time),
                None,
            )
            baseline_direction = (
                _sign(origin["close"] - baseline_origin["close"], origin["close"] * 1e-10)
                if baseline_origin else 0
            )
            origins += 1
            if forecast["direction"]:
                calls += 1
                wins += forecast["direction"] == actual_direction
                baseline_wins += baseline_direction == actual_direction
        index += horizon

    return {
        "origins": origins,
        "calls": calls,
        "wins": wins,
        "hit_rate_pct": wins / calls * 100 if calls else None,
        "baseline_hit_rate_pct": baseline_wins / calls * 100 if calls else None,
        "baseline_wins": baseline_wins,
        "horizon_minutes": horizon,
        "validated": False,
    }
