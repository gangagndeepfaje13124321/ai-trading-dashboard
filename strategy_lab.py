"""Small, transparent model selection and chronological holdout for Kraken bars.

This is a rule-selection experiment, not a neural network. Candidate rules are
fit on the earlier part of the available history and measured once on its later
holdout. The holdout is never used to select the live rule or its expected move.
"""

from bisect import bisect_left
from datetime import timedelta
import math


MODEL_LABELS = {
    "momentum": "Recent momentum",
    "ema_trend": "EMA trend (8/21)",
    "mean_reversion": "Short-term mean reversion",
    "range_breakout": "10-candle range breakout",
}
MIN_HISTORY = 120
MIN_TRAIN_CALLS = 12


def _clean(candles):
    """Return valid, unique, UTC-aware OHLC rows in timestamp order."""
    by_time = {}
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
        values = (row["open"], row["high"], row["low"], row["close"], row["volume"])
        if not all(math.isfinite(value) for value in values):
            continue
        if min(row["open"], row["close"]) < row["low"] or max(row["open"], row["close"]) > row["high"]:
            continue
        if row["close"] <= 0 or row["low"] <= 0:
            continue
        by_time[row["time"]] = row
    return [by_time[key] for key in sorted(by_time)]


def _median(values):
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return 0.0
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return (clean[middle - 1] + clean[middle]) / 2


def _ema(values, span):
    if not values:
        return None
    alpha = 2 / (span + 1)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1 - alpha) * result
    return result


def _sign(value, threshold):
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _prior_index(rows, index, horizon_minutes, times=None):
    """Find the closest candle at or just before the requested lookback time."""
    target_time = rows[index]["time"] - timedelta(minutes=horizon_minutes)
    times = times if times is not None else [row["time"] for row in rows[:index + 1]]
    prior = bisect_left(times, target_time + timedelta(microseconds=1)) - 1
    if prior < 0 or target_time - rows[prior]["time"] > timedelta(minutes=1):
        return None
    return prior


def _signal(name, rows, index, horizon_minutes, times=None):
    """Return a rule's direction using information available at `index`."""
    current = rows[index]
    price = current["close"]
    if price <= 0:
        return 0
    ranges = [row["high"] - row["low"] for row in rows[max(0, index - 14):index + 1]]
    noise = max(_median(ranges) * 0.08, price * 0.0000001)

    if name == "momentum":
        prior_index = _prior_index(rows, index, horizon_minutes, times)
        if prior_index is None:
            return 0
        return _sign(price - rows[prior_index]["close"], noise * 0.1)

    if name == "ema_trend":
        if index < 21:
            return 0
        closes = [row["close"] for row in rows[max(0, index - 59):index + 1]]
        difference = _ema(closes[-8:], 8) - _ema(closes, 21)
        return _sign(difference, noise * 0.05)

    if name == "mean_reversion":
        lookback = max(3, horizon_minutes * 2)
        prior_index = max(0, index - lookback)
        movement = price - rows[prior_index]["close"]
        threshold = max(noise * 0.5, price * 0.000001)
        return -_sign(movement, threshold)

    if name == "range_breakout":
        if index < 10:
            return 0
        previous = rows[index - 10:index]
        prior_high = max(row["high"] for row in previous)
        prior_low = min(row["low"] for row in previous)
        return _sign(price - prior_high, noise * 0.05) if price > prior_high else (
            -1 if price < prior_low - noise * 0.05 else 0
        )

    return 0


def _target_index(rows, times, origin_index, horizon_minutes):
    target_time = rows[origin_index]["time"] + timedelta(minutes=horizon_minutes)
    index = bisect_left(times, target_time, lo=origin_index + 1)
    if index >= len(rows):
        return None
    # Skip large gaps rather than pretending a late candle was the requested horizon.
    if rows[index]["time"] > target_time + timedelta(minutes=1):
        return None
    return index


def _training_records(rows, model, horizon, warmup, end_index):
    times = [row["time"] for row in rows]
    records = []
    for index in range(warmup, end_index, horizon):
        target_index = _target_index(rows, times, index, horizon)
        if target_index is None or target_index >= end_index:
            continue
        direction = _signal(model, rows, index, horizon, times)
        if not direction:
            continue
        origin = rows[index]["close"]
        actual_return = (rows[target_index]["close"] - origin) / origin
        records.append({"direction": direction, "actual_return": actual_return})
    return records


def _fit_returns(records):
    return {
        direction: _median([row["actual_return"] for row in records if row["direction"] == direction])
        for direction in (-1, 1)
        if any(row["direction"] == direction for row in records)
    }


def _evaluate_holdout(rows, model, expected_returns, horizon, start_index, warmup):
    times = [row["time"] for row in rows]
    calls = wins = momentum_wins = 0
    origins = 0
    momentum_calls = 0
    model_errors = []
    model_last_close_errors = []
    naive_errors = []
    momentum_errors = []
    for index in range(max(start_index, warmup), len(rows), horizon):
        target_index = _target_index(rows, times, index, horizon)
        if target_index is None:
            continue
        origin = rows[index]["close"]
        actual = rows[target_index]["close"]
        actual_return = (actual - origin) / origin
        origins += 1
        naive_errors.append(abs(actual_return) * 100)

        momentum_direction = _signal("momentum", rows, index, horizon, times)
        momentum_prior_index = _prior_index(rows, index, horizon, times)
        momentum_prior = rows[momentum_prior_index]["close"] if momentum_prior_index is not None else origin
        momentum_return = (origin - momentum_prior) / momentum_prior
        momentum_errors.append(abs(momentum_return - actual_return) * 100)
        if momentum_direction:
            momentum_calls += 1
            momentum_wins += int(momentum_direction == _sign(actual_return, 1e-12))

        direction = _signal(model, rows, index, horizon, times)
        expected_return = expected_returns.get(direction) if direction else None
        if expected_return is None:
            continue
        calls += 1
        wins += int(direction == _sign(actual_return, 1e-12))
        model_errors.append(abs(expected_return - actual_return) * 100)
        model_last_close_errors.append(abs(actual_return) * 100)

    lower_bound = _wilson_lower_bound_pct(wins, calls)

    return {
        "origins": origins,
        "calls": calls,
        "directional_accuracy_pct": wins / calls * 100 if calls else None,
        "coverage_pct": calls / origins * 100 if origins else None,
        "forecast_mae_pct": sum(model_errors) / len(model_errors) if model_errors else None,
        "last_close_mae_on_calls_pct": (
            sum(model_last_close_errors) / len(model_last_close_errors)
            if model_last_close_errors else None
        ),
        "last_close_mae_pct": sum(naive_errors) / len(naive_errors) if naive_errors else None,
        "directional_accuracy_lower_95_pct": lower_bound,
        "momentum_mae_pct": sum(momentum_errors) / len(momentum_errors) if momentum_errors else None,
        "momentum_directional_accuracy_pct": momentum_wins / momentum_calls * 100 if momentum_calls else None,
        "momentum_calls": momentum_calls,
    }


def _wilson_lower_bound_pct(wins, samples, z=1.96):
    """Conservative lower end of a 95% Wilson interval for hit rate."""
    if samples <= 0:
        return None
    rate = wins / samples
    z2 = z * z
    denominator = 1 + z2 / samples
    center = rate + z2 / (2 * samples)
    margin = z * math.sqrt(rate * (1 - rate) / samples + z2 / (4 * samples * samples))
    return max(0.0, (center - margin) / denominator * 100)


def fit_and_evaluate(candles, horizon_minutes=1):
    """Choose a rule on earlier bars; evaluate it on a later untouched block."""
    rows = _clean(candles)
    horizon = max(1, int(horizon_minutes))
    if len(rows) < MIN_HISTORY:
        return {"ready": False, "bars": len(rows), "minimum_bars": MIN_HISTORY}

    warmup = max(30, horizon * 5)
    split = int(len(rows) * 0.70)
    if split <= warmup + horizon * MIN_TRAIN_CALLS:
        return {"ready": False, "bars": len(rows), "minimum_bars": warmup + horizon * MIN_TRAIN_CALLS + 1}

    trained = {}
    for name in MODEL_LABELS:
        records = _training_records(rows, name, horizon, warmup, split)
        if len(records) < MIN_TRAIN_CALLS:
            continue
        expected = _fit_returns(records)
        if not expected:
            continue
        errors = [abs(row["actual_return"] - expected[row["direction"]]) * 100 for row in records]
        trained[name] = {
            "returns": expected,
            "train_calls": len(records),
            "train_mae_pct": sum(errors) / len(errors),
            "train_directional_accuracy_pct": sum(
                _sign(row["actual_return"], 1e-12) == row["direction"] for row in records
            ) / len(records) * 100,
        }

    if not trained:
        return {"ready": False, "bars": len(rows), "minimum_bars": MIN_HISTORY, "reason": "Not enough directional examples."}

    # Select using only the earlier training block. A small hit-rate tie-breaker
    # prefers comparable-error rules with more directionally useful outcomes.
    selected = min(
        trained,
        key=lambda name: (
            trained[name]["train_mae_pct"],
            -trained[name]["train_directional_accuracy_pct"],
            -trained[name]["train_calls"],
        ),
    )
    selected_fit = trained[selected]
    typical_range_pct = _median([
        (row["high"] - row["low"]) / row["close"]
        for row in rows[max(0, warmup):split]
        if row["close"] > 0
    ])
    holdout = _evaluate_holdout(
        rows, selected, selected_fit["returns"], horizon, split, warmup
    )

    return {
        "ready": True,
        "model": selected,
        "model_label": MODEL_LABELS[selected],
        "horizon_minutes": horizon,
        "bars": len(rows),
        "train_bars": split,
        "holdout_bars": len(rows) - split,
        "train_calls": selected_fit["train_calls"],
        "train_directional_accuracy_pct": selected_fit["train_directional_accuracy_pct"],
        "train_mae_pct": selected_fit["train_mae_pct"],
        "expected_returns": selected_fit["returns"],
        "typical_range_pct": typical_range_pct,
        "holdout": holdout,
        "history_start": rows[0]["time"],
        "history_end": rows[-1]["time"],
    }


def make_minute_forecast(candles, fitted):
    """Create OHLC scenario bars using the selected, training-only rule fit."""
    rows = _clean(candles)
    if not rows or not fitted or not fitted.get("ready"):
        return None
    horizon = max(1, int(fitted["horizon_minutes"]))
    model = fitted["model"]
    scenario_direction = _signal(model, rows, len(rows) - 1, horizon)
    expected_return = fitted["expected_returns"].get(scenario_direction) if scenario_direction else 0.0
    # The model may find that this rule's historical direction had no same-way
    # expected move. In that case display neutral scenario candles, not a false call.
    if expected_return is not None and scenario_direction and _sign(expected_return, 1e-12) != scenario_direction:
        scenario_direction = 0
        expected_return = 0.0
    if expected_return is None:
        scenario_direction = 0
        expected_return = 0.0

    holdout = fitted.get("holdout", {})
    fair_mae = holdout.get("last_close_mae_on_calls_pct")
    forecast_mae = holdout.get("forecast_mae_pct")
    signal_validated = (
        holdout.get("calls", 0) >= 30
        and holdout.get("directional_accuracy_pct") is not None
        and holdout["directional_accuracy_pct"] >= 55.0
        and forecast_mae is not None
        and fair_mae is not None
        and forecast_mae < fair_mae
    )
    direction = scenario_direction if signal_validated else 0

    start = rows[-1]["close"]
    total_move = start * expected_return
    volatility = max(fitted.get("typical_range_pct", 0.0) * start, start * 0.000002)
    previous_close = start
    forecast_candles = []
    for step in range(1, horizon + 1):
        close = start + total_move * step / horizon
        width = volatility * math.sqrt(step) * 0.45
        forecast_candles.append({
            "time": rows[-1]["time"] + timedelta(minutes=step),
            "open": previous_close,
            "high": max(previous_close, close) + width,
            "low": min(previous_close, close) - width,
            "close": close,
        })
        previous_close = close

    return {
        "time": rows[-1]["time"],
        "entry_price": start,
        "direction": direction,
        "scenario_direction": scenario_direction,
        "signal_validated": signal_validated,
        "model": model,
        "model_label": fitted["model_label"],
        "expected_return_pct": expected_return * 100,
        "forecast_candles": forecast_candles,
        "note": "OHLC scenario based on a rule selected from earlier candles; wick size reflects typical past range, not a probability interval.",
    }
