"""Transparent short-horizon forecasts and online scoring for Kraken trade candles.

This module detects a small set of candle-pattern breakouts and otherwise falls
back to a short momentum baseline. It does not place orders or estimate profit.
"""

from collections import deque
from datetime import timedelta
import math
import threading


DEFAULT_HORIZON_SECONDS = 5
DEFAULT_MIN_PATTERN_SAMPLES = 200
REQUIRED_PATTERN_ACCURACY_PCT = 55.0


def _clean_candles(candles):
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
                "trade_count": int(candle.get("trade_count", 0)),
            }
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            continue
        values = (row["open"], row["high"], row["low"], row["close"], row["volume"])
        if not all(math.isfinite(value) for value in values):
            continue
        if row["low"] > min(row["open"], row["close"]):
            continue
        if row["high"] < max(row["open"], row["close"]):
            continue
        rows.append(row)
    return sorted(rows, key=lambda row: row["time"])


def _median(values):
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def _swing_points(values, kind):
    """Return simple one-bar local turning points as (index, price) pairs."""
    points = []
    for index in range(1, len(values) - 1):
        before, value, after = values[index - 1], values[index], values[index + 1]
        if kind == "high":
            is_pivot = value >= before and value >= after and (value > before or value > after)
        else:
            is_pivot = value <= before and value <= after and (value < before or value < after)
        if is_pivot:
            points.append((index, float(value)))
    return points


def _line_slope(points):
    if len(points) < 2:
        return 0.0
    first_x, first_y = points[0]
    last_x, last_y = points[-1]
    return (last_y - first_y) / max(1, last_x - first_x)


def _setup(rows):
    """Detect simple, confirmed short-term chart-pattern breakouts.

    These are transparent price-geometry rules, not a trained or predictive
    pattern-recognition model. A name is shown while a shape forms, but a
    directional pattern result is issued only after its boundary breaks.
    """
    neutral = {"name": None, "state": "none", "direction": 0, "method": "none"}
    if len(rows) < 16:
        return neutral

    history = rows[-41:-1]
    current = rows[-1]
    if len(history) < 14:
        return neutral

    close = current["close"]
    ranges = [max(0.0, row["high"] - row["low"]) for row in history[-30:]]
    typical_range = _median(ranges)
    tolerance = max(typical_range * 1.25, abs(close) * 0.00002)
    breakout_buffer = max(typical_range * 0.2, abs(close) * 0.000005)
    highs = [row["high"] for row in history]
    lows = [row["low"] for row in history]
    high_pivots = _swing_points(highs, "high")[-4:]
    low_pivots = _swing_points(lows, "low")[-4:]

    # Head-and-shoulders and inverse head-and-shoulders need three distinct
    # turning points and a neckline break before they get a direction.
    all_high_pivots = _swing_points(highs, "high")
    if len(all_high_pivots) >= 3:
        left, head, right = all_high_pivots[-3:]
        shoulders_match = abs(left[1] - right[1]) <= tolerance * 2
        head_is_higher = head[1] >= max(left[1], right[1]) + tolerance * 0.75
        if right[0] - left[0] >= 6 and shoulders_match and head_is_higher:
            neckline = min(min(lows[left[0]:head[0] + 1]), min(lows[head[0]:right[0] + 1]))
            if close < neckline - breakout_buffer:
                return {"name": "Head and shoulders", "state": "neckline breakdown", "direction": -1, "method": "pattern"}
            return {"name": "Head and shoulders", "state": "forming", "direction": 0, "method": "none"}

    all_low_pivots = _swing_points(lows, "low")
    ascending_triangle_context = False
    descending_triangle_context = False
    if len(high_pivots) >= 3 and len(low_pivots) >= 3:
        recent_highs = [point[1] for point in high_pivots[-3:]]
        recent_lows = [point[1] for point in low_pivots[-3:]]
        ascending_triangle_context = (
            max(recent_highs) - min(recent_highs) <= tolerance * 2
            and recent_lows[-1] > recent_lows[0] + tolerance
        )
        descending_triangle_context = (
            max(recent_lows) - min(recent_lows) <= tolerance * 2
            and recent_highs[-1] < recent_highs[0] - tolerance
        )
    if len(all_low_pivots) >= 3:
        left, head, right = all_low_pivots[-3:]
        shoulders_match = abs(left[1] - right[1]) <= tolerance * 2
        head_is_lower = head[1] <= min(left[1], right[1]) - tolerance * 0.75
        if right[0] - left[0] >= 6 and shoulders_match and head_is_lower:
            neckline = max(max(highs[left[0]:head[0] + 1]), max(highs[head[0]:right[0] + 1]))
            if close > neckline + breakout_buffer:
                return {"name": "Inverse head and shoulders", "state": "neckline breakout", "direction": 1, "method": "pattern"}
            return {"name": "Inverse head and shoulders", "state": "forming", "direction": 0, "method": "none"}

    # Three similar swing highs/lows are a triple top/bottom. They are only
    # directional after price crosses the intervening neckline.
    if len(all_high_pivots) >= 3 and not (ascending_triangle_context or descending_triangle_context):
        tops = all_high_pivots[-3:]
        prices = [point[1] for point in tops]
        if tops[-1][0] - tops[0][0] >= 6 and max(prices) - min(prices) <= tolerance * 2:
            neckline = min(lows[tops[0][0]:tops[-1][0] + 1])
            if close < neckline - breakout_buffer:
                return {"name": "Triple top", "state": "neckline breakdown", "direction": -1, "method": "pattern"}
            return {"name": "Triple top", "state": "forming", "direction": 0, "method": "none"}

    if len(all_low_pivots) >= 3 and not (ascending_triangle_context or descending_triangle_context):
        bottoms = all_low_pivots[-3:]
        prices = [point[1] for point in bottoms]
        if bottoms[-1][0] - bottoms[0][0] >= 6 and max(prices) - min(prices) <= tolerance * 2:
            neckline = max(highs[bottoms[0][0]:bottoms[-1][0] + 1])
            if close > neckline + breakout_buffer:
                return {"name": "Triple bottom", "state": "neckline breakout", "direction": 1, "method": "pattern"}
            return {"name": "Triple bottom", "state": "forming", "direction": 0, "method": "none"}

    if len(high_pivots) >= 3 and len(low_pivots) >= 3:
        high_values = [point[1] for point in high_pivots]
        low_values = [point[1] for point in low_pivots]
        highs_flat = max(high_values) - min(high_values) <= tolerance * 2
        lows_flat = max(low_values) - min(low_values) <= tolerance * 2
        highs_falling = high_values[-1] < high_values[0] - tolerance
        lows_rising = low_values[-1] > low_values[0] + tolerance

        if highs_flat and lows_rising:
            resistance = _median(high_values)
            if close > resistance + breakout_buffer:
                return {"name": "Ascending triangle", "state": "breakout", "direction": 1, "method": "pattern"}
            return {"name": "Ascending triangle", "state": "forming", "direction": 0, "method": "momentum"}

        if lows_flat and highs_falling:
            support = _median(low_values)
            if close < support - breakout_buffer:
                return {"name": "Descending triangle", "state": "breakdown", "direction": -1, "method": "pattern"}
            return {"name": "Descending triangle", "state": "forming", "direction": 0, "method": "momentum"}

        if highs_falling and lows_rising:
            resistance = high_values[-1]
            support = low_values[-1]
            if close > resistance + breakout_buffer:
                return {"name": "Symmetrical triangle", "state": "upward breakout", "direction": 1, "method": "pattern"}
            if close < support - breakout_buffer:
                return {"name": "Symmetrical triangle", "state": "downward breakout", "direction": -1, "method": "pattern"}
            return {"name": "Symmetrical triangle", "state": "forming", "direction": 0, "method": "momentum"}

    # A double top/bottom only becomes a directional setup after its neckline breaks.
    if len(all_high_pivots) >= 2 and not (ascending_triangle_context or descending_triangle_context):
        first, second = all_high_pivots[-2:]
        if second[0] - first[0] >= 4 and abs(first[1] - second[1]) <= tolerance * 2:
            neckline = min(lows[first[0]:second[0] + 1])
            if close < neckline - breakout_buffer:
                return {"name": "Double top", "state": "neckline breakdown", "direction": -1, "method": "pattern"}
            return {"name": "Double top", "state": "forming", "direction": 0, "method": "momentum"}

    if len(all_low_pivots) >= 2 and not (ascending_triangle_context or descending_triangle_context):
        first, second = all_low_pivots[-2:]
        if second[0] - first[0] >= 4 and abs(first[1] - second[1]) <= tolerance * 2:
            neckline = max(highs[first[0]:second[0] + 1])
            if close > neckline + breakout_buffer:
                return {"name": "Double bottom", "state": "neckline breakout", "direction": 1, "method": "pattern"}
            return {"name": "Double bottom", "state": "forming", "direction": 0, "method": "momentum"}

    # Wedges have converging boundaries; rising wedges break down and falling
    # wedges break up. Require three pivots on both sides to reduce noise.
    if len(high_pivots) >= 3 and len(low_pivots) >= 3:
        high_slope = _line_slope(high_pivots[-3:])
        low_slope = _line_slope(low_pivots[-3:])
        span = max(high_pivots[-1][0], low_pivots[-1][0]) - min(high_pivots[-3][0], low_pivots[-3][0])
        upper_now = high_pivots[-1][1] + high_slope * (len(history) - high_pivots[-1][0])
        lower_now = low_pivots[-1][1] + low_slope * (len(history) - low_pivots[-1][0])
        if span >= 8 and high_slope > 0 and low_slope > high_slope:
            if close < lower_now - breakout_buffer:
                return {"name": "Rising wedge", "state": "downward breakout", "direction": -1, "method": "pattern"}
            return {"name": "Rising wedge", "state": "forming", "direction": 0, "method": "none"}
        if span >= 8 and low_slope < 0 and high_slope > low_slope:
            if close > upper_now + breakout_buffer:
                return {"name": "Falling wedge", "state": "upward breakout", "direction": 1, "method": "pattern"}
            return {"name": "Falling wedge", "state": "forming", "direction": 0, "method": "none"}

    # Bull/bear flags: an impulse followed by a short, sloping, relatively
    # narrow channel. The continuation direction is only emitted on breakout.
    if len(history) >= 26:
        impulse = history[-26:-14]
        channel = history[-14:]
        impulse_move = impulse[-1]["close"] - impulse[0]["close"]
        channel_highs = [row["high"] for row in channel]
        channel_lows = [row["low"] for row in channel]
        channel_high_slope = _line_slope(list(enumerate(channel_highs)))
        channel_low_slope = _line_slope(list(enumerate(channel_lows)))
        channel_parallel = abs(channel_high_slope - channel_low_slope) <= max(typical_range * 0.3, abs(close) * 0.000001)
        channel_width = max(channel_highs) - min(channel_lows)
        if abs(impulse_move) >= max(typical_range * 4, abs(close) * 0.00003) and channel_width <= abs(impulse_move) * 1.5 and channel_parallel:
            if impulse_move > 0 and channel_high_slope < 0 and channel_low_slope < 0:
                if close > max(channel_highs) + breakout_buffer:
                    return {"name": "Bull flag", "state": "upward breakout", "direction": 1, "method": "pattern"}
                return {"name": "Bull flag", "state": "forming", "direction": 0, "method": "none"}
            if impulse_move < 0 and channel_high_slope > 0 and channel_low_slope > 0:
                if close < min(channel_lows) - breakout_buffer:
                    return {"name": "Bear flag", "state": "downward breakout", "direction": -1, "method": "pattern"}
                return {"name": "Bear flag", "state": "forming", "direction": 0, "method": "none"}

    prior_high = max(row["high"] for row in history[-20:])
    prior_low = min(row["low"] for row in history[-20:])
    if close > prior_high + breakout_buffer:
        return {"name": "20-second range", "state": "upward breakout", "direction": 1, "method": "breakout"}
    if close < prior_low - breakout_buffer:
        return {"name": "20-second range", "state": "downward breakdown", "direction": -1, "method": "breakout"}
    return neutral


def _momentum_direction(rows, horizon_seconds):
    if len(rows) < 2:
        return 0
    latest = rows[-1]
    cutoff = latest["time"] - timedelta(seconds=horizon_seconds)
    recent = [row for row in rows if row["time"] >= cutoff]
    if len(recent) < 2:
        return 0
    movement = recent[-1]["close"] - recent[0]["close"]
    typical_range = _median([row["high"] - row["low"] for row in recent[-10:]])
    threshold = max(typical_range * 0.25, abs(latest["close"]) * 0.000001)
    if movement > threshold:
        return 1
    if movement < -threshold:
        return -1
    return 0


def make_live_forecast(candles, horizon_seconds=DEFAULT_HORIZON_SECONDS):
    """Make a small OHLC path from the same recent one-second trade candles."""
    rows = _clean_candles(candles)
    if not rows:
        return None

    horizon_seconds = max(1, int(horizon_seconds))
    setup = _setup(rows)
    direction = setup["direction"] or _momentum_direction(rows, horizon_seconds)
    method = setup["method"] if setup["direction"] else ("momentum" if direction else "none")
    closes = [row["close"] for row in rows[-20:]]
    changes = [abs(current - previous) for previous, current in zip(closes, closes[1:])]
    typical_step = _median([change for change in changes if change > 0])
    typical_range = _median([row["high"] - row["low"] for row in rows[-20:]])
    if typical_step <= 0:
        typical_step = typical_range * 0.25
    typical_wick = max(typical_range * 0.2, rows[-1]["close"] * 0.000001)

    entry_price = rows[-1]["close"]
    previous_close = entry_price
    forecast_candles = []
    for step in range(1, horizon_seconds + 1):
        predicted_close = entry_price + direction * typical_step * math.sqrt(step)
        forecast_candles.append({
            "time": rows[-1]["time"] + timedelta(seconds=step),
            "open": previous_close,
            "high": max(previous_close, predicted_close) + typical_wick,
            "low": min(previous_close, predicted_close) - typical_wick,
            "close": predicted_close,
        })
        previous_close = predicted_close

    if setup["name"]:
        pattern_label = f"{setup['name']} — {setup['state']}"
    else:
        pattern_label = "No confirmed classic pattern"
    if setup["direction"] and setup["method"] == "pattern":
        reason = f"Confirmed {setup['name'].lower()} {setup['state']}."
    elif setup["direction"] and setup["method"] == "breakout":
        reason = "A recent price range broke; this is tracked separately from classic chart patterns."
    elif direction:
        reason = "No confirmed pattern breakout; the direction uses the recent 5-second momentum baseline."
    else:
        reason = "No confirmed pattern breakout or clear short-term momentum."

    return {
        "time": rows[-1]["time"],
        "entry_price": rows[-1]["close"],
        "direction": direction,
        "method": method,
        "pattern": pattern_label,
        "reason": reason,
        "forecast_candles": forecast_candles,
    }


def wilson_lower_bound_pct(wins, samples, z=1.96):
    """Return a conservative 95% Wilson lower bound for a measured hit rate."""
    samples = int(samples)
    wins = int(wins)
    if samples <= 0 or wins < 0 or wins > samples:
        return 0.0
    rate = wins / samples
    z2 = z * z
    denominator = 1 + z2 / samples
    center = rate + z2 / (2 * samples)
    margin = z * math.sqrt((rate * (1 - rate) / samples) + (z2 / (4 * samples * samples)))
    return max(0.0, (center - margin) / denominator * 100)


class LiveForecastEvaluator:
    """Score non-overlapping five-second forecasts against later trade candles."""

    def __init__(self, horizon_seconds=DEFAULT_HORIZON_SECONDS, max_results=2000,
                 min_pattern_samples=DEFAULT_MIN_PATTERN_SAMPLES):
        self.horizon_seconds = max(1, int(horizon_seconds))
        self.min_pattern_samples = max(1, int(min_pattern_samples))
        self._results = deque(maxlen=max(1, int(max_results)))
        self._pending = None
        self._last_issue_time = None
        self._lock = threading.Lock()

    def update(self, candles):
        rows = _clean_candles(candles)
        if not rows:
            return self.snapshot(None)

        current = make_live_forecast(rows, horizon_seconds=self.horizon_seconds)
        latest_time = rows[-1]["time"]
        with self._lock:
            if self._pending is not None:
                target = self._pending["target_time"]
                realized = next((row for row in rows if row["time"] >= target), None)
                if realized is not None:
                    actual_direction = (realized["close"] > self._pending["entry_price"]) - (realized["close"] < self._pending["entry_price"])
                    record = dict(self._pending)
                    record.update({
                        "actual_time": realized["time"],
                        "actual_close": realized["close"],
                        "correct": actual_direction == self._pending["direction"],
                        "forecast_close_mae_pct": (
                            abs(self._pending["predicted_close"] - realized["close"])
                            / self._pending["entry_price"] * 100
                        ),
                        "last_close_mae_pct": (
                            abs(self._pending["entry_price"] - realized["close"])
                            / self._pending["entry_price"] * 100
                        ),
                    })
                    self._results.append(record)
                    self._pending = None

            if self._pending is None:
                due = self._last_issue_time is None or latest_time >= self._last_issue_time + timedelta(seconds=self.horizon_seconds)
                if due:
                    self._last_issue_time = latest_time
                    if current["direction"] != 0:
                        self._pending = {
                            "issue_time": latest_time,
                            "target_time": latest_time + timedelta(seconds=self.horizon_seconds),
                            "entry_price": current["entry_price"],
                            "direction": current["direction"],
                            "method": current["method"],
                            "pattern": current["pattern"],
                            "predicted_close": current["forecast_candles"][-1]["close"],
                        }
            return self._summary(current)

    def snapshot(self, current):
        with self._lock:
            return self._summary(current)

    def _summary(self, current):
        results = list(self._results)
        pattern_results = [row for row in results if row["method"] == "pattern"]
        breakout_results = [row for row in results if row["method"] == "breakout"]
        momentum_results = [row for row in results if row["method"] == "momentum"]

        def rate(rows):
            wins = sum(bool(row["correct"]) for row in rows)
            count = len(rows)
            return {
                "samples": count,
                "wins": wins,
                "hit_rate_pct": (wins / count * 100) if count else None,
                "lower_bound_pct": wilson_lower_bound_pct(wins, count),
                "forecast_close_mae_pct": (
                    sum(row["forecast_close_mae_pct"] for row in rows) / count if count else None
                ),
                "last_close_mae_pct": (
                    sum(row["last_close_mae_pct"] for row in rows) / count if count else None
                ),
            }

        pattern_rate = rate(pattern_results)
        pattern_rate["validated"] = (
            pattern_rate["samples"] >= self.min_pattern_samples
            and pattern_rate["lower_bound_pct"] >= REQUIRED_PATTERN_ACCURACY_PCT
        )
        return {
            "current": current,
            "pattern": pattern_rate,
            "breakout": rate(breakout_results),
            "momentum": rate(momentum_results),
            "all": rate(results),
            "pending_evaluation": self._pending is not None,
        }
