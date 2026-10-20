"""Simple, explicitly labelled price baselines and walk-forward evaluation."""

import math

import numpy as np
import pandas as pd


OHLC_COLUMNS = ("Open", "High", "Low", "Close")


def _finite_prices(values):
    prices = np.asarray(values, dtype=float).reshape(-1)
    return prices[np.isfinite(prices)]


def _clean_ohlc(frame):
    missing = set(OHLC_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("OHLC data must include Open, High, Low, and Close columns.")
    clean = frame.loc[:, OHLC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    return clean


def linear_trend_forecast(close_values, steps=1, lookback=100):
    """Extrapolate a least-squares line; this is a baseline, not trained ML."""
    values = _finite_prices(close_values)
    if len(values) == 0:
        raise ValueError("At least one finite close price is required.")
    steps = max(1, int(steps))
    lookback = max(1, int(lookback))
    recent = values[-lookback:]
    if len(recent) < 2:
        return [float(recent[-1])] * steps

    x = np.arange(len(recent), dtype=float)
    slope, intercept = np.polyfit(x, recent, 1)
    future_x = np.arange(len(recent), len(recent) + steps, dtype=float)
    return (slope * future_x + intercept).astype(float).tolist()


def build_forecast_candles(frame, future_closes, lookback=100):
    """Build point OHLC baseline candles using prior typical wick sizes."""
    history = _clean_ohlc(frame).tail(max(1, int(lookback)))
    if history.empty:
        raise ValueError("At least one complete historical OHLC candle is required.")

    upper_wicks = history["High"] - history[["Open", "Close"]].max(axis=1)
    lower_wicks = history[["Open", "Close"]].min(axis=1) - history["Low"]
    typical_upper_wick = max(0.0, float(upper_wicks.clip(lower=0).median()))
    typical_lower_wick = max(0.0, float(lower_wicks.clip(lower=0).median()))

    previous_close = float(history["Close"].iloc[-1])
    candles = []
    for step, value in enumerate(future_closes, start=1):
        predicted_close = float(value)
        predicted_open = previous_close
        predicted_high = max(predicted_open, predicted_close) + typical_upper_wick
        predicted_low = min(predicted_open, predicted_close) - typical_lower_wick
        candles.append({
            "step": step,
            "open": predicted_open,
            "high": predicted_high,
            "low": predicted_low,
            "close": predicted_close,
        })
        previous_close = predicted_close
    return candles


def rolling_origin_backtest(frame, lookback=100, min_train=20, max_origins=500,
                            test_fraction=0.2):
    """Compare next-candle forecasts with last-close using only prior candles.

    Every prediction is generated before its target candle. Metrics are reported
    on the final chronological portion of those rolling-origin predictions.
    """
    history = _clean_ohlc(frame).reset_index(drop=True)
    min_train = max(2, int(min_train))
    lookback = max(1, int(lookback))
    if len(history) <= min_train:
        return None

    first_target = max(min_train, len(history) - max(1, int(max_origins)))
    predictions = []
    for target_index in range(first_target, len(history)):
        training = history.iloc[max(0, target_index - lookback):target_index]
        forecast = build_forecast_candles(
            training, linear_trend_forecast(training["Close"], steps=1, lookback=lookback), lookback
        )[0]
        target = history.iloc[target_index]
        previous_close = float(training["Close"].iloc[-1])
        predictions.append({
            "actual": target,
            "forecast": forecast,
            "previous_close": previous_close,
        })

    if len(predictions) < 5:
        return None

    holdout_count = max(5, int(math.ceil(len(predictions) * float(test_fraction))))
    holdout = predictions[-holdout_count:]
    actual_close = np.asarray([row["actual"]["Close"] for row in holdout], dtype=float)
    trend_close = np.asarray([row["forecast"]["close"] for row in holdout], dtype=float)
    last_close = np.asarray([row["previous_close"] for row in holdout], dtype=float)
    actual_direction = np.sign(actual_close - last_close)
    predicted_direction = np.sign(trend_close - last_close)

    component_mae = {}
    for component in ("open", "high", "low", "close"):
        predicted = np.asarray([row["forecast"][component] for row in holdout], dtype=float)
        actual = np.asarray([row["actual"][component.capitalize()] for row in holdout], dtype=float)
        component_mae[component] = float(np.mean(np.abs(predicted - actual)))

    trend_errors = trend_close - actual_close
    persistence_errors = last_close - actual_close
    return {
        "holdout_candles": len(holdout),
        "trend_close_mae": float(np.mean(np.abs(trend_errors))),
        "trend_close_rmse": float(np.sqrt(np.mean(np.square(trend_errors)))),
        "last_close_mae": float(np.mean(np.abs(persistence_errors))),
        "directional_accuracy_pct": float(np.mean(predicted_direction == actual_direction) * 100),
        "ohlc_mae": component_mae,
    }
