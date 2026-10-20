# Live Crypto Dashboard

A Streamlit dashboard for viewing Kraken one-minute OHLC candles and a transparent, multi-signal short-horizon scenario.

## Main screen

- Kraken's public REST OHLC endpoint supplies about 12 hours of recent one-minute bars. Kraken's public trade WebSocket updates the current minute candle as new trades arrive. The page redraws every second; the current candle is still forming until its minute ends.
- Historical bars are kept from the OHLC endpoint. Recent streamed one-second trade candles are aggregated into minute candles and used to extend/update the current bar; empty minutes are not fabricated by the stream aggregator.
- The right-hand view shows a broad scenario range, not a predicted candle path. Its width uses recent average true range; this is an observed-volatility range, not a calibrated probability interval or exact target.
- The displayed direction combines fixed, readable rules: 9/21 EMA trend, 3- and 10-minute momentum, RSI extremes, 20-minute high/low breakouts, 20-minute volume-weighted price, and a confirmed classic-pattern break. A direction requires at least two active votes and 60% agreement among active votes. Votes can be correlated and are not probabilities.
- This is **not a trained AI model** and does not place orders. The historical check walks through Kraken's recent candles in time order, only uses information available at each forecast origin, spaces test forecasts by the selected horizon, and compares direction calls with a simple continuation baseline. It does not account for spread, fees, slippage, or execution, and it is not proof of a profitable edge.
- The WebSocket trade snapshot contains only the most recent 50 trades. Kraken's REST OHLC endpoint includes an unfinished current candle and is limited to roughly 12 hours of one-minute history; the app uses this for recent chart history and updates the current candle with the live stream.

## Historical analysis

The separate **Historical Analysis** page retains Yahoo Finance history, indicators, the candle table, and simple trend-baseline evaluation. It is separate from the Kraken live dashboard.

## Data and deployment

The app uses Kraken public market-data endpoints and does not need API keys. `requirements.txt` lists the app dependencies. Streamlit Community Cloud can deploy from this GitHub repository with `app.py` selected as the main file.

## Tests

Run the repository's available unit tests with:

```bash
python -m unittest discover -s tests -v
```
