# Live Crypto Dashboard

A Streamlit dashboard for watching Kraken trade-based one-second candles and a short experimental OHLC scenario.

## Main screen

- The home page shows a live Kraken candle chart beside a short forecast chart. Both use the same public Kraken trade feed.
- Kraken exchange timestamps are grouped into UTC seconds. Seconds without a trade are not invented as candles.
- The dashboard redraws once per second while open. A forecast changes when new trade data arrives; an empty second does not create a new market observation.
- The forecast uses simple price-pattern geometry (such as triangles, flags, wedges, and top/bottom formations) after a boundary break. Without a confirmed pattern, it can use a short momentum rule. It is not a trained machine-learning model.
- Forecast OHLC wicks reuse recent movement sizes. They are illustrative scenarios, not calibrated uncertainty ranges or trading instructions.
- Pattern results are checked against later Kraken trade candles in memory. They reset when the Streamlit app process restarts. A pattern is labelled validated only after at least 200 outcomes and a 95% Wilson lower bound of 55% or more. This directional hit-rate check does not prove profitability after spread, fees, and slippage.

## Historical analysis

The separate **Historical Analysis** page retains the existing Yahoo Finance history, indicators, candle table, simple trend baseline, and historical holdout metrics. Those tools are slower and are not required by the live dashboard page.

## Data and deployment

The app subscribes to Kraken's public WebSocket trade channel; no API key is needed. `requirements.txt` lists the packages used by the app. Streamlit Community Cloud can deploy from this GitHub repository by selecting `app.py` as the main file.

## Tests

Run the repository's available unit tests with:

```bash
python -m unittest discover -s tests -v
```
