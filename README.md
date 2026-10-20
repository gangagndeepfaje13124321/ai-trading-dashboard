# Live Crypto Dashboard

A compact Streamlit dashboard with a TradingView one-minute chart and a Kraken live one-second candle/scenario view.

## Main screen

- The left chart is TradingView's hosted one-minute chart with its drawing toolbar. TradingView supplies this chart's market data, which can differ slightly from Kraken prices shown on the right.
- The right chart uses Kraken's public trade WebSocket and groups real exchange trades into genuine one-second OHLC candles. The page redraws every second. Seconds with no trades are left blank, and no past one-second bars are downloaded.
- The right chart adds a small number of hypothetical one-second OHLC scenario candles. The direction comes from a confirmed classic-pattern/range break, or otherwise recent price movement. It is not a trained model, and candle values are not actual future exchange prices.
- A “buy-side watch” or “sell-side watch” is only a short-term direction label from those simple rules. “Wait” means they found no clear direction. These labels are not orders or evidence of profitability.
- The app scores non-neutral scenarios against later Kraken trade candles while it is running. This short live sample is held in memory, resets when the app restarts, and does not account for spread, fees, slippage, or execution. It is not proof of a lasting edge.
- Kraken's trade WebSocket sends only a recent-trade snapshot on connection, then live updates. The app keeps up to 15 minutes of observed trade-second candles in memory.

The hosted TradingView widget does not let this Streamlit app add its own forecast series. The forecast is therefore shown as candles on the Kraken one-second chart, where its time interval and data source match the live trades.

## Historical analysis

The separate **Historical Analysis** page retains Yahoo Finance history, indicators, the candle table, and simple trend-baseline evaluation. It is separate from the Kraken live dashboard.

## Data and deployment

The app uses Kraken public market-data endpoints and does not need API keys. `requirements.txt` lists the app dependencies. Streamlit Community Cloud can deploy from this GitHub repository with `app.py` selected as the main file.

## Tests

Run the repository's available unit tests with:

```bash
python -m unittest discover -s tests -v
```
