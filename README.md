# Live Crypto Dashboard

A Streamlit dashboard with matching Kraken data views: historical/live one-minute candles on the left and trade-based one-second candles on the right.

## Main screen

- **Left chart:** Kraken REST minute history, with the current minute updated from the live Kraken trade feed. Blue/orange future OHLC candles show the historical rule-selection experiment. The chart opens on about two hours of candles; its bottom range slider can reach the available history. Mouse-wheel zoom and line/shape drawing tools are enabled.
- **Why the chart is not the hosted TradingView widget:** that widget is a third-party iframe and cannot accept this app's custom forecast candles. The in-app Plotly chart supports the forecast overlay, drawing, zooming, and the same Kraken data as the right chart. It is not the TradingView product.
- **Minute method:** four transparent rules are compared on the earlier 70% of the available minute history: momentum, EMA trend, mean reversion, and range breakout. The rule and its typical expected move are fit only on that earlier block. The last 30% is kept aside as a chronological holdout, then compared with last-close and momentum baselines. The display includes call count, hit rate, a 95% lower confidence bound, and close-price MAE.
- The minute chart labels its direction **WAIT** unless the held-out test has at least 30 directional calls, at least 55% direction matches, and lower close-price MAE than keeping the last close on the same calls. Passing this screen is not proof of profitability; fees, spread, slippage, and execution are not modeled.
- **Right chart:** Kraken's public trade WebSocket groups actual trades by exchange timestamp into one-second OHLC candles. Empty seconds are not invented. The scenario horizon can be set to 10, 30, 60, or 120 seconds and refreshes each second while the feed is current.
- The second-level scenario is a transparent extrapolation from recent movement and observed candle ranges. It is not historically trained or backtested because the app does not load historical one-second trades. Its live directional score begins only after startup and is kept in memory; it resets when the app process restarts.
- A buy-side or sell-side watch is an experimental direction label, never an order instruction. No model promises profitable or consistently accurate predictions.

## Data limits

The public Kraken OHLC endpoint supplies a recent window of one-minute history, not full market history. The live WebSocket only gives a recent trade snapshot at startup and then live trades, so second-candle history starts building while the app is running. Kraken also publishes much larger downloadable historical trade archives; importing and maintaining those is a separate data-storage stage and is not required for this dashboard to run.

## Historical analysis page

The **Historical Analysis** page remains separate and retains the earlier Yahoo Finance chart and trend-baseline evaluation.

## Deployment

The app uses public market-data endpoints and does not require API keys. Deploy from this GitHub repository on Streamlit Community Cloud with `app.py` selected as the main file. Dependencies are listed in `requirements.txt`.

## Tests

Run the available unit tests with:

```bash
python -m unittest discover -s tests -v
```
