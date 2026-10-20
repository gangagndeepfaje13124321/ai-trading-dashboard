# Market Dashboard

A Streamlit market dashboard with TradingView charts, real trade-based one-second candles, and a transparent minute-candle research test.

## Main screen

- The asset selector includes Kraken crypto pairs plus Apple (AAPL) and Tesla (TSLA).
- The larger left panel is TradingView's hosted 1-minute chart. Use its own toolbar for chart intervals, indicators, zooming, and drawings.
- The right panel plots actual received trades grouped into UTC one-second OHLC candles. Solid lines trace recent swing-high/resistance and swing-low/support boundaries; dashed extensions continue those lines for context. These guides are not forecast prices or trading signals. Kraken second-level history is not loaded, so the guides are not historically evaluated.
- Kraken supplies public crypto trades and minute history. Apple and Tesla use Alpaca's authenticated IEX stock feed and historical bars. Alpaca's free Basic market-data plan covers IEX, not every US stock exchange; the TradingView chart may use a different source.
- The historical minute experiment compares momentum, EMA trend, mean reversion, and range breakout. It selects a rule on the earlier 70% of loaded bars, then reports results on the later 30% held aside. A short plain-English summary and charts compare directional accuracy with a 50% reference and close error with the unchanged-price baseline.
- The minute experiment is not a second-by-second forecast. One-second history is not loaded, so the app does not draw invented forecast candles or label an untested one-second guess as a trade signal. Any brief live score is in-memory and resets when the Streamlit process restarts.
- No model guarantees profit. The evaluation does not include fees, bid/ask spread, slippage, or whether an order could be filled.

## Enable Apple and Tesla data

Create Alpaca market-data API keys in your Alpaca account, then add them to the app's **Streamlit Community Cloud → App settings → Secrets** area. Use TOML syntax:

```toml
ALPACA_API_KEY = "your-key"
ALPACA_SECRET_KEY = "your-secret"
```

Keep these values in Streamlit Secrets. Never add them to GitHub, a source file, or a chat message. If the keys are absent, the TradingView stock chart still displays, while stock history and the one-second stock feed stay disabled.

## Deployment

Deploy the repository on Streamlit Community Cloud with `app.py` as the main file. Python dependencies are listed in `requirements.txt`. After a GitHub change is merged into the branch Streamlit Cloud deploys, its app rebuilds from that branch.

## Tests

Run the available unit tests with:

```bash
python -m unittest discover -s tests -v
```
