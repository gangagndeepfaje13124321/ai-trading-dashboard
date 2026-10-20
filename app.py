import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Market Pattern Explorer", page_icon="📈", layout="wide")
st.title("📈 Market Pattern Explorer")
st.caption("Candle-based market analysis and a short-range statistical projection. Educational only — not trading advice.")

INTERVALS = {
    "1 minute":  {"yf": "1m",  "tv": "1",  "periods": ["1d", "5d", "7d"], "default": "1d",  "minutes": 1},
    "5 minutes": {"yf": "5m",  "tv": "5",  "periods": ["1d", "5d", "1mo"], "default": "5d", "minutes": 5},
    "15 minutes":{"yf": "15m", "tv": "15", "periods": ["5d", "1mo", "3mo"], "default": "5d", "minutes": 15},
    "30 minutes":{"yf": "30m", "tv": "30", "periods": ["5d", "1mo", "2mo"], "default": "5d", "minutes": 30},
    "1 hour":    {"yf": "60m", "tv": "60", "periods": ["5d", "1mo", "3mo", "6mo", "1y"], "default": "1mo", "minutes": 60},
    "1 day":     {"yf": "1d",  "tv": "D",  "periods": ["1mo", "3mo", "6mo", "1y", "2y"], "default": "3mo", "minutes": 1440},
}

def tv_symbol_for(ticker):
    """Convert common Yahoo symbols into TradingView symbols."""
    t = ticker.strip().upper()
    if t == "BTC-USD":
        return "BINANCE:BTCUSDT"
    if t == "ETH-USD":
        return "BINANCE:ETHUSDT"
    if t == "SOL-USD":
        return "BINANCE:SOLUSDT"
    if t.endswith("-USD"):
        return "COINBASE:" + t.replace("-USD", "USD")
    return t

with st.sidebar:
    st.header("Market settings")
    symbol = st.text_input(
        "Ticker symbol", value="BTC-USD",
        help="Use Yahoo Finance tickers for analysis. Examples: BTC-USD, ETH-USD, AAPL, MSFT."
    )
    interval_label = st.selectbox("Candle interval", list(INTERVALS.keys()), index=0,
        help="Each candle summarises price movement over this time.")
    config = INTERVALS[interval_label]
    period = st.selectbox("History to analyse", config["periods"],
        index=config["periods"].index(config["default"]),
        help="How much past data the analysis uses. Intraday data availability is limited by the provider.")
    refresh = st.button("Refresh market data", type="primary", use_container_width=True)
    st.caption("Yahoo Finance supplies analysis data; TradingView supplies the separate chart. Their candles may not match perfectly.")
    st.info("For BTC-USD, Bitcoin trades 24/7. A missing-data error is usually a data-provider or ticker issue, not market closing.")

symbol_clean = symbol.strip().upper().replace(" ", "")
if not symbol_clean:
    st.warning("Enter a ticker symbol.")
    st.stop()

tv_symbol = tv_symbol_for(symbol_clean)
st.subheader(f"TradingView chart — {tv_symbol} ({interval_label})")
st.caption("The chart's interval selector is controlled by TradingView. If it resets, use the chart's own interval menu after it loads.")
tv_html = f"""
<div class="tradingview-widget-container" style="height:560px;width:100%">
  <div id="tv_chart_{config['tv']}" style="height:560px;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
    new TradingView.widget({{
      "width": "100%",
      "height": 560,
      "symbol": "{tv_symbol}",
      "interval": "{config['tv']}",
      "timezone": "Australia/Melbourne",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#131722",
      "enable_publishing": false,
      "allow_symbol_change": true,
      "hide_side_toolbar": false,
      "withdateranges": true,
      "container_id": "tv_chart_{config['tv']}"
    }});
  </script>
</div>
"""
st.components.v1.html(tv_html, height=575, scrolling=False)

@st.cache_data(ttl=60, show_spinner=False)
def load_data(ticker, history_period, interval):
    df = yf.download(
        ticker, period=history_period, interval=interval,
        auto_adjust=False, progress=False, prepost=False,
        threads=False, timeout=20, multi_level_index=False
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in required):
