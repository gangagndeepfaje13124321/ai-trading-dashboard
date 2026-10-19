import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="AI Trading Assistant", page_icon="📈", layout="wide")

st.title("📈 AI Trading Assistant")
st.caption("Version 1 • TradingView chart, technical report, and historical backtesting")

with st.sidebar:
    st.header("Market settings")
    symbol = st.text_input("Ticker symbol", value="AAPL", help="Examples: AAPL, MSFT, NVDA, BTC-USD")
    interval_label = st.selectbox("Chart interval", ["1 minute", "5 minutes", "15 minutes", "1 hour", "1 day"], index=4)
    period = st.selectbox("History for analysis", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    run = st.button("Run analysis", type="primary", use_container_width=True)
    st.caption("Tip: enter a ticker here to analyse it. This does not automatically read the symbol selected in a separate TradingView tab.")

interval_map = {"1 minute": "1", "5 minutes": "5", "15 minutes": "15", "1 hour": "60", "1 day": "D"}
tv_interval = interval_map[interval_label]
clean_symbol = symbol.strip().upper().replace(" ", "")
if not clean_symbol:
    st.warning("Enter a ticker symbol in the sidebar.")
    st.stop()

# TradingView's public Advanced Chart widget. It is a chart embed, not a feed
# that gives this Python app direct access to TradingView's underlying data.
tv_symbol = clean_symbol if ":" in clean_symbol or "-" in clean_symbol else f"{clean_symbol}"
tv_html = f"""
<div class="tradingview-widget-container" style="height:520px;width:100%">
  <div id="tradingview_chart" style="height:520px;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({{
    "autosize": true,
    "symbol": "{tv_symbol}",
    "interval": "{tv_interval}",
    "timezone": "Australia/Melbourne",
    "theme": "dark",
    "style": "1",
    "locale": "en",
    "allow_symbol_change": true,
    "calendar": false,
    "support_host": "https://www.tradingview.com",
    "container_id": "tradingview_chart"
  }});
  </script>
</div>
"""
st.subheader(f"TradingView chart — {clean_symbol}")
st.components.v1.html(tv_html, height=530, scrolling=False)

# Fetch data from Yahoo Finance for analysis/backtesting. Availability and
# delay depend on ticker, exchange, interval, and provider.
@st.cache_data(ttl=60, show_spinner=False)
def load_data(ticker, history_period):
    df = yf.download(ticker, period=history_period, interval="1d",
                      auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"]).copy()

if run or "market_data" not in st.session_state or st.session_state.get("loaded_symbol") != clean_symbol or st.session_state.get("loaded_period") != period:
    with st.spinner(f"Downloading historical daily data for {clean_symbol}..."):
        try:
            st.session_state.market_data = load_data(clean_symbol, period)
            st.session_state.loaded_symbol = clean_symbol
            st.session_state.loaded_period = period
        except Exception as exc:
            st.error(f"Could not download data: {exc}")
            st.stop()

data = st.session_state.market_data
if data.empty:
    st.error("No historical data was returned for this symbol. Check the ticker and try again.")
    st.stop()

close = data["Close"].astype(float)
data["SMA20"] = close.rolling(20).mean()
data["SMA50"] = close.rolling(50).mean()
delta = close.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
data["RSI14"] = 100 - (100 / (1 + rs))

latest = float(close.iloc[-1])
previous = float(close.iloc[-2]) if len(close) > 1 else latest
change_pct = ((latest / previous) - 1) * 100 if previous else 0
sma20 = data["SMA20"].iloc[-1]
sma50 = data["SMA50"].iloc[-1]
rsi = data["RSI14"].iloc[-1]

st.divider()
st.subheader("Market snapshot")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest available close", f"{latest:,.2f}", f"{change_pct:+.2f}% vs prior close")
c2.metric("20-day average", f"{sma20:,.2f}" if pd.notna(sma20) else "Not enough data")
c3.metric("50-day average", f"{sma50:,.2f}" if pd.notna(sma50) else "Not enough data")
c4.metric("14-day RSI", f"{rsi:.1f}" if pd.notna(rsi) else "Not enough data")

st.subheader("Technical analysis report")
report = []
if pd.notna(sma20) and pd.notna(sma50):
    if latest > sma20 and sma20 > sma50:
        report.append("Trend snapshot: bullish alignment — latest close is above the 20-day average, which is above the 50-day average.")
    elif latest < sma20 and sma20 < sma50:
        report.append("Trend snapshot: bearish alignment — latest close is below the 20-day average, which is below the 50-day average.")
    else:
        report.append("Trend snapshot: mixed — price and moving averages do not show a clear bullish or bearish alignment.")
else:
    report.append("Trend snapshot: not enough history to calculate both moving averages.")
if pd.notna(rsi):
    if rsi >= 70:
        report.append(f"Momentum: RSI is {rsi:.1f}, commonly interpreted as overbought territory. This is not, by itself, a sell signal.")
    elif rsi <= 30:
        report.append(f"Momentum: RSI is {rsi:.1f}, commonly interpreted as oversold territory. This is not, by itself, a buy signal.")
    else:
        report.append(f"Momentum: RSI is {rsi:.1f}, between the commonly watched 30 and 70 levels.")
report.append(f"Data note: analysis uses daily adjusted prices from Yahoo Finance, last available date {data.index[-1].date()}. This is not guaranteed to be real-time.")
report.append("This report describes indicators; it does not know why a move happened. News/social analysis is planned for a later version.")
for item in report:
    st.write("• " + item)

with st.expander("Show recent price history"):
    st.dataframe(data[["Close", "SMA20", "SMA50", "RSI14"]].tail(20).sort_index(ascending=False), use_container_width=True)

st.divider()
st.subheader("Historical backtest: simple moving-average strategy")
st.write("Educational example only: hold the asset when its 20-day average is above its 50-day average; otherwise stay in cash. The signal is shifted one day to reduce look-ahead bias. This simplified test excludes fees, slippage, taxes, and execution delays.")

bt = data[["Close", "SMA20", "SMA50"]].dropna().copy()
if len(bt) < 5:
    st.info("Not enough history to backtest. Choose a longer history period.")
else:
    bt["Market_Return"] = bt["Close"].pct_change().fillna(0)
    bt["Signal"] = (bt["SMA20"] > bt["SMA50"]).astype(int).shift(1).fillna(0)
    bt["Strategy_Return"] = bt["Signal"] * bt["Market_Return"]
    bt["Buy_Hold_Equity"] = (1 + bt["Market_Return"]).cumprod()
    bt["Strategy_Equity"] = (1 + bt["Strategy_Return"]).cumprod()
    market_return = (bt["Buy_Hold_Equity"].iloc[-1] - 1) * 100
    strategy_return = (bt["Strategy_Equity"].iloc[-1] - 1) * 100
    b1, b2, b3 = st.columns(3)
    b1.metric("Strategy total return", f"{strategy_return:+.2f}%")
    b2.metric("Buy & hold return", f"{market_return:+.2f}%")
    b3.metric("Test observations", f"{len(bt):,} days")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Strategy_Equity"], mode="lines", name="Moving-average strategy"))
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Buy_Hold_Equity"], mode="lines", name="Buy & hold"))
    fig.update_layout(title="Growth of $1 (before fees and taxes)", xaxis_title="Date", yaxis_title="Portfolio value ($)", template="plotly_dark", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

st.caption("Important: this first version does not place trades, does not guarantee minute-by-minute predictions, and does not yet analyse news or social media. The embedded TradingView chart and Yahoo Finance analysis may use different data feeds or update schedules.")
