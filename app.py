import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="AI Trading Assistant", page_icon="📈", layout="wide")
st.title("📈 AI Trading Assistant")
st.caption("Version 2 • Intraday technical analysis and educational backtesting")

INTERVALS = {
    "1 minute": {"yf": "1m", "tv": "1", "periods": ["1d", "5d", "7d"], "default": "5d", "max_note": "Yahoo Finance typically limits 1-minute history to about 7 days."},
    "5 minutes": {"yf": "5m", "tv": "5", "periods": ["1d", "5d", "1mo", "2mo", "60d"], "default": "1mo", "max_note": "Yahoo Finance typically limits 5-minute history to about 60 days."},
    "15 minutes": {"yf": "15m", "tv": "15", "periods": ["5d", "1mo", "3mo", "60d"], "default": "1mo", "max_note": "Intraday history limits depend on ticker and provider."},
    "1 hour": {"yf": "60m", "tv": "60", "periods": ["5d", "1mo", "3mo", "6mo", "1y"], "default": "3mo", "max_note": "Hourly history limits depend on ticker and provider."},
    "1 day": {"yf": "1d", "tv": "D", "periods": ["1mo", "3mo", "6mo", "1y", "2y", "5y"], "default": "6mo", "max_note": "Daily data is suitable for longer historical comparisons."},
}

with st.sidebar:
    st.header("Market settings")
    symbol = st.text_input("Ticker symbol", value="AAPL", help="Examples: AAPL, MSFT, NVDA, BTC-USD")
    interval_label = st.selectbox("Analysis + chart interval", list(INTERVALS.keys()), index=0)
    config = INTERVALS[interval_label]
    period = st.selectbox("History for analysis", config["periods"], index=config["periods"].index(config["default"]))
    run = st.button("Run analysis", type="primary", use_container_width=True)
    st.caption("The selected interval now controls both the embedded chart and Python analysis/backtest, where data is available.")
    st.caption(config["max_note"])

clean_symbol = symbol.strip().upper().replace(" ", "")
if not clean_symbol:
    st.warning("Enter a ticker symbol in the sidebar.")
    st.stop()

# TradingView chart is embedded for viewing; Python analysis independently uses Yahoo Finance.
tv_symbol = clean_symbol
tv_html = f"""
<div class="tradingview-widget-container" style="height:520px;width:100%">
  <div id="tradingview_chart" style="height:520px;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({{
    "autosize": true, "symbol": "{tv_symbol}", "interval": "{config['tv']}",
    "timezone": "Australia/Melbourne", "theme": "dark", "style": "1", "locale": "en",
    "allow_symbol_change": true, "calendar": false,
    "support_host": "https://www.tradingview.com", "container_id": "tradingview_chart"
  }});
  </script>
</div>
"""
st.subheader(f"TradingView chart — {clean_symbol} ({interval_label})")
st.components.v1.html(tv_html, height=530, scrolling=False)

@st.cache_data(ttl=60, show_spinner=False)
def load_data(ticker, history_period, interval):
    df = yf.download(ticker, period=history_period, interval=interval, auto_adjust=True, progress=False, prepost=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[needed].copy()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["Close"])

cache_key = (clean_symbol, period, config["yf"])
if run or "market_data" not in st.session_state or st.session_state.get("loaded_key") != cache_key:
    with st.spinner(f"Downloading {interval_label.lower()} data for {clean_symbol}..."):
        try:
            st.session_state.market_data = load_data(clean_symbol, period, config["yf"])
            st.session_state.loaded_key = cache_key
        except Exception as exc:
            st.error(f"Could not download data: {exc}")
            st.stop()

data = st.session_state.market_data.copy()
if data.empty:
    st.error("No data was returned for this ticker, interval, and history period. Try a shorter period, a different ticker, or daily data. Provider availability varies by exchange and symbol.")
    st.stop()

close = data["Close"].astype(float)
data["SMA20"] = close.rolling(20, min_periods=20).mean()
data["SMA50"] = close.rolling(50, min_periods=50).mean()
delta = close.diff()
gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
rs = gain / loss.replace(0, np.nan)
data["RSI14"] = 100 - (100 / (1 + rs))

latest = float(close.iloc[-1])
previous = float(close.iloc[-2]) if len(close) > 1 else latest
change_pct = ((latest / previous) - 1) * 100 if previous else 0
sma20 = data["SMA20"].iloc[-1]
sma50 = data["SMA50"].iloc[-1]
rsi = data["RSI14"].iloc[-1]

st.divider()
st.subheader(f"Market snapshot — {interval_label} candles")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest available close", f"{latest:,.2f}", f"{change_pct:+.2f}% vs prior candle")
c2.metric("20-candle average", f"{sma20:,.2f}" if pd.notna(sma20) else "Not enough data")
c3.metric("50-candle average", f"{sma50:,.2f}" if pd.notna(sma50) else "Not enough data")
c4.metric("14-candle RSI", f"{rsi:.1f}" if pd.notna(rsi) else "Not enough data")

st.subheader("Technical analysis report")
report = []
if pd.notna(sma20) and pd.notna(sma50):
    if latest > sma20 and sma20 > sma50:
        report.append("Trend snapshot: bullish alignment — latest close is above the 20-candle average, which is above the 50-candle average.")
    elif latest < sma20 and sma20 < sma50:
        report.append("Trend snapshot: bearish alignment — latest close is below the 20-candle average, which is below the 50-candle average.")
    else:
        report.append("Trend snapshot: mixed — price and moving averages do not show a clear bullish or bearish alignment.")
else:
    report.append("Trend snapshot: not enough candles to calculate both moving averages.")
if pd.notna(rsi):
    if rsi >= 70:
        report.append(f"Momentum: RSI is {rsi:.1f}, commonly interpreted as overbought territory. This is not, by itself, a sell signal.")
    elif rsi <= 30:
        report.append(f"Momentum: RSI is {rsi:.1f}, commonly interpreted as oversold territory. This is not, by itself, a buy signal.")
    else:
        report.append(f"Momentum: RSI is {rsi:.1f}, between the commonly watched 30 and 70 levels.")
report.append(f"Data note: Python analysis uses {interval_label.lower()} adjusted candles from Yahoo Finance. Latest candle timestamp returned: {data.index[-1]}.")
report.append("The TradingView embed and Yahoo Finance may use different feeds, timestamps, or update schedules. This is indicator-based analysis, not a guaranteed prediction.")
for item in report:
    st.write("• " + item)

with st.expander("Show recent candles"):
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "SMA20", "SMA50", "RSI14"] if c in data.columns]
    st.dataframe(data[cols].tail(50).sort_index(ascending=False), use_container_width=True)

st.divider()
st.subheader(f"Historical backtest — {interval_label} candles")
st.write("Educational example: hold the asset when its 20-candle average is above its 50-candle average; otherwise stay in cash. The signal is shifted by one candle to reduce look-ahead bias. This simplified test excludes fees, spread, slippage, taxes, and execution delays.")
bt = data[["Close", "SMA20", "SMA50"]].dropna().copy()
if len(bt) < 5:
    st.info("Not enough candles to backtest. Choose a longer history period or a slower interval.")
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
    b3.metric("Test observations", f"{len(bt):,} candles")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Strategy_Equity"], mode="lines", name="Moving-average strategy"))
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Buy_Hold_Equity"], mode="lines", name="Buy & hold"))
    fig.update_layout(title="Growth of $1 (before fees and taxes)", xaxis_title="Timestamp", yaxis_title="Portfolio value ($)", template="plotly_dark", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

st.caption("Important: this app does not place trades or guarantee predictions. Intraday data availability is limited by the data provider and may be delayed or incomplete. Use paper trading and realistic costs before considering real-money decisions.")
