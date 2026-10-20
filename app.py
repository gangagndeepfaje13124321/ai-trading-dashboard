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
        return pd.DataFrame()
    keep = required + (["Volume"] if "Volume" in df.columns else [])
    df = df[keep].copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=required)

cache_key = (symbol_clean, period, config["yf"])
if refresh:
    load_data.clear()
    st.session_state.pop("market_data", None)
if "market_data" not in st.session_state or st.session_state.get("loaded_key") != cache_key:
    try:
        with st.spinner(f"Loading {period} of {interval_label.lower()} candles for {symbol_clean}..."):
            st.session_state.market_data = load_data(symbol_clean, period, config["yf"])
            st.session_state.loaded_key = cache_key
    except Exception as exc:
        st.error(f"Could not load market data: {type(exc).__name__}: {exc}")
        st.stop()

data = st.session_state.market_data.copy()
if data.empty:
    st.error(
        f"No candles were returned for {symbol_clean} with interval {config['yf']} and history {period}. "
        "Try BTC-USD, 5 minutes, and 5d, then press Refresh market data. Yahoo Finance can temporarily omit intraday data."
    )
    st.stop()

# Indicators are computed across the entire selected history, not only the last few candles.
data["Change"] = data["Close"].diff()
data["Change_pct"] = data["Close"].pct_change() * 100
data["Candle_range"] = data["High"] - data["Low"]
data["Body"] = data["Close"] - data["Open"]
data["Body_pct"] = data["Body"] / data["Open"].replace(0, np.nan) * 100
data["Upper_wick"] = data["High"] - data[["Open", "Close"]].max(axis=1)
data["Lower_wick"] = data[["Open", "Close"]].min(axis=1) - data["Low"]
data["Range_pct"] = data["Candle_range"] / data["Close"].replace(0, np.nan) * 100
data["EMA20"] = data["Close"].ewm(span=20, adjust=False, min_periods=1).mean()
data["EMA50"] = data["Close"].ewm(span=50, adjust=False, min_periods=1).mean()
data["EMA20_slope_pct"] = data["EMA20"].pct_change(5) * 100
delta = data["Close"].diff()
gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
rs = gain / loss.replace(0, np.nan)
data["RSI14"] = 100 - (100 / (1 + rs))
prev_close = data["Close"].shift(1)
true_range = pd.concat([
    data["High"] - data["Low"],
    (data["High"] - prev_close).abs(),
    (data["Low"] - prev_close).abs()
], axis=1).max(axis=1)
data["ATR14"] = true_range.ewm(alpha=1/14, adjust=False, min_periods=1).mean()
data["Move_5"] = data["Close"].pct_change(5) * 100
data["Move_10"] = data["Close"].pct_change(10) * 100

close = data["Close"].astype(float).dropna()
if len(close) < 2:
    st.error("Not enough candles were returned to analyse this market. Try a longer history or a larger candle interval.")
    st.stop()
latest = float(close.iloc[-1])
previous = float(close.iloc[-2])
last_change = (latest / previous - 1) * 100 if previous else 0
recent_n = min(10, len(data))
recent_move = (float(close.iloc[-1]) / float(close.iloc[-recent_n]) - 1) * 100 if recent_n > 1 else 0.0
ema20 = float(data["EMA20"].iloc[-1])
ema50 = float(data["EMA50"].iloc[-1])
rsi_now = float(data["RSI14"].dropna().iloc[-1]) if data["RSI14"].notna().any() else np.nan
atr_now = float(data["ATR14"].iloc[-1])
if latest > ema20 > ema50:
    trend = "Uptrend alignment"
    trend_detail = "Price is above EMA20, and EMA20 is above EMA50. This supports an upward trend reading, but does not guarantee continuation."
elif latest < ema20 < ema50:
    trend = "Downtrend alignment"
    trend_detail = "Price is below EMA20, and EMA20 is below EMA50. This supports a downward trend reading, but does not guarantee continuation."
else:
    trend = "Mixed / sideways signals"
    trend_detail = "Price and moving averages are not aligned in one clear direction. Choppy conditions may be present."

st.subheader("Market snapshot")
a, b, c, d = st.columns(4)
a.metric("Latest available close", f"{latest:,.4f}", f"{last_change:+.3f}% vs prior candle")
b.metric("Change over last 10 candles", f"{recent_move:+.3f}%")
c.metric("Trend structure", trend)
d.metric("RSI (14)", "—" if np.isnan(rsi_now) else f"{rsi_now:.1f}")
st.caption(f"Data: Yahoo Finance · Latest candle: {data.index[-1]} · {len(data):,} candles analysed. This is the latest returned candle, not necessarily a live quote.")
st.write(f"**Trend interpretation:** {trend_detail}")
if not np.isnan(rsi_now):
    if rsi_now >= 70:
        st.write("**Momentum note:** RSI is above 70, often described as strong/overbought momentum. It can remain elevated during strong trends.")
    elif rsi_now <= 30:
        st.write("**Momentum note:** RSI is below 30, often described as weak/oversold momentum. It can remain low during strong declines.")
    else:
        st.write("**Momentum note:** RSI is between 30 and 70, so it is not at the usual extreme levels.")

# Main chart is a genuine OHLC candlestick chart, with moving averages.
st.divider()
st.subheader(f"Candlestick chart — {interval_label}")
st.caption("Each candle shows Open, High, Low and Close. EMA20 and EMA50 are smoothed average-price lines used to help describe trend direction.")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
    name="Candles", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
))
fig.add_trace(go.Scatter(x=data.index, y=data["EMA20"], name="EMA20", line=dict(width=1.5, color="#f5c542")))
fig.add_trace(go.Scatter(x=data.index, y=data["EMA50"], name="EMA50", line=dict(width=1.5, color="#42a5f5")))
fig.update_layout(template="plotly_dark", height=600, xaxis_title="Time", yaxis_title="Price",
                  xaxis_rangeslider_visible=False, hovermode="x unified", legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig, use_container_width=True)

# Full-history pattern table: every candle gets simple descriptive labels.
def candle_label(row):
    body = abs(row["Close"] - row["Open"])
    rng = max(row["High"] - row["Low"], 1e-12)
    body_ratio = body / rng
    if body_ratio < 0.15:
        return "Small body / indecision"
    return "Bullish candle" if row["Close"] > row["Open"] else "Bearish candle" if row["Close"] < row["Open"] else "Flat candle"

data["Candle read"] = data.apply(candle_label, axis=1)
data["Trend read"] = np.select(
    [data["Close"] > data["EMA20"], data["Close"] < data["EMA20"]],
    ["Above EMA20", "Below EMA20"], default="At EMA20"
)
data["Momentum read"] = np.select(
    [data["RSI14"] >= 70, data["RSI14"] <= 30],
    ["RSI high", "RSI low"], default="RSI middle"
)
st.divider()
st.subheader("Candle-by-candle trend table")
st.write("This table uses candle records, not a line chart. It includes the whole selected history; use the table controls to sort and inspect individual candles.")
table = data.copy().sort_index(ascending=False)
table = table.rename(columns={
    "Open":"Open", "High":"High", "Low":"Low", "Close":"Close",
    "Change_pct":"Change %", "Body_pct":"Body %", "Range_pct":"Range %",
    "Move_5":"Change over 5 candles %", "Move_10":"Change over 10 candles %",
    "RSI14":"RSI (14)", "EMA20":"EMA20", "EMA50":"EMA50", "ATR14":"ATR (14)"
})
cols = ["Open","High","Low","Close","Volume","Change %","Body %","Range %",
        "EMA20","EMA50","RSI (14)","ATR (14)","Candle read","Trend read","Momentum read"]
cols = [c for c in cols if c in table.columns]
st.dataframe(table[cols].style.format(precision=4, na_rep="—"), use_container_width=True, height=420)

# Short projection: regression across recent candles, with residual/ATR-based band.
# The band is not forced narrow: it reflects observed variability.
st.divider()
st.subheader("Short-range projection")
st.caption("The projection is a statistical baseline based on recent closes. The shaded area uses recent forecast errors and ATR to communicate uncertainty; it is not a calibrated probability interval.")
max_projection = 10
projection_steps = min(max_projection, max(5, 10 // max(config["minutes"], 1)))
lookback = min(100, len(close))
recent = close.iloc[-lookback:].to_numpy(dtype=float)
x = np.arange(lookback, dtype=float)
slope, intercept = np.polyfit(x, recent, 1) if lookback >= 3 else (0.0, latest)
fitted = slope * x + intercept
resid_sd = float(np.std(recent - fitted, ddof=1)) if lookback > 3 else atr_now
future_x = np.arange(lookback, lookback + projection_steps)
projected = slope * future_x + intercept
step_sd = float(data["Change"].tail(lookback).dropna().std()) if len(data["Change"].tail(lookback).dropna()) > 2 else atr_now
base_uncertainty = max(resid_sd, atr_now, step_sd if np.isfinite(step_sd) else 0, latest * 0.0001)
spread = np.sqrt(np.arange(1, projection_steps + 1)) * base_uncertainty
step_delta = pd.Timedelta(minutes=config["minutes"])
future_times = [data.index[-1] + step_delta * i for i in range(1, projection_steps + 1)]
proj_fig = go.Figure()
visible = data.tail(min(200, len(data)))
proj_fig.add_trace(go.Candlestick(
    x=visible.index, open=visible["Open"], high=visible["High"], low=visible["Low"], close=visible["Close"],
    name="Historical candles", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
))
proj_fig.add_trace(go.Scatter(x=future_times, y=projected, mode="lines+markers", name="Trend baseline",
    line=dict(dash="dash", width=2, color="#f5c542")))
proj_fig.add_trace(go.Scatter(x=future_times, y=projected + spread, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
proj_fig.add_trace(go.Scatter(x=future_times, y=projected - spread, mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(100,149,237,0.18)", name="Observed-variability band", hoverinfo="skip"))
proj_fig.update_layout(template="plotly_dark", height=520, xaxis_title="Time", yaxis_title="Price",
    xaxis_rangeslider_visible=False, hovermode="x unified")
st.plotly_chart(proj_fig, use_container_width=True)

projected_change = ((float(projected[-1]) / latest) - 1) * 100 if latest else 0
if projected_change > 0.05:
    projected_direction = "upward"
elif projected_change < -0.05:
    projected_direction = "downward"
else:
    projected_direction = "roughly flat"
st.write(f"**Baseline reading:** the simple fitted trend points {projected_direction} over the next {projection_steps * config['minutes']} minutes ({projected_change:+.3f}% from the latest close).")
st.write("**What could change this reading:** a move back across EMA20/EMA50, a new swing high/low, or a sharp increase in candle range. The projection does not know future news or order-book changes.")
st.caption("This dashboard describes historical price behaviour. It does not provide buy/sell signals, does not use a trained AI model or order-book data, and cannot guarantee a forecast. More historical candles improve context, but do not remove market uncertainty.")
