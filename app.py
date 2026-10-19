import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Market Pattern Explorer", page_icon="📈", layout="wide")
st.title("📈 Market Pattern Explorer")
st.caption("A plain-English view of recent price behaviour and a short-range statistical projection. Not a trading-signal service.")

INTERVALS = {
    "1 minute": {"yf": "1m", "tv": "1", "periods": ["1d", "5d", "7d"], "default": "5d", "limit": "1-minute history is usually limited to around 7 days by Yahoo Finance. You cannot reliably request a full month of 1-minute candles from this source."},
    "5 minutes": {"yf": "5m", "tv": "5", "periods": ["5d", "1mo", "2mo"], "default": "1mo", "limit": "5-minute history is usually limited to around 60 days. One month is a useful recent-history starting point."},
    "15 minutes": {"yf": "15m", "tv": "15", "periods": ["5d", "1mo", "3mo", "60d"], "default": "1mo", "limit": "Intraday history availability varies by ticker and provider."},
    "1 hour": {"yf": "60m", "tv": "60", "periods": ["5d", "1mo", "3mo", "6mo", "1y"], "default": "1mo", "limit": "Hourly history availability varies by ticker and provider."},
    "1 day": {"yf": "1d", "tv": "D", "periods": ["1mo", "3mo", "6mo", "1y", "2y"], "default": "1mo", "limit": "Daily candles are better for looking at longer-term movement, not the next few minutes."},
}

with st.sidebar:
    st.header("Market settings")
    symbol = st.text_input("Ticker symbol", value="AAPL", help="The market code for the asset. Examples: AAPL (Apple), MSFT (Microsoft), NVDA (Nvidia), BTC-USD (Bitcoin).")
    st.caption("**Ticker = what to study.** Use the code shown by your data provider.")
    interval_label = st.selectbox("Candle interval", list(INTERVALS.keys()), index=0, help="Each candle summarises price movement over this amount of time. A 1-minute candle covers one minute; a 5-minute candle covers five minutes.")
    st.caption("**Candle interval = the size of each step on the graph.** Short intervals show more detail but also more random noise.")
    config = INTERVALS[interval_label]
    period = st.selectbox("Recent history to analyse", config["periods"], index=config["periods"].index(config["default"]), help="How far back the app looks to find patterns. Recent history can reflect current conditions better, but short samples can also be misleading.")
    st.caption("**History = the sample of past candles studied.** For 1-minute data, choose up to 7 days; for 5-minute data, 1 month is a sensible starting point.")
    refresh = st.button("Refresh market data", type="primary", use_container_width=True, help="Fetch the latest candles available from Yahoo Finance. This does not guarantee exchange-live prices.")
    st.info(config["limit"])
    st.caption("Data is refreshed when you press the button or reload the page. Provider updates may be delayed.")

symbol_clean = symbol.strip().upper().replace(" ", "")
if not symbol_clean:
    st.warning("Enter a ticker symbol in Market settings.")
    st.stop()

# Keep the familiar TradingView chart; analysis and forecast below use Yahoo Finance candles.
tv_symbol = symbol_clean
tv_html = f"""
<div class="tradingview-widget-container" style="height:500px;width:100%">
  <div id="tradingview_chart" style="height:500px;width:100%"></div>
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
st.subheader(f"Market chart — {symbol_clean} ({interval_label})")
st.components.v1.html(tv_html, height=510, scrolling=False)

@st.cache_data(ttl=60, show_spinner=False)
def load_data(ticker, history_period, interval):
    df = yf.download(ticker, period=history_period, interval=interval, auto_adjust=True, progress=False, prepost=False, threads=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    wanted = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[wanted].copy()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["Open", "High", "Low", "Close"])

cache_key = (symbol_clean, period, config["yf"])
if refresh or "market_data" not in st.session_state or st.session_state.get("loaded_key") != cache_key:
    try:
        with st.spinner(f"Loading recent {interval_label.lower()} candles for {symbol_clean}..."):
            st.session_state.market_data = load_data(symbol_clean, period, config["yf"])
            st.session_state.loaded_key = cache_key
    except Exception as exc:
        st.error(f"Could not load market data: {exc}")
        st.stop()

data = st.session_state.market_data.copy()
if data.empty:
    st.error("No candles were returned. Check the ticker, try a shorter history, or try another interval. Some markets have limited data access.")
    st.stop()

# Candle-by-candle measurements: describe the candle's shape and change, not a trading instruction.
data["Change"] = data["Close"].diff()
data["Change_pct"] = data["Close"].pct_change() * 100
data["Candle_range"] = data["High"] - data["Low"]
data["Body"] = data["Close"] - data["Open"]
data["Body_pct"] = np.where(data["Open"] != 0, data["Body"] / data["Open"] * 100, np.nan)
data["Upper_wick"] = data["High"] - data[["Open", "Close"]].max(axis=1)
data["Lower_wick"] = data[["Open", "Close"]].min(axis=1) - data["Low"]
data["Range_pct"] = np.where(data["Close"] != 0, data["Candle_range"] / data["Close"] * 100, np.nan)
data["Move_5"] = data["Close"].pct_change(5) * 100
data["Move_10"] = data["Close"].pct_change(10) * 100

close = data["Close"].astype(float).dropna()
latest = float(close.iloc[-1])
previous = float(close.iloc[-2]) if len(close) > 1 else latest
last_change = (latest / previous - 1) * 100 if previous else 0

# Use the most recent 30 candle closes for a simple linear-trend extrapolation.
# The uncertainty band is estimated from recent one-candle changes and grows with forecast distance.
lookback = min(30, len(close))
recent = close.iloc[-lookback:].to_numpy(dtype=float)
steps = min(10, max(5, len(recent) // 3))
future_x = np.arange(lookback, lookback + steps)
if lookback >= 3:
    x = np.arange(lookback, dtype=float)
    slope, intercept = np.polyfit(x, recent, 1)
    fitted = slope * x + intercept
    residuals = recent - fitted
    residual_sd = float(np.std(residuals, ddof=1)) if len(residuals) > 2 else 0.0
    projected = slope * future_x + intercept
    recent_changes = np.diff(recent)
    step_sd = float(np.std(recent_changes, ddof=1)) if len(recent_changes) > 2 else 0.0
    # A broad, increasing band communicates uncertainty; not a calibrated confidence interval.
    spread = np.sqrt(np.arange(1, steps + 1)) * max(step_sd, residual_sd * 0.5, latest * 0.0001)
    upper = projected + 1.5 * spread
    lower = projected - 1.5 * spread
    start_fit = slope * x + intercept
else:
    projected = np.repeat(latest, steps)
    upper = projected + latest * 0.002
    lower = projected - latest * 0.002
    start_fit = recent

# Human-readable summary based on the most recent 10 candles and slope fit.
recent_n = min(10, len(close))
recent_move = (float(close.iloc[-1]) / float(close.iloc[-recent_n]) - 1) * 100 if recent_n > 1 else 0.0
if len(recent) >= 3:
    fitted_change = ((float(projected[-1]) / float(projected[0])) - 1) * 100 if projected[0] else 0.0
else:
    fitted_change = 0.0
if recent_move > 0.05:
    direction_words = "has generally moved up"
elif recent_move < -0.05:
    direction_words = "has generally moved down"
else:
    direction_words = "has mostly moved sideways"
if abs(fitted_change) < 0.05:
    forecast_words = "roughly flat in the simple trend projection"
elif fitted_change > 0:
    forecast_words = "slightly upward in the simple trend projection"
else:
    forecast_words = "slightly downward in the simple trend projection"

st.subheader("At a glance")
a, b, c = st.columns(3)
a.metric("Latest available price", f"{latest:,.4f}", f"{last_change:+.3f}% vs previous candle")
b.metric(f"Price change over last {recent_n} candles", f"{recent_move:+.3f}%")
c.metric("Candles analysed", f"{len(data):,}", help="Each row is one candle in the selected interval and history window.")
st.caption(f"Data source: Yahoo Finance · Latest candle timestamp: {data.index[-1]} · This is the latest data returned by the provider, not necessarily the current live market price.")

st.divider()
st.subheader("Recent price and 5–10 minute projection")
st.write("**Solid line:** observed closing prices. **Dashed line:** a straight-line trend fitted to the latest candles and extended forward. **Shaded band:** a rough uncertainty range that widens with time. It is an estimate, not a promise or a statistically validated confidence interval.")
fig = go.Figure()
show_n = min(100, len(data))
visible = data.iloc[-show_n:]
fig.add_trace(go.Scatter(x=visible.index, y=visible["Close"], mode="lines", name="Observed close", line=dict(width=2)))
if lookback >= 3:
    fit_index = data.index[-lookback:]
    fig.add_trace(go.Scatter(x=fit_index, y=start_fit, mode="lines", name="Recent straight-line fit", line=dict(dash="dot", width=1.5)))
# Build future timestamps using the median spacing between available candles.
if len(data.index) >= 2:
    diffs = pd.Series(data.index).diff().dropna()
    step_delta = diffs.median()
else:
    step_delta = pd.Timedelta(minutes=1)
future_times = [data.index[-1] + step_delta * i for i in range(1, steps + 1)]
forecast_x = [data.index[-1]] + future_times
forecast_y = [latest] + list(projected)
upper_y = [latest] + list(upper)
lower_y = [latest] + list(lower)
fig.add_trace(go.Scatter(x=forecast_x, y=upper_y, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", name="Upper range"))
fig.add_trace(go.Scatter(x=forecast_x, y=lower_y, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(100, 149, 237, 0.18)", name="Uncertainty range", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=forecast_x, y=forecast_y, mode="lines+markers", name="Simple projection", line=dict(dash="dash", width=2)))
fig.add_vline(x=data.index[-1], line_dash="dash", line_width=1, annotation_text="Now (latest data)")
fig.update_layout(template="plotly_dark", height=520, xaxis_title="Time", yaxis_title="Price", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
st.plotly_chart(fig, use_container_width=True)
st.info(f"Plain-English reading: over the latest {recent_n} candles, price {direction_words}. Extending a straight line through the most recent {lookback} closes gives a projection that is {forecast_words}. Short-term prices can change direction quickly; treat this as one simple mathematical baseline, not a reliable prediction of what must happen.")

st.divider()
st.subheader("What each candle says")
st.write("A candle summarises the price during one interval. The app measures every candle in the selected history, then shows the latest 30 below so the table stays readable.")
with st.expander("How to read a candle (plain English)"):
    st.markdown("""
- **Open:** first price recorded during the interval.
- **High / Low:** highest and lowest prices recorded during the interval.
- **Close:** last price recorded during the interval.
- **Candle body:** close minus open. Positive means the interval ended above its opening price; negative means it ended below.
- **Wicks:** the distance from the body to the high or low. Long wicks mean price travelled further but did not finish there.
- **Change %:** percentage change from the previous candle's close to this candle's close.
- **Range %:** high-to-low distance as a percentage of price. A larger number means a wider move during that candle.
""")

recent_table = data.tail(30).copy().sort_index(ascending=False)
recent_table = recent_table.rename(columns={"Open":"Open", "High":"High", "Low":"Low", "Close":"Close", "Volume":"Volume", "Change_pct":"Change %", "Body_pct":"Body %", "Range_pct":"Range %", "Move_5":"Change over 5 candles %", "Move_10":"Change over 10 candles %"})
show_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Change %", "Body %", "Range %", "Change over 5 candles %", "Change over 10 candles %"] if c in recent_table.columns]
st.dataframe(recent_table[show_cols].style.format(precision=4, na_rep="—"), use_container_width=True)

with st.expander("Summary of patterns in the selected history"):
    returns = data["Change_pct"].dropna()
    if len(returns):
        up = int((returns > 0).sum())
        down = int((returns < 0).sum())
        unchanged = int((returns == 0).sum())
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Candles closing up", f"{up:,}", f"{up/len(returns)*100:.1f}% of changes")
        s2.metric("Candles closing down", f"{down:,}", f"{down/len(returns)*100:.1f}% of changes")
        s3.metric("Unchanged closes", f"{unchanged:,}")
        s4.metric("Typical candle range", f"{data['Range_pct'].median():.3f}% of price")
        st.write("These are descriptions of the selected past sample. They do not show that the same pattern will continue.")
    else:
        st.write("There are not enough candles to summarise price changes.")

st.caption("This version does not recommend trades or strategies. The forecast is a simple linear extrapolation of recent closes, with a rough uncertainty band. It does not yet use a trained AI model, order-book data, or a guaranteed live exchange feed. Yahoo Finance can delay, omit, or limit intraday data.")
