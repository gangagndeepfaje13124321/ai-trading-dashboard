import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from live_feed import get_first_valid_price
from trade_candles import KrakenTradeFeed
from forecasting import build_forecast_candles, linear_trend_forecast, rolling_origin_backtest

st.set_page_config(page_title="Market Pattern Explorer", page_icon="📈", layout="wide")
st.title("📈 Market Pattern Explorer")
st.caption("Candle-based market analysis and a short-range statistical projection. Educational only — not trading advice.")


@st.cache_resource(show_spinner=False)
def get_kraken_trade_feed(symbol):
    """Share one thread-safe Kraken trade connection per market across sessions."""
    feed = KrakenTradeFeed(symbol)
    feed.start()
    return feed

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
st.caption("Each candle shows Open, High, Low and Close. EMA20 = average-like line that responds to the last 20 candles; EMA50 responds more slowly to the last 50. They are context, not predictions. Use the toolbar to box-zoom, double-click to reset, and the mouse wheel to zoom.")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
    name="Candles", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
))
fig.add_trace(go.Scatter(x=data.index, y=data["EMA20"], name="EMA20", line=dict(width=1.5, color="#f5c542")))
fig.add_trace(go.Scatter(x=data.index, y=data["EMA50"], name="EMA50", line=dict(width=1.5, color="#42a5f5")))
fig.update_layout(template="plotly_dark", height=600, xaxis_title="Time", yaxis_title="Price",
                  xaxis_rangeslider_visible=False, hovermode="x unified", legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"]}, key="history_candles")

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

# Short projection: a transparent linear-trend baseline, checked against later candles.
st.divider()
st.subheader("Short-range baseline forecast")
st.caption("This is a simple line fitted to recent closes, not a trained AI model. Forecast OHLC high/low values use typical recent candle wicks. The shaded range estimates uncertainty from recent variation; it is not a calibrated probability interval.")
max_projection = 10
projection_steps = min(max_projection, max(5, 10 // max(config["minutes"], 1)))
lookback = min(100, len(close))
recent = close.iloc[-lookback:].to_numpy(dtype=float)
x = np.arange(lookback, dtype=float)
slope, intercept = np.polyfit(x, recent, 1) if lookback >= 3 else (0.0, latest)
fitted = slope * x + intercept
resid_sd = float(np.std(recent - fitted, ddof=1)) if lookback > 3 else atr_now
projected = np.asarray(
    linear_trend_forecast(close.to_numpy(dtype=float), steps=projection_steps, lookback=lookback),
    dtype=float,
)
forecast_candles = build_forecast_candles(data, projected, lookback=lookback)
step_sd = float(data["Change"].tail(lookback).dropna().std()) if len(data["Change"].tail(lookback).dropna()) > 2 else atr_now
base_uncertainty = max(resid_sd, atr_now, step_sd if np.isfinite(step_sd) else 0, latest * 0.0001)
spread = np.sqrt(np.arange(1, projection_steps + 1)) * base_uncertainty
step_delta = pd.Timedelta(minutes=config["minutes"])
future_times = [data.index[-1] + step_delta * i for i in range(1, projection_steps + 1)]

backtest = rolling_origin_backtest(
    data.loc[:, ["Open", "High", "Low", "Close"]], lookback=lookback
)
st.markdown("**How this baseline performed on later historical candles**")
if backtest is None:
    st.info("There are not enough historical candles yet for a separate walk-forward test. Load a longer history or use a larger candle interval.")
else:
    test_columns = st.columns(5)
    test_columns[0].metric("Trend close MAE", f"{backtest['trend_close_mae']:,.4f}", help="Average absolute close-price miss on the held-out test candles. Lower is better.")
    test_columns[1].metric("Last-close baseline MAE", f"{backtest['last_close_mae']:,.4f}", help="Simple comparison: predict that the next close stays at the latest close.")
    test_columns[2].metric("Trend close RMSE", f"{backtest['trend_close_rmse']:,.4f}", help="Root mean square error. Like MAE, lower is better; it gives extra weight to larger misses.")
    test_columns[3].metric("Test candles", f"{backtest['holdout_candles']}", help="Final chronological portion of one-step walk-forward predictions.")
    test_columns[4].metric("Trend direction correct", f"{backtest['directional_accuracy_pct']:.1f}%", help="Share of test candles where the predicted up/down/flat direction matched the next close.")
    st.caption(
        "MAE is the average size of the miss. RMSE gives larger misses more weight. "
        f"Last-close RMSE: {backtest['last_close_rmse']:,.4f}."
    )
    st.caption(
        "At each test step, the forecast used only candles available before that target candle. "
        f"OHLC average absolute miss (price units): open {backtest['ohlc_mae']['open']:,.4f}, "
        f"high {backtest['ohlc_mae']['high']:,.4f}, low {backtest['ohlc_mae']['low']:,.4f}, "
        f"close {backtest['ohlc_mae']['close']:,.4f}. This is historical testing, not a guarantee of future accuracy."
    )
    if backtest["trend_close_mae"] < backtest["last_close_mae"]:
        improvement = (1 - backtest["trend_close_mae"] / backtest["last_close_mae"]) * 100 if backtest["last_close_mae"] > 0 else 0
        st.caption(f"On this selected history, the trend baseline's average close-price miss was {improvement:.1f}% lower than the last-close comparison.")
    elif backtest["last_close_mae"] > 0:
        ratio = backtest["trend_close_mae"] / backtest["last_close_mae"]
        st.caption(f"On this selected history, the trend baseline's average close-price miss was {ratio:.1f}× the last-close comparison's miss.")
    else:
        st.caption("The last-close comparison had zero average close-price error on these test candles; the trend baseline did not beat it.")

proj_fig = go.Figure()
max_visible_candles = min(200, len(data))
if max_visible_candles > 20:
    history_options = [count for count in (20, 40, 60, 100, 150, 200) if count < max_visible_candles]
    history_options.append(max_visible_candles)
    default_visible_candles = min(60, max_visible_candles)
    visible_candle_count = st.select_slider(
        "Recent history shown with the forecast",
        options=history_options,
        value=default_visible_candles,
        format_func=lambda count: f"{count} candles",
        help="Choose fewer candles to zoom in on the forecast. The full history remains available in the main chart above.",
    )
else:
    visible_candle_count = max_visible_candles
visible = data.tail(visible_candle_count)
proj_fig.add_trace(go.Candlestick(
    x=visible.index, open=visible["Open"], high=visible["High"], low=visible["Low"], close=visible["Close"],
    name="Historical candles", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    legendrank=1,
))
proj_fig.add_trace(go.Scatter(x=future_times, y=projected, mode="lines", name="Trend baseline",
    line=dict(dash="dash", width=1.5, color="#f5c542"), legendrank=3))
proj_fig.add_trace(go.Scatter(x=future_times, y=projected + spread, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
proj_fig.add_trace(go.Scatter(x=future_times, y=projected - spread, mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(100,149,237,0.18)", name="Observed-variability band", hoverinfo="skip", legendrank=4))
proj_fig.add_trace(go.Candlestick(
    x=future_times,
    open=[candle["open"] for candle in forecast_candles],
    high=[candle["high"] for candle in forecast_candles],
    low=[candle["low"] for candle in forecast_candles],
    close=[candle["close"] for candle in forecast_candles],
    name="Forecast OHLC baseline",
    increasing_line_color="#64b5f6", decreasing_line_color="#ffab91",
    increasing_fillcolor="rgba(100,181,246,0.7)", decreasing_fillcolor="rgba(255,171,145,0.7)",
    legendrank=2,
))
proj_fig.update_layout(template="plotly_dark", height=520, xaxis_title="Time", yaxis_title="Price",
    xaxis_rangeslider_visible=False, hovermode="x unified",
    uirevision=f"{symbol_clean}-{config['yf']}-{visible_candle_count}")
st.caption(f"Showing the last {visible_candle_count} historical candles plus the forecast. Use fewer candles in the selector to zoom in; hover over a forecast candle to inspect its Open, High, Low and Close.")
st.plotly_chart(
    proj_fig, use_container_width=True,
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "doubleClick": "reset",
        "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
    },
    key="historical_projection",
)
st.caption("Blue/orange future candles show estimated Open, High, Low and Close. Their open/high/low use simple historical assumptions; the model being tested forecasts Close only. The shaded band is an observed-variation guide, not a probability promise.")

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


# Live sampled price and projection. This is deliberately separate from Yahoo historical candles.
# Public spot ticker requests give sampled quotes; the trade stream below builds one-second OHLC candles.
st.divider()
st.subheader("Live spot tracker + rolling short-range projection")
st.caption(
    "For supported markets this samples a public spot ticker about every 2 seconds (Kraken first, Binance fallback). "
    "These are sampled observations, not one-second candles. The dashed line and shaded range are "
    "simple baselines, not calibrated predictions."
)

live_symbol_map = {
    "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "SOL-USD": "SOLUSDT",
    "BNB-USD": "BNBUSDT", "XRP-USD": "XRPUSDT", "DOGE-USD": "DOGEUSDT"
}
kraken_trade_symbols = {
    "BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD", "SOL-USD": "SOL/USD",
    "BNB-USD": "BNB/USD", "XRP-USD": "XRP/USD", "DOGE-USD": "DOGE/USD"
}
live_pair = live_symbol_map.get(symbol_clean)
trade_symbol = kraken_trade_symbols.get(symbol_clean)
if not live_pair:
    st.info("Live spot sampling currently supports BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD and DOGE-USD. Other tickers still use the historical chart above.")
else:
    @st.fragment(run_every="2s")
    def live_market_panel():
        import requests
        try:
            def fetch_kraken_price():
                kraken_pairs = {
                    "BTC-USD": "XBTUSD", "ETH-USD": "ETHUSD", "SOL-USD": "SOLUSD",
                    "BNB-USD": "BNBUSD", "XRP-USD": "XRPUSD", "DOGE-USD": "DOGEUSD"
                }
                response = requests.get(
                    "https://api.kraken.com/0/public/Ticker",
                    params={"pair": kraken_pairs[symbol_clean]}, timeout=5,
                    headers={"User-Agent": "Mozilla/5.0 MarketPatternExplorer/1.0"}
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("error"):
                    raise ValueError("Kraken API returned an error")
                result = payload.get("result", {})
                if not result:
                    raise ValueError("Kraken returned no ticker data")
                return next(iter(result.values()))["c"][0]

            def fetch_binance_price():
                response = requests.get(
                    f"https://api.binance.com/api/v3/ticker/price?symbol={live_pair}",
                    timeout=4, headers={"User-Agent": "Mozilla/5.0 MarketPatternExplorer/1.0"}
                )
                response.raise_for_status()
                return response.json()["price"]

            # Kraken is first because it is currently reachable from Streamlit Cloud;
            # Binance remains as a fallback if Kraken is unavailable.
            spot, provider, errors = get_first_valid_price([
                ("Kraken", fetch_kraken_price),
                ("Binance", fetch_binance_price),
            ])
            now = pd.Timestamp.now(tz="UTC")
            key = f"live_samples_{live_pair}"
            samples = st.session_state.get(key, [])
            # Do not combine prices from two exchanges in the same chart or volatility estimate.
            if samples and samples[-1].get("provider") != provider:
                samples = []
            samples.append({"time": now, "price": spot, "provider": provider})
            samples = samples[-180:]
            st.session_state[key] = samples
            st.metric("Live sampled spot price", f"{spot:,.6f}", help=f"Current {provider} spot quote. This is a sampled quote, not a one-second OHLC candle.")
            st.caption(f"Source: {provider} · received {now.strftime('%Y-%m-%d %H:%M:%S')} UTC · polled about every 2 seconds.")
            if errors:
                st.caption("Fallback detail: " + "; ".join(errors))
            live_df = pd.DataFrame(samples)
            if len(live_df) >= 2:
                # Calculate the recent historical baseline's direction, then anchor it to live price.
                base_close = close.tail(min(60, len(close))).to_numpy(dtype=float)
                bx = np.arange(len(base_close), dtype=float)
                bslope = np.polyfit(bx, base_close, 1)[0] if len(base_close) >= 3 else 0.0
                # Forecast horizon is 10 minutes, represented as 2-second steps.
                n_steps = 300
                last_time = live_df["time"].iloc[-1]
                future_t = [last_time + pd.Timedelta(seconds=2 * i) for i in range(1, n_steps + 1)]
                hist_t = live_df["time"].tolist()
                hist_p = live_df["price"].astype(float).tolist()
                # Drift is deliberately damped: short history is not a reliable predictor.
                hist_move = (hist_p[-1] / hist_p[0] - 1) if hist_p[0] else 0.0
                drift_per_step = np.clip(hist_move / max(len(hist_p) - 1, 1), -0.00002, 0.00002)
                horizon_steps = np.arange(1, n_steps + 1)
                # Tiny baseline drift from recent candles, damped to avoid pretending confidence.
                baseline_drift = np.clip((bslope / max(abs(latest), 1e-12)) / max(config["minutes"] * 30, 1), -0.00001, 0.00001)
                pred = spot * (1 + (drift_per_step * 0.35 + baseline_drift * 0.25) * horizon_steps)
                # Band grows with horizon and sampled volatility.
                changes = pd.Series(hist_p).pct_change().dropna()
                vol = float(changes.std()) if len(changes) > 2 and np.isfinite(changes.std()) else 0.00005
                width = spot * max(vol, 0.00003) * np.sqrt(horizon_steps) * 1.5
                live_fig = go.Figure()
                live_fig.add_trace(go.Scatter(x=hist_t, y=hist_p, mode="lines", name="Live sampled price", line=dict(width=2, color="#26a69a")))
                live_fig.add_trace(go.Scatter(x=future_t, y=pred, mode="lines", name="Rolling baseline", line=dict(width=2, dash="dash", color="#f5c542")))
                live_fig.add_trace(go.Scatter(x=future_t, y=pred + width, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                live_fig.add_trace(go.Scatter(x=future_t, y=pred - width, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(100,149,237,0.18)", name="Uncertainty range", hoverinfo="skip"))
                live_fig.update_layout(template="plotly_dark", height=430, hovermode="x unified",
                    xaxis_title="Time (UTC)", yaxis_title="Price", margin=dict(l=10,r=10,t=35,b=10))
                st.plotly_chart(live_fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False}, key=f"live_chart_{live_pair}")
                st.caption(f"Live observations retained in this session: {len(live_df)}. The projection refreshes with the spot price; it is a baseline, not an AI/news-aware forecast.")
            else:
                st.info("Collecting live samples. The chart will appear after the next update.")
        except Exception as exc:
            st.warning(f"Live spot feed is temporarily unavailable: {exc}. The historical chart above remains available.")
    live_market_panel()

if trade_symbol:
    st.subheader("Kraken trade-based 1-second candles")
    st.caption(
        "Each candle groups actual Kraken trades by their exchange timestamp into a UTC second. "
        "Seconds with no trades are left empty. Kraken sends a small recent-trades snapshot when the feed connects; "
        "the chart then builds history while this app is running."
    )
    trade_feed = get_kraken_trade_feed(trade_symbol)

    @st.fragment(run_every="1s")
    def kraken_candle_panel():
        snapshot = trade_feed.snapshot()
        candles = snapshot["candles"]
        if snapshot["status"] == "connected":
            last_trade_at = snapshot["last_trade_at"]
            detail = "Kraken trade feed connected"
            if last_trade_at is not None:
                detail += f" · last exchange trade {last_trade_at.strftime('%H:%M:%S')} UTC"
            st.caption(detail)
        elif snapshot["status"] == "unavailable":
            st.warning("Kraken trade feed unavailable: " + snapshot["detail"])
        else:
            st.caption("Kraken trade feed: " + snapshot["detail"])

        if candles:
            candle_fig = go.Figure(data=[go.Candlestick(
                x=[candle["time"] for candle in candles],
                open=[candle["open"] for candle in candles],
                high=[candle["high"] for candle in candles],
                low=[candle["low"] for candle in candles],
                close=[candle["close"] for candle in candles],
                name="Kraken trades"
            )])
            candle_fig.update_layout(
                template="plotly_dark", height=440, hovermode="x unified",
                xaxis_title="Time (UTC)", yaxis_title="Price (USD)",
                xaxis_rangeslider_visible=True,
                margin=dict(l=10, r=10, t=25, b=10)
            )
            st.plotly_chart(
                candle_fig, use_container_width=True,
                config={"scrollZoom": True, "displaylogo": False},
                key=f"kraken_trade_candles_{symbol_clean}"
            )
            st.caption(f"Showing {len(candles)} recent trade-containing seconds. The current second can update as more trades arrive.")
        else:
            st.info("Waiting for Kraken trades. The candle chart will appear when trade data arrives.")

    kraken_candle_panel()
else:
    st.info("Kraken one-second trade candles currently support BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD, and DOGE-USD.")

st.divider()
st.subheader("Recent news and community discussion")
st.caption("Headlines and public discussion can provide context, but this simple feed does not measure sentiment reliably and does not alter the price projection. Social platforms may limit access or change feeds.")
news_col, social_col = st.columns(2)

def load_rss_items(url, limit=6):
    import requests
    import xml.etree.ElementTree as ET
    try:
        r = requests.get(url, timeout=6, headers={"User-Agent": "MarketPatternExplorer/1.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = []
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "Untitled").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            items.append((title, link, pub))
        return items
    except Exception:
        return []

with news_col:
    st.markdown("**News headlines**")
    news_query = symbol_clean.replace("-USD", " crypto")
    news_items = load_rss_items(
        "https://news.google.com/rss/search?q=" + requests.utils.quote(news_query + " market") +
        "&hl=en-US&gl=US&ceid=US:en", 6
    ) if "requests" in globals() else []
    if not news_items:
        # requests is imported inside the live fragment, so import explicitly for the RSS fetch.
        import requests
        news_items = load_rss_items(
            "https://news.google.com/rss/search?q=" + requests.utils.quote(news_query + " market") +
            "&hl=en-US&gl=US&ceid=US:en", 6
        )
    if news_items:
        for title, link, pub in news_items:
            st.markdown(f"- [{title}]({link})" + (f"  \n  <small>{pub}</small>" if pub else ""), unsafe_allow_html=True)
    else:
        st.write("News feed unavailable right now.")

with social_col:
    st.markdown("**Public Reddit discussions**")
    import requests
    social_query = symbol_clean.replace("-USD", "")
    social_items = load_rss_items(
        "https://www.reddit.com/search.rss?q=" + requests.utils.quote(social_query + " crypto") +
        "&sort=new&t=day", 6
    )
    if social_items:
        for title, link, pub in social_items:
            st.markdown(f"- [{title}]({link})" + (f"  \n  <small>{pub}</small>" if pub else ""), unsafe_allow_html=True)
    else:
        st.write("Reddit feed unavailable or rate-limited.")
