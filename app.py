"""Compact TradingView dashboard with tested minute methods and live trade bars."""

from datetime import datetime, timezone
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from alpaca_market import AlpacaTradeFeed, fetch_alpaca_minute_history
from live_forecast import LiveForecastEvaluator, make_live_forecast
from market_history import fetch_kraken_minute_history, merge_history_with_live_trades
from strategy_lab import fit_and_evaluate, make_minute_forecast
from trade_candles import KrakenTradeFeed


MARKETS = {
    "Bitcoin (BTC/USD)": {"kind": "crypto", "pair": "BTC/USD", "tv": "KRAKEN:BTCUSD"},
    "Ethereum (ETH/USD)": {"kind": "crypto", "pair": "ETH/USD", "tv": "KRAKEN:ETHUSD"},
    "Solana (SOL/USD)": {"kind": "crypto", "pair": "SOL/USD", "tv": "KRAKEN:SOLUSD"},
    "BNB (BNB/USD)": {"kind": "crypto", "pair": "BNB/USD", "tv": "KRAKEN:BNBUSD"},
    "XRP (XRP/USD)": {"kind": "crypto", "pair": "XRP/USD", "tv": "KRAKEN:XRPUSD"},
    "Dogecoin (DOGE/USD)": {"kind": "crypto", "pair": "DOGE/USD", "tv": "KRAKEN:DOGEUSD"},
    "Apple (AAPL)": {"kind": "stock", "ticker": "AAPL", "tv": "NASDAQ:AAPL"},
    "Tesla (TSLA)": {"kind": "stock", "ticker": "TSLA", "tv": "NASDAQ:TSLA"},
}
HORIZONS_SECONDS = (10, 30, 60, 120)
MAX_STALE_SECONDS = 10

st.set_page_config(
    page_title="Market Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
      section.main > div { padding-top: .35rem; padding-bottom: .2rem; }
      [data-testid="stAppViewContainer"] { background: radial-gradient(ellipse at 15% 0%, #111c32 0%, #0b0f18 58%); }
      h1 { letter-spacing: -.035em; }
      div[data-testid="stMetric"] { background: linear-gradient(145deg, rgba(26,39,60,.92), rgba(15,22,34,.94)); border: 1px solid rgba(107,145,194,.25); border-radius: 10px; padding: .6rem .8rem; }
      div[data-testid="stPlotlyChart"] { margin-bottom: -.3rem; border: 1px solid rgba(107,145,194,.22); border-radius: 10px; overflow: hidden; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_kraken_trade_feed(pair):
    """Share one reconnecting Kraken trade connection for this market."""
    feed = KrakenTradeFeed(pair, max_candles=900)
    feed.start()
    return feed


@st.cache_resource(show_spinner=False)
def get_alpaca_trade_feed(symbol, api_key, secret_key):
    """Share one authenticated Alpaca IEX stock trade connection."""
    feed = AlpacaTradeFeed(symbol, api_key, secret_key, max_candles=900)
    feed.start()
    return feed


@st.cache_resource(show_spinner=False)
def get_second_evaluator(pair, horizon_seconds):
    """Keep a short, in-memory, live check for this market and horizon."""
    return LiveForecastEvaluator(horizon_seconds=horizon_seconds)


@st.cache_data(ttl=30, show_spinner=False)
def get_minute_history(pair):
    """Share Kraken's recent minute history and refresh it at most twice/minute."""
    return fetch_kraken_minute_history(pair)


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_minute_history(symbol, api_key, secret_key):
    """Cache a month of authenticated one-minute IEX history."""
    return fetch_alpaca_minute_history(symbol, api_key, secret_key, days=30)


@st.cache_data(ttl=300, show_spinner=False)
def get_historical_model(candles, horizon_minutes):
    """Cache historical rule selection until the source bars or horizon change."""
    return fit_and_evaluate(candles, horizon_minutes)


def _price_label(price):
    return f"${price:,.2f}" if abs(price) >= 100 else f"${price:,.5f}"


def _read_secret(name):
    """Read private provider credentials from Streamlit Secrets or environment."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.environ.get(name)


def _tradingview_chart(symbol, height=590):
    """Render TradingView's hosted Advanced Chart widget for the selected asset."""
    config = {
        "autosize": True,
        "symbol": symbol,
        "interval": "1",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "withdateranges": True,
        "hide_side_toolbar": False,
        "hide_top_toolbar": False,
        "allow_symbol_change": False,
        "save_image": False,
        "locale": "en",
        "calendar": False,
        "support_host": "https://www.tradingview.com",
    }
    markup = f"""
    <div class="tradingview-widget-container" style="height:{height}px;width:100%">
      <div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
        {json.dumps(config)}
      </script>
    </div>
    """
    components.html(markup, height=height, scrolling=False)


def _help_heading(label, explanation):
    """Show a visible question-mark button beside a section heading."""
    heading, help_button = st.columns([0.91, 0.09], vertical_alignment="center")
    with heading:
        st.markdown(f"**{label}**")
    with help_button:
        with st.popover("?", help=explanation):
            st.write(explanation)


def _second_figure(candles, name, asset_label, height=500):
    """Plot only real trades grouped into one-second OHLC candles."""
    fig = go.Figure()
    recent = candles[-120:]
    if recent:
        frame = pd.DataFrame(recent)
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        fig.add_trace(go.Candlestick(
            x=frame["time"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name=f"{asset_label} · actual 1s candles",
            increasing_line_color="#20b89a",
            decreasing_line_color="#ff5b64",
            increasing_fillcolor="rgba(32,184,154,.62)",
            decreasing_fillcolor="rgba(255,91,100,.62)",
        ))

    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis_title=None,
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        uirevision=name,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
        showlegend=True,
    )
    fig.update_xaxes(showgrid=False, tickformat="%H:%M:%S\n%b %d")
    fig.update_yaxes(fixedrange=False, tickformat=".6~g", gridcolor="rgba(150,160,180,.18)")
    return fig


header, market_picker, horizon_picker = st.columns([1.8, 1.4, 1.1], vertical_alignment="center")
with header:
    st.title("⚡ Market Dashboard")
with market_picker:
    selected_market = st.selectbox("Market", list(MARKETS), label_visibility="collapsed")
with horizon_picker:
    horizon_seconds = st.selectbox(
        "Seconds to score",
        HORIZONS_SECONDS,
        index=2,
        format_func=lambda seconds: f"Score next {seconds}s",
        label_visibility="collapsed",
        help="This sets how far ahead the live evaluator waits before scoring an experimental direction. It does not create a forecast candle path.",
    )

asset = MARKETS[selected_market]
is_stock = asset["kind"] == "stock"
asset_label = asset.get("ticker", asset.get("pair"))
api_key = _read_secret("ALPACA_API_KEY") if is_stock else None
secret_key = _read_secret("ALPACA_SECRET_KEY") if is_stock else None
if is_stock:
    feed = get_alpaca_trade_feed(asset_label, api_key, secret_key) if api_key and secret_key else None
    provider_label = "Alpaca IEX"
    pair = asset_label
else:
    pair = asset["pair"]
    feed = get_kraken_trade_feed(pair)
    provider_label = "Kraken"

evaluator = get_second_evaluator(pair, horizon_seconds) if feed else None

try:
    raw_history = (
        get_stock_minute_history(asset_label, api_key, secret_key)
        if is_stock and api_key and secret_key
        else get_minute_history(pair) if not is_stock
        else []
    )
    history_error = None
except Exception as exc:
    raw_history = []
    history_error = str(exc)

initial_snapshot = feed.snapshot() if feed else {"candles": [], "status": "credentials required"}
second_candles = initial_snapshot.get("candles", [])
minute_candles = merge_history_with_live_trades(raw_history, second_candles) if raw_history else []
# Do not train on an actively forming last minute candle.
training_rows = minute_candles[:-1] if len(minute_candles) > 1 else minute_candles
fitted = get_historical_model(training_rows, horizon_minutes=1) if training_rows else {"ready": False, "bars": 0}
minute_forecast = make_minute_forecast(minute_candles, fitted) if minute_candles and fitted.get("ready") else None


@st.fragment(run_every="1s")
def live_summary():
    """Refresh feed status and withhold one-second directions without evidence."""
    snapshot = feed.snapshot() if feed else {"candles": [], "status": "credentials required"}
    rows = snapshot.get("candles", [])
    last_trade_at = snapshot.get("last_trade_at")
    age = (datetime.now(timezone.utc) - last_trade_at).total_seconds() if last_trade_at else None
    live = snapshot.get("status") == "connected" and age is not None and age <= MAX_STALE_SECONDS

    live_check = evaluator.update(rows) if evaluator and rows else evaluator.snapshot(None) if evaluator else None
    candidate = live_check["current"] if live_check and live else None
    pattern_score = live_check["pattern"] if live_check else {"samples": 0}
    # The second-level candidate is not historically backtested. Only a
    # confirmed chart-pattern method with a large, conservative live sample
    # may appear as a watch; everything else remains WAIT.
    if not live:
        signal = "WAIT · feed not current"
    elif not candidate or not candidate.get("direction"):
        signal = "WAIT · no confirmed pattern break"
    elif candidate.get("method") != "pattern" or not pattern_score.get("validated"):
        signal = "WAIT · one-second rule not validated"
    else:
        signal = "BUY WATCH · live sample passed" if candidate["direction"] > 0 else "SELL WATCH · live sample passed"

    if live:
        status = f"🟢 **{provider_label} live** · {asset_label} · trade {last_trade_at:%H:%M:%S UTC}"
    elif snapshot.get("status") in {"connecting", "subscribing", "authenticating", "reconnecting"}:
        status = f"🟡 **{provider_label} {snapshot.get('status')}** · {asset_label}"
    elif snapshot.get("status") == "connected" and age is None:
        status = f"🟡 **{provider_label} connected** · waiting for the first {asset_label} trade"
    elif age is not None and age > MAX_STALE_SECONDS:
        status = f"🟡 **No recent {provider_label} trade** · last trade {age:.0f}s ago"
    elif is_stock and not feed:
        status = "🔒 **Stock feed needs Alpaca keys** · TradingView chart still works"
    else:
        status = f"🔴 **{provider_label} feed {snapshot.get('status', 'unavailable')}** · {asset_label}"
    st.markdown(status)

    last_price = rows[-1]["close"] if rows else None
    all_score = live_check["all"] if live_check else {"samples": 0}
    metrics = st.columns(4)
    metrics[0].metric("Latest trade", _price_label(last_price) if last_price is not None else "Waiting",
                      help="Price from the most recent actual trade received by the selected exchange feed.")
    metrics[1].metric("1-second signal", signal,
                      help="WAIT is intentional: minute-bar tests do not prove a one-second method works. A directional watch requires a confirmed pattern and strong completed live outcomes; it is still experimental.")
    metrics[2].metric("Pattern label", candidate["pattern"] if candidate else "No live reading",
                      help="A pattern name is only descriptive while forming. It does not imply direction unless a boundary break is confirmed.")
    if pattern_score.get("samples"):
        metrics[3].metric("Pattern outcomes", f"{pattern_score['hit_rate_pct']:.1f}% · {pattern_score['samples']} tests",
                          help="This is a temporary live check since the app process started, not a historical backtest. It resets when Streamlit restarts and ignores fees, spread, and slippage.")
    else:
        metrics[3].metric("Pattern outcomes", "No completed tests",
                          help="The app has not yet collected enough later trades to score a confirmed pattern rule.")
    st.caption(
        "Right chart contains real trade-based one-second candles only. The old glowing forecast path was removed because it was an untested extrapolation. "
        "Second-level history is not available to this app yet, so no forecast path or buy/sell claim is shown."
    )


live_summary()

_help_heading(
    "What the minute-history test found",
    "Four simple rules are compared using earlier data, the best one is selected without looking at the test period, and its performance is reported on later candles that were kept aside. These results cover one-minute bars only; they do not validate a one-second setup.",
)
if is_stock and not (api_key and secret_key):
    st.caption("Add Alpaca credentials in Streamlit Cloud Secrets to run this test on Apple or Tesla history. The TradingView chart works without them.")
elif history_error:
    st.warning(f"Could not load {provider_label} minute history: {history_error}")
elif not fitted.get("ready"):
    st.caption(f"Waiting for enough completed one-minute candles ({fitted.get('bars', 0)} loaded; {fitted.get('minimum_bars', 120)} needed).")
else:
    holdout = fitted["holdout"]
    accuracy = holdout.get("directional_accuracy_pct")
    calls = holdout.get("calls", 0)
    model_mae = holdout.get("forecast_mae_pct")
    baseline_mae = holdout.get("last_close_mae_on_calls_pct")
    mae_label = (
        f"{model_mae:.4f}% vs {baseline_mae:.4f}%"
        if model_mae is not None and baseline_mae is not None else "Not enough calls"
    )
    if minute_forecast and minute_forecast["signal_validated"] and minute_forecast["direction"]:
        minute_signal = "BUY WATCH" if minute_forecast["direction"] > 0 else "SELL WATCH"
    else:
        minute_signal = "WAIT"
    test_metrics = st.columns(4)
    test_metrics[0].metric("Chosen rule", fitted["model_label"])
    test_metrics[1].metric("Later test", f"{accuracy:.1f}%" if accuracy is not None else "No calls",
                          help=f"Direction matches across {calls} calls on later candles kept out of selection.")
    test_metrics[2].metric("Close MAE vs unchanged", mae_label,
                          help="MAE is average forecast error. Smaller is better; the second number is the error from simply assuming no price change.")
    test_metrics[3].metric("1-minute result", minute_signal,
                          help="WATCH appears only if there are at least 30 later test calls, at least 55% direction matches, and close error beats the unchanged-price baseline.")
    if minute_forecast:
        scenario = {1: "up", -1: "down", 0: "no passing direction"}[minute_forecast["scenario_direction"]]
        if minute_forecast["signal_validated"] and minute_forecast["direction"]:
            st.caption(f"The one-minute rule passed this recent holdout screen and points {scenario} for the next minute. This is separate from the unvalidated one-second feed.")
        elif minute_forecast["scenario_direction"]:
            st.caption(f"An unvalidated minute candidate points {scenario}, but it did not pass the holdout screen, so the result is WAIT. This does not validate a one-second trade.")
        else:
            st.caption("The minute rule has no directional candidate right now, so the result is WAIT.")
    with st.expander("How this test works", expanded=False):
        st.write(f"The app loaded {fitted['bars']} one-minute candles. It used the first {fitted['train_bars']} candles to choose a method and estimate its expected move, then measured it on the next {fitted['holdout_bars']} candles. The displayed rule is {fitted['model_label']}.")
        accuracy_text = f"{accuracy:.2f}%" if accuracy is not None else "not available"
        model_error_text = f"{model_mae:.5f}%" if model_mae is not None else "not available"
        baseline_error_text = f"{baseline_mae:.5f}%" if baseline_mae is not None else "not available"
        momentum_error = holdout.get("momentum_mae_pct")
        momentum_error_text = f"{momentum_error:.5f}%" if momentum_error is not None else "not available"
        st.write(f"On that held-back period, the rule made {calls} directional calls across {holdout.get('origins', 0)} possible test points. It matched {accuracy_text} of directions. Close MAE was {model_error_text}; the unchanged-price baseline was {baseline_error_text}; momentum baseline MAE was {momentum_error_text}.")
        st.write("This is a single recent chronological holdout, not a guarantee. The test does not include fees, spread, slippage, or real order fills.")

left, right = st.columns(2, gap="small")
with left:
    _help_heading(
        f"TradingView · {asset_label} · 1-minute chart",
        "This is TradingView's hosted chart, with its normal timeframe, zoom, indicators, and drawing tools. The market selector above changes the ticker. The hosted widget cannot receive a custom forecast overlay from this app, so the model result is explained below the chart.",
    )
    _tradingview_chart(asset["tv"], height=500)
    st.caption("TradingView chart and exchange feed are provided by TradingView. A custom model drawing cannot be injected into this hosted widget.")

with right:
    _help_heading(
        f"Live {provider_label} · actual 1-second candles",
        "Each bar groups real received trades by their exchange timestamp second. Seconds with no trade are omitted. The plot updates every second while the feed is active. No invented forecast bars are drawn.",
    )

    @st.fragment(run_every="1s")
    def live_second_chart():
        if feed is None:
            st.info("To activate AAPL/TSLA one-second candles, add ALPACA_API_KEY and ALPACA_SECRET_KEY as Streamlit Cloud Secrets. Do not put those values in GitHub or send them in chat.")
            return
        snapshot = feed.snapshot()
        rows = snapshot.get("candles", [])
        if rows:
            st.plotly_chart(
                _second_figure(rows, name=f"second-candles-{pair}", asset_label=provider_label, height=480),
                use_container_width=True,
                config={"scrollZoom": True, "displaylogo": False},
                key=f"second-candles-{pair}",
            )
            st.caption(f"Showing the latest {min(120, len(rows))} seconds that had at least one actual trade. This feed keeps a short rolling window in memory; it does not load old one-second trade history.")
        elif snapshot.get("status") == "unavailable":
            st.error(f"{provider_label} feed unavailable: {snapshot.get('detail', 'Check the data provider settings.')}")
        else:
            st.info(f"Waiting for actual {asset_label} trades. Stock trades arrive only during eligible market sessions; crypto trades arrive continuously.")

    live_second_chart()
