"""Compact live Kraken trade-candle dashboard."""

from datetime import timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from live_forecast import LiveForecastEvaluator
from trade_candles import KrakenTradeFeed


MARKETS = {
    "Bitcoin (BTC/USD)": "BTC/USD",
    "Ethereum (ETH/USD)": "ETH/USD",
    "Solana (SOL/USD)": "SOL/USD",
    "BNB (BNB/USD)": "BNB/USD",
    "XRP (XRP/USD)": "XRP/USD",
    "Dogecoin (DOGE/USD)": "DOGE/USD",
}
HORIZONS = (3, 5, 10)

st.set_page_config(
    page_title="Live Crypto Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
      section.main > div { padding-top: .55rem; padding-bottom: .25rem; }
      div[data-testid="stPlotlyChart"] { margin-bottom: -.3rem; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_kraken_trade_feed(pair):
    """Share the existing reconnecting Kraken feed across app reruns."""
    feed = KrakenTradeFeed(pair, max_candles=900)
    feed.start()
    return feed


@st.cache_resource(show_spinner=False)
def get_live_evaluator(pair, horizon_seconds):
    """Keep online scoring alive across Streamlit reruns for this market."""
    return LiveForecastEvaluator(horizon_seconds=horizon_seconds)


def _direction_label(direction):
    return {1: "Upward watch", -1: "Downward watch", 0: "No clear direction"}[direction]


def _price_label(price):
    return f"${price:,.2f}" if abs(price) >= 100 else f"${price:,.5f}"


def _candlestick_figure(candles, forecast_candles=None, name="", height=410):
    fig = go.Figure()
    if candles:
        actual = candles[-90:]
        frame = pd.DataFrame(actual)
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        fig.add_trace(go.Candlestick(
            x=frame["time"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Kraken trades · 1s",
            increasing_line_color="#20b89a",
            decreasing_line_color="#ff5b64",
            increasing_fillcolor="rgba(32,184,154,.48)",
            decreasing_fillcolor="rgba(255,91,100,.48)",
        ))
    if forecast_candles:
        forecast = pd.DataFrame(forecast_candles)
        forecast["time"] = pd.to_datetime(forecast["time"], utc=True)
        fig.add_trace(go.Candlestick(
            x=forecast["time"],
            open=forecast["open"],
            high=forecast["high"],
            low=forecast["low"],
            close=forecast["close"],
            name="Experimental forecast OHLC",
            increasing_line_color="#4c9dff",
            decreasing_line_color="#ff9f43",
            increasing_fillcolor="rgba(76,157,255,.5)",
            decreasing_fillcolor="rgba(255,159,67,.5)",
        ))
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=12, r=12, t=8, b=8),
        xaxis_title=None,
        yaxis_title="USD",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        uirevision=name,
        showlegend=True,
    )
    fig.update_xaxes(showgrid=False, tickformat="%H:%M:%S\n%b %d", rangeslider_visible=False)
    fig.update_yaxes(fixedrange=False, tickformat=".6~g", gridcolor="rgba(150,160,180,.18)")
    return fig


header, market_control, horizon_control = st.columns([2.2, 1.2, 0.8])
with header:
    st.title("⚡ Live Crypto Dashboard")
with market_control:
    selected_market = st.selectbox("Kraken market", list(MARKETS), label_visibility="collapsed")
with horizon_control:
    horizon_seconds = st.selectbox(
        "Forecast horizon in seconds", HORIZONS, index=1,
        format_func=lambda seconds: f"{seconds}-second view",
        label_visibility="collapsed",
    )

pair = MARKETS[selected_market]
feed = get_kraken_trade_feed(pair)
evaluator = get_live_evaluator(pair, horizon_seconds)

st.caption(
    "Two live views of the same Kraken trade feed. The forecast is an experimental, watch-only scenario; "
    "it is not a buy/sell instruction and does not place orders."
)


@st.fragment(run_every="1s")
def live_dashboard():
    snapshot = feed.snapshot()
    candles = snapshot.get("candles", [])
    evaluation = evaluator.update(candles)
    current = evaluation["current"]
    last_trade_at = snapshot.get("last_trade_at")

    if snapshot.get("status") == "connected":
        status_text = f"🟢 Kraken connected · {pair}"
    elif snapshot.get("status") in {"reconnecting", "connecting", "subscribing"}:
        status_text = f"🟡 Kraken {snapshot.get('status')} · {pair}"
    else:
        status_text = f"🔴 Kraken {snapshot.get('status', 'unavailable')} · {pair}"
    if last_trade_at:
        status_text += f" · last exchange trade {last_trade_at.astimezone(timezone.utc):%H:%M:%S} UTC"
    st.markdown(status_text)

    if current:
        bias = _direction_label(current["direction"])
        setup = current["pattern"]
        price = current["entry_price"]
    elif candles:
        bias = "Waiting for another trade"
        setup = "Not enough recent trade candles yet"
        price = candles[-1]["close"]
    else:
        bias = "Waiting for Kraken trades"
        setup = "No pattern to evaluate yet"
        price = None

    stats = evaluation["pattern"]
    score_label = (
        f"{stats['hit_rate_pct']:.1f}% · {stats['samples']} outcomes"
        if stats["hit_rate_pct"] is not None
        else "Collecting outcomes"
    )
    score_help = (
        f"95% Wilson lower bound: {stats['lower_bound_pct']:.1f}%. "
        "A pattern result is only labelled validated after at least 200 non-overlapping outcomes "
        "and a lower bound of 55%. "
        + (
            f"Close MAE: {stats['forecast_close_mae_pct']:.4f}% of entry versus "
            f"{stats['last_close_mae_pct']:.4f}% for the last-close baseline. "
            if stats["samples"] else "Close-error metrics appear after the first scored setup. "
        )
        + "Results reset when this app process restarts."
    )
    metrics = st.columns(4)
    metrics[0].metric("Latest trade price", _price_label(price) if price is not None else "—")
    metrics[1].metric(f"{horizon_seconds}s watch scenario", bias)
    metrics[2].metric("Current chart setup", setup)
    metrics[3].metric("Classic-pattern test", score_label, help=score_help)

    if stats["validated"]:
        st.success("Classic-pattern direction has passed the current in-session validation threshold. This still does not show profitability after trading costs.")
    elif stats["samples"]:
        st.caption(
            f"Classic-pattern direction so far: {stats['wins']}/{stats['samples']} correct "
            f"({stats['hit_rate_pct']:.1f}%); 95% lower bound {stats['lower_bound_pct']:.1f}%. "
            "This is too early to call validated."
        )
    else:
        st.caption("Pattern scoring starts when a confirmed setup appears and is checked against a later Kraken trade candle.")

    left, right = st.columns(2, gap="small")
    with left:
        st.markdown("**Live 1-second Kraken trade candles**")
        st.plotly_chart(
            _candlestick_figure(candles, name=f"live-{pair}", height=300),
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False},
            key=f"live-candles-{pair}",
        )
    with right:
        st.markdown(f"**Live {horizon_seconds}-second forecast · OHLC candles**")
        forecast = current["forecast_candles"] if current else None
        st.plotly_chart(
            _candlestick_figure(candles[-35:], forecast, name=f"forecast-{pair}-{horizon_seconds}", height=300),
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False},
            key=f"live-forecast-{pair}-{horizon_seconds}",
        )
    reason = current["reason"] if current else snapshot.get("detail", "Waiting for trade data.")
    st.caption(
        f"Forecast basis: {reason} Forecast candles reuse recent {horizon_seconds}-second movement sizes and wick ranges; "
        "they are scenarios, not calibrated probability bounds. This view redraws every second while open; "
        "the forecast changes when new trade data arrives."
    )


live_dashboard()
