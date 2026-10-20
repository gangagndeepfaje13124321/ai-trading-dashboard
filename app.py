"""Compact live dashboard with one-minute Kraken candles and a broad watch scenario."""

from datetime import timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_history import fetch_kraken_minute_history, merge_history_with_live_trades
from minute_forecast import make_minute_forecast, walk_forward_direction_test
from trade_candles import KrakenTradeFeed


MARKETS = {
    "Bitcoin (BTC/USD)": "BTC/USD",
    "Ethereum (ETH/USD)": "ETH/USD",
    "Solana (SOL/USD)": "SOL/USD",
    "BNB (BNB/USD)": "BNB/USD",
    "XRP (XRP/USD)": "XRP/USD",
    "Dogecoin (DOGE/USD)": "DOGE/USD",
}
HORIZONS_MINUTES = (1, 3, 5, 10)

st.set_page_config(
    page_title="Live Crypto Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
      section.main > div { padding-top: .45rem; padding-bottom: .2rem; }
      div[data-testid="stPlotlyChart"] { margin-bottom: -.35rem; }
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


@st.cache_data(ttl=60, show_spinner=False)
def get_kraken_minute_history(pair):
    """Refresh REST history once a minute and keep the last error visible."""
    try:
        return {"candles": fetch_kraken_minute_history(pair), "error": None}
    except Exception as exc:
        return {"candles": [], "error": f"Kraken minute history unavailable: {type(exc).__name__}: {str(exc)[:160]}"}


@st.cache_data(ttl=60, show_spinner=False)
def get_walk_forward_test(candles, horizon_minutes):
    """Cache the historical directional check until fresh REST data arrives."""
    return walk_forward_direction_test(candles, horizon_minutes=horizon_minutes)


def _direction_label(direction):
    return {1: "Upward bias", -1: "Downward bias", 0: "Mixed / no clear bias"}[direction]


def _price_label(price):
    return f"${price:,.2f}" if abs(price) >= 100 else f"${price:,.5f}"


def _minute_figure(candles, name, height=370):
    fig = go.Figure()
    if candles:
        frame = pd.DataFrame(candles[-60:])
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        fig.add_trace(go.Candlestick(
            x=frame["time"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Kraken · 1m OHLC",
            increasing_line_color="#20b89a",
            decreasing_line_color="#ff5b64",
            increasing_fillcolor="rgba(32,184,154,.55)",
            decreasing_fillcolor="rgba(255,91,100,.55)",
        ))
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis_title=None,
        yaxis_title="USD",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        uirevision=name,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, tickformat="%H:%M\n%b %d")
    fig.update_yaxes(fixedrange=False, tickformat=".6~g", gridcolor="rgba(150,160,180,.18)")
    return fig


def _scenario_figure(candles, forecast, name, height=370):
    fig = go.Figure()
    if candles:
        recent = pd.DataFrame(candles[-25:])
        recent["time"] = pd.to_datetime(recent["time"], utc=True)
        fig.add_trace(go.Scatter(
            x=recent["time"] + pd.Timedelta(minutes=1),
            y=recent["close"],
            mode="lines",
            name="Recent minute closes",
            line=dict(color="#8ea1b8", width=1.5),
        ))

    if forecast:
        start = forecast["time"] + timedelta(minutes=1)
        horizon = forecast["horizon_minutes"]
        x_values, upper, lower = [], [], []
        for step in range(horizon + 1):
            progress = step / horizon
            center = forecast["price"] + (
                forecast["direction"] * forecast["range_radius"] * 0.12 * progress
            )
            radius = forecast["range_radius"] * progress
            x_values.append(start + timedelta(minutes=step))
            upper.append(center + radius)
            lower.append(center - radius)
        shade = (
            "rgba(32,184,154,.22)" if forecast["direction"] > 0 else
            "rgba(255,91,100,.22)" if forecast["direction"] < 0 else
            "rgba(142,161,184,.22)"
        )
        fig.add_trace(go.Scatter(
            x=x_values, y=upper, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x_values, y=lower, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=shade, name="Scenario range",
            hovertemplate="Range boundary: %{y:$,.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[start], y=[forecast["price"]], mode="markers",
            marker=dict(color="#ffffff", size=8, line=dict(color="#242a35", width=1)),
            name="Current price",
            hovertemplate="Current price: %{y:$,.2f}<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis_title=None,
        yaxis_title="USD",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.03, x=0),
        uirevision=name,
    )
    fig.update_xaxes(showgrid=False, tickformat="%H:%M\n%b %d")
    fig.update_yaxes(fixedrange=False, tickformat=".6~g", gridcolor="rgba(150,160,180,.18)")
    return fig


title_col, market_col, horizon_col = st.columns([2.2, 1.2, 0.9])
with title_col:
    st.title("⚡ Live Crypto Dashboard")
with market_col:
    selected_market = st.selectbox("Kraken market", list(MARKETS), label_visibility="collapsed")
with horizon_col:
    horizon_minutes = st.selectbox(
        "Forecast horizon",
        HORIZONS_MINUTES,
        index=1,
        format_func=lambda minutes: f"Next {minutes} minute" + ("s" if minutes != 1 else ""),
        label_visibility="collapsed",
    )

pair = MARKETS[selected_market]
feed = get_kraken_trade_feed(pair)
st.caption(
    "The chart uses one-minute candles. The current candle refreshes every second from Kraken trades; "
    "a candle closes at the end of its minute."
)


@st.fragment(run_every="1s")
def live_dashboard():
    snapshot = feed.snapshot()
    history_result = get_kraken_minute_history(pair)
    history = history_result["candles"]
    second_candles = snapshot.get("candles", [])
    candles = merge_history_with_live_trades(history, second_candles)
    forecast = make_minute_forecast(candles, horizon_minutes=horizon_minutes)

    last_trade_at = snapshot.get("last_trade_at")
    if snapshot.get("status") == "connected":
        status_text = f"🟢 Kraken trade feed connected · {pair}"
    elif snapshot.get("status") in {"reconnecting", "connecting", "subscribing"}:
        status_text = f"🟡 Kraken {snapshot.get('status')} · {pair}"
    else:
        status_text = f"🔴 Kraken {snapshot.get('status', 'unavailable')} · {pair}"
    if last_trade_at:
        status_text += f" · last exchange trade {last_trade_at.astimezone(timezone.utc):%H:%M:%S} UTC"
    st.markdown(status_text)

    if history_result["error"] and not history:
        st.warning(history_result["error"] + " The candle chart will show only trades received since this app started.")
    elif history_result["error"]:
        st.caption(history_result["error"] + " Showing the most recently cached candle history plus live trades.")

    price = candles[-1]["close"] if candles else None
    test_rows = history[:-1] if len(history) > 1 else []  # Exclude Kraken's unfinished current REST candle.
    test = get_walk_forward_test(test_rows, horizon_minutes) if len(test_rows) >= 90 else None

    metric_cols = st.columns(4)
    metric_cols[0].metric("Latest trade price", _price_label(price) if price is not None else "—")
    metric_cols[1].metric(
        f"Next {horizon_minutes} min outlook",
        forecast["bias"] if forecast else "Waiting for candles",
    )
    metric_cols[2].metric(
        "Signal agreement",
        f"{max(forecast['up_votes'], forecast['down_votes'])}/{forecast['active_votes']} active votes"
        if forecast and forecast["active_votes"] else "No directional votes",
        help="Agreement among the indicators that currently give a direction. It is not a probability.",
    )
    if test and test["calls"]:
        metric_cols[3].metric(
            "Historical direction check",
            f"{test['hit_rate_pct']:.0f}% · {test['calls']} calls",
            help=(
                f"Walk-forward on recent Kraken one-minute candles: {test['wins']}/{test['calls']} directional calls matched the later close. "
                f"The same timestamps' simple momentum baseline matched {test['baseline_wins']}/{test['calls']} calls. "
                "This is not a profitability test and excludes spread, fees, and slippage."
            ),
        )
    else:
        metric_cols[3].metric("Historical direction check", "Collecting data")

    left, right = st.columns(2, gap="small")
    with left:
        st.markdown("**BTC/USD · 1-minute candles**" if pair == "BTC/USD" else f"**{pair} · 1-minute candles**")
        st.plotly_chart(
            _minute_figure(candles, name=f"minute-candles-{pair}"),
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False},
            key=f"minute-candles-{pair}",
        )
    with right:
        st.markdown(f"**Next {horizon_minutes} minutes · scenario range**")
        st.plotly_chart(
            _scenario_figure(candles, forecast, name=f"minute-scenario-{pair}-{horizon_minutes}"),
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False},
            key=f"minute-scenario-{pair}-{horizon_minutes}",
        )

    if forecast:
        st.caption(
            f"Scenario range at the horizon: {_price_label(forecast['scenario_low'])}–{_price_label(forecast['scenario_high'])}. "
            "The shaded area shows a broad range based on recent volatility; it is not a calibrated probability interval or exact price target."
        )

    with st.expander("How to read the outlook and its test"):
        st.write(
            "This is a fixed-rule signal blend, not a trained AI model. It checks EMA trend, 3- and 10-minute momentum, "
            "RSI reversal extremes, breaks of the recent 20-minute high/low, price versus volume-weighted average price, "
            "and confirmed classic-pattern breaks. The displayed direction requires at least two active methods to agree. "
            "These indicators can overlap and agreement is not a probability."
        )
        st.write(
            "The historical check replays the same fixed rules through recent Kraken one-minute candles in time order. "
            "Each prediction sees only earlier candles, is checked against the later close, and is spaced by the selected "
            "horizon so test targets do not overlap. It compares directional calls with a simple continuation baseline "
            "based on the prior move of the same length. Kraken supplies about 12 hours of recent one-minute candles."
        )
        if test:
            baseline_text = (
                f"The current check has {test['origins']} forecast origins and {test['calls']} non-neutral directional calls. "
                f"On those same calls, the ensemble matched {test['wins']}/{test['calls']} future closes "
                f"({test['hit_rate_pct']:.1f}%), while the continuation baseline matched "
                f"{test['baseline_wins']}/{test['calls']} ({test['baseline_hit_rate_pct']:.1f}%)."
                if test["calls"] else
                f"The current check has {test['origins']} forecast origins but no directional calls to score yet."
            )
            st.write(baseline_text)
        st.write(
            "This small recent-history check is not proof of a durable edge or profitable trades. It does not model "
            "spread, fees, slippage, order-book changes, or execution. The shaded range is observed volatility, not a "
            "probability forecast. No orders are placed."
        )
        if forecast:
            st.markdown("**Current inputs**")
            for signal in forecast["signals"]:
                direction = {1: "up", -1: "down", 0: "neutral"}[signal["direction"]]
                st.write(f"- **{signal['name']} — {direction}:** {signal['detail']}")


live_dashboard()
