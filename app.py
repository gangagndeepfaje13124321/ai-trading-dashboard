"""Compact dashboard: TradingView one-minute chart plus Kraken second candles."""

from datetime import datetime, timezone
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from live_forecast import LiveForecastEvaluator, make_live_forecast
from trade_candles import KrakenTradeFeed


MARKETS = {
    "Bitcoin (BTC/USD)": ("BTC/USD", "KRAKEN:BTCUSD"),
    "Ethereum (ETH/USD)": ("ETH/USD", "KRAKEN:ETHUSD"),
    "Solana (SOL/USD)": ("SOL/USD", "KRAKEN:SOLUSD"),
    "BNB (BNB/USD)": ("BNB/USD", "KRAKEN:BNBUSD"),
    "XRP (XRP/USD)": ("XRP/USD", "KRAKEN:XRPUSD"),
    "Dogecoin (DOGE/USD)": ("DOGE/USD", "KRAKEN:DOGEUSD"),
}
HORIZONS_SECONDS = (3, 5, 10)
MAX_STALE_SECONDS = 10

st.set_page_config(
    page_title="Live Crypto Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
      section.main > div { padding-top: .35rem; padding-bottom: .2rem; }
      div[data-testid="stPlotlyChart"] { margin-bottom: -.3rem; }
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
def get_second_evaluator(pair, horizon_seconds):
    """Keep a short, in-memory, live check for this market and horizon."""
    return LiveForecastEvaluator(horizon_seconds=horizon_seconds)


def _price_label(price):
    return f"${price:,.2f}" if abs(price) >= 100 else f"${price:,.5f}"


def _method_label(method):
    return {
        "pattern": "Confirmed chart-pattern break",
        "breakout": "Recent price-range break",
        "momentum": "Recent price movement",
        "none": "No rule gave a direction",
    }.get(method, "No rule gave a direction")


def _watch_label(forecast):
    if not forecast or forecast["direction"] == 0:
        return "WAIT · no clear setup"
    if forecast["direction"] > 0:
        return "BUY-SIDE WATCH · upward bias"
    return "SELL-SIDE WATCH · downward bias"


def _help_heading(label, explanation):
    """Show a visible question-mark button beside a section heading."""
    heading, help_button = st.columns([0.91, 0.09], vertical_alignment="center")
    with heading:
        st.markdown(f"**{label}**")
    with help_button:
        with st.popover("?", help=explanation):
            st.write(explanation)


def _tradingview_embed(symbol, height=500):
    """Embed TradingView's hosted chart and its user drawing toolbar."""
    config = {
        "autosize": True,
        "symbol": symbol,
        "interval": "1",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "withdateranges": True,
        "hide_side_toolbar": False,
        "hide_top_toolbar": False,
        "hide_legend": False,
        "hide_volume": False,
        "allow_symbol_change": False,
        "save_image": False,
        "calendar": False,
        "support_host": "https://www.tradingview.com",
    }
    html = f"""
    <div class="tradingview-widget-container" style="height:{height}px;width:100%">
      <div class="tradingview-widget-container__widget" style="height:calc(100% - 28px);width:100%"></div>
      <div class="tradingview-widget-copyright" style="height:24px;font:12px Arial,sans-serif;color:#9aa4b2">
        <a href="https://www.tradingview.com/" target="_blank" rel="noopener nofollow" style="color:#8ab4f8">Chart by TradingView</a>
      </div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
        {json.dumps(config)}
      </script>
    </div>
    """
    components.html(html, height=height + 4, scrolling=False)


def _second_figure(candles, forecast, name, height=500):
    """Plot observed trade-second OHLC bars and forecast scenario OHLC bars."""
    fig = go.Figure()
    recent = candles[-60:]
    if recent:
        frame = pd.DataFrame(recent)
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        fig.add_trace(go.Candlestick(
            x=frame["time"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Kraken trades · actual 1s candles",
            increasing_line_color="#20b89a",
            decreasing_line_color="#ff5b64",
            increasing_fillcolor="rgba(32,184,154,.62)",
            decreasing_fillcolor="rgba(255,91,100,.62)",
        ))

    future = forecast.get("forecast_candles", []) if forecast else []
    if future:
        frame = pd.DataFrame(future)
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        up_line, down_line = ("#75a7ff", "#ffb454")
        fig.add_trace(go.Candlestick(
            x=frame["time"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Forecast scenario · 1s OHLC",
            increasing_line_color=up_line,
            decreasing_line_color=down_line,
            increasing_fillcolor="rgba(117,167,255,.62)",
            decreasing_fillcolor="rgba(255,180,84,.62)",
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
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
        showlegend=True,
    )
    fig.update_xaxes(showgrid=False, tickformat="%H:%M:%S\n%b %d")
    fig.update_yaxes(fixedrange=False, tickformat=".6~g", gridcolor="rgba(150,160,180,.18)")
    return fig


title_col, market_col, horizon_col = st.columns([2.0, 1.2, 0.9], vertical_alignment="center")
with title_col:
    st.title("⚡ Live Crypto Dashboard")
with market_col:
    picker, picker_help = st.columns([0.91, 0.09], vertical_alignment="center")
    with picker:
        selected_market = st.selectbox("Kraken market", list(MARKETS), label_visibility="collapsed")
    with picker_help:
        with st.popover("?", help="Choose a market"):
            st.write("This selects which Kraken exchange market supplies the live one-second candles and short forecast.")
with horizon_col:
    picker, picker_help = st.columns([0.88, 0.12], vertical_alignment="center")
    with picker:
        horizon_seconds = st.selectbox(
            "Forecast length",
            HORIZONS_SECONDS,
            index=1,
            format_func=lambda seconds: f"Next {seconds} seconds",
            label_visibility="collapsed",
        )
    with picker_help:
        with st.popover("?", help="Choose the scenario length"):
            st.write("This is how many hypothetical one-second candles the right chart shows. It is not a promise that price will reach any level.")

pair, tradingview_symbol = MARKETS[selected_market]
feed = get_kraken_trade_feed(pair)
evaluator = get_second_evaluator(pair, horizon_seconds)

@st.fragment(run_every="1s")
def live_summary():
    """Update feed status and simple forecast/evaluation labels each second."""
    snapshot = feed.snapshot()
    second_candles = snapshot.get("candles", [])
    last_trade_at = snapshot.get("last_trade_at")
    now = datetime.now(timezone.utc)
    age_seconds = (now - last_trade_at).total_seconds() if last_trade_at else None
    feed_is_live = (
        snapshot.get("status") == "connected"
        and age_seconds is not None
        and age_seconds <= MAX_STALE_SECONDS
    )

    live_check = evaluator.snapshot(None)
    if second_candles:
        live_check = evaluator.update(second_candles)
    forecast = live_check["current"] if feed_is_live else None

    if feed_is_live:
        status_text = f"🟢 **Kraken live** · {pair} · last trade {last_trade_at:%H:%M:%S UTC}"
    elif snapshot.get("status") in {"connecting", "subscribing", "reconnecting"}:
        status_text = f"🟡 **Kraken {snapshot.get('status')}** · {pair}"
    elif age_seconds is not None and age_seconds > MAX_STALE_SECONDS:
        status_text = f"🟡 **No recent Kraken trade** · last trade {age_seconds:.0f}s ago"
    else:
        status_text = f"🔴 **Kraken feed {snapshot.get('status', 'unavailable')}** · {pair}"
    status_col, status_help_col = st.columns([0.95, 0.05], vertical_alignment="center")
    with status_col:
        st.markdown(status_text)
    with status_help_col:
        with st.popover("?", help="What the connection light means"):
            st.write(
                "Green means Kraken is connected and a trade arrived within the last 10 seconds. "
                "Yellow means it is reconnecting or there has not been a recent trade. The app hides the setup label when the feed is stale."
            )

    recent_candles = second_candles[-60:]
    price = recent_candles[-1]["close"] if recent_candles else None
    score = live_check["all"]
    metrics = st.columns(4)
    metrics[0].metric(
        "Latest Kraken trade",
        _price_label(price) if price is not None else "Waiting",
        help="Most recent price recorded in an actual Kraken trade. It is not a buy or sell price guarantee.",
    )
    metrics[1].metric(
        "Potential setup",
        _watch_label(forecast) if feed_is_live else "WAIT · feed not current",
        help=(
            "BUY-SIDE WATCH or SELL-SIDE WATCH means one short-term rule points that way. WAIT means no rule has a clear direction. "
            "This describes the chart; it is not an instruction to place an order or evidence of profitability."
        ),
    )
    metrics[2].metric(
        "Rule used",
        _method_label(forecast["method"]) if forecast else "No fresh reading",
        help=(
            "A confirmed pattern/range break gets priority. Otherwise the app checks whether recent one-second prices moved clearly "
            "over the selected horizon. It does not combine these into a probability or a trained AI prediction."
        ),
    )
    if score["samples"]:
        metrics[3].metric(
            f"Live check · {score['samples']} results",
            f"{score['hit_rate_pct']:.0f}% matched",
            help=(
                f"Since this app process started, {score['wins']} of {score['samples']} completed non-neutral scenarios matched the direction "
                f"of the first later trade candle at about {horizon_seconds} seconds. Direction-only; it excludes spread, fees, slippage, "
                "and execution. This in-memory sample resets when the app restarts and is not proof of a lasting edge."
            ),
        )
    else:
        metrics[3].metric(
            "Live direction check",
            "Waiting for results",
            help=(
                "The app needs to issue a directional scenario, wait for its future time, and compare it with later Kraken trades. "
                "No result means there is not enough live evidence yet."
            ),
        )

    why_col, why_help_col = st.columns([0.95, 0.05], vertical_alignment="center")
    with why_col:
        if forecast:
            st.caption(f"{_watch_label(forecast)} · {_method_label(forecast['method'])}. {forecast['reason']}")
        else:
            st.caption("No fresh directional setup is being shown while the trade feed is connecting or stale.")
    with why_help_col:
        with st.popover("?", help="Plain-English explanation of the displayed setup"):
            st.write(
                "The app looks for recognizable short price shapes such as a triangle, double top/bottom, flag, or wedge. "
                "A shape only points up or down after price breaks its boundary. If there is no confirmed shape break, it checks recent price movement. "
                "A forming shape by itself does not count as a buy or sell direction."
            )

    with st.expander("What does the live check mean?", expanded=False):
        st.write(
            "This is a small, real-time check, not a backtest of all market conditions. Each directional scenario waits for the selected number of seconds, "
            "then gets scored against the first later Kraken trade candle. A match means only that the later close was in the predicted direction from the starting price. "
            "It does not measure whether a trade could have been entered or exited profitably."
        )
        if score["samples"]:
            st.write(
                f"Current in-memory sample: {score['wins']} of {score['samples']} directions matched ({score['hit_rate_pct']:.1f}%). "
                f"Average absolute close error was {score['forecast_close_mae_pct']:.5f}% for the scenario endpoint, "
                f"versus {score['last_close_mae_pct']:.5f}% for simply keeping the last price."
            )
        else:
            st.write("No completed scenario has been scored yet. The count starts when the app receives live trades.")
        st.write(
            "The Kraken WebSocket provides genuine trade timestamps, grouped into one-second OHLC bars. Only seconds with trades appear. "
            "This process keeps up to 15 minutes of received candles in memory; it does not load historical one-second trades."
        )


live_summary()

left, right = st.columns(2, gap="small")
with left:
    _help_heading(
        f"{pair} · 1-minute TradingView chart",
        "TradingView supplies this one-minute chart and its drawing toolbar. Its data provider may differ slightly from the Kraken trade prices on the right. "
        "The hosted TradingView widget does not accept this app's custom forecast candles, so the forecast appears on the Kraken chart instead.",
    )
    # Keep this widget outside the one-second fragment so it does not reload
    # while the user is drawing on the chart.
    _tradingview_embed(tradingview_symbol, height=440)

with right:
    @st.fragment(run_every="1s")
    def live_second_chart():
        snapshot = feed.snapshot()
        second_candles = snapshot.get("candles", [])
        last_trade_at = snapshot.get("last_trade_at")
        age_seconds = (datetime.now(timezone.utc) - last_trade_at).total_seconds() if last_trade_at else None
        feed_is_live = (
            snapshot.get("status") == "connected"
            and age_seconds is not None
            and age_seconds <= MAX_STALE_SECONDS
        )
        forecast = make_live_forecast(second_candles, horizon_seconds) if feed_is_live else None

        _help_heading(
            f"{horizon_seconds}-second forecast and live candles",
            "Green/red candles are actual seconds that contained Kraken trades; a gap means no trade candle was received for that second. "
            "Blue/orange candles are hypothetical OHLC bars for the next few seconds. They update each second while the feed is fresh, "
            "but are scenarios, not real future prices.",
        )
        st.plotly_chart(
            _second_figure(second_candles, forecast, name=f"second-candles-{pair}-{horizon_seconds}", height=440),
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False},
            key=f"second-candles-{pair}-{horizon_seconds}",
        )
        if forecast:
            st.caption(f"{_watch_label(forecast)} · {_method_label(forecast['method'])}.")
        else:
            st.caption("Waiting for fresh Kraken trades; forecast candles are hidden until the feed is current.")

    live_second_chart()
