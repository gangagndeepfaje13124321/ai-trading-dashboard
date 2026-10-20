"""Compact Kraken dashboard with minute-level validation and second bars."""

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from live_forecast import LiveForecastEvaluator, make_live_forecast
from market_history import fetch_kraken_minute_history, merge_history_with_live_trades
from strategy_lab import fit_and_evaluate, make_minute_forecast
from trade_candles import KrakenTradeFeed


MARKETS = {
    "Bitcoin (BTC/USD)": "BTC/USD",
    "Ethereum (ETH/USD)": "ETH/USD",
    "Solana (SOL/USD)": "SOL/USD",
    "BNB (BNB/USD)": "BNB/USD",
    "XRP (XRP/USD)": "XRP/USD",
    "Dogecoin (DOGE/USD)": "DOGE/USD",
}
HORIZONS_SECONDS = (10, 30, 60, 120)
HORIZONS_MINUTES = (1, 3, 5)
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
def get_second_evaluator(pair, horizon_seconds):
    """Keep a short, in-memory, live check for this market and horizon."""
    return LiveForecastEvaluator(horizon_seconds=horizon_seconds)


@st.cache_data(ttl=30, show_spinner=False)
def get_minute_history(pair):
    """Share Kraken's recent minute history and refresh it at most twice/minute."""
    return fetch_kraken_minute_history(pair)


@st.cache_data(ttl=300, show_spinner=False)
def get_historical_model(candles, horizon_minutes):
    """Cache historical rule selection until the source bars or horizon change."""
    return fit_and_evaluate(candles, horizon_minutes)


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


def _second_figure(candles, forecast, name, height=500):
    """Plot observed trade-second OHLC bars and forecast scenario OHLC bars."""
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


def _minute_figure(candles, forecast, name, height=500):
    """Show Kraken minute history and the historical model's future OHLC bars."""
    fig = go.Figure()
    if candles:
        frame = pd.DataFrame(candles)
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        fig.add_trace(go.Candlestick(
            x=frame["time"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Kraken · actual 1m candles",
            increasing_line_color="#24c7a2",
            decreasing_line_color="#ff6270",
            increasing_fillcolor="rgba(36,199,162,.65)",
            decreasing_fillcolor="rgba(255,98,112,.65)",
        ))

    future = forecast.get("forecast_candles", []) if forecast else []
    if future:
        frame = pd.DataFrame(future)
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        fig.add_trace(go.Candlestick(
            x=frame["time"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Historical model · scenario OHLC",
            increasing_line_color="#78aaff",
            decreasing_line_color="#ffb454",
            increasing_fillcolor="rgba(120,170,255,.72)",
            decreasing_fillcolor="rgba(255,180,84,.72)",
        ))

    latest = candles[-1]["time"] if candles else datetime.now(timezone.utc)
    horizon = len(future)
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        yaxis_title="USD",
        hovermode="x unified",
        uirevision=name,
        editrevision=name,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.07),
            range=[latest - pd.Timedelta(minutes=120), latest + pd.Timedelta(minutes=horizon + 1)],
            showgrid=False,
            tickformat="%H:%M\n%b %d",
        ),
        yaxis=dict(fixedrange=False, tickformat=".6~g", gridcolor="rgba(150,160,180,.18)"),
    )
    return fig


title_col, market_col, second_horizon_col, minute_horizon_col = st.columns(
    [1.75, 1.15, 1.05, 1.05], vertical_alignment="center"
)
with title_col:
    st.title("⚡ Live Crypto Dashboard")
with market_col:
    picker, picker_help = st.columns([0.91, 0.09], vertical_alignment="center")
    with picker:
        selected_market = st.selectbox("Kraken market", list(MARKETS), label_visibility="collapsed")
    with picker_help:
        with st.popover("?", help="Choose a market"):
            st.write("This selects which Kraken exchange market supplies the live one-second candles and short forecast.")
with second_horizon_col:
    picker, picker_help = st.columns([0.88, 0.12], vertical_alignment="center")
    with picker:
        horizon_seconds = st.selectbox(
            "Second forecast",
            HORIZONS_SECONDS,
            index=2,
            format_func=lambda seconds: f"Next {seconds} seconds",
            label_visibility="collapsed",
        )
    with picker_help:
        with st.popover("?", help="Choose the scenario length"):
            st.write("This is how many hypothetical one-second candles the right chart shows. Longer scenarios are more uncertain and are not actual future prices.")

with minute_horizon_col:
    picker, picker_help = st.columns([0.88, 0.12], vertical_alignment="center")
    with picker:
        horizon_minutes = st.selectbox(
            "Minute forecast",
            HORIZONS_MINUTES,
            index=0,
            format_func=lambda minutes: f"Next {minutes} minute" + ("s" if minutes != 1 else ""),
            label_visibility="collapsed",
        )
    with picker_help:
        with st.popover("?", help="Choose the historical model forecast length"):
            st.write("This changes the OHLC forecast on the left chart. Its rule is chosen using earlier Kraken minute candles and then checked on a later, separate block of candles.")

pair = MARKETS[selected_market]
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
        "Second-level setup",
        _watch_label(forecast) if feed_is_live else "WAIT · feed not current",
        help=(
            "BUY-SIDE WATCH or SELL-SIDE WATCH means one short-term rule points that way. WAIT means no rule has a clear direction. "
            "The second-level rule has no historical one-second backtest yet; it is not an instruction to place an order or evidence of profitability."
        ),
    )
    metrics[2].metric(
        "Rule used",
        _method_label(forecast["method"]) if forecast else "No fresh reading",
        help=(
            "A confirmed pattern/range break gets priority. Otherwise the app checks recent one-second prices for movement. "
            "This is a fixed rule, not a trained AI model, and it has not been tested on historical one-second candles."
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
            st.caption(f"EXPERIMENTAL · {_watch_label(forecast)} · {_method_label(forecast['method'])}. {forecast['reason']}")
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
    @st.fragment(run_every="1s")
    def minute_chart():
        """Refresh Kraken's current minute candle and its historical-model overlay."""
        try:
            history = get_minute_history(pair)
        except Exception:
            _help_heading(
                f"{pair} · 1-minute chart",
                "Kraken supplies both this historical minute chart and the live trade candles. A forecast is hidden if the source history could not be loaded.",
            )
            st.warning("Kraken minute history is temporarily unavailable. Check the app's Streamlit Cloud logs if this continues.")
            return

        # Kraken's final REST row is the still-forming minute. Keep it out of
        # training so the historical test only uses completed candles.
        training_rows = history[:-1] if len(history) > 1 else history
        fitted = get_historical_model(training_rows, horizon_minutes)
        snapshot = feed.snapshot()
        second_candles = snapshot.get("candles", [])
        minute_candles = merge_history_with_live_trades(history, second_candles)
        latest_trade_at = snapshot.get("last_trade_at")
        age = (datetime.now(timezone.utc) - latest_trade_at).total_seconds() if latest_trade_at else None
        live = snapshot.get("status") == "connected" and age is not None and age <= MAX_STALE_SECONDS
        forecast = make_minute_forecast(minute_candles, fitted) if live else None

        selected_label = fitted.get("model_label", "Not enough history")
        _help_heading(
            f"{pair} · 1-minute candles + {horizon_minutes}-minute OHLC scenario",
            "Kraken minute candles are shown in green/red. Blue/orange future candles are a scenario from the rule selected on the earlier 70% of history. The last 30% is held out for testing. Use the bottom range slider or mouse wheel to inspect history; the toolbar includes line and shape drawing tools. This chart replaces the hosted TradingView embed because that embed cannot accept this app's forecast overlay.",
        )
        holdout = fitted.get("holdout", {})
        if fitted.get("ready"):
            test_accuracy = holdout.get("directional_accuracy_pct")
            lower_bound = holdout.get("directional_accuracy_lower_95_pct")
            calls = holdout.get("calls", 0)
            accuracy_text = (
                f"{test_accuracy:.1f}% correct from {calls} calls"
                + (f" (95% lower bound {lower_bound:.1f}%)" if lower_bound is not None else "")
                if test_accuracy is not None else "no test calls"
            )
            mae = holdout.get("forecast_mae_pct")
            naive = holdout.get("last_close_mae_on_calls_pct")
            if mae is not None and naive is not None:
                test_summary = f"close MAE {mae:.4f}% vs same-call last-close {naive:.4f}%"
            else:
                test_summary = "Holdout needs more directional examples"
            st.caption(
                f"Selected on earlier candles: {selected_label} · untouched later test: {accuracy_text} · {test_summary}. "
                "Scenario only; not a proven edge."
            )
        else:
            st.caption("Historical selection is waiting for enough Kraken candles. The forecast will stay hidden until the history is sufficient.")

        st.plotly_chart(
            _minute_figure(
                minute_candles,
                forecast,
                name=f"minute-chart-{pair}-{horizon_minutes}",
                height=450,
            ),
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "eraseshape"],
            },
            key=f"minute-chart-{pair}-{horizon_minutes}",
        )
        if forecast:
            scenario_direction = forecast["scenario_direction"]
            scenario = "UPWARD" if scenario_direction > 0 else ("DOWNWARD" if scenario_direction < 0 else "FLAT")
            if forecast["signal_validated"]:
                signal_text = f"{_watch_label(forecast)} · historical screen passed"
            else:
                signal_text = "WAIT · this rule did not pass the minimum holdout screen"
            st.caption(f"{signal_text} · charted candidate: {scenario} OHLC · {forecast['model_label']} · {forecast['note']}")
            with st.expander("How the history test works", expanded=False):
                st.write(
                    f"Kraken REST supplied {fitted['bars']} one-minute candles. The rule was selected and its expected move estimated on the earlier {fitted['train_bars']} bars. "
                    f"Its final {fitted['holdout_bars']} bars were kept aside for the holdout check. The selected rule made {calls} directional calls across {holdout.get('origins', 0)} test opportunities."
                )
                st.write(
                    f"Holdout direction match: {accuracy_text}. On the same {calls} calls, forecast close MAE was {mae:.5f}% "
                    f"and simply keeping the last close had MAE {naive:.5f}%. The all-opportunity momentum baseline had MAE {holdout.get('momentum_mae_pct', float('nan')):.5f}%."
                    if mae is not None and naive is not None else "There were not enough test calls to compare forecast error."
                )
                st.write(
                    "The dashboard only calls a minute direction historically screened when there are at least 30 holdout calls, "
                    "at least 55% matched directions, and its same-call close error beats the no-change baseline. Fees, spread, slippage, and order execution are not included."
                )
        elif not live:
            st.caption("Minute history is loaded, but the current trade feed is stale; the forecast is hidden until trades resume.")

    minute_chart()

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
            f"Kraken live · 1-second candles + next {horizon_seconds} seconds",
            "Green/red candles are actual seconds that contained Kraken trades; a gap means no trade candle was received for that second. "
            "Blue/orange candles are hypothetical OHLC bars for the selected future seconds. They recalculate each second while the feed is fresh, "
            "but are simple scenarios, not real future prices. Their live hit-rate check resets when the app restarts.",
        )
        st.plotly_chart(
            _second_figure(second_candles, forecast, name=f"second-candles-{pair}-{horizon_seconds}", height=450),
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False},
            key=f"second-candles-{pair}-{horizon_seconds}",
        )
        if forecast:
            st.caption(f"EXPERIMENTAL · {_watch_label(forecast)} · {_method_label(forecast['method'])}.")
        else:
            st.caption("Waiting for fresh Kraken trades; forecast candles are hidden until the feed is current.")

    live_second_chart()
