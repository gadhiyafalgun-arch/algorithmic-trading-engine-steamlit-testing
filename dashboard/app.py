"""
Algorithmic Trading Engine — Live Dashboard
=============================================
Streamlit web app that runs the full trading engine live.
Visitors can select stocks, strategies, date ranges, and drag
the risk slider to see how results change in real time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import sys
import os
from datetime import datetime, date

# ── Make src/ importable from dashboard/ ──────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.indicators import TechnicalIndicators
from src.strategy import TradingStrategy
from src.backtester import Backtester
from src.performance import PerformanceAnalyzer

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Algorithmic Trading Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — dark, premium, quant-terminal aesthetic
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Exo+2:wght@300;400;600;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #080c14;
    color: #c9d1d9;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0f1a 100%);
    border-right: 1px solid #1a2332;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Rajdhani', sans-serif;
    color: #58a6ff;
    font-size: 1.1rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid #1a2332;
    padding-bottom: 6px;
    margin-bottom: 12px;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1117 0%, #111827 100%);
    border: 1px solid #1a2332;
    border-radius: 8px;
    padding: 14px 18px;
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #58a6ff, #00e676);
}
[data-testid="metric-container"] label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 1.6rem;
    color: #e6edf3;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 6px 0 6px 12px;
    border-left: 3px solid #58a6ff;
    margin: 24px 0 14px 0;
}

/* ── Grade badge ── */
.grade-badge {
    display: inline-block;
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    padding: 8px 28px;
    border-radius: 8px;
    letter-spacing: 0.05em;
}
.grade-a-plus { background: linear-gradient(135deg,#0a2a1a,#0d3320); color:#00e676; border:2px solid #00e676; }
.grade-a      { background: linear-gradient(135deg,#0a2a1a,#0d3320); color:#26c95a; border:2px solid #26c95a; }
.grade-b      { background: linear-gradient(135deg,#1a1a0a,#25250d); color:#ffd60a; border:2px solid #ffd60a; }
.grade-c      { background: linear-gradient(135deg,#1a100a,#251a0d); color:#ff9800; border:2px solid #ff9800; }
.grade-d      { background: linear-gradient(135deg,#1a0a0a,#25100d); color:#ff5722; border:2px solid #ff5722; }
.grade-f      { background: linear-gradient(135deg,#1a0a0a,#250d0d); color:#ff1744; border:2px solid #ff1744; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #0a0f1a 40%, #0d1a2a 100%);
    border: 1px solid #1a2332;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #58a6ff 0%, #00e676 50%, #ff9800 100%);
}
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #e6edf3;
    letter-spacing: 0.05em;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #58a6ff;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ── Stat pill ── */
.stat-pill {
    display: inline-block;
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.3);
    border-radius: 20px;
    padding: 3px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #58a6ff;
    margin-right: 8px;
    margin-top: 10px;
}

/* ── Risk bar label ── */
.risk-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    color: #8b949e;
    text-transform: uppercase;
}

/* ── Trade table ── */
.trade-table {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
}

/* ── Divider ── */
hr { border-color: #1a2332 !important; }

/* ── Plotly chart container ── */
.stPlotlyChart {
    border: 1px solid #1a2332;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #58a6ff !important; }

/* ── Selectbox / slider labels ── */
.stSelectbox label, .stSlider label, .stDateInput label,
.stMultiSelect label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #8b949e;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Button ── */
.stButton > button {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: linear-gradient(135deg, #1a2a4a, #0d1a2a);
    border: 1px solid #58a6ff;
    color: #58a6ff;
    border-radius: 6px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #58a6ff, #1a6be0);
    color: white;
    border-color: #58a6ff;
}

/* ── Info / warning boxes ── */
.stAlert {
    border-radius: 8px;
    font-family: 'Exo 2', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY", "NVDA", "META"]
STRATEGIES = {
    "Combined (Multi-Signal)": "combined_signal",
    "SMA Crossover":           "sma_signal",
    "RSI (Trend-Aware)":       "rsi_signal",
    "MACD (Filtered)":         "macd_trade_signal",
    "Bollinger Bands":         "bb_signal",
}
RISK_MULTIPLIERS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_and_process(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch from yfinance, clean, add indicators + strategies. Cached 1 hr."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval="1d")
    if df.empty:
        return pd.DataFrame()

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    needed = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in needed if c in df.columns]]
    df = df[df["close"] > 0].dropna()

    # daily return (needed by backtester portfolio tracking)
    df["daily_return"] = df["close"].pct_change()

    ti = TechnicalIndicators()
    df = ti.add_all_indicators(df)

    strat = TradingStrategy()
    df = strat.apply_all_strategies(df)

    return df


def build_mock_config(
    initial_capital: float,
    commission: float,
    slippage: float,
    max_position: float,
    risk_per_trade: float,
    stop_loss: float,
    take_profit: float,
    max_drawdown: float = 0.15,
    max_positions: int = 5,
) -> dict:
    """Build a config dict that Backtester can use."""
    return {
        "trading": {
            "initial_capital": initial_capital,
            "commission": commission / 100,
            "slippage": slippage / 100,
            "max_position_size": max_position / 100,
            "risk_per_trade": risk_per_trade / 100,
        },
        "risk": {
            "stop_loss": stop_loss / 100,
            "take_profit": take_profit / 100,
            "max_drawdown": max_drawdown,
            "max_open_positions": max_positions,
        },
    }


class DashboardBacktester(Backtester):
    """Backtester that takes a config dict instead of reading a yaml file."""
    def __init__(self, config: dict):
        self.config = config
        t = config["trading"]
        r = config["risk"]
        self.initial_capital   = t["initial_capital"]
        self.commission_rate   = t["commission"]
        self.slippage_rate     = t["slippage"]
        self.max_position_size = t["max_position_size"]
        self.risk_per_trade    = t["risk_per_trade"]
        self.stop_loss_pct     = r["stop_loss"]
        self.take_profit_pct   = r["take_profit"]
        self.max_drawdown_limit = r["max_drawdown"]
        self.max_open_positions = r["max_open_positions"]


def run_backtest_at_risk(df, signal_col, config, risk_mult):
    """Run backtest at a given risk multiplier."""
    cfg = {
        "trading": {
            "initial_capital":   config["trading"]["initial_capital"],
            "commission":        config["trading"]["commission"],
            "slippage":          config["trading"]["slippage"],
            "max_position_size": min(config["trading"]["max_position_size"] * risk_mult, 0.95),
            "risk_per_trade":    min(config["trading"]["risk_per_trade"] * risk_mult, 0.20),
        },
        "risk": config["risk"],
    }
    bt = DashboardBacktester(cfg)
    results = bt.run(df, signal_column=signal_col, symbol="STOCK")
    if results:
        results["risk_multiplier"] = risk_mult
        results["position_size_pct"] = cfg["trading"]["max_position_size"] * 100
    return results


def grade_color(grade: str) -> str:
    if "A+" in grade: return "grade-a-plus"
    if grade.startswith("A"):  return "grade-a"
    if grade.startswith("B"):  return "grade-b"
    if grade.startswith("C"):  return "grade-c"
    if grade.startswith("D"):  return "grade-d"
    return "grade-f"


def fmt_pct(v, decimals=2):
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%"


def fmt_dollar(v):
    return f"${v:,.2f}"


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS (Plotly, dark theme)
# ══════════════════════════════════════════════════════════════════════════════

DARK = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="Exo 2, sans-serif"),
)


def chart_price_signals(df: pd.DataFrame, symbol: str, signal_col: str) -> go.Figure:
    """3D price + volume + buy/sell signals."""
    days = np.arange(len(df), dtype=float)
    prices = df["close"].values
    vols = df["volume"].values
    vol_norm = vols / vols.max() * prices.max() * 0.25
    dates = [str(d)[:10] for d in df.index]

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter3d(
        x=days, y=prices, z=vol_norm,
        mode="lines",
        line=dict(color="#00e676", width=3),
        name="Price",
        text=dates,
        hovertemplate="Date: %{text}<br>Price: $%{y:.2f}<extra></extra>",
    ))

    # SMA overlays
    for col, color, label in [("sma_20","#ff9800","SMA 20"),("sma_50","#2196f3","SMA 50")]:
        if col in df.columns:
            fig.add_trace(go.Scatter3d(
                x=days, y=df[col].values, z=np.zeros_like(days),
                mode="lines", line=dict(color=color, width=2, dash="dot"),
                name=label, opacity=0.7,
            ))

    # Volume dots on floor
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Scatter3d(
        x=days, y=np.full_like(days, prices.min()*0.96), z=vol_norm,
        mode="markers", marker=dict(size=2, color=vol_colors, opacity=0.35),
        name="Volume", hoverinfo="skip",
    ))

    # BUY signals
    buys = df[df[signal_col] == 1]
    if len(buys):
        bi = [df.index.get_loc(i) for i in buys.index]
        fig.add_trace(go.Scatter3d(
            x=np.array(bi, dtype=float), y=buys["close"].values,
            z=buys["volume"].values / vols.max() * prices.max() * 0.25,
            mode="markers",
            marker=dict(size=8, color="#00e676", symbol="diamond",
                        line=dict(width=2, color="white")),
            name="BUY ▲",
            text=[str(d)[:10] for d in buys.index],
            hovertemplate="BUY %{text}<br>$%{y:.2f}<extra></extra>",
        ))

    # SELL signals
    sells = df[df[signal_col] == -1]
    if len(sells):
        si = [df.index.get_loc(i) for i in sells.index]
        fig.add_trace(go.Scatter3d(
            x=np.array(si, dtype=float), y=sells["close"].values,
            z=sells["volume"].values / vols.max() * prices.max() * 0.25,
            mode="markers",
            marker=dict(size=8, color="#ff1744", symbol="diamond",
                        line=dict(width=2, color="white")),
            name="SELL ▼",
            text=[str(d)[:10] for d in sells.index],
            hovertemplate="SELL %{text}<br>$%{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        **DARK,
        title=dict(text=f"⚡ {symbol} — Price, Volume & Trade Signals (3D)",
                   font=dict(size=16, color="#e6edf3", family="Rajdhani, sans-serif"),
                   x=0.5, xanchor="center"),
        height=620,
        scene=dict(
            xaxis=dict(title="Trading Days", gridcolor="#21262d", backgroundcolor="#0d1117", color="#8b949e"),
            yaxis=dict(title="Price ($)",    gridcolor="#21262d", backgroundcolor="#0d1117", color="#8b949e"),
            zaxis=dict(title="Volume",       gridcolor="#21262d", backgroundcolor="#0d1117", color="#8b949e"),
            bgcolor="#0d1117",
            camera=dict(eye=dict(x=1.8, y=-1.4, z=0.8)),
        ),
        legend=dict(bgcolor="rgba(13,17,23,0.8)", bordercolor="#21262d", borderwidth=1),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def chart_equity_risk(risk_results: dict, initial_capital: float,
                      symbol: str, strategy: str, active_mult: float) -> go.Figure:
    """3D equity curves for all risk levels — visibility toggled by slider."""
    sorted_levels = sorted(risk_results.keys())

    def risk_color(mult):
        lo, hi = sorted_levels[0], sorted_levels[-1]
        ratio = (mult - lo) / (hi - lo) if hi > lo else 0
        r = int(min(255, 80  + ratio * 175))
        g = int(max(0,   230 - ratio * 200))
        b = int(max(0,   118 - ratio * 118))
        return f"rgb({r},{g},{b})"

    fig = go.Figure()
    traces_per = 3
    default_idx = sorted_levels.index(active_mult) if active_mult in sorted_levels else \
                  sorted_levels.index(min(sorted_levels, key=lambda x: abs(x-1.0)))

    for idx, mult in enumerate(sorted_levels):
        res = risk_results[mult]
        pdf = res["portfolio_history"]
        days = np.arange(len(pdf), dtype=float)
        vals = pdf["total_value"].values
        rets = pdf["daily_return"].fillna(0).values * 100
        color = risk_color(mult)
        vis = (idx == default_idx)
        pos_pct = res.get("position_size_pct", 0)
        final = res.get("final_value", 0)
        ret_pct = res.get("total_return", 0) * 100
        dates = [str(d)[:10] for d in pdf.index]

        fig.add_trace(go.Scatter3d(
            x=days, y=vals, z=rets,
            mode="lines", line=dict(color=color, width=5),
            name=f"Risk {mult:.2f}×",
            visible=vis,
            text=dates,
            hovertemplate=(
                f"<b>Risk {mult:.2f}× | Pos: {pos_pct:.0f}%</b><br>"
                "Date: %{text}<br>Value: $%{y:,.0f}<br>Daily Ret: %{z:.2f}%<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter3d(
            x=[days[0], days[-1]], y=[initial_capital]*2, z=[0,0],
            mode="lines", line=dict(color="white", width=1, dash="dash"),
            name="Initial Capital", visible=vis, opacity=0.4,
            showlegend=(idx == 0),
        ))
        mc = "#00e676" if ret_pct >= 0 else "#ff1744"
        fig.add_trace(go.Scatter3d(
            x=[days[-1]], y=[final], z=[0],
            mode="markers+text",
            marker=dict(size=8, color=mc, line=dict(width=2, color="white")),
            text=[f"${final:,.0f} ({ret_pct:+.1f}%)"],
            textposition="top center",
            textfont=dict(size=12, color=color),
            visible=vis, showlegend=False,
        ))

    # Build slider
    steps = []
    for idx, mult in enumerate(sorted_levels):
        vis_arr = [False] * (len(sorted_levels) * traces_per)
        base = idx * traces_per
        vis_arr[base] = vis_arr[base+1] = vis_arr[base+2] = True
        res = risk_results[mult]
        final = res.get("final_value", 0)
        ret_pct = res.get("total_return", 0) * 100
        pos_pct = res.get("position_size_pct", 0)
        steps.append(dict(
            args=[{"visible": vis_arr},
                  {"title.text": f"📊 {symbol} | {strategy} | Risk {mult:.2f}× | Pos {pos_pct:.0f}% | ${final:,.0f} ({ret_pct:+.1f}%)"}],
            label=f"{mult:.2f}×",
            method="update",
        ))

    res0 = risk_results[sorted_levels[default_idx]]
    final0 = res0.get("final_value", 0)
    ret0 = res0.get("total_return", 0) * 100
    pos0 = res0.get("position_size_pct", 0)

    fig.update_layout(
        **DARK,
        title=dict(
            text=f"📊 {symbol} | {strategy} | Risk {sorted_levels[default_idx]:.2f}× | Pos {pos0:.0f}% | ${final0:,.0f} ({ret0:+.1f}%)",
            font=dict(size=15, color="#e6edf3", family="Rajdhani, sans-serif"),
            x=0.5, xanchor="center",
        ),
        height=700,
        scene=dict(
            xaxis=dict(title="Trading Days", gridcolor="#21262d", backgroundcolor="#0d1117", color="#8b949e"),
            yaxis=dict(title="Portfolio Value ($)", gridcolor="#21262d", backgroundcolor="#0d1117", color="#8b949e"),
            zaxis=dict(title="Daily Return (%)", gridcolor="#21262d", backgroundcolor="#0d1117", color="#8b949e"),
            bgcolor="#0d1117",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        legend=dict(bgcolor="rgba(13,17,23,0.8)", bordercolor="#21262d", borderwidth=1),
        margin=dict(l=0, r=0, t=60, b=80),
        sliders=[dict(
            active=default_idx,
            currentvalue=dict(prefix="⚡ Risk Level: ", visible=True,
                              xanchor="center",
                              font=dict(size=14, color="#e0e0e0")),
            pad=dict(b=10, t=40),
            len=0.9, x=0.05,
            steps=steps,
            bgcolor="#21262d",
            activebgcolor="#58a6ff",
            bordercolor="#30363d",
            font=dict(size=10, color="#c9d1d9"),
        )],
    )
    return fig


def chart_macd(df: pd.DataFrame, symbol: str) -> go.Figure:
    """2-panel MACD chart with price and MACD histogram."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.04,
    )

    # Price + SMAs
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], mode="lines",
        line=dict(color="#00e676", width=2), name="Price",
    ), row=1, col=1)
    for col, color, label in [("sma_20","#ff9800","SMA 20"),("sma_50","#2196f3","SMA 50")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode="lines",
                line=dict(color=color, width=1.5, dash="dot"), name=label,
            ), row=1, col=1)

    # MACD line + signal
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd_line"], mode="lines",
        line=dict(color="#2196f3", width=1.5), name="MACD",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd_signal"], mode="lines",
        line=dict(color="#ff9800", width=1.5), name="Signal",
    ), row=2, col=1)

    # Histogram
    hist = df["macd_histogram"].values
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in hist]
    fig.add_trace(go.Bar(
        x=df.index, y=hist, marker_color=colors,
        name="Histogram", opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        **DARK,
        title=dict(text=f"📉 {symbol} — MACD Analysis",
                   font=dict(size=16, color="#e6edf3", family="Rajdhani, sans-serif"),
                   x=0.5, xanchor="center"),
        height=520,
        hovermode="x unified",
        xaxis2=dict(gridcolor="#21262d", color="#8b949e"),
        yaxis=dict(gridcolor="#21262d", color="#8b949e", title="Price ($)"),
        yaxis2=dict(gridcolor="#21262d", color="#8b949e", title="MACD"),
        legend=dict(bgcolor="rgba(13,17,23,0.8)", bordercolor="#21262d", borderwidth=1),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def chart_drawdown(portfolio_df: pd.DataFrame, symbol: str) -> go.Figure:
    """Drawdown chart."""
    daily = portfolio_df["daily_return"].fillna(0)
    cum = (1 + daily).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_df.index, y=dd,
        mode="lines", fill="tozeroy",
        line=dict(color="#ff1744", width=1.5),
        fillcolor="rgba(255,23,68,0.15)",
        name="Drawdown",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **DARK,
        title=dict(text=f"📉 {symbol} — Portfolio Drawdown",
                   font=dict(size=16, color="#e6edf3", family="Rajdhani, sans-serif"),
                   x=0.5, xanchor="center"),
        height=300,
        xaxis=dict(gridcolor="#21262d", color="#8b949e"),
        yaxis=dict(gridcolor="#21262d", color="#8b949e", title="Drawdown (%)"),
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode="x unified",
    )
    return fig


def chart_equity_2d(portfolio_df: pd.DataFrame, symbol: str,
                    initial_capital: float, strategy: str) -> go.Figure:
    """Clean 2D equity curve."""
    vals = portfolio_df["total_value"].values
    fig = go.Figure()
    fig.add_hline(y=initial_capital, line=dict(color="white", width=1, dash="dash"), opacity=0.4)
    colors_fill = ["rgba(0,230,118,0.12)" if vals[-1] >= initial_capital
                   else "rgba(255,23,68,0.12)"]
    line_color  = "#00e676" if vals[-1] >= initial_capital else "#ff1744"
    fig.add_trace(go.Scatter(
        x=portfolio_df.index, y=vals,
        mode="lines", line=dict(color=line_color, width=2),
        fill="tozeroy", fillcolor=colors_fill[0],
        name="Portfolio Value",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **DARK,
        title=dict(text=f"📈 {symbol} — Equity Curve ({strategy})",
                   font=dict(size=16, color="#e6edf3", family="Rajdhani, sans-serif"),
                   x=0.5, xanchor="center"),
        height=340,
        xaxis=dict(gridcolor="#21262d", color="#8b949e"),
        yaxis=dict(gridcolor="#21262d", color="#8b949e", title="Portfolio Value ($)"),
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Engine Config")

    symbol = st.selectbox("Stock Symbol", SYMBOLS, index=0)
    strategy_name = st.selectbox("Trading Strategy", list(STRATEGIES.keys()), index=0)
    signal_col = STRATEGIES[strategy_name]

    st.markdown("---")
    st.markdown("## 📅 Date Range")
    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Start", value=date(2020, 1, 1),
                                   min_value=date(2015, 1, 1), max_value=date(2024, 1, 1))
    with col_b:
        end_date = st.date_input("End", value=date(2024, 12, 31),
                                 min_value=date(2016, 1, 1), max_value=date(2025, 12, 31))

    st.markdown("---")
    st.markdown("## 💰 Capital & Costs")
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000,
                                       max_value=10_000_000, value=100_000, step=5000)
    commission = st.slider("Commission (%)", 0.0, 1.0, 0.1, 0.01,
                           help="% charged per trade (buy + sell)")
    slippage   = st.slider("Slippage (%)", 0.0, 0.5, 0.05, 0.01,
                           help="Price impact of executing trades")

    st.markdown("---")
    st.markdown("## 🎚️ Risk Parameters")
    max_position = st.slider("Max Position Size (%)", 5, 95, 20, 5,
                             help="Max % of capital in one trade")
    risk_per_trade = st.slider("Risk Per Trade (%)", 0.5, 20.0, 2.0, 0.5)
    stop_loss   = st.slider("Stop Loss (%)", 1, 20, 5, 1)
    take_profit = st.slider("Take Profit (%)", 1, 50, 10, 1)

    st.markdown("---")
    run_btn = st.button("▶  RUN ENGINE", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">⚡ ALGORITHMIC TRADING ENGINE</div>
    <div class="hero-subtitle">Quantitative Strategy Backtester · 3D Interactive Analytics · ML-Ready</div>
    <span class="stat-pill">5 Strategies</span>
    <span class="stat-pill">8 Stocks</span>
    <span class="stat-pill">3D Charts</span>
    <span class="stat-pill">Live Risk Slider</span>
    <span class="stat-pill">Real Market Data</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ══════════════════════════════════════════════════════════════════════════════

if not run_btn and "results" not in st.session_state:
    st.info("👈  Configure your parameters in the sidebar, then hit **RUN ENGINE** to start.")
    st.markdown("""
    <div style='padding:24px; background:#0d1117; border:1px solid #1a2332; border-radius:10px; margin-top:20px;'>
    <h4 style='font-family:Rajdhani,sans-serif;color:#58a6ff;margin:0 0 12px 0;'>What this engine does</h4>
    <p style='color:#8b949e;font-size:0.9rem;line-height:1.7;'>
    Fetches real historical market data from Yahoo Finance, computes 8 technical indicators
    (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, VWAP), applies your chosen trading
    strategy, and runs a full day-by-day backtest with realistic commission &amp; slippage simulation.<br><br>
    The <strong style='color:#e6edf3;'>risk slider</strong> re-runs the backtest at 10 different capital
    allocation levels so you can instantly see how more or less aggressive position sizing changes
    your returns, drawdown, and Sharpe ratio.
    </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Run / cache results ────────────────────────────────────────────────────────
if run_btn:
    st.session_state.pop("results", None)  # Force fresh run

if "results" not in st.session_state or run_btn:
    config = build_mock_config(
        initial_capital, commission, slippage,
        max_position, risk_per_trade, stop_loss, take_profit,
    )

    with st.spinner(f"Fetching {symbol} data and running engine…"):
        df = fetch_and_process(symbol, str(start_date), str(end_date))

    if df.empty:
        st.error(f"Could not fetch data for {symbol}. Try a different symbol or date range.")
        st.stop()

    if signal_col not in df.columns or df[signal_col].abs().sum() == 0:
        st.warning(f"Strategy **{strategy_name}** generated no signals for {symbol} in this date range. Try a different combination.")
        st.stop()

    with st.spinner("Running backtests across all risk levels…"):
        risk_results = {}
        for mult in RISK_MULTIPLIERS:
            res = run_backtest_at_risk(df, signal_col, config, mult)
            if res:
                risk_results[mult] = res

    if not risk_results:
        st.error("Backtest produced no results. Try adjusting your parameters.")
        st.stop()

    # Baseline = 1.0× (or closest)
    base_mult = 1.0 if 1.0 in risk_results else min(risk_results, key=lambda x: abs(x-1.0))
    base_res = risk_results[base_mult]

    perf = PerformanceAnalyzer()
    metrics = perf.calculate_all_metrics(
        base_res["portfolio_history"],
        base_res["trades"],
        initial_capital,
    )
    grade_str = perf._grade_strategy(metrics)

    st.session_state["results"] = dict(
        df=df, risk_results=risk_results, base_res=base_res,
        metrics=metrics, grade_str=grade_str,
        symbol=symbol, strategy_name=strategy_name,
        signal_col=signal_col, initial_capital=initial_capital,
        config=config,
    )

R = st.session_state["results"]
df          = R["df"]
risk_results = R["risk_results"]
base_res    = R["base_res"]
metrics     = R["metrics"]
grade_str   = R["grade_str"]
sym         = R["symbol"]
strat_name  = R["strategy_name"]
sig_col     = R["signal_col"]
init_cap    = R["initial_capital"]

# ══════════════════════════════════════════════════════════════════════════════
# KEY METRICS ROW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Performance Summary</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
ret_pct = metrics["total_return_pct"]
c1.metric("Total Return",    fmt_pct(ret_pct),
          delta=fmt_pct(metrics["annualized_return"]*100) + " ann.")
c2.metric("Final Value",     f"${metrics['final_value']:,.0f}",
          delta=f"${metrics['final_value']-init_cap:+,.0f}")
c3.metric("Sharpe Ratio",    f"{metrics['sharpe_ratio']:.3f}",
          delta="Sortino: " + f"{metrics['sortino_ratio']:.2f}")
c4.metric("Max Drawdown",    fmt_pct(metrics['max_drawdown_pct']),
          delta=f"Calmar: {metrics['calmar_ratio']:.2f}")
c5.metric("Win Rate",        f"{metrics['win_rate']*100:.1f}%",
          delta=f"{metrics['total_trades']} trades")
c6.metric("Profit Factor",   f"{metrics['profit_factor']:.2f}",
          delta=f"VaR 95%: {metrics['var_95']*100:.2f}%")

# Grade
st.markdown("---")
gcls = grade_color(grade_str)
letter = grade_str.split()[0]
desc   = " ".join(grade_str.split()[1:])
st.markdown(
    f'<span class="grade-badge {gcls}">{letter}</span>'
    f'<span style="margin-left:14px;font-family:Exo 2,sans-serif;font-size:1rem;color:#8b949e;">{desc}</span>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# RISK SLIDER + EQUITY CHART (3D)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎚️ Interactive Risk Bar — 3D Equity Curve</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="risk-label">drag the slider inside the chart ↓ to change risk level · '
    'green→yellow→red = low→medium→high risk · chart updates live</p>',
    unsafe_allow_html=True,
)

sorted_lvls = sorted(risk_results.keys())
active_mult = 1.0 if 1.0 in risk_results else sorted_lvls[len(sorted_lvls)//2]
fig_risk = chart_equity_risk(risk_results, init_cap, sym, strat_name, active_mult)
st.plotly_chart(fig_risk, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PRICE + SIGNALS (3D) + EQUITY 2D + DRAWDOWN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📈 Price, Signals & Equity</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2])
with col_left:
    st.plotly_chart(chart_price_signals(df, sym, sig_col), use_container_width=True)
with col_right:
    st.plotly_chart(chart_equity_2d(base_res["portfolio_history"], sym,
                                    init_cap, strat_name), use_container_width=True)
    st.plotly_chart(chart_drawdown(base_res["portfolio_history"], sym),
                    use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MACD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📉 MACD Analysis</div>', unsafe_allow_html=True)
st.plotly_chart(chart_macd(df, sym), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TRADE LOG
# ══════════════════════════════════════════════════════════════════════════════
trades_df = base_res["trades"]
if not trades_df.empty:
    st.markdown('<div class="section-header">📋 Trade Log</div>', unsafe_allow_html=True)

    display = trades_df.copy()
    display["pnl"] = display["pnl"].map(lambda x: f"${x:+,.2f}")
    display["pnl_percent"] = display["pnl_percent"].map(lambda x: f"{x*100:+.2f}%")
    display["entry_price"] = display["entry_price"].map(lambda x: f"${x:.2f}")
    display["exit_price"]  = display["exit_price"].map(lambda x: f"${x:.2f}" if pd.notna(x) else "—")

    show_cols = ["symbol","entry_date","entry_price","exit_date",
                 "exit_price","shares","pnl","pnl_percent","status"]
    show_cols = [c for c in show_cols if c in display.columns]

    st.dataframe(
        display[show_cols].tail(50),
        use_container_width=True,
        height=320,
    )

# ══════════════════════════════════════════════════════════════════════════════
# DETAILED METRICS TABLE
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🔬 Full Metrics Detail", expanded=False):
    m = metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Returns**")
        st.json({
            "Total Return":      fmt_pct(m["total_return_pct"]),
            "Annualized Return": fmt_pct(m["annualized_return"]*100),
            "Initial Capital":   fmt_dollar(m["initial_capital"]),
            "Final Value":       fmt_dollar(m["final_value"]),
        })
    with col2:
        st.markdown("**Risk**")
        st.json({
            "Sharpe Ratio":       round(m["sharpe_ratio"],3),
            "Sortino Ratio":      round(m["sortino_ratio"],3),
            "Calmar Ratio":       round(m["calmar_ratio"],3),
            "Max Drawdown":       fmt_pct(m["max_drawdown_pct"]),
            "Annual Volatility":  fmt_pct(m["volatility_annual"]*100),
            "VaR 95%":            fmt_pct(m["var_95"]*100,3),
            "VaR 99%":            fmt_pct(m["var_99"]*100,3),
        })
    with col3:
        st.markdown("**Trades**")
        st.json({
            "Total Trades":   m.get("total_trades",0),
            "Win Rate":       fmt_pct(m.get("win_rate",0)*100,1),
            "Profit Factor":  round(m.get("profit_factor",0),2),
            "Avg Win":        fmt_dollar(m.get("avg_win",0)),
            "Avg Loss":       fmt_dollar(m.get("avg_loss",0)),
            "Best Trade":     fmt_dollar(m.get("best_trade",0)),
            "Worst Trade":    fmt_dollar(m.get("worst_trade",0)),
            "Expectancy":     fmt_dollar(m.get("expectancy",0)),
        })

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p style="text-align:center;font-family:Share Tech Mono,monospace;'
    'font-size:0.72rem;color:#30363d;letter-spacing:0.15em;">'
    'ALGORITHMIC TRADING ENGINE · BUILT BY FALGUN GADHIYA · '
    'DATA: YAHOO FINANCE · NOT FINANCIAL ADVICE'
    '</p>',
    unsafe_allow_html=True,
)
