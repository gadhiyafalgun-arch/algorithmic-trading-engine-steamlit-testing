"""
Interactive 3D Trading Dashboard — Plotly Dash
================================================
A stunning interactive dashboard with:
- 3D equity surface (Date × Risk × Portfolio Value)
- Risk slider (1× to 6×)
- Toggle checkboxes (equity, drawdown, benchmark, signals)
- Live metrics panel

Run: python dashboard/dash_app.py
Then open: http://localhost:8050
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go


# ==========================================
# LOAD DATA
# ==========================================

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "dashboard", "dash_data.pkl"
)


def load_data():
    """Load pre-computed dashboard data."""
    if not os.path.exists(DATA_PATH):
        print("❌ Data file not found! Run first:")
        print("   python dashboard/prepare_dash_data.py")
        sys.exit(1)

    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    print(f"✅ Loaded data: {data['symbol']} | {len(data['risk_multipliers'])} risk levels")
    return data


# CSS is auto-loaded from dashboard/assets/style.css


# ==========================================
# BUILD LAYOUT
# ==========================================

def build_layout(data):
    """Build the Dash layout."""
    risk_multipliers = data["risk_multipliers"]
    symbol = data["symbol"]
    signal_col = data["signal_column"]

    slider_marks = {}
    for m in risk_multipliers:
        if m == 1.0:
            slider_marks[m] = {"label": "1× Baseline", "style": {"color": "#00e676", "fontWeight": "700"}}
        elif m == 6.0:
            slider_marks[m] = {"label": "6× Max", "style": {"color": "#ff1744", "fontWeight": "700"}}
        elif m % 1.0 == 0:
            slider_marks[m] = {"label": f"{m:.0f}×", "style": {"color": "#8b949e"}}
        else:
            slider_marks[m] = {"label": f"{m:.1f}×", "style": {"color": "#6c7a89", "fontSize": "10px"}}

    return html.Div(className="app-container", children=[
        # CSS auto-loaded from assets/style.css

        # Hero
        html.Div(className="hero-title", children="🤖 Interactive Trading Dashboard"),
        html.Div(className="hero-subtitle", children=[
            f"3D Equity Explorer — ",
            html.Span(f"{symbol}"),
            f" | Strategy: {signal_col} | Drag the risk slider to see how leverage changes everything."
        ]),

        # ===== CONTROLS BAR =====
        html.Div(className="controls-bar", children=[
            # Risk Slider
            html.Div(className="control-group", children=[
                html.Div(className="section-label", children="⚡ SELECT RISK MULTIPLIER"),
                html.Div(className="risk-slider-container", children=[
                    dcc.Slider(
                        id="risk-slider",
                        min=min(risk_multipliers),
                        max=max(risk_multipliers),
                        step=0.5,
                        value=1.0,
                        marks=slider_marks,
                        tooltip={"placement": "top", "always_visible": True},
                        updatemode="drag",
                    ),
                ]),
                html.Div(className="risk-label-bar", children=[
                    html.Span("🟢 Conservative", className="low"),
                    html.Span("🟡 Balanced", className="mid"),
                    html.Span("🔴 Aggressive", className="high"),
                ]),
            ]),

            # Toggle Checkboxes
            html.Div(className="control-group", style={"marginTop": "18px"}, children=[
                html.Div(className="control-label", children="OVERLAY TOGGLES"),
                dcc.Checklist(
                    id="line-toggles",
                    options=[
                        {"label": " 📈 Equity Curve", "value": "equity"},
                        {"label": " 📉 Drawdown", "value": "drawdown"},
                        {"label": " 📊 Benchmark (SPY)", "value": "benchmark"},
                        {"label": " 🎯 Trading Signals", "value": "signals"},
                    ],
                    value=["equity", "drawdown"],
                    inline=True,
                    className="toggle-checklist",
                ),
            ]),
        ]),

        # ===== LIVE METRICS =====
        html.Div(className="section-label", children="📊 LIVE METRICS", style={"marginBottom": "12px"}),
        html.Div(id="metrics-display", className="metrics-grid"),

        # ===== 3D CHART =====
        html.Div(className="chart-container", children=[
            dcc.Graph(
                id="main-3d-chart",
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToAdd": ["orbitRotation", "resetCameraDefault3d"],
                },
                style={"height": "700px"},
            ),
        ]),

        # ===== INSTRUCTIONS =====
        html.Div(className="instructions-box", children=[
            html.H3("📖 How to Read This Dashboard"),
            html.P([
                "• Drag the RISK SLIDER to change leverage (1× = safe, 6× = aggressive). ",
                "All metrics and the chart update instantly.", html.Br(),
                "• The 3D chart shows your portfolio value over time. ",
                "Green surface = profit, Red = loss. Drag to rotate, scroll to zoom.", html.Br(),
                "• Toggle DRAWDOWN to see the red shaded area below the equity curve — ",
                "this shows peak-to-trough decline (how much you'd lose at the worst moment).", html.Br(),
                "• Toggle BENCHMARK to compare your strategy against buying and holding SPY (S&P 500).", html.Br(),
                "• Toggle SIGNALS to see green ▲ (buy) and red ▼ (sell) markers on the price curve.", html.Br(),
                "• Look at the METRICS PANEL above the chart — Sharpe Ratio > 1.0 is good, ",
                "Win Rate > 50% means the strategy wins more often than it loses.", html.Br(),
                "• Higher risk → higher potential returns BUT also higher drawdowns and volatility.",
            ]),
        ]),

        # Footer
        html.Div(className="footer", children=[
            "🤖 Algorithmic Trading Engine — Built with Python, Dash & Plotly | ",
            "Data: Yahoo Finance | ML: XGBoost, Random Forest, Logistic Regression"
        ]),
    ])


# ==========================================
# CHART BUILDER
# ==========================================

def risk_color(mult, min_m=1.0, max_m=6.0):
    """Generate color from green → yellow → red based on risk level."""
    ratio = (mult - min_m) / (max_m - min_m) if max_m > min_m else 0
    r = int(min(255, 80 + ratio * 175))
    g = int(max(0, 230 - ratio * 200))
    b = int(max(0, 118 - ratio * 118))
    return f"rgb({r},{g},{b})"


def build_chart(data, risk_level, toggles):
    """Build the 3D chart based on current slider + toggle state."""
    fig = go.Figure()

    results = data["risk_level_results"].get(risk_level)
    if results is None:
        # Find nearest available
        available = sorted(data["risk_level_results"].keys())
        risk_level = min(available, key=lambda x: abs(x - risk_level))
        results = data["risk_level_results"][risk_level]

    dates = results["dates"]
    values = np.array(results["total_value"])
    returns = np.array(results["daily_return"])
    days = np.arange(len(dates))
    date_labels = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in dates]
    color = risk_color(risk_level)

    initial_capital = data["initial_capital"]
    final_value = results["final_value"]
    total_return = results["total_return"] * 100

    # ===== Equity Curve =====
    if "equity" in toggles:
        fig.add_trace(go.Scatter3d(
            x=days, y=values, z=returns * 100,
            mode="lines",
            line=dict(color=color, width=6),
            name=f"Equity Curve ({risk_level:.1f}×)",
            text=date_labels,
            hovertemplate=(
                f"<b>Risk: {risk_level:.1f}× | Pos: {results.get('position_size_pct',0):.0f}%</b><br>"
                "Date: %{text}<br>"
                "Value: $%{y:,.0f}<br>"
                "Daily Ret: %{z:.2f}%<extra></extra>"
            ),
        ))

        # Initial capital reference
        fig.add_trace(go.Scatter3d(
            x=[days[0], days[-1]],
            y=[initial_capital, initial_capital],
            z=[0, 0],
            mode="lines",
            line=dict(color="white", width=2, dash="dash"),
            name=f"Initial: ${initial_capital:,.0f}",
            opacity=0.4,
        ))

        # Final value marker
        marker_color = "#00e676" if total_return >= 0 else "#ff1744"
        fig.add_trace(go.Scatter3d(
            x=[days[-1]], y=[final_value], z=[0],
            mode="markers+text",
            marker=dict(size=8, color=marker_color, line=dict(width=2, color="white")),
            text=[f"${final_value:,.0f} ({total_return:+.1f}%)"],
            textposition="top center",
            textfont=dict(size=13, color=color),
            name="Final Value", showlegend=False,
        ))

    # ===== Drawdown =====
    if "drawdown" in toggles:
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        drawdown_values = initial_capital * cumulative + drawdown * initial_capital * 0.01

        fig.add_trace(go.Scatter3d(
            x=days, y=values,
            z=drawdown,
            mode="lines",
            line=dict(color="#ff1744", width=3),
            name="Drawdown (%)",
            text=date_labels,
            opacity=0.8,
            hovertemplate="Date: %{text}<br>Drawdown: %{z:.2f}%<extra></extra>",
        ))

    # ===== Benchmark (SPY) =====
    if "benchmark" in toggles and data.get("benchmark"):
        bench = data["benchmark"]
        bench_returns = np.array(bench["daily_return"])
        bench_cumulative = (1 + bench_returns).cumprod()
        bench_values = initial_capital * bench_cumulative
        bench_days = np.arange(len(bench_values))
        bench_dates = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d)
                       for d in bench["dates"]]

        # Truncate to same length
        min_len = min(len(bench_days), len(days))
        fig.add_trace(go.Scatter3d(
            x=bench_days[:min_len],
            y=bench_values[:min_len],
            z=np.zeros(min_len),
            mode="lines",
            line=dict(color="#9c27b0", width=3, dash="dot"),
            name="Benchmark (SPY)",
            text=bench_dates[:min_len],
            opacity=0.7,
            hovertemplate="SPY %{text}<br>Value: $%{y:,.0f}<extra></extra>",
        ))

    # ===== Trading Signals =====
    if "signals" in toggles:
        signal_col = data.get("signal_column", "combined_signal")
        sig_data = data.get("signal_data", {}).get(signal_col, {})
        close_prices = np.array(data.get("close_prices", []))

        if sig_data:
            buy_dates = sig_data.get("buy_dates", [])
            buy_prices = sig_data.get("buy_prices", [])
            sell_dates = sig_data.get("sell_dates", [])
            sell_prices = sig_data.get("sell_prices", [])

            all_dates = data["dates"]
            date_to_idx = {}
            for i, d in enumerate(all_dates):
                date_to_idx[d] = i

            # Buy markers
            if buy_dates and buy_prices:
                buy_x = [date_to_idx.get(d, 0) for d in buy_dates if d in date_to_idx]
                buy_y = buy_prices[:len(buy_x)]
                fig.add_trace(go.Scatter3d(
                    x=buy_x, y=buy_y,
                    z=np.zeros(len(buy_x)),
                    mode="markers",
                    marker=dict(size=6, color="#00e676", symbol="diamond",
                               line=dict(width=1, color="white")),
                    name="BUY ▲",
                    hovertemplate="BUY<br>$%{y:.2f}<extra></extra>",
                ))

            # Sell markers
            if sell_dates and sell_prices:
                sell_x = [date_to_idx.get(d, 0) for d in sell_dates if d in date_to_idx]
                sell_y = sell_prices[:len(sell_x)]
                fig.add_trace(go.Scatter3d(
                    x=sell_x, y=sell_y,
                    z=np.zeros(len(sell_x)),
                    mode="markers",
                    marker=dict(size=6, color="#ff1744", symbol="diamond",
                               line=dict(width=1, color="white")),
                    name="SELL ▼",
                    hovertemplate="SELL<br>$%{y:.2f}<extra></extra>",
                ))

    # ===== Layout =====
    fig.update_layout(
        title=dict(
            text=(f"📊 {data['symbol']} — Risk {risk_level:.1f}× | "
                  f"Final: ${final_value:,.0f} ({total_return:+.1f}%)"),
            font=dict(size=18, color="#e0e0e0", family="Inter, Arial"),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", size=12, family="Inter, Arial"),
        height=700,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(22,27,34,0.9)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=11),
            x=0.01, y=0.99,
        ),
        scene=dict(
            xaxis=dict(title="Trading Days", color="#8b949e",
                      gridcolor="#21262d", backgroundcolor="#0d1117"),
            yaxis=dict(title="Portfolio Value ($)", color="#8b949e",
                      gridcolor="#21262d", backgroundcolor="#0d1117"),
            zaxis=dict(title="Daily Return / Drawdown (%)", color="#8b949e",
                      gridcolor="#21262d", backgroundcolor="#0d1117"),
            bgcolor="#0d1117",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


# ==========================================
# METRICS BUILDER
# ==========================================

def build_metrics(data, risk_level):
    """Build the metrics cards HTML for the given risk level."""
    metrics = data["all_metrics"].get(risk_level)
    if metrics is None:
        available = sorted(data["all_metrics"].keys())
        risk_level = min(available, key=lambda x: abs(x - risk_level))
        metrics = data["all_metrics"][risk_level]

    total_return = metrics.get("total_return_pct", 0)
    annual_return = metrics.get("annualized_return", 0) * 100
    sharpe = metrics.get("sharpe_ratio", 0)
    max_dd = metrics.get("max_drawdown_pct", 0)
    win_rate = metrics.get("win_rate", 0) * 100
    profit_factor = metrics.get("profit_factor", 0)
    total_trades = metrics.get("total_trades", 0)
    avg_trade = metrics.get("avg_trade_return", 0) * 100
    pos_pct = metrics.get("position_size_pct", 0)
    risk_pct = metrics.get("risk_per_trade_pct", 0)

    def fmt(val, suffix="", prefix="", decimals=1):
        if isinstance(val, str):
            return val
        return f"{prefix}{val:,.{decimals}f}{suffix}"

    def color_class(val, threshold=0, invert=False):
        if isinstance(val, str):
            return "blue"
        if invert:
            return "green" if val <= threshold else "red"
        return "green" if val >= threshold else "red"

    cards = [
        ("⚡ Risk Level", f"{risk_level:.1f}×", "blue",
         f"Pos: {pos_pct:.0f}% | Risk/Trade: {risk_pct:.1f}%"),
        ("💰 Total Return", fmt(total_return, "%"), color_class(total_return), ""),
        ("📈 Annual Return", fmt(annual_return, "%"), color_class(annual_return), ""),
        ("📐 Sharpe Ratio", fmt(sharpe, "", "", 2), color_class(sharpe, 1.0), ""),
        ("📉 Max Drawdown", fmt(max_dd, "%"), color_class(max_dd, -10, True), ""),
        ("🎯 Win Rate", fmt(win_rate, "%"), color_class(win_rate, 50), ""),
        ("⚖️ Profit Factor", fmt(profit_factor, "", "", 2),
         color_class(profit_factor, 1.0), ""),
        ("🔢 Total Trades", str(int(total_trades)), "cyan", ""),
        ("💵 Avg Trade", fmt(avg_trade, "%", "", 2), color_class(avg_trade), ""),
    ]

    children = []
    for label, value, cls, subtitle in cards:
        card_children = [
            html.Div(value, className=f"metric-value {cls}"),
            html.Div(label, className="metric-label"),
        ]
        if subtitle:
            card_children.append(
                html.Div(subtitle, style={
                    "fontSize": "10px", "color": "#4a5568", "marginTop": "4px"
                })
            )
        children.append(html.Div(className="metric-card", children=card_children))

    return children


# ==========================================
# CREATE APP
# ==========================================

DATA = load_data()

app = Dash(
    __name__,
    title=f"Trading Dashboard — {DATA['symbol']}",
    update_title="Updating...",
)
app.layout = build_layout(DATA)


# ==========================================
# CALLBACKS
# ==========================================

@callback(
    [Output("main-3d-chart", "figure"),
     Output("metrics-display", "children")],
    [Input("risk-slider", "value"),
     Input("line-toggles", "value")]
)
def update_dashboard(risk_level, toggles):
    """Main callback: updates chart + metrics when slider or toggles change."""
    fig = build_chart(DATA, risk_level, toggles or [])
    metrics = build_metrics(DATA, risk_level)
    return fig, metrics


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 INTERACTIVE 3D TRADING DASHBOARD")
    print("=" * 60)
    print(f"   Symbol: {DATA['symbol']}")
    print(f"   Strategy: {DATA['signal_column']}")
    print(f"   Risk levels: {DATA['risk_multipliers']}")
    print(f"   Open: http://localhost:8050")
    print("=" * 60 + "\n")

    app.run(debug=True, host="0.0.0.0", port=8050)
