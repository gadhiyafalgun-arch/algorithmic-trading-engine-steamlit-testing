"""
Visualizer Module — 3D Edition
================================
Creates stunning 3D animated charts and visualizations for trading data.
All charts use Plotly 3D surfaces, animated traces, and interactive controls.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from loguru import logger


class Visualizer:
    """
    Creates interactive 3D charts using Plotly.
    Charts are saved as HTML files you can open in browser.

    Features:
    - 3D surface charts for price/volume data
    - Animated buy/sell markers
    - Interactive risk bar slider on backtest chart
    - 3D ribbon equity comparison
    - Animated MACD histogram
    """

    def __init__(self, output_dir: str = "docs/charts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("3D Visualizer initialized")

    # ==========================================
    # SHARED HELPERS
    # ==========================================

    def _get_dark_layout(self, title: str, height: int = 900) -> dict:
        """Return a premium dark layout config shared by all charts."""
        return dict(
            title=dict(
                text=title,
                font=dict(size=22, color="#e0e0e0", family="Arial Black"),
                x=0.5, xanchor="center"
            ),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9", size=12),
            height=height,
            showlegend=True,
            legend=dict(
                bgcolor="rgba(22,27,34,0.8)",
                bordercolor="#30363d",
                borderwidth=1,
                font=dict(size=11)
            ),
        )

    def _date_to_numeric(self, dates):
        """Convert datetime index to numeric days for 3D axes."""
        origin = dates.min()
        return np.array([(d - origin).days for d in dates], dtype=float)

    def _make_animation_frames(self, x, y, z, name, color, n_frames=30):
        """Create animation frames that progressively reveal data."""
        frames = []
        total = len(x)
        step = max(1, total // n_frames)
        for i in range(step, total + 1, step):
            frames.append(go.Frame(
                data=[go.Scatter3d(
                    x=x[:i], y=y[:i], z=z[:i],
                    mode="lines",
                    line=dict(color=color, width=4),
                    name=name
                )],
                name=str(i)
            ))
        # Ensure final frame has all data
        if total > 0:
            frames.append(go.Frame(
                data=[go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="lines",
                    line=dict(color=color, width=4),
                    name=name
                )],
                name=str(total)
            ))
        return frames

    def _add_instructions(self, fig, instructions: list, y_start: float = -0.06):
        """Add reading instructions as annotations below the chart."""
        # Header
        fig.add_annotation(
            text="📖 HOW TO READ THIS CHART",
            showarrow=False,
            x=0.0, y=y_start,
            xref="paper", yref="paper",
            font=dict(size=14, color="#58a6ff", family="Arial Black"),
            xanchor="left", yanchor="top"
        )
        # Individual instruction lines
        for i, instruction in enumerate(instructions):
            fig.add_annotation(
                text=instruction,
                showarrow=False,
                x=0.0, y=y_start - 0.03 * (i + 1),
                xref="paper", yref="paper",
                font=dict(size=11, color="#8b949e"),
                xanchor="left", yanchor="top"
            )

    # ==========================================
    # CHART 1 & 4: Price with Signals (3D)
    # ==========================================

    def plot_price_with_signals(self, df: pd.DataFrame, symbol: str,
                                 signal_column: str = "sma_signal") -> None:
        """
        3D chart: Date × Price × Volume surface with animated buy/sell markers.
        """
        fig = go.Figure()

        days = self._date_to_numeric(df.index)
        date_labels = [d.strftime("%Y-%m-%d") for d in df.index]
        prices = df["close"].values
        volumes = df["volume"].values
        vol_normalized = volumes / volumes.max() * prices.max() * 0.3

        # 3D price line (main trace)
        fig.add_trace(go.Scatter3d(
            x=days, y=prices, z=vol_normalized,
            mode="lines",
            line=dict(color="#00e676", width=3),
            name="Price",
            text=date_labels,
            hovertemplate=(
                "Date: %{text}<br>"
                "Price: $%{y:.2f}<br>"
                "Volume: %{z:.0f}<extra></extra>"
            )
        ))

        # SMA overlays in 3D
        if "sma_20" in df.columns:
            sma20 = df["sma_20"].values
            fig.add_trace(go.Scatter3d(
                x=days, y=sma20, z=np.zeros_like(days),
                mode="lines",
                line=dict(color="#ff9800", width=2, dash="dot"),
                name="SMA 20",
                opacity=0.7
            ))

        if "sma_50" in df.columns:
            sma50 = df["sma_50"].values
            fig.add_trace(go.Scatter3d(
                x=days, y=sma50, z=np.zeros_like(days),
                mode="lines",
                line=dict(color="#2196f3", width=2, dash="dot"),
                name="SMA 50",
                opacity=0.7
            ))

        # Volume bars as 3D scatter on the floor
        vol_colors = ["#26a69a" if c >= o else "#ef5350"
                      for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Scatter3d(
            x=days, y=np.full_like(days, prices.min() * 0.95),
            z=vol_normalized,
            mode="markers",
            marker=dict(size=2, color=vol_colors, opacity=0.4),
            name="Volume",
            hoverinfo="skip"
        ))

        # BUY signals — glowing green diamonds
        buys = df[df[signal_column] == 1]
        if len(buys) > 0:
            buy_days = self._date_to_numeric(buys.index)
            buy_vol = buys["volume"].values / volumes.max() * prices.max() * 0.3
            fig.add_trace(go.Scatter3d(
                x=buy_days, y=buys["close"].values, z=buy_vol,
                mode="markers",
                marker=dict(
                    size=8, color="#00e676",
                    symbol="diamond",
                    line=dict(width=2, color="white")
                ),
                name="BUY ▲",
                text=[d.strftime("%Y-%m-%d") for d in buys.index],
                hovertemplate="BUY %{text}<br>$%{y:.2f}<extra></extra>"
            ))

        # SELL signals — glowing red diamonds
        sells = df[df[signal_column] == -1]
        if len(sells) > 0:
            sell_days = self._date_to_numeric(sells.index)
            sell_vol = sells["volume"].values / volumes.max() * prices.max() * 0.3
            fig.add_trace(go.Scatter3d(
                x=sell_days, y=sells["close"].values, z=sell_vol,
                mode="markers",
                marker=dict(
                    size=8, color="#ff1744",
                    symbol="diamond",
                    line=dict(width=2, color="white")
                ),
                name="SELL ▼",
                text=[d.strftime("%Y-%m-%d") for d in sells.index],
                hovertemplate="SELL %{text}<br>$%{y:.2f}<extra></extra>"
            ))

        # Bollinger Bands
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            fig.add_trace(go.Scatter3d(
                x=days, y=df["bb_upper"].values, z=np.zeros_like(days),
                mode="lines",
                line=dict(color="rgba(150,150,150,0.4)", width=1),
                name="BB Upper", showlegend=True
            ))
            fig.add_trace(go.Scatter3d(
                x=days, y=df["bb_lower"].values, z=np.zeros_like(days),
                mode="lines",
                line=dict(color="rgba(150,150,150,0.4)", width=1),
                name="BB Lower", showlegend=True
            ))

        # Animation frames — progressive reveal
        frames = self._make_animation_frames(
            days, prices, vol_normalized, "Price", "#00e676"
        )
        fig.frames = frames

        # Layout
        layout = self._get_dark_layout(
            f"🚀 {symbol} — 3D Price & Signals ({signal_column})", 1000
        )
        layout.update(
            scene=dict(
                xaxis=dict(title="Trading Days", color="#8b949e",
                          gridcolor="#21262d", backgroundcolor="#0d1117"),
                yaxis=dict(title="Price ($)", color="#8b949e",
                          gridcolor="#21262d", backgroundcolor="#0d1117"),
                zaxis=dict(title="Volume", color="#8b949e",
                          gridcolor="#21262d", backgroundcolor="#0d1117"),
                bgcolor="#0d1117",
                camera=dict(
                    eye=dict(x=1.8, y=-1.4, z=0.8),
                    up=dict(x=0, y=0, z=1)
                ),
            ),
            margin=dict(b=200),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.0, x=0.0, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="▶ Animate", method="animate",
                         args=[None, dict(frame=dict(duration=50, redraw=True),
                                         fromcurrent=True,
                                         transition=dict(duration=30))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode="immediate",
                                           transition=dict(duration=0))])
                ]
            )]
        )
        fig.update_layout(**layout)

        # Add reading instructions
        is_ml = "ml" in signal_column.lower()
        instructions = [
            "• The GREEN line traces the stock's closing price over time. Height on the Z-axis shows trading volume.",
            "• GREEN ♦ diamonds = BUY signals (strategy says buy here). RED ♦ diamonds = SELL signals.",
            "• ORANGE dotted line = SMA 20 (short-term trend). BLUE dotted line = SMA 50 (longer-term trend).",
            "• When SMA 20 crosses ABOVE SMA 50, it's a bullish sign. When it crosses BELOW, it's bearish.",
            "• Drag to rotate the 3D view. Scroll to zoom. Hover over any point for exact values.",
        ]
        if is_ml:
            instructions[1] = "• GREEN ♦ = ML model predicts price will GO UP. RED ♦ = ML model predicts price will GO DOWN."
            instructions.append("• ML signals are generated by the trained machine learning model, not traditional indicators.")
        self._add_instructions(fig, instructions)

        filepath = os.path.join(self.output_dir, f"{symbol}_{signal_column}_3d.html")
        fig.write_html(filepath, auto_play=False)
        logger.info(f"3D chart saved: {filepath}")
        fig.show()

    # ==========================================
    # CHART 2: Backtest Results with Risk Bar
    # ==========================================

    def plot_backtest_results(self, portfolio_df: pd.DataFrame,
                               trades_df: pd.DataFrame,
                               symbol: str, strategy: str,
                               initial_capital: float,
                               risk_level_results: dict = None) -> None:
        """
        3D Backtest chart with interactive risk bar slider.
        """
        has_risk_levels = risk_level_results and len(risk_level_results) > 1

        if has_risk_levels:
            self._plot_backtest_with_risk_bar(
                portfolio_df, trades_df, symbol, strategy,
                initial_capital, risk_level_results
            )
        else:
            self._plot_backtest_3d_single(
                portfolio_df, trades_df, symbol, strategy, initial_capital
            )

    def _plot_backtest_3d_single(self, portfolio_df, trades_df,
                                  symbol, strategy, initial_capital):
        """3D backtest chart without risk bar (single risk level)."""
        fig = go.Figure()

        days = self._date_to_numeric(portfolio_df.index)
        values = portfolio_df["total_value"].values
        returns = portfolio_df["daily_return"].values * 100

        fig.add_trace(go.Scatter3d(
            x=days, y=values, z=returns,
            mode="lines",
            line=dict(color="#00e676", width=4),
            name="Portfolio Value",
            text=[d.strftime("%Y-%m-%d") for d in portfolio_df.index],
            hovertemplate=(
                "Date: %{text}<br>"
                "Value: $%{y:,.0f}<br>"
                "Daily Ret: %{z:.2f}%<extra></extra>"
            )
        ))

        fig.add_trace(go.Scatter3d(
            x=[days[0], days[-1]], y=[initial_capital, initial_capital],
            z=[0, 0], mode="lines",
            line=dict(color="white", width=2, dash="dash"),
            name=f"Initial: ${initial_capital:,.0f}", opacity=0.5
        ))

        frames = self._make_animation_frames(days, values, returns, "Portfolio", "#00e676")
        fig.frames = frames

        layout = self._get_dark_layout(
            f"📊 {symbol} Backtest — {strategy} (3D)", 1000
        )
        layout.update(
            scene=dict(
                xaxis=dict(title="Trading Days", color="#8b949e",
                          gridcolor="#21262d", backgroundcolor="#0d1117"),
                yaxis=dict(title="Portfolio Value ($)", color="#8b949e",
                          gridcolor="#21262d", backgroundcolor="#0d1117"),
                zaxis=dict(title="Daily Return (%)", color="#8b949e",
                          gridcolor="#21262d", backgroundcolor="#0d1117"),
                bgcolor="#0d1117",
                camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
            ),
            margin=dict(b=200),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.0, x=0.0, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="▶ Animate", method="animate",
                         args=[None, dict(frame=dict(duration=50, redraw=True),
                                         fromcurrent=True,
                                         transition=dict(duration=30))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode="immediate")])
                ]
            )]
        )
        fig.update_layout(**layout)

        instructions = [
            "• The GREEN line shows your portfolio's total value over time, starting from initial capital.",
            "• The Z-axis (depth) shows daily returns — spikes forward = good day, spikes back = bad day.",
            "• The WHITE dashed line marks your starting capital — above it means profit, below means loss.",
            "• Drag to rotate the 3D view. Scroll to zoom. Hover over any point for exact values.",
        ]
        self._add_instructions(fig, instructions)

        filepath = os.path.join(self.output_dir, f"{symbol}_{strategy}_backtest_3d.html")
        fig.write_html(filepath, auto_play=False)
        logger.info(f"3D Backtest chart saved: {filepath}")
        fig.show()

    def _plot_backtest_with_risk_bar(self, portfolio_df, trades_df,
                                      symbol, strategy, initial_capital,
                                      risk_level_results):
        """
        3D backtest chart with interactive risk bar slider.
        Uses trace visibility toggling (not animation frames) for reliable slider behavior.
        """
        sorted_levels = sorted(risk_level_results.keys())

        # Color gradient: green → yellow → red
        def risk_color(mult):
            ratio = (mult - sorted_levels[0]) / (sorted_levels[-1] - sorted_levels[0]) \
                if sorted_levels[-1] > sorted_levels[0] else 0
            r = int(min(255, 80 + ratio * 175))
            g = int(max(0, 230 - ratio * 200))
            b = int(max(0, 118 - ratio * 118))
            return f"rgb({r},{g},{b})"

        fig = go.Figure()

        # For each risk level, add 3 traces: equity line, initial capital line, final marker
        # Total traces per level = 3
        traces_per_level = 3
        default_mult = 1.0 if 1.0 in risk_level_results else sorted_levels[len(sorted_levels) // 2]

        for level_idx, mult in enumerate(sorted_levels):
            results = risk_level_results[mult]
            pdf = results["portfolio_history"]
            days = self._date_to_numeric(pdf.index)
            values = pdf["total_value"].values
            returns_arr = pdf["daily_return"].values * 100
            color = risk_color(mult)

            pos_pct = results.get("position_size_pct", 0)
            risk_pct = results.get("risk_per_trade_pct", 0)
            final_val = results.get("final_value", 0)
            total_ret = results.get("total_return", 0) * 100

            is_default = (mult == default_mult)

            # Trace 1: Equity curve
            fig.add_trace(go.Scatter3d(
                x=days, y=values, z=returns_arr,
                mode="lines",
                line=dict(color=color, width=5),
                name=f"Risk {mult:.2f}× | Pos: {pos_pct:.0f}%",
                visible=is_default,
                text=[d.strftime("%Y-%m-%d") for d in pdf.index],
                hovertemplate=(
                    f"<b>Risk: {mult:.2f}× | Pos Size: {pos_pct:.1f}% | Risk/Trade: {risk_pct:.1f}%</b><br>"
                    "Date: %{text}<br>"
                    "Value: $%{y:,.0f}<br>"
                    "Daily Ret: %{z:.2f}%<extra></extra>"
                )
            ))

            # Trace 2: Initial capital reference line
            fig.add_trace(go.Scatter3d(
                x=[days[0], days[-1]],
                y=[initial_capital, initial_capital],
                z=[0, 0],
                mode="lines",
                line=dict(color="white", width=2, dash="dash"),
                name=f"Initial: ${initial_capital:,.0f}",
                visible=is_default, opacity=0.4, showlegend=(level_idx == 0)
            ))

            # Trace 3: Final value marker with label
            marker_color = "#00e676" if total_ret >= 0 else "#ff1744"
            fig.add_trace(go.Scatter3d(
                x=[days[-1]], y=[final_val], z=[0],
                mode="markers+text",
                marker=dict(size=8, color=marker_color,
                           line=dict(width=2, color="white")),
                text=[f"${final_val:,.0f} ({total_ret:+.1f}%)"],
                textposition="top center",
                textfont=dict(size=13, color=color),
                name=f"Final: ${final_val:,.0f}",
                visible=is_default, showlegend=False
            ))

        # Build slider steps — each step toggles visibility
        slider_steps = []
        for step_idx, mult in enumerate(sorted_levels):
            # Build visibility array: True for 3 traces of this level, False for all others
            visibility = [False] * (len(sorted_levels) * traces_per_level)
            base = step_idx * traces_per_level
            visibility[base] = True      # equity line
            visibility[base + 1] = True  # initial capital
            visibility[base + 2] = True  # final marker

            results = risk_level_results[mult]
            pos_pct = results.get("position_size_pct", 0)
            risk_pct = results.get("risk_per_trade_pct", 0)
            final_val = results.get("final_value", 0)
            total_ret = results.get("total_return", 0) * 100
            color = risk_color(mult)

            label = f"{mult:.2f}×"
            if mult == default_mult:
                label += " ◄"

            slider_steps.append(dict(
                args=[{"visible": visibility},
                      {"title.text": (
                          f"📊 {symbol} Backtest — {strategy} | "
                          f"Risk: {mult:.2f}× | Pos Size: {pos_pct:.0f}% | "
                          f"Final: ${final_val:,.0f} ({total_ret:+.1f}%)"
                      )}],
                label=label,
                method="update"
            ))

        layout = self._get_dark_layout(
            f"📊 {symbol} Backtest — {strategy} | Risk: {default_mult:.2f}× (drag slider below)", 1050
        )
        layout.update(
            scene=dict(
                xaxis=dict(title="Trading Days", color="#8b949e",
                          gridcolor="#21262d", backgroundcolor="#0d1117"),
                yaxis=dict(title="Portfolio Value ($)", color="#8b949e",
                          gridcolor="#21262d", backgroundcolor="#0d1117"),
                zaxis=dict(title="Daily Return (%)", color="#8b949e",
                          gridcolor="#21262d", backgroundcolor="#0d1117"),
                bgcolor="#0d1117",
                camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
            ),
            margin=dict(b=280),
            sliders=[dict(
                active=sorted_levels.index(default_mult),
                currentvalue=dict(
                    prefix="⚡ Risk Level: ",
                    visible=True,
                    xanchor="center",
                    font=dict(size=16, color="#e0e0e0")
                ),
                pad=dict(b=10, t=40),
                len=0.9,
                x=0.05,
                xanchor="left",
                y=0,
                yanchor="top",
                steps=slider_steps,
                bgcolor="#21262d",
                activebgcolor="#58a6ff",
                bordercolor="#30363d",
                ticklen=5,
                font=dict(size=10, color="#c9d1d9"),
                transition=dict(duration=300)
            )],
        )
        fig.update_layout(**layout)

        # Add reading instructions
        instructions = [
            "◀ LOW RISK ━━━━━━━━━━━━━━━━━━━━ RISK BAR ━━━━━━━━━━━━━━━━━━━━ HIGH RISK ▶",
            "• DRAG THE SLIDER above to change risk level. Each level re-runs the backtest with different capital allocation.",
            "• LOW risk (0.25×) = 5% of capital per trade → smoother, smaller returns. Current config is 1.0× (marked with ◄).",
            "• HIGH risk (3.0×) = 60% of capital per trade → more volatile, bigger wins AND bigger losses.",
            "• The equity line color shifts GREEN → YELLOW → RED as risk increases. Watch how the curve shape changes!",
            "• The Z-axis shows daily returns — taller spikes at high risk = more volatile daily swings.",
            "• Drag to rotate the 3D view. Scroll to zoom. Hover over the line for exact values.",
        ]
        self._add_instructions(fig, instructions, y_start=-0.05)

        filepath = os.path.join(self.output_dir,
                               f"{symbol}_{strategy}_backtest_risk_bar_3d.html")
        fig.write_html(filepath, auto_play=False)
        logger.info(f"3D Risk Bar Backtest chart saved: {filepath}")
        fig.show()

    # ==========================================
    # CHART 3: Equity Comparison (3D Ribbon)
    # ==========================================

    def plot_equity_comparison(self, all_results: dict, symbol: str) -> None:
        """
        3D ribbon chart comparing strategy equity curves.
        Each strategy is a ribbon at a different Z-depth.
        """
        fig = go.Figure()

        colors = ["#00e676", "#2196f3", "#ff9800", "#e91e63", "#9c27b0",
                  "#00bcd4", "#ffeb3b", "#4caf50"]

        initial_capital = 100000
        strategy_names = list(all_results.keys())

        for i, (strategy_name, results) in enumerate(all_results.items()):
            pdf = results["portfolio_history"]
            color = colors[i % len(colors)]
            initial_capital = results["initial_capital"]

            days = self._date_to_numeric(pdf.index)
            values = pdf["total_value"].values
            z_depth = np.full_like(days, float(i))

            # Main ribbon line
            fig.add_trace(go.Scatter3d(
                x=days, y=values, z=z_depth,
                mode="lines",
                line=dict(color=color, width=5),
                name=strategy_name,
                text=[d.strftime("%Y-%m-%d") for d in pdf.index],
                hovertemplate=(
                    f"<b>{strategy_name}</b><br>"
                    "Date: %{text}<br>"
                    "Value: $%{y:,.0f}<extra></extra>"
                )
            ))

            # Floor line for ribbon effect
            fig.add_trace(go.Scatter3d(
                x=days, y=np.full_like(values, initial_capital), z=z_depth,
                mode="lines",
                line=dict(color=color, width=1),
                opacity=0.15, showlegend=False, hoverinfo="skip"
            ))

            # Final value marker
            final_val = values[-1]
            total_ret = (final_val - initial_capital) / initial_capital * 100
            marker_color = "#00e676" if total_ret >= 0 else "#ff1744"
            fig.add_trace(go.Scatter3d(
                x=[days[-1]], y=[final_val], z=[float(i)],
                mode="markers+text",
                marker=dict(size=7, color=marker_color,
                           line=dict(width=2, color="white")),
                text=[f"${final_val:,.0f} ({total_ret:+.1f}%)"],
                textposition="middle right",
                textfont=dict(size=10, color=color),
                showlegend=False
            ))

        # Initial capital reference
        max_days = max(
            self._date_to_numeric(r["portfolio_history"].index)[-1]
            for r in all_results.values()
        )
        fig.add_trace(go.Scatter3d(
            x=[0, max_days], y=[initial_capital, initial_capital], z=[0, 0],
            mode="lines",
            line=dict(color="white", width=2, dash="dash"),
            name=f"Initial: ${initial_capital:,.0f}", opacity=0.3
        ))

        # Animation
        first_results = list(all_results.values())[0]
        first_pdf = first_results["portfolio_history"]
        first_days = self._date_to_numeric(first_pdf.index)
        first_vals = first_pdf["total_value"].values
        frames = self._make_animation_frames(
            first_days, first_vals, np.zeros_like(first_days),
            strategy_names[0], colors[0]
        )
        fig.frames = frames

        layout = self._get_dark_layout(
            f"🏆 Strategy Equity Comparison — {symbol} (3D Ribbon)", 1000
        )
        layout.update(
            scene=dict(
                xaxis=dict(title="Trading Days", color="#8b949e",
                          gridcolor="#21262d", backgroundcolor="#0d1117"),
                yaxis=dict(title="Portfolio Value ($)", color="#8b949e",
                          gridcolor="#21262d", backgroundcolor="#0d1117"),
                zaxis=dict(
                    title="Strategy",
                    color="#8b949e",
                    gridcolor="#21262d",
                    backgroundcolor="#0d1117",
                    tickvals=list(range(len(strategy_names))),
                    ticktext=strategy_names
                ),
                bgcolor="#0d1117",
                camera=dict(eye=dict(x=1.5, y=-1.8, z=1.0)),
            ),
            margin=dict(b=200),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.0, x=0.0, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="▶ Animate", method="animate",
                         args=[None, dict(frame=dict(duration=50, redraw=True),
                                         fromcurrent=True,
                                         transition=dict(duration=30))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode="immediate")])
                ]
            )]
        )
        fig.update_layout(**layout)

        strategy_list = ", ".join(strategy_names)
        instructions = [
            f"• Each colored ribbon represents a different trading strategy: {strategy_list}.",
            "• The Y-axis shows portfolio value ($). Lines ABOVE the white dashed line = profit.",
            "• The Z-axis separates strategies into lanes so you can compare them side by side.",
            "• The strategy with the HIGHEST final value (shown at the right end) performed best.",
            "• Look for smooth upward curves — jagged lines indicate high volatility / inconsistency.",
            "• Drag to rotate. Try viewing from the side to compare final values directly.",
        ]
        self._add_instructions(fig, instructions)

        filepath = os.path.join(self.output_dir, f"{symbol}_equity_comparison_3d.html")
        fig.write_html(filepath, auto_play=False)
        logger.info(f"3D Equity comparison saved: {filepath}")
        fig.show()

    # ==========================================
    # CHART 5: MACD Analysis (3D)
    # ==========================================

    def plot_macd(self, df: pd.DataFrame, symbol: str) -> None:
        """
        3D MACD chart with animated histogram bars rising from the plane.
        """
        fig = go.Figure()

        days = self._date_to_numeric(df.index)
        prices = df["close"].values
        date_labels = [d.strftime("%Y-%m-%d") for d in df.index]

        # Price line (back layer z=0)
        fig.add_trace(go.Scatter3d(
            x=days, y=prices, z=np.zeros_like(days),
            mode="lines",
            line=dict(color="white", width=2),
            name="Close Price",
            text=date_labels,
            hovertemplate="Date: %{text}<br>Price: $%{y:.2f}<extra></extra>"
        ))

        # MACD components
        macd_line = df["macd_line"].values
        macd_signal_line = df["macd_signal"].values
        macd_hist = df["macd_histogram"].values

        # Normalize MACD to price scale
        macd_scale = prices.max() * 0.2
        macd_offset = prices.min() * 0.8
        macd_abs_max = np.abs(macd_line).max() + 1e-8
        signal_abs_max = np.abs(macd_signal_line).max() + 1e-8
        hist_abs_max = np.abs(macd_hist).max() + 1e-8

        macd_scaled = macd_offset + (macd_line / macd_abs_max) * macd_scale
        signal_scaled = macd_offset + (macd_signal_line / signal_abs_max) * macd_scale
        hist_scaled = (macd_hist / hist_abs_max) * macd_scale

        # MACD Line (z=2)
        fig.add_trace(go.Scatter3d(
            x=days, y=macd_scaled, z=np.ones_like(days) * 2,
            mode="lines",
            line=dict(color="#2196f3", width=3),
            name="MACD Line",
            text=date_labels,
            hovertemplate="Date: %{text}<br>MACD: %{customdata:.4f}<extra></extra>",
            customdata=macd_line
        ))

        # Signal Line (z=2)
        fig.add_trace(go.Scatter3d(
            x=days, y=signal_scaled, z=np.ones_like(days) * 2,
            mode="lines",
            line=dict(color="#ff9800", width=3),
            name="Signal Line",
            text=date_labels,
            hovertemplate="Date: %{text}<br>Signal: %{customdata:.4f}<extra></extra>",
            customdata=macd_signal_line
        ))

        # 3D histogram bars (z=4)
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_hist]
        sample_step = max(1, len(days) // 200)
        for idx in range(0, len(days), sample_step):
            fig.add_trace(go.Scatter3d(
                x=[days[idx], days[idx]],
                y=[macd_offset, macd_offset + hist_scaled[idx]],
                z=[4, 4],
                mode="lines",
                line=dict(color=hist_colors[idx], width=3),
                showlegend=False, hoverinfo="skip"
            ))

        # Histogram hover scatter (z=4)
        fig.add_trace(go.Scatter3d(
            x=days, y=macd_offset + hist_scaled,
            z=np.ones_like(days) * 4,
            mode="markers",
            marker=dict(size=2, color=hist_colors, opacity=0.7),
            name="Histogram",
            text=date_labels,
            hovertemplate="Date: %{text}<br>Histogram: %{customdata:.4f}<extra></extra>",
            customdata=macd_hist
        ))

        # Animation
        frames = self._make_animation_frames(
            days, prices, np.zeros_like(days), "Close Price", "white"
        )
        fig.frames = frames

        layout = self._get_dark_layout(
            f"📉 MACD Analysis — {symbol} (3D)", 1000
        )
        layout.update(
            scene=dict(
                xaxis=dict(title="Trading Days", color="#8b949e",
                          gridcolor="#21262d", backgroundcolor="#0d1117"),
                yaxis=dict(title="Price / MACD", color="#8b949e",
                          gridcolor="#21262d", backgroundcolor="#0d1117"),
                zaxis=dict(
                    title="Layer",
                    color="#8b949e",
                    gridcolor="#21262d",
                    backgroundcolor="#0d1117",
                    tickvals=[0, 2, 4],
                    ticktext=["Price", "MACD", "Histogram"]
                ),
                bgcolor="#0d1117",
                camera=dict(eye=dict(x=1.7, y=-1.5, z=0.7)),
            ),
            margin=dict(b=220),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.0, x=0.0, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="▶ Animate", method="animate",
                         args=[None, dict(frame=dict(duration=50, redraw=True),
                                         fromcurrent=True,
                                         transition=dict(duration=30))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode="immediate")])
                ]
            )]
        )
        fig.update_layout(**layout)

        instructions = [
            "• This chart has 3 LAYERS (Z-axis): Price (z=0), MACD Lines (z=2), and Histogram (z=4).",
            "• WHITE line = stock's closing price. This is the raw price movement your strategy trades on.",
            "• BLUE line (MACD) vs ORANGE line (Signal): when BLUE crosses ABOVE orange → bullish momentum.",
            "• When BLUE crosses BELOW orange → bearish momentum. These crossovers generate trade signals.",
            "• GREEN histogram bars = positive momentum (bullish). RED bars = negative momentum (bearish).",
            "• TALLER bars = STRONGER momentum. Watch for bars shrinking → momentum is fading, reversal may come.",
            "• Rotate the 3D view to see each layer independently. Try looking from the side (z-axis view).",
        ]
        self._add_instructions(fig, instructions)

        filepath = os.path.join(self.output_dir, f"{symbol}_macd_3d.html")
        fig.write_html(filepath, auto_play=False)
        logger.info(f"3D MACD chart saved: {filepath}")
        fig.show()

    # ==========================================
    # LEGACY: Strategy Comparison (kept for compat)
    # ==========================================

    def plot_strategy_comparison(self, df: pd.DataFrame, symbol: str) -> None:
        """Plot all strategy signals on one chart (3D scatter). Kept for compatibility."""
        signal_cols = [col for col in df.columns
                      if "signal" in col.lower() and df[col].abs().sum() > 0]

        fig = go.Figure()
        days = self._date_to_numeric(df.index)
        colors = ["#00e676", "#2196f3", "#ff9800", "#e91e63", "#9c27b0"]

        for i, col in enumerate(signal_cols):
            color = colors[i % len(colors)]
            signals = df[col].values
            fig.add_trace(go.Scatter3d(
                x=days, y=df["close"].values, z=signals.astype(float),
                mode="markers",
                marker=dict(
                    size=3,
                    color=[color if s != 0 else "rgba(0,0,0,0)" for s in signals],
                    opacity=0.7
                ),
                name=col
            ))

        layout = self._get_dark_layout(
            f"Strategy Comparison — {symbol} (3D)", 800
        )
        layout.update(
            scene=dict(
                xaxis=dict(title="Trading Days", backgroundcolor="#0d1117"),
                yaxis=dict(title="Price ($)", backgroundcolor="#0d1117"),
                zaxis=dict(title="Signal", backgroundcolor="#0d1117"),
                bgcolor="#0d1117",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
            )
        )
        fig.update_layout(**layout)

        filepath = os.path.join(self.output_dir, f"{symbol}_comparison_3d.html")
        fig.write_html(filepath, auto_play=False)
        logger.info(f"3D Comparison chart saved: {filepath}")
        fig.show()