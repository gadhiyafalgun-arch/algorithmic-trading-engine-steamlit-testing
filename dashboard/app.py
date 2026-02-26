"""
Algorithmic Trading Engine — Interactive Dashboard
====================================================
Built with Streamlit.

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import DataPipeline
from src.indicators import TechnicalIndicators
from src.strategy import TradingStrategy
from src.backtester import Backtester
from src.performance import PerformanceAnalyzer
from src.risk_manager import RiskManager
from models.feature_engineer import FeatureEngineer
from models.predict import MLPredictor

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Algo Trading Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #1f4068;
        margin: 5px 0;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #00e676;
    }
    
    .metric-label {
        font-size: 14px;
        color: #8892b0;
    }
    
    .positive { color: #00e676; }
    .negative { color: #ff1744; }
    
    /* Header */
    .main-header {
        font-size: 42px;
        font-weight: bold;
        background: linear-gradient(90deg, #00e676, #2196f3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 10px 0;
    }
    
    .sub-header {
        text-align: center;
        color: #8892b0;
        font-size: 16px;
        margin-bottom: 30px;
    }
    
    /* Divider */
    .custom-divider {
        border-top: 1px solid #1f4068;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# CACHED DATA LOADING
# ==========================================
@st.cache_data(ttl=300)
def load_data():
    """Load and process all data."""
    pipeline = DataPipeline()
    data = pipeline.run_pipeline()

    indicators = TechnicalIndicators()
    strategy = TradingStrategy()
    feature_eng = FeatureEngineer()

    processed = {}
    for symbol, df in data.items():
        df = indicators.add_all_indicators(df)
        df = strategy.apply_all_strategies(df)
        df = feature_eng.create_all_features(df)
        processed[symbol] = df

    return processed


@st.cache_data(ttl=300)
def run_backtest(df_dict, symbol, signal_column):
    """Run backtest for a specific stock and strategy."""
    df = df_dict[symbol]
    backtester = Backtester()
    results = backtester.run(df, signal_column=signal_column, symbol=symbol)
    return results


@st.cache_data(ttl=300)
def get_performance_metrics(portfolio_history, trades, initial_capital):
    """Calculate performance metrics."""
    perf = PerformanceAnalyzer()
    metrics = perf.calculate_all_metrics(portfolio_history, trades, initial_capital)
    return metrics


# ==========================================
# CHART FUNCTIONS
# ==========================================
def create_candlestick_chart(df, symbol, signal_column="combined_signal"):
    """Create interactive candlestick chart with signals."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(
            f"{symbol} Price Action",
            "Volume",
            "RSI (14)",
            "MACD"
        )
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )

    # Moving Averages
    if "sma_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["sma_20"], name="SMA 20",
                      line=dict(color="orange", width=1)),
            row=1, col=1
        )
    if "sma_50" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50",
                      line=dict(color="#2196f3", width=1)),
            row=1, col=1
        )

    # Bollinger Bands
    if "bb_upper" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
                      line=dict(color="gray", width=0.5, dash="dash"), opacity=0.4),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
                      line=dict(color="gray", width=0.5, dash="dash"),
                      fill="tonexty", fillcolor="rgba(128,128,128,0.05)", opacity=0.4),
            row=1, col=1
        )

    # BUY/SELL Signals
    if signal_column in df.columns:
        buys = df[df[signal_column] == 1]
        sells = df[df[signal_column] == -1]

        if len(buys) > 0:
            fig.add_trace(
                go.Scatter(x=buys.index, y=buys["close"], mode="markers",
                          name="BUY", marker=dict(symbol="triangle-up", size=12,
                          color="#00e676", line=dict(width=1, color="darkgreen"))),
                row=1, col=1
            )
        if len(sells) > 0:
            fig.add_trace(
                go.Scatter(x=sells.index, y=sells["close"], mode="markers",
                          name="SELL", marker=dict(symbol="triangle-down", size=12,
                          color="#ff1744", line=dict(width=1, color="darkred"))),
                row=1, col=1
            )

    # Volume
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume",
               marker_color=colors, opacity=0.5),
        row=2, col=1
    )

    # RSI
    if "rsi_14" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["rsi_14"], name="RSI",
                      line=dict(color="#9c27b0", width=1.5)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

    # MACD
    if "macd_line" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd_line"], name="MACD",
                      line=dict(color="#2196f3", width=1.5)),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                      line=dict(color="#ff9800", width=1.5)),
            row=4, col=1
        )
        macd_colors = ["#26a69a" if v >= 0 else "#ef5350"
                       for v in df["macd_histogram"]]
        fig.add_trace(
            go.Bar(x=df.index, y=df["macd_histogram"], name="Histogram",
                   marker_color=macd_colors, opacity=0.5),
            row=4, col=1
        )

    fig.update_layout(
        template="plotly_dark",
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig


def create_equity_chart(portfolio_df, initial_capital):
    """Create equity curve chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Portfolio Value", "Drawdown")
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index, y=portfolio_df["total_value"],
            name="Portfolio", line=dict(color="#00e676", width=2),
            fill="tozeroy", fillcolor="rgba(0, 230, 118, 0.05)"
        ),
        row=1, col=1
    )

    fig.add_hline(y=initial_capital, line_dash="dash", line_color="white",
                  opacity=0.3, row=1, col=1)

    # Drawdown
    cumulative = (1 + portfolio_df["daily_return"]).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100

    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index, y=drawdown,
            name="Drawdown", line=dict(color="#ef5350", width=1),
            fill="tozeroy", fillcolor="rgba(239, 83, 80, 0.1)"
        ),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    return fig


def create_returns_distribution(portfolio_df):
    """Create returns distribution histogram."""
    returns = portfolio_df["daily_return"].dropna() * 100

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns, nbinsx=50, name="Daily Returns",
            marker_color="#2196f3", opacity=0.7
        )
    )

    # Add mean line
    mean_ret = returns.mean()
    fig.add_vline(x=mean_ret, line_dash="dash", line_color="#00e676",
                  annotation_text=f"Mean: {mean_ret:.3f}%")
    fig.add_vline(x=0, line_dash="solid", line_color="white", opacity=0.3)

    fig.update_layout(
        template="plotly_dark",
        height=350,
        title="Daily Returns Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


def create_monthly_returns_heatmap(portfolio_df):
    """Create monthly returns heatmap."""
    portfolio_df = portfolio_df.copy()
    portfolio_df["month"] = portfolio_df.index.month
    portfolio_df["year"] = portfolio_df.index.year

    monthly = portfolio_df.groupby(["year", "month"])["daily_return"].apply(
        lambda x: (1 + x).prod() - 1
    ).unstack() * 100

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(data=go.Heatmap(
        z=monthly.values,
        x=month_names[:monthly.shape[1]],
        y=monthly.index.astype(str),
        colorscale=[[0, "#ef5350"], [0.5, "#1a1a2e"], [1, "#00e676"]],
        text=np.round(monthly.values, 2),
        texttemplate="%{text:.1f}%",
        textfont={"size": 11},
        colorbar=dict(title="Return %")
    ))

    fig.update_layout(
        template="plotly_dark",
        height=300,
        title="Monthly Returns Heatmap (%)",
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


def create_strategy_comparison_chart(all_results):
    """Create strategy comparison bar chart."""
    strategies = list(all_results.keys())
    returns = [all_results[s]["total_return"] * 100 for s in strategies]
    colors = ["#00e676" if r > 0 else "#ef5350" for r in returns]

    fig = go.Figure(data=[
        go.Bar(x=strategies, y=returns, marker_color=colors, opacity=0.8,
               text=[f"{r:.1f}%" for r in returns], textposition="outside")
    ])

    fig.update_layout(
        template="plotly_dark",
        height=350,
        title="Strategy Returns Comparison",
        yaxis_title="Return (%)",
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


# ==========================================
# MAIN DASHBOARD
# ==========================================
def main():
    """Main dashboard function."""

    # Header
    st.markdown('<div class="main-header">🤖 Algorithmic Trading Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ML-Powered Quantitative Trading System | Built with Python</div>', unsafe_allow_html=True)

    # ==========================================
    # SIDEBAR
    # ==========================================
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        # Load data
        with st.spinner("Loading market data..."):
            try:
                data = load_data()
                symbols = list(data.keys())
                st.success(f"✅ Loaded {len(symbols)} stocks")
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return

        # Stock selection
        selected_symbol = st.selectbox(
            "📊 Select Stock",
            symbols,
            index=0
        )

        # Strategy selection
        strategy_options = [
            "sma_signal",
            "rsi_signal",
            "macd_trade_signal",
            "bb_signal",
            "combined_signal"
        ]

        # Check if ML signal exists
        if "ml_signal" in data[selected_symbol].columns:
            if data[selected_symbol]["ml_signal"].abs().sum() > 0:
                strategy_options.append("ml_signal")

        selected_strategy = st.selectbox(
            "🎯 Select Strategy",
            strategy_options,
            index=strategy_options.index("combined_signal")
        )

        st.markdown("---")
        st.markdown("## 📅 Date Range")

        df = data[selected_symbol]
        min_date = df.index.min().date()
        max_date = df.index.max().date()

        date_range = st.date_input(
            "Select Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[start_date:end_date]

        st.markdown("---")
        st.markdown("## 💰 Backtest Settings")

        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000
        )

        st.markdown("---")
        st.markdown("### 📊 Stock Info")
        latest = df.iloc[-1]
        st.metric("Current Price", f"${latest['close']:.2f}")

        if "daily_return" in df.columns:
            daily_ret = latest["daily_return"]
            st.metric("Daily Return", f"{daily_ret*100:.2f}%",
                      delta=f"{daily_ret*100:.2f}%")

        if "rsi_14" in df.columns:
            rsi_val = latest["rsi_14"]
            rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Normal"
            st.metric("RSI (14)", f"{rsi_val:.1f}", delta=rsi_status)

    # ==========================================
    # MAIN CONTENT
    # ==========================================

    # Run backtest
    with st.spinner("Running backtest..."):
        try:
            backtester = Backtester()
            results = backtester.run(df, signal_column=selected_strategy, symbol=selected_symbol)

            if results and not results["portfolio_history"].empty:
                metrics = get_performance_metrics(
                    results["portfolio_history"],
                    results["trades"],
                    results["initial_capital"]
                )
            else:
                metrics = None
                results = None
        except Exception as e:
            st.error(f"Backtest error: {e}")
            metrics = None
            results = None

    # ==========================================
    # KEY METRICS ROW
    # ==========================================
    st.markdown("### 📊 Key Performance Metrics")

    if metrics:
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            total_ret = metrics["total_return_pct"]
            st.metric(
                "Total Return",
                f"{total_ret:.2f}%",
                delta=f"{total_ret:.2f}%"
            )

        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.3f}"
            )

        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics['max_drawdown_pct']:.2f}%"
            )

        with col4:
            st.metric(
                "Win Rate",
                f"{metrics['win_rate']*100:.1f}%"
            )

        with col5:
            st.metric(
                "Total Trades",
                f"{metrics['total_trades']}"
            )

        with col6:
            st.metric(
                "Profit Factor",
                f"{metrics['profit_factor']:.2f}"
            )
    else:
        st.warning("No backtest results available. Try a different strategy or date range.")

    st.markdown("---")

    # ==========================================
    # TABS
    # ==========================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price Chart",
        "💰 Backtest Results",
        "📊 Strategy Comparison",
        "🛡️ Risk Analysis",
        "📋 Trade Log"
    ])

    # --- TAB 1: Price Chart ---
    with tab1:
        st.markdown(f"### {selected_symbol} — Price Action with {selected_strategy}")
        fig_price = create_candlestick_chart(df, selected_symbol, selected_strategy)
        st.plotly_chart(fig_price, use_container_width=True)

    # --- TAB 2: Backtest Results ---
    with tab2:
        if results and metrics:
            st.markdown("### Portfolio Performance")

            col1, col2 = st.columns([2, 1])

            with col1:
                # Equity curve
                fig_equity = create_equity_chart(
                    results["portfolio_history"],
                    results["initial_capital"]
                )
                st.plotly_chart(fig_equity, use_container_width=True)

                # Monthly returns
                fig_monthly = create_monthly_returns_heatmap(results["portfolio_history"])
                st.plotly_chart(fig_monthly, use_container_width=True)

            with col2:
                # Detailed metrics
                st.markdown("#### 📊 Detailed Metrics")

                metrics_display = {
                    "💰 Initial Capital": f"${metrics['initial_capital']:,.2f}",
                    "💵 Final Value": f"${metrics['final_value']:,.2f}",
                    "📈 Total Return": f"{metrics['total_return_pct']:.2f}%",
                    "📅 Annual Return": f"{metrics['annualized_return']*100:.2f}%",
                    "📊 Sharpe Ratio": f"{metrics['sharpe_ratio']:.3f}",
                    "📉 Sortino Ratio": f"{metrics['sortino_ratio']:.3f}",
                    "🔻 Max Drawdown": f"{metrics['max_drawdown_pct']:.2f}%",
                    "📏 Calmar Ratio": f"{metrics['calmar_ratio']:.3f}",
                    "📊 Volatility (Annual)": f"{metrics['volatility_annual']*100:.2f}%",
                    "⚠️ VaR (95%)": f"{metrics['var_95']*100:.3f}%",
                    "🎯 Win Rate": f"{metrics['win_rate']*100:.1f}%",
                    "💪 Profit Factor": f"{metrics['profit_factor']:.2f}",
                    "📊 Total Trades": f"{metrics['total_trades']}",
                    "✅ Winning Trades": f"{metrics['winning_trades']}",
                    "❌ Losing Trades": f"{metrics['losing_trades']}",
                    "💵 Avg Win": f"${metrics.get('avg_win', 0):,.2f}",
                    "💸 Avg Loss": f"${metrics.get('avg_loss', 0):,.2f}",
                    "🏆 Best Trade": f"${metrics.get('best_trade', 0):,.2f}",
                    "😢 Worst Trade": f"${metrics.get('worst_trade', 0):,.2f}",
                }

                for label, value in metrics_display.items():
                    st.markdown(f"**{label}:** {value}")

                # Returns distribution
                fig_dist = create_returns_distribution(results["portfolio_history"])
                st.plotly_chart(fig_dist, use_container_width=True)

        else:
            st.info("Run a backtest to see results.")

    # --- TAB 3: Strategy Comparison ---
    with tab3:
        st.markdown("### 🔄 Strategy Comparison")
        st.markdown("Comparing all strategies on the same stock and time period.")

        with st.spinner("Running all strategies..."):
            all_strategies = [
                "sma_signal", "rsi_signal", "macd_trade_signal",
                "bb_signal", "combined_signal"
            ]

            if "ml_signal" in df.columns and df["ml_signal"].abs().sum() > 0:
                all_strategies.append("ml_signal")

            all_results = {}
            comparison_data = []

            for strat in all_strategies:
                if strat in df.columns and df[strat].abs().sum() > 0:
                    try:
                        bt = Backtester()
                        res = bt.run(df, signal_column=strat, symbol=selected_symbol)

                        if res and not res["portfolio_history"].empty:
                            perf = PerformanceAnalyzer()
                            met = perf.calculate_all_metrics(
                                res["portfolio_history"],
                                res["trades"],
                                res["initial_capital"]
                            )
                            all_results[strat] = met

                            comparison_data.append({
                                "Strategy": strat,
                                "Return (%)": f"{met['total_return_pct']:.2f}",
                                "Sharpe": f"{met['sharpe_ratio']:.3f}",
                                "Sortino": f"{met['sortino_ratio']:.3f}",
                                "Max DD (%)": f"{met['max_drawdown_pct']:.2f}",
                                "Win Rate (%)": f"{met['win_rate']*100:.1f}",
                                "Profit Factor": f"{met['profit_factor']:.2f}",
                                "Trades": met['total_trades']
                            })
                    except Exception:
                        pass

            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                # Bar chart
                if all_results:
                    fig_comp = create_strategy_comparison_chart(
                        {k: {"total_return": v["total_return"]} for k, v in all_results.items()}
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("No strategy results available.")

    # --- TAB 4: Risk Analysis ---
    with tab4:
        st.markdown("### 🛡️ Risk Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Risk Score
            st.markdown("#### Stock Risk Scores")
            risk_mgr = RiskManager()

            risk_data = []
            for symbol in symbols:
                risk = risk_mgr.calculate_risk_score(data[symbol])
                risk_data.append({
                    "Stock": symbol,
                    "Risk Score": risk["risk_score"],
                    "Risk Level": risk["risk_level"],
                    "Volatility": risk["component_scores"].get("volatility", 0),
                    "Drawdown": risk["component_scores"].get("drawdown", 0),
                    "Trend": risk["component_scores"].get("trend", 0)
                })

            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True, hide_index=True)

        with col2:
            # Correlation Matrix
            st.markdown("#### Correlation Matrix")
            corr_matrix = risk_mgr.calculate_correlation_matrix(data)

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale=[[0, "#ef5350"], [0.5, "#1a1a2e"], [1, "#00e676"]],
                text=np.round(corr_matrix.values, 3),
                texttemplate="%{text:.3f}",
                textfont={"size": 12},
                zmin=-1, zmax=1
            ))

            fig_corr.update_layout(
                template="plotly_dark",
                height=400,
                margin=dict(l=50, r=50, t=20, b=50)
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # Correlation warnings
        corr_risk = risk_mgr.check_correlation_risk(corr_matrix)
        if corr_risk["high_correlation_pairs"]:
            st.warning("⚠️ High Correlation Detected!")
            for pair in corr_risk["high_correlation_pairs"]:
                st.markdown(f"- **{pair['stock1']}** ↔ **{pair['stock2']}**: {pair['correlation']:.3f}")
        else:
            st.success("✅ Portfolio is well diversified!")

    # --- TAB 5: Trade Log ---
    with tab5:
        st.markdown("### 📋 Trade History")

        if results and not results["trades"].empty:
            trades_df = results["trades"].copy()

            # Color PnL
            st.dataframe(
                trades_df.style.applymap(
                    lambda v: "color: #00e676" if isinstance(v, (int, float)) and v > 0
                    else "color: #ef5350" if isinstance(v, (int, float)) and v < 0
                    else "",
                    subset=["pnl", "pnl_percent"]
                ),
                use_container_width=True,
                hide_index=True
            )

            # Trade stats
            col1, col2, col3 = st.columns(3)
            with col1:
                total_pnl = trades_df["pnl"].sum()
                color = "positive" if total_pnl > 0 else "negative"
                st.markdown(f"**Total P&L:** <span class='{color}'>${total_pnl:,.2f}</span>",
                           unsafe_allow_html=True)
            with col2:
                avg_pnl = trades_df["pnl"].mean()
                st.markdown(f"**Avg Trade P&L:** ${avg_pnl:,.2f}")
            with col3:
                total_comm = trades_df["commission"].sum()
                st.markdown(f"**Total Commission:** ${total_comm:,.2f}")
        else:
            st.info("No trades to display.")

    # ==========================================
    # FOOTER
    # ==========================================
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #8892b0; font-size: 13px;'>"
        "🤖 Algorithmic Trading Engine | Built with Python, Streamlit, Plotly | "
        "Powered by ML (XGBoost, Random Forest)"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()