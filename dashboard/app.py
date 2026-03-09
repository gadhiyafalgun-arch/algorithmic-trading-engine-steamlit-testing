"""
Algorithmic Trading Engine — Interactive Dashboard
====================================================
A stunning interactive dashboard showcasing a complete
algorithmic trading system built from scratch.

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ==========================================
# PAGE CONFIG — must be first Streamlit call
# ==========================================
st.set_page_config(
    page_title="Algo Trading Engine | Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# DATA PATH
# ==========================================
# Works both locally and on Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "dashboard")
SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY"]


# ==========================================
# CUSTOM CSS — Dark Trading Terminal Theme
# ==========================================
def inject_css():
    st.markdown("""
    <style>
        /* ===== GLOBAL ===== */
        .stApp {
            background-color: #0a0e17;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #0d1117;
            border-right: 1px solid #1a2332;
        }

        /* ===== HERO SECTION ===== */
        .hero-container {
            text-align: center;
            padding: 40px 20px 20px 20px;
        }

        .hero-title {
            font-size: 52px;
            font-weight: 800;
            background: linear-gradient(135deg, #00e676 0%, #00bcd4 50%, #2196f3 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
            letter-spacing: -1px;
        }

        .hero-subtitle {
            font-size: 18px;
            color: #6c7a89;
            margin-bottom: 30px;
            font-weight: 300;
        }

        .hero-subtitle span {
            color: #00e676;
            font-weight: 600;
        }

        /* ===== STAT CARDS ===== */
        .stat-row {
            display: flex;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
            margin: 20px 0 30px 0;
        }

        .stat-card {
            background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
            border: 1px solid #1e2d3d;
            border-radius: 12px;
            padding: 20px 28px;
            text-align: center;
            min-width: 160px;
            transition: transform 0.2s, border-color 0.2s;
        }

        .stat-card:hover {
            transform: translateY(-3px);
            border-color: #00e676;
        }

        .stat-number {
            font-size: 32px;
            font-weight: 800;
            color: #00e676;
            line-height: 1.2;
        }

        .stat-number.blue { color: #2196f3; }
        .stat-number.cyan { color: #00bcd4; }
        .stat-number.orange { color: #ff9800; }
        .stat-number.purple { color: #9c27b0; }

        .stat-label {
            font-size: 13px;
            color: #6c7a89;
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* ===== SECTION HEADERS ===== */
        .section-header {
            font-size: 28px;
            font-weight: 700;
            color: #e2e8f0;
            margin: 40px 0 8px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #1a2332;
        }

        .section-subtext {
            color: #6c7a89;
            font-size: 15px;
            margin-bottom: 24px;
        }

        /* ===== NAVIGATION ===== */
        .nav-header {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #4a5568;
            margin-bottom: 8px;
            font-weight: 600;
        }

        /* ===== DIVIDER ===== */
        .glow-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #00e676, transparent);
            border: none;
            margin: 30px 0;
        }

        /* ===== FOOTER ===== */
        .footer {
            text-align: center;
            color: #4a5568;
            font-size: 13px;
            padding: 40px 0 20px 0;
            border-top: 1px solid #1a2332;
            margin-top: 60px;
        }

        .footer a {
            color: #00e676;
            text-decoration: none;
        }

        /* ===== HIDE STREAMLIT DEFAULTS ===== */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* ===== SIDEBAR RADIO STYLING ===== */
        div[data-testid="stSidebar"] .stRadio > label {
            display: none;
        }

        div[data-testid="stSidebar"] .stRadio > div {
            gap: 2px;
        }

        div[data-testid="stSidebar"] .stRadio > div > label {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 8px 16px;
            color: #8892b0;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
        }

        div[data-testid="stSidebar"] .stRadio > div > label:hover {
            background: #111827;
            border-color: #1e2d3d;
            color: #e2e8f0;
        }

        div[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
            background: linear-gradient(135deg, #0d2818 0%, #0a1628 100%);
            border-color: #00e676;
            color: #00e676;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# CACHED DATA LOADING
# ==========================================
@st.cache_data
def load_stock_data(symbol):
    """Load pre-computed data for a single stock."""
    filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.set_index("date")
    df.index = df.index.tz_localize(None)  # Strip any timezone
    return df


@st.cache_data
def load_all_stocks():
    """Load all stocks into a dictionary."""
    data = {}
    for symbol in SYMBOLS:
        try:
            data[symbol] = load_stock_data(symbol)
        except Exception as e:
            st.error(f"Failed to load {symbol}: {e}")
    return data


# ==========================================
# HERO SECTION
# ==========================================
def render_hero():
    """Render the hero section with key stats."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🤖 Algorithmic Trading Engine</div>
        <div class="hero-subtitle">
            A complete quantitative trading system built from scratch —
            <span>data pipeline → indicators → strategies → backtesting → ML → risk management</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key stats row
    st.markdown("""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-number">6</div>
            <div class="stat-label">Stocks Tracked</div>
        </div>
        <div class="stat-card">
            <div class="stat-number blue">5</div>
            <div class="stat-label">Trading Strategies</div>
        </div>
        <div class="stat-card">
            <div class="stat-number cyan">8</div>
            <div class="stat-label">Technical Indicators</div>
        </div>
        <div class="stat-card">
            <div class="stat-number orange">3</div>
            <div class="stat-label">ML Models</div>
        </div>
        <div class="stat-card">
            <div class="stat-number purple">1,257</div>
            <div class="stat-label">Trading Days</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">\$100K</div>
            <div class="stat-label">Initial Capital</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)


# ==========================================
# PLACEHOLDER SECTIONS (built in later steps)
# ==========================================
def render_story():
    st.markdown('<div class="section-header">📖 The Story</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Every strategy failed. Here\'s how I found out why — and fixed all of them.</div>', unsafe_allow_html=True)

    # ===== ACT 1: THE FAILURE =====
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a0a0a 0%, #1a1a2e 100%);
        border: 1px solid #3d1515;
        border-left: 4px solid #ff1744;
        border-radius: 12px;
        padding: 28px;
        margin-bottom: 24px;
    ">
        <div style="font-size: 22px; font-weight: 700; color: #ff1744; margin-bottom: 12px;">
            💀 Act 1: Total Failure
        </div>
        <div style="color: #c0c0c0; font-size: 15px; line-height: 1.8;">
            I built 5 trading strategies from scratch — SMA Crossover, RSI, MACD, Bollinger Bands, 
            and a Combined strategy. Ran them through a full backtest on AAPL from 2020-2024 with \$100,000.<br><br>
            <b style="color: #ff1744;">Every single strategy got an F grade.</b> 
            The portfolio was losing money. Not just underperforming — actively destroying capital.
            Commission costs alone ate 28-33% of all gains. Something was fundamentally broken.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Failure metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number" style="color: #ff1744;">F</div>
            <div class="stat-label">Strategy Grades</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number" style="color: #ff1744;">-33%</div>
            <div class="stat-label">Lost to Commissions</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number" style="color: #ff1744;">28%</div>
            <div class="stat-label">Whipsaw Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number" style="color: #ff1744;">0-20%</div>
            <div class="stat-label">Trend Alignment</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ===== ACT 2: THE DIAGNOSIS =====
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0a1a1a 0%, #1a1a2e 100%);
        border: 1px solid #1a3d3d;
        border-left: 4px solid #ff9800;
        border-radius: 12px;
        padding: 28px;
        margin-bottom: 24px;
    ">
        <div style="font-size: 22px; font-weight: 700; color: #ff9800; margin-bottom: 12px;">
            🔬 Act 2: The Diagnosis
        </div>
        <div style="color: #c0c0c0; font-size: 15px; line-height: 1.8;">
            Instead of tweaking parameters randomly, I built a <b style="color: #ff9800;">diagnostic tool</b> 
            — an X-ray machine for trading strategies. It analyzed every signal, every trade, every dollar lost. 
            It found <b>5 critical problems</b> that were killing performance:
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 5 Problems - expandable
    problems = [
        {
            "icon": "💸",
            "title": "Commission Death",
            "severity": "CRITICAL",
            "color": "#ff1744",
            "detail": "28-33% of all gains eaten by trading costs. The strategies were over-trading — generating hundreds of signals that triggered tiny trades, each one paying commission. Death by a thousand cuts.",
            "metric_before": "28-33% commission drag",
            "metric_after": "3-5% commission drag",
        },
        {
            "icon": "📉",
            "title": "Bad Sell Timing",
            "severity": "HIGH",
            "color": "#ff5722",
            "detail": "Strategies were selling during a bull market. AAPL went up 246% in this period, but the strategies kept triggering sell signals during uptrends. Selling winners and holding losers.",
            "metric_before": "Selling in uptrends",
            "metric_after": "Strict sell conditions only",
        },
        {
            "icon": "🔄",
            "title": "Whipsaw Chaos",
            "severity": "HIGH",
            "color": "#ff9800",
            "detail": "20-28% of all signals were rapid flips — BUY today, SELL tomorrow, BUY the next day. Each flip costs commission and locks in small losses. The signal noise was drowning out real opportunities.",
            "metric_before": "20-28% whipsaw rate",
            "metric_after": "0% whipsaw rate",
        },
        {
            "icon": "🔀",
            "title": "RSI Running Backwards",
            "severity": "MEDIUM",
            "color": "#ffc107",
            "detail": "RSI was buying 'oversold' bounces during downtrends — catching falling knives. Only 0-20% of RSI signals aligned with the actual market trend. The indicator was technically correct but strategically useless.",
            "metric_before": "0-20% trend alignment",
            "metric_after": "100% trend alignment",
        },
        {
            "icon": "🙈",
            "title": "No Trend Filter",
            "severity": "MEDIUM",
            "color": "#8bc34a",
            "detail": "None of the strategies checked whether the market was in a bull or bear trend before trading. They were flying blind — buying in downtrends, selling in uptrends, with no context about market conditions.",
            "metric_before": "No trend awareness",
            "metric_after": "SMA 200 trend filter on all",
        },
    ]

    for i, p in enumerate(problems):
        with st.expander(f"{p['icon']}  Problem {i+1}: {p['title']}  —  Severity: {p['severity']}", expanded=(i == 0)):
            st.markdown(f"""
            <div style="
                background: #111827;
                border-radius: 8px;
                padding: 20px;
                border-left: 3px solid {p['color']};
            ">
                <div style="color: #c0c0c0; font-size: 14px; line-height: 1.8; margin-bottom: 16px;">
                    {p['detail']}
                </div>
                <div style="display: flex; gap: 20px;">
                    <div style="
                        flex: 1; background: #1a0a0a; border: 1px solid #3d1515;
                        border-radius: 8px; padding: 12px; text-align: center;
                    ">
                        <div style="font-size: 11px; color: #ff1744; text-transform: uppercase; letter-spacing: 1px;">Before</div>
                        <div style="font-size: 16px; color: #ff1744; font-weight: 600; margin-top: 4px;">{p['metric_before']}</div>
                    </div>
                    <div style="
                        flex: 1; background: #0a1a0a; border: 1px solid #153d15;
                        border-radius: 8px; padding: 12px; text-align: center;
                    ">
                        <div style="font-size: 11px; color: #00e676; text-transform: uppercase; letter-spacing: 1px;">After</div>
                        <div style="font-size: 16px; color: #00e676; font-weight: 600; margin-top: 4px;">{p['metric_after']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ===== ACT 3: THE FIX =====
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0a1a0a 0%, #1a1a2e 100%);
        border: 1px solid #153d15;
        border-left: 4px solid #00e676;
        border-radius: 12px;
        padding: 28px;
        margin-bottom: 24px;
    ">
        <div style="font-size: 22px; font-weight: 700; color: #00e676; margin-bottom: 12px;">
            ✅ Act 3: The Fix — Strategy v2.1
        </div>
        <div style="color: #c0c0c0; font-size: 15px; line-height: 1.8;">
            I redesigned all 5 strategies with systematic fixes. Not random parameter tuning — 
            <b style="color: #00e676;">targeted engineering</b> based on the diagnostic findings:
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fixes applied
    fixes = [
        {"icon": "📊", "name": "SMA 200 Trend Filter", "desc": "Only buy when price is ABOVE the 200-day moving average. Forces alignment with the long-term trend. Simple physics: don't swim against the current."},
        {"icon": "⏱️", "name": "Cooldown Periods", "desc": "10-15 bar minimum between signals. Eliminates whipsaw by enforcing a waiting period after each trade. Like a refractory period in neural firing."},
        {"icon": "🎯", "name": "Trend-Aware RSI", "desc": "RSI now only triggers buys during confirmed uptrends (buying dips), not random oversold bounces. Context-dependent signal interpretation."},
        {"icon": "🔍", "name": "Lookback Window", "desc": "Combined strategy uses a 5-bar lookback window — requires 2+ strategies to agree within 5 days, not just on the same bar. Reduces noise, increases conviction."},
        {"icon": "🚪", "name": "Strict Sell Conditions", "desc": "Sells only trigger on strong reversal signals, not minor fluctuations. Lets winners run instead of cutting them at the first dip."},
    ]

    fix_cols = st.columns(len(fixes))
    for col, fix in zip(fix_cols, fixes):
        with col:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #111827 0%, #0d1a0d 100%);
                border: 1px solid #1e2d3d;
                border-radius: 10px;
                padding: 16px;
                height: 220px;
                transition: border-color 0.2s;
            ">
                <div style="font-size: 28px; margin-bottom: 8px;">{fix['icon']}</div>
                <div style="color: #00e676; font-weight: 600; font-size: 14px; margin-bottom: 8px;">{fix['name']}</div>
                <div style="color: #8892b0; font-size: 12px; line-height: 1.6;">{fix['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ===== BEFORE vs AFTER TABLE =====
    st.markdown("""
    <div style="font-size: 20px; font-weight: 700; color: #e2e8f0; margin: 20px 0 16px 0;">
        📋 Before vs After — Full Comparison
    </div>
    """, unsafe_allow_html=True)

    comparison_data = {
        "Metric": [
            "Trend Alignment",
            "Whipsaw Rate",
            "Commission Drag",
            "RSI Accuracy",
            "Trend Filter",
            "Portfolio Result",
        ],
        "❌ Before (v1)": [
            "0-20%",
            "20-28%",
            "28-33%",
            "Buying against trend",
            "None",
            "📉 Net Loss",
        ],
        "✅ After (v2.1)": [
            "100%",
            "0%",
            "3-5%",
            "Buying with trend",
            "SMA 200 on all",
            "📈 Profitable",
        ],
        "Improvement": [
            "🟢 +80-100%",
            "🟢 -28%",
            "🟢 -28%",
            "🟢 Fixed",
            "🟢 Added",
            "🟢 Loss → Profit",
        ],
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "❌ Before (v1)": st.column_config.TextColumn("❌ Before (v1)", width="medium"),
            "✅ After (v2.1)": st.column_config.TextColumn("✅ After (v2.1)", width="medium"),
            "Improvement": st.column_config.TextColumn("Improvement", width="medium"),
        }
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ===== KEY INSIGHT BOX =====
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d1a2e 0%, #1a1a2e 100%);
        border: 1px solid #1f4068;
        border-radius: 12px;
        padding: 24px;
        margin-top: 12px;
    ">
        <div style="font-size: 18px; font-weight: 700; color: #2196f3; margin-bottom: 12px;">
            💡 The Lesson
        </div>
        <div style="color: #c0c0c0; font-size: 15px; line-height: 1.8;">
            The problem was never the indicators — <b style="color: #2196f3;">RSI, MACD, and Bollinger Bands all work correctly.</b> 
            The problem was using them without context. An RSI oversold signal means something completely different 
            in a bull market vs a bear market. The fix wasn't more complexity — it was 
            <b style="color: #2196f3;">adding awareness</b>.<br><br>
            This mirrors a principle from physics: <i>a measurement only has meaning within its reference frame.</i>
            A trading signal only has meaning within its market regime.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # After metrics row
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">100%</div>
            <div class="stat-label">Trend Alignment</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">0%</div>
            <div class="stat-label">Whipsaw Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number blue">3-5%</div>
            <div class="stat-label">Commission Drag</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number cyan">+3.36%</div>
            <div class="stat-label">Combined Return</div>
        </div>
        """, unsafe_allow_html=True)


def render_risk_slider(data):
    st.markdown('<div class="section-header">🎚️ Risk Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Drag the slider to see how position sizing impacts returns, risk, and drawdowns — live.</div>', unsafe_allow_html=True)
    st.info("🚧 Coming in Step 3 — Interactive risk slider with live equity curve.")


def render_3d_surface(data):
    st.markdown('<div class="section-header">🏔️ Risk Landscape (3D)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Portfolio value across time × risk level — a 3D mountain you can rotate and explore.</div>', unsafe_allow_html=True)
    st.info("🚧 Coming in Step 4 — Rotatable 3D surface chart.")


def render_price_charts(data):
    st.markdown('<div class="section-header">📈 Price Action & Signals</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Interactive candlestick charts with indicators and buy/sell signals.</div>', unsafe_allow_html=True)
    st.info("🚧 Coming in Step 5 — Candlestick + RSI + MACD + signal markers.")


def render_strategy_comparison(data):
    st.markdown('<div class="section-header">⚔️ Strategy Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">How each strategy performed — trades, returns, signal counts.</div>', unsafe_allow_html=True)
    st.info("🚧 Coming in Step 6.")


def render_backtest_results(data):
    st.markdown('<div class="section-header">📊 Backtest Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Full performance metrics for all strategies.</div>', unsafe_allow_html=True)
    st.info("🚧 Coming in Step 6.")


def render_ml_results():
    st.markdown('<div class="section-header">🧠 Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">ML model performance — and why markets are hard to predict.</div>', unsafe_allow_html=True)
    st.info("🚧 Coming in Step 6.")


def render_risk_analysis(data):
    st.markdown('<div class="section-header">🛡️ Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Correlation matrix, risk scores, and diversification analysis.</div>', unsafe_allow_html=True)
    st.info("🚧 Coming in Step 6.")


def render_architecture():
    st.markdown('<div class="section-header">🏗️ System Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">How the entire engine is built — 6 phases, end to end.</div>', unsafe_allow_html=True)
    st.info("🚧 Coming in Step 6.")


# ==========================================
# FOOTER
# ==========================================
def render_footer():
    st.markdown("""
    <div class="footer">
        🤖 Algorithmic Trading Engine — Built with Python, Streamlit & Plotly<br>
        <span style="font-size: 12px; color: #3a4556;">
            Data: Yahoo Finance | ML: XGBoost, Random Forest, Logistic Regression | 
            Risk: Correlation Analysis, VaR, Drawdown Metrics
        </span>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# MAIN
# ==========================================
def main():
    # Inject custom CSS
    def inject_css()
    st.markdown("""
    <style>
        /* ============================================
           ANIMATED TRADING TERMINAL THEME
           ============================================ */

        /* ===== KEYFRAME ANIMATIONS ===== */

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes fadeInRight {
            from {
                opacity: 0;
                transform: translateX(30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes fadeInScale {
            from {
                opacity: 0;
                transform: scale(0.9);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes pulseGlow {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 1; }
        }

        @keyframes borderGlow {
            0%, 100% { border-color: #1e2d3d; }
            50% { border-color: #00e676; }
        }

        @keyframes textGradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes floatUp {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-6px); }
        }

        @keyframes scanline {
            0% { top: -10%; }
            100% { top: 110%; }
        }

        @keyframes particleFloat {
            0%, 100% {
                transform: translateY(0) translateX(0);
                opacity: 0;
            }
            10% { opacity: 0.4; }
            90% { opacity: 0.4; }
            50% {
                transform: translateY(-120px) translateX(30px);
            }
        }

        @keyframes tickerScroll {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }

        @keyframes countUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* ===== GLOBAL ===== */
        .stApp {
            background-color: #0a0e17;
            overflow-x: hidden;
        }

        /* Animated background grid */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image:
                linear-gradient(rgba(0, 230, 118, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 230, 118, 0.03) 1px, transparent 1px);
            background-size: 60px 60px;
            pointer-events: none;
            z-index: 0;
        }

        /* Scanning line effect */
        .stApp::after {
            content: '';
            position: fixed;
            top: -10%;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(0, 230, 118, 0.15), transparent);
            animation: scanline 8s linear infinite;
            pointer-events: none;
            z-index: 0;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #0d1117;
            border-right: 1px solid #1a2332;
        }

        /* ===== HERO SECTION ===== */
        .hero-container {
            text-align: center;
            padding: 40px 20px 20px 20px;
            animation: fadeInUp 0.8s ease-out;
        }

        .hero-title {
            font-size: 52px;
            font-weight: 800;
            background: linear-gradient(135deg, #00e676 0%, #00bcd4 25%, #2196f3 50%, #00e676 75%, #00bcd4 100%);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: textGradient 6s ease infinite;
            margin-bottom: 8px;
            letter-spacing: -1px;
        }

        .hero-subtitle {
            font-size: 18px;
            color: #6c7a89;
            margin-bottom: 30px;
            font-weight: 300;
            animation: fadeInUp 0.8s ease-out 0.2s both;
        }

        .hero-subtitle span {
            color: #00e676;
            font-weight: 600;
        }

        /* ===== STAT CARDS ===== */
        .stat-row {
            display: flex;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
            margin: 20px 0 30px 0;
            animation: fadeInUp 0.8s ease-out 0.4s both;
        }

        .stat-card {
            background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
            border: 1px solid #1e2d3d;
            border-radius: 12px;
            padding: 20px 28px;
            text-align: center;
            min-width: 160px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        /* Card shimmer effect on hover */
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(0, 230, 118, 0.05),
                transparent
            );
            transition: left 0.5s ease;
        }

        .stat-card:hover::before {
            left: 100%;
        }

        .stat-card:hover {
            transform: translateY(-4px);
            border-color: #00e676;
            box-shadow: 0 8px 30px rgba(0, 230, 118, 0.15);
        }

        .stat-number {
            font-size: 32px;
            font-weight: 800;
            color: #00e676;
            line-height: 1.2;
            animation: countUp 0.6s ease-out;
        }

        .stat-number.blue { color: #2196f3; }
        .stat-number.cyan { color: #00bcd4; }
        .stat-number.orange { color: #ff9800; }
        .stat-number.purple { color: #9c27b0; }

        .stat-label {
            font-size: 13px;
            color: #6c7a89;
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* ===== SECTION HEADERS ===== */
        .section-header {
            font-size: 28px;
            font-weight: 700;
            color: #e2e8f0;
            margin: 40px 0 8px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #1a2332;
            animation: fadeInLeft 0.6s ease-out;
        }

        .section-subtext {
            color: #6c7a89;
            font-size: 15px;
            margin-bottom: 24px;
            animation: fadeInLeft 0.6s ease-out 0.1s both;
        }

        /* ===== NAVIGATION ===== */
        .nav-header {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #4a5568;
            margin-bottom: 8px;
            font-weight: 600;
        }

        /* ===== GLOWING DIVIDER ===== */
        .glow-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #00e676, transparent);
            background-size: 200% 100%;
            border: none;
            margin: 30px 0;
            animation: gradientShift 3s ease infinite;
        }

        /* ===== FOOTER ===== */
        .footer {
            text-align: center;
            color: #4a5568;
            font-size: 13px;
            padding: 40px 0 20px 0;
            border-top: 1px solid #1a2332;
            margin-top: 60px;
            animation: fadeInUp 0.6s ease-out;
        }

        .footer a {
            color: #00e676;
            text-decoration: none;
        }

        /* ===== HIDE STREAMLIT DEFAULTS ===== */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* ===== SIDEBAR RADIO STYLING ===== */
        div[data-testid="stSidebar"] .stRadio > label {
            display: none;
        }

        div[data-testid="stSidebar"] .stRadio > div {
            gap: 2px;
        }

        div[data-testid="stSidebar"] .stRadio > div > label {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 8px 16px;
            color: #8892b0;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            font-size: 14px;
        }

        div[data-testid="stSidebar"] .stRadio > div > label:hover {
            background: #111827;
            border-color: #1e2d3d;
            color: #e2e8f0;
            transform: translateX(4px);
        }

        div[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
            background: linear-gradient(135deg, #0d2818 0%, #0a1628 100%);
            border-color: #00e676;
            color: #00e676;
            font-weight: 600;
        }

        /* ===== STREAMLIT ELEMENTS STYLING ===== */

        /* Expanders */
        .streamlit-expanderHeader {
            background: #111827 !important;
            border: 1px solid #1e2d3d !important;
            border-radius: 8px !important;
            color: #e2e8f0 !important;
            transition: all 0.3s ease !important;
        }

        .streamlit-expanderHeader:hover {
            border-color: #00e676 !important;
            background: #0d1a2e !important;
        }

        /* DataFrames */
        .stDataFrame {
            animation: fadeInUp 0.5s ease-out;
        }

        /* Metrics */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
            border: 1px solid #1e2d3d;
            border-radius: 10px;
            padding: 12px 16px;
            transition: all 0.3s ease;
        }

        [data-testid="stMetric"]:hover {
            border-color: #00e676;
            box-shadow: 0 4px 20px rgba(0, 230, 118, 0.1);
        }

        /* Plotly charts */
        .stPlotlyChart {
            animation: fadeInScale 0.6s ease-out;
            border-radius: 12px;
            overflow: hidden;
        }

        /* Info/Warning/Error boxes */
        .stAlert {
            animation: fadeInUp 0.5s ease-out;
        }

        /* ===== OVERVIEW CARDS STAGGER ===== */
        [data-testid="stVerticalBlock"] > div {
            animation: fadeInUp 0.5s ease-out both;
        }

        [data-testid="stHorizontalBlock"] > div:nth-child(1) > div {
            animation-delay: 0s;
        }

        [data-testid="stHorizontalBlock"] > div:nth-child(2) > div {
            animation-delay: 0.1s;
        }

        [data-testid="stHorizontalBlock"] > div:nth-child(3) > div {
            animation-delay: 0.2s;
        }

        /* ===== RESPONSIVE ===== */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 32px;
            }

            .stat-row {
                gap: 8px;
            }

            .stat-card {
                min-width: 120px;
                padding: 14px 18px;
            }

            .stat-number {
                font-size: 24px;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # Data status
    st.markdown("### 📡 Data Status")
    data = load_all_stocks()
    if data:
        st.success(f"✅ {len(data)} stocks loaded")
        for sym in data:
            rows = len(data[sym])
            price = data[sym]["close"].iloc[-1]
            st.markdown(
                f"**{sym}** — ${price:.2f} "
                f"<span style='color:#4a5568;font-size:12px;'>({rows} days)</span>",
                unsafe_allow_html=True
            )
    else:
        st.error("❌ No data found. Run generate_data.py first.")
        return

    st.markdown("---")
    st.markdown(
        "<div style='color:#4a5568; font-size:12px; text-align:center;'>"
        "Built by Falgun Gadhiya<br>"
        "Phase 6 of 6"
        "</div>",
        unsafe_allow_html=True
    )

    # ===== ROUTE TO PAGE =====
    if page == "🏠 Overview":
        render_hero()
        # Show mini previews of each section
        st.markdown('<div class="section-header">🗺️ What\'s Inside</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="stat-card" style="text-align:left; min-width:auto;">
                <div style="font-size:24px; margin-bottom:8px;">📖</div>
                <div style="color:#e2e8f0; font-weight:600; margin-bottom:4px;">The Story</div>
                <div style="color:#6c7a89; font-size:13px;">Every strategy got an F. Here's how I diagnosed 5 critical bugs and fixed them all.</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="stat-card" style="text-align:left; min-width:auto; margin-top:12px;">
                <div style="font-size:24px; margin-bottom:8px;">📈</div>
                <div style="color:#e2e8f0; font-weight:600; margin-bottom:4px;">Price Charts</div>
                <div style="color:#6c7a89; font-size:13px;">Candlestick charts with buy/sell signals, RSI, MACD, and Bollinger Bands.</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="stat-card" style="text-align:left; min-width:auto; margin-top:12px;">
                <div style="font-size:24px; margin-bottom:8px;">🧠</div>
                <div style="color:#e2e8f0; font-weight:600; margin-bottom:4px;">Machine Learning</div>
                <div style="color:#6c7a89; font-size:13px;">3 ML models, 118 features, walk-forward validation. Markets are hard.</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="stat-card" style="text-align:left; min-width:auto;">
                <div style="font-size:24px; margin-bottom:8px;">🎚️</div>
                <div style="color:#e2e8f0; font-weight:600; margin-bottom:4px;">Risk Simulator</div>
                <div style="color:#6c7a89; font-size:13px;">Drag a slider from 10% to 100% position size and watch the equity curve update live.</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="stat-card" style="text-align:left; min-width:auto; margin-top:12px;">
                <div style="font-size:24px; margin-bottom:8px;">⚔️</div>
                <div style="color:#e2e8f0; font-weight:600; margin-bottom:4px;">Strategy Battle</div>
                <div style="color:#6c7a89; font-size:13px;">5 strategies head-to-head: SMA, RSI, MACD, Bollinger, Combined.</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="stat-card" style="text-align:left; min-width:auto; margin-top:12px;">
                <div style="font-size:24px; margin-bottom:8px;">🛡️</div>
                <div style="color:#e2e8f0; font-weight:600; margin-bottom:4px;">Risk Analysis</div>
                <div style="color:#6c7a89; font-size:13px;">Correlation heatmap, risk scores, TSLA flagged HIGH risk at 6.6/10.</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="stat-card" style="text-align:left; min-width:auto;">
                <div style="font-size:24px; margin-bottom:8px;">🏔️</div>
                <div style="color:#e2e8f0; font-weight:600; margin-bottom:4px;">3D Risk Landscape</div>
                <div style="color:#6c7a89; font-size:13px;">A 3D surface showing portfolio value across time × risk level. Rotate and explore.</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="stat-card" style="text-align:left; min-width:auto; margin-top:12px;">
                <div style="font-size:24px; margin-bottom:8px;">📊</div>
                <div style="color:#e2e8f0; font-weight:600; margin-bottom:4px;">Backtest Results</div>
                <div style="color:#6c7a89; font-size:13px;">Return, Sharpe, Win Rate, Max Drawdown — every metric for every strategy.</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="stat-card" style="text-align:left; min-width:auto; margin-top:12px;">
                <div style="font-size:24px; margin-bottom:8px;">🏗️</div>
                <div style="color:#e2e8f0; font-weight:600; margin-bottom:4px;">Architecture</div>
                <div style="color:#6c7a89; font-size:13px;">6-phase system design: Pipeline → Indicators → Strategy → Backtest → ML → Dashboard.</div>
            </div>
            """, unsafe_allow_html=True)

    elif page == "📖 The Story":
        render_hero()
        render_story()

    elif page == "🎚️ Risk Simulator":
        render_hero()
        render_risk_slider(data)

    elif page == "🏔️ 3D Risk Landscape":
        render_hero()
        render_3d_surface(data)

    elif page == "📈 Price Charts":
        render_hero()
        render_price_charts(data)

    elif page == "⚔️ Strategy Comparison":
        render_hero()
        render_strategy_comparison(data)

    elif page == "📊 Backtest Results":
        render_hero()
        render_backtest_results(data)

    elif page == "🧠 Machine Learning":
        render_hero()
        render_ml_results()

    elif page == "🛡️ Risk Analysis":
        render_hero()
        render_risk_analysis(data)

    elif page == "🏗️ Architecture":
        render_hero()
        render_architecture()

    # Footer on every page
    render_footer()


if __name__ == "__main__":
    main()