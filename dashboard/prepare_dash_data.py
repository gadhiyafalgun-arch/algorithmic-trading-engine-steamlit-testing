"""
Prepare pre-computed data for the Dash Trading Dashboard.
Run ONCE: python dashboard/prepare_dash_data.py

Runs the full pipeline → backtester at risk levels 1×–6× → saves results
as pickle files to data/dashboard/ for instant Dash loading.
"""

import sys
import os
import pickle
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import DataPipeline
from src.indicators import TechnicalIndicators
from src.strategy import TradingStrategy
from src.backtester import Backtester
from src.performance import PerformanceAnalyzer
from loguru import logger


def main():
    print("=" * 60)
    print("🚀 Preparing Dash Dashboard Data")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "dashboard")
    os.makedirs(output_dir, exist_ok=True)

    # ===== Step 1: Run data pipeline =====
    print("\n📊 Step 1: Loading data pipeline...")
    pipeline = DataPipeline()
    data = pipeline.run_pipeline()

    # ===== Step 2: Add indicators & strategies =====
    print("\n📈 Step 2: Adding indicators & strategies...")
    indicators = TechnicalIndicators()
    strategy = TradingStrategy()

    processed_data = {}
    for symbol, df in data.items():
        print(f"  Processing {symbol}...")
        df = indicators.add_all_indicators(df)
        df = strategy.apply_all_strategies(df)
        processed_data[symbol] = df

    first_symbol = list(processed_data.keys())[0]
    first_df = processed_data[first_symbol]

    # ===== Step 3: Run backtests at multiple risk levels =====
    print("\n🎚️ Step 3: Running backtests at risk levels 1×–6×...")
    backtester = Backtester()
    performance = PerformanceAnalyzer()

    risk_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    signal_col = "combined_signal" if "combined_signal" in first_df.columns else "sma_signal"

    risk_level_results = backtester.run_with_risk_levels(
        first_df,
        signal_column=signal_col,
        symbol=first_symbol,
        risk_multipliers=risk_multipliers
    )

    # ===== Step 4: Calculate metrics for each risk level =====
    print("\n📊 Step 4: Calculating performance metrics...")
    all_metrics = {}
    for mult, results in risk_level_results.items():
        metrics = performance.calculate_all_metrics(
            results["portfolio_history"],
            results["trades"],
            results["initial_capital"]
        )
        metrics["risk_multiplier"] = mult
        metrics["position_size_pct"] = results.get("position_size_pct", 0)
        metrics["risk_per_trade_pct"] = results.get("risk_per_trade_pct", 0)
        all_metrics[mult] = metrics

    # ===== Step 5: Get benchmark data (SPY) =====
    print("\n📈 Step 5: Extracting benchmark (SPY) data...")
    benchmark_data = None
    if "SPY" in processed_data:
        spy_df = processed_data["SPY"]
        benchmark_data = {
            "dates": spy_df.index.tolist(),
            "close": spy_df["close"].values.tolist(),
            "daily_return": spy_df["close"].pct_change().fillna(0).values.tolist()
        }

    # ===== Step 6: Extract trade signals for the first symbol =====
    print("\n🎯 Step 6: Extracting trading signals...")
    signal_data = {}
    for col in first_df.columns:
        if "signal" in col.lower() and first_df[col].abs().sum() > 0:
            buys = first_df[first_df[col] == 1].index.tolist()
            sells = first_df[first_df[col] == -1].index.tolist()
            buy_prices = first_df.loc[first_df[col] == 1, "close"].values.tolist()
            sell_prices = first_df.loc[first_df[col] == -1, "close"].values.tolist()
            signal_data[col] = {
                "buy_dates": buys,
                "sell_dates": sells,
                "buy_prices": buy_prices,
                "sell_prices": sell_prices
            }

    # ===== Step 7: Save everything =====
    print("\n💾 Step 7: Saving data...")

    dashboard_data = {
        "symbol": first_symbol,
        "signal_column": signal_col,
        "dates": first_df.index.tolist(),
        "close_prices": first_df["close"].values.tolist(),
        "risk_level_results": {},
        "all_metrics": {},
        "benchmark": benchmark_data,
        "signal_data": signal_data,
        "initial_capital": backtester.initial_capital,
        "risk_multipliers": risk_multipliers,
    }

    # Serialize portfolio histories
    for mult, results in risk_level_results.items():
        pdf = results["portfolio_history"]
        dashboard_data["risk_level_results"][mult] = {
            "dates": pdf.index.tolist(),
            "total_value": pdf["total_value"].values.tolist(),
            "daily_return": pdf["daily_return"].values.tolist(),
            "cash": pdf["cash"].values.tolist(),
            "final_value": results["final_value"],
            "total_return": results["total_return"],
            "total_trades": results["total_trades"],
            "position_size_pct": results.get("position_size_pct", 0),
            "risk_per_trade_pct": results.get("risk_per_trade_pct", 0),
        }

    # Serialize metrics (make JSON-safe)
    for mult, metrics in all_metrics.items():
        safe_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                safe_metrics[k] = v
            elif hasattr(v, 'item'):
                safe_metrics[k] = v.item()
            else:
                safe_metrics[k] = str(v)
        dashboard_data["all_metrics"][mult] = safe_metrics

    filepath = os.path.join(output_dir, "dash_data.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(dashboard_data, f)

    print(f"\n✅ Dashboard data saved to: {filepath}")
    print(f"   Symbol: {first_symbol}")
    print(f"   Signal: {signal_col}")
    print(f"   Risk levels: {len(risk_multipliers)}")
    print(f"   Date range: {first_df.index[0]} → {first_df.index[-1]}")
    print(f"   File size: {os.path.getsize(filepath) / 1024:.1f} KB")
    print("\n" + "=" * 60)
    print("✅ DONE! Now run: python dashboard/dash_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
