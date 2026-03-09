"""
Generate pre-computed data files for the dashboard.
Run ONCE: python dashboard/generate_data.py
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import DataPipeline
from src.indicators import TechnicalIndicators
from src.strategy import TradingStrategy

def main():
    print("=" * 60)
    print("🚀 Generating Dashboard Data")
    print("=" * 60)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "dashboard")
    os.makedirs(output_dir, exist_ok=True)

    # Load raw data
    print("\n📊 Loading data pipeline...")
    pipeline = DataPipeline()
    data = pipeline.run_pipeline()

    # Process each symbol
    indicators = TechnicalIndicators()
    strategy = TradingStrategy()

    for symbol, df in data.items():
        print(f"\n🔧 Processing {symbol}...")

        # Add indicators
        df = indicators.add_all_indicators(df)

        # Add strategies
        df = strategy.apply_all_strategies(df)

        # Fix the date index for clean CSV storage
        # If index is datetime, reset it to a column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        # Find the date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break

        if date_col:
            # Remove timezone info and ensure clean datetime
            df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
            # Rename to 'date' if it's not already
            if date_col != 'date':
                df = df.rename(columns={date_col: 'date'})

        # Save
        filepath = os.path.join(output_dir, f"{symbol}.csv")
        df.to_csv(filepath, index=False)

        # Report
        signal_cols = [c for c in df.columns if 'signal' in c.lower()]
        print(f"   ✅ {symbol}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"   📡 Signal columns: {signal_cols}")

    # Generate summary stats for quick dashboard loading
    print("\n📋 Generating summary stats...")
    summary = {}
    for symbol in data.keys():
        filepath = os.path.join(output_dir, f"{symbol}.csv")
        df = pd.read_csv(filepath)
        summary[symbol] = {
            "rows": len(df),
            "columns": len(df.columns),
            "date_start": df["date"].iloc[0],
            "date_end": df["date"].iloc[-1],
            "price_start": round(df["close"].iloc[0], 2),
            "price_end": round(df["close"].iloc[-1], 2),
            "total_return_pct": round(
                (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 2
            ),
        }

    # Save summary
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(os.path.join(output_dir, "_summary.csv"))
    print(f"\n✅ Summary saved to data/dashboard/_summary.csv")

    print("\n" + "=" * 60)
    print("✅ DONE! Dashboard data ready.")
    print(f"📁 Files saved to: {output_dir}")
    print("=" * 60)

    # Final verification
    print("\n🔍 Verification:")
    test_file = os.path.join(output_dir, "AAPL.csv")
    test_df = pd.read_csv(test_file)
    print(f"   AAPL columns: {test_df.shape[1]}")
    print(f"   Date sample: {test_df['date'].iloc[0]}")
    print(f"   Signal columns found:")
    for col in ['sma_signal', 'rsi_signal', 'macd_trade_signal', 'bb_signal', 'combined_signal']:
        if col in test_df.columns:
            buys = (test_df[col] == 1).sum()
            sells = (test_df[col] == -1).sum()
            print(f"      ✅ {col}: {buys} buys, {sells} sells")
        else:
            print(f"      ❌ {col}: NOT FOUND")


if __name__ == "__main__":
    main()