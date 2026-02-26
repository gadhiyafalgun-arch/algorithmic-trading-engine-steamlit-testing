"""
Algorithmic Trading Engine
==========================
Main entry point for the application.

Author: Your Name
Version: 2.0.0 — Phase 2
"""

from src.data_pipeline import DataPipeline
from src.indicators import TechnicalIndicators
from src.strategy import TradingStrategy
from src.visualizer import Visualizer
from src.utils import ensure_directories
from loguru import logger
import sys

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add("logs/main.log", rotation="5 MB", level="DEBUG")


def main():
    """Main function — runs the trading engine."""

    logger.info("=" * 60)
    logger.info("🚀 ALGORITHMIC TRADING ENGINE v2.0.0")
    logger.info("=" * 60)

    # Step 0: Setup
    ensure_directories()

    # ==========================================
    # PHASE 1: Data Pipeline
    # ==========================================
    logger.info("\n📊 PHASE 1: Data Pipeline")
    pipeline = DataPipeline()
    data = pipeline.run_pipeline()

    # Show summary
    summary = pipeline.get_summary(data)
    print("\n📋 DATA SUMMARY:")
    print(summary.to_string(index=False))
    print()

    # ==========================================
    # PHASE 2: Technical Indicators & Strategy
    # ==========================================
    logger.info("\n📈 PHASE 2: Technical Indicators & Strategy")

    # Initialize modules
    indicators = TechnicalIndicators()
    strategy = TradingStrategy()
    visualizer = Visualizer()

    # Process each stock
    processed_data = {}

    for symbol, df in data.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {symbol}...")
        logger.info(f"{'='*40}")

        # Add technical indicators
        df = indicators.add_all_indicators(df)

        # Apply all trading strategies
        df = strategy.apply_all_strategies(df)

        # Store processed data
        processed_data[symbol] = df

        # Show latest indicator values
        ind_summary = indicators.get_indicator_summary(df)
        logger.info(f"\n📊 {symbol} Latest Indicators:")
        for key, value in ind_summary.items():
            if value is not None:
                logger.info(f"   {key}: {value:.2f}")

    # ==========================================
    # SIGNAL SUMMARY
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("📋 SIGNAL SUMMARY FOR ALL STOCKS")
    logger.info("=" * 60)

    for symbol, df in processed_data.items():
        logger.info(f"\n--- {symbol} ---")
        signal_summary = strategy.get_signal_summary(df)
        print(signal_summary.to_string(index=False))
        print()

    # ==========================================
    # VISUALIZATION
    # ==========================================
    logger.info("\n📊 PHASE 2: Generating Charts...")

    # Pick first stock for detailed charts
    first_symbol = list(processed_data.keys())[0]
    first_df = processed_data[first_symbol]

    # Chart 1: Price with SMA Crossover signals
    visualizer.plot_price_with_signals(
        first_df, first_symbol, signal_column="sma_signal"
    )

    # Chart 2: Price with Combined Strategy signals
    visualizer.plot_price_with_signals(
        first_df, first_symbol, signal_column="combined_signal"
    )

    # Chart 3: MACD Analysis
    visualizer.plot_macd(first_df, first_symbol)

    # Chart 4: Strategy Comparison
    visualizer.plot_strategy_comparison(first_df, first_symbol)

    # ==========================================
    # SAVE ALL PROCESSED DATA
    # ==========================================
    logger.info("\n💾 Saving processed data with indicators & signals...")
    pipeline.save_data(processed_data, data_type="processed")

    # ==========================================
    # DONE
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("✅ PHASE 2 COMPLETE!")
    logger.info("=" * 60)
    logger.info("📁 Charts saved in: docs/charts/")
    logger.info("📁 Data saved in: data/processed/")
    logger.info("🔜 Next: Phase 3 — Backtesting Engine")


if __name__ == "__main__":
    main()