"""
Trading Strategy Module
========================
Contains trading strategies that generate BUY/SELL signals.

Strategies included:
1. SMA Crossover
2. RSI Strategy
3. MACD Strategy
4. Bollinger Bands Strategy
5. Combined Strategy (uses multiple indicators)
"""

import pandas as pd
import numpy as np
from loguru import logger


class TradingStrategy:
    """
    Generates trading signals based on technical indicators.
    
    Signals:
        1  = BUY
       -1  = SELL
        0  = HOLD (do nothing)
    """

    def __init__(self):
        logger.info("TradingStrategy initialized")

    # ==========================================
    # STRATEGY 1: SMA CROSSOVER
    # ==========================================

    def sma_crossover(self, df: pd.DataFrame, 
                      fast_period: int = 20, 
                      slow_period: int = 50) -> pd.DataFrame:
        """
        SMA Crossover Strategy
        
        Logic:
            BUY  → When fast SMA crosses ABOVE slow SMA (Golden Cross)
            SELL → When fast SMA crosses BELOW slow SMA (Death Cross)
            
        This is the most classic and simple strategy.
        Great for trending markets.
        
        Args:
            df: DataFrame with OHLCV + indicators
            fast_period: Short-term SMA period
            slow_period: Long-term SMA period
        """
        df = df.copy()

        fast_col = f"sma_{fast_period}"
        slow_col = f"sma_{slow_period}"

        # Check if indicators exist
        if fast_col not in df.columns or slow_col not in df.columns:
            logger.error(f"Missing columns: {fast_col} or {slow_col}")
            logger.error("Run TechnicalIndicators.add_all_indicators() first!")
            return df

        # Generate position: 1 when fast > slow, 0 otherwise
        df["sma_position"] = np.where(df[fast_col] > df[slow_col], 1, 0)

        # Generate signal: detect the CROSSOVER moment
        # Signal = 1 (BUY) when position changes from 0 to 1
        # Signal = -1 (SELL) when position changes from 1 to 0
        df["sma_signal"] = df["sma_position"].diff()

        # Clean up: 
        # diff() gives 1.0 for BUY, -1.0 for SELL, 0.0 for HOLD
        df["sma_signal"] = df["sma_signal"].fillna(0).astype(int)

        # Count signals
        buys = (df["sma_signal"] == 1).sum()
        sells = (df["sma_signal"] == -1).sum()
        logger.info(f"📊 SMA Crossover ({fast_period}/{slow_period}): {buys} BUYs, {sells} SELLs")

        return df

    # ==========================================
    # STRATEGY 2: RSI STRATEGY
    # ==========================================

    def rsi_strategy(self, df: pd.DataFrame,
                     overbought: int = 70,
                     oversold: int = 30) -> pd.DataFrame:
        """
        RSI Strategy
        
        Logic:
            BUY  → When RSI crosses ABOVE oversold level (30)
                   (stock was oversold, now recovering)
            SELL → When RSI crosses BELOW overbought level (70)
                   (stock was overbought, now declining)
                   
        Best for: Range-bound / sideways markets
        """
        df = df.copy()

        if "rsi_14" not in df.columns:
            logger.error("Missing rsi_14 column!")
            return df

        # Initialize signal column
        df["rsi_signal"] = 0

        # BUY: RSI was below oversold, now crosses above it
        df.loc[
            (df["rsi_14"] > oversold) & 
            (df["rsi_14"].shift(1) <= oversold),
            "rsi_signal"
        ] = 1

        # SELL: RSI was above overbought, now crosses below it
        df.loc[
            (df["rsi_14"] < overbought) & 
            (df["rsi_14"].shift(1) >= overbought),
            "rsi_signal"
        ] = -1

        buys = (df["rsi_signal"] == 1).sum()
        sells = (df["rsi_signal"] == -1).sum()
        logger.info(f"📊 RSI Strategy ({oversold}/{overbought}): {buys} BUYs, {sells} SELLs")

        return df

    # ==========================================
    # STRATEGY 3: MACD STRATEGY
    # ==========================================

    def macd_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD Strategy
        
        Logic:
            BUY  → When MACD line crosses ABOVE signal line
            SELL → When MACD line crosses BELOW signal line
            
        Best for: Trending markets, medium-term trading
        """
        df = df.copy()

        if "macd_line" not in df.columns or "macd_signal" not in df.columns:
            logger.error("Missing MACD columns!")
            return df

        # Position: 1 when MACD > Signal, 0 otherwise
        df["macd_position"] = np.where(
            df["macd_line"] > df["macd_signal"], 1, 0
        )

        # Signal: detect crossover
        df["macd_trade_signal"] = df["macd_position"].diff().fillna(0).astype(int)

        buys = (df["macd_trade_signal"] == 1).sum()
        sells = (df["macd_trade_signal"] == -1).sum()
        logger.info(f"📊 MACD Strategy: {buys} BUYs, {sells} SELLs")

        return df

    # ==========================================
    # STRATEGY 4: BOLLINGER BANDS STRATEGY
    # ==========================================

    def bollinger_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bollinger Bands Strategy
        
        Logic:
            BUY  → When price crosses ABOVE lower band
                   (was below lower band = oversold, now recovering)
            SELL → When price crosses BELOW upper band
                   (was above upper band = overbought, now declining)
                   
        Best for: Mean-reversion in range-bound markets
        """
        df = df.copy()

        if "bb_lower" not in df.columns or "bb_upper" not in df.columns:
            logger.error("Missing Bollinger Band columns!")
            return df

        df["bb_signal"] = 0

        # BUY: Price was below lower band, now crosses above
        df.loc[
            (df["close"] > df["bb_lower"]) & 
            (df["close"].shift(1) <= df["bb_lower"].shift(1)),
            "bb_signal"
        ] = 1

        # SELL: Price was above upper band, now crosses below
        df.loc[
            (df["close"] < df["bb_upper"]) & 
            (df["close"].shift(1) >= df["bb_upper"].shift(1)),
            "bb_signal"
        ] = -1

        buys = (df["bb_signal"] == 1).sum()
        sells = (df["bb_signal"] == -1).sum()
        logger.info(f"📊 Bollinger Strategy: {buys} BUYs, {sells} SELLs")

        return df

    # ==========================================
    # STRATEGY 5: COMBINED STRATEGY ⭐
    # ==========================================

    def combined_strategy(self, df: pd.DataFrame,
                          min_confirmations: int = 2) -> pd.DataFrame:
        """
        Combined Strategy (Multi-Indicator Confirmation)
        
        Logic:
            Uses MULTIPLE indicators to confirm signals.
            Only triggers BUY/SELL when enough indicators agree.
            
            BUY when at least {min_confirmations} of these are true:
            - SMA fast > SMA slow (uptrend)
            - RSI < 40 (not overbought)
            - MACD line > Signal line (bullish momentum)
            - Price near lower Bollinger Band
            
            SELL when at least {min_confirmations} of these are true:
            - SMA fast < SMA slow (downtrend)
            - RSI > 60 (not oversold)
            - MACD line < Signal line (bearish momentum)
            - Price near upper Bollinger Band
            
        This is MORE RELIABLE than single-indicator strategies.
        Fewer trades but higher quality signals.
        """
        df = df.copy()

        # --- BUY Conditions ---
        buy_conditions = pd.DataFrame(index=df.index)

        # Condition 1: SMA trend (fast > slow)
        buy_conditions["sma_bullish"] = (df["sma_20"] > df["sma_50"]).astype(int)

        # Condition 2: RSI not overbought
        buy_conditions["rsi_ok"] = (df["rsi_14"] < 40).astype(int)

        # Condition 3: MACD bullish
        buy_conditions["macd_bullish"] = (df["macd_line"] > df["macd_signal"]).astype(int)

        # Condition 4: Price near lower Bollinger Band
        buy_conditions["bb_oversold"] = (df["bb_percent_b"] < 0.3).astype(int)

        # Count how many conditions are TRUE
        buy_conditions["buy_score"] = buy_conditions.sum(axis=1)

        # --- SELL Conditions ---
        sell_conditions = pd.DataFrame(index=df.index)

        # Condition 1: SMA trend (fast < slow)
        sell_conditions["sma_bearish"] = (df["sma_20"] < df["sma_50"]).astype(int)

        # Condition 2: RSI overbought
        sell_conditions["rsi_high"] = (df["rsi_14"] > 60).astype(int)

        # Condition 3: MACD bearish
        sell_conditions["macd_bearish"] = (df["macd_line"] < df["macd_signal"]).astype(int)

        # Condition 4: Price near upper Bollinger Band
        sell_conditions["bb_overbought"] = (df["bb_percent_b"] > 0.7).astype(int)

        # Count how many conditions are TRUE
        sell_conditions["sell_score"] = sell_conditions.sum(axis=1)

        # --- Generate Combined Signal ---
        df["combined_buy_score"] = buy_conditions["buy_score"]
        df["combined_sell_score"] = sell_conditions["sell_score"]

        # Signal: only when enough confirmations
        df["combined_signal"] = 0

        # BUY when buy score meets threshold AND changes
        buy_mask = (
            (buy_conditions["buy_score"] >= min_confirmations) &
            (buy_conditions["buy_score"].shift(1) < min_confirmations)
        )
        df.loc[buy_mask, "combined_signal"] = 1

        # SELL when sell score meets threshold AND changes
        sell_mask = (
            (sell_conditions["sell_score"] >= min_confirmations) &
            (sell_conditions["sell_score"].shift(1) < min_confirmations)
        )
        df.loc[sell_mask, "combined_signal"] = -1

        buys = (df["combined_signal"] == 1).sum()
        sells = (df["combined_signal"] == -1).sum()
        logger.info(f"📊 Combined Strategy (min {min_confirmations} confirmations): {buys} BUYs, {sells} SELLs")

        return df

    # ==========================================
    # RUN ALL STRATEGIES
    # ==========================================

    def apply_all_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ALL strategies to the DataFrame.
        
        Args:
            df: DataFrame with indicators already added
            
        Returns:
            DataFrame with all strategy signals
        """
        logger.info("Applying all trading strategies...")

        df = self.sma_crossover(df)
        df = self.rsi_strategy(df)
        df = self.macd_strategy(df)
        df = self.bollinger_strategy(df)
        df = self.combined_strategy(df)

        logger.info("✅ All strategies applied")
        return df

    def get_signal_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of all signals for reporting.
        """
        signal_cols = [col for col in df.columns if "signal" in col.lower()]

        summary_data = []
        for col in signal_cols:
            buys = (df[col] == 1).sum()
            sells = (df[col] == -1).sum()
            holds = (df[col] == 0).sum()
            summary_data.append({
                "strategy": col,
                "buy_signals": buys,
                "sell_signals": sells,
                "hold_signals": holds,
                "total_trades": buys + sells
            })

        return pd.DataFrame(summary_data)