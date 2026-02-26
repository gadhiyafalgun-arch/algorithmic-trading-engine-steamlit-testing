"""
Technical Indicators Module
============================
Contains all technical analysis indicators used for trading signals.

Indicators included:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- VWAP (Volume Weighted Average Price)
- Stochastic Oscillator
"""

import pandas as pd
import numpy as np
from loguru import logger


class TechnicalIndicators:
    """
    Calculate technical indicators on OHLCV data.
    
    Usage:
        ti = TechnicalIndicators()
        df = ti.add_all_indicators(df)
    """

    def __init__(self):
        logger.info("TechnicalIndicators initialized")

    # ==========================================
    # MOVING AVERAGES
    # ==========================================

    def sma(self, df: pd.DataFrame, column: str = "close", period: int = 20) -> pd.Series:
        """
        Simple Moving Average (SMA)
        
        What it does:
            Average of closing prices over a period.
            
        Trading use:
            - Price above SMA = Bullish
            - Price below SMA = Bearish
            - SMA crossovers = Buy/Sell signals
        
        Args:
            df: DataFrame with OHLCV data
            column: Column to calculate SMA on
            period: Number of periods
            
        Returns:
            Series with SMA values
        """
        return df[column].rolling(window=period).mean()

    def ema(self, df: pd.DataFrame, column: str = "close", period: int = 20) -> pd.Series:
        """
        Exponential Moving Average (EMA)
        
        What it does:
            Like SMA but gives MORE weight to recent prices.
            Reacts faster to price changes.
            
        Trading use:
            - Faster than SMA for signals
            - Better for short-term trading
        """
        return df[column].ewm(span=period, adjust=False).mean()

    # ==========================================
    # RSI (Relative Strength Index)
    # ==========================================

    def rsi(self, df: pd.DataFrame, column: str = "close", period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI)
        
        What it does:
            Measures speed and magnitude of price changes.
            Ranges from 0 to 100.
            
        Trading use:
            - RSI > 70 = Overbought (potential SELL)
            - RSI < 30 = Oversold (potential BUY)
            - RSI crossing 50 = trend confirmation
            
        Args:
            df: DataFrame with OHLCV data
            column: Column to calculate RSI on
            period: RSI period (default 14)
            
        Returns:
            Series with RSI values (0-100)
        """
        # Calculate price changes
        delta = df[column].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Calculate average gain and loss (Wilder's smoothing)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi

    # ==========================================
    # MACD (Moving Average Convergence Divergence)
    # ==========================================

    def macd(self, df: pd.DataFrame, column: str = "close",
             fast_period: int = 12, slow_period: int = 26,
             signal_period: int = 9) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence)
        
        What it does:
            Shows relationship between two moving averages.
            
        Components:
            - MACD Line = Fast EMA - Slow EMA
            - Signal Line = EMA of MACD Line
            - Histogram = MACD Line - Signal Line
            
        Trading use:
            - MACD crosses ABOVE signal = BUY
            - MACD crosses BELOW signal = SELL
            - Histogram growing = trend strengthening
            - Histogram shrinking = trend weakening
        """
        # Calculate fast and slow EMA
        fast_ema = df[column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df[column].ewm(span=slow_period, adjust=False).mean()

        # MACD Line
        macd_line = fast_ema - slow_ema

        # Signal Line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        result = pd.DataFrame({
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": histogram
        }, index=df.index)

        return result

    # ==========================================
    # BOLLINGER BANDS
    # ==========================================

    def bollinger_bands(self, df: pd.DataFrame, column: str = "close",
                        period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Bollinger Bands
        
        What it does:
            Creates upper and lower bands around a moving average.
            Bands expand in volatile markets, contract in calm markets.
            
        Components:
            - Middle Band = SMA
            - Upper Band = SMA + (std_dev × Standard Deviation)
            - Lower Band = SMA - (std_dev × Standard Deviation)
            
        Trading use:
            - Price touching upper band = potentially overbought
            - Price touching lower band = potentially oversold
            - Band squeeze = big move coming
            - Price outside bands = extreme move
        """
        middle = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        # Bandwidth (how wide the bands are)
        bandwidth = (upper - lower) / middle

        # %B (where price is relative to the bands)
        percent_b = (df[column] - lower) / (upper - lower)

        result = pd.DataFrame({
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_percent_b": percent_b
        }, index=df.index)

        return result

    # ==========================================
    # ATR (Average True Range)
    # ==========================================

    def atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR)
        
        What it does:
            Measures market volatility.
            Higher ATR = more volatile market.
            
        Trading use:
            - Setting stop-loss levels
            - Position sizing
            - Identifying volatile periods
        """
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)

        # True Range is the maximum of:
        tr1 = high - low                    # Current High - Current Low
        tr2 = abs(high - close)             # Current High - Previous Close
        tr3 = abs(low - close)              # Current Low - Previous Close

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Average True Range
        atr = true_range.ewm(com=period - 1, min_periods=period).mean()

        return atr

    # ==========================================
    # STOCHASTIC OSCILLATOR
    # ==========================================

    def stochastic(self, df: pd.DataFrame, k_period: int = 14,
                   d_period: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator
        
        What it does:
            Compares closing price to price range over a period.
            Ranges from 0 to 100.
            
        Trading use:
            - %K > 80 = Overbought
            - %K < 20 = Oversold
            - %K crossing above %D = BUY
            - %K crossing below %D = SELL
        """
        lowest_low = df["low"].rolling(window=k_period).min()
        highest_high = df["high"].rolling(window=k_period).max()

        # %K line
        k_line = 100 * ((df["close"] - lowest_low) / (highest_high - lowest_low))

        # %D line (SMA of %K)
        d_line = k_line.rolling(window=d_period).mean()

        result = pd.DataFrame({
            "stoch_k": k_line,
            "stoch_d": d_line
        }, index=df.index)

        return result

    # ==========================================
    # VWAP (Volume Weighted Average Price)
    # ==========================================

    def vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP)
        
        What it does:
            Average price weighted by volume.
            Shows the "true" average price.
            
        Trading use:
            - Price above VWAP = Bullish
            - Price below VWAP = Bearish
            - Institutional traders use this heavily
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

        return vwap

    # ==========================================
    # ADD ALL INDICATORS TO DATAFRAME
    # ==========================================

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ALL technical indicators to the DataFrame.
        
        This is the main method — call this to get everything at once.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        if df.empty:
            logger.warning("Empty DataFrame — cannot add indicators")
            return df

        df = df.copy()
        logger.info("Adding technical indicators...")

        # --- Moving Averages ---
        df["sma_10"] = self.sma(df, period=10)
        df["sma_20"] = self.sma(df, period=20)
        df["sma_50"] = self.sma(df, period=50)
        df["sma_200"] = self.sma(df, period=200)
        df["ema_10"] = self.ema(df, period=10)
        df["ema_20"] = self.ema(df, period=20)
        df["ema_50"] = self.ema(df, period=50)

        # --- RSI ---
        df["rsi_14"] = self.rsi(df, period=14)

        # --- MACD ---
        macd_data = self.macd(df)
        df["macd_line"] = macd_data["macd_line"]
        df["macd_signal"] = macd_data["macd_signal"]
        df["macd_histogram"] = macd_data["macd_histogram"]

        # --- Bollinger Bands ---
        bb_data = self.bollinger_bands(df)
        df["bb_upper"] = bb_data["bb_upper"]
        df["bb_middle"] = bb_data["bb_middle"]
        df["bb_lower"] = bb_data["bb_lower"]
        df["bb_bandwidth"] = bb_data["bb_bandwidth"]
        df["bb_percent_b"] = bb_data["bb_percent_b"]

        # --- ATR ---
        df["atr_14"] = self.atr(df, period=14)

        # --- Stochastic ---
        stoch_data = self.stochastic(df)
        df["stoch_k"] = stoch_data["stoch_k"]
        df["stoch_d"] = stoch_data["stoch_d"]

        # --- VWAP ---
        df["vwap"] = self.vwap(df)

        logger.info(f"✅ All indicators added — {len(df.columns)} total columns")
        return df

    def get_indicator_summary(self, df: pd.DataFrame) -> dict:
        """
        Get current indicator values (latest row).
        Useful for quick overview.
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]

        summary = {
            "price": latest["close"],
            "sma_20": latest.get("sma_20", None),
            "sma_50": latest.get("sma_50", None),
            "rsi_14": latest.get("rsi_14", None),
            "macd_line": latest.get("macd_line", None),
            "macd_signal": latest.get("macd_signal", None),
            "bb_upper": latest.get("bb_upper", None),
            "bb_lower": latest.get("bb_lower", None),
            "atr_14": latest.get("atr_14", None),
        }

        return summary