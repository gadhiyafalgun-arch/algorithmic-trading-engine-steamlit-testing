"""
Backtesting Engine
===================
Simulates trading strategies on historical data.

This is the CORE of the algo trading engine.
It tells you: "Would this strategy have made money?"

Features:
- Simulates trades based on signals
- Tracks portfolio value over time
- Accounts for transaction costs & slippage
- Tracks individual trades with entry/exit
- Generates detailed performance report
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger
import yaml


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    direction: str           # 'LONG' or 'SHORT'
    shares: int
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0        # Profit and Loss
    pnl_percent: float = 0.0
    status: str = "OPEN"     # 'OPEN' or 'CLOSED'
    commission_paid: float = 0.0


@dataclass
class PortfolioState:
    """Snapshot of portfolio at a point in time."""
    date: pd.Timestamp
    cash: float
    holdings_value: float
    total_value: float
    num_positions: int
    daily_return: float = 0.0


class Backtester:
    """
    Core backtesting engine.
    
    Simulates trading on historical data with realistic conditions:
    - Commission costs
    - Slippage
    - Position sizing
    - Portfolio tracking
    
    Usage:
        bt = Backtester()
        results = bt.run(df, signal_column='sma_signal', symbol='AAPL')
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize backtester with configuration."""

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Trading parameters
        trading = self.config["trading"]
        self.initial_capital = trading["initial_capital"]
        self.commission_rate = trading["commission"]
        self.slippage_rate = trading["slippage"]
        self.max_position_size = trading["max_position_size"]
        self.risk_per_trade = trading["risk_per_trade"]

        # Risk parameters
        risk = self.config["risk"]
        self.stop_loss_pct = risk["stop_loss"]
        self.take_profit_pct = risk["take_profit"]
        self.max_drawdown_limit = risk["max_drawdown"]
        self.max_open_positions = risk["max_open_positions"]

        logger.info("Backtester initialized")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Commission: {self.commission_rate*100}%")
        logger.info(f"Slippage: {self.slippage_rate*100}%")

    def _calculate_shares(self, price: float, cash: float) -> int:
        """
        Calculate number of shares to buy.
        Uses position sizing based on max_position_size.
        
        Args:
            price: Current stock price
            cash: Available cash
            
        Returns:
            Number of shares to buy
        """
        # Maximum amount to spend on one position
        max_spend = cash * self.max_position_size

        # Account for commission and slippage
        effective_price = price * (1 + self.commission_rate + self.slippage_rate)

        # Calculate shares (round down)
        shares = int(max_spend / effective_price)

        return max(0, shares)

    def _apply_slippage(self, price: float, direction: str) -> float:
        """
        Apply slippage to price.
        
        When BUYING: price goes UP slightly (you pay more)
        When SELLING: price goes DOWN slightly (you receive less)
        """
        if direction == "BUY":
            return price * (1 + self.slippage_rate)
        else:  # SELL
            return price * (1 - self.slippage_rate)

    def _calculate_commission(self, price: float, shares: int) -> float:
        """Calculate commission for a trade."""
        return price * shares * self.commission_rate

    def run(self, df: pd.DataFrame, signal_column: str = "sma_signal",
            symbol: str = "UNKNOWN") -> dict:
        """
        Run backtest on a single stock.
        
        This is the MAIN method. It simulates trading day by day.
        
        Args:
            df: DataFrame with OHLCV data and signals
            signal_column: Column containing trading signals (1, -1, 0)
            symbol: Stock ticker
            
        Returns:
            Dictionary with:
            - 'portfolio_history': Daily portfolio values
            - 'trades': List of all trades
            - 'final_value': Final portfolio value
            - 'total_return': Total percentage return
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"🏃 Running Backtest: {symbol} | Strategy: {signal_column}")
        logger.info(f"{'='*50}")

        if signal_column not in df.columns:
            logger.error(f"Signal column '{signal_column}' not found!")
            return {}

        # --- Initialize Portfolio ---
        cash = self.initial_capital
        shares_held = 0
        entry_price = 0.0
        entry_date = None

        # Tracking lists
        trades: List[Trade] = []
        portfolio_history: List[PortfolioState] = []
        prev_total_value = self.initial_capital

        # --- Day by Day Simulation ---
        for date, row in df.iterrows():
            signal = row[signal_column]
            current_price = row["close"]
            high = row["high"]
            low = row["low"]

            # ===== CHECK STOP LOSS / TAKE PROFIT =====
            if shares_held > 0:
                current_return = (current_price - entry_price) / entry_price

                # Stop Loss Hit
                if low <= entry_price * (1 - self.stop_loss_pct):
                    exit_price = self._apply_slippage(
                        entry_price * (1 - self.stop_loss_pct), "SELL"
                    )
                    commission = self._calculate_commission(exit_price, shares_held)
                    cash += (exit_price * shares_held) - commission

                    # Record trade
                    pnl = (exit_price - entry_price) * shares_held - commission
                    pnl_pct = (exit_price - entry_price) / entry_price

                    trades.append(Trade(
                        symbol=symbol,
                        entry_date=entry_date,
                        entry_price=entry_price,
                        direction="LONG",
                        shares=shares_held,
                        exit_date=date,
                        exit_price=exit_price,
                        pnl=pnl,
                        pnl_percent=pnl_pct,
                        status="CLOSED",
                        commission_paid=commission
                    ))

                    logger.info(f"🛑 STOP LOSS: {date.strftime('%Y-%m-%d')} | "
                              f"Exit: ${exit_price:.2f} | PnL: ${pnl:.2f}")

                    shares_held = 0
                    entry_price = 0.0
                    entry_date = None

                # Take Profit Hit
                elif high >= entry_price * (1 + self.take_profit_pct):
                    exit_price = self._apply_slippage(
                        entry_price * (1 + self.take_profit_pct), "SELL"
                    )
                    commission = self._calculate_commission(exit_price, shares_held)
                    cash += (exit_price * shares_held) - commission

                    pnl = (exit_price - entry_price) * shares_held - commission
                    pnl_pct = (exit_price - entry_price) / entry_price

                    trades.append(Trade(
                        symbol=symbol,
                        entry_date=entry_date,
                        entry_price=entry_price,
                        direction="LONG",
                        shares=shares_held,
                        exit_date=date,
                        exit_price=exit_price,
                        pnl=pnl,
                        pnl_percent=pnl_pct,
                        status="CLOSED",
                        commission_paid=commission
                    ))

                    logger.info(f"🎯 TAKE PROFIT: {date.strftime('%Y-%m-%d')} | "
                              f"Exit: ${exit_price:.2f} | PnL: ${pnl:.2f}")

                    shares_held = 0
                    entry_price = 0.0
                    entry_date = None

            # ===== PROCESS SIGNALS =====

            # BUY SIGNAL
            if signal == 1 and shares_held == 0:
                buy_price = self._apply_slippage(current_price, "BUY")
                num_shares = self._calculate_shares(buy_price, cash)

                if num_shares > 0:
                    commission = self._calculate_commission(buy_price, num_shares)
                    total_cost = (buy_price * num_shares) + commission

                    if total_cost <= cash:
                        cash -= total_cost
                        shares_held = num_shares
                        entry_price = buy_price
                        entry_date = date

                        logger.info(f"🟢 BUY: {date.strftime('%Y-%m-%d')} | "
                                  f"Price: ${buy_price:.2f} | "
                                  f"Shares: {num_shares} | "
                                  f"Cost: ${total_cost:.2f}")

            # SELL SIGNAL
            elif signal == -1 and shares_held > 0:
                sell_price = self._apply_slippage(current_price, "SELL")
                commission = self._calculate_commission(sell_price, shares_held)
                cash += (sell_price * shares_held) - commission

                pnl = (sell_price - entry_price) * shares_held - commission
                pnl_pct = (sell_price - entry_price) / entry_price

                trades.append(Trade(
                    symbol=symbol,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    direction="LONG",
                    shares=shares_held,
                    exit_date=date,
                    exit_price=sell_price,
                    pnl=pnl,
                    pnl_percent=pnl_pct,
                    status="CLOSED",
                    commission_paid=commission
                ))

                logger.info(f"🔴 SELL: {date.strftime('%Y-%m-%d')} | "
                          f"Price: ${sell_price:.2f} | "
                          f"PnL: ${pnl:.2f} ({pnl_pct*100:.2f}%)")

                shares_held = 0
                entry_price = 0.0
                entry_date = None

            # ===== RECORD DAILY PORTFOLIO STATE =====
            holdings_value = shares_held * current_price
            total_value = cash + holdings_value
            daily_return = (total_value - prev_total_value) / prev_total_value if prev_total_value > 0 else 0

            portfolio_history.append(PortfolioState(
                date=date,
                cash=cash,
                holdings_value=holdings_value,
                total_value=total_value,
                num_positions=1 if shares_held > 0 else 0,
                daily_return=daily_return
            ))

            prev_total_value = total_value

        # --- Close any remaining open position ---
        if shares_held > 0:
            final_price = df["close"].iloc[-1]
            sell_price = self._apply_slippage(final_price, "SELL")
            commission = self._calculate_commission(sell_price, shares_held)
            cash += (sell_price * shares_held) - commission

            pnl = (sell_price - entry_price) * shares_held - commission
            pnl_pct = (sell_price - entry_price) / entry_price

            trades.append(Trade(
                symbol=symbol,
                entry_date=entry_date,
                entry_price=entry_price,
                direction="LONG",
                shares=shares_held,
                exit_date=df.index[-1],
                exit_price=sell_price,
                pnl=pnl,
                pnl_percent=pnl_pct,
                status="CLOSED",
                commission_paid=commission
            ))

            logger.info(f"📌 CLOSED REMAINING: PnL: ${pnl:.2f}")
            shares_held = 0

        # --- Build Results ---
        final_value = cash
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Convert portfolio history to DataFrame
        portfolio_df = pd.DataFrame([
            {
                "date": state.date,
                "cash": state.cash,
                "holdings_value": state.holdings_value,
                "total_value": state.total_value,
                "num_positions": state.num_positions,
                "daily_return": state.daily_return
            }
            for state in portfolio_history
        ])
        portfolio_df.set_index("date", inplace=True)

        # Convert trades to DataFrame
        trades_df = pd.DataFrame([
            {
                "symbol": t.symbol,
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "direction": t.direction,
                "pnl": t.pnl,
                "pnl_percent": t.pnl_percent,
                "commission": t.commission_paid,
                "status": t.status
            }
            for t in trades
        ])

        results = {
            "symbol": symbol,
            "strategy": signal_column,
            "portfolio_history": portfolio_df,
            "trades": trades_df,
            "final_value": final_value,
            "initial_capital": self.initial_capital,
            "total_return": total_return,
            "total_trades": len(trades),
        }

        # Summary
        logger.info(f"\n{'='*50}")
        logger.info(f"📊 BACKTEST RESULTS: {symbol} ({signal_column})")
        logger.info(f"{'='*50}")
        logger.info(f"Initial Capital:  ${self.initial_capital:>12,.2f}")
        logger.info(f"Final Value:      ${final_value:>12,.2f}")
        logger.info(f"Total Return:     {total_return*100:>11.2f}%")
        logger.info(f"Total Trades:     {len(trades):>12}")

        if len(trades) > 0:
            winning = [t for t in trades if t.pnl > 0]
            losing = [t for t in trades if t.pnl <= 0]
            logger.info(f"Winning Trades:   {len(winning):>12}")
            logger.info(f"Losing Trades:    {len(losing):>12}")
            logger.info(f"Win Rate:         {len(winning)/len(trades)*100:>11.2f}%")

        return results

    def run_with_risk_levels(self, df: pd.DataFrame, signal_column: str = "sma_signal",
                              symbol: str = "UNKNOWN",
                              risk_multipliers: list = None) -> dict:
        """
        Run backtest at multiple risk levels for the interactive risk bar.

        Scales max_position_size and risk_per_trade by each multiplier.
        Multiplier 1.0 = current config (baseline).

        Args:
            df: DataFrame with OHLCV data and signals
            signal_column: Column containing trading signals
            symbol: Stock ticker
            risk_multipliers: List of floats, e.g. [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

        Returns:
            Dictionary of {multiplier: backtest_results}
        """
        if risk_multipliers is None:
            risk_multipliers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

        # Save original config
        orig_max_position = self.max_position_size
        orig_risk_per_trade = self.risk_per_trade

        all_risk_results = {}

        for mult in risk_multipliers:
            # Scale risk parameters (cap position size at 95% to stay realistic)
            self.max_position_size = min(orig_max_position * mult, 0.95)
            self.risk_per_trade = min(orig_risk_per_trade * mult, 0.20)

            logger.info(f"Running risk level {mult:.2f}x "
                       f"(pos_size={self.max_position_size*100:.1f}%, "
                       f"risk/trade={self.risk_per_trade*100:.1f}%)")

            results = self.run(df, signal_column=signal_column, symbol=symbol)
            if results:
                results["risk_multiplier"] = mult
                results["position_size_pct"] = self.max_position_size * 100
                results["risk_per_trade_pct"] = self.risk_per_trade * 100
                all_risk_results[mult] = results

        # Restore original config
        self.max_position_size = orig_max_position
        self.risk_per_trade = orig_risk_per_trade

        logger.info(f"Completed {len(all_risk_results)} risk-level backtests")
        return all_risk_results

    def run_multiple_strategies(self, df: pd.DataFrame, symbol: str,
                                 strategies: list = None) -> dict:
        """
        Run backtest for multiple strategies on the same stock.
        
        Args:
            df: DataFrame with signals
            symbol: Stock ticker
            strategies: List of signal column names
            
        Returns:
            Dictionary of {strategy_name: results}
        """
        if strategies is None:
            strategies = [
                "sma_signal",
                "rsi_signal",
                "macd_trade_signal",
                "bb_signal",
                "combined_signal"
            ]

        all_results = {}

        for strat in strategies:
            if strat in df.columns:
                results = self.run(df, signal_column=strat, symbol=symbol)
                if results:
                    all_results[strat] = results
            else:
                logger.warning(f"Strategy column '{strat}' not found — skipping")

        return all_results

    def compare_strategies(self, all_results: dict) -> pd.DataFrame:
        """
        Compare results from multiple strategies.
        
        Args:
            all_results: Dictionary from run_multiple_strategies()
            
        Returns:
            Comparison DataFrame
        """
        comparison = []

        for strategy_name, results in all_results.items():
            trades_df = results["trades"]

            row = {
                "strategy": strategy_name,
                "final_value": results["final_value"],
                "total_return": f"{results['total_return']*100:.2f}%",
                "total_trades": results["total_trades"],
            }

            if not trades_df.empty:
                winning = trades_df[trades_df["pnl"] > 0]
                losing = trades_df[trades_df["pnl"] <= 0]

                row["winning_trades"] = len(winning)
                row["losing_trades"] = len(losing)
                row["win_rate"] = f"{len(winning)/len(trades_df)*100:.1f}%"
                row["avg_win"] = f"${winning['pnl'].mean():.2f}" if len(winning) > 0 else "\$0.00"
                row["avg_loss"] = f"${losing['pnl'].mean():.2f}" if len(losing) > 0 else "\$0.00"
                row["total_commission"] = f"${trades_df['commission'].sum():.2f}"
                row["best_trade"] = f"${trades_df['pnl'].max():.2f}"
                row["worst_trade"] = f"${trades_df['pnl'].min():.2f}"
            else:
                row["winning_trades"] = 0
                row["losing_trades"] = 0
                row["win_rate"] = "0%"

            comparison.append(row)

        comp_df = pd.DataFrame(comparison)
        comp_df = comp_df.sort_values("final_value", ascending=False)

        return comp_df