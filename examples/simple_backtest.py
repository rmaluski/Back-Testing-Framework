#!/usr/bin/env python3
"""
Simple backtest example using the trading framework.

This example demonstrates how to:
1. Create a simple strategy
2. Load market data
3. Run a backtest
4. Generate reports
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading_core import StrategyBase, MarketEvent
from trading_core.engine import BacktestEngine
from trading_core.config import Config, DataConfig, StrategyConfig, ExecutionConfig, RiskConfig, ReportingConfig


class SimpleMovingAverageStrategy(StrategyBase):
    """
    Simple moving average crossover strategy.
    
    This strategy:
    - Calculates short and long moving averages
    - Buys when short MA crosses above long MA
    - Sells when short MA crosses below long MA
    """
    
    def __init__(self, short_window=10, long_window=30, **kwargs):
        super().__init__(**kwargs)
        self.short_window = short_window
        self.long_window = long_window
        self.price_history = {}
        self.positions = {}
    
    def initialize(self):
        """Initialize strategy state."""
        for symbol in self.symbols:
            self.price_history[symbol] = []
            self.positions[symbol] = 0
    
    def on_market_event(self, event: MarketEvent):
        """Handle market events and generate signals."""
        symbol = event.symbol
        
        # Get current price
        if event.close is not None:
            current_price = event.close
        elif event.mid_price is not None:
            current_price = event.mid_price
        else:
            return
        
        # Update price history
        self.price_history[symbol].append(current_price)
        
        # Keep only recent prices
        max_window = max(self.short_window, self.long_window)
        if len(self.price_history[symbol]) > max_window * 2:
            self.price_history[symbol] = self.price_history[symbol][-max_window:]
        
        # Need enough data
        if len(self.price_history[symbol]) < max_window:
            return
        
        # Calculate moving averages
        prices = np.array(self.price_history[symbol])
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])
        
        # Generate signals
        current_position = self.positions[symbol]
        
        # Buy signal: short MA crosses above long MA
        if short_ma > long_ma and current_position <= 0:
            quantity = self._calculate_position_size(current_price)
            if quantity > 0:
                self.buy(symbol, quantity)
                self.positions[symbol] = quantity
                print(f"BUY {quantity} {symbol} @ {current_price:.2f}")
        
        # Sell signal: short MA crosses below long MA
        elif short_ma < long_ma and current_position >= 0:
            if current_position > 0:
                self.sell(symbol, current_position)
                self.positions[symbol] = 0
                print(f"SELL {current_position} {symbol} @ {current_price:.2f}")
    
    def _calculate_position_size(self, price: float) -> int:
        """Calculate position size."""
        position_value = self.current_capital * 0.1  # 10% of capital
        quantity = int(position_value / price)
        return max(1, min(quantity, 100))  # Between 1 and 100


def create_sample_data():
    """Create sample market data for testing."""
    # Generate 1 year of daily data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Create synthetic price data with trend and noise
    np.random.seed(42)  # For reproducible results
    
    # Start price
    price = 100.0
    prices = []
    
    for _ in range(len(dates)):
        # Add trend and noise
        trend = 0.0001  # Small upward trend
        noise = np.random.normal(0, 0.02)  # 2% daily volatility
        
        price = price * (1 + trend + noise)
        prices.append(price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'AAPL',
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    return df


def run_simple_backtest():
    """Run a simple backtest example."""
    print("Creating sample market data...")
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Save to Parquet file
    os.makedirs('data', exist_ok=True)
    sample_data.to_parquet('data/sample_data.parquet', index=False)
    
    print(f"Created {len(sample_data)} data points")
    
    # Create configuration
    config = Config(
        data=DataConfig(
            source="data/sample_data.parquet",
            start_date="2022-01-01",
            end_date="2022-12-31",
            symbols=["AAPL"],
            bar_size="1D"
        ),
        strategy=StrategyConfig(
            class_name="SimpleMovingAverageStrategy",
            module_path="examples.simple_backtest",
            params={
                "short_window": 10,
                "long_window": 30
            },
            initial_capital=100000.0
        ),
        execution=ExecutionConfig(
            slippage_bps=1.0,
            commission_bps=0.5,
            min_tick_size=0.01
        ),
        risk=RiskConfig(
            max_position_size=100,
            max_drawdown=0.15,
            position_limit_pct=0.1
        ),
        reporting=ReportingConfig(
            output_dir="./results/simple_example",
            save_equity_curve=True,
            save_fills=True,
            save_risk_metrics=True,
            generate_html_report=True
        )
    )
    
    print("Running backtest...")
    
    # Create engine
    engine = BacktestEngine(config)
    
    # Create strategy
    strategy = SimpleMovingAverageStrategy(
        short_window=10,
        long_window=30,
        initial_capital=100000.0,
        symbols=["AAPL"]
    )
    
    # Set strategy
    engine.set_strategy(strategy)
    
    # Run backtest
    results = engine.run()
    
    # Print results
    print("\nBacktest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    
    # Generate reports
    from trading_core.reporter import Reporter
    reporter = Reporter(config, results)
    reporter.generate_reports()
    
    print(f"\nResults saved to: {config.reporting.output_dir}")
    print("Open report.html in your browser to view the detailed report.")


if __name__ == "__main__":
    run_simple_backtest() 