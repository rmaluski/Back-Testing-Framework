"""
Main backtesting engine for the framework.

This module contains the core engine that orchestrates the backtesting simulation.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from .config import Config
from .strategy import StrategyBase
from .events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from .orders import Order, OrderType, Side
from .portfolio import Portfolio


class BacktestEngine:
    """
    Main backtesting engine.
    
    This class orchestrates the entire backtesting simulation, including
    data loading, event processing, order execution, and performance tracking.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the backtesting engine.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.strategy: Optional[StrategyBase] = None
        self.portfolio: Optional[Portfolio] = None
        
        # Data storage
        self.market_data: Optional[pd.DataFrame] = None
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.start_time = None
        self.end_time = None
        self.total_events = 0
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize engine components."""
        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_capital=self.config.strategy.initial_capital,
            symbols=self.config.data.symbols,
            slippage_bps=self.config.execution.slippage_bps,
            commission_bps=self.config.execution.commission_bps
        )
    
    def set_strategy(self, strategy: StrategyBase):
        """
        Set the strategy for the backtest.
        
        Args:
            strategy: Strategy instance
        """
        self.strategy = strategy
        self.strategy.portfolio = self.portfolio
    
    def load_data(self) -> pd.DataFrame:
        """
        Load market data from the configured source.
        
        Returns:
            DataFrame with market data
        """
        source = self.config.data.source
        
        if source.endswith('.parquet'):
            # Load from Parquet file
            df = pd.read_parquet(source)
        elif source.endswith('.csv'):
            # Load from CSV file
            df = pd.read_csv(source)
        else:
            raise ValueError(f"Unsupported data source format: {source}")
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'symbol']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by date range
        start_date = pd.to_datetime(self.config.data.start_date)
        end_date = pd.to_datetime(self.config.data.end_date)
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        # Filter by symbols
        if self.config.data.symbols:
            df = df[df['symbol'].isin(self.config.data.symbols)]
        
        # Sort by timestamp
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        
        return df
    
    def run(self) -> Dict[str, Any]:
        """
        Run the backtest simulation.
        
        Returns:
            Dictionary with backtest results
        """
        if self.strategy is None:
            raise ValueError("Strategy not set. Call set_strategy() first.")
        
        print(f"Starting backtest: {self.config.strategy.class_name}")
        print(f"Date range: {self.config.data.start_date} to {self.config.data.end_date}")
        print(f"Symbols: {self.config.data.symbols}")
        print(f"Initial capital: ${self.config.strategy.initial_capital:,.2f}")
        
        # Load market data
        print("Loading market data...")
        self.market_data = self.load_data()
        print(f"Loaded {len(self.market_data)} data points")
        
        # Initialize strategy
        self.strategy.on_start()
        
        # Process market data
        print("Processing market data...")
        self.start_time = time.time()
        
        # Group data by timestamp for bar processing
        if self.config.data.bar_size:
            # Process as bars
            self._process_bars()
        else:
            # Process as ticks
            self._process_ticks()
        
        self.end_time = time.time()
        
        # Finalize strategy
        self.strategy.on_end()
        
        # Calculate final results
        results = self._calculate_results()
        
        print(f"Backtest completed in {self.end_time - self.start_time:.2f} seconds")
        print(f"Total return: {results['total_return']:.2%}")
        print(f"Max drawdown: {results['max_drawdown']:.2%}")
        print(f"Total trades: {results['total_trades']}")
        
        return results
    
    def _process_ticks(self):
        """Process market data as individual ticks."""
        for _, row in self.market_data.iterrows():
            # Create market event
            event = MarketEvent(
                timestamp=row['timestamp'],
                symbol=row['symbol'],
                bid=row.get('bid'),
                ask=row.get('ask'),
                bid_size=row.get('bid_size'),
                ask_size=row.get('ask_size'),
                last=row.get('last'),
                last_size=row.get('last_size'),
                volume=row.get('volume')
            )
            
            # Update current time
            self.strategy.current_time = event.timestamp
            
            # Process event
            self._process_market_event(event)
            self.total_events += 1
    
    def _process_bars(self):
        """Process market data as bars."""
        # Resample data to bars
        bar_size = self.config.data.bar_size
        
        for symbol in self.config.data.symbols:
            symbol_data = self.market_data[self.market_data['symbol'] == symbol].copy()
            
            if len(symbol_data) == 0:
                continue
            
            # Set timestamp as index for resampling
            symbol_data = symbol_data.set_index('timestamp')
            
            # Resample to bars
            if bar_size == '1min':
                bars = symbol_data.resample('1T').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            elif bar_size == '5min':
                bars = symbol_data.resample('5T').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            elif bar_size == '1D':
                bars = symbol_data.resample('D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            else:
                raise ValueError(f"Unsupported bar size: {bar_size}")
            
            # Process each bar
            for timestamp, bar in bars.iterrows():
                event = MarketEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    open=bar.get('open'),
                    high=bar.get('high'),
                    low=bar.get('low'),
                    close=bar.get('close'),
                    volume=bar.get('volume')
                )
                
                # Update current time
                self.strategy.current_time = event.timestamp
                
                # Process event
                self._process_market_event(event)
                self.total_events += 1
    
    def _process_market_event(self, event: MarketEvent):
        """
        Process a market event.
        
        Args:
            event: Market event to process
        """
        # Update portfolio with current market data
        if event.close is not None:
            self.portfolio.update_position_value(event.symbol, event.close)
        
        # Call strategy
        self.strategy.on_market_event(event)
        
        # Process any orders generated by strategy
        self._process_strategy_orders()
        
        # Update equity curve
        self._update_equity_curve(event.timestamp)
    
    def _process_strategy_orders(self):
        """Process orders generated by the strategy."""
        if not self.strategy.orders:
            return
        
        # Process each order
        for order in self.strategy.orders:
            # Execute order
            fill = self.portfolio.execute_order(order)
            
            if fill:
                # Create fill event
                fill_event = FillEvent(
                    timestamp=order.timestamp,
                    symbol=order.symbol,
                    quantity=fill['quantity'],
                    side=order.side.value,
                    fill_price=fill['price'],
                    commission=fill['commission'],
                    order_id=order.order_id,
                    strategy_id=order.strategy_id
                )
                
                # Notify strategy
                self.strategy.on_fill_event(fill_event)
                
                # Record trade
                self.trades.append({
                    'timestamp': fill_event.timestamp,
                    'symbol': fill_event.symbol,
                    'side': fill_event.side,
                    'quantity': fill_event.quantity,
                    'price': fill_event.fill_price,
                    'commission': fill_event.commission,
                    'order_id': fill_event.order_id
                })
        
        # Clear processed orders
        self.strategy.orders = []
    
    def _update_equity_curve(self, timestamp: datetime):
        """Update equity curve with current portfolio value."""
        portfolio_value = self.portfolio.get_total_value()
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.portfolio.cash,
            'positions_value': portfolio_value - self.portfolio.cash
        })
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate final backtest results."""
        if not self.equity_curve:
            return {
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df = equity_df.set_index('timestamp')
        
        # Calculate returns
        equity_df['returns'] = equity_df['portfolio_value'].pct_change().fillna(0)
        
        # Calculate metrics
        initial_value = self.config.strategy.initial_capital
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate max drawdown
        rolling_max = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Calculate Sharpe ratio (assuming 252 trading days)
        daily_returns = equity_df['returns'].resample('D').sum()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Calculate win rate
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            if len(trades_df) > 0:
                # Simple win rate calculation (can be improved)
                win_rate = 0.5  # Placeholder
            else:
                win_rate = 0.0
        else:
            win_rate = 0.0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'initial_capital': initial_value,
            'final_capital': final_value,
            'equity_curve': equity_df,
            'trades': self.trades,
            'execution_time': self.end_time - self.start_time,
            'total_events': self.total_events
        }
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get the equity curve as a DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp')
    
    def get_trades(self) -> pd.DataFrame:
        """Get the trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp') 