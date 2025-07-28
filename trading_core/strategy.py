"""
Strategy base class for the backtesting framework.

This module provides the base class that all trading strategies inherit from.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from .orders import Order, OrderType, Side, TimeInForce


class StrategyBase(ABC):
    """
    Base class for all trading strategies.
    
    This class provides the interface and common functionality for strategies.
    Strategies should inherit from this class and implement the required methods.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 symbols: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize the strategy.
        
        Args:
            initial_capital: Starting capital for the strategy
            symbols: List of symbols to trade
            **kwargs: Additional strategy-specific parameters
        """
        self.initial_capital = initial_capital
        self.symbols = symbols or []
        self.current_capital = initial_capital
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.orders = []
        self.fills = []
        self.equity_curve = []
        
        # Strategy state
        self.is_running = False
        self.current_time = None
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
        
        # Store strategy parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def initialize(self):
        """
        Initialize strategy state before backtesting begins.
        
        Override this method to set up strategy-specific initialization.
        """
        pass
    
    def on_start(self):
        """
        Called when the backtest starts.
        
        Override this method to perform any start-of-backtest logic.
        """
        self.is_running = True
        self.initialize()
    
    def on_end(self):
        """
        Called when the backtest ends.
        
        Override this method to perform any end-of-backtest logic.
        """
        self.is_running = False
    
    @abstractmethod
    def on_market_event(self, event: MarketEvent):
        """
        Handle market data events (ticks or bars).
        
        This is the main method that strategies should implement.
        
        Args:
            event: Market data event
        """
        pass
    
    def on_signal_event(self, event: SignalEvent):
        """
        Handle signal events.
        
        Args:
            event: Signal event
        """
        pass
    
    def on_order_event(self, event: OrderEvent):
        """
        Handle order events.
        
        Args:
            event: Order event
        """
        pass
    
    def on_fill_event(self, event: FillEvent):
        """
        Handle fill events.
        
        Args:
            event: Fill event
        """
        # Update positions
        if event.side == "BUY":
            self.positions[event.symbol] += event.quantity
        else:
            self.positions[event.symbol] -= event.quantity
        
        # Update capital
        cost = event.cost
        if event.side == "BUY":
            self.current_capital -= cost
        else:
            self.current_capital += cost
        
        # Track fills
        self.fills.append(event)
        self.total_trades += 1
        
        # Update performance metrics
        self._update_performance_metrics()
    
    def buy(self, 
            symbol: str, 
            quantity: int, 
            order_type: Union[OrderType, str] = OrderType.MARKET,
            limit_price: Optional[float] = None,
            **kwargs) -> Order:
        """
        Place a buy order.
        
        Args:
            symbol: Symbol to buy
            quantity: Quantity to buy
            order_type: Type of order
            limit_price: Limit price (for limit orders)
            **kwargs: Additional order parameters
            
        Returns:
            Order object
        """
        order = Order(
            symbol=symbol,
            order_type=order_type,
            side=Side.BUY,
            quantity=quantity,
            limit_price=limit_price,
            strategy_id=self.__class__.__name__,
            **kwargs
        )
        self.orders.append(order)
        return order
    
    def sell(self, 
             symbol: str, 
             quantity: int, 
             order_type: Union[OrderType, str] = OrderType.MARKET,
             limit_price: Optional[float] = None,
             **kwargs) -> Order:
        """
        Place a sell order.
        
        Args:
            symbol: Symbol to sell
            quantity: Quantity to sell
            order_type: Type of order
            limit_price: Limit price (for limit orders)
            **kwargs: Additional order parameters
            
        Returns:
            Order object
        """
        order = Order(
            symbol=symbol,
            order_type=order_type,
            side=Side.SELL,
            quantity=quantity,
            limit_price=limit_price,
            strategy_id=self.__class__.__name__,
            **kwargs
        )
        self.orders.append(order)
        return order
    
    def close_position(self, symbol: str, **kwargs) -> Optional[Order]:
        """
        Close all positions in a symbol.
        
        Args:
            symbol: Symbol to close position in
            **kwargs: Additional order parameters
            
        Returns:
            Order object if position exists, None otherwise
        """
        position = self.positions.get(symbol, 0)
        if position == 0:
            return None
        
        if position > 0:
            return self.sell(symbol, abs(position), **kwargs)
        else:
            return self.buy(symbol, abs(position), **kwargs)
    
    def get_position(self, symbol: str) -> int:
        """
        Get current position in a symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Current position (positive for long, negative for short)
        """
        return self.positions.get(symbol, 0)
    
    def get_total_position_value(self, symbol: str, current_price: float) -> float:
        """
        Get total value of position in a symbol.
        
        Args:
            symbol: Symbol to get position value for
            current_price: Current price of the symbol
            
        Returns:
            Total position value
        """
        position = self.get_position(symbol)
        return abs(position * current_price)
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Get total portfolio value.
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Total portfolio value
        """
        portfolio_value = self.current_capital
        
        for symbol, position in self.positions.items():
            if position != 0 and symbol in current_prices:
                portfolio_value += position * current_prices[symbol]
        
        return portfolio_value
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics."""
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Calculate drawdown
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        # Update equity curve
        self.equity_curve.append({
            'timestamp': self.current_time,
            'capital': self.current_capital,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.equity_curve:
            return {}
        
        initial_capital = self.initial_capital
        final_capital = self.current_capital
        total_return = (final_capital - initial_capital) / initial_capital
        
        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'max_drawdown': self.max_drawdown,
            'total_pnl': self.total_pnl,
            'positions': self.positions.copy()
        }
    
    def reset(self):
        """Reset strategy state for a new backtest."""
        self.current_capital = self.initial_capital
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.orders = []
        self.fills = []
        self.equity_curve = []
        self.is_running = False
        self.current_time = None
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = self.initial_capital 