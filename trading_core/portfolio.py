"""
Portfolio management for the backtesting framework.

This module handles position tracking, cash management, and order execution.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .orders import Order, OrderType, Side


class Portfolio:
    """
    Portfolio management class.
    
    This class handles position tracking, cash management, and order execution
    with realistic slippage and commission models.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 symbols: Optional[List[str]] = None,
                 slippage_bps: float = 1.0,
                 commission_bps: float = 0.5):
        """
        Initialize the portfolio.
        
        Args:
            initial_capital: Starting capital
            symbols: List of symbols to track
            slippage_bps: Slippage in basis points
            commission_bps: Commission in basis points
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.symbols = symbols or []
        
        # Position tracking
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.position_values = {symbol: 0.0 for symbol in self.symbols}
        self.current_prices = {symbol: 0.0 for symbol in self.symbols}
        
        # Execution parameters
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        
        # Performance tracking
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Trade history
        self.trades = []
    
    def update_position_value(self, symbol: str, price: float):
        """
        Update the current price for a symbol.
        
        Args:
            symbol: Symbol to update
            price: Current price
        """
        if symbol in self.symbols:
            self.current_prices[symbol] = price
            self.position_values[symbol] = self.positions[symbol] * price
    
    def get_position(self, symbol: str) -> int:
        """
        Get current position in a symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Current position (positive for long, negative for short)
        """
        return self.positions.get(symbol, 0)
    
    def get_position_value(self, symbol: str) -> float:
        """
        Get current position value in a symbol.
        
        Args:
            symbol: Symbol to get position value for
            
        Returns:
            Current position value
        """
        return self.position_values.get(symbol, 0.0)
    
    def get_total_position_value(self) -> float:
        """
        Get total value of all positions.
        
        Returns:
            Total position value
        """
        return sum(self.position_values.values())
    
    def get_total_value(self) -> float:
        """
        Get total portfolio value (cash + positions).
        
        Returns:
            Total portfolio value
        """
        return self.cash + self.get_total_position_value()
    
    def execute_order(self, order: Order) -> Optional[Dict[str, Any]]:
        """
        Execute an order and return fill information.
        
        Args:
            order: Order to execute
            
        Returns:
            Fill information or None if order cannot be executed
        """
        if order.symbol not in self.symbols:
            return None
        
        current_price = self.current_prices.get(order.symbol, 0.0)
        if current_price <= 0:
            return None
        
        # Calculate fill price with slippage
        fill_price = self._calculate_fill_price(order, current_price)
        
        # Calculate commission
        commission = self._calculate_commission(order, fill_price)
        
        # Check if we have enough cash for buy orders
        if order.side == Side.BUY:
            total_cost = order.quantity * fill_price + commission
            if total_cost > self.cash:
                return None
        
        # Execute the trade
        self._execute_trade(order, fill_price, commission)
        
        return {
            'quantity': order.quantity,
            'price': fill_price,
            'commission': commission,
            'timestamp': order.timestamp
        }
    
    def _calculate_fill_price(self, order: Order, current_price: float) -> float:
        """
        Calculate fill price with slippage.
        
        Args:
            order: Order to calculate fill price for
            current_price: Current market price
            
        Returns:
            Fill price including slippage
        """
        slippage_multiplier = self.slippage_bps / 10000.0
        
        if order.order_type == OrderType.MARKET:
            # Market orders get filled at current price with slippage
            if order.side == Side.BUY:
                # Buy orders pay slightly more (ask side)
                fill_price = current_price * (1 + slippage_multiplier)
            else:
                # Sell orders receive slightly less (bid side)
                fill_price = current_price * (1 - slippage_multiplier)
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders get filled at limit price if marketable
            if order.side == Side.BUY and order.limit_price >= current_price:
                fill_price = order.limit_price
            elif order.side == Side.SELL and order.limit_price <= current_price:
                fill_price = order.limit_price
            else:
                # Order not marketable
                return 0.0
        
        else:
            # Other order types not implemented yet
            return 0.0
        
        return fill_price
    
    def _calculate_commission(self, order: Order, fill_price: float) -> float:
        """
        Calculate commission for an order.
        
        Args:
            order: Order to calculate commission for
            fill_price: Fill price of the order
            
        Returns:
            Commission amount
        """
        trade_value = order.quantity * fill_price
        commission = trade_value * (self.commission_bps / 10000.0)
        return commission
    
    def _execute_trade(self, order: Order, fill_price: float, commission: float):
        """
        Execute a trade and update portfolio.
        
        Args:
            order: Order being executed
            fill_price: Fill price
            commission: Commission amount
        """
        symbol = order.symbol
        quantity = order.quantity
        side = order.side
        
        # Update position
        if side == Side.BUY:
            self.positions[symbol] += quantity
            self.cash -= (quantity * fill_price + commission)
        else:
            self.positions[symbol] -= quantity
            self.cash += (quantity * fill_price - commission)
        
        # Update position value
        self.position_values[symbol] = self.positions[symbol] * fill_price
        
        # Record trade
        self.trades.append({
            'timestamp': order.timestamp,
            'symbol': symbol,
            'side': side.value,
            'quantity': quantity,
            'price': fill_price,
            'commission': commission,
            'order_id': order.order_id
        })
    
    def get_pnl(self) -> Dict[str, float]:
        """
        Get current P&L breakdown.
        
        Returns:
            Dictionary with P&L components
        """
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        for symbol in self.symbols:
            position = self.positions[symbol]
            current_price = self.current_prices[symbol]
            
            # This is a simplified calculation - in practice you'd track cost basis
            if position != 0 and current_price > 0:
                # For simplicity, assume average cost basis
                unrealized_pnl += position * current_price
        
        self.unrealized_pnl = unrealized_pnl
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        
        return {
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl
        }
    
    def get_exposure(self) -> Dict[str, float]:
        """
        Get current portfolio exposure by symbol.
        
        Returns:
            Dictionary with exposure by symbol
        """
        total_value = self.get_total_value()
        if total_value <= 0:
            return {}
        
        exposure = {}
        for symbol in self.symbols:
            position_value = self.position_values[symbol]
            exposure[symbol] = position_value / total_value
        
        return exposure
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate basic risk metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        total_value = self.get_total_value()
        initial_value = self.initial_capital
        
        # Calculate return
        total_return = (total_value - initial_value) / initial_value if initial_value > 0 else 0.0
        
        # Calculate leverage
        total_position_value = self.get_total_position_value()
        leverage = total_position_value / total_value if total_value > 0 else 0.0
        
        # Calculate concentration (max position as % of total)
        max_position_pct = 0.0
        for symbol in self.symbols:
            position_pct = abs(self.position_values[symbol]) / total_value if total_value > 0 else 0.0
            max_position_pct = max(max_position_pct, position_pct)
        
        return {
            'total_return': total_return,
            'leverage': leverage,
            'max_position_pct': max_position_pct,
            'cash_pct': self.cash / total_value if total_value > 0 else 1.0
        }
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.position_values = {symbol: 0.0 for symbol in self.symbols}
        self.current_prices = {symbol: 0.0 for symbol in self.symbols}
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.trades = []
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary.
        
        Returns:
            Dictionary with portfolio summary
        """
        return {
            'initial_capital': self.initial_capital,
            'current_cash': self.cash,
            'total_position_value': self.get_total_position_value(),
            'total_value': self.get_total_value(),
            'positions': self.positions.copy(),
            'exposure': self.get_exposure(),
            'risk_metrics': self.get_risk_metrics(),
            'total_trades': len(self.trades)
        } 