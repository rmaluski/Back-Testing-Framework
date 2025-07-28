"""
Order management for the backtesting framework.

This module defines order types, sides, and time-in-force options.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Union


class OrderType(Enum):
    """Types of orders."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    ICEBERG = "ICEBERG"


class Side(Enum):
    """Order sides."""
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    """Time in force options."""
    DAY = "DAY"  # Good for the day
    GTC = "GTC"  # Good till cancelled
    IOC = "IOC"  # Immediate or cancel
    FOK = "FOK"  # Fill or kill


@dataclass
class Order:
    """Order representation."""
    symbol: str
    order_type: OrderType
    side: Side
    quantity: int
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    strategy_id: Optional[str] = None
    timestamp: Optional[Union[datetime, str]] = None
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type)
        if isinstance(self.side, str):
            self.side = Side(self.side)
        if isinstance(self.time_in_force, str):
            self.time_in_force = TimeInForce(self.time_in_force)
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        elif self.timestamp is None:
            self.timestamp = datetime.now()
        if self.order_id is None:
            self.order_id = f"{self.symbol}_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        if self.metadata is None:
            self.metadata = {}
    
    def __str__(self) -> str:
        return f"{self.side.value} {self.quantity} {self.symbol} @ {self.order_type.value}"
    
    def __repr__(self) -> str:
        return f"Order({self.symbol}, {self.order_type.value}, {self.side.value}, {self.quantity})"


# Order factory functions
def create_market_order(
    symbol: str,
    side: Union[Side, str],
    quantity: int,
    **kwargs
) -> Order:
    """Create a market order."""
    return Order(
        symbol=symbol,
        order_type=OrderType.MARKET,
        side=side,
        quantity=quantity,
        **kwargs
    )


def create_limit_order(
    symbol: str,
    side: Union[Side, str],
    quantity: int,
    limit_price: float,
    **kwargs
) -> Order:
    """Create a limit order."""
    return Order(
        symbol=symbol,
        order_type=OrderType.LIMIT,
        side=side,
        quantity=quantity,
        limit_price=limit_price,
        **kwargs
    )


def create_stop_order(
    symbol: str,
    side: Union[Side, str],
    quantity: int,
    stop_price: float,
    **kwargs
) -> Order:
    """Create a stop order."""
    return Order(
        symbol=symbol,
        order_type=OrderType.STOP,
        side=side,
        quantity=quantity,
        stop_price=stop_price,
        **kwargs
    )


def create_stop_limit_order(
    symbol: str,
    side: Union[Side, str],
    quantity: int,
    stop_price: float,
    limit_price: float,
    **kwargs
) -> Order:
    """Create a stop-limit order."""
    return Order(
        symbol=symbol,
        order_type=OrderType.STOP_LIMIT,
        side=side,
        quantity=quantity,
        stop_price=stop_price,
        limit_price=limit_price,
        **kwargs
    )


# Order validation
def validate_order(order: Order) -> bool:
    """Validate order parameters."""
    if order.quantity <= 0:
        return False
    
    if order.order_type == OrderType.LIMIT and order.limit_price is None:
        return False
    
    if order.order_type == OrderType.STOP and order.stop_price is None:
        return False
    
    if order.order_type == OrderType.STOP_LIMIT:
        if order.stop_price is None or order.limit_price is None:
            return False
        if order.side == Side.BUY and order.limit_price <= order.stop_price:
            return False
        if order.side == Side.SELL and order.limit_price >= order.stop_price:
            return False
    
    return True


# Order comparison
def order_matches(order1: Order, order2: Order) -> bool:
    """Check if two orders match for execution."""
    if order1.symbol != order2.symbol:
        return False
    
    if order1.side == order2.side:
        return False  # Same side orders don't match
    
    if order1.order_type == OrderType.MARKET or order2.order_type == OrderType.MARKET:
        return True
    
    if order1.order_type == OrderType.LIMIT and order2.order_type == OrderType.LIMIT:
        if order1.side == Side.BUY:
            return order1.limit_price >= order2.limit_price
        else:
            return order1.limit_price <= order2.limit_price
    
    return False 