"""
High-performance event-driven backtesting framework.

This package provides a C++ core with Python bindings for quantitative trading research.
"""

__version__ = "0.1.0"
__author__ = "Ryan Maluski"
__email__ = "ryan.maluski@example.com"

# Core imports
try:
    from ._core import (
        EventLoop,
        MarketDataFeed,
        OrderBook,
        FillEngine,
        Portfolio,
        StrategyAPI,
        MarketEvent,
        SignalEvent,
        OrderEvent,
        FillEvent,
    )
except ImportError:
    # Fallback for when C++ extensions aren't built
    pass

# Python layer imports
from .strategy import StrategyBase
from .events import (
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    EventType,
)
from .orders import (
    Order,
    OrderType,
    TimeInForce,
    Side,
)
from .portfolio import Portfolio
from .reporter import Reporter
from .config import Config

# CLI
from .cli import main

__all__ = [
    # Core C++ components
    "EventLoop",
    "MarketDataFeed", 
    "OrderBook",
    "FillEngine",
    "Portfolio",
    "StrategyAPI",
    
    # Python layer
    "StrategyBase",
    "MarketEvent",
    "SignalEvent", 
    "OrderEvent",
    "FillEvent",
    "EventType",
    "Order",
    "OrderType",
    "TimeInForce",
    "Side",
    "Portfolio",
    "Reporter",
    "Config",
    
    # CLI
    "main",
] 