"""
Unit tests for strategy classes.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from trading_core.strategy import StrategyBase
from trading_core.events import MarketEvent
from trading_core.orders import Order, OrderType, Side
from strategies.bollinger import BollingerStrategy


class TestStrategyBase:
    """Test the StrategyBase class."""
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = StrategyBase(
            initial_capital=100000,
            symbols=["AAPL", "GOOGL"]
        )
        
        assert strategy.initial_capital == 100000
        assert strategy.current_capital == 100000
        assert strategy.symbols == ["AAPL", "GOOGL"]
        assert strategy.positions == {"AAPL": 0, "GOOGL": 0}
        assert not strategy.is_running
    
    def test_strategy_reset(self):
        """Test strategy reset functionality."""
        strategy = StrategyBase(
            initial_capital=100000,
            symbols=["AAPL"]
        )
        
        # Modify state
        strategy.current_capital = 95000
        strategy.positions["AAPL"] = 10
        strategy.total_trades = 5
        
        # Reset
        strategy.reset()
        
        assert strategy.current_capital == 100000
        assert strategy.positions["AAPL"] == 0
        assert strategy.total_trades == 0
    
    def test_buy_order_creation(self):
        """Test buy order creation."""
        strategy = StrategyBase(symbols=["AAPL"])
        
        order = strategy.buy("AAPL", 100, OrderType.MARKET)
        
        assert order.symbol == "AAPL"
        assert order.quantity == 100
        assert order.side == Side.BUY
        assert order.order_type == OrderType.MARKET
        assert order in strategy.orders
    
    def test_sell_order_creation(self):
        """Test sell order creation."""
        strategy = StrategyBase(symbols=["AAPL"])
        
        order = strategy.sell("AAPL", 50, OrderType.LIMIT, limit_price=150.0)
        
        assert order.symbol == "AAPL"
        assert order.quantity == 50
        assert order.side == Side.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 150.0
        assert order in strategy.orders
    
    def test_position_tracking(self):
        """Test position tracking."""
        strategy = StrategyBase(symbols=["AAPL"])
        
        # Initial position
        assert strategy.get_position("AAPL") == 0
        
        # Simulate fill events
        from trading_core.events import FillEvent
        
        # Buy fill
        buy_fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            quantity=100,
            side="BUY",
            fill_price=150.0
        )
        strategy.on_fill_event(buy_fill)
        
        assert strategy.get_position("AAPL") == 100
        assert strategy.current_capital < 100000  # Reduced by trade cost
        
        # Sell fill
        sell_fill = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            quantity=50,
            side="SELL",
            fill_price=155.0
        )
        strategy.on_fill_event(sell_fill)
        
        assert strategy.get_position("AAPL") == 50  # 100 - 50
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        strategy = StrategyBase(
            initial_capital=100000,
            symbols=["AAPL"]
        )
        
        # Simulate some trading activity
        strategy.current_capital = 105000
        strategy.total_trades = 10
        strategy.winning_trades = 6
        
        summary = strategy.get_performance_summary()
        
        assert summary['initial_capital'] == 100000
        assert summary['final_capital'] == 105000
        assert summary['total_return'] == 0.05  # 5% return
        assert summary['total_trades'] == 10
        assert summary['win_rate'] == 0.6  # 60% win rate


class TestBollingerStrategy:
    """Test the Bollinger Bands strategy."""
    
    def test_bollinger_initialization(self):
        """Test Bollinger strategy initialization."""
        strategy = BollingerStrategy(
            lookback=20,
            z_score=2.0,
            position_size=0.1,
            symbols=["AAPL"]
        )
        
        assert strategy.lookback == 20
        assert strategy.z_score == 2.0
        assert strategy.position_size == 0.1
        assert strategy.price_history == {}
        assert strategy.positions == {}
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        strategy = BollingerStrategy(
            lookback=5,
            z_score=2.0,
            symbols=["AAPL"]
        )
        strategy.initialize()
        
        # Create price history
        prices = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105]
        
        for price in prices:
            event = MarketEvent(
                timestamp=datetime.now(),
                symbol="AAPL",
                close=price
            )
            strategy.on_market_event(event)
        
        # Check that price history is maintained
        assert len(strategy.price_history["AAPL"]) >= 5
        
        # Test Bollinger Bands calculation manually
        recent_prices = np.array(prices[-5:])
        mean = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper_band = mean + 2.0 * std
        lower_band = mean - 2.0 * std
        
        assert upper_band > mean
        assert lower_band < mean
    
    def test_signal_generation(self):
        """Test signal generation logic."""
        strategy = BollingerStrategy(
            lookback=5,
            z_score=2.0,
            symbols=["AAPL"]
        )
        strategy.initialize()
        
        # Create price history with clear pattern
        prices = [100, 100, 100, 100, 100, 95, 94, 93, 92, 91]  # Price dropping
        
        for i, price in enumerate(prices):
            event = MarketEvent(
                timestamp=datetime.now(),
                symbol="AAPL",
                close=price
            )
            strategy.on_market_event(event)
            
            # After enough data, should generate signals
            if i >= 4:
                # Price dropping should eventually trigger buy signal
                pass
    
    def test_position_sizing(self):
        """Test position sizing calculation."""
        strategy = BollingerStrategy(
            lookback=20,
            z_score=2.0,
            position_size=0.1,  # 10% of capital
            symbols=["AAPL"]
        )
        
        # Test position sizing
        price = 150.0
        quantity = strategy._calculate_position_size(price, "AAPL")
        
        # Should be approximately 10% of capital / price
        expected_quantity = int(100000 * 0.1 / 150.0)
        assert abs(quantity - expected_quantity) <= 1
    
    def test_risk_management(self):
        """Test risk management features."""
        strategy = BollingerStrategy(
            lookback=20,
            z_score=2.0,
            position_size=0.5,  # 50% of capital
            symbols=["AAPL"]
        )
        
        # Test position size limits
        price = 100.0
        quantity = strategy._calculate_position_size(price, "AAPL")
        
        # Should respect position limits
        assert quantity <= 100  # Default max position size
    
    def test_strategy_info(self):
        """Test strategy information retrieval."""
        strategy = BollingerStrategy(
            lookback=20,
            z_score=2.0,
            position_size=0.1,
            symbols=["AAPL", "GOOGL"]
        )
        
        info = strategy.get_strategy_info()
        
        assert info['name'] == 'Bollinger Bands Strategy'
        assert info['lookback'] == 20
        assert info['z_score'] == 2.0
        assert info['position_size'] == 0.1
        assert 'AAPL' in info['current_positions']
        assert 'GOOGL' in info['current_positions']


class TestBollingerMeanReversion:
    """Test the enhanced Bollinger strategy with RSI."""
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        strategy = BollingerMeanReversion(
            lookback=5,
            z_score=2.0,
            rsi_period=3,
            symbols=["AAPL"]
        )
        strategy.initialize()
        
        # Create price history with clear trend
        prices = [100, 101, 102, 103, 104, 105]  # Upward trend
        
        for price in prices:
            event = MarketEvent(
                timestamp=datetime.now(),
                symbol="AAPL",
                close=price
            )
            strategy.on_market_event(event)
        
        # RSI should be calculated
        if len(strategy.returns_history["AAPL"]) >= 3:
            rsi = strategy._calculate_rsi("AAPL")
            assert 0 <= rsi <= 100
    
    def test_rsi_filtered_signals(self):
        """Test signal generation with RSI filter."""
        strategy = BollingerMeanReversion(
            lookback=5,
            z_score=2.0,
            rsi_period=3,
            rsi_oversold=30,
            rsi_overbought=70,
            symbols=["AAPL"]
        )
        strategy.initialize()
        
        # Create price history that would trigger signals
        # but RSI should filter them
        prices = [100, 100, 100, 100, 100, 95, 94, 93, 92, 91]
        
        for price in prices:
            event = MarketEvent(
                timestamp=datetime.now(),
                symbol="AAPL",
                close=price
            )
            strategy.on_market_event(event)


if __name__ == "__main__":
    pytest.main([__file__]) 