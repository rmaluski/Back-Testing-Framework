"""
Bollinger Bands strategy implementation.

This strategy trades based on price movements relative to Bollinger Bands.
"""

import numpy as np
from trading_core import StrategyBase, MarketEvent


class BollingerStrategy(StrategyBase):
    """
    Bollinger Bands mean reversion strategy.
    
    This strategy:
    - Calculates Bollinger Bands using a rolling window
    - Buys when price touches the lower band
    - Sells when price touches the upper band
    - Uses position sizing based on volatility
    """
    
    def __init__(self, 
                 lookback: int = 20,
                 z_score: float = 2.0,
                 position_size: float = 0.1,
                 **kwargs):
        """
        Initialize the Bollinger Bands strategy.
        
        Args:
            lookback: Number of periods for moving average
            z_score: Number of standard deviations for bands
            position_size: Fraction of capital to risk per trade
            **kwargs: Additional strategy parameters
        """
        super().__init__(**kwargs)
        
        self.lookback = lookback
        self.z_score = z_score
        self.position_size = position_size
        
        # Data storage
        self.price_history = {}
        self.positions = {}
        
        # Strategy state
        self.last_signal = {}
    
    def initialize(self):
        """Initialize strategy state."""
        for symbol in self.symbols:
            self.price_history[symbol] = []
            self.positions[symbol] = 0
            self.last_signal[symbol] = None
    
    def on_market_event(self, event: MarketEvent):
        """
        Handle market events and generate trading signals.
        
        Args:
            event: Market data event
        """
        symbol = event.symbol
        
        # Get current price (use close for bars, mid for ticks)
        if event.close is not None:
            current_price = event.close
        elif event.mid_price is not None:
            current_price = event.mid_price
        else:
            return  # No valid price
        
        # Update price history
        self.price_history[symbol].append(current_price)
        
        # Keep only recent prices
        if len(self.price_history[symbol]) > self.lookback * 2:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback:]
        
        # Need enough data for calculation
        if len(self.price_history[symbol]) < self.lookback:
            return
        
        # Calculate Bollinger Bands
        prices = np.array(self.price_history[symbol][-self.lookback:])
        mean = np.mean(prices)
        std = np.std(prices)
        
        upper_band = mean + self.z_score * std
        lower_band = mean - self.z_score * std
        
        # Generate signals
        signal = self._generate_signal(current_price, upper_band, lower_band, symbol)
        
        if signal:
            self._execute_signal(signal, current_price, symbol)
    
    def _generate_signal(self, price: float, upper_band: float, lower_band: float, symbol: str) -> str:
        """
        Generate trading signal based on Bollinger Bands.
        
        Args:
            price: Current price
            upper_band: Upper Bollinger Band
            lower_band: Lower Bollinger Band
            symbol: Symbol being analyzed
            
        Returns:
            Signal type ('BUY', 'SELL', or None)
        """
        current_position = self.positions[symbol]
        
        # Buy signal: price touches lower band and we're not long
        if price <= lower_band and current_position <= 0:
            return 'BUY'
        
        # Sell signal: price touches upper band and we're not short
        elif price >= upper_band and current_position >= 0:
            return 'SELL'
        
        # Exit signal: price crosses back to mean
        elif current_position > 0 and price >= (upper_band + lower_band) / 2:
            return 'EXIT_LONG'
        elif current_position < 0 and price <= (upper_band + lower_band) / 2:
            return 'EXIT_SHORT'
        
        return None
    
    def _execute_signal(self, signal: str, price: float, symbol: str):
        """
        Execute trading signal.
        
        Args:
            signal: Trading signal
            price: Current price
            symbol: Symbol to trade
        """
        current_position = self.positions[symbol]
        
        if signal == 'BUY' and current_position <= 0:
            # Calculate position size
            quantity = self._calculate_position_size(price, symbol)
            if quantity > 0:
                self.buy(symbol, quantity)
                self.positions[symbol] = quantity
                self.last_signal[symbol] = 'BUY'
        
        elif signal == 'SELL' and current_position >= 0:
            # Calculate position size
            quantity = self._calculate_position_size(price, symbol)
            if quantity > 0:
                self.sell(symbol, quantity)
                self.positions[symbol] = -quantity
                self.last_signal[symbol] = 'SELL'
        
        elif signal == 'EXIT_LONG' and current_position > 0:
            # Close long position
            self.sell(symbol, current_position)
            self.positions[symbol] = 0
            self.last_signal[symbol] = 'EXIT_LONG'
        
        elif signal == 'EXIT_SHORT' and current_position < 0:
            # Close short position
            self.buy(symbol, abs(current_position))
            self.positions[symbol] = 0
            self.last_signal[symbol] = 'EXIT_SHORT'
    
    def _calculate_position_size(self, price: float, symbol: str) -> int:
        """
        Calculate position size based on risk management rules.
        
        Args:
            price: Current price
            symbol: Symbol to trade
            
        Returns:
            Position size in shares/contracts
        """
        # Calculate position value based on capital
        position_value = self.current_capital * self.position_size
        
        # Convert to quantity
        quantity = int(position_value / price)
        
        # Apply position limits
        max_position = self.config.risk.max_position_size if hasattr(self, 'config') else 100
        quantity = min(quantity, max_position)
        
        return max(1, quantity)  # Minimum 1 share/contract
    
    def get_strategy_info(self) -> dict:
        """Get strategy information."""
        return {
            'name': 'Bollinger Bands Strategy',
            'lookback': self.lookback,
            'z_score': self.z_score,
            'position_size': self.position_size,
            'current_positions': self.positions.copy()
        }


class BollingerMeanReversion(BollingerStrategy):
    """
    Enhanced Bollinger Bands strategy with mean reversion logic.
    
    This strategy adds:
    - RSI filter to avoid trading in trending markets
    - Volume confirmation
    - Dynamic position sizing based on volatility
    """
    
    def __init__(self, 
                 lookback: int = 20,
                 z_score: float = 2.0,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 **kwargs):
        """
        Initialize the enhanced Bollinger Bands strategy.
        
        Args:
            lookback: Number of periods for moving average
            z_score: Number of standard deviations for bands
            rsi_period: Period for RSI calculation
            rsi_oversold: RSI level considered oversold
            rsi_overbought: RSI level considered overbought
            **kwargs: Additional strategy parameters
        """
        super().__init__(lookback=lookback, z_score=z_score, **kwargs)
        
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # Additional data storage
        self.returns_history = {}
    
    def initialize(self):
        """Initialize strategy state."""
        super().initialize()
        
        for symbol in self.symbols:
            self.returns_history[symbol] = []
    
    def on_market_event(self, event: MarketEvent):
        """Enhanced market event handling with RSI filter."""
        symbol = event.symbol
        
        # Get current price
        if event.close is not None:
            current_price = event.close
        elif event.mid_price is not None:
            current_price = event.mid_price
        else:
            return
        
        # Update price and returns history
        self.price_history[symbol].append(current_price)
        
        if len(self.price_history[symbol]) > 1:
            returns = (current_price - self.price_history[symbol][-2]) / self.price_history[symbol][-2]
            self.returns_history[symbol].append(returns)
        
        # Keep only recent data
        if len(self.price_history[symbol]) > self.lookback * 2:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback:]
        if len(self.returns_history[symbol]) > self.rsi_period * 2:
            self.returns_history[symbol] = self.returns_history[symbol][-self.rsi_period:]
        
        # Need enough data
        if len(self.price_history[symbol]) < self.lookback or len(self.returns_history[symbol]) < self.rsi_period:
            return
        
        # Calculate indicators
        prices = np.array(self.price_history[symbol][-self.lookback:])
        mean = np.mean(prices)
        std = np.std(prices)
        
        upper_band = mean + self.z_score * std
        lower_band = mean - self.z_score * std
        
        # Calculate RSI
        rsi = self._calculate_rsi(symbol)
        
        # Generate signal with RSI filter
        signal = self._generate_signal_with_rsi(current_price, upper_band, lower_band, rsi, symbol)
        
        if signal:
            self._execute_signal(signal, current_price, symbol)
    
    def _calculate_rsi(self, symbol: str) -> float:
        """Calculate RSI for a symbol."""
        if len(self.returns_history[symbol]) < self.rsi_period:
            return 50.0  # Neutral RSI
        
        returns = np.array(self.returns_history[symbol][-self.rsi_period:])
        
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _generate_signal_with_rsi(self, price: float, upper_band: float, lower_band: float, rsi: float, symbol: str) -> str:
        """Generate signal with RSI filter."""
        current_position = self.positions[symbol]
        
        # Buy signal: price at lower band and RSI oversold
        if price <= lower_band and rsi <= self.rsi_oversold and current_position <= 0:
            return 'BUY'
        
        # Sell signal: price at upper band and RSI overbought
        elif price >= upper_band and rsi >= self.rsi_overbought and current_position >= 0:
            return 'SELL'
        
        # Exit signals
        elif current_position > 0 and (price >= (upper_band + lower_band) / 2 or rsi >= 70):
            return 'EXIT_LONG'
        elif current_position < 0 and (price <= (upper_band + lower_band) / 2 or rsi <= 30):
            return 'EXIT_SHORT'
        
        return None
    
    def _calculate_position_size(self, price: float, symbol: str) -> int:
        """Enhanced position sizing with volatility adjustment."""
        # Base position size
        base_position_value = self.current_capital * self.position_size
        
        # Adjust for volatility
        if len(self.returns_history[symbol]) >= 20:
            volatility = np.std(self.returns_history[symbol][-20:])
            # Reduce position size in high volatility
            volatility_adjustment = 1 / (1 + volatility * 10)
            position_value = base_position_value * volatility_adjustment
        else:
            position_value = base_position_value
        
        # Convert to quantity
        quantity = int(position_value / price)
        
        # Apply limits
        max_position = getattr(self, 'config', None)
        if max_position and hasattr(max_position.risk, 'max_position_size'):
            quantity = min(quantity, max_position.risk.max_position_size)
        else:
            quantity = min(quantity, 100)
        
        return max(1, quantity) 