"""
Momentum and Technical Analysis Strategies (Categories D & E: 161-340)

This module implements momentum and technical analysis strategies including:
- Cross-sectional momentum
- Time-series momentum
- Technical indicators
- Price patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from trading_core import StrategyBase, MarketEvent
from trading_core.events import SignalEvent


class MomentumStrategyBase(StrategyBase):
    """Base class for momentum strategies."""
    
    def __init__(self, 
                 lookback_period: int = 252,
                 momentum_period: int = 12,
                 **kwargs):
        """
        Initialize momentum strategy base.
        
        Args:
            lookback_period: Number of days to look back for calculations
            momentum_period: Period for momentum calculation
            **kwargs: Additional strategy parameters
        """
        super().__init__(**kwargs)
        self.lookback_period = lookback_period
        self.momentum_period = momentum_period
        
        # Price history storage
        self.price_history = {}
        self.returns_history = {}
        self.volatility_history = {}
        
        # Strategy state
        self.current_positions = {}
        self.momentum_signals = {}
    
    def initialize(self):
        """Initialize strategy state."""
        super().initialize()
        for symbol in self.symbols:
            self.price_history[symbol] = []
            self.returns_history[symbol] = []
            self.volatility_history[symbol] = []
            self.current_positions[symbol] = 0
            self.momentum_signals[symbol] = 0.0
    
    def update_price_history(self, symbol: str, price: float):
        """Update price history for a symbol."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # Keep only recent prices
        if len(self.price_history[symbol]) > self.lookback_period:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback_period:]
        
        # Calculate returns
        if len(self.price_history[symbol]) > 1:
            returns = (price - self.price_history[symbol][-2]) / self.price_history[symbol][-2]
            self.returns_history[symbol].append(returns)
            
            # Keep only recent returns
            if len(self.returns_history[symbol]) > self.lookback_period:
                self.returns_history[symbol] = self.returns_history[symbol][-self.lookback_period:]
    
    def calculate_momentum(self, symbol: str) -> float:
        """Calculate momentum for a symbol. Override in subclasses."""
        raise NotImplementedError
    
    def generate_momentum_signals(self) -> List[SignalEvent]:
        """Generate momentum-based trading signals."""
        signals = []
        
        # Calculate momentum for all symbols
        momentum_scores = {}
        for symbol in self.symbols:
            if len(self.price_history.get(symbol, [])) >= self.momentum_period:
                momentum = self.calculate_momentum(symbol)
                momentum_scores[symbol] = momentum
                self.momentum_signals[symbol] = momentum
        
        # Sort by momentum
        sorted_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Generate long signals for top momentum
        for symbol, momentum in sorted_symbols[:len(sorted_symbols) // 3]:  # Top third
            if momentum > 0 and self.current_positions.get(symbol, 0) <= 0:
                signals.append(SignalEvent(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type="LONG",
                    strength=momentum,
                    metadata={"strategy": self.__class__.__name__}
                ))
        
        # Generate short signals for bottom momentum
        for symbol, momentum in sorted_symbols[-len(sorted_symbols) // 3:]:  # Bottom third
            if momentum < 0 and self.current_positions.get(symbol, 0) >= 0:
                signals.append(SignalEvent(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type="SHORT",
                    strength=abs(momentum),
                    metadata={"strategy": self.__class__.__name__}
                ))
        
        return signals
    
    def on_market_event(self, event: MarketEvent):
        """Handle market events and generate momentum signals."""
        symbol = event.symbol
        current_price = event.close or event.mid_price
        
        if current_price is None:
            return
        
        # Update price history
        self.update_price_history(symbol, current_price)
        
        # Generate signals periodically
        if len(self.price_history.get(symbol, [])) >= self.momentum_period:
            signals = self.generate_momentum_signals()
            
            # Execute signals
            for signal in signals:
                if signal.symbol == symbol:
                    self._execute_signal(signal, event)
    
    def _execute_signal(self, signal: SignalEvent, market_event: MarketEvent):
        """Execute a trading signal."""
        symbol = signal.symbol
        current_price = market_event.close or market_event.mid_price
        
        if current_price is None:
            return
        
        if signal.signal_type == "LONG":
            # Close short position if exists
            if self.current_positions.get(symbol, 0) < 0:
                self.buy(symbol, abs(self.current_positions[symbol]))
            
            # Open long position
            quantity = self._calculate_position_size(current_price, signal.strength)
            if quantity > 0:
                self.buy(symbol, quantity)
                self.current_positions[symbol] = quantity
        
        elif signal.signal_type == "SHORT":
            # Close long position if exists
            if self.current_positions.get(symbol, 0) > 0:
                self.sell(symbol, self.current_positions[symbol])
            
            # Open short position
            quantity = self._calculate_position_size(current_price, signal.strength)
            if quantity > 0:
                self.sell(symbol, quantity)
                self.current_positions[symbol] = -quantity
    
    def _calculate_position_size(self, price: float, signal_strength: float) -> int:
        """Calculate position size based on signal strength."""
        base_value = self.current_capital * 0.01  # 1% per position
        adjusted_value = base_value * abs(signal_strength)
        quantity = int(adjusted_value / price)
        return max(1, min(quantity, 100))  # Between 1 and 100


class CrossSectionalMomentum(MomentumStrategyBase):
    """
    Cross-Sectional Momentum Strategy (#161)
    
    12-1 Month Relative Strength.
    """
    
    def __init__(self, **kwargs):
        super().__init__(momentum_period=252, **kwargs)  # 12 months
    
    def calculate_momentum(self, symbol: str) -> float:
        """Calculate 12-1 month momentum."""
        prices = self.price_history.get(symbol, [])
        if len(prices) < self.momentum_period:
            return 0.0
        
        # 12-month return minus 1-month return
        twelve_month_return = (prices[-1] - prices[-252]) / prices[-252]
        one_month_return = (prices[-1] - prices[-21]) / prices[-21]
        
        return twelve_month_return - one_month_return


class TimeSeriesMomentum(MomentumStrategyBase):
    """
    Time-Series Momentum Strategy (#162)
    
    6-1 Month Momentum.
    """
    
    def __init__(self, **kwargs):
        super().__init__(momentum_period=126, **kwargs)  # 6 months
    
    def calculate_momentum(self, symbol: str) -> float:
        """Calculate 6-1 month momentum."""
        prices = self.price_history.get(symbol, [])
        if len(prices) < self.momentum_period:
            return 0.0
        
        # 6-month return minus 1-month return
        six_month_return = (prices[-1] - prices[-126]) / prices[-126]
        one_month_return = (prices[-1] - prices[-21]) / prices[-21]
        
        return six_month_return - one_month_return


class ShortTermMomentum(MomentumStrategyBase):
    """
    Short-Term Momentum Strategy (#163)
    
    3-1 Month Short-Term Momentum.
    """
    
    def __init__(self, **kwargs):
        super().__init__(momentum_period=63, **kwargs)  # 3 months
    
    def calculate_momentum(self, symbol: str) -> float:
        """Calculate 3-1 month momentum."""
        prices = self.price_history.get(symbol, [])
        if len(prices) < self.momentum_period:
            return 0.0
        
        # 3-month return minus 1-month return
        three_month_return = (prices[-1] - prices[-63]) / prices[-63]
        one_month_return = (prices[-1] - prices[-21]) / prices[-21]
        
        return three_month_return - one_month_return


class VolatilityAdjustedMomentum(MomentumStrategyBase):
    """
    Volatility-Adjusted Momentum Strategy (#164)
    
    Momentum adjusted for volatility.
    """
    
    def __init__(self, vol_window: int = 60, **kwargs):
        super().__init__(**kwargs)
        self.vol_window = vol_window
    
    def calculate_momentum(self, symbol: str) -> float:
        """Calculate volatility-adjusted momentum."""
        returns = self.returns_history.get(symbol, [])
        if len(returns) < self.momentum_period:
            return 0.0
        
        # Calculate momentum
        momentum_return = np.mean(returns[-self.momentum_period:])
        
        # Calculate volatility
        volatility = np.std(returns[-self.vol_window:])
        
        # Volatility-adjusted momentum
        if volatility > 0:
            return momentum_return / volatility
        else:
            return 0.0


class MovingAverageCrossover(MomentumStrategyBase):
    """
    Moving Average Crossover Strategy (#165)
    
    Fast MA vs Slow MA crossover.
    """
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def calculate_momentum(self, symbol: str) -> float:
        """Calculate MA crossover momentum."""
        prices = self.price_history.get(symbol, [])
        if len(prices) < self.slow_period:
            return 0.0
        
        # Calculate moving averages
        fast_ma = np.mean(prices[-self.fast_period:])
        slow_ma = np.mean(prices[-self.slow_period:])
        
        # Crossover signal
        if fast_ma > slow_ma:
            return (fast_ma - slow_ma) / slow_ma
        else:
            return (fast_ma - slow_ma) / slow_ma


class RSIMomentum(MomentumStrategyBase):
    """
    RSI Momentum Strategy (#166)
    
    RSI-based momentum signals.
    """
    
    def __init__(self, rsi_period: int = 14, **kwargs):
        super().__init__(**kwargs)
        self.rsi_period = rsi_period
    
    def calculate_rsi(self, symbol: str) -> float:
        """Calculate RSI for a symbol."""
        returns = self.returns_history.get(symbol, [])
        if len(returns) < self.rsi_period:
            return 50.0  # Neutral RSI
        
        gains = np.where(np.array(returns[-self.rsi_period:]) > 0, 
                        np.array(returns[-self.rsi_period:]), 0)
        losses = np.where(np.array(returns[-self.rsi_period:]) < 0, 
                         -np.array(returns[-self.rsi_period:]), 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_momentum(self, symbol: str) -> float:
        """Calculate RSI-based momentum."""
        rsi = self.calculate_rsi(symbol)
        
        # RSI momentum signal
        if rsi > 70:
            return -1.0  # Overbought - negative momentum
        elif rsi < 30:
            return 1.0   # Oversold - positive momentum
        else:
            return (rsi - 50) / 50  # Normalized RSI


class MACDMomentum(MomentumStrategyBase):
    """
    MACD Momentum Strategy (#167)
    
    MACD histogram momentum.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, **kwargs):
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate_macd(self, symbol: str) -> Tuple[float, float, float]:
        """Calculate MACD components."""
        prices = self.price_history.get(symbol, [])
        if len(prices) < self.slow_period:
            return 0.0, 0.0, 0.0
        
        # Calculate EMAs
        fast_ema = self._calculate_ema(prices, self.fast_period)
        slow_ema = self._calculate_ema(prices, self.slow_period)
        
        # MACD line
        macd_line = fast_ema - slow_ema
        
        # Signal line (EMA of MACD)
        macd_values = []
        for i in range(len(prices) - self.slow_period + 1):
            fast_ema_i = self._calculate_ema(prices[i:i+self.slow_period], self.fast_period)
            slow_ema_i = self._calculate_ema(prices[i:i+self.slow_period], self.slow_period)
            macd_values.append(fast_ema_i - slow_ema_i)
        
        if len(macd_values) >= self.signal_period:
            signal_line = self._calculate_ema(macd_values, self.signal_period)
        else:
            signal_line = 0.0
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_ema(self, data: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return np.mean(data)
        
        alpha = 2.0 / (period + 1)
        ema = data[0]
        
        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def calculate_momentum(self, symbol: str) -> float:
        """Calculate MACD momentum."""
        macd_line, signal_line, histogram = self.calculate_macd(symbol)
        
        # Use histogram as momentum signal
        return histogram


class BollingerBandsMomentum(MomentumStrategyBase):
    """
    Bollinger Bands Momentum Strategy (#168)
    
    Price position relative to Bollinger Bands.
    """
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def calculate_momentum(self, symbol: str) -> float:
        """Calculate Bollinger Bands momentum."""
        prices = self.price_history.get(symbol, [])
        if len(prices) < self.bb_period:
            return 0.0
        
        # Calculate Bollinger Bands
        recent_prices = prices[-self.bb_period:]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        upper_band = mean_price + self.bb_std * std_price
        lower_band = mean_price - self.bb_std * std_price
        
        current_price = prices[-1]
        
        # Position relative to bands
        if current_price > upper_band:
            return -1.0  # Above upper band - negative momentum
        elif current_price < lower_band:
            return 1.0   # Below lower band - positive momentum
        else:
            # Normalized position within bands
            return (current_price - mean_price) / (upper_band - mean_price)


class VolumeWeightedMomentum(MomentumStrategyBase):
    """
    Volume-Weighted Momentum Strategy (#169)
    
    Momentum weighted by volume.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.volume_history = {}
    
    def update_volume_history(self, symbol: str, volume: float):
        """Update volume history for a symbol."""
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        
        self.volume_history[symbol].append(volume)
        
        # Keep only recent volumes
        if len(self.volume_history[symbol]) > self.lookback_period:
            self.volume_history[symbol] = self.volume_history[symbol][-self.lookback_period:]
    
    def calculate_momentum(self, symbol: str) -> float:
        """Calculate volume-weighted momentum."""
        returns = self.returns_history.get(symbol, [])
        volumes = self.volume_history.get(symbol, [])
        
        if len(returns) < self.momentum_period or len(volumes) < self.momentum_period:
            return 0.0
        
        # Volume-weighted returns
        recent_returns = returns[-self.momentum_period:]
        recent_volumes = volumes[-self.momentum_period:]
        
        # Normalize volumes
        avg_volume = np.mean(recent_volumes)
        if avg_volume > 0:
            volume_weights = np.array(recent_volumes) / avg_volume
        else:
            volume_weights = np.ones(len(recent_returns))
        
        # Volume-weighted momentum
        vw_momentum = np.average(recent_returns, weights=volume_weights)
        
        return vw_momentum


class MomentumCrashesHedge(MomentumStrategyBase):
    """
    Momentum Crashes Hedge Strategy (#170)
    
    Momentum with crash protection.
    """
    
    def __init__(self, crash_threshold: float = -0.05, **kwargs):
        super().__init__(**kwargs)
        self.crash_threshold = crash_threshold
    
    def calculate_momentum(self, symbol: str) -> float:
        """Calculate momentum with crash protection."""
        returns = self.returns_history.get(symbol, [])
        if len(returns) < self.momentum_period:
            return 0.0
        
        # Calculate momentum
        momentum = np.mean(returns[-self.momentum_period:])
        
        # Check for crash conditions
        recent_returns = returns[-21:]  # Last month
        crash_probability = np.mean(np.array(recent_returns) < self.crash_threshold)
        
        # Reduce momentum signal during crash conditions
        if crash_probability > 0.3:  # More than 30% crash probability
            momentum *= 0.5  # Reduce signal strength
        
        return momentum


class TechnicalIndicatorStrategy(StrategyBase):
    """Base class for technical indicator strategies."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.price_history = {}
        self.indicator_values = {}
    
    def initialize(self):
        """Initialize strategy state."""
        super().initialize()
        for symbol in self.symbols:
            self.price_history[symbol] = []
            self.indicator_values[symbol] = {}
    
    def update_price_data(self, symbol: str, price: float, volume: float = 0):
        """Update price data for a symbol."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'price': price,
            'volume': volume,
            'timestamp': datetime.now()
        })
    
    def calculate_indicators(self, symbol: str) -> Dict[str, float]:
        """Calculate technical indicators. Override in subclasses."""
        raise NotImplementedError
    
    def generate_signals(self, symbol: str) -> List[SignalEvent]:
        """Generate trading signals based on indicators. Override in subclasses."""
        raise NotImplementedError
    
    def on_market_event(self, event: MarketEvent):
        """Handle market events and generate signals."""
        symbol = event.symbol
        current_price = event.close or event.mid_price
        volume = event.volume or 0
        
        if current_price is None:
            return
        
        # Update price data
        self.update_price_data(symbol, current_price, volume)
        
        # Calculate indicators
        indicators = self.calculate_indicators(symbol)
        self.indicator_values[symbol] = indicators
        
        # Generate signals
        signals = self.generate_signals(symbol)
        
        # Execute signals
        for signal in signals:
            self._execute_signal(signal, event)


class RSISignalStrategy(TechnicalIndicatorStrategy):
    """
    RSI Signal Strategy (#221)
    
    RSI 14 70/30 rule.
    """
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70, **kwargs):
        super().__init__(**kwargs)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_indicators(self, symbol: str) -> Dict[str, float]:
        """Calculate RSI indicator."""
        prices = [d['price'] for d in self.price_history.get(symbol, [])]
        if len(prices) < self.rsi_period + 1:
            return {'rsi': 50.0}
        
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if len(returns) < self.rsi_period:
            return {'rsi': 50.0}
        
        # Calculate RSI
        gains = np.where(np.array(returns[-self.rsi_period:]) > 0, 
                        np.array(returns[-self.rsi_period:]), 0)
        losses = np.where(np.array(returns[-self.rsi_period:]) < 0, 
                         -np.array(returns[-self.rsi_period:]), 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        return {'rsi': rsi}
    
    def generate_signals(self, symbol: str) -> List[SignalEvent]:
        """Generate RSI-based signals."""
        signals = []
        indicators = self.indicator_values.get(symbol, {})
        rsi = indicators.get('rsi', 50.0)
        
        current_position = self.get_position(symbol)
        
        # Oversold signal
        if rsi < self.oversold and current_position <= 0:
            signals.append(SignalEvent(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type="LONG",
                strength=(self.oversold - rsi) / self.oversold,
                metadata={"strategy": self.__class__.__name__, "rsi": rsi}
            ))
        
        # Overbought signal
        elif rsi > self.overbought and current_position >= 0:
            signals.append(SignalEvent(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type="SHORT",
                strength=(rsi - self.overbought) / (100 - self.overbought),
                metadata={"strategy": self.__class__.__name__, "rsi": rsi}
            ))
        
        return signals


class MACDSignalStrategy(TechnicalIndicatorStrategy):
    """
    MACD Signal Strategy (#222)
    
    MACD histogram surge.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, **kwargs):
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate_indicators(self, symbol: str) -> Dict[str, float]:
        """Calculate MACD indicators."""
        prices = [d['price'] for d in self.price_history.get(symbol, [])]
        if len(prices) < self.slow_period:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        # Calculate EMAs
        fast_ema = self._calculate_ema(prices, self.fast_period)
        slow_ema = self._calculate_ema(prices, self.slow_period)
        
        # MACD line
        macd_line = fast_ema - slow_ema
        
        # Signal line
        macd_values = []
        for i in range(len(prices) - self.slow_period + 1):
            fast_ema_i = self._calculate_ema(prices[i:i+self.slow_period], self.fast_period)
            slow_ema_i = self._calculate_ema(prices[i:i+self.slow_period], self.slow_period)
            macd_values.append(fast_ema_i - slow_ema_i)
        
        if len(macd_values) >= self.signal_period:
            signal_line = self._calculate_ema(macd_values, self.signal_period)
        else:
            signal_line = 0.0
        
        # Histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_ema(self, data: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return np.mean(data)
        
        alpha = 2.0 / (period + 1)
        ema = data[0]
        
        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def generate_signals(self, symbol: str) -> List[SignalEvent]:
        """Generate MACD-based signals."""
        signals = []
        indicators = self.indicator_values.get(symbol, {})
        histogram = indicators.get('histogram', 0.0)
        
        current_position = self.get_position(symbol)
        
        # Positive histogram surge
        if histogram > 0.01 and current_position <= 0:  # Threshold for surge
            signals.append(SignalEvent(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type="LONG",
                strength=min(histogram * 100, 1.0),
                metadata={"strategy": self.__class__.__name__, "histogram": histogram}
            ))
        
        # Negative histogram surge
        elif histogram < -0.01 and current_position >= 0:
            signals.append(SignalEvent(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type="SHORT",
                strength=min(abs(histogram) * 100, 1.0),
                metadata={"strategy": self.__class__.__name__, "histogram": histogram}
            ))
        
        return signals


# Factory function to create momentum strategies
def create_momentum_strategy(strategy_name: str, **kwargs) -> MomentumStrategyBase:
    """
    Factory function to create momentum strategies.
    
    Args:
        strategy_name: Name of the strategy to create
        **kwargs: Strategy parameters
        
    Returns:
        Strategy instance
    """
    strategy_map = {
        'cross_sectional_momentum': CrossSectionalMomentum,
        'time_series_momentum': TimeSeriesMomentum,
        'short_term_momentum': ShortTermMomentum,
        'volatility_adjusted_momentum': VolatilityAdjustedMomentum,
        'moving_average_crossover': MovingAverageCrossover,
        'rsi_momentum': RSIMomentum,
        'macd_momentum': MACDMomentum,
        'bollinger_bands_momentum': BollingerBandsMomentum,
        'volume_weighted_momentum': VolumeWeightedMomentum,
        'momentum_crashes_hedge': MomentumCrashesHedge,
    }
    
    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategy_map[strategy_name](**kwargs)


# Factory function to create technical indicator strategies
def create_technical_strategy(strategy_name: str, **kwargs) -> TechnicalIndicatorStrategy:
    """
    Factory function to create technical indicator strategies.
    
    Args:
        strategy_name: Name of the strategy to create
        **kwargs: Strategy parameters
        
    Returns:
        Strategy instance
    """
    strategy_map = {
        'rsi_signal': RSISignalStrategy,
        'macd_signal': MACDSignalStrategy,
    }
    
    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategy_map[strategy_name](**kwargs) 