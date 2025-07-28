"""
Strategy Combination Framework

This module provides a framework for combining multiple strategies together,
allowing for sophisticated multi-factor approaches and strategy overlays.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from trading_core import StrategyBase, MarketEvent
from trading_core.events import SignalEvent
from .fundamental_strategies import FundamentalStrategyBase, create_fundamental_strategy
from .momentum_strategies import MomentumStrategyBase, create_momentum_strategy
from .quality_strategies import QualityStrategyBase, create_quality_strategy


class StrategyCombiner(StrategyBase):
    """
    Combines multiple strategies with configurable weights and combination methods.
    
    This allows for sophisticated multi-factor approaches and strategy overlays.
    """
    
    def __init__(self, 
                 strategies: List[Dict[str, Any]],
                 combination_method: str = "weighted_sum",
                 rebalance_frequency: str = "daily",
                 **kwargs):
        """
        Initialize strategy combiner.
        
        Args:
            strategies: List of strategy configurations
                Each dict should contain:
                - 'name': Strategy name
                - 'type': 'fundamental', 'momentum', 'quality', 'technical'
                - 'weight': Weight in combination (0-1)
                - 'params': Strategy-specific parameters
            combination_method: How to combine signals
                - 'weighted_sum': Weighted average of signals
                - 'majority_vote': Majority vote on direction
                - 'consensus': All strategies must agree
                - 'hierarchical': Apply strategies in order
            rebalance_frequency: How often to rebalance
            **kwargs: Additional strategy parameters
        """
        super().__init__(**kwargs)
        self.strategies = strategies
        self.combination_method = combination_method
        self.rebalance_frequency = rebalance_frequency
        
        # Initialize sub-strategies
        self.sub_strategies = []
        self.strategy_weights = []
        self.strategy_signals = {}
        
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all sub-strategies."""
        for strategy_config in self.strategies:
            strategy_name = strategy_config['name']
            strategy_type = strategy_config['type']
            weight = strategy_config.get('weight', 1.0)
            params = strategy_config.get('params', {})
            
            # Create strategy based on type
            if strategy_type == 'fundamental':
                strategy = create_fundamental_strategy(strategy_name, **params)
            elif strategy_type == 'momentum':
                strategy = create_momentum_strategy(strategy_name, **params)
            elif strategy_type == 'quality':
                strategy = create_quality_strategy(strategy_name, **params)
            else:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            # Set symbols and capital for sub-strategy
            strategy.symbols = self.symbols
            strategy.initial_capital = self.initial_capital * weight
            
            self.sub_strategies.append(strategy)
            self.strategy_weights.append(weight)
            self.strategy_signals[strategy_name] = {}
    
    def initialize(self):
        """Initialize strategy state."""
        super().initialize()
        
        # Initialize all sub-strategies
        for strategy in self.sub_strategies:
            strategy.initialize()
    
    def update_fundamental_data(self, symbol: str, data: Dict[str, float]):
        """Update fundamental data for all fundamental strategies."""
        for strategy in self.sub_strategies:
            if isinstance(strategy, FundamentalStrategyBase):
                strategy.update_fundamental_data(symbol, data)
    
    def combine_signals(self, symbol: str) -> SignalEvent:
        """Combine signals from all strategies."""
        signals = []
        weights = []
        
        # Collect signals from all strategies
        for i, strategy in enumerate(self.sub_strategies):
            strategy_name = self.strategies[i]['name']
            
            # Get signal from strategy
            if hasattr(strategy, 'generate_signals'):
                strategy_signals = strategy.generate_signals()
                for signal in strategy_signals:
                    if signal.symbol == symbol:
                        signals.append(signal)
                        weights.append(self.strategy_weights[i])
                        break
            else:
                # For strategies that don't generate signals directly
                # Use their current position or momentum
                if hasattr(strategy, 'momentum_signals'):
                    momentum = strategy.momentum_signals.get(symbol, 0)
                    if momentum != 0:
                        signal_type = "LONG" if momentum > 0 else "SHORT"
                        signals.append(SignalEvent(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            signal_type=signal_type,
                            strength=abs(momentum),
                            metadata={"strategy": strategy_name}
                        ))
                        weights.append(self.strategy_weights[i])
        
        if not signals:
            return None
        
        # Combine signals based on method
        if self.combination_method == "weighted_sum":
            return self._weighted_sum_combination(signals, weights)
        elif self.combination_method == "majority_vote":
            return self._majority_vote_combination(signals, weights)
        elif self.combination_method == "consensus":
            return self._consensus_combination(signals, weights)
        elif self.combination_method == "hierarchical":
            return self._hierarchical_combination(signals, weights)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    def _weighted_sum_combination(self, signals: List[SignalEvent], weights: List[float]) -> SignalEvent:
        """Combine signals using weighted sum."""
        if not signals:
            return None
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return None
        
        normalized_weights = [w / total_weight for w in weights]
        
        # Calculate weighted signal strength
        weighted_strength = 0.0
        long_votes = 0
        short_votes = 0
        
        for signal, weight in zip(signals, normalized_weights):
            if signal.signal_type == "LONG":
                weighted_strength += signal.strength * weight
                long_votes += weight
            else:  # SHORT
                weighted_strength -= signal.strength * weight
                short_votes += weight
        
        # Determine signal type
        if weighted_strength > 0:
            signal_type = "LONG"
            strength = weighted_strength
        else:
            signal_type = "SHORT"
            strength = abs(weighted_strength)
        
        return SignalEvent(
            timestamp=datetime.now(),
            symbol=signals[0].symbol,
            signal_type=signal_type,
            strength=strength,
            metadata={"strategy": "StrategyCombiner", "method": "weighted_sum"}
        )
    
    def _majority_vote_combination(self, signals: List[SignalEvent], weights: List[float]) -> SignalEvent:
        """Combine signals using majority vote."""
        if not signals:
            return None
        
        long_weight = 0.0
        short_weight = 0.0
        
        for signal, weight in zip(signals, weights):
            if signal.signal_type == "LONG":
                long_weight += weight
            else:  # SHORT
                short_weight += weight
        
        # Determine majority
        if long_weight > short_weight:
            signal_type = "LONG"
            strength = long_weight / (long_weight + short_weight)
        elif short_weight > long_weight:
            signal_type = "SHORT"
            strength = short_weight / (long_weight + short_weight)
        else:
            return None  # Tie
        
        return SignalEvent(
            timestamp=datetime.now(),
            symbol=signals[0].symbol,
            signal_type=signal_type,
            strength=strength,
            metadata={"strategy": "StrategyCombiner", "method": "majority_vote"}
        )
    
    def _consensus_combination(self, signals: List[SignalEvent], weights: List[float]) -> SignalEvent:
        """Combine signals using consensus (all must agree)."""
        if not signals:
            return None
        
        # Check if all signals agree
        signal_types = [s.signal_type for s in signals]
        if len(set(signal_types)) != 1:
            return None  # No consensus
        
        signal_type = signal_types[0]
        
        # Average strength
        avg_strength = np.mean([s.strength for s in signals])
        
        return SignalEvent(
            timestamp=datetime.now(),
            symbol=signals[0].symbol,
            signal_type=signal_type,
            strength=avg_strength,
            metadata={"strategy": "StrategyCombiner", "method": "consensus"}
        )
    
    def _hierarchical_combination(self, signals: List[SignalEvent], weights: List[float]) -> SignalEvent:
        """Combine signals using hierarchical approach (first non-zero signal wins)."""
        if not signals:
            return None
        
        # Return the first signal with non-zero strength
        for signal in signals:
            if signal.strength > 0:
                return signal
        
        return None
    
    def on_market_event(self, event: MarketEvent):
        """Handle market events and generate combined signals."""
        # Update all sub-strategies
        for strategy in self.sub_strategies:
            strategy.on_market_event(event)
        
        # Generate combined signal
        combined_signal = self.combine_signals(event.symbol)
        
        if combined_signal:
            self._execute_signal(combined_signal, event)
    
    def _execute_signal(self, signal: SignalEvent, market_event: MarketEvent):
        """Execute a combined trading signal."""
        symbol = signal.symbol
        current_price = market_event.close or market_event.mid_price
        
        if current_price is None:
            return
        
        if signal.signal_type == "LONG":
            # Close short position if exists
            current_position = self.get_position(symbol)
            if current_position < 0:
                self.buy(symbol, abs(current_position))
            
            # Open long position
            quantity = self._calculate_position_size(current_price, signal.strength)
            if quantity > 0:
                self.buy(symbol, quantity)
        
        elif signal.signal_type == "SHORT":
            # Close long position if exists
            current_position = self.get_position(symbol)
            if current_position > 0:
                self.sell(symbol, current_position)
            
            # Open short position
            quantity = self._calculate_position_size(current_price, signal.strength)
            if quantity > 0:
                self.sell(symbol, quantity)
    
    def _calculate_position_size(self, price: float, signal_strength: float) -> int:
        """Calculate position size based on signal strength."""
        base_value = self.current_capital * 0.01  # 1% per position
        adjusted_value = base_value * signal_strength
        quantity = int(adjusted_value / price)
        return max(1, min(quantity, 100))  # Between 1 and 100
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance metrics for each sub-strategy."""
        performance = {}
        
        for i, strategy in enumerate(self.sub_strategies):
            strategy_name = self.strategies[i]['name']
            if hasattr(strategy, 'get_performance_summary'):
                performance[strategy_name] = strategy.get_performance_summary()
            else:
                performance[strategy_name] = {
                    'total_trades': strategy.total_trades,
                    'total_pnl': strategy.total_pnl,
                    'max_drawdown': strategy.max_drawdown
                }
        
        return performance


class ValueMomentumCombo(StrategyCombiner):
    """
    Value + Momentum Combination Strategy
    
    Combines fundamental value strategies with momentum strategies.
    """
    
    def __init__(self, 
                 value_weight: float = 0.6,
                 momentum_weight: float = 0.4,
                 **kwargs):
        """
        Initialize Value + Momentum combination.
        
        Args:
            value_weight: Weight for value strategies
            momentum_weight: Weight for momentum strategies
            **kwargs: Additional parameters
        """
        strategies = [
            {
                'name': 'magic_formula',
                'type': 'fundamental',
                'weight': value_weight,
                'params': {}
            },
            {
                'name': 'cross_sectional_momentum',
                'type': 'momentum',
                'weight': momentum_weight,
                'params': {}
            }
        ]
        
        super().__init__(strategies, combination_method="weighted_sum", **kwargs)


class QualityValueMomentumCombo(StrategyCombiner):
    """
    Quality + Value + Momentum Combination Strategy
    
    Three-factor combination approach.
    """
    
    def __init__(self, 
                 quality_weight: float = 0.4,
                 value_weight: float = 0.3,
                 momentum_weight: float = 0.3,
                 **kwargs):
        """
        Initialize Quality + Value + Momentum combination.
        
        Args:
            quality_weight: Weight for quality strategies
            value_weight: Weight for value strategies
            momentum_weight: Weight for momentum strategies
            **kwargs: Additional parameters
        """
        strategies = [
            {
                'name': 'gross_profitability_factor',
                'type': 'quality',
                'weight': quality_weight,
                'params': {}
            },
            {
                'name': 'price_to_book_deep_value',
                'type': 'fundamental',
                'weight': value_weight,
                'params': {}
            },
            {
                'name': 'time_series_momentum',
                'type': 'momentum',
                'weight': momentum_weight,
                'params': {}
            }
        ]
        
        super().__init__(strategies, combination_method="weighted_sum", **kwargs)


class TechnicalFundamentalCombo(StrategyCombiner):
    """
    Technical + Fundamental Combination Strategy
    
    Combines technical indicators with fundamental analysis.
    """
    
    def __init__(self, 
                 fundamental_weight: float = 0.7,
                 technical_weight: float = 0.3,
                 **kwargs):
        """
        Initialize Technical + Fundamental combination.
        
        Args:
            fundamental_weight: Weight for fundamental strategies
            technical_weight: Weight for technical strategies
            **kwargs: Additional parameters
        """
        strategies = [
            {
                'name': 'ev_ebitda_cheapness',
                'type': 'fundamental',
                'weight': fundamental_weight,
                'params': {}
            },
            {
                'name': 'rsi_signal',
                'type': 'momentum',  # RSI is in momentum module
                'weight': technical_weight,
                'params': {'rsi_period': 14, 'oversold': 30, 'overbought': 70}
            }
        ]
        
        super().__init__(strategies, combination_method="consensus", **kwargs)


class MultiFactorModel(StrategyCombiner):
    """
    Multi-Factor Model Strategy
    
    Combines multiple factors with equal weights (similar to Fama-French).
    """
    
    def __init__(self, **kwargs):
        """Initialize Multi-Factor Model."""
        strategies = [
            {
                'name': 'price_to_book_deep_value',
                'type': 'fundamental',
                'weight': 0.25,
                'params': {}
            },
            {
                'name': 'gross_profitability_factor',
                'type': 'quality',
                'weight': 0.25,
                'params': {}
            },
            {
                'name': 'cross_sectional_momentum',
                'type': 'momentum',
                'weight': 0.25,
                'params': {}
            },
            {
                'name': 'return_on_invested_capital',
                'type': 'quality',
                'weight': 0.25,
                'params': {}
            }
        ]
        
        super().__init__(strategies, combination_method="weighted_sum", **kwargs)


# Factory function to create strategy combinations
def create_strategy_combo(combo_name: str, **kwargs) -> StrategyCombiner:
    """
    Factory function to create strategy combinations.
    
    Args:
        combo_name: Name of the combination strategy
        **kwargs: Strategy parameters
        
    Returns:
        StrategyCombiner instance
    """
    combo_map = {
        'value_momentum_combo': ValueMomentumCombo,
        'quality_value_momentum_combo': QualityValueMomentumCombo,
        'technical_fundamental_combo': TechnicalFundamentalCombo,
        'multi_factor_model': MultiFactorModel,
    }
    
    if combo_name not in combo_map:
        raise ValueError(f"Unknown combination strategy: {combo_name}")
    
    return combo_map[combo_name](**kwargs)


# Utility function to create custom combinations
def create_custom_combo(strategies: List[Dict[str, Any]], 
                       combination_method: str = "weighted_sum",
                       **kwargs) -> StrategyCombiner:
    """
    Create a custom strategy combination.
    
    Args:
        strategies: List of strategy configurations
        combination_method: How to combine signals
        **kwargs: Additional parameters
        
    Returns:
        StrategyCombiner instance
    """
    return StrategyCombiner(
        strategies=strategies,
        combination_method=combination_method,
        **kwargs
    ) 