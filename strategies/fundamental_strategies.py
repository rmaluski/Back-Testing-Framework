"""
Fundamental and Valuation Strategies (Category A: 1-100)

This module implements fundamental analysis strategies including:
- Price-to-Book Deep Value
- EV/EBITDA Cheapness Spread
- Net-Net Graham Basket
- Shiller CAPE Tilt
- And many more...
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from trading_core import StrategyBase, MarketEvent
from trading_core.events import SignalEvent


class FundamentalStrategyBase(StrategyBase):
    """Base class for fundamental strategies with common functionality."""
    
    def __init__(self, 
                 rebalance_frequency: str = "monthly",
                 universe_size: int = 1000,
                 **kwargs):
        """
        Initialize fundamental strategy base.
        
        Args:
            rebalance_frequency: How often to rebalance ("daily", "weekly", "monthly")
            universe_size: Number of stocks in universe
            **kwargs: Additional strategy parameters
        """
        super().__init__(**kwargs)
        self.rebalance_frequency = rebalance_frequency
        self.universe_size = universe_size
        
        # Fundamental data storage
        self.fundamental_data = {}
        self.rankings = {}
        self.last_rebalance = None
        
        # Strategy state
        self.current_positions = {}
        self.target_positions = {}
    
    def initialize(self):
        """Initialize strategy state."""
        super().initialize()
        for symbol in self.symbols:
            self.current_positions[symbol] = 0
            self.target_positions[symbol] = 0
    
    def should_rebalance(self, event: MarketEvent) -> bool:
        """Check if it's time to rebalance."""
        if self.last_rebalance is None:
            return True
        
        current_date = event.timestamp.date()
        
        if self.rebalance_frequency == "daily":
            return current_date > self.last_rebalance
        elif self.rebalance_frequency == "weekly":
            return (current_date - self.last_rebalance).days >= 7
        elif self.rebalance_frequency == "monthly":
            return (current_date - self.last_rebalance).days >= 30
        
        return False
    
    def update_fundamental_data(self, symbol: str, data: Dict[str, float]):
        """Update fundamental data for a symbol."""
        self.fundamental_data[symbol] = data
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate rankings for all symbols. Override in subclasses."""
        raise NotImplementedError
    
    def generate_signals(self, rankings: Dict[str, float]) -> List[SignalEvent]:
        """Generate trading signals based on rankings."""
        signals = []
        
        # Sort by ranking (higher is better for long signals)
        sorted_symbols = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        
        # Generate long signals for top performers
        for symbol, rank in sorted_symbols[:self.universe_size // 10]:  # Top 10%
            if self.current_positions.get(symbol, 0) <= 0:
                signals.append(SignalEvent(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type="LONG",
                    strength=rank,
                    metadata={"strategy": self.__class__.__name__}
                ))
        
        # Generate short signals for bottom performers
        for symbol, rank in sorted_symbols[-self.universe_size // 10:]:  # Bottom 10%
            if self.current_positions.get(symbol, 0) >= 0:
                signals.append(SignalEvent(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type="SHORT",
                    strength=abs(rank),
                    metadata={"strategy": self.__class__.__name__}
                ))
        
        return signals
    
    def on_market_event(self, event: MarketEvent):
        """Handle market events and trigger rebalancing."""
        if self.should_rebalance(event):
            # Calculate new rankings
            rankings = self.calculate_rankings()
            
            # Generate signals
            signals = self.generate_signals(rankings)
            
            # Execute signals
            for signal in signals:
                self._execute_signal(signal, event)
            
            self.last_rebalance = event.timestamp.date()
    
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
        adjusted_value = base_value * signal_strength
        quantity = int(adjusted_value / price)
        return max(1, min(quantity, 100))  # Between 1 and 100


class PriceToBookDeepValue(FundamentalStrategyBase):
    """
    Price-to-Book Deep Value Strategy (#1)
    
    Long lowest decile, short highest decile by P/B ratio.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate P/B rankings (lower P/B = higher rank)."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if 'price_to_book' in data and data['price_to_book'] > 0:
                # Invert P/B so lower values get higher ranks
                rankings[symbol] = 1.0 / data['price_to_book']
            else:
                rankings[symbol] = 0.0
        
        return rankings


class EVEBITDACheapnessSpread(FundamentalStrategyBase):
    """
    EV/EBITDA Cheapness Spread Strategy (#2)
    
    Monthly rebalance long low quintile by EV/EBITDA.
    """
    
    def __init__(self, **kwargs):
        super().__init__(rebalance_frequency="monthly", **kwargs)
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate EV/EBITDA rankings (lower EV/EBITDA = higher rank)."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if 'ev_ebitda' in data and data['ev_ebitda'] > 0:
                # Invert EV/EBITDA so lower values get higher ranks
                rankings[symbol] = 1.0 / data['ev_ebitda']
            else:
                rankings[symbol] = 0.0
        
        return rankings


class NetNetGrahamBasket(FundamentalStrategyBase):
    """
    Net-Net Graham Basket Strategy (#3)
    
    Buy if Price < 0.66 × NCAV (Net Current Asset Value).
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate Net-Net rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if all(key in data for key in ['price', 'current_assets', 'total_liabilities']):
                ncav = data['current_assets'] - data['total_liabilities']
                if ncav > 0 and data['price'] > 0:
                    # Calculate discount to NCAV
                    discount = (ncav - data['price']) / ncav
                    if data['price'] < 0.66 * ncav:
                        rankings[symbol] = discount  # Higher discount = higher rank
                    else:
                        rankings[symbol] = 0.0
                else:
                    rankings[symbol] = 0.0
            else:
                rankings[symbol] = 0.0
        
        return rankings


class ShillerCAPETilt(FundamentalStrategyBase):
    """
    Shiller CAPE Tilt Strategy (#4)
    
    Overweight markets with CAPE < 15.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate CAPE-based rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if 'cape_ratio' in data and data['cape_ratio'] > 0:
                if data['cape_ratio'] < 15:
                    # Higher rank for lower CAPE
                    rankings[symbol] = 15.0 / data['cape_ratio']
                else:
                    rankings[symbol] = 0.0
            else:
                rankings[symbol] = 0.0
        
        return rankings


class PriceToSalesLowMultiple(FundamentalStrategyBase):
    """
    Price-to-Sales Low Multiple Strategy (#5)
    
    Long bottom decile P/S ratio.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate P/S rankings (lower P/S = higher rank)."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if 'price_to_sales' in data and data['price_to_sales'] > 0:
                # Invert P/S so lower values get higher ranks
                rankings[symbol] = 1.0 / data['price_to_sales']
            else:
                rankings[symbol] = 0.0
        
        return rankings


class PEGRatioSweetSpot(FundamentalStrategyBase):
    """
    PEG Ratio Sweet-Spot Strategy (#6)
    
    Long PEG 0.8-1.2 zone; avoid extremes.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate PEG-based rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if 'peg_ratio' in data and data['peg_ratio'] > 0:
                peg = data['peg_ratio']
                if 0.8 <= peg <= 1.2:
                    # Higher rank for PEG closer to 1.0
                    rankings[symbol] = 1.0 / abs(peg - 1.0)
                else:
                    rankings[symbol] = 0.0
            else:
                rankings[symbol] = 0.0
        
        return rankings


class OwnerEarningsYield(FundamentalStrategyBase):
    """
    Owner-Earnings Yield Strategy (#7)
    
    Rank by Buffett-style OE/market cap.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate Owner Earnings Yield rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if all(key in data for key in ['owner_earnings', 'market_cap']) and data['market_cap'] > 0:
                oey = data['owner_earnings'] / data['market_cap']
                rankings[symbol] = oey
            else:
                rankings[symbol] = 0.0
        
        return rankings


class PiotroskiFScoreBooster(FundamentalStrategyBase):
    """
    Piotroski F-Score Booster Strategy (#8)
    
    Value basket, keep only F-Score ≥ 7.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate F-Score rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if 'f_score' in data:
                f_score = data['f_score']
                if f_score >= 7:
                    rankings[symbol] = f_score
                else:
                    rankings[symbol] = 0.0
            else:
                rankings[symbol] = 0.0
        
        return rankings


class AltmanZScoreSafety(FundamentalStrategyBase):
    """
    Altman Z-Score Safety Strategy (#9)
    
    Long Z > 3; short Z < 1.8.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate Z-Score rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if 'altman_z_score' in data:
                z_score = data['altman_z_score']
                if z_score > 3:
                    rankings[symbol] = z_score  # Long signal
                elif z_score < 1.8:
                    rankings[symbol] = -z_score  # Short signal
                else:
                    rankings[symbol] = 0.0
            else:
                rankings[symbol] = 0.0
        
        return rankings


class NetDebtEBITDAConservative(FundamentalStrategyBase):
    """
    Net Debt / EBITDA Conservative Strategy (#10)
    
    Avoid leverage > 3×.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate Net Debt/EBITDA rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if all(key in data for key in ['net_debt', 'ebitda']) and data['ebitda'] > 0:
                leverage = data['net_debt'] / data['ebitda']
                if leverage <= 3:
                    # Higher rank for lower leverage
                    rankings[symbol] = 1.0 / leverage
                else:
                    rankings[symbol] = 0.0
            else:
                rankings[symbol] = 0.0
        
        return rankings


class FranchiseROIC(FundamentalStrategyBase):
    """
    Franchise ROIC Strategy (#11)
    
    Hold names with 5-yr avg ROIC > 15%.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate ROIC rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if 'roic_5yr_avg' in data:
                roic = data['roic_5yr_avg']
                if roic > 0.15:  # 15%
                    rankings[symbol] = roic
                else:
                    rankings[symbol] = 0.0
            else:
                rankings[symbol] = 0.0
        
        return rankings


class EBITEnterpriseValue(FundamentalStrategyBase):
    """
    EBIT/Enterprise Value Strategy (#12)
    
    Top decile long, beta-hedged.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate EBIT/EV rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if all(key in data for key in ['ebit', 'enterprise_value']) and data['enterprise_value'] > 0:
                ebit_ev = data['ebit'] / data['enterprise_value']
                rankings[symbol] = ebit_ev
            else:
                rankings[symbol] = 0.0
        
        return rankings


class MagicFormula(FundamentalStrategyBase):
    """
    "Magic Formula" Strategy (#13)
    
    (EBIT/EV + ROIC) - top 30 names.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate Magic Formula rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if all(key in data for key in ['ebit', 'enterprise_value', 'roic']):
                if data['enterprise_value'] > 0:
                    ebit_ev = data['ebit'] / data['enterprise_value']
                    roic = data['roic']
                    magic_score = ebit_ev + roic
                    rankings[symbol] = magic_score
                else:
                    rankings[symbol] = 0.0
            else:
                rankings[symbol] = 0.0
        
        return rankings


class QualityValueCombo(FundamentalStrategyBase):
    """
    Quality-Value Combo Strategy (#14)
    
    Rank equally by P/B and ROE.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate Quality-Value combo rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if all(key in data for key in ['price_to_book', 'roe']):
                if data['price_to_book'] > 0:
                    # Invert P/B for value component
                    value_score = 1.0 / data['price_to_book']
                    quality_score = data['roe']
                    # Equal weight combination
                    combo_score = (value_score + quality_score) / 2
                    rankings[symbol] = combo_score
                else:
                    rankings[symbol] = 0.0
            else:
                rankings[symbol] = 0.0
        
        return rankings


class DividendGrowers10Yr(FundamentalStrategyBase):
    """
    Dividend Growers 10-yr Strategy (#15)
    
    CAGR div > 5%.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate dividend growth rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if 'dividend_cagr_10yr' in data:
                div_cagr = data['dividend_cagr_10yr']
                if div_cagr > 0.05:  # 5%
                    rankings[symbol] = div_cagr
                else:
                    rankings[symbol] = 0.0
            else:
                rankings[symbol] = 0.0
        
        return rankings


class ShareholderYield(FundamentalStrategyBase):
    """
    Shareholder Yield Strategy (#16)
    
    Aggregate buybacks + dividend yield.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate shareholder yield rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            buyback_yield = data.get('buyback_yield', 0)
            dividend_yield = data.get('dividend_yield', 0)
            total_yield = buyback_yield + dividend_yield
            rankings[symbol] = total_yield
        
        return rankings


class LowPayoutHighReinvest(FundamentalStrategyBase):
    """
    Low Payout High Reinvest Strategy (#17)
    
    Long payout < 30% & ROE > 15%.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate payout/ROE combo rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if all(key in data for key in ['payout_ratio', 'roe']):
                payout = data['payout_ratio']
                roe = data['roe']
                if payout < 0.30 and roe > 0.15:  # 30% and 15%
                    # Higher rank for lower payout and higher ROE
                    rankings[symbol] = (0.30 - payout) + roe
                else:
                    rankings[symbol] = 0.0
            else:
                rankings[symbol] = 0.0
        
        return rankings


class NegativeEnterpriseValue(FundamentalStrategyBase):
    """
    Negative Enterprise Value Strategy (#18)
    
    Buy EV < 0 situations.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate negative EV rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if 'enterprise_value' in data:
                ev = data['enterprise_value']
                if ev < 0:
                    # More negative EV = higher rank
                    rankings[symbol] = abs(ev)
                else:
                    rankings[symbol] = 0.0
            else:
                rankings[symbol] = 0.0
        
        return rankings


class CashRichNetNet(FundamentalStrategyBase):
    """
    Cash-Rich Net Net Strategy (#19)
    
    Price < cash + S/T securities.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate cash-rich net net rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if all(key in data for key in ['price', 'cash', 'short_term_securities']):
                cash_rich = data['cash'] + data['short_term_securities']
                if data['price'] > 0 and cash_rich > data['price']:
                    # Higher discount = higher rank
                    discount = (cash_rich - data['price']) / data['price']
                    rankings[symbol] = discount
                else:
                    rankings[symbol] = 0.0
            else:
                rankings[symbol] = 0.0
        
        return rankings


class DeferredRevenueGrowth(FundamentalStrategyBase):
    """
    Deferred Revenue Growth Strategy (#20)
    
    Long when unearned rev yoy > 20%.
    """
    
    def calculate_rankings(self) -> Dict[str, float]:
        """Calculate deferred revenue growth rankings."""
        rankings = {}
        
        for symbol, data in self.fundamental_data.items():
            if 'deferred_revenue_yoy_growth' in data:
                growth = data['deferred_revenue_yoy_growth']
                if growth > 0.20:  # 20%
                    rankings[symbol] = growth
                else:
                    rankings[symbol] = 0.0
            else:
                rankings[symbol] = 0.0
        
        return rankings


# Factory function to create fundamental strategies
def create_fundamental_strategy(strategy_name: str, **kwargs) -> FundamentalStrategyBase:
    """
    Factory function to create fundamental strategies.
    
    Args:
        strategy_name: Name of the strategy to create
        **kwargs: Strategy parameters
        
    Returns:
        Strategy instance
    """
    strategy_map = {
        'price_to_book_deep_value': PriceToBookDeepValue,
        'ev_ebitda_cheapness': EVEBITDACheapnessSpread,
        'net_net_graham': NetNetGrahamBasket,
        'shiller_cape_tilt': ShillerCAPETilt,
        'price_to_sales_low_multiple': PriceToSalesLowMultiple,
        'peg_ratio_sweet_spot': PEGRatioSweetSpot,
        'owner_earnings_yield': OwnerEarningsYield,
        'piotroski_f_score': PiotroskiFScoreBooster,
        'altman_z_score': AltmanZScoreSafety,
        'net_debt_ebitda': NetDebtEBITDAConservative,
        'franchise_roic': FranchiseROIC,
        'ebit_enterprise_value': EBITEnterpriseValue,
        'magic_formula': MagicFormula,
        'quality_value_combo': QualityValueCombo,
        'dividend_growers': DividendGrowers10Yr,
        'shareholder_yield': ShareholderYield,
        'low_payout_high_reinvest': LowPayoutHighReinvest,
        'negative_enterprise_value': NegativeEnterpriseValue,
        'cash_rich_net_net': CashRichNetNet,
        'deferred_revenue_growth': DeferredRevenueGrowth,
    }
    
    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategy_map[strategy_name](**kwargs) 