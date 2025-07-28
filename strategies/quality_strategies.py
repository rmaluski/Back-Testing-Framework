"""
Profitability and Quality Strategies (Category B: 101-140)

This module implements profitability and quality-based strategies including:
- Gross Profitability Factor
- Return on Invested Capital
- Earnings Stability
- Low Debt-to-Equity Quality
- And many more...
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from trading_core import StrategyBase, MarketEvent
from trading_core.events import SignalEvent


class QualityStrategyBase(StrategyBase):
    """Base class for quality strategies."""
    
    def __init__(self, 
                 rebalance_frequency: str = "quarterly",
                 universe_size: int = 1000,
                 **kwargs):
        """
        Initialize quality strategy base.
        
        Args:
            rebalance_frequency: How often to rebalance ("monthly", "quarterly", "yearly")
            universe_size: Number of stocks in universe
            **kwargs: Additional strategy parameters
        """
        super().__init__(**kwargs)
        self.rebalance_frequency = rebalance_frequency
        self.universe_size = universe_size
        
        # Fundamental data storage
        self.fundamental_data = {}
        self.quality_scores = {}
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
        
        if self.rebalance_frequency == "monthly":
            return (current_date - self.last_rebalance).days >= 30
        elif self.rebalance_frequency == "quarterly":
            return (current_date - self.last_rebalance).days >= 90
        elif self.rebalance_frequency == "yearly":
            return (current_date - self.last_rebalance).days >= 365
        
        return False
    
    def update_fundamental_data(self, symbol: str, data: Dict[str, float]):
        """Update fundamental data for a symbol."""
        self.fundamental_data[symbol] = data
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate quality score for a symbol. Override in subclasses."""
        raise NotImplementedError
    
    def generate_quality_signals(self) -> List[SignalEvent]:
        """Generate quality-based trading signals."""
        signals = []
        
        # Calculate quality scores for all symbols
        quality_scores = {}
        for symbol in self.symbols:
            if symbol in self.fundamental_data:
                score = self.calculate_quality_score(symbol)
                quality_scores[symbol] = score
                self.quality_scores[symbol] = score
        
        # Sort by quality score
        sorted_symbols = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Generate long signals for top quality
        for symbol, score in sorted_symbols[:self.universe_size // 10]:  # Top 10%
            if score > 0 and self.current_positions.get(symbol, 0) <= 0:
                signals.append(SignalEvent(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type="LONG",
                    strength=score,
                    metadata={"strategy": self.__class__.__name__}
                ))
        
        # Generate short signals for low quality
        for symbol, score in sorted_symbols[-self.universe_size // 10:]:  # Bottom 10%
            if score < 0 and self.current_positions.get(symbol, 0) >= 0:
                signals.append(SignalEvent(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type="SHORT",
                    strength=abs(score),
                    metadata={"strategy": self.__class__.__name__}
                ))
        
        return signals
    
    def on_market_event(self, event: MarketEvent):
        """Handle market events and trigger rebalancing."""
        if self.should_rebalance(event):
            # Generate signals
            signals = self.generate_quality_signals()
            
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


class GrossProfitabilityFactor(QualityStrategyBase):
    """
    Gross Profitability Factor Strategy (#101)
    
    Long top GP/A (Gross Profit to Assets).
    """
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate Gross Profitability Factor score."""
        data = self.fundamental_data.get(symbol, {})
        
        if all(key in data for key in ['gross_profit', 'total_assets']) and data['total_assets'] > 0:
            gp_a = data['gross_profit'] / data['total_assets']
            return gp_a
        else:
            return 0.0


class ReturnOnInvestedCapital(QualityStrategyBase):
    """
    Return on Invested Capital Strategy (#102)
    
    Long sustained ROIC > 15%.
    """
    
    def __init__(self, roic_threshold: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.roic_threshold = roic_threshold
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate ROIC quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        if 'roic' in data:
            roic = data['roic']
            if roic > self.roic_threshold:
                return roic
            else:
                return 0.0
        else:
            return 0.0


class EarningsStabilityLong(QualityStrategyBase):
    """
    Earnings Stability Long Strategy (#103)
    
    σ(EPS) < σthreshold.
    """
    
    def __init__(self, stability_threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.stability_threshold = stability_threshold
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate earnings stability score."""
        data = self.fundamental_data.get(symbol, {})
        
        if 'eps_history' in data and len(data['eps_history']) > 1:
            eps_history = data['eps_history']
            eps_std = np.std(eps_history)
            eps_mean = np.mean(eps_history)
            
            if eps_mean > 0:
                coefficient_of_variation = eps_std / abs(eps_mean)
                if coefficient_of_variation < self.stability_threshold:
                    return 1.0 - coefficient_of_variation  # Higher score for lower volatility
                else:
                    return 0.0
            else:
                return 0.0
        else:
            return 0.0


class LowDebtToEquityQuality(QualityStrategyBase):
    """
    Low Debt-to-Equity Quality Strategy (#104)
    
    Long bottom decile D/E.
    """
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate debt-to-equity quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        if all(key in data for key in ['total_debt', 'total_equity']) and data['total_equity'] > 0:
            de_ratio = data['total_debt'] / data['total_equity']
            # Invert so lower D/E gets higher score
            return 1.0 / (1.0 + de_ratio)
        else:
            return 0.0


class PiotroskiMomentumCombo(QualityStrategyBase):
    """
    Piotroski + Momentum Combo Strategy (#105)
    
    Combine F-Score with momentum.
    """
    
    def __init__(self, momentum_weight: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.momentum_weight = momentum_weight
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate Piotroski + Momentum combo score."""
        data = self.fundamental_data.get(symbol, {})
        
        # Piotroski F-Score component
        f_score = data.get('f_score', 0)
        piotroski_score = f_score / 9.0  # Normalize to 0-1
        
        # Momentum component (if available)
        momentum_score = data.get('momentum_score', 0.5)  # Default to neutral
        
        # Combine scores
        combo_score = (1 - self.momentum_weight) * piotroski_score + self.momentum_weight * momentum_score
        
        return combo_score


class HighInterestCoverageFilter(QualityStrategyBase):
    """
    High Interest Coverage Filter Strategy (#106)
    
    Long high interest coverage ratio.
    """
    
    def __init__(self, coverage_threshold: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.coverage_threshold = coverage_threshold
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate interest coverage quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        if all(key in data for key in ['ebit', 'interest_expense']) and data['interest_expense'] > 0:
            coverage_ratio = data['ebit'] / data['interest_expense']
            if coverage_ratio > self.coverage_threshold:
                return coverage_ratio
            else:
                return 0.0
        else:
            return 0.0


class NegativeNetDebtScreen(QualityStrategyBase):
    """
    Negative Net Debt Screen Strategy (#107)
    
    Long companies with negative net debt.
    """
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate negative net debt quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        if all(key in data for key in ['total_debt', 'cash']):
            net_debt = data['total_debt'] - data['cash']
            if net_debt < 0:
                return abs(net_debt)  # Higher score for more negative net debt
            else:
                return 0.0
        else:
            return 0.0


class FreeCashFlowMarginGrowth(QualityStrategyBase):
    """
    Free Cash Flow Margin Growth Strategy (#108)
    
    Long growing FCF margins.
    """
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate FCF margin growth quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        if all(key in data for key in ['fcf_margin_current', 'fcf_margin_previous']) and data['revenue'] > 0:
            fcf_margin_growth = data['fcf_margin_current'] - data['fcf_margin_previous']
            if fcf_margin_growth > 0:
                return fcf_margin_growth
            else:
                return 0.0
        else:
            return 0.0


class SGAEfficiency(QualityStrategyBase):
    """
    SG&A Efficiency Strategy (#109)
    
    SG&A% rev falling.
    """
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate SG&A efficiency quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        if all(key in data for key in ['sga_current', 'sga_previous', 'revenue']) and data['revenue'] > 0:
            sga_current_pct = data['sga_current'] / data['revenue']
            sga_previous_pct = data['sga_previous'] / data['revenue']
            
            sga_improvement = sga_previous_pct - sga_current_pct  # Falling SG&A%
            if sga_improvement > 0:
                return sga_improvement
            else:
                return 0.0
        else:
            return 0.0


class HighCashReturnOnCapital(QualityStrategyBase):
    """
    High Cash Return on Capital Strategy (#110)
    
    High CROCI (Cash Return on Capital Invested).
    """
    
    def __init__(self, croci_threshold: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.croci_threshold = croci_threshold
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate CROCI quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        if all(key in data for key in ['cash_flow_from_operations', 'invested_capital']) and data['invested_capital'] > 0:
            croci = data['cash_flow_from_operations'] / data['invested_capital']
            if croci > self.croci_threshold:
                return croci
            else:
                return 0.0
        else:
            return 0.0


class ConservativeAcquisitionStrategy(QualityStrategyBase):
    """
    Conservative Acquisition Strategy (#111)
    
    Avoid serial acquirers.
    """
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate acquisition conservatism quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        # Score based on acquisition frequency and size
        acquisition_frequency = data.get('acquisitions_last_5_years', 0)
        acquisition_size_pct = data.get('acquisition_size_pct_of_assets', 0)
        
        # Lower score for frequent/large acquirers
        if acquisition_frequency <= 1 and acquisition_size_pct < 0.1:  # Conservative
            return 1.0
        elif acquisition_frequency <= 3 and acquisition_size_pct < 0.2:  # Moderate
            return 0.5
        else:  # Aggressive
            return 0.0


class WarrantyLiabilityDrop(QualityStrategyBase):
    """
    Warranty Liability Drop Strategy (#112)
    
    Long companies with declining warranty liabilities.
    """
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate warranty liability drop quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        if all(key in data for key in ['warranty_liability_current', 'warranty_liability_previous']):
            warranty_drop = data['warranty_liability_previous'] - data['warranty_liability_current']
            if warranty_drop > 0:
                return warranty_drop
            else:
                return 0.0
        else:
            return 0.0


class EmployeeSatisfactionAlpha(QualityStrategyBase):
    """
    Employee Satisfaction Alpha Strategy (#113)
    
    Long companies with high employee satisfaction.
    """
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate employee satisfaction quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        # Glassdoor rating or similar metric
        employee_rating = data.get('employee_satisfaction_rating', 0)
        if employee_rating > 3.5:  # Above average
            return employee_rating / 5.0  # Normalize to 0-1
        else:
            return 0.0


class CEOTenureStability(QualityStrategyBase):
    """
    CEO Tenure > 5 yr Stability Strategy (#114)
    
    Long companies with stable leadership.
    """
    
    def __init__(self, tenure_threshold: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.tenure_threshold = tenure_threshold
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate CEO tenure stability quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        ceo_tenure = data.get('ceo_tenure_years', 0)
        if ceo_tenure >= self.tenure_threshold:
            return min(ceo_tenure / 10.0, 1.0)  # Cap at 10 years
        else:
            return 0.0


class AuditFeeRatioDecline(QualityStrategyBase):
    """
    Audit-Fee Ratio Decline Strategy (#115)
    
    Long companies with declining audit fees relative to assets.
    """
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate audit fee ratio decline quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        if all(key in data for key in ['audit_fees_current', 'audit_fees_previous', 'total_assets']) and data['total_assets'] > 0:
            current_ratio = data['audit_fees_current'] / data['total_assets']
            previous_ratio = data['audit_fees_previous'] / data['total_assets']
            
            ratio_decline = previous_ratio - current_ratio
            if ratio_decline > 0:
                return ratio_decline
            else:
                return 0.0
        else:
            return 0.0


class RDtoSalesEfficiencyBand(QualityStrategyBase):
    """
    R&D to Sales Efficiency Band Strategy (#116)
    
    Long companies with efficient R&D spending.
    """
    
    def __init__(self, min_rd_pct: float = 0.02, max_rd_pct: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.min_rd_pct = min_rd_pct
        self.max_rd_pct = max_rd_pct
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate R&D efficiency quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        if all(key in data for key in ['rd_expense', 'revenue']) and data['revenue'] > 0:
            rd_pct = data['rd_expense'] / data['revenue']
            
            if self.min_rd_pct <= rd_pct <= self.max_rd_pct:
                # Optimal R&D spending
                return 1.0
            elif rd_pct < self.min_rd_pct:
                # Too little R&D
                return rd_pct / self.min_rd_pct
            else:
                # Too much R&D
                return max(0, 1.0 - (rd_pct - self.max_rd_pct) / self.max_rd_pct)
        else:
            return 0.0


class CapexToDepreciationRatio(QualityStrategyBase):
    """
    Capex to Depreciation < 1 Strategy (#117)
    
    Long companies with capex < depreciation (maintenance mode).
    """
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate capex to depreciation quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        if all(key in data for key in ['capital_expenditure', 'depreciation']) and data['depreciation'] > 0:
            capex_dep_ratio = data['capital_expenditure'] / data['depreciation']
            if capex_dep_ratio < 1.0:
                return 1.0 - capex_dep_ratio  # Higher score for lower ratio
            else:
                return 0.0
        else:
            return 0.0


class BeneishMScoreFraudShort(QualityStrategyBase):
    """
    Beneish M-Score Fraud Short Strategy (#118)
    
    Short companies with high M-Score (potential fraud).
    """
    
    def __init__(self, m_score_threshold: float = -2.22, **kwargs):
        super().__init__(**kwargs)
        self.m_score_threshold = m_score_threshold
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate Beneish M-Score quality score (negative for fraud)."""
        data = self.fundamental_data.get(symbol, {})
        
        m_score = data.get('beneish_m_score', 0)
        if m_score > self.m_score_threshold:
            # Potential fraud - negative score
            return -(m_score - self.m_score_threshold)
        else:
            # Clean - positive score
            return abs(m_score - self.m_score_threshold)


class HighPiotroskiLowAccrualCombo(QualityStrategyBase):
    """
    High Piotroski & Low Accrual Combo Strategy (#119)
    
    Combine high F-Score with low accruals.
    """
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate Piotroski + Accrual combo quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        # Piotroski F-Score component
        f_score = data.get('f_score', 0)
        piotroski_score = f_score / 9.0  # Normalize to 0-1
        
        # Accrual component (lower is better)
        accrual_ratio = data.get('accrual_ratio', 0)
        accrual_score = max(0, 1.0 - abs(accrual_ratio))  # Higher score for lower accruals
        
        # Combine scores (equal weight)
        combo_score = (piotroski_score + accrual_score) / 2.0
        
        return combo_score


class AssetTurnoverImprovement(QualityStrategyBase):
    """
    Asset Turnover Improvement Strategy (#120)
    
    Long companies with improving asset turnover.
    """
    
    def calculate_quality_score(self, symbol: str) -> float:
        """Calculate asset turnover improvement quality score."""
        data = self.fundamental_data.get(symbol, {})
        
        if all(key in data for key in ['asset_turnover_current', 'asset_turnover_previous']):
            turnover_improvement = data['asset_turnover_current'] - data['asset_turnover_previous']
            if turnover_improvement > 0:
                return turnover_improvement
            else:
                return 0.0
        else:
            return 0.0


# Factory function to create quality strategies
def create_quality_strategy(strategy_name: str, **kwargs) -> QualityStrategyBase:
    """
    Factory function to create quality strategies.
    
    Args:
        strategy_name: Name of the strategy to create
        **kwargs: Strategy parameters
        
    Returns:
        Strategy instance
    """
    strategy_map = {
        'gross_profitability_factor': GrossProfitabilityFactor,
        'return_on_invested_capital': ReturnOnInvestedCapital,
        'earnings_stability_long': EarningsStabilityLong,
        'low_debt_to_equity_quality': LowDebtToEquityQuality,
        'piotroski_momentum_combo': PiotroskiMomentumCombo,
        'high_interest_coverage_filter': HighInterestCoverageFilter,
        'negative_net_debt_screen': NegativeNetDebtScreen,
        'free_cash_flow_margin_growth': FreeCashFlowMarginGrowth,
        'sga_efficiency': SGAEfficiency,
        'high_cash_return_on_capital': HighCashReturnOnCapital,
        'conservative_acquisition_strategy': ConservativeAcquisitionStrategy,
        'warranty_liability_drop': WarrantyLiabilityDrop,
        'employee_satisfaction_alpha': EmployeeSatisfactionAlpha,
        'ceo_tenure_stability': CEOTenureStability,
        'audit_fee_ratio_decline': AuditFeeRatioDecline,
        'rd_to_sales_efficiency_band': RDtoSalesEfficiencyBand,
        'capex_to_depreciation_ratio': CapexToDepreciationRatio,
        'beneish_m_score_fraud_short': BeneishMScoreFraudShort,
        'high_piotroski_low_accrual_combo': HighPiotroskiLowAccrualCombo,
        'asset_turnover_improvement': AssetTurnoverImprovement,
    }
    
    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategy_map[strategy_name](**kwargs) 