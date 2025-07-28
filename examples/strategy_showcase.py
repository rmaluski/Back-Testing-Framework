#!/usr/bin/env python3
"""
Strategy Showcase Example

This script demonstrates how to use the comprehensive strategy library
including fundamental, momentum, quality, and combination strategies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading_core import StrategyBase, MarketEvent
from trading_core.engine import BacktestEngine
from trading_core.config import Config, DataConfig, StrategyConfig, ExecutionConfig, RiskConfig, ReportingConfig

# Import strategy modules
from strategies.fundamental_strategies import (
    create_fundamental_strategy,
    PriceToBookDeepValue,
    MagicFormula,
    QualityValueCombo
)
from strategies.momentum_strategies import (
    create_momentum_strategy,
    CrossSectionalMomentum,
    RSISignalStrategy
)
from strategies.quality_strategies import (
    create_quality_strategy,
    GrossProfitabilityFactor,
    ReturnOnInvestedCapital
)
from strategies.strategy_combiner import (
    create_strategy_combo,
    ValueMomentumCombo,
    QualityValueMomentumCombo,
    create_custom_combo
)


def create_sample_fundamental_data():
    """Create sample fundamental data for testing."""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    fundamental_data = {}
    for symbol in symbols:
        fundamental_data[symbol] = {
            'price_to_book': np.random.uniform(1.0, 10.0),
            'ev_ebitda': np.random.uniform(5.0, 25.0),
            'price_to_sales': np.random.uniform(1.0, 15.0),
            'peg_ratio': np.random.uniform(0.5, 2.0),
            'roe': np.random.uniform(0.05, 0.30),
            'roic': np.random.uniform(0.08, 0.25),
            'f_score': np.random.randint(0, 10),
            'altman_z_score': np.random.uniform(1.0, 5.0),
            'net_debt': np.random.uniform(0, 1000000000),
            'ebitda': np.random.uniform(100000000, 5000000000),
            'ebit': np.random.uniform(50000000, 3000000000),
            'enterprise_value': np.random.uniform(1000000000, 10000000000),
            'owner_earnings': np.random.uniform(10000000, 1000000000),
            'market_cap': np.random.uniform(500000000, 5000000000),
            'dividend_cagr_10yr': np.random.uniform(0.02, 0.15),
            'buyback_yield': np.random.uniform(0.01, 0.08),
            'dividend_yield': np.random.uniform(0.0, 0.05),
            'payout_ratio': np.random.uniform(0.1, 0.6),
            'current_assets': np.random.uniform(50000000, 2000000000),
            'total_liabilities': np.random.uniform(20000000, 1000000000),
            'price': np.random.uniform(50, 500),
            'cash': np.random.uniform(10000000, 500000000),
            'short_term_securities': np.random.uniform(5000000, 200000000),
            'deferred_revenue_yoy_growth': np.random.uniform(0.05, 0.40),
            'gross_profit': np.random.uniform(10000000, 1000000000),
            'total_assets': np.random.uniform(100000000, 5000000000),
            'eps_history': [np.random.uniform(1, 10) for _ in range(20)],
            'total_debt': np.random.uniform(0, 2000000000),
            'total_equity': np.random.uniform(50000000, 3000000000),
            'momentum_score': np.random.uniform(0.3, 0.7),
            'interest_expense': np.random.uniform(1000000, 100000000),
            'fcf_margin_current': np.random.uniform(0.05, 0.25),
            'fcf_margin_previous': np.random.uniform(0.03, 0.20),
            'revenue': np.random.uniform(100000000, 5000000000),
            'sga_current': np.random.uniform(5000000, 500000000),
            'sga_previous': np.random.uniform(6000000, 600000000),
            'cash_flow_from_operations': np.random.uniform(20000000, 2000000000),
            'invested_capital': np.random.uniform(50000000, 3000000000),
            'acquisitions_last_5_years': np.random.randint(0, 5),
            'acquisition_size_pct_of_assets': np.random.uniform(0.01, 0.3),
            'warranty_liability_current': np.random.uniform(1000000, 50000000),
            'warranty_liability_previous': np.random.uniform(1200000, 60000000),
            'employee_satisfaction_rating': np.random.uniform(3.0, 4.5),
            'ceo_tenure_years': np.random.uniform(1, 15),
            'audit_fees_current': np.random.uniform(100000, 5000000),
            'audit_fees_previous': np.random.uniform(120000, 6000000),
            'rd_expense': np.random.uniform(1000000, 100000000),
            'capital_expenditure': np.random.uniform(5000000, 500000000),
            'depreciation': np.random.uniform(3000000, 300000000),
            'beneish_m_score': np.random.uniform(-3.0, 1.0),
            'accrual_ratio': np.random.uniform(-0.1, 0.2),
            'asset_turnover_current': np.random.uniform(0.5, 2.0),
            'asset_turnover_previous': np.random.uniform(0.4, 1.8),
        }
    
    return fundamental_data


def create_sample_price_data():
    """Create sample price data for testing."""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Generate daily data
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    price_data = []
    for symbol in symbols:
        # Generate random walk price data
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        initial_price = np.random.uniform(50, 500)
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        for i, date in enumerate(dates):
            price_data.append({
                'timestamp': date,
                'symbol': symbol,
                'open': prices[i] * (1 + np.random.normal(0, 0.005)),
                'high': prices[i] * (1 + abs(np.random.normal(0, 0.01))),
                'low': prices[i] * (1 - abs(np.random.normal(0, 0.01))),
                'close': prices[i],
                'volume': np.random.randint(1000000, 10000000)
            })
    
    return pd.DataFrame(price_data)


def run_fundamental_strategy_example():
    """Run fundamental strategy example."""
    print("\n=== Fundamental Strategy Example ===")
    
    # Create sample data
    fundamental_data = create_sample_fundamental_data()
    price_data = create_sample_price_data()
    
    # Save data
    os.makedirs('data', exist_ok=True)
    price_data.to_parquet('data/price_data.parquet', index=False)
    
    # Create strategy
    strategy = create_fundamental_strategy('magic_formula', 
                                         symbols=list(fundamental_data.keys()),
                                         initial_capital=100000.0)
    
    # Update fundamental data
    for symbol, data in fundamental_data.items():
        strategy.update_fundamental_data(symbol, data)
    
    # Create configuration
    config = Config(
        data=DataConfig(
            source="data/price_data.parquet",
            start_date="2020-01-01",
            end_date="2023-12-31",
            symbols=list(fundamental_data.keys()),
            bar_size="1D"
        ),
        strategy=StrategyConfig(
            class_name="MagicFormula",
            module_path="strategies.fundamental_strategies",
            params={},
            initial_capital=100000.0
        ),
        execution=ExecutionConfig(
            slippage_bps=1.0,
            commission_bps=0.5,
            min_tick_size=0.01
        ),
        risk=RiskConfig(
            max_position_size=100,
            max_drawdown=0.15,
            position_limit_pct=0.1
        ),
        reporting=ReportingConfig(
            output_dir="./results/magic_formula_example",
            save_equity_curve=True,
            save_fills=True,
            save_risk_metrics=True,
            generate_html_report=True
        )
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    engine.set_strategy(strategy)
    results = engine.run()
    
    print(f"Magic Formula Strategy Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Total Trades: {results['total_trades']}")


def run_momentum_strategy_example():
    """Run momentum strategy example."""
    print("\n=== Momentum Strategy Example ===")
    
    # Create sample data
    price_data = create_sample_price_data()
    
    # Save data
    os.makedirs('data', exist_ok=True)
    price_data.to_parquet('data/price_data.parquet', index=False)
    
    # Create strategy
    strategy = create_momentum_strategy('cross_sectional_momentum',
                                      symbols=price_data['symbol'].unique().tolist(),
                                      initial_capital=100000.0)
    
    # Create configuration
    config = Config(
        data=DataConfig(
            source="data/price_data.parquet",
            start_date="2020-01-01",
            end_date="2023-12-31",
            symbols=price_data['symbol'].unique().tolist(),
            bar_size="1D"
        ),
        strategy=StrategyConfig(
            class_name="CrossSectionalMomentum",
            module_path="strategies.momentum_strategies",
            params={},
            initial_capital=100000.0
        ),
        execution=ExecutionConfig(
            slippage_bps=1.0,
            commission_bps=0.5,
            min_tick_size=0.01
        ),
        risk=RiskConfig(
            max_position_size=100,
            max_drawdown=0.15,
            position_limit_pct=0.1
        ),
        reporting=ReportingConfig(
            output_dir="./results/momentum_example",
            save_equity_curve=True,
            save_fills=True,
            save_risk_metrics=True,
            generate_html_report=True
        )
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    engine.set_strategy(strategy)
    results = engine.run()
    
    print(f"Cross-Sectional Momentum Strategy Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Total Trades: {results['total_trades']}")


def run_quality_strategy_example():
    """Run quality strategy example."""
    print("\n=== Quality Strategy Example ===")
    
    # Create sample data
    fundamental_data = create_sample_fundamental_data()
    price_data = create_sample_price_data()
    
    # Save data
    os.makedirs('data', exist_ok=True)
    price_data.to_parquet('data/price_data.parquet', index=False)
    
    # Create strategy
    strategy = create_quality_strategy('gross_profitability_factor',
                                     symbols=list(fundamental_data.keys()),
                                     initial_capital=100000.0)
    
    # Update fundamental data
    for symbol, data in fundamental_data.items():
        strategy.update_fundamental_data(symbol, data)
    
    # Create configuration
    config = Config(
        data=DataConfig(
            source="data/price_data.parquet",
            start_date="2020-01-01",
            end_date="2023-12-31",
            symbols=list(fundamental_data.keys()),
            bar_size="1D"
        ),
        strategy=StrategyConfig(
            class_name="GrossProfitabilityFactor",
            module_path="strategies.quality_strategies",
            params={},
            initial_capital=100000.0
        ),
        execution=ExecutionConfig(
            slippage_bps=1.0,
            commission_bps=0.5,
            min_tick_size=0.01
        ),
        risk=RiskConfig(
            max_position_size=100,
            max_drawdown=0.15,
            position_limit_pct=0.1
        ),
        reporting=ReportingConfig(
            output_dir="./results/quality_example",
            save_equity_curve=True,
            save_fills=True,
            save_risk_metrics=True,
            generate_html_report=True
        )
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    engine.set_strategy(strategy)
    results = engine.run()
    
    print(f"Gross Profitability Factor Strategy Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Total Trades: {results['total_trades']}")


def run_strategy_combination_example():
    """Run strategy combination example."""
    print("\n=== Strategy Combination Example ===")
    
    # Create sample data
    fundamental_data = create_sample_fundamental_data()
    price_data = create_sample_price_data()
    
    # Save data
    os.makedirs('data', exist_ok=True)
    price_data.to_parquet('data/price_data.parquet', index=False)
    
    # Create strategy combination
    strategy = create_strategy_combo('value_momentum_combo',
                                   symbols=list(fundamental_data.keys()),
                                   initial_capital=100000.0)
    
    # Update fundamental data for fundamental strategies
    for symbol, data in fundamental_data.items():
        strategy.update_fundamental_data(symbol, data)
    
    # Create configuration
    config = Config(
        data=DataConfig(
            source="data/price_data.parquet",
            start_date="2020-01-01",
            end_date="2023-12-31",
            symbols=list(fundamental_data.keys()),
            bar_size="1D"
        ),
        strategy=StrategyConfig(
            class_name="ValueMomentumCombo",
            module_path="strategies.strategy_combiner",
            params={},
            initial_capital=100000.0
        ),
        execution=ExecutionConfig(
            slippage_bps=1.0,
            commission_bps=0.5,
            min_tick_size=0.01
        ),
        risk=RiskConfig(
            max_position_size=100,
            max_drawdown=0.15,
            position_limit_pct=0.1
        ),
        reporting=ReportingConfig(
            output_dir="./results/combination_example",
            save_equity_curve=True,
            save_fills=True,
            save_risk_metrics=True,
            generate_html_report=True
        )
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    engine.set_strategy(strategy)
    results = engine.run()
    
    print(f"Value + Momentum Combination Strategy Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Total Trades: {results['total_trades']}")
    
    # Get sub-strategy performance
    sub_performance = strategy.get_strategy_performance()
    print(f"\nSub-Strategy Performance:")
    for strategy_name, perf in sub_performance.items():
        print(f"  {strategy_name}: {perf.get('total_pnl', 0):.2f}")


def run_custom_combination_example():
    """Run custom strategy combination example."""
    print("\n=== Custom Strategy Combination Example ===")
    
    # Create sample data
    fundamental_data = create_sample_fundamental_data()
    price_data = create_sample_price_data()
    
    # Save data
    os.makedirs('data', exist_ok=True)
    price_data.to_parquet('data/price_data.parquet', index=False)
    
    # Create custom strategy combination
    custom_strategies = [
        {
            'name': 'price_to_book_deep_value',
            'type': 'fundamental',
            'weight': 0.4,
            'params': {}
        },
        {
            'name': 'gross_profitability_factor',
            'type': 'quality',
            'weight': 0.3,
            'params': {}
        },
        {
            'name': 'cross_sectional_momentum',
            'type': 'momentum',
            'weight': 0.3,
            'params': {}
        }
    ]
    
    strategy = create_custom_combo(
        strategies=custom_strategies,
        combination_method="weighted_sum",
        symbols=list(fundamental_data.keys()),
        initial_capital=100000.0
    )
    
    # Update fundamental data
    for symbol, data in fundamental_data.items():
        strategy.update_fundamental_data(symbol, data)
    
    # Create configuration
    config = Config(
        data=DataConfig(
            source="data/price_data.parquet",
            start_date="2020-01-01",
            end_date="2023-12-31",
            symbols=list(fundamental_data.keys()),
            bar_size="1D"
        ),
        strategy=StrategyConfig(
            class_name="StrategyCombiner",
            module_path="strategies.strategy_combiner",
            params={
                'strategies': custom_strategies,
                'combination_method': 'weighted_sum'
            },
            initial_capital=100000.0
        ),
        execution=ExecutionConfig(
            slippage_bps=1.0,
            commission_bps=0.5,
            min_tick_size=0.01
        ),
        risk=RiskConfig(
            max_position_size=100,
            max_drawdown=0.15,
            position_limit_pct=0.1
        ),
        reporting=ReportingConfig(
            output_dir="./results/custom_combination_example",
            save_equity_curve=True,
            save_fills=True,
            save_risk_metrics=True,
            generate_html_report=True
        )
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    engine.set_strategy(strategy)
    results = engine.run()
    
    print(f"Custom Strategy Combination Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Total Trades: {results['total_trades']}")


def main():
    """Run all strategy examples."""
    print("Strategy Showcase - Comprehensive Strategy Library Demo")
    print("=" * 60)
    
    try:
        # Run individual strategy examples
        run_fundamental_strategy_example()
        run_momentum_strategy_example()
        run_quality_strategy_example()
        run_strategy_combination_example()
        run_custom_combination_example()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("\nAvailable Strategy Categories:")
        print("1. Fundamental Strategies (Category A: 1-100)")
        print("   - Price-to-Book Deep Value")
        print("   - EV/EBITDA Cheapness Spread")
        print("   - Magic Formula")
        print("   - Quality-Value Combo")
        print("   - And many more...")
        
        print("\n2. Momentum Strategies (Categories D & E: 161-340)")
        print("   - Cross-Sectional Momentum")
        print("   - Time-Series Momentum")
        print("   - Moving Average Crossover")
        print("   - RSI/MACD Signals")
        print("   - And many more...")
        
        print("\n3. Quality Strategies (Category B: 101-140)")
        print("   - Gross Profitability Factor")
        print("   - Return on Invested Capital")
        print("   - Earnings Stability")
        print("   - And many more...")
        
        print("\n4. Strategy Combinations")
        print("   - Value + Momentum")
        print("   - Quality + Value + Momentum")
        print("   - Technical + Fundamental")
        print("   - Multi-Factor Model")
        print("   - Custom combinations")
        
        print("\nUsage Examples:")
        print("  # Run individual strategy")
        print("  bt run configs/fundamental_strategies.yaml --strategy magic_formula")
        print("  bt run configs/momentum_strategies.yaml --strategy cross_sectional_momentum")
        
        print("\n  # Run strategy combination")
        print("  bt run configs/strategy_combinations.yaml --strategy value_momentum_combo")
        
        print("\n  # Parameter sweep")
        print("  bt sweep configs/fundamental_strategies.yaml configs/parameter_sweep_grid.yaml --workers 4")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 