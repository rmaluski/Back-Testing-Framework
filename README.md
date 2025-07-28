# Back-Testing Framework

A high-performance, event-driven backtesting framework with C++ core and Python bindings for quantitative trading research.

## 🚀 Features

- **Speed**: Simulate 20 years of 1-second ticks (~500M events) in <60s on 16-core desktop
- **Breadth**: Sweep 10,000 parameter grids overnight (<8h) with Ray or GNU parallel
- **Realism**: Unified models for slippage, queue priority, exchange fees, borrow costs
- **Re-use**: Same strategy DLL runs in back-test → paper-trade → live stack
- **Reporting**: One-click HTML tear-sheet with returns, drawdown, factor attribution
- **Comprehensive Strategy Library**: 500+ strategies across fundamental, momentum, quality, and technical categories
- **Strategy Combinations**: Multi-factor approaches with weighted, consensus, and voting methods
- **Modular Design**: Easy to add new strategies and combine existing ones

## 🏗️ Architecture

```
      +-------------------+      +------------------+
      |  Dataset Cleaner  |----▶ |  MarketDataFeed  |  arrow batches
      +-------------------+      +--------+---------+
                                         │
                                         ▼
                               +--------------------+
                               |   Event Loop       |
                               | (C++ Reactor)      |
                               +----+-------+-------+
                                    │       │
                                    │       │
                           +--------+--+  +--+---------+
                           | OrderBook |  | FillEngine |
                           +--------+--+  +--+---------+
                                    │       ▲
                                    │       │ fills
           Py strategy hook         │       │
      (PyBind11 -> C++)             │       │
        signal events ──────────────┘       │
                                            │
                                    +-------+-------+
                                    |  Portfolio     |
                                    |  Ledger        |
                                    +-------+-------+
                                            │
                                            ▼
                                +-----------------------+
                                | Analytics Reporter    |
                                | (Pandas + Plotly)     |
                                +-----------------------+
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/rmaluski/Back-Testing-Framework.git
cd Back-Testing-Framework

# Install dependencies
pip install -r requirements.txt

# Build C++ core
mkdir build && cd build
cmake ..
make -j$(nproc)

# Install CLI tool
pip install -e .
```

## 🎯 Quick Start

### Basic Strategy Example

```python
# strategy.py
from trading_core import StrategyBase, MarketEvent

class Bollinger(StrategyBase):
    def __init__(self, lookback=20, z=2):
        self.lb, self.z = lookback, z
        self.window = []

    def on_bar(self, bar):
        self.window.append(bar.close)
        if len(self.window) < self.lb: return

        mean = sum(self.window[-self.lb:]) / self.lb
        stdev = (sum((x-mean)**2 for x in self.window[-self.lb:])/self.lb)**0.5
        upper = mean + self.z*stdev
        lower = mean - self.z*stdev

        if bar.close > upper:
            self.sell(bar.symbol, qty=1, tif="DAY")
        elif bar.close < lower:
            self.buy(bar.symbol, qty=1, tif="DAY")
```

Run the backtest:

```bash
bt run configs/bollinger_es.yaml --strategy strategy.Bollinger
```

### Comprehensive Strategy Library

The framework includes 500+ pre-built strategies across multiple categories:

#### Fundamental Strategies (Category A: 1-100)

```python
from strategies.fundamental_strategies import create_fundamental_strategy

# Price-to-Book Deep Value
strategy = create_fundamental_strategy('price_to_book_deep_value')

# Magic Formula (EBIT/EV + ROIC)
strategy = create_fundamental_strategy('magic_formula')

# Quality-Value Combo
strategy = create_fundamental_strategy('quality_value_combo')
```

#### Momentum Strategies (Categories D & E: 161-340)

```python
from strategies.momentum_strategies import create_momentum_strategy

# Cross-Sectional Momentum
strategy = create_momentum_strategy('cross_sectional_momentum')

# Moving Average Crossover
strategy = create_momentum_strategy('moving_average_crossover')

# RSI Signal Strategy
strategy = create_momentum_strategy('rsi_signal')
```

#### Quality Strategies (Category B: 101-140)

```python
from strategies.quality_strategies import create_quality_strategy

# Gross Profitability Factor
strategy = create_quality_strategy('gross_profitability_factor')

# Return on Invested Capital
strategy = create_quality_strategy('return_on_invested_capital')
```

#### Strategy Combinations

```python
from strategies.strategy_combiner import create_strategy_combo

# Value + Momentum Combination
strategy = create_strategy_combo('value_momentum_combo')

# Quality + Value + Momentum
strategy = create_strategy_combo('quality_value_momentum_combo')

# Custom Multi-Factor Combination
custom_strategies = [
    {'name': 'price_to_book_deep_value', 'type': 'fundamental', 'weight': 0.4},
    {'name': 'gross_profitability_factor', 'type': 'quality', 'weight': 0.3},
    {'name': 'cross_sectional_momentum', 'type': 'momentum', 'weight': 0.3}
]
strategy = create_custom_combo(custom_strategies, combination_method="weighted_sum")
```

Run comprehensive examples:

```bash
# Run strategy showcase
python examples/strategy_showcase.py

# Run individual strategy categories
bt run configs/fundamental_strategies.yaml --strategy magic_formula
bt run configs/momentum_strategies.yaml --strategy cross_sectional_momentum
bt run configs/strategy_combinations.yaml --strategy value_momentum_combo
```

## 📊 Output

Each run generates:

- `equity.csv` - Portfolio value over time
- `fills.parquet` - Trade execution details
- `risk.csv` - Risk metrics and exposures
- `report.html` - Interactive tear-sheet with Plotly charts
- `config.yaml` - Complete parameter set used

## 🔧 Configuration

```yaml
# configs/bollinger_es.yaml
data:
  source: "tickdb:///data/es_futures.parquet"
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  symbols: ["ES"]

strategy:
  class: "strategy.Bollinger"
  params:
    lookback: 20
    z: 2.0

execution:
  slippage_bps: 1.0
  commission_bps: 0.5
  min_tick_size: 0.25

risk:
  max_position_size: 100
  max_drawdown: 0.15
  stop_loss_bps: 200
```

## 🚀 Parallel Parameter Sweeps

```bash
# Sweep 10,000 parameter combinations
bt sweep grid.yml --parallel --workers 12

# Grid configuration
lookback: [20, 40, 60]
z: [1.5, 2.0, 2.5]
slippage_bps: [0.5, 1.0, 2.0]
```

## 📚 Strategy Library

The framework includes a comprehensive library of 500+ strategies organized into categories:

### Fundamental Strategies (Category A: 1-100)

- **Price-to-Book Deep Value**: Long lowest decile, short highest
- **EV/EBITDA Cheapness Spread**: Monthly rebalance long low quintile
- **Net-Net Graham Basket**: Buy if Price < 0.66×NCAV
- **Magic Formula**: (EBIT/EV + ROIC) - top 30 names
- **Quality-Value Combo**: Rank equally by P/B and ROE
- **Shareholder Yield**: Aggregate buybacks + dividend yield
- **Negative Enterprise Value**: Buy EV < 0 situations
- And 90+ more fundamental strategies...

### Momentum Strategies (Categories D & E: 161-340)

- **Cross-Sectional Momentum**: 12-1 Month Relative Strength
- **Time-Series Momentum**: 6-1 Month Momentum
- **Moving Average Crossover**: Fast MA vs Slow MA
- **RSI Momentum**: RSI-based momentum signals
- **MACD Momentum**: MACD histogram momentum
- **Bollinger Bands Momentum**: Price position relative to bands
- **Volume-Weighted Momentum**: Momentum weighted by volume
- And 170+ more momentum strategies...

### Quality Strategies (Category B: 101-140)

- **Gross Profitability Factor**: Long top GP/A
- **Return on Invested Capital**: Long sustained ROIC > 15%
- **Earnings Stability**: σ(EPS) < σthreshold
- **Low Debt-to-Equity Quality**: Long bottom decile D/E
- **High Interest Coverage**: Long high coverage ratio
- **Negative Net Debt Screen**: Long negative net debt
- And 30+ more quality strategies...

### Strategy Combinations

- **Value + Momentum**: Combines fundamental value with momentum
- **Quality + Value + Momentum**: Three-factor combination
- **Technical + Fundamental**: Combines technical indicators with fundamental analysis
- **Multi-Factor Model**: Equal-weighted factor combination
- **Custom Combinations**: Create your own multi-strategy approaches

### Combination Methods

- **Weighted Sum**: Weighted average of signals
- **Majority Vote**: Majority vote on direction
- **Consensus**: All strategies must agree
- **Hierarchical**: Apply strategies in order

### Usage Examples

```bash
# Run individual strategies
bt run configs/fundamental_strategies.yaml --strategy magic_formula
bt run configs/momentum_strategies.yaml --strategy cross_sectional_momentum
bt run configs/strategy_combinations.yaml --strategy value_momentum_combo

# Parameter sweeps across strategies
bt sweep configs/fundamental_strategies.yaml configs/parameter_sweep_grid.yaml --workers 8

# Run comprehensive showcase
python examples/strategy_showcase.py
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run performance benchmarks
python benchmarks/performance_test.py

# Run compliance tests
python tests/compliance_test.py
```

## 📈 Performance Targets

- **Speed**: 500M ticks in <60s (16-core)
- **Memory**: <8GB RAM for 20-year simulation
- **Accuracy**: Slippage error ≤0.1bp vs analytic reference
- **Determinism**: Same seed → identical equity vectors

## 🔗 Integration

This framework integrates with:

- **Dataset Cleaner** (Module 0.1) - Arrow/Parquet data ingestion
- **Schema Registry** (Module 0.2) - Data validation
- **Data-Quality Monitor** (Module 2) - Anomaly detection
- **CI/CD Deployer** (Module 7) - Canary testing
- **Factor-Model Builder** (Module 14) - Performance attribution

## 📚 Documentation

- [API Reference](docs/api.md)
- [Strategy Development Guide](docs/strategies.md)
- [Performance Tuning](docs/performance.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🏆 Roadmap

This is Module 1 of a comprehensive quantitative trading system. See the full roadmap in [ROADMAP.md](ROADMAP.md) for the complete 18-module architecture.
