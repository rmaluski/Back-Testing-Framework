# Back-Testing Framework

A high-performance, event-driven backtesting framework with C++ core and Python bindings for quantitative trading research.

## 🚀 Features

- **Speed**: Simulate 20 years of 1-second ticks (~500M events) in <60s on 16-core desktop
- **Breadth**: Sweep 10,000 parameter grids overnight (<8h) with Ray or GNU parallel
- **Realism**: Unified models for slippage, queue priority, exchange fees, borrow costs
- **Re-use**: Same strategy DLL runs in back-test → paper-trade → live stack
- **Reporting**: One-click HTML tear-sheet with returns, drawdown, factor attribution

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
