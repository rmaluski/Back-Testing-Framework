"""
Reporting module for the backtesting framework.

This module handles generation of HTML reports and saving of results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import jinja2

from .config import Config


class Reporter:
    """
    Report generator for backtest results.
    
    This class generates HTML reports with interactive charts and saves
    results in various formats.
    """
    
    def __init__(self, config: Config, results: Dict[str, Any]):
        """
        Initialize the reporter.
        
        Args:
            config: Configuration object
            results: Backtest results
        """
        self.config = config
        self.results = results
        self.output_dir = Path(config.reporting.output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_reports(self):
        """Generate all reports and save results."""
        # Save data files
        if self.config.reporting.save_equity_curve:
            self._save_equity_curve()
        
        if self.config.reporting.save_fills:
            self._save_fills()
        
        if self.config.reporting.save_risk_metrics:
            self._save_risk_metrics()
        
        # Generate HTML report
        if self.config.reporting.generate_html_report:
            self._generate_html_report()
        
        # Save configuration
        self._save_config()
    
    def _save_equity_curve(self):
        """Save equity curve data."""
        if 'equity_curve' in self.results:
            equity_df = self.results['equity_curve']
            equity_df.to_csv(self.output_dir / "equity.csv")
            equity_df.to_parquet(self.output_dir / "equity.parquet")
    
    def _save_fills(self):
        """Save fill/trade data."""
        if 'trades' in self.results and self.results['trades']:
            trades_df = pd.DataFrame(self.results['trades'])
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df.to_csv(self.output_dir / "fills.csv", index=False)
            trades_df.to_parquet(self.output_dir / "fills.parquet", index=False)
    
    def _save_risk_metrics(self):
        """Save risk metrics."""
        risk_data = {
            'total_return': self.results.get('total_return', 0.0),
            'max_drawdown': self.results.get('max_drawdown', 0.0),
            'sharpe_ratio': self.results.get('sharpe_ratio', 0.0),
            'total_trades': self.results.get('total_trades', 0),
            'win_rate': self.results.get('win_rate', 0.0),
            'initial_capital': self.results.get('initial_capital', 0.0),
            'final_capital': self.results.get('final_capital', 0.0),
            'execution_time': self.results.get('execution_time', 0.0),
            'total_events': self.results.get('total_events', 0)
        }
        
        risk_df = pd.DataFrame([risk_data])
        risk_df.to_csv(self.output_dir / "risk.csv", index=False)
        
        # Save as JSON for easy parsing
        with open(self.output_dir / "risk.json", 'w') as f:
            json.dump(risk_data, f, indent=2, default=str)
    
    def _save_config(self):
        """Save configuration used for this run."""
        self.config.save(self.output_dir / "config.yaml")
    
    def _generate_html_report(self):
        """Generate interactive HTML report."""
        # Create HTML content
        html_content = self._create_html_content()
        
        # Save HTML file
        with open(self.output_dir / "report.html", 'w') as f:
            f.write(html_content)
    
    def _create_html_content(self) -> str:
        """Create the HTML content for the report."""
        # Get data for charts
        equity_df = self.results.get('equity_curve', pd.DataFrame())
        trades_df = pd.DataFrame(self.results.get('trades', []))
        
        # Create charts
        equity_chart = self._create_equity_chart(equity_df)
        drawdown_chart = self._create_drawdown_chart(equity_df)
        returns_chart = self._create_returns_chart(equity_df)
        trades_chart = self._create_trades_chart(trades_df) if not trades_df.empty else None
        
        # Create summary statistics
        summary_stats = self._create_summary_stats()
        
        # Generate HTML using template
        template = self._get_html_template()
        
        return template.render(
            title=f"Backtest Report - {self.config.strategy.class_name}",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary_stats=summary_stats,
            equity_chart=equity_chart,
            drawdown_chart=drawdown_chart,
            returns_chart=returns_chart,
            trades_chart=trades_chart,
            config=self.config.dict()
        )
    
    def _create_equity_chart(self, equity_df: pd.DataFrame) -> str:
        """Create equity curve chart."""
        if equity_df.empty:
            return "<p>No equity curve data available</p>"
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equity_df.index,
            y=equity_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Portfolio Value Over Time',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            height=400
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _create_drawdown_chart(self, equity_df: pd.DataFrame) -> str:
        """Create drawdown chart."""
        if equity_df.empty:
            return "<p>No equity curve data available</p>"
        
        # Calculate drawdown
        rolling_max = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equity_df.index,
            y=drawdown,
            mode='lines',
            name='Drawdown (%)',
            line=dict(color='red', width=2),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title='Drawdown Over Time',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            height=400
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _create_returns_chart(self, equity_df: pd.DataFrame) -> str:
        """Create returns distribution chart."""
        if equity_df.empty:
            return "<p>No equity curve data available</p>"
        
        # Calculate daily returns
        daily_returns = equity_df['portfolio_value'].resample('D').last().pct_change().dropna()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=daily_returns,
            nbinsx=50,
            name='Daily Returns',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Daily Returns Distribution',
            xaxis_title='Daily Return',
            yaxis_title='Frequency',
            height=400
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _create_trades_chart(self, trades_df: pd.DataFrame) -> str:
        """Create trades analysis chart."""
        if trades_df.empty:
            return "<p>No trades data available</p>"
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in trades_df.columns:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.set_index('timestamp')
        
        # Create subplot with trade prices and volumes
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Trade Prices', 'Trade Volumes'),
            vertical_spacing=0.1
        )
        
        # Trade prices
        fig.add_trace(
            go.Scatter(
                x=trades_df.index,
                y=trades_df['price'],
                mode='markers',
                name='Trade Price',
                marker=dict(
                    color=trades_df['side'].map({'BUY': 'green', 'SELL': 'red'}),
                    size=8
                )
            ),
            row=1, col=1
        )
        
        # Trade volumes
        fig.add_trace(
            go.Bar(
                x=trades_df.index,
                y=trades_df['quantity'],
                name='Trade Volume',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Trade Analysis',
            height=600,
            showlegend=False
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)
    
    def _create_summary_stats(self) -> Dict[str, Any]:
        """Create summary statistics."""
        return {
            'strategy_name': self.config.strategy.class_name,
            'start_date': self.config.data.start_date,
            'end_date': self.config.data.end_date,
            'symbols': ', '.join(self.config.data.symbols),
            'initial_capital': f"${self.results.get('initial_capital', 0):,.2f}",
            'final_capital': f"${self.results.get('final_capital', 0):,.2f}",
            'total_return': f"{self.results.get('total_return', 0):.2%}",
            'max_drawdown': f"{self.results.get('max_drawdown', 0):.2%}",
            'sharpe_ratio': f"{self.results.get('sharpe_ratio', 0):.2f}",
            'total_trades': self.results.get('total_trades', 0),
            'win_rate': f"{self.results.get('win_rate', 0):.2%}",
            'execution_time': f"{self.results.get('execution_time', 0):.2f}s",
            'total_events': f"{self.results.get('total_events', 0):,}"
        }
    
    def _get_html_template(self) -> jinja2.Template:
        """Get the HTML template for the report."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }
        .header h1 {
            color: #333;
            margin: 0;
        }
        .header p {
            color: #666;
            margin: 5px 0;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }
        .summary-card .value {
            font-size: 24px;
            font-weight: bold;
        }
        .chart-section {
            margin-bottom: 40px;
        }
        .chart-section h2 {
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .config-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
        }
        .config-section h2 {
            color: #333;
            margin-bottom: 15px;
        }
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        .config-item {
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        .config-item h4 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .config-item pre {
            margin: 0;
            font-size: 12px;
            color: #666;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 3px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <p>Generated on {{ timestamp }}</p>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Return</h3>
                <div class="value">{{ summary_stats.total_return }}</div>
            </div>
            <div class="summary-card">
                <h3>Max Drawdown</h3>
                <div class="value">{{ summary_stats.max_drawdown }}</div>
            </div>
            <div class="summary-card">
                <h3>Sharpe Ratio</h3>
                <div class="value">{{ summary_stats.sharpe_ratio }}</div>
            </div>
            <div class="summary-card">
                <h3>Total Trades</h3>
                <div class="value">{{ summary_stats.total_trades }}</div>
            </div>
            <div class="summary-card">
                <h3>Win Rate</h3>
                <div class="value">{{ summary_stats.win_rate }}</div>
            </div>
            <div class="summary-card">
                <h3>Execution Time</h3>
                <div class="value">{{ summary_stats.execution_time }}</div>
            </div>
        </div>
        
        <div class="chart-section">
            <h2>Portfolio Performance</h2>
            <div class="chart-container">
                {{ equity_chart | safe }}
            </div>
        </div>
        
        <div class="chart-section">
            <h2>Drawdown Analysis</h2>
            <div class="chart-container">
                {{ drawdown_chart | safe }}
            </div>
        </div>
        
        <div class="chart-section">
            <h2>Returns Distribution</h2>
            <div class="chart-container">
                {{ returns_chart | safe }}
            </div>
        </div>
        
        {% if trades_chart %}
        <div class="chart-section">
            <h2>Trade Analysis</h2>
            <div class="chart-container">
                {{ trades_chart | safe }}
            </div>
        </div>
        {% endif %}
        
        <div class="config-section">
            <h2>Configuration</h2>
            <div class="config-grid">
                <div class="config-item">
                    <h4>Strategy</h4>
                    <pre>{{ summary_stats.strategy_name }}</pre>
                </div>
                <div class="config-item">
                    <h4>Date Range</h4>
                    <pre>{{ summary_stats.start_date }} to {{ summary_stats.end_date }}</pre>
                </div>
                <div class="config-item">
                    <h4>Symbols</h4>
                    <pre>{{ summary_stats.symbols }}</pre>
                </div>
                <div class="config-item">
                    <h4>Capital</h4>
                    <pre>{{ summary_stats.initial_capital }} → {{ summary_stats.final_capital }}</pre>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        return jinja2.Template(template_str) 