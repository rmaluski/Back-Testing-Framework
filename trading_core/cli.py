"""
Command-line interface for the backtesting framework.

This module provides the CLI tools for running backtests and parameter sweeps.
"""

import importlib
import sys
from pathlib import Path
from typing import Optional
import click
import yaml

from .config import Config, load_config_with_defaults, create_sweep_configs
from .engine import BacktestEngine
from .reporter import Reporter


@click.group()
@click.version_option()
def main():
    """High-performance event-driven backtesting framework."""
    pass


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--strategy', '-s', help='Strategy class to use')
@click.option('--output-dir', '-o', help='Output directory for results')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--dry-run', is_flag=True, help='Validate config without running')
def run(config_file: str, strategy: Optional[str], output_dir: Optional[str], 
        verbose: bool, dry_run: bool):
    """Run a single backtest."""
    try:
        # Load configuration
        config = load_config_with_defaults(config_file)
        
        # Override strategy if specified
        if strategy:
            config.strategy.class_name = strategy
        
        # Override output directory if specified
        if output_dir:
            config.reporting.output_dir = output_dir
        
        if verbose:
            click.echo(f"Configuration loaded from: {config_file}")
            click.echo(f"Strategy: {config.strategy.class_name}")
            click.echo(f"Symbols: {config.data.symbols}")
            click.echo(f"Date range: {config.data.start_date} to {config.data.end_date}")
            click.echo(f"Initial capital: ${config.strategy.initial_capital:,.2f}")
        
        if dry_run:
            click.echo("Configuration validation passed!")
            return
        
        # Create output directory
        output_path = Path(config.reporting.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save config to output directory
        config.save(output_path / "config.yaml")
        
        # Initialize engine
        engine = BacktestEngine(config)
        
        # Load strategy
        strategy_instance = _load_strategy(config)
        engine.set_strategy(strategy_instance)
        
        if verbose:
            click.echo("Starting backtest...")
        
        # Run backtest
        results = engine.run()
        
        if verbose:
            click.echo("Backtest completed!")
            click.echo(f"Total return: {results['total_return']:.2%}")
            click.echo(f"Max drawdown: {results['max_drawdown']:.2%}")
            click.echo(f"Total trades: {results['total_trades']}")
        
        # Generate reports
        reporter = Reporter(config, results)
        reporter.generate_reports()
        
        click.echo(f"Results saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.argument('grid_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', help='Output directory for results')
@click.option('--workers', '-w', default=1, help='Number of parallel workers')
@click.option('--max-configs', default=None, type=int, help='Maximum number of configs to run')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def sweep(config_file: str, grid_file: str, output_dir: Optional[str], 
          workers: int, max_configs: Optional[int], verbose: bool):
    """Run parameter sweep across multiple configurations."""
    try:
        # Load base configuration
        base_config = load_config_with_defaults(config_file)
        
        # Load parameter grid
        with open(grid_file, 'r') as f:
            param_grid = yaml.safe_load(f)
        
        if verbose:
            click.echo(f"Parameter grid loaded from: {grid_file}")
            total_combinations = 1
            for param, values in param_grid.items():
                total_combinations *= len(values)
                click.echo(f"  {param}: {values}")
            click.echo(f"Total combinations: {total_combinations}")
        
        # Create sweep configurations
        sweep_configs = create_sweep_configs(base_config, param_grid)
        
        if max_configs:
            sweep_configs = sweep_configs[:max_configs]
            if verbose:
                click.echo(f"Limited to {max_configs} configurations")
        
        # Set output directory
        if output_dir:
            base_output_dir = output_dir
        else:
            base_output_dir = f"./sweep_results_{Path(config_file).stem}"
        
        if verbose:
            click.echo(f"Running {len(sweep_configs)} configurations...")
        
        # Run sweep
        results = _run_sweep(sweep_configs, base_output_dir, workers, verbose)
        
        # Generate sweep summary
        _generate_sweep_summary(results, base_output_dir)
        
        click.echo(f"Sweep completed! Results saved to: {base_output_dir}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
def validate(config_file: str):
    """Validate a configuration file."""
    try:
        config = load_config_with_defaults(config_file)
        click.echo("Configuration is valid!")
        
        # Display configuration summary
        click.echo(f"\nConfiguration Summary:")
        click.echo(f"  Strategy: {config.strategy.class_name}")
        click.echo(f"  Symbols: {config.data.symbols}")
        click.echo(f"  Date range: {config.data.start_date} to {config.data.end_date}")
        click.echo(f"  Initial capital: ${config.strategy.initial_capital:,.2f}")
        click.echo(f"  Slippage: {config.execution.slippage_bps} bps")
        click.echo(f"  Commission: {config.execution.commission_bps} bps")
        
    except Exception as e:
        click.echo(f"Configuration validation failed: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option('--template', '-t', default='basic', help='Template to use')
@click.option('--output', '-o', default='config.yaml', help='Output file')
def init(template: str, output: str):
    """Initialize a new configuration file."""
    try:
        config = _create_template_config(template)
        config.save(output)
        click.echo(f"Configuration template created: {output}")
        
    except Exception as e:
        click.echo(f"Error creating template: {str(e)}", err=True)
        sys.exit(1)


def _load_strategy(config: Config):
    """Load strategy class from configuration."""
    try:
        # Try to import from module path if specified
        if config.strategy.module_path:
            module = importlib.import_module(config.strategy.module_path)
        else:
            # Try to import from current directory
            module = importlib.import_module('strategy')
        
        # Get strategy class
        strategy_class = getattr(module, config.strategy.class_name)
        
        # Create strategy instance with parameters
        strategy_params = config.strategy.params.copy()
        strategy_params['initial_capital'] = config.strategy.initial_capital
        strategy_params['symbols'] = config.data.symbols
        
        return strategy_class(**strategy_params)
        
    except Exception as e:
        raise ValueError(f"Failed to load strategy '{config.strategy.class_name}': {str(e)}")


def _run_sweep(configs: list, output_dir: str, workers: int, verbose: bool) -> list:
    """Run parameter sweep."""
    results = []
    
    if workers == 1:
        # Sequential execution
        for i, config in enumerate(configs):
            if verbose:
                click.echo(f"Running config {i+1}/{len(configs)}")
            
            try:
                # Create config-specific output directory
                config_output_dir = Path(output_dir) / f"run_{i:04d}"
                config.reporting.output_dir = str(config_output_dir)
                
                # Run backtest
                engine = BacktestEngine(config)
                strategy_instance = _load_strategy(config)
                engine.set_strategy(strategy_instance)
                
                result = engine.run()
                result['config_id'] = i
                result['config'] = config
                results.append(result)
                
                # Save individual results
                config.save(config_output_dir / "config.yaml")
                reporter = Reporter(config, result)
                reporter.generate_reports()
                
            except Exception as e:
                click.echo(f"Error in config {i}: {str(e)}", err=True)
                results.append({
                    'config_id': i,
                    'config': config,
                    'error': str(e)
                })
    
    else:
        # Parallel execution using Ray
        try:
            import ray
            
            if not ray.is_initialized():
                ray.init(num_cpus=workers)
            
            @ray.remote
            def run_single_backtest(config, config_id, output_dir):
                try:
                    config.reporting.output_dir = str(Path(output_dir) / f"run_{config_id:04d}")
                    
                    engine = BacktestEngine(config)
                    strategy_instance = _load_strategy(config)
                    engine.set_strategy(strategy_instance)
                    
                    result = engine.run()
                    result['config_id'] = config_id
                    result['config'] = config
                    
                    # Save results
                    config.save(Path(config.reporting.output_dir) / "config.yaml")
                    reporter = Reporter(config, result)
                    reporter.generate_reports()
                    
                    return result
                    
                except Exception as e:
                    return {
                        'config_id': config_id,
                        'config': config,
                        'error': str(e)
                    }
            
            # Submit all tasks
            futures = []
            for i, config in enumerate(configs):
                future = run_single_backtest.remote(config, i, output_dir)
                futures.append(future)
            
            # Collect results
            results = ray.get(futures)
            
        except ImportError:
            click.echo("Ray not available, falling back to sequential execution", err=True)
            return _run_sweep(configs, output_dir, 1, verbose)
    
    return results


def _generate_sweep_summary(results: list, output_dir: str):
    """Generate summary of sweep results."""
    import pandas as pd
    
    # Filter successful results
    successful_results = [r for r in results if 'error' not in r]
    
    if not successful_results:
        click.echo("No successful runs to summarize")
        return
    
    # Create summary DataFrame
    summary_data = []
    for result in successful_results:
        summary_data.append({
            'config_id': result['config_id'],
            'total_return': result['total_return'],
            'max_drawdown': result['max_drawdown'],
            'total_trades': result['total_trades'],
            'win_rate': result.get('win_rate', 0),
            'sharpe_ratio': result.get('sharpe_ratio', 0),
            **{k: v for k, v in result['config'].strategy.params.items()}
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save summary
    output_path = Path(output_dir)
    df.to_csv(output_path / "sweep_summary.csv", index=False)
    
    # Display top performers
    click.echo("\nTop 5 configurations by total return:")
    top_return = df.nlargest(5, 'total_return')
    for _, row in top_return.iterrows():
        click.echo(f"  Config {row['config_id']}: {row['total_return']:.2%} return, "
                  f"{row['max_drawdown']:.2%} max drawdown")
    
    click.echo("\nTop 5 configurations by Sharpe ratio:")
    top_sharpe = df.nlargest(5, 'sharpe_ratio')
    for _, row in top_sharpe.iterrows():
        click.echo(f"  Config {row['config_id']}: {row['sharpe_ratio']:.2f} Sharpe, "
                  f"{row['total_return']:.2%} return")


def _create_template_config(template: str) -> Config:
    """Create a template configuration."""
    if template == 'basic':
        return Config(
            data=DataConfig(
                source="data/market_data.parquet",
                start_date="2020-01-01",
                end_date="2023-12-31",
                symbols=["AAPL", "GOOGL"],
                bar_size="1min"
            ),
            strategy=StrategyConfig(
                class_name="MyStrategy",
                module_path="strategy",
                params={
                    "lookback": 20,
                    "threshold": 0.02
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
                output_dir="./results",
                save_equity_curve=True,
                save_fills=True,
                save_risk_metrics=True,
                generate_html_report=True
            )
        )
    else:
        raise ValueError(f"Unknown template: {template}")


if __name__ == '__main__':
    main() 