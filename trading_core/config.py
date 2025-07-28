"""
Configuration management for the backtesting framework.

This module handles loading and validation of configuration files.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    """Configuration for data sources."""
    source: str = Field(..., description="Data source path or URL")
    start_date: str = Field(..., description="Start date for backtest")
    end_date: str = Field(..., description="End date for backtest")
    symbols: List[str] = Field(default_factory=list, description="Symbols to trade")
    bar_size: Optional[str] = Field(None, description="Bar size (e.g., '1min', '5min', '1D')")
    
    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use ISO format (YYYY-MM-DD)")


class StrategyConfig(BaseModel):
    """Configuration for strategy parameters."""
    class_name: str = Field(..., description="Strategy class name")
    module_path: Optional[str] = Field(None, description="Module path for strategy")
    params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    initial_capital: float = Field(100000.0, description="Initial capital")
    
    @validator('class_name')
    def validate_class_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Class name cannot be empty")
        return v.strip()


class ExecutionConfig(BaseModel):
    """Configuration for execution parameters."""
    slippage_bps: float = Field(1.0, description="Slippage in basis points")
    commission_bps: float = Field(0.5, description="Commission in basis points")
    min_tick_size: float = Field(0.01, description="Minimum tick size")
    max_spread_bps: Optional[float] = Field(None, description="Maximum spread in basis points")
    fill_delay_ms: int = Field(0, description="Fill delay in milliseconds")
    
    @validator('slippage_bps', 'commission_bps', 'min_tick_size')
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError(f"Value must be positive: {v}")
        return v


class RiskConfig(BaseModel):
    """Configuration for risk management."""
    max_position_size: int = Field(100, description="Maximum position size")
    max_drawdown: float = Field(0.15, description="Maximum drawdown (0.15 = 15%)")
    stop_loss_bps: Optional[float] = Field(None, description="Stop loss in basis points")
    position_limit_pct: float = Field(0.1, description="Position limit as % of capital")
    
    @validator('max_drawdown', 'position_limit_pct')
    def validate_percentage(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"Percentage must be between 0 and 1: {v}")
        return v


class ReportingConfig(BaseModel):
    """Configuration for reporting."""
    output_dir: str = Field("./results", description="Output directory for results")
    save_equity_curve: bool = Field(True, description="Save equity curve data")
    save_fills: bool = Field(True, description="Save fill data")
    save_risk_metrics: bool = Field(True, description="Save risk metrics")
    generate_html_report: bool = Field(True, description="Generate HTML report")
    plot_style: str = Field("plotly", description="Plotting style (plotly, matplotlib)")


class Config(BaseModel):
    """Main configuration class."""
    data: DataConfig
    strategy: StrategyConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        extra = "forbid"  # Don't allow extra fields
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.dict()
    
    def save(self, filepath: Union[str, Path]):
        """Save config to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> 'Config':
        """Load config from YAML file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(**data)


def create_default_config() -> Config:
    """Create a default configuration."""
    return Config(
        data=DataConfig(
            source="data/market_data.parquet",
            start_date="2020-01-01",
            end_date="2023-12-31",
            symbols=["AAPL", "GOOGL"],
            bar_size="1min"
        ),
        strategy=StrategyConfig(
            class_name="BollingerStrategy",
            module_path="strategies.bollinger",
            params={
                "lookback": 20,
                "z_score": 2.0
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


def validate_config(config: Config) -> List[str]:
    """
    Validate configuration and return list of errors.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    try:
        # Validate dates
        start_date = datetime.fromisoformat(config.data.start_date)
        end_date = datetime.fromisoformat(config.data.end_date)
        
        if start_date >= end_date:
            errors.append("Start date must be before end date")
        
        # Validate symbols
        if not config.data.symbols:
            errors.append("At least one symbol must be specified")
        
        # Validate capital
        if config.strategy.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        
        # Validate execution parameters
        if config.execution.slippage_bps < 0:
            errors.append("Slippage must be non-negative")
        
        if config.execution.commission_bps < 0:
            errors.append("Commission must be non-negative")
        
        # Validate risk parameters
        if config.risk.max_drawdown <= 0 or config.risk.max_drawdown >= 1:
            errors.append("Max drawdown must be between 0 and 1")
        
        if config.risk.position_limit_pct <= 0 or config.risk.position_limit_pct >= 1:
            errors.append("Position limit percentage must be between 0 and 1")
        
    except Exception as e:
        errors.append(f"Configuration validation error: {str(e)}")
    
    return errors


def load_config_with_defaults(filepath: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file with defaults.
    
    Args:
        filepath: Path to config file (optional)
        
    Returns:
        Configuration object
    """
    if filepath is None:
        return create_default_config()
    
    try:
        config = Config.from_file(filepath)
        errors = validate_config(config)
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        return config
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {str(e)}")


# Utility functions for parameter sweeps
def create_parameter_grid(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Create a grid of parameter combinations.
    
    Args:
        params: Dictionary of parameter names to lists of values
        
    Returns:
        List of parameter combinations
    """
    import itertools
    
    keys = list(params.keys())
    values = list(params.values())
    
    combinations = []
    for combination in itertools.product(*values):
        combinations.append(dict(zip(keys, combination)))
    
    return combinations


def create_sweep_configs(base_config: Config, param_grid: Dict[str, List[Any]]) -> List[Config]:
    """
    Create multiple configurations for parameter sweeping.
    
    Args:
        base_config: Base configuration
        param_grid: Dictionary of parameter names to lists of values
        
    Returns:
        List of configurations
    """
    configs = []
    combinations = create_parameter_grid(param_grid)
    
    for i, params in enumerate(combinations):
        # Create a copy of the base config
        config_dict = base_config.to_dict()
        
        # Update strategy parameters
        for key, value in params.items():
            if key in config_dict['strategy']['params']:
                config_dict['strategy']['params'][key] = value
            else:
                # Add to metadata for tracking
                config_dict['metadata'][f'sweep_param_{key}'] = value
        
        # Add sweep metadata
        config_dict['metadata']['sweep_id'] = i
        config_dict['metadata']['sweep_total'] = len(combinations)
        
        configs.append(Config.from_dict(config_dict))
    
    return configs 