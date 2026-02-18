"""
核心模块

提供系统基础组件
"""

from core.config import Config, BacktestConfig, TradingConfig
from core.exceptions import (
    QuantitativeError,
    DataError,
    StrategyError,
    BacktestError,
    TradingError,
)
from core.constants import (
    Interval,
    OrderSide,
    OrderType,
    PositionSide,
    SignalType,
)

__all__ = [
    "Config",
    "BacktestConfig",
    "TradingConfig",
    "QuantitativeError",
    "DataError",
    "StrategyError",
    "BacktestError",
    "TradingError",
    "Interval",
    "OrderSide",
    "OrderType",
    "PositionSide",
    "SignalType",
]
