"""
回测模块

提供回测引擎、绩效指标和可视化
"""

from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import PerformanceMetrics
from backtest.report import BacktestReport

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "PerformanceMetrics",
    "BacktestReport",
]
