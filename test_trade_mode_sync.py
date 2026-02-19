"""测试trade_mode参数同步"""
import pandas as pd
import numpy as np
from datetime import datetime

from Strategy.templates import get_strategy
from Strategy.parameter_optimizer import ParameterOptimizer, ParameterRange, get_all_optimizable_params
from backtest.engine import BacktestEngine
from core.config import BacktestConfig

print("测试trade_mode参数优化和同步...")

np.random.seed(42)
n = 500
dates = pd.date_range(start='2024-01-01', periods=n, freq='30min')
prices = [100000]
for i in range(n-1):
    change = np.random.uniform(-0.02, 0.02)
    prices.append(prices[-1] * (1 + change))

data = pd.DataFrame({
    'open': prices,
    'high': [p * 1.005 for p in prices],
    'low': [p * 0.995 for p in prices],
    'close': prices,
    'volume': [1000] * n
}, index=dates)

strategy = get_strategy("MACDStrategy")

print("\n可优化参数:")
all_params = get_all_optimizable_params(strategy, include_risk_params=False)
for pr in all_params["strategy"]:
    if pr.values:
        print(f"  {pr.name}: 离散值 {pr.values}")
    else:
        print(f"  {pr.name}: {pr.min_value} - {pr.max_value}")

print("\n运行参数优化（包含trade_mode）...")

optimizer = ParameterOptimizer(
    strategy_class=type(strategy),
    data=data,
    base_config=BacktestConfig(initial_capital=10000),
)

param_ranges = [
    ParameterRange("trade_mode", 0, 0, values=["long_only", "short_only", "both"]),
    ParameterRange("fast_period", 10, 15, 1),
    ParameterRange("slow_period", 25, 30, 1),
]

result = optimizer.random_search(param_ranges, n_iterations=10)

print(f"\n最优参数:")
for key, value in result.best_params.items():
    print(f"  {key}: {value}")

print(f"\n最优得分: {result.best_score:.4f}")

if "trade_mode" in result.best_params:
    print(f"\n✅ trade_mode已包含在最优参数中: {result.best_params['trade_mode']}")
else:
    print(f"\n❌ trade_mode未包含在最优参数中")
