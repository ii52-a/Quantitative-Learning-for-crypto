"""测试复合参数优化"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from Strategy.composite_param_optimizer import CompositeParameterOptimizer, run_composite_optimization
from Strategy.parameter_optimizer import ParameterRange
from Strategy.templates import get_strategy
from core.config import BacktestConfig

print("=" * 60)
print("复合参数优化测试")
print("=" * 60)

np.random.seed(42)
n = 5000
dates = pd.date_range(start='2024-01-01', periods=n, freq='30min')

base_price = 100000
prices = [base_price]
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

print(f"数据量: {len(data)} 条")
print(f"时间跨度: {(data.index[-1] - data.index[0]).total_seconds() / 3600:.1f} 小时")

config = BacktestConfig(initial_capital=10000, leverage=5)
strategy = get_strategy("MACDStrategy")

param_ranges = [
    ParameterRange("fast_period", 10, 15, 1),
    ParameterRange("slow_period", 24, 28, 1),
]

print("\n运行复合优化 (每个算法10次迭代)...")

result = run_composite_optimization(
    strategy_class=type(strategy),
    data=data,
    base_config=config,
    param_ranges=param_ranges,
    iterations_per_algorithm=10,
)

print(f"\n最优算法: {result.best_algorithm}")
print(f"最优得分: {result.best_score:.4f}")
print(f"最优参数: {result.best_params}")
print(f"总耗时: {result.execution_time:.1f}秒")

print(f"\n算法排名:")
for i, (name, score) in enumerate(result.get_ranking(), 1):
    print(f"  {i}. {name}: {score:.4f}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
