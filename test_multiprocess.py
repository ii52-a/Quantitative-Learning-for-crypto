"""测试多进程优化器"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import multiprocessing as mp
import numpy as np
import pandas as pd

from Strategy.templates import get_strategy
from Strategy.parameter_optimizer import ParameterOptimizer, ParameterRange
from core.config import BacktestConfig

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
config = BacktestConfig(initial_capital=10000)

param_ranges = [
    ParameterRange("fast_period", 10, 20, 1),
    ParameterRange("slow_period", 25, 35, 1),
]

print("测试单线程优化...")
optimizer1 = ParameterOptimizer(
    strategy_class=type(strategy),
    data=data,
    base_config=config,
)
start = time.time()
result1 = optimizer1.random_search(param_ranges, n_iterations=50)
single_time = time.time() - start
print(f"  耗时: {single_time:.2f}秒")

from Strategy.multiprocess_optimizer import MultiProcessOptimizer

print(f"\n测试多进程优化 ({max(1, mp.cpu_count() - 1)}进程)...")
optimizer2 = MultiProcessOptimizer(
    strategy_class=type(strategy),
    data=data,
    base_config=config,
)
start = time.time()
result2 = optimizer2.parallel_random_search(param_ranges, n_iterations=50)
multi_time = time.time() - start
print(f"  耗时: {multi_time:.2f}秒")

print(f"\n加速比: {single_time/multi_time:.2f}x")
