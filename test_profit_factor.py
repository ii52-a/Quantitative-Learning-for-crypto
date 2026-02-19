"""测试盈亏比计算"""
import pandas as pd
import numpy as np
from datetime import datetime

from Strategy.templates import get_strategy
from backtest.engine import BacktestEngine
from core.config import BacktestConfig

print("测试盈亏比计算...")

np.random.seed(42)
n = 200
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

config = BacktestConfig(
    symbol="BTCUSDT",
    interval="30min",
    initial_capital=10000,
    leverage=5,
)

strategy = get_strategy("MACDStrategy", {"fast_period": 12, "slow_period": 26})
engine = BacktestEngine(strategy, config)
result = engine.run(data)

print(f"\n交易统计:")
print(f"  总交易次数: {result.total_trades}")
print(f"  盈利次数: {result.winning_trades}")
print(f"  亏损次数: {result.losing_trades}")

print(f"\n盈亏分析:")
print(f"  平均盈利: {result.avg_win:.2f}")
print(f"  平均亏损: {result.avg_loss:.2f}")
print(f"  盈亏比: {result.profit_factor:.2f}")

if result.avg_loss > 0:
    expected = result.avg_win / result.avg_loss
    print(f"\n验证: 平均盈利/平均亏损 = {result.avg_win:.2f}/{result.avg_loss:.2f} = {expected:.2f}")
    print(f"盈亏比 > 1 表示盈利大于亏损，是好情况")
    print(f"盈亏比 < 1 表示亏损大于盈利，是坏情况")
    
if result.profit_factor > 1:
    print(f"\n当前盈亏比 {result.profit_factor:.2f} > 1，表示平均盈利大于平均亏损")
elif result.profit_factor < 1:
    print(f"\n当前盈亏比 {result.profit_factor:.2f} < 1，表示平均亏损大于平均盈利")
