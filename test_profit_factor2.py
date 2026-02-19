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
print(f"  胜率: {result.win_rate:.2f}%")

print(f"\n盈亏比计算:")
print(f"  胜率 = {result.win_rate:.2f}% = {result.win_rate/100:.2f}")
print(f"  亏损率 = {100 - result.win_rate:.2f}% = {(100 - result.win_rate)/100:.2f}")
print(f"  盈亏比 = 盈利次数/亏损次数 = {result.winning_trades}/{result.losing_trades} = {result.profit_factor:.2f}")

expected = result.winning_trades / result.losing_trades if result.losing_trades > 0 else float('inf')
print(f"\n验证: {result.winning_trades}/{result.losing_trades} = {expected:.2f}")

if result.profit_factor < 1:
    print(f"\n盈亏比 {result.profit_factor:.2f} < 1，亏损次数多于盈利次数")
elif result.profit_factor > 1:
    print(f"\n盈亏比 {result.profit_factor:.2f} > 1，盈利次数多于亏损次数")
else:
    print(f"\n盈亏比 = 1，盈利亏损次数相等")
