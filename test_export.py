"""测试导出数据包含回测参数"""
import pandas as pd
import numpy as np
from datetime import datetime

from Strategy.templates import get_strategy
from backtest.engine import BacktestEngine
from core.config import BacktestConfig

print("测试导出数据...")

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
    stop_loss_pct=5.0,
    take_profit_pct=10.0,
    position_size=0.2,
)

strategy = get_strategy("MACDStrategy", {"fast_period": 12, "slow_period": 26})
engine = BacktestEngine(strategy, config)
result = engine.run(data)

print("\n导出数据字段:")
export_data = result.to_dict()
for key, value in export_data.items():
    print(f"  {key}: {value}")

print("\n新增回测参数字段:")
print(f"  symbol: {result.symbol}")
print(f"  interval: {result.interval}")
print(f"  strategy_name: {result.strategy_name}")
print(f"  strategy_params: {result.strategy_params}")
print(f"  commission_rate: {result.commission_rate}")
print(f"  position_size: {result.position_size}")
print(f"  stop_loss_pct: {result.stop_loss_pct}")
print(f"  take_profit_pct: {result.take_profit_pct}")

print("\n✅ 测试完成!")
