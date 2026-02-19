"""测试爆仓检测功能"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest.engine import BacktestEngine
from core.config import BacktestConfig
from Strategy.base import BaseStrategy, StrategyResult, Signal
from core.constants import SignalType

class TestStrategy(BaseStrategy):
    """测试策略：第一根K线开多"""
    name = "TestStrategy"
    display_name = "测试策略"
    parameters = []
    
    def __init__(self, params=None):
        super().__init__(params)
        self._opened = False
    
    def initialize(self, context):
        self._opened = False
        self._initialized = True
    
    def on_bar(self, bar, context):
        if not self._opened:
            self._opened = True
            return StrategyResult(
                signal=Signal(type=SignalType.OPEN_LONG, price=bar.close, reason="测试开多")
            )
        return StrategyResult()

print("=" * 60)
print("爆仓检测测试")
print("=" * 60)

np.random.seed(42)
n = 100
dates = pd.date_range(start='2024-01-01', periods=n, freq='30min')

base_price = 100000
prices = [base_price]
for i in range(n-1):
    if i < 30:
        change = np.random.uniform(-0.02, 0.01)
    else:
        change = np.random.uniform(-0.05, 0.02)
    prices.append(prices[-1] * (1 + change))

data = pd.DataFrame({
    'open': prices,
    'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
    'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
    'close': prices,
    'volume': [1000] * n
}, index=dates)

print(f"数据: {len(data)} 条")
print(f"价格范围: {data['low'].min():.2f} ~ {data['high'].max():.2f}")

print("\n测试不同杠杆下的爆仓情况:")
print("-" * 60)

for leverage in [5, 10, 20, 50]:
    config = BacktestConfig(
        initial_capital=10000,
        leverage=leverage,
    )
    
    strategy = TestStrategy()
    engine = BacktestEngine(strategy, config)
    result = engine.run(data)
    
    print(f"\n杠杆: {leverage}x")
    print(f"  最终资金: {result.final_capital:.2f}")
    print(f"  总收益率: {result.total_return_pct:.2f}%")
    print(f"  最大回撤: {result.max_drawdown_pct:.2f}%")
    print(f"  爆仓次数: {result.liquidation_hits}")
    print(f"  止损次数: {result.stop_loss_hits}")
    
    if result.liquidation_hits > 0:
        print(f"  ⚠️ 发生爆仓！")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
