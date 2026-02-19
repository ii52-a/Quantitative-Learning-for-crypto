"""测试复合策略和盈亏比修复"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from Strategy.composite import CompositeStrategy, create_composite_strategy
from Strategy.templates import get_strategy
from backtest.engine import BacktestEngine
from core.config import BacktestConfig

print("=" * 60)
print("测试1: 盈亏比修复验证")
print("=" * 60)

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

from Strategy.base import BaseStrategy, StrategyResult, Signal
from core.constants import SignalType

class TestStrategy(BaseStrategy):
    name = "TestStrategy"
    display_name = "测试策略"
    parameters = []
    
    def __init__(self, params=None):
        super().__init__(params)
        self._count = 0
    
    def initialize(self, context):
        self._initialized = True
    
    def on_bar(self, bar, context):
        self._count += 1
        if self._count % 20 == 0:
            if np.random.random() > 0.4:
                return StrategyResult(signal=Signal(type=SignalType.OPEN_LONG, price=bar.close))
            else:
                return StrategyResult(signal=Signal(type=SignalType.CLOSE_LONG, price=bar.close))
        return StrategyResult()

strategy = TestStrategy()
config = BacktestConfig(initial_capital=10000, leverage=5)
engine = BacktestEngine(strategy, config)
result = engine.run(data)

print(f"总交易次数: {result.total_trades}")
print(f"盈利次数: {result.winning_trades}")
print(f"亏损次数: {result.losing_trades}")
print(f"平均盈利: {result.avg_win:.2f}")
print(f"平均亏损: {result.avg_loss:.2f}")
print(f"盈亏比: {result.profit_factor:.2f}")
expected_ratio = result.avg_win / result.avg_loss if result.avg_loss > 0 else float('inf')
print(f"验证: 平均盈利/平均亏损 = {result.avg_win:.2f}/{result.avg_loss:.2f} = {expected_ratio:.2f}")

print("\n" + "=" * 60)
print("测试2: 复合策略")
print("=" * 60)

macd1 = get_strategy("MACDStrategy", {"fast_period": 12, "slow_period": 26})
macd2 = get_strategy("MACDStrategy", {"fast_period": 8, "slow_period": 20})

composite = create_composite_strategy(
    strategies=[(macd1, 1.0), (macd2, 0.8)],
    combine_method="weighted",
    min_vote_ratio=0.5,
)

print(f"复合策略包含 {len(composite._strategies)} 个子策略")
for sw in composite._strategies:
    print(f"  - {sw.strategy.name}: 权重={sw.weight}")

engine2 = BacktestEngine(composite, config)
result2 = engine2.run(data)

print(f"\n复合策略回测结果:")
print(f"  总收益率: {result2.total_return_pct:.2f}%")
print(f"  最大回撤: {result2.max_drawdown_pct:.2f}%")
print(f"  夏普比率: {result2.sharpe_ratio:.2f}")
print(f"  交易次数: {result2.total_trades}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
