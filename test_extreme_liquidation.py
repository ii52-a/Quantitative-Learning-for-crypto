"""极端爆仓测试"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest.engine import BacktestEngine
from core.config import BacktestConfig
from Strategy.base import BaseStrategy, StrategyResult, Signal
from core.constants import SignalType

class SimpleLongStrategy(BaseStrategy):
    """简单做多策略"""
    name = "SimpleLong"
    display_name = "简单做多"
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
                signal=Signal(type=SignalType.OPEN_LONG, price=bar.close, reason="开多")
            )
        return StrategyResult()

print("=" * 70)
print("极端爆仓测试")
print("=" * 70)

np.random.seed(42)
n = 50
dates = pd.date_range(start='2024-01-01', periods=n, freq='30min')

base_price = 100000
prices = [base_price]
for i in range(n-1):
    if i < 10:
        change = np.random.uniform(-0.005, 0.005)
    else:
        change = np.random.uniform(-0.05, -0.02)
    prices.append(prices[-1] * (1 + change))

data = pd.DataFrame({
    'open': prices,
    'high': [p * (1 + np.random.uniform(0, 0.005)) for p in prices],
    'low': [p * (1 - np.random.uniform(0, 0.005)) for p in prices],
    'close': prices,
    'volume': [1000] * n
}, index=dates)

print(f"数据: {len(data)} 条")
print(f"价格范围: {min(prices):.2f} ~ {max(prices):.2f}")
print(f"最大跌幅: {(min(prices)/max(prices)-1)*100:.2f}%")

print("\n" + "=" * 70)
print("不同杠杆下的爆仓测试")
print("=" * 70)

for leverage in [5, 10, 20, 50]:
    print(f"\n--- 杠杆: {leverage}x ---")
    print(f"爆仓阈值: 价格下跌 {100/leverage:.1f}%")
    
    config = BacktestConfig(
        initial_capital=10000,
        leverage=leverage,
    )
    
    strategy = SimpleLongStrategy()
    engine = BacktestEngine(strategy, config)
    result = engine.run(data)
    
    print(f"最终资金: {result.final_capital:.2f}")
    print(f"总收益率: {result.total_return_pct:.2f}%")
    print(f"最大回撤: {result.max_drawdown_pct:.2f}%")
    print(f"爆仓次数: {result.liquidation_hits}次")
    
    if result.liquidation_hits > 0:
        print("⚠️ 发生爆仓！")
        for trade in result.completed_trades:
            if hasattr(trade, 'exit_type') and trade.exit_type == 'liquidation':
                print(f"  爆仓价格: {trade.exit_price:.2f}")
                print(f"  入场价格: {trade.entry_price:.2f}")
                print(f"  价格变动: {(trade.exit_price/trade.entry_price-1)*100:.2f}%")
    elif result.final_capital <= 0:
        print("❌ 资金归零但未检测到爆仓！")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
