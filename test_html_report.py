"""测试HTML报告生成"""
import pandas as pd
import numpy as np
from datetime import datetime

from Strategy.templates import get_strategy
from backtest.engine import BacktestEngine
from backtest.visualization import BacktestVisualizer
from core.config import BacktestConfig

print("测试HTML报告生成...")

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

print(f"\n回测结果:")
print(f"  总收益率: {result.total_return_pct:.2f}%")
print(f"  交易次数: {result.total_trades}")
print(f"  盈亏比: {result.profit_factor:.2f}")

print(f"\n回测参数:")
print(f"  交易对: {result.symbol}")
print(f"  周期: {result.interval}")
print(f"  策略: {result.strategy_name}")
print(f"  策略参数: {result.strategy_params}")

visualizer = BacktestVisualizer()
html = visualizer.generate_html_report(data, result, "30min", "测试报告")

with open("test_report.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✅ HTML报告已生成: test_report.html")
print(f"   包含: 回测参数 + MACD指标图表")
