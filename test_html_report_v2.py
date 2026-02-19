"""测试HTML报告交易记录更新"""
import pandas as pd
import numpy as np
from datetime import datetime

from Strategy.templates import get_strategy
from backtest.engine import BacktestEngine
from backtest.visualization import BacktestVisualizer
from core.config import BacktestConfig

print("测试HTML报告交易记录更新...")

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

print(f"\n交易记录详情:")
for i, t in enumerate(result.completed_trades[:3]):
    print(f"  交易{i+1}:")
    print(f"    方向: {t.side}")
    print(f"    开仓价: {t.entry_price:.2f}")
    print(f"    平仓价: {t.exit_price:.2f}")
    print(f"    数量: {t.quantity:.4f}")
    print(f"    仓位金额: {t.position_value:.2f}")
    print(f"    保证金: {t.margin_used:.2f}")
    print(f"    手续费: {t.commission:.2f}")
    print(f"    盈亏: {t.pnl:.2f}")

visualizer = BacktestVisualizer()
html = visualizer.generate_html_report(data, result, "30min", "测试报告")

with open("test_report_v2.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✅ HTML报告已生成: test_report_v2.html")
print(f"   新增字段: 方向、仓位金额、保证金、手续费")
