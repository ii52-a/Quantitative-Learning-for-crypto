"""测试MACD策略修复"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from Data.data_service import get_data_service, DataServiceConfig
from Strategy.templates import MACDStrategy
from backtest.engine import BacktestEngine
from core.config import BacktestConfig

print("=" * 50)
print("MACD策略测试")
print("=" * 50)

service = get_data_service(DataServiceConfig(prefer_database=True, auto_init=True))
data = service.get_klines("BTCUSDT", "30min", 1000)

if data.empty:
    print("无法获取数据")
    exit()

print(f"数据: {len(data)} 条")
print(f"时间范围: {data.index[0]} 至 {data.index[-1]}")

config = BacktestConfig(
    symbol="BTCUSDT",
    interval="30min",
    initial_capital=10000,
    leverage=5,
)

strategy = MACDStrategy()
engine = BacktestEngine(strategy, config)

print("\n运行回测...")
result = engine.run(data)

print(f"\n回测结果:")
print(f"  总收益率: {result.total_return_pct:.2f}%")
print(f"  最大回撤: {result.max_drawdown_pct:.2f}%")
print(f"  夏普比率: {result.sharpe_ratio:.2f}")
print(f"  交易次数: {result.total_trades}")
print(f"  胜率: {result.win_rate:.2f}%")

print(f"\n交易记录:")
for i, trade in enumerate(result.completed_trades[:10]):
    entry_time = trade.entry_time.strftime('%m-%d %H:%M') if trade.entry_time else 'N/A'
    exit_time = trade.exit_time.strftime('%m-%d %H:%M') if trade.exit_time else 'N/A'
    pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price * 100 if trade.entry_price > 0 else 0
    print(f"  {i+1}. {entry_time} @ {trade.entry_price:.2f} -> {exit_time} @ {trade.exit_price:.2f} | 收益: {pnl_pct:.2f}%")

if len(result.completed_trades) > 10:
    print(f"  ... 共 {len(result.completed_trades)} 笔交易")
