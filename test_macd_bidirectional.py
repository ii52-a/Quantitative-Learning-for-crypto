"""测试MACD双向策略"""
import pandas as pd
from datetime import datetime

from Data.data_service import get_data_service, DataServiceConfig
from Strategy.templates import MACDStrategy
from backtest.engine import BacktestEngine
from core.config import BacktestConfig

print("=" * 60)
print("MACD双向策略测试")
print("=" * 60)

service = get_data_service(DataServiceConfig(prefer_database=True, auto_init=True))
data = service.get_klines("BTCUSDT", "30min", 1000)

if data.empty:
    print("无法获取数据")
    exit()

print(f"数据: {len(data)} 条")
print(f"时间范围: {data.index[0]} 至 {data.index[-1]}")

modes = ["long_only", "short_only", "both"]

for mode in modes:
    print(f"\n{'='*60}")
    print(f"交易模式: {mode}")
    print("=" * 60)
    
    config = BacktestConfig(
        symbol="BTCUSDT",
        interval="30min",
        initial_capital=10000,
        leverage=5,
    )
    
    strategy = MACDStrategy({"trade_mode": mode})
    engine = BacktestEngine(strategy, config)
    result = engine.run(data)
    
    print(f"总收益率: {result.total_return_pct:.2f}%")
    print(f"最大回撤: {result.max_drawdown_pct:.2f}%")
    print(f"夏普比率: {result.sharpe_ratio:.2f}")
    print(f"交易次数: {result.total_trades}")
    print(f"胜率: {result.win_rate:.2f}%")
