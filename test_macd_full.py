"""测试MACD双向回测和爆仓显示"""
import pandas as pd
import numpy as np
from datetime import datetime

from Data.data_service import get_data_service, DataServiceConfig
from Strategy.templates import MACDStrategy
from backtest.engine import BacktestEngine
from core.config import BacktestConfig

print("=" * 70)
print("MACD双向回测 + 爆仓检测测试")
print("=" * 70)

service = get_data_service(DataServiceConfig(prefer_database=True, auto_init=True))
data = service.get_klines("BTCUSDT", "30min", 500)

print(f"\n数据: {len(data)} 条")
print(f"时间范围: {data.index[0]} 至 {data.index[-1]}")

modes = ["long_only", "short_only", "both"]

print("\n" + "=" * 70)
print("【测试1】正常杠杆 (5x)")
print("=" * 70)

for mode in modes:
    print(f"\n--- 交易模式: {mode} ---")
    
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
    print(f"止损触发: {result.stop_loss_hits}次")
    print(f"止盈触发: {result.take_profit_hits}次")
    print(f"爆仓次数: {result.liquidation_hits}次")
    
    if result.liquidation_hits > 0:
        print("⚠️ 发生爆仓！")

print("\n" + "=" * 70)
print("【测试2】高杠杆爆仓测试 (20x)")
print("=" * 70)

for mode in modes:
    print(f"\n--- 交易模式: {mode} ---")
    
    config = BacktestConfig(
        symbol="BTCUSDT",
        interval="30min",
        initial_capital=10000,
        leverage=20,
    )
    
    strategy = MACDStrategy({"trade_mode": mode})
    engine = BacktestEngine(strategy, config)
    result = engine.run(data)
    
    print(f"最终资金: {result.final_capital:.2f}")
    print(f"总收益率: {result.total_return_pct:.2f}%")
    print(f"最大回撤: {result.max_drawdown_pct:.2f}%")
    print(f"交易次数: {result.total_trades}")
    print(f"爆仓次数: {result.liquidation_hits}次")
    
    if result.liquidation_hits > 0:
        print("⚠️ 发生爆仓！资金归零")
    
    if result.completed_trades:
        for trade in result.completed_trades[-3:]:
            exit_type = getattr(trade, 'exit_type', 'signal')
            if exit_type == 'liquidation':
                print(f"  爆仓交易: 入场价={trade.entry_price:.2f}, 出场价={trade.exit_price:.2f}")

print("\n" + "=" * 70)
print("【测试3】极端杠杆爆仓测试 (50x)")
print("=" * 70)

config = BacktestConfig(
    symbol="BTCUSDT",
    interval="30min",
    initial_capital=10000,
    leverage=50,
)

strategy = MACDStrategy({"trade_mode": "both"})
engine = BacktestEngine(strategy, config)
result = engine.run(data)

print(f"最终资金: {result.final_capital:.2f}")
print(f"总收益率: {result.total_return_pct:.2f}%")
print(f"爆仓次数: {result.liquidation_hits}次")

if result.liquidation_hits > 0:
    print("\n⚠️ 爆仓详情:")
    for trade in result.completed_trades:
        if hasattr(trade, 'exit_type') and trade.exit_type == 'liquidation':
            print(f"  时间: {trade.entry_time} -> {trade.exit_time}")
            print(f"  价格: {trade.entry_price:.2f} -> {trade.exit_price:.2f}")
            print(f"  亏损: {trade.pnl:.2f}")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
