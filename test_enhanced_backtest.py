"""测试强化回测模块"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from Strategy.templates import get_strategy
from backtest.enhanced_backtester import (
    EnhancedBacktester,
    MarketScenario,
    ScenarioConfig,
    run_enhanced_backtest,
)
from core.config import BacktestConfig

print("="*60)
print("强化回测模块测试")
print("="*60)

np.random.seed(42)
n = 500
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
backtester = EnhancedBacktester(type(strategy), config)

print("\n[1] 市场情景测试")
print("-"*40)

scenarios_to_test = [
    (MarketScenario.NORMAL, "正常市场"),
    (MarketScenario.BULL_MARKET, "牛市"),
    (MarketScenario.BEAR_MARKET, "熊市"),
    (MarketScenario.HIGH_VOLATILITY, "高波动"),
    (MarketScenario.FLASH_CRASH, "闪崩"),
]

for scenario_type, name in scenarios_to_test:
    if scenario_type == MarketScenario.BULL_MARKET:
        scenario = ScenarioConfig(scenario_type, trend_bias=0.3, volatility_multiplier=1.2)
    elif scenario_type == MarketScenario.BEAR_MARKET:
        scenario = ScenarioConfig(scenario_type, trend_bias=0.3, volatility_multiplier=1.5)
    elif scenario_type == MarketScenario.HIGH_VOLATILITY:
        scenario = ScenarioConfig(scenario_type, volatility_multiplier=3.0)
    elif scenario_type == MarketScenario.FLASH_CRASH:
        scenario = ScenarioConfig(scenario_type, trend_bias=0.3, volatility_multiplier=2.0)
    else:
        scenario = ScenarioConfig(scenario_type)
    
    result = backtester.run_scenario_test(data, scenario)
    print(f"\n{name}:")
    print(f"  收益率: {result.total_return_pct:.2f}%")
    print(f"  最大回撤: {result.max_drawdown_pct:.2f}%")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  胜率: {result.win_rate:.1f}%")
    print(f"  交易次数: {result.total_trades}")

print("\n[2] 蒙特卡洛模拟 (30次)")
print("-"*40)

mc_result = backtester.run_monte_carlo(data, n_simulations=30, random_seed=42)
print(f"模拟次数: {mc_result.simulations}")
print(f"平均收益: {mc_result.mean_return:.2f}%")
print(f"收益标准差: {mc_result.std_return:.2f}%")
print(f"VaR(95%): {mc_result.var_95:.2f}%")
print(f"盈利概率: {mc_result.profit_probability:.1f}%")
print(f"回撤分布: {mc_result.max_drawdown_distribution}")

print("\n[3] 滚动窗口测试")
print("-"*40)

wf_result = backtester.run_walk_forward(data, train_window=200, test_window=50, step=50)
print(f"测试窗口数: {len(wf_result.window_results)}")
print(f"平均收益: {wf_result.avg_return:.2f}%")
print(f"平均夏普: {wf_result.avg_sharpe:.2f}")
print(f"一致性得分: {wf_result.consistency_score:.1f}%")

if wf_result.window_results:
    print("\n窗口详情:")
    for wr in wf_result.window_results[:3]:
        print(f"  窗口{wr['window']}: 收益={wr['return']:.2f}%, 夏普={wr['sharpe']:.2f}")

print("\n" + "="*60)
print("✅ 强化回测模块测试完成")
print("="*60)
