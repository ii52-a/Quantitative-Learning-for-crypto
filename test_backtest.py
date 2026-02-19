"""
测试回测功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 50)
print("回测功能测试")
print("=" * 50)

print("\n1. 测试数据服务...")
try:
    from Data.data_service import get_data_service, DataServiceConfig
    
    service = get_data_service(DataServiceConfig(prefer_database=True, auto_init=True))
    
    print("   获取BTCUSDT 30min数据...")
    data = service.get_klines("BTCUSDT", "30min", 100)
    
    if not data.empty:
        print(f"   获取到 {len(data)} 条数据")
        print(f"   时间范围: {data.index[0]} 至 {data.index[-1]}")
        print(f"   最新价格: ${data['close'].iloc[-1]:,.2f}")
    else:
        print("   数据为空，尝试生成测试数据...")
        dates = pd.date_range(start='2024-01-01', periods=100, freq='30min')
        data = pd.DataFrame({
            'open': np.random.uniform(49000, 51000, 100),
            'high': np.random.uniform(50000, 52000, 100),
            'low': np.random.uniform(48000, 50000, 100),
            'close': np.random.uniform(49000, 51000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }, index=dates)
        print(f"   生成 {len(data)} 条测试数据")
    
except Exception as e:
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()
    
    print("   使用测试数据...")
    dates = pd.date_range(start='2024-01-01', periods=100, freq='30min')
    data = pd.DataFrame({
        'open': np.random.uniform(49000, 51000, 100),
        'high': np.random.uniform(50000, 52000, 100),
        'low': np.random.uniform(48000, 50000, 100),
        'close': np.random.uniform(49000, 51000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    }, index=dates)

print("\n2. 测试策略加载...")
try:
    from Strategy.templates import get_strategy, list_strategies
    
    strategies = list_strategies()
    print(f"   可用策略: {[s['name'] for s in strategies]}")
    
    strategy = get_strategy("MACDStrategy")
    print(f"   策略: {strategy.display_name}")
    
except Exception as e:
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

print("\n3. 测试回测引擎...")
try:
    from backtest.engine import BacktestEngine
    from core.config import BacktestConfig
    
    config = BacktestConfig(
        symbol="BTCUSDT",
        interval="30min",
        initial_capital=10000,
        leverage=5,
    )
    
    strategy = get_strategy("MACDStrategy")
    engine = BacktestEngine(strategy, config)
    
    print("   运行回测...")
    result = engine.run(data)
    
    print(f"   回测完成!")
    print(f"   总收益率: {result.total_return_pct:.2f}%")
    print(f"   最大回撤: {result.max_drawdown_pct:.2f}%")
    print(f"   夏普比率: {result.sharpe_ratio:.2f}")
    print(f"   交易次数: {result.total_trades}")
    print(f"   胜率: {result.win_rate:.2f}%")
    
except Exception as e:
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

print("\n4. 测试参数优化...")
try:
    from Strategy.parameter_optimizer import ParameterOptimizer, ParameterRange
    
    config = BacktestConfig(
        symbol="BTCUSDT",
        interval="30min",
        initial_capital=10000,
    )
    
    from Strategy.templates import MACDStrategy
    optimizer = ParameterOptimizer(
        strategy_class=MACDStrategy,
        data=data,
        base_config=config,
    )
    
    param_ranges = [
        ParameterRange("fast_period", 10, 14, 2),
        ParameterRange("slow_period", 24, 28, 2),
    ]
    
    print("   运行随机搜索优化 (5次迭代)...")
    result = optimizer.random_search(
        param_ranges=param_ranges,
        n_iterations=5,
    )
    
    print(f"   优化完成!")
    print(f"   最优参数: {result.best_params}")
    print(f"   最优得分: {result.best_score:.4f}")
    
except Exception as e:
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("测试完成!")
print("=" * 50)
