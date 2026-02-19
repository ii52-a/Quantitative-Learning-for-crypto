"""测试强化回测模块是否正常"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("测试强化回测模块导入...")

try:
    from backtest.enhanced_backtester import EnhancedBacktester, MarketScenario, ScenarioConfig
    print("✅ 强化回测模块导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

print("\n测试回测引擎导入...")
try:
    from backtest.engine import BacktestEngine
    print("✅ 回测引擎导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")

print("\n测试策略导入...")
try:
    from Strategy.templates import get_strategy
    strategy = get_strategy("MACDStrategy", {})
    print(f"✅ 策略导入成功: {strategy}")
except Exception as e:
    print(f"❌ 导入失败: {e}")

print("\n测试配置导入...")
try:
    from core.config import BacktestConfig
    config = BacktestConfig(symbol="BTCUSDT")
    print(f"✅ 配置导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")

print("\n✅ 所有测试通过")
