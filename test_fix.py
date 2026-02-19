"""测试UI导入和参数优化修复"""
print("测试UI导入...")
from UI.main_ui import TradingUI
print("✅ UI导入成功")

print("\n测试参数优化零交易惩罚...")
from Strategy.parameter_optimizer import ParameterOptimizer, ParameterRange
from Strategy.templates import MACDStrategy
from Data.data_service import get_data_service, DataServiceConfig
from core.config import BacktestConfig

service = get_data_service(DataServiceConfig(prefer_database=True, auto_init=True))
data = service.get_klines("BTCUSDT", "30min", 200)

config = BacktestConfig(initial_capital=10000, leverage=5)
optimizer = ParameterOptimizer(
    strategy_class=MACDStrategy,
    data=data,
    base_config=config,
)

param_ranges = [
    ParameterRange("fast_period", 10, 15, 5),
    ParameterRange("slow_period", 24, 28, 2),
]

result = optimizer.random_search(param_ranges=param_ranges, n_iterations=10)
print(f"最优参数: {result.best_params}")
print(f"最优得分: {result.best_score:.4f}")
print(f"交易次数: {result.to_dict()['top_results'][0].get('total_trades', 'N/A')}")

if result.best_score == float('-inf'):
    print("⚠️ 所有参数组合都没有交易")
else:
    print("✅ 参数优化正常")
