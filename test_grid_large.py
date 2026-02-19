"""测试大组合数网格搜索"""
from Strategy.parameter_optimizer import ParameterOptimizer, ParameterRange
from Strategy.templates import get_strategy
from Data.data_service import get_data_service, DataServiceConfig
from core.config import BacktestConfig

print("测试大组合数网格搜索...")

service = get_data_service(DataServiceConfig(prefer_database=True, auto_init=True))
data = service.get_klines("BTCUSDT", "30min", 100)

config = BacktestConfig(initial_capital=10000, leverage=5)
strategy = get_strategy("MACDStrategy")

optimizer = ParameterOptimizer(
    strategy_class=type(strategy),
    data=data,
    base_config=config,
)

param_ranges = [
    ParameterRange("fast_period", 5, 30, 1),
    ParameterRange("slow_period", 20, 50, 1),
]

total_combinations = 1
for pr in param_ranges:
    values = pr.get_values()
    total_combinations *= len(values)

print(f"总组合数: {total_combinations}")
print(f"将限制为前50个组合")

print("\n开始网格搜索...")
result = optimizer.grid_search(
    param_ranges=param_ranges,
    max_iterations=50,
)

print(f"\n优化完成!")
print(f"最优参数: {result.best_params}")
print(f"最优得分: {result.best_score:.4f}")
print(f"执行时间: {result.execution_time:.2f}秒")
print(f"总迭代数: {result.total_iterations}")
