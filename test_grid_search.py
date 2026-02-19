"""测试网格搜索"""
from Strategy.parameter_optimizer import ParameterOptimizer, ParameterRange
from Strategy.templates import get_strategy
from Data.data_service import get_data_service, DataServiceConfig
from core.config import BacktestConfig

print("测试网格搜索...")

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
    ParameterRange("fast_period", 10, 14, 2),
    ParameterRange("slow_period", 24, 28, 2),
]

total_combinations = 1
for pr in param_ranges:
    values = pr.get_values()
    print(f"  {pr.name}: {values} ({len(values)}个)")
    total_combinations *= len(values)

print(f"\n总组合数: {total_combinations}")

print("\n开始网格搜索...")
result = optimizer.grid_search(
    param_ranges=param_ranges,
    max_iterations=20,
)

print(f"\n优化完成!")
print(f"最优参数: {result.best_params}")
print(f"最优得分: {result.best_score:.4f}")
print(f"执行时间: {result.execution_time:.2f}秒")
print(f"总迭代数: {result.total_iterations}")
