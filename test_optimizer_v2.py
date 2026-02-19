"""测试参数优化器v2.0"""
print("测试参数优化器v2.0导入...")
from Strategy.parameter_optimizer import ParameterOptimizer, ParameterRange, get_all_optimizable_params
from Strategy.templates import get_strategy
from Data.data_service import get_data_service, DataServiceConfig
from core.config import BacktestConfig

print("✅ 导入成功")

print("\n测试遗传算法...")
service = get_data_service(DataServiceConfig(prefer_database=True, auto_init=True))
data = service.get_klines("BTCUSDT", "30min", 200)

config = BacktestConfig(initial_capital=10000, leverage=5)
strategy = get_strategy("MACDStrategy")

optimizer = ParameterOptimizer(
    strategy_class=type(strategy),
    data=data,
    base_config=config,
)

param_ranges = [
    ParameterRange("fast_period", 10, 15, 1),
    ParameterRange("slow_period", 24, 28, 1),
]

result = optimizer.genetic_algorithm(
    param_ranges=param_ranges,
    n_generations=3,
    population_size=10,
)

print(f"优化方法: {result.optimization_method}")
print(f"最优参数: {result.best_params}")
print(f"最优得分: {result.best_score:.4f}")
print(f"参数重要性: {result.parameter_importance}")
print(f"收敛数据点: {len(result.convergence_data) if result.convergence_data else 0}")
print("✅ 遗传算法测试通过")
