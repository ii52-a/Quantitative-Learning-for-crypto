"""测试强化学习优化修复"""
from Strategy.parameter_optimizer import ParameterOptimizer, ParameterRange
from Strategy.templates import get_strategy
from Data.data_service import get_data_service, DataServiceConfig
from core.config import BacktestConfig

print("测试强化学习优化修复...")

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
    ParameterRange("fast_period", 12, 12, 1),
    ParameterRange("slow_period", 26, 26, 1),
]

print("\n测试相同最小最大值的情况:")
result = optimizer.reinforcement_learning_optimize(
    param_ranges=param_ranges,
    n_episodes=5,
)
print(f"最优参数: {result.best_params}")
print(f"最优得分: {result.best_score:.4f}")

param_ranges2 = [
    ParameterRange("fast_period", 10, 15, 1),
    ParameterRange("slow_period", 24, 28, 1),
]

print("\n测试正常范围:")
result2 = optimizer.reinforcement_learning_optimize(
    param_ranges=param_ranges2,
    n_episodes=10,
)
print(f"最优参数: {result2.best_params}")
print(f"最优得分: {result2.best_score:.4f}")

print("\n✅ 测试通过!")
