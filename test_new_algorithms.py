"""测试新优化算法"""
from Strategy.parameter_optimizer import ParameterOptimizer, ParameterRange
from Strategy.templates import get_strategy
from Data.data_service import get_data_service, DataServiceConfig
from core.config import BacktestConfig

print("测试新优化算法...")

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
    ParameterRange("fast_period", 10, 15, 1),
    ParameterRange("slow_period", 24, 28, 1),
]

print("\n" + "=" * 60)
print("测试模拟退火")
print("=" * 60)
result = optimizer.simulated_annealing(
    param_ranges=param_ranges,
    n_iterations=20,
)
print(f"最优参数: {result.best_params}")
print(f"最优得分: {result.best_score:.4f}")
print(f"执行时间: {result.execution_time:.2f}秒")

print("\n" + "=" * 60)
print("测试粒子群优化")
print("=" * 60)
result = optimizer.particle_swarm_optimization(
    param_ranges=param_ranges,
    n_particles=10,
    n_iterations=5,
)
print(f"最优参数: {result.best_params}")
print(f"最优得分: {result.best_score:.4f}")
print(f"执行时间: {result.execution_time:.2f}秒")

print("\n" + "=" * 60)
print("测试强化学习优化")
print("=" * 60)
result = optimizer.reinforcement_learning_optimize(
    param_ranges=param_ranges,
    n_episodes=20,
)
print(f"最优参数: {result.best_params}")
print(f"最优得分: {result.best_score:.4f}")
print(f"执行时间: {result.execution_time:.2f}秒")

print("\n所有测试通过!")
