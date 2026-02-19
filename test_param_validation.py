"""测试参数验证"""
from Strategy.parameter_optimizer import ParameterOptimizer, ParameterRange
from Strategy.templates import get_strategy
from Data.data_service import get_data_service, DataServiceConfig
from core.config import BacktestConfig

print("测试参数验证...")

service = get_data_service(DataServiceConfig(prefer_database=True, auto_init=True))
data = service.get_klines("BTCUSDT", "30min", 100)

config = BacktestConfig(initial_capital=10000, leverage=5)
strategy = get_strategy("MACDStrategy")

optimizer = ParameterOptimizer(
    strategy_class=type(strategy),
    data=data,
    base_config=config,
)

print("\n测试无效参数 (fast_period >= slow_period):")
invalid_params = {"fast_period": 30, "slow_period": 20}
is_valid = optimizer._validate_params(invalid_params)
print(f"  参数: {invalid_params}")
print(f"  验证结果: {'有效' if is_valid else '无效'}")

print("\n测试有效参数 (fast_period < slow_period):")
valid_params = {"fast_period": 12, "slow_period": 26}
is_valid = optimizer._validate_params(valid_params)
print(f"  参数: {valid_params}")
print(f"  验证结果: {'有效' if is_valid else '无效'}")

print("\n测试遗传算法参数约束:")
result = optimizer.genetic_algorithm(
    param_ranges=[
        ParameterRange("fast_period", 10, 30, 2),
        ParameterRange("slow_period", 20, 40, 2),
    ],
    n_generations=2,
    population_size=5,
)

print(f"最优参数: {result.best_params}")
if "fast_period" in result.best_params and "slow_period" in result.best_params:
    if result.best_params["fast_period"] < result.best_params["slow_period"]:
        print("✅ 参数约束满足: fast_period < slow_period")
    else:
        print("❌ 参数约束不满足!")

print("\n测试完成")
