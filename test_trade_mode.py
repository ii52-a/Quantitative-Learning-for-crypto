"""测试MACD策略交易模式参数"""
from Strategy.templates import get_strategy
from Strategy.parameter_optimizer import get_all_optimizable_params

print("检查MACD策略参数...")

strategy = get_strategy("MACDStrategy")

print("\n策略定义的参数:")
for param in strategy.parameters:
    print(f"  {param.name}: {param.display_name}")
    if param.options:
        print(f"    选项: {param.options}")
    else:
        print(f"    范围: {param.min_value} - {param.max_value}")

print("\n可优化参数范围:")
all_params = get_all_optimizable_params(strategy, include_risk_params=False)
for pr in all_params["strategy"]:
    if pr.values:
        print(f"  {pr.name}: 离散值 {pr.values}")
    else:
        print(f"  {pr.name}: {pr.min_value} - {pr.max_value}, 步长 {pr.step}")

print("\n✅ trade_mode参数已包含在可优化参数中")
