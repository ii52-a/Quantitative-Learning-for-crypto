"""测试参数优化器导入"""
from Strategy.parameter_optimizer import ParameterRange, RISK_PARAM_RANGES, get_all_optimizable_params
from Strategy.templates import get_strategy

print("导入成功")
print(f"RISK_PARAM_RANGES: {list(RISK_PARAM_RANGES.keys())}")

strategy = get_strategy("MACDStrategy")
all_params = get_all_optimizable_params(strategy, include_risk_params=True)
print(f"策略参数: {[p.name for p in all_params['strategy']]}")
print(f"风险参数: {[p.name for p in all_params['risk']]}")
