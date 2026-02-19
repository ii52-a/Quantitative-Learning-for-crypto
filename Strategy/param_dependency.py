"""参数依赖关系定义和处理"""
from dataclasses import dataclass, field
from typing import Any, Callable
import random


@dataclass
class ParameterDependency:
    """参数依赖关系"""
    param_name: str
    depends_on: str
    constraint_func: Callable[[Any, Any], bool]
    value_func: Callable[[Any], list[Any]] | None = None
    
    def validate(self, param_value: Any, depend_value: Any) -> bool:
        """验证约束"""
        return self.constraint_func(param_value, depend_value)
    
    def get_valid_values(self, depend_value: Any) -> list[Any] | None:
        """根据依赖参数获取有效值"""
        if self.value_func:
            return self.value_func(depend_value)
        return None


@dataclass
class ParameterRangeV2:
    """支持依赖关系的参数范围"""
    name: str
    min_value: float
    max_value: float
    step: float = 1.0
    values: list[Any] | None = None
    dependency: ParameterDependency | None = None
    
    def get_values(self, context: dict[str, Any] | None = None) -> list[Any]:
        """获取有效参数值，考虑依赖关系"""
        if self.values is not None:
            if self.dependency and context:
                depend_value = context.get(self.dependency.depends_on)
                if depend_value is not None:
                    valid_values = self.dependency.get_valid_values(depend_value)
                    if valid_values:
                        return [v for v in self.values if v in valid_values]
            return self.values
        
        base_values = []
        current = self.min_value
        while current <= self.max_value + 1e-9:
            base_values.append(current)
            current += self.step
        
        if self.dependency and context:
            depend_value = context.get(self.dependency.depends_on)
            if depend_value is not None:
                valid_values = self.dependency.get_valid_values(depend_value)
                if valid_values:
                    return [v for v in base_values if v in valid_values]
        
        return base_values
    
    def get_random_value(self, context: dict[str, Any] | None = None) -> Any:
        """随机获取有效值"""
        values = self.get_values(context)
        if not values:
            return self.min_value
        return random.choice(values)


def create_macd_dependencies() -> dict[str, ParameterDependency]:
    """创建MACD策略的参数依赖关系"""
    
    def slow_period_values(fast_period: float) -> list[float]:
        gap = max(5, fast_period * 0.5)
        return [v for v in range(int(fast_period + gap), 60)]
    
    def stop_loss_values(leverage: float) -> list[float]:
        if leverage >= 20:
            return [1, 2, 3, 4, 5]
        elif leverage >= 10:
            return [2, 3, 4, 5, 7, 10]
        else:
            return [3, 5, 7, 10, 15, 20]
    
    return {
        "slow_period": ParameterDependency(
            param_name="slow_period",
            depends_on="fast_period",
            constraint_func=lambda slow, fast: slow > fast,
            value_func=slow_period_values,
        ),
        "stop_loss_pct": ParameterDependency(
            param_name="stop_loss_pct",
            depends_on="leverage",
            constraint_func=lambda sl, lev: (lev >= 20 and sl <= 5) or (lev < 20 and sl <= 20),
            value_func=stop_loss_values,
        ),
    }


def create_bollinger_dependencies() -> dict[str, ParameterDependency]:
    """创建布林带策略的参数依赖关系"""
    
    def std_values(period: float) -> list[float]:
        if period <= 10:
            return [1.0, 1.5, 2.0]
        elif period <= 20:
            return [1.5, 2.0, 2.5]
        else:
            return [2.0, 2.5, 3.0]
    
    return {
        "std_dev": ParameterDependency(
            param_name="std_dev",
            depends_on="period",
            constraint_func=lambda std, period: True,
            value_func=std_values,
        ),
    }


class DependentParameterOptimizer:
    """支持参数依赖的优化器"""
    
    def __init__(self, dependencies: dict[str, ParameterDependency] | None = None):
        self.dependencies = dependencies or {}
    
    def generate_valid_params(
        self,
        param_ranges: list[ParameterRangeV2],
        n_samples: int = 100,
    ) -> list[dict[str, Any]]:
        """生成有效的参数组合"""
        results = []
        param_order = self._get_param_order(param_ranges)
        
        for _ in range(n_samples):
            params = {}
            context = {}
            valid = True
            
            for param_name in param_order:
                pr = next((p for p in param_ranges if p.name == param_name), None)
                if not pr:
                    continue
                
                value = pr.get_random_value(context)
                
                if pr.name in self.dependencies:
                    dep = self.dependencies[pr.name]
                    depend_value = context.get(dep.depends_on)
                    if depend_value is not None:
                        if not dep.validate(value, depend_value):
                            valid = False
                            break
                
                params[param_name] = value
                context[param_name] = value
            
            if valid:
                results.append(params)
        
        return results
    
    def _get_param_order(self, param_ranges: list[ParameterRangeV2]) -> list[str]:
        """确定参数评估顺序（依赖参数在后）"""
        order = []
        remaining = [pr.name for pr in param_ranges]
        
        while remaining:
            for name in remaining[:]:
                dep = self.dependencies.get(name)
                if dep is None or dep.depends_on in order:
                    order.append(name)
                    remaining.remove(name)
                    break
            else:
                order.extend(remaining)
                break
        
        return order
    
    def validate_params(self, params: dict[str, Any]) -> bool:
        """验证参数组合是否满足所有约束"""
        for param_name, dep in self.dependencies.items():
            if param_name in params:
                depend_value = params.get(dep.depends_on)
                if depend_value is not None:
                    if not dep.validate(params[param_name], depend_value):
                        return False
        return True
