"""
弱机器学习参数优化模块

提供策略参数优化功能：
- 网格搜索
- 随机搜索
- 贝叶斯优化（简化版）
- 多目标优化支持
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
import itertools
import random
import math

import pandas as pd
import numpy as np

from Strategy.base import BaseStrategy, StrategyParameter
from backtest.engine import BacktestEngine, BacktestResult
from core.config import BacktestConfig
from app_logger.logger_setup import Logger

logger = Logger(__name__)


@dataclass
class ParameterRange:
    """参数范围定义"""
    name: str
    min_value: float
    max_value: float
    step: float = 1.0
    values: list[Any] | None = None
    
    def get_values(self) -> list[Any]:
        if self.values is not None:
            return self.values
        
        if self.step == 0:
            return [self.min_value]
        
        values = []
        current = self.min_value
        while current <= self.max_value + 1e-9:
            values.append(current)
            current += self.step
        return values


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: dict[str, Any]
    best_score: float
    best_result: BacktestResult | None
    all_results: list[dict[str, Any]]
    optimization_method: str
    total_iterations: int
    execution_time: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "total_iterations": self.total_iterations,
            "execution_time": round(self.execution_time, 2),
            "optimization_method": self.optimization_method,
            "top_results": [
                {
                    "params": r["params"],
                    "score": r["score"],
                    "total_return_pct": r.get("total_return_pct", 0),
                    "max_drawdown_pct": r.get("max_drawdown_pct", 0),
                    "win_rate": r.get("win_rate", 0),
                }
                for r in sorted(self.all_results, key=lambda x: x["score"], reverse=True)[:5]
            ],
        }


class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        data: pd.DataFrame,
        base_config: BacktestConfig,
        optimization_metric: str = "sharpe_ratio",
    ):
        self.strategy_class = strategy_class
        self.data = data
        self.base_config = base_config
        self.optimization_metric = optimization_metric
        self._results: list[dict[str, Any]] = []
    
    def grid_search(
        self,
        param_ranges: list[ParameterRange],
        max_iterations: int = 1000,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OptimizationResult:
        """网格搜索优化"""
        start_time = datetime.now()
        
        param_combinations = list(itertools.product(
            *[pr.get_values() for pr in param_ranges]
        ))
        
        param_names = [pr.name for pr in param_ranges]
        total = min(len(param_combinations), max_iterations)
        
        logger.info(f"网格搜索: 总组合数={len(param_combinations)}, 执行={total}")
        
        self._results = []
        best_score = float('-inf')
        best_params = {}
        best_result = None
        
        for i, combo in enumerate(param_combinations[:max_iterations]):
            params = dict(zip(param_names, combo))
            
            result = self._evaluate_params(params)
            self._results.append(result)
            
            if result["score"] > best_score:
                best_score = result["score"]
                best_params = params.copy()
                best_result = result.get("backtest_result")
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_result=best_result,
            all_results=self._results,
            optimization_method="grid_search",
            total_iterations=len(self._results),
            execution_time=execution_time,
        )
    
    def random_search(
        self,
        param_ranges: list[ParameterRange],
        n_iterations: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OptimizationResult:
        """随机搜索优化"""
        start_time = datetime.now()
        
        logger.info(f"随机搜索: 迭代次数={n_iterations}")
        
        self._results = []
        best_score = float('-inf')
        best_params = {}
        best_result = None
        
        for i in range(n_iterations):
            params = {}
            for pr in param_ranges:
                if pr.values is not None:
                    params[pr.name] = random.choice(pr.values)
                else:
                    if isinstance(pr.min_value, int) and isinstance(pr.step, int):
                        params[pr.name] = random.randint(
                            int(pr.min_value), int(pr.max_value)
                        )
                    else:
                        params[pr.name] = random.uniform(pr.min_value, pr.max_value)
            
            result = self._evaluate_params(params)
            self._results.append(result)
            
            if result["score"] > best_score:
                best_score = result["score"]
                best_params = params.copy()
                best_result = result.get("backtest_result")
            
            if progress_callback:
                progress_callback(i + 1, n_iterations)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_result=best_result,
            all_results=self._results,
            optimization_method="random_search",
            total_iterations=n_iterations,
            execution_time=execution_time,
        )
    
    def bayesian_optimization(
        self,
        param_ranges: list[ParameterRange],
        n_iterations: int = 50,
        n_initial: int = 10,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OptimizationResult:
        """简化版贝叶斯优化"""
        start_time = datetime.now()
        
        logger.info(f"贝叶斯优化: 初始采样={n_initial}, 总迭代={n_iterations}")
        
        self._results = []
        best_score = float('-inf')
        best_params = {}
        best_result = None
        
        sampled_points: list[dict[str, float]] = []
        sampled_scores: list[float] = []
        
        for i in range(n_initial):
            params = {}
            for pr in param_ranges:
                if pr.values is not None:
                    params[pr.name] = random.choice(pr.values)
                else:
                    params[pr.name] = random.uniform(pr.min_value, pr.max_value)
            
            result = self._evaluate_params(params)
            self._results.append(result)
            sampled_points.append(params)
            sampled_scores.append(result["score"])
            
            if result["score"] > best_score:
                best_score = result["score"]
                best_params = params.copy()
                best_result = result.get("backtest_result")
            
            if progress_callback:
                progress_callback(i + 1, n_iterations)
        
        for i in range(n_initial, n_iterations):
            next_params = self._select_next_point(
                param_ranges, sampled_points, sampled_scores
            )
            
            result = self._evaluate_params(next_params)
            self._results.append(result)
            sampled_points.append(next_params)
            sampled_scores.append(result["score"])
            
            if result["score"] > best_score:
                best_score = result["score"]
                best_params = next_params.copy()
                best_result = result.get("backtest_result")
            
            if progress_callback:
                progress_callback(i + 1, n_iterations)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_result=best_result,
            all_results=self._results,
            optimization_method="bayesian_optimization",
            total_iterations=n_iterations,
            execution_time=execution_time,
        )
    
    def _select_next_point(
        self,
        param_ranges: list[ParameterRange],
        sampled_points: list[dict[str, float]],
        sampled_scores: list[float],
    ) -> dict[str, float]:
        """选择下一个采样点（基于期望改进）"""
        best_score = max(sampled_scores)
        
        best_idx = sampled_scores.index(best_score)
        best_point = sampled_points[best_idx]
        
        next_point = {}
        for pr in param_ranges:
            if pr.values is not None:
                candidates = [v for v in pr.values if v != best_point.get(pr.name)]
                if candidates:
                    weights = []
                    for v in candidates:
                        min_dist = min(
                            abs(v - p.get(pr.name, 0)) 
                            for p in sampled_points
                        )
                        weights.append(min_dist + 0.1)
                    total_weight = sum(weights)
                    weights = [w / total_weight for w in weights]
                    next_point[pr.name] = random.choices(candidates, weights=weights)[0]
                else:
                    next_point[pr.name] = random.choice(pr.values)
            else:
                exploration = random.uniform(-1, 1) * (pr.max_value - pr.min_value) * 0.2
                new_value = best_point.get(pr.name, (pr.min_value + pr.max_value) / 2) + exploration
                next_point[pr.name] = max(pr.min_value, min(pr.max_value, new_value))
        
        return next_point
    
    def _evaluate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """评估参数组合"""
        try:
            strategy = self.strategy_class(params)
            engine = BacktestEngine(strategy, self.base_config)
            result = engine.run(self.data)
            
            score = self._calculate_score(result)
            
            return {
                "params": params,
                "score": score,
                "total_return_pct": result.total_return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "win_rate": result.win_rate,
                "sharpe_ratio": result.sharpe_ratio,
                "total_trades": result.total_trades,
                "backtest_result": result,
            }
        except Exception as e:
            logger.warning(f"参数评估失败: {params}, 错误: {e}")
            return {
                "params": params,
                "score": float('-inf'),
                "total_return_pct": 0,
                "max_drawdown_pct": 100,
                "win_rate": 0,
                "sharpe_ratio": 0,
                "total_trades": 0,
                "backtest_result": None,
            }
    
    def _calculate_score(self, result: BacktestResult) -> float:
        """计算综合得分"""
        if self.optimization_metric == "sharpe_ratio":
            return result.sharpe_ratio
        elif self.optimization_metric == "total_return":
            return result.total_return_pct
        elif self.optimization_metric == "calmar_ratio":
            return result.calmar_ratio
        elif self.optimization_metric == "composite":
            if result.max_drawdown_pct == 0:
                return result.total_return_pct
            return result.total_return_pct / result.max_drawdown_pct * result.win_rate / 100
        else:
            return result.sharpe_ratio


class MultiObjectiveOptimizer(ParameterOptimizer):
    """多目标优化器"""
    
    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        data: pd.DataFrame,
        base_config: BacktestConfig,
        objectives: list[str] = None,
    ):
        super().__init__(strategy_class, data, base_config)
        self.objectives = objectives or ["sharpe_ratio", "max_drawdown_pct"]
    
    def pareto_optimize(
        self,
        param_ranges: list[ParameterRange],
        n_iterations: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OptimizationResult:
        """帕累托优化"""
        start_time = datetime.now()
        
        logger.info(f"帕累托优化: 迭代次数={n_iterations}")
        
        self._results = []
        pareto_front: list[dict[str, Any]] = []
        
        for i in range(n_iterations):
            params = {}
            for pr in param_ranges:
                if pr.values is not None:
                    params[pr.name] = random.choice(pr.values)
                else:
                    params[pr.name] = random.uniform(pr.min_value, pr.max_value)
            
            result = self._evaluate_params(params)
            self._results.append(result)
            
            is_dominated = False
            to_remove = []
            
            for existing in pareto_front:
                if self._dominates(existing, result):
                    is_dominated = True
                    break
                if self._dominates(result, existing):
                    to_remove.append(existing)
            
            if not is_dominated:
                pareto_front.append(result)
                for r in to_remove:
                    if r in pareto_front:
                        pareto_front.remove(r)
            
            if progress_callback:
                progress_callback(i + 1, n_iterations)
        
        best_result = max(pareto_front, key=lambda x: x["score"]) if pareto_front else None
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_result["params"] if best_result else {},
            best_score=best_result["score"] if best_result else 0,
            best_result=best_result.get("backtest_result") if best_result else None,
            all_results=self._results,
            optimization_method="pareto_optimization",
            total_iterations=n_iterations,
            execution_time=execution_time,
        )
    
    def _dominates(self, a: dict[str, Any], b: dict[str, Any]) -> bool:
        """判断a是否支配b"""
        a_values = [
            a.get("sharpe_ratio", 0),
            -a.get("max_drawdown_pct", 100),
            a.get("win_rate", 0),
        ]
        b_values = [
            b.get("sharpe_ratio", 0),
            -b.get("max_drawdown_pct", 100),
            b.get("win_rate", 0),
        ]
        
        better_in_any = False
        for av, bv in zip(a_values, b_values):
            if av > bv:
                better_in_any = True
            elif av < bv:
                return False
        
        return better_in_any


def get_parameter_ranges_from_strategy(
    strategy: BaseStrategy,
) -> list[ParameterRange]:
    """从策略定义获取参数范围"""
    ranges = []
    
    for param in strategy.parameters:
        if param.value_type == int:
            ranges.append(ParameterRange(
                name=param.name,
                min_value=param.min_value or param.default_value - 10,
                max_value=param.max_value or param.default_value + 10,
                step=1,
            ))
        elif param.value_type == float:
            ranges.append(ParameterRange(
                name=param.name,
                min_value=param.min_value or 0,
                max_value=param.max_value or param.default_value * 2,
                step=(param.max_value - param.min_value) / 10 if param.max_value and param.min_value else 0.1,
            ))
        elif param.options is not None:
            ranges.append(ParameterRange(
                name=param.name,
                min_value=0,
                max_value=0,
                values=param.options,
            ))
    
    return ranges


def tune_parameter_ranges(
    ranges: list[ParameterRange],
    breadth: float = 1.0,
    depth: int = 1,
) -> list[ParameterRange]:
    """根据优化广度/深度调整参数搜索空间。

    breadth:
        控制搜索范围，>1 扩大区间，<1 缩小区间。
    depth:
        控制搜索精度，值越大步长越小。
    """
    tuned_ranges: list[ParameterRange] = []
    safe_breadth = max(0.2, float(breadth))
    safe_depth = max(1, int(depth))

    for pr in ranges:
        if pr.values is not None:
            values = list(pr.values)
            if safe_breadth < 1 and len(values) > 2:
                keep = max(2, int(round(len(values) * safe_breadth)))
                step = max(1, len(values) // keep)
                sampled = values[::step][:keep]
                if values[-1] not in sampled:
                    sampled[-1] = values[-1]
                values = sampled

            tuned_ranges.append(ParameterRange(
                name=pr.name,
                min_value=pr.min_value,
                max_value=pr.max_value,
                step=pr.step,
                values=values,
            ))
            continue

        min_v = float(pr.min_value)
        max_v = float(pr.max_value)
        center = (min_v + max_v) / 2
        half_span = (max_v - min_v) / 2
        tuned_min = center - (half_span * safe_breadth)
        tuned_max = center + (half_span * safe_breadth)

        if min_v >= 0:
            tuned_min = max(0.0, tuned_min)

        step = pr.step
        if step:
            step = step / safe_depth
            if isinstance(pr.step, int) and pr.step > 0:
                step = max(1, int(round(step)))

        tuned_ranges.append(ParameterRange(
            name=pr.name,
            min_value=tuned_min,
            max_value=tuned_max,
            step=step,
        ))

    return tuned_ranges
