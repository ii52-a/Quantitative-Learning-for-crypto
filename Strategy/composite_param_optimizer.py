"""复合参数优化策略 - 组合多种优化算法寻找最优解"""
from dataclasses import dataclass, field
from typing import Any, Callable
from datetime import datetime
import pandas as pd
import numpy as np

from Strategy.parameter_optimizer import (
    ParameterOptimizer,
    ParameterRange,
    OptimizationResult,
)
from core.config import BacktestConfig


MIN_DATA_POINTS = 5000


@dataclass
class AlgorithmResult:
    """单个算法的优化结果"""
    algorithm_name: str
    best_params: dict[str, Any]
    best_score: float
    execution_time: float
    iterations: int
    convergence_speed: float = 0.0


@dataclass
class CompositeOptResult:
    """复合优化结果"""
    best_params: dict[str, Any]
    best_score: float
    best_algorithm: str
    all_results: list[AlgorithmResult] = field(default_factory=list)
    execution_time: float = 0.0
    data_points: int = 0
    time_span_hours: float = 0.0
    
    def get_ranking(self) -> list[tuple[str, float]]:
        """获取算法排名"""
        return sorted(
            [(r.algorithm_name, r.best_score) for r in self.all_results],
            key=lambda x: x[1],
            reverse=True
        )


class CompositeParameterOptimizer:
    """复合参数优化器 - 组合多种优化算法"""
    
    ALGORITHMS = [
        "random_search",
        "genetic_algorithm", 
        "simulated_annealing",
        "particle_swarm",
        "reinforcement_learning",
        "bayesian",
    ]
    
    ALGORITHM_NAMES = {
        "random_search": "随机搜索",
        "genetic_algorithm": "遗传算法",
        "simulated_annealing": "模拟退火",
        "particle_swarm": "粒子群优化",
        "reinforcement_learning": "强化学习",
        "bayesian": "贝叶斯优化",
    }
    
    def __init__(
        self,
        strategy_class,
        data: pd.DataFrame,
        base_config: BacktestConfig,
        optimization_metric: str = "sharpe_ratio",
    ):
        self.strategy_class = strategy_class
        self.data = data
        self.base_config = base_config
        self.optimization_metric = optimization_metric
        
        self._validate_data()
        
        self._optimizer = ParameterOptimizer(
            strategy_class=strategy_class,
            data=data,
            base_config=base_config,
            optimization_metric=optimization_metric,
        )
        
        self._stop_flag = False
    
    def _validate_data(self) -> None:
        """验证数据量是否满足最低要求"""
        data_count = len(self.data)
        
        if data_count < MIN_DATA_POINTS:
            raise ValueError(
                f"数据量不足: 当前{data_count}条, 最低要求{MIN_DATA_POINTS}条。"
                f"数据量过小可能导致模型过拟合。"
            )
    
    def stop(self) -> None:
        """停止优化"""
        self._stop_flag = True
        self._optimizer.stop()
    
    def run_single_algorithm(
        self,
        algorithm: str,
        param_ranges: list[ParameterRange],
        iterations: int = 50,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> AlgorithmResult:
        """运行单个优化算法"""
        start_time = datetime.now()
        
        if algorithm == "random_search":
            result = self._optimizer.random_search(
                param_ranges=param_ranges,
                n_iterations=iterations,
                progress_callback=progress_callback,
            )
        elif algorithm == "genetic_algorithm":
            result = self._optimizer.genetic_algorithm(
                param_ranges=param_ranges,
                n_generations=max(5, iterations // 10),
                population_size=20,
                progress_callback=progress_callback,
            )
        elif algorithm == "simulated_annealing":
            result = self._optimizer.simulated_annealing(
                param_ranges=param_ranges,
                n_iterations=iterations,
                progress_callback=progress_callback,
            )
        elif algorithm == "particle_swarm":
            result = self._optimizer.particle_swarm_optimization(
                param_ranges=param_ranges,
                n_particles=15,
                n_iterations=max(5, iterations // 15),
                progress_callback=progress_callback,
            )
        elif algorithm == "reinforcement_learning":
            result = self._optimizer.reinforcement_learning_optimize(
                param_ranges=param_ranges,
                n_episodes=iterations,
                progress_callback=progress_callback,
            )
        elif algorithm == "bayesian":
            result = self._optimizer.bayesian_optimization(
                param_ranges=param_ranges,
                n_iterations=iterations,
                progress_callback=progress_callback,
            )
        else:
            raise ValueError(f"未知算法: {algorithm}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        convergence_speed = result.best_score / execution_time if execution_time > 0 else 0
        
        return AlgorithmResult(
            algorithm_name=algorithm,
            best_params=result.best_params,
            best_score=result.best_score,
            execution_time=execution_time,
            iterations=result.total_iterations,
            convergence_speed=convergence_speed,
        )
    
    def run_composite(
        self,
        param_ranges: list[ParameterRange],
        algorithms: list[str] | None = None,
        iterations_per_algorithm: int = 30,
        progress_callback: Callable[[str, int, int], None] | None = None,
        result_callback: Callable[[AlgorithmResult], None] | None = None,
    ) -> CompositeOptResult:
        """
        运行复合优化 - 依次执行多种算法并比较结果
        
        Args:
            param_ranges: 参数范围列表
            algorithms: 要使用的算法列表，None则使用全部
            iterations_per_algorithm: 每个算法的迭代次数
            progress_callback: 进度回调 (算法名, 当前, 总数)
            result_callback: 单算法结果回调
        """
        start_time = datetime.now()
        
        if algorithms is None:
            algorithms = self.ALGORITHMS.copy()
        
        all_results: list[AlgorithmResult] = []
        best_score = float('-inf')
        best_params = {}
        best_algorithm = ""
        
        total_algorithms = len(algorithms)
        
        for algo_idx, algorithm in enumerate(algorithms):
            if self._stop_flag:
                break
            
            algo_name = self.ALGORITHM_NAMES.get(algorithm, algorithm)
            
            def algo_progress(current, total):
                if progress_callback:
                    progress_callback(algo_name, current, total)
            
            try:
                result = self.run_single_algorithm(
                    algorithm=algorithm,
                    param_ranges=param_ranges,
                    iterations=iterations_per_algorithm,
                    progress_callback=algo_progress,
                )
                
                all_results.append(result)
                
                if result_callback:
                    result_callback(result)
                
                if result.best_score > best_score:
                    best_score = result.best_score
                    best_params = result.best_params.copy()
                    best_algorithm = algorithm
                    
            except Exception as e:
                print(f"算法 {algo_name} 执行失败: {e}")
                continue
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        time_span = 0
        if len(self.data.index) > 1:
            time_span = (self.data.index[-1] - self.data.index[0]).total_seconds() / 3600
        
        return CompositeOptResult(
            best_params=best_params,
            best_score=best_score,
            best_algorithm=best_algorithm,
            all_results=all_results,
            execution_time=execution_time,
            data_points=len(self.data),
            time_span_hours=time_span,
        )
    
    def generate_report(self, result: CompositeOptResult) -> str:
        """生成优化报告"""
        lines = [
            "═" * 50,
            "复合参数优化报告",
            "═" * 50,
            f"数据量: {result.data_points} 条",
            f"时间跨度: {result.time_span_hours:.1f} 小时",
            f"总执行时间: {result.execution_time:.1f} 秒",
            "",
            f"最优算法: {self.ALGORITHM_NAMES.get(result.best_algorithm, result.best_algorithm)}",
            f"最优得分: {result.best_score:.4f}",
            "",
            "最优参数:",
        ]
        
        for key, value in result.best_params.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        
        lines.extend([
            "",
            "算法排名:",
        ])
        
        ranking = result.get_ranking()
        for i, (name, score) in enumerate(ranking, 1):
            algo_name = self.ALGORITHM_NAMES.get(name, name)
            lines.append(f"  {i}. {algo_name}: {score:.4f}")
        
        lines.extend([
            "",
            "各算法详情:",
        ])
        
        for r in result.all_results:
            algo_name = self.ALGORITHM_NAMES.get(r.algorithm_name, r.algorithm_name)
            lines.extend([
                f"  {algo_name}:",
                f"    得分: {r.best_score:.4f}",
                f"    迭代: {r.iterations}次",
                f"    耗时: {r.execution_time:.1f}s",
                f"    效率: {r.convergence_speed:.4f}/s",
            ])
        
        lines.append("═" * 50)
        
        return "\n".join(lines)


def run_composite_optimization(
    strategy_class,
    data: pd.DataFrame,
    base_config: BacktestConfig,
    param_ranges: list[ParameterRange],
    algorithms: list[str] | None = None,
    iterations_per_algorithm: int = 30,
    optimization_metric: str = "sharpe_ratio",
) -> CompositeOptResult:
    """运行复合优化的便捷函数"""
    optimizer = CompositeParameterOptimizer(
        strategy_class=strategy_class,
        data=data,
        base_config=base_config,
        optimization_metric=optimization_metric,
    )
    
    return optimizer.run_composite(
        param_ranges=param_ranges,
        algorithms=algorithms,
        iterations_per_algorithm=iterations_per_algorithm,
    )
