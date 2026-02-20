"""
参数优化模块 v2.0

提供策略参数优化功能：
- 单一参数最优探索
- 网格搜索
- 随机搜索
- 贝叶斯优化
- 多目标优化
- 遗传算法优化
- 参数探索结果可视化
- 探索中断与恢复
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
import itertools
import random
import math
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    category: str = "strategy"
    display_name: str = ""
    fixed: bool = False
    fixed_value: Any = None
    
    def get_values(self) -> list[Any]:
        if self.fixed and self.fixed_value is not None:
            return [self.fixed_value]
        if self.values is not None:
            return self.values
        if self.step == 0:
            return [self.min_value]

        step = float(self.step)
        if step < 0:
            step = abs(step)
        span = float(self.max_value) - float(self.min_value)
        if span < 0:
            return [self.min_value]

        count = int(math.floor(span / step + 1e-12)) + 1
        values = []
        for i in range(max(1, count)):
            value = float(self.min_value) + i * step
            value = min(value, float(self.max_value))
            value = round(value, 10)
            if abs(value - round(value)) < 1e-10:
                value = int(round(value))
            values.append(value)

        if values and values[-1] != self.max_value and abs(float(values[-1]) - float(self.max_value)) > 1e-8:
            max_value = round(float(self.max_value), 10)
            if abs(max_value - round(max_value)) < 1e-10:
                max_value = int(round(max_value))
            values.append(max_value)

        return values
    
    def get_random_value(self) -> Any:
        if self.fixed and self.fixed_value is not None:
            return self.fixed_value
        if self.values is not None:
            return random.choice(self.values)
        if isinstance(self.min_value, int) and isinstance(self.step, int):
            return random.randint(int(self.min_value), int(self.max_value))
        return random.uniform(self.min_value, self.max_value)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "values": self.values,
            "category": self.category,
            "display_name": self.display_name or self.name,
            "fixed": self.fixed,
            "fixed_value": self.fixed_value,
        }


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
    checkpoint_path: str | None = None
    parameter_importance: dict[str, float] | None = None
    convergence_data: list[dict[str, Any]] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "total_iterations": self.total_iterations,
            "execution_time": round(self.execution_time, 2),
            "optimization_method": self.optimization_method,
            "parameter_importance": self.parameter_importance,
            "top_results": [
                {
                    "params": r["params"],
                    "score": r["score"],
                    "total_return_pct": r.get("total_return_pct", 0),
                    "max_drawdown_pct": r.get("max_drawdown_pct", 0),
                    "win_rate": r.get("win_rate", 0),
                    "total_trades": r.get("total_trades", 0),
                }
                for r in sorted(self.all_results, key=lambda x: x["score"], reverse=True)[:10]
            ],
        }
    
    def get_visualization_data(self) -> dict[str, Any]:
        """获取可视化数据"""
        df = pd.DataFrame(self.all_results)
        
        param_names = list(self.best_params.keys())
        
        viz_data = {
            "convergence": self.convergence_data or [],
            "parameter_distribution": {},
            "score_distribution": df["score"].tolist() if "score" in df.columns else [],
            "top_10_params": [],
        }
        
        for param in param_names:
            if param in df.columns:
                viz_data["parameter_distribution"][param] = {
                    "values": df[param].tolist(),
                    "scores": df["score"].tolist() if "score" in df.columns else [],
                }
        
        sorted_results = sorted(self.all_results, key=lambda x: x["score"], reverse=True)[:10]
        viz_data["top_10_params"] = [
            {"rank": i+1, "params": r["params"], "score": r["score"]}
            for i, r in enumerate(sorted_results)
        ]
        
        return viz_data


class ParameterOptimizer:
    """参数优化器 v2.0"""
    
    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        data: pd.DataFrame,
        base_config: BacktestConfig,
        optimization_metric: str = "sharpe_ratio",
        n_workers: int = 1,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.strategy_class = strategy_class
        self.data = data
        self.base_config = base_config
        self.optimization_metric = optimization_metric
        self.n_workers = n_workers
        self.checkpoint_dir = checkpoint_dir
        
        self._results: list[dict[str, Any]] = []
        self._best_score = float('-inf')
        self._best_params = {}
        self._best_result = None
        self._convergence_data: list[dict[str, Any]] = []
        self._stop_flag = threading.Event()
        self._iteration_count = 0
        self._result_callback: Callable | None = None
    
    def stop(self):
        """停止优化"""
        self._stop_flag.set()
    
    def is_stopped(self) -> bool:
        """检查是否已停止"""
        return self._stop_flag.is_set()
    
    def save_checkpoint(self, path: str | None = None) -> str:
        """保存检查点"""
        if path is None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.checkpoint_dir, f"opt_checkpoint_{timestamp}.json")
        
        checkpoint = {
            "results": self._results,
            "best_score": self._best_score,
            "best_params": self._best_params,
            "iteration_count": self._iteration_count,
            "optimization_metric": self.optimization_metric,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(path, 'w') as f:
            json.dump(checkpoint, f, default=str)
        
        return path
    
    def load_checkpoint(self, path: str) -> bool:
        """加载检查点"""
        try:
            with open(path, 'r') as f:
                checkpoint = json.load(f)
            
            self._results = checkpoint.get("results", [])
            self._best_score = checkpoint.get("best_score", float('-inf'))
            self._best_params = checkpoint.get("best_params", {})
            self._iteration_count = checkpoint.get("iteration_count", 0)
            
            logger.info(f"加载检查点: {len(self._results)} 条结果, 最优得分: {self._best_score}")
            return True
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return False
    
    def single_parameter_search(
        self,
        target_param: ParameterRange,
        fixed_params: dict[str, Any],
        progress_callback: Callable[[int, int], None] | None = None,
        result_callback: Callable[[dict], None] | None = None,
    ) -> OptimizationResult:
        """单一参数最优探索"""
        start_time = datetime.now()
        self._result_callback = result_callback
        self._stop_flag.clear()
        
        values = target_param.get_values()
        total = len(values)
        
        logger.info(f"单一参数探索: {target_param.name}, 范围: {values}")
        
        self._results = []
        self._best_score = float('-inf')
        self._best_params = {}
        self._convergence_data = []
        
        for i, value in enumerate(values):
            if self._stop_flag.is_set():
                break
            
            params = fixed_params.copy()
            params[target_param.name] = value
            
            result = self._evaluate_params(params)
            self._results.append(result)
            
            if result["score"] > self._best_score:
                self._best_score = result["score"]
                self._best_params = params.copy()
                self._best_result = result.get("backtest_result")
            
            self._convergence_data.append({
                "iteration": i + 1,
                "best_score": self._best_score,
                "current_score": result["score"],
                "param_value": value,
            })
            
            if result_callback:
                result_callback(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        param_importance = self._calculate_parameter_importance()
        
        return OptimizationResult(
            best_params=self._best_params,
            best_score=self._best_score,
            best_result=self._best_result,
            all_results=self._results,
            optimization_method="single_parameter_search",
            total_iterations=len(self._results),
            execution_time=execution_time,
            parameter_importance=param_importance,
            convergence_data=self._convergence_data,
        )
    
    def grid_search(
        self,
        param_ranges: list[ParameterRange],
        max_iterations: int = 1000,
        progress_callback: Callable[[int, int], None] | None = None,
        result_callback: Callable[[dict], None] | None = None,
    ) -> OptimizationResult:
        """网格搜索优化"""
        start_time = datetime.now()
        self._result_callback = result_callback
        self._stop_flag.clear()
        
        all_values = [pr.get_values() for pr in param_ranges]
        total_combinations = 1
        for values in all_values:
            total_combinations *= len(values)
        
        total = min(total_combinations, max_iterations)
        
        if total_combinations > 10000:
            logger.warning(f"网格搜索组合数过大: {total_combinations}, 建议使用随机搜索")
        
        logger.info(f"网格搜索: 总组合数={total_combinations}, 执行={total}")
        
        self._results = []
        self._best_score = float('-inf')
        self._best_params = {}
        self._convergence_data = []
        
        param_names = [pr.name for pr in param_ranges]
        
        iteration = 0
        for combo in itertools.product(*all_values):
            if iteration >= max_iterations:
                break
            if self._stop_flag.is_set():
                break
            
            params = dict(zip(param_names, combo))
            
            result = self._evaluate_params(params)
            self._results.append(result)
            
            if result["score"] > self._best_score:
                self._best_score = result["score"]
                self._best_params = params.copy()
                self._best_result = result.get("backtest_result")
            
            self._convergence_data.append({
                "iteration": iteration + 1,
                "best_score": self._best_score,
                "current_score": result["score"],
            })
            
            if result_callback:
                result_callback(result)
            
            if progress_callback:
                progress_callback(iteration + 1, total)
            
            iteration += 1
        
        execution_time = (datetime.now() - start_time).total_seconds()
        param_importance = self._calculate_parameter_importance()
        
        return OptimizationResult(
            best_params=self._best_params,
            best_score=self._best_score,
            best_result=self._best_result,
            all_results=self._results,
            optimization_method="grid_search",
            total_iterations=len(self._results),
            execution_time=execution_time,
            parameter_importance=param_importance,
            convergence_data=self._convergence_data,
        )
    
    def random_search(
        self,
        param_ranges: list[ParameterRange],
        n_iterations: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
        result_callback: Callable[[dict], None] | None = None,
    ) -> OptimizationResult:
        """随机搜索优化"""
        start_time = datetime.now()
        self._result_callback = result_callback
        self._stop_flag.clear()
        
        logger.info(f"随机搜索: 迭代次数={n_iterations}")
        
        self._results = []
        self._best_score = float('-inf')
        self._best_params = {}
        self._convergence_data = []
        
        for i in range(n_iterations):
            if self._stop_flag.is_set():
                break
            
            params = {pr.name: pr.get_random_value() for pr in param_ranges}
            
            result = self._evaluate_params(params)
            self._results.append(result)
            
            if result["score"] > self._best_score:
                self._best_score = result["score"]
                self._best_params = params.copy()
                self._best_result = result.get("backtest_result")
            
            self._convergence_data.append({
                "iteration": i + 1,
                "best_score": self._best_score,
                "current_score": result["score"],
            })
            
            if result_callback:
                result_callback(result)
            
            if progress_callback:
                progress_callback(i + 1, n_iterations)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        param_importance = self._calculate_parameter_importance()
        
        return OptimizationResult(
            best_params=self._best_params,
            best_score=self._best_score,
            best_result=self._best_result,
            all_results=self._results,
            optimization_method="random_search",
            total_iterations=len(self._results),
            execution_time=execution_time,
            parameter_importance=param_importance,
            convergence_data=self._convergence_data,
        )
    
    def bayesian_optimization(
        self,
        param_ranges: list[ParameterRange],
        n_iterations: int = 50,
        n_initial: int = 10,
        acquisition_type: str = "ei",
        progress_callback: Callable[[int, int], None] | None = None,
        result_callback: Callable[[dict], None] | None = None,
    ) -> OptimizationResult:
        """贝叶斯优化"""
        start_time = datetime.now()
        self._result_callback = result_callback
        self._stop_flag.clear()
        
        logger.info(f"贝叶斯优化: 初始采样={n_initial}, 总迭代={n_iterations}")
        
        self._results = []
        self._best_score = float('-inf')
        self._best_params = {}
        self._convergence_data = []
        
        sampled_points: list[dict[str, float]] = []
        sampled_scores: list[float] = []
        
        for i in range(n_initial):
            if self._stop_flag.is_set():
                break
            
            params = {pr.name: pr.get_random_value() for pr in param_ranges}
            
            result = self._evaluate_params(params)
            self._results.append(result)
            sampled_points.append(params)
            sampled_scores.append(result["score"])
            
            if result["score"] > self._best_score:
                self._best_score = result["score"]
                self._best_params = params.copy()
                self._best_result = result.get("backtest_result")
            
            self._convergence_data.append({
                "iteration": i + 1,
                "best_score": self._best_score,
                "current_score": result["score"],
                "phase": "initial",
            })
            
            if result_callback:
                result_callback(result)
            
            if progress_callback:
                progress_callback(i + 1, n_iterations)
        
        for i in range(n_initial, n_iterations):
            if self._stop_flag.is_set():
                break
            
            next_params = self._acquisition_function(
                param_ranges, sampled_points, sampled_scores, acquisition_type
            )
            
            result = self._evaluate_params(next_params)
            self._results.append(result)
            sampled_points.append(next_params)
            sampled_scores.append(result["score"])
            
            if result["score"] > self._best_score:
                self._best_score = result["score"]
                self._best_params = next_params.copy()
                self._best_result = result.get("backtest_result")
            
            self._convergence_data.append({
                "iteration": i + 1,
                "best_score": self._best_score,
                "current_score": result["score"],
                "phase": "bayesian",
            })
            
            if result_callback:
                result_callback(result)
            
            if progress_callback:
                progress_callback(i + 1, n_iterations)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        param_importance = self._calculate_parameter_importance()
        
        return OptimizationResult(
            best_params=self._best_params,
            best_score=self._best_score,
            best_result=self._best_result,
            all_results=self._results,
            optimization_method="bayesian_optimization",
            total_iterations=len(self._results),
            execution_time=execution_time,
            parameter_importance=param_importance,
            convergence_data=self._convergence_data,
        )
    
    def genetic_algorithm(
        self,
        param_ranges: list[ParameterRange],
        n_generations: int = 20,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism: int = 2,
        progress_callback: Callable[[int, int], None] | None = None,
        result_callback: Callable[[dict], None] | None = None,
    ) -> OptimizationResult:
        """遗传算法优化"""
        start_time = datetime.now()
        self._result_callback = result_callback
        self._stop_flag.clear()
        
        logger.info(f"遗传算法: 种群={population_size}, 代数={n_generations}")
        
        self._results = []
        self._best_score = float('-inf')
        self._best_params = {}
        self._convergence_data = []
        
        population = []
        for _ in range(population_size):
            individual = {pr.name: pr.get_random_value() for pr in param_ranges}
            population.append(individual)
        
        total_iterations = n_generations * population_size
        current_iteration = 0
        
        for gen in range(n_generations):
            if self._stop_flag.is_set():
                break
            
            fitness_scores = []
            for individual in population:
                result = self._evaluate_params(individual)
                self._results.append(result)
                fitness_scores.append(result["score"])
                
                if result["score"] > self._best_score:
                    self._best_score = result["score"]
                    self._best_params = individual.copy()
                    self._best_result = result.get("backtest_result")
                
                if result_callback:
                    result_callback(result)
                
                current_iteration += 1
                if progress_callback:
                    progress_callback(current_iteration, total_iterations)
            
            self._convergence_data.append({
                "iteration": current_iteration,
                "generation": gen + 1,
                "best_score": self._best_score,
                "avg_score": np.mean(fitness_scores),
                "best_in_gen": max(fitness_scores),
            })
            
            sorted_indices = np.argsort(fitness_scores)[::-1]
            new_population = []
            
            for i in range(elitism):
                new_population.append(population[sorted_indices[i]].copy())
            
            while len(new_population) < population_size:
                parent1 = self._tournament_select(population, fitness_scores)
                parent2 = self._tournament_select(population, fitness_scores)
                
                if random.random() < crossover_rate:
                    child = self._crossover(parent1, parent2, param_ranges)
                else:
                    child = parent1.copy()
                
                if random.random() < mutation_rate:
                    child = self._mutate(child, param_ranges)
                
                new_population.append(child)
            
            population = new_population
        
        execution_time = (datetime.now() - start_time).total_seconds()
        param_importance = self._calculate_parameter_importance()
        
        return OptimizationResult(
            best_params=self._best_params,
            best_score=self._best_score,
            best_result=self._best_result,
            all_results=self._results,
            optimization_method="genetic_algorithm",
            total_iterations=len(self._results),
            execution_time=execution_time,
            parameter_importance=param_importance,
            convergence_data=self._convergence_data,
        )
    
    def simulated_annealing(
        self,
        param_ranges: list[ParameterRange],
        n_iterations: int = 100,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        progress_callback: Callable[[int, int], None] | None = None,
        result_callback: Callable[[dict], None] | None = None,
    ) -> OptimizationResult:
        """模拟退火优化
        
        模拟金属退火过程，通过逐渐降低"温度"来减少接受劣解的概率。
        适合解决复杂的非凸优化问题。
        """
        start_time = datetime.now()
        self._result_callback = result_callback
        self._stop_flag.clear()
        
        logger.info(f"模拟退火: 迭代次数={n_iterations}, 初始温度={initial_temp}, 冷却率={cooling_rate}")
        
        self._results = []
        self._best_score = float('-inf')
        self._best_params = {}
        self._convergence_data = []
        
        current_params = {pr.name: pr.get_random_value() for pr in param_ranges}
        current_result = self._evaluate_params(current_params)
        current_score = current_result["score"]
        self._results.append(current_result)
        
        if current_score > self._best_score:
            self._best_score = current_score
            self._best_params = current_params.copy()
            self._best_result = current_result.get("backtest_result")
        
        temperature = initial_temp
        
        for i in range(n_iterations):
            if self._stop_flag.is_set():
                break
            
            neighbor_params = {}
            for pr in param_ranges:
                if pr.values is not None:
                    neighbor_params[pr.name] = random.choice(pr.values)
                else:
                    mutation = random.gauss(0, (pr.max_value - pr.min_value) * 0.1)
                    new_value = current_params[pr.name] + mutation
                    neighbor_params[pr.name] = max(pr.min_value, min(pr.max_value, new_value))
            
            result = self._evaluate_params(neighbor_params)
            self._results.append(result)
            neighbor_score = result["score"]
            
            if neighbor_score > self._best_score:
                self._best_score = neighbor_score
                self._best_params = neighbor_params.copy()
                self._best_result = result.get("backtest_result")
            
            delta = neighbor_score - current_score
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 0.01)):
                current_params = neighbor_params
                current_score = neighbor_score
            
            self._convergence_data.append({
                "iteration": i + 1,
                "best_score": self._best_score,
                "current_score": current_score,
                "temperature": temperature,
            })
            
            temperature *= cooling_rate
            
            if result_callback:
                result_callback(result)
            
            if progress_callback:
                progress_callback(i + 1, n_iterations)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        param_importance = self._calculate_parameter_importance()
        
        return OptimizationResult(
            best_params=self._best_params,
            best_score=self._best_score,
            best_result=self._best_result,
            all_results=self._results,
            optimization_method="simulated_annealing",
            total_iterations=len(self._results),
            execution_time=execution_time,
            parameter_importance=param_importance,
            convergence_data=self._convergence_data,
        )
    
    def particle_swarm_optimization(
        self,
        param_ranges: list[ParameterRange],
        n_particles: int = 20,
        n_iterations: int = 50,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        progress_callback: Callable[[int, int], None] | None = None,
        result_callback: Callable[[dict], None] | None = None,
    ) -> OptimizationResult:
        """粒子群优化
        
        模拟鸟群觅食行为，每个粒子根据自己的历史最优和群体最优调整位置。
        适合连续参数空间的优化问题。
        """
        start_time = datetime.now()
        self._result_callback = result_callback
        self._stop_flag.clear()
        
        logger.info(f"粒子群优化: 粒子数={n_particles}, 迭代次数={n_iterations}")
        
        self._results = []
        self._best_score = float('-inf')
        self._best_params = {}
        self._convergence_data = []
        
        particles = []
        velocities = []
        personal_best = []
        personal_best_scores = []
        
        for _ in range(n_particles):
            particle = {pr.name: pr.get_random_value() for pr in param_ranges}
            particles.append(particle)
            
            velocity = {}
            for pr in param_ranges:
                if pr.values is None:
                    velocity[pr.name] = random.uniform(
                        -(pr.max_value - pr.min_value) * 0.1,
                        (pr.max_value - pr.min_value) * 0.1
                    )
                else:
                    velocity[pr.name] = 0
            velocities.append(velocity)
            
            personal_best.append(particle.copy())
            personal_best_scores.append(float('-inf'))
        
        global_best = None
        global_best_score = float('-inf')
        
        total_iterations = n_particles * n_iterations
        current_iteration = 0
        
        for iteration in range(n_iterations):
            if self._stop_flag.is_set():
                break
            
            for i, particle in enumerate(particles):
                result = self._evaluate_params(particle)
                self._results.append(result)
                score = result["score"]
                
                if score > self._best_score:
                    self._best_score = score
                    self._best_params = particle.copy()
                    self._best_result = result.get("backtest_result")
                
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best[i] = particle.copy()
                
                if score > global_best_score:
                    global_best_score = score
                    global_best = particle.copy()
                
                if result_callback:
                    result_callback(result)
                
                current_iteration += 1
                if progress_callback:
                    progress_callback(current_iteration, total_iterations)
            
            self._convergence_data.append({
                "iteration": iteration + 1,
                "best_score": self._best_score,
                "global_best_score": global_best_score,
            })
            
            for i, particle in enumerate(particles):
                for pr in param_ranges:
                    if pr.values is not None:
                        continue
                    
                    r1, r2 = random.random(), random.random()
                    
                    velocities[i][pr.name] = (
                        w * velocities[i][pr.name] +
                        c1 * r1 * (personal_best[i][pr.name] - particle[pr.name]) +
                        c2 * r2 * (global_best[pr.name] - particle[pr.name])
                    )
                    
                    particles[i][pr.name] += velocities[i][pr.name]
                    particles[i][pr.name] = max(pr.min_value, min(pr.max_value, particles[i][pr.name]))
        
        execution_time = (datetime.now() - start_time).total_seconds()
        param_importance = self._calculate_parameter_importance()
        
        return OptimizationResult(
            best_params=self._best_params,
            best_score=self._best_score,
            best_result=self._best_result,
            all_results=self._results,
            optimization_method="particle_swarm_optimization",
            total_iterations=len(self._results),
            execution_time=execution_time,
            parameter_importance=param_importance,
            convergence_data=self._convergence_data,
        )
    
    def reinforcement_learning_optimize(
        self,
        param_ranges: list[ParameterRange],
        n_episodes: int = 100,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.3,
        progress_callback: Callable[[int, int], None] | None = None,
        result_callback: Callable[[dict], None] | None = None,
    ) -> OptimizationResult:
        """强化学习优化（简化版Q-Learning）
        
        使用Q-Learning算法学习最优参数选择策略。
        将参数空间离散化为状态，通过探索和利用找到最优参数。
        """
        start_time = datetime.now()
        self._result_callback = result_callback
        self._stop_flag.clear()
        
        logger.info(f"强化学习优化: 回合数={n_episodes}, 学习率={learning_rate}")
        
        self._results = []
        self._best_score = float('-inf')
        self._best_params = {}
        self._convergence_data = []
        
        n_bins = 10
        q_tables = {}
        for pr in param_ranges:
            if pr.values is None:
                q_tables[pr.name] = [0.0] * n_bins
        
        def get_bin_index(value, pr):
            if pr.values is not None:
                return 0
            if pr.max_value == pr.min_value:
                return 0
            normalized = (value - pr.min_value) / (pr.max_value - pr.min_value)
            return min(max(int(normalized * n_bins), 0), n_bins - 1)
        
        def get_value_from_bin(bin_idx, pr):
            if pr.values is not None:
                return pr.values[bin_idx % len(pr.values)]
            if pr.max_value == pr.min_value:
                return pr.min_value
            return pr.min_value + (bin_idx + 0.5) * (pr.max_value - pr.min_value) / n_bins
        
        current_params = {pr.name: pr.get_random_value() for pr in param_ranges}
        
        for episode in range(n_episodes):
            if self._stop_flag.is_set():
                break
            
            action = {}
            for pr in param_ranges:
                if pr.values is not None:
                    action[pr.name] = random.choice(pr.values)
                else:
                    if random.random() < epsilon:
                        action[pr.name] = pr.get_random_value()
                    else:
                        bin_idx = q_tables[pr.name].index(max(q_tables[pr.name]))
                        action[pr.name] = get_value_from_bin(bin_idx, pr)
            
            result = self._evaluate_params(action)
            self._results.append(result)
            reward = result["score"]
            
            if reward > self._best_score:
                self._best_score = reward
                self._best_params = action.copy()
                self._best_result = result.get("backtest_result")
            
            for pr in param_ranges:
                if pr.values is None:
                    bin_idx = get_bin_index(action[pr.name], pr)
                    old_q = q_tables[pr.name][bin_idx]
                    max_next_q = max(q_tables[pr.name])
                    q_tables[pr.name][bin_idx] = old_q + learning_rate * (
                        reward + discount_factor * max_next_q - old_q
                    )
            
            self._convergence_data.append({
                "episode": episode + 1,
                "best_score": self._best_score,
                "reward": reward,
                "epsilon": epsilon,
            })
            
            epsilon = max(0.01, epsilon * 0.99)
            
            if result_callback:
                result_callback(result)
            
            if progress_callback:
                progress_callback(episode + 1, n_episodes)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        param_importance = self._calculate_parameter_importance()
        
        return OptimizationResult(
            best_params=self._best_params,
            best_score=self._best_score,
            best_result=self._best_result,
            all_results=self._results,
            optimization_method="reinforcement_learning",
            total_iterations=len(self._results),
            execution_time=execution_time,
            parameter_importance=param_importance,
            convergence_data=self._convergence_data,
        )
    
    def _acquisition_function(
        self,
        param_ranges: list[ParameterRange],
        sampled_points: list[dict[str, float]],
        sampled_scores: list[float],
        acquisition_type: str = "ei",
    ) -> dict[str, float]:
        """采集函数"""
        best_score = max(sampled_scores)
        best_idx = sampled_scores.index(best_score)
        best_point = sampled_points[best_idx]
        
        next_point = {}
        
        for pr in param_ranges:
            if pr.values is not None:
                candidates = [v for v in pr.values]
                if acquisition_type == "ei":
                    weights = []
                    for v in candidates:
                        min_dist = 1.0
                        for p in sampled_points:
                            if p.get(pr.name) != v:
                                min_dist += 1.0
                        improvement = max(0, best_score - min(sampled_scores)) if sampled_scores else 1.0
                        if not sampled_scores or best_score == float('-inf') or best_score == float('inf'):
                            improvement = 1.0
                        ei = improvement * min_dist
                        weights.append(max(0.1, ei))
                    
                    total_weight = sum(weights)
                    if total_weight <= 0 or not np.isfinite(total_weight):
                        weights = [1.0] * len(candidates)
                        total_weight = len(candidates)
                    weights = [w / total_weight for w in weights]
                    next_point[pr.name] = random.choices(candidates, weights=weights)[0]
                else:
                    next_point[pr.name] = random.choice(candidates)
            else:
                if acquisition_type == "ei":
                    exploration = random.uniform(-1, 1) * (pr.max_value - pr.min_value) * 0.3
                else:
                    exploration = random.gauss(0, (pr.max_value - pr.min_value) * 0.1)
                
                new_value = best_point.get(pr.name, (pr.min_value + pr.max_value) / 2) + exploration
                next_point[pr.name] = max(pr.min_value, min(pr.max_value, new_value))
        
        return next_point
    
    def _tournament_select(
        self,
        population: list[dict],
        fitness_scores: list[float],
        tournament_size: int = 3,
    ) -> dict:
        """锦标赛选择"""
        indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()
    
    def _crossover(
        self,
        parent1: dict,
        parent2: dict,
        param_ranges: list[ParameterRange],
    ) -> dict:
        """交叉操作"""
        child = {}
        for pr in param_ranges:
            if random.random() < 0.5:
                child[pr.name] = parent1.get(pr.name)
            else:
                child[pr.name] = parent2.get(pr.name)
        return child
    
    def _mutate(
        self,
        individual: dict,
        param_ranges: list[ParameterRange],
    ) -> dict:
        """变异操作"""
        mutated = individual.copy()
        pr = random.choice(param_ranges)
        
        if pr.values is not None:
            mutated[pr.name] = random.choice(pr.values)
        else:
            mutation = random.gauss(0, (pr.max_value - pr.min_value) * 0.1)
            new_value = mutated[pr.name] + mutation
            mutated[pr.name] = max(pr.min_value, min(pr.max_value, new_value))
        
        return mutated
    
    def _evaluate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """评估参数组合"""
        try:
            if not self._validate_params(params):
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
            
            strategy_params = {}
            risk_params = {}
            
            for key, value in params.items():
                if key in ["leverage", "stop_loss_pct", "take_profit_pct", "position_size"]:
                    risk_params[key] = value
                else:
                    strategy_params[key] = value

            strategy = self.strategy_class(strategy_params)

            config = BacktestConfig(
                symbol=self.base_config.symbol,
                interval=self.base_config.interval,
                initial_capital=self.base_config.initial_capital,
                leverage=int(risk_params.get("leverage", self.base_config.leverage)),
                position_size=risk_params.get("position_size", self.base_config.position_size),
                stop_loss_pct=risk_params.get("stop_loss_pct", self.base_config.stop_loss_pct),
                take_profit_pct=risk_params.get("take_profit_pct", self.base_config.take_profit_pct),
                commission_rate=0.0004,  # 固定手续费率（开仓时收取）
                slippage=0.0001,  # 固定滑点率
            )
            
            engine = BacktestEngine(strategy, config)
            result = engine.run(self.data)
            
            score = self._calculate_score(result)
            
            return {
                "params": params,
                "strategy_params": strategy_params,
                "risk_params": risk_params,
                "score": score,
                "total_return_pct": result.total_return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "win_rate": result.win_rate,
                "sharpe_ratio": result.sharpe_ratio,
                "total_trades": result.total_trades,
                "stop_loss_hits": result.stop_loss_hits,
                "take_profit_hits": result.take_profit_hits,
                "liquidation_hits": result.liquidation_hits,
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
    
    def _validate_params(self, params: dict[str, Any]) -> bool:
        """验证参数约束"""
        if "fast_period" in params and "slow_period" in params:
            if params["fast_period"] >= params["slow_period"]:
                return False
        
        if "leverage" in params:
            if params["leverage"] < 1:
                return False
        
        return True
    
    def _calculate_score(self, result: BacktestResult) -> float:
        """计算综合得分"""
        if result.total_trades == 0:
            return float('-inf')
        
        min_trades_penalty = 0
        if result.total_trades < 5:
            min_trades_penalty = (5 - result.total_trades) * 0.1
        
        liquidation_penalty = 0
        if hasattr(result, 'liquidation_hits') and result.liquidation_hits > 0:
            liquidation_penalty = result.liquidation_hits * 10
        
        if self.optimization_metric == "sharpe_ratio":
            base_score = result.sharpe_ratio
        elif self.optimization_metric == "total_return":
            base_score = result.total_return_pct
        elif self.optimization_metric == "calmar_ratio":
            base_score = result.calmar_ratio
        elif self.optimization_metric == "composite":
            if result.max_drawdown_pct == 0:
                base_score = result.total_return_pct
            else:
                base_score = result.total_return_pct / result.max_drawdown_pct * result.win_rate / 100
        else:
            base_score = result.sharpe_ratio
        
        return base_score - min_trades_penalty - liquidation_penalty
    
    def _calculate_parameter_importance(self) -> dict[str, float]:
        """计算参数重要性"""
        if len(self._results) < 10:
            return {}
        
        df = pd.DataFrame(self._results)
        
        if "score" not in df.columns:
            return {}
        
        param_names = list(self._best_params.keys())
        importance = {}
        
        for param in param_names:
            if param not in df.columns:
                continue
            
            try:
                valid_data = df[[param, "score"]].dropna()
                if len(valid_data) > 5:
                    correlation = valid_data[param].corr(valid_data["score"])
                    importance[param] = abs(correlation) if not pd.isna(correlation) else 0
            except Exception:
                importance[param] = 0
        
        return importance


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
                category="strategy",
                display_name=param.display_name,
            ))
        elif param.value_type == float:
            min_val = param.min_value if param.min_value is not None else 0
            max_val = param.max_value if param.max_value is not None else param.default_value * 2
            raw_step = (max_val - min_val) / 10 if max_val is not None and min_val is not None else 0.1
            step = max(1e-6, round(float(raw_step), 6))
            ranges.append(ParameterRange(
                name=param.name,
                min_value=min_val,
                max_value=max_val,
                step=step,
                category="strategy",
                display_name=param.display_name,
            ))
        elif param.options is not None:
            ranges.append(ParameterRange(
                name=param.name,
                min_value=0,
                max_value=0,
                values=param.options,
                category="strategy",
                display_name=param.display_name,
            ))
    
    return ranges


def get_all_optimizable_params(
    strategy: BaseStrategy,
    include_risk_params: bool = True,
) -> dict[str, list[ParameterRange]]:
    """获取所有可优化参数"""
    result = {
        "strategy": get_parameter_ranges_from_strategy(strategy),
        "risk": [],
    }

    if include_risk_params:
        result["risk"] = [
            ParameterRange(
                name="leverage",
                min_value=1,
                max_value=20,
                step=1,
                category="risk",
                display_name="杠杆倍数",
            ),
            ParameterRange(
                name="stop_loss_pct",
                min_value=0,
                max_value=20,
                step=0.5,
                category="risk",
                display_name="止损率(%)",
            ),
            ParameterRange(
                name="take_profit_pct",
                min_value=0,
                max_value=50,
                step=2.5,
                category="risk",
                display_name="止盈率(%)",
            ),
            ParameterRange(
                name="position_size",
                min_value=0.05,
                max_value=0.5,
                step=0.05,
                category="risk",
                display_name="仓位比例",
            ),
            # 手续费和滑点使用固定值，不在参数列表中显示
            # commission_rate: 固定 0.0004 (开仓时收取)
            # slippage: 固定 0.0001
        ]

    return result


def get_default_param_ranges(
    strategy: BaseStrategy,
    include_risk: bool = True,
) -> list[ParameterRange]:
    """获取默认参数范围列表"""
    all_params = get_all_optimizable_params(strategy, include_risk)
    return all_params["strategy"] + all_params["risk"]
