"""多进程参数优化器"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
import multiprocessing as mp
import time
import os

import pandas as pd
import numpy as np

from Strategy.base import BaseStrategy
from Strategy.parameter_optimizer import ParameterOptimizer, ParameterRange, OptimizationResult
from backtest.engine import BacktestEngine
from core.config import BacktestConfig
from app_logger.logger_setup import Logger

logger = Logger(__name__)


def _evaluate_single_param(args):
    """评估单个参数组合（用于多进程）"""
    params, strategy_class, data_dict, config_dict, metric = args
    
    try:
        data = pd.DataFrame(data_dict)
        config = BacktestConfig(**config_dict)
        
        strategy = strategy_class(params)
        engine = BacktestEngine(strategy, config)
        result = engine.run(data)
        
        score = getattr(result, metric, 0)
        
        return {
            "params": params,
            "score": score,
            "total_return_pct": result.total_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
        }
    except Exception as e:
        return {
            "params": params,
            "score": float('-inf'),
            "error": str(e),
        }


class MultiProcessOptimizer(ParameterOptimizer):
    """多进程参数优化器"""
    
    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        data: pd.DataFrame,
        base_config: BacktestConfig,
        optimization_metric: str = "sharpe_ratio",
        n_workers: int | None = None,
    ):
        super().__init__(strategy_class, data, base_config, optimization_metric)
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self._stop_flag = mp.Value('i', 0)
    
    def stop(self):
        """停止优化"""
        self._stop_flag.value = 1
    
    def parallel_random_search(
        self,
        param_ranges: list[ParameterRange],
        n_iterations: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
        result_callback: Callable[[dict], None] | None = None,
    ) -> OptimizationResult:
        """并行随机搜索"""
        start_time = datetime.now()
        self._stop_flag.value = 0
        
        all_combinations = []
        for _ in range(n_iterations):
            params = {pr.name: pr.get_random_value() for pr in param_ranges}
            all_combinations.append(params)
        
        data_dict = self.data.to_dict('list')
        config_dict = {
            'symbol': self.base_config.symbol,
            'interval': self.base_config.interval,
            'initial_capital': self.base_config.initial_capital,
            'data_limit': self.base_config.data_limit,
            'leverage': self.base_config.leverage,
            'stop_loss_pct': self.base_config.stop_loss_pct,
            'take_profit_pct': self.base_config.take_profit_pct,
            'position_size': self.base_config.position_size,
            'commission_rate': self.base_config.commission_rate,
        }
        
        args_list = [
            (params, self.strategy_class, data_dict, config_dict, self.optimization_metric)
            for params in all_combinations
        ]
        
        all_results = []
        completed = 0
        
        with mp.Pool(processes=self.n_workers) as pool:
            for result in pool.imap(_evaluate_single_param, args_list):
                if self._stop_flag.value:
                    pool.terminate()
                    break
                
                all_results.append(result)
                completed += 1
                
                if progress_callback and completed % 5 == 0:
                    progress_callback(completed, n_iterations)
                
                if result_callback:
                    result_callback(result)
        
        best_score = float('-inf')
        best_params = {}
        
        for result in all_results:
            if result["score"] > best_score:
                best_score = result["score"]
                best_params = result["params"].copy()
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_result=None,
            all_results=all_results,
            optimization_method="parallel_random_search",
            total_iterations=len(all_results),
            execution_time=execution_time,
        )
    
    def parallel_grid_search(
        self,
        param_ranges: list[ParameterRange],
        max_iterations: int = 10000,
        progress_callback: Callable[[int, int], None] | None = None,
        result_callback: Callable[[dict], None] | None = None,
    ) -> OptimizationResult:
        """并行网格搜索"""
        import itertools
        
        start_time = datetime.now()
        self._stop_flag.value = 0
        
        all_values = [pr.get_values() for pr in param_ranges]
        param_names = [pr.name for pr in param_ranges]
        
        all_combinations = []
        for combo in itertools.product(*all_values):
            if len(all_combinations) >= max_iterations:
                break
            params = dict(zip(param_names, combo))
            all_combinations.append(params)
        
        data_dict = self.data.to_dict('list')
        config_dict = {
            'symbol': self.base_config.symbol,
            'interval': self.base_config.interval,
            'initial_capital': self.base_config.initial_capital,
            'data_limit': self.base_config.data_limit,
            'leverage': self.base_config.leverage,
            'stop_loss_pct': self.base_config.stop_loss_pct,
            'take_profit_pct': self.base_config.take_profit_pct,
            'position_size': self.base_config.position_size,
            'commission_rate': self.base_config.commission_rate,
        }
        
        args_list = [
            (params, self.strategy_class, data_dict, config_dict, self.optimization_metric)
            for params in all_combinations
        ]
        
        all_results = []
        completed = 0
        
        with mp.Pool(processes=self.n_workers) as pool:
            for result in pool.imap(_evaluate_single_param, args_list):
                if self._stop_flag.value:
                    pool.terminate()
                    break
                
                all_results.append(result)
                completed += 1
                
                if progress_callback and completed % 5 == 0:
                    progress_callback(completed, len(all_combinations))
                
                if result_callback:
                    result_callback(result)
        
        best_score = float('-inf')
        best_params = {}
        
        for result in all_results:
            if result["score"] > best_score:
                best_score = result["score"]
                best_params = result["params"].copy()
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_result=None,
            all_results=all_results,
            optimization_method="parallel_grid_search",
            total_iterations=len(all_results),
            execution_time=execution_time,
        )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    np.random.seed(42)
    n = 500
    dates = pd.date_range(start='2024-01-01', periods=n, freq='30min')
    prices = [100000]
    for i in range(n-1):
        change = np.random.uniform(-0.02, 0.02)
        prices.append(prices[-1] * (1 + change))
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.005 for p in prices],
        'low': [p * 0.995 for p in prices],
        'close': prices,
        'volume': [1000] * n
    }, index=dates)
    
    from Strategy.templates import get_strategy
    
    strategy = get_strategy("MACDStrategy")
    config = BacktestConfig(initial_capital=10000)
    
    param_ranges = [
        ParameterRange("fast_period", 10, 20, 1),
        ParameterRange("slow_period", 25, 35, 1),
    ]
    
    print("测试单线程优化...")
    optimizer1 = ParameterOptimizer(
        strategy_class=type(strategy),
        data=data,
        base_config=config,
    )
    start = time.time()
    result1 = optimizer1.random_search(param_ranges, n_iterations=50)
    single_time = time.time() - start
    print(f"  耗时: {single_time:.2f}秒")
    
    print(f"\n测试多进程优化 ({max(1, mp.cpu_count() - 1)}进程)...")
    optimizer2 = MultiProcessOptimizer(
        strategy_class=type(strategy),
        data=data,
        base_config=config,
    )
    start = time.time()
    result2 = optimizer2.parallel_random_search(param_ranges, n_iterations=50)
    multi_time = time.time() - start
    print(f"  耗时: {multi_time:.2f}秒")
    
    print(f"\n加速比: {single_time/multi_time:.2f}x")
