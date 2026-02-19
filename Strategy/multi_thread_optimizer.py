"""多线程参数优化器"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
import threading
import queue
import time

import pandas as pd
import numpy as np

from Strategy.base import BaseStrategy
from Strategy.parameter_optimizer import ParameterOptimizer, ParameterRange, OptimizationResult
from backtest.engine import BacktestEngine
from core.config import BacktestConfig
from app_logger.logger_setup import Logger

logger = Logger(__name__)


class MultiThreadOptimizer(ParameterOptimizer):
    """多线程参数优化器"""
    
    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        data: pd.DataFrame,
        base_config: BacktestConfig,
        optimization_metric: str = "sharpe_ratio",
        n_workers: int = 4,
    ):
        super().__init__(strategy_class, data, base_config, optimization_metric)
        self.n_workers = n_workers
        self._stop_flag = threading.Event()
        self._result_queue = queue.Queue()
        self._progress_lock = threading.Lock()
        self._completed_count = 0
    
    def stop(self):
        """停止优化"""
        self._stop_flag.set()
    
    def _worker(
        self,
        worker_id: int,
        param_combinations: list[dict],
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        """工作线程"""
        local_results = []
        
        for i, params in enumerate(param_combinations):
            if self._stop_flag.is_set():
                break
            
            try:
                result = self._evaluate_params(params)
                local_results.append(result)
                
                with self._progress_lock:
                    self._completed_count += 1
                    if progress_callback:
                        progress_callback(self._completed_count, len(param_combinations) * self.n_workers)
                        
            except Exception as e:
                logger.warning(f"Worker {worker_id} 评估失败: {e}")
        
        self._result_queue.put(local_results)
    
    def parallel_random_search(
        self,
        param_ranges: list[ParameterRange],
        n_iterations: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
        result_callback: Callable[[dict], None] | None = None,
    ) -> OptimizationResult:
        """并行随机搜索"""
        start_time = datetime.now()
        self._stop_flag.clear()
        
        all_combinations = []
        for _ in range(n_iterations):
            params = {pr.name: pr.get_random_value() for pr in param_ranges}
            all_combinations.append(params)
        
        chunk_size = (len(all_combinations) + self.n_workers - 1) // self.n_workers
        chunks = [
            all_combinations[i:i + chunk_size]
            for i in range(0, len(all_combinations), chunk_size)
        ]
        
        self._completed_count = 0
        threads = []
        
        for i, chunk in enumerate(chunks):
            t = threading.Thread(
                target=self._worker,
                args=(i, chunk, progress_callback),
            )
            threads.append(t)
            t.start()
        
        all_results = []
        for t in threads:
            t.join()
        
        while not self._result_queue.empty():
            all_results.extend(self._result_queue.get())
        
        best_score = float('-inf')
        best_params = {}
        
        for result in all_results:
            if result_callback:
                result_callback(result)
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
        self._stop_flag.clear()
        
        all_values = [pr.get_values() for pr in param_ranges]
        param_names = [pr.name for pr in param_ranges]
        
        all_combinations = []
        for combo in itertools.product(*all_values):
            if len(all_combinations) >= max_iterations:
                break
            params = dict(zip(param_names, combo))
            all_combinations.append(params)
        
        chunk_size = (len(all_combinations) + self.n_workers - 1) // self.n_workers
        chunks = [
            all_combinations[i:i + chunk_size]
            for i in range(0, len(all_combinations), chunk_size)
        ]
        
        self._completed_count = 0
        threads = []
        
        for i, chunk in enumerate(chunks):
            t = threading.Thread(
                target=self._worker,
                args=(i, chunk, progress_callback),
            )
            threads.append(t)
            t.start()
        
        all_results = []
        for t in threads:
            t.join()
        
        while not self._result_queue.empty():
            all_results.extend(self._result_queue.get())
        
        best_score = float('-inf')
        best_params = {}
        
        for result in all_results:
            if result_callback:
                result_callback(result)
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


def benchmark_optimization():
    """测试多线程优化性能"""
    import numpy as np
    
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
    
    print("\n测试多线程优化 (4线程)...")
    optimizer2 = MultiThreadOptimizer(
        strategy_class=type(strategy),
        data=data,
        base_config=config,
        n_workers=4,
    )
    start = time.time()
    result2 = optimizer2.parallel_random_search(param_ranges, n_iterations=50)
    multi_time = time.time() - start
    print(f"  耗时: {multi_time:.2f}秒")
    
    print(f"\n加速比: {single_time/multi_time:.2f}x")


if __name__ == "__main__":
    benchmark_optimization()
