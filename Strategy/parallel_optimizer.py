"""并行参数优化器 - 一次数据循环评估多组参数"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
import threading

import pandas as pd
import numpy as np

from Strategy.base import BaseStrategy
from backtest.engine import BacktestEngine, BacktestResult
from core.config import BacktestConfig
from Strategy.parameter_optimizer import ParameterRange, OptimizationResult
from app_logger.logger_setup import Logger

logger = Logger(__name__)


@dataclass
class ParallelBacktestState:
    """并行回测状态"""
    params: dict[str, Any]
    position: int = 0
    entry_price: float = 0.0
    equity: float = 0.0
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    total_pnl: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0


class ParallelParameterOptimizer:
    """并行参数优化器 - 一次数据循环评估多组参数"""
    
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
        
        self._stop_flag = threading.Event()
    
    def stop(self):
        """停止优化"""
        self._stop_flag.set()
    
    def evaluate_params_parallel(
        self,
        param_combinations: list[dict[str, Any]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        并行评估多组参数 - 一次数据循环
        
        Args:
            param_combinations: 参数组合列表
            progress_callback: 进度回调
        
        Returns:
            每组参数的评估结果
        """
        start_time = datetime.now()
        self._stop_flag.clear()
        
        n_params = len(param_combinations)
        initial_capital = self.base_config.initial_capital
        leverage = self.base_config.leverage
        position_size = self.base_config.position_size
        stop_loss_pct = self.base_config.stop_loss_pct
        take_profit_pct = self.base_config.take_profit_pct
        commission_rate = self.base_config.commission_rate
        
        strategies = []
        states = []
        
        for params in param_combinations:
            strategy_params = {k: v for k, v in params.items() 
                             if k not in ["leverage", "stop_loss_pct", "take_profit_pct", "position_size"]}
            
            strategy = self.strategy_class(strategy_params)
            strategy.initialize(None)
            strategies.append(strategy)
            
            risk_params = {k: params.get(k, getattr(self.base_config, k, 0)) 
                          for k in ["leverage", "stop_loss_pct", "take_profit_pct", "position_size"]}
            
            state = ParallelBacktestState(
                params=params,
                equity=initial_capital,
            )
            state.risk_params = risk_params
            states.append(state)
        
        data_len = len(self.data)
        
        for idx, (timestamp, row) in enumerate(self.data.iterrows()):
            if self._stop_flag.is_set():
                break
            
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            for i, (strategy, state) in enumerate(zip(strategies, states)):
                if state.position != 0:
                    lev = state.risk_params.get("leverage", leverage)
                    sl = state.risk_params.get("stop_loss_pct", stop_loss_pct)
                    tp = state.risk_params.get("take_profit_pct", take_profit_pct)
                    
                    if state.position > 0:
                        if sl > 0 and low_price <= state.entry_price * (1 - sl/100):
                            pnl = (state.entry_price * (1 - sl/100) - state.entry_price) * state.position_size
                            state.equity += pnl * lev
                            state.position = 0
                            state.total_pnl += pnl * lev
                            if pnl * lev > 0:
                                state.winning_trades += 1
                            else:
                                state.losing_trades += 1
                            continue
                        if tp > 0 and high_price >= state.entry_price * (1 + tp/100):
                            pnl = (state.entry_price * (1 + tp/100) - state.entry_price) * state.position_size
                            state.equity += pnl * lev
                            state.position = 0
                            state.total_pnl += pnl * lev
                            state.winning_trades += 1
                            continue
                
                from core.data_types import Bar
                from core.constants import SignalType
                
                bar = Bar(
                    timestamp=timestamp,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=row.get('volume', 0),
                )
                
                result = strategy.on_bar(bar, None)
                
                if result and result.signal:
                    signal = result.signal
                    
                    if signal.type == SignalType.OPEN_LONG and state.position == 0:
                        state.position = 1
                        state.entry_price = close_price
                        state.position_size = (state.equity * position_size) / close_price
                    elif signal.type == SignalType.CLOSE_LONG and state.position > 0:
                        pnl = (close_price - state.entry_price) * state.position_size
                        commission = state.position_size * close_price * commission_rate
                        state.equity += pnl * leverage - commission
                        state.position = 0
                        state.total_pnl += pnl * leverage
                        if pnl * leverage > 0:
                            state.winning_trades += 1
                        else:
                            state.losing_trades += 1
                
                state.equity_curve.append(state.equity)
            
            if progress_callback and idx % 100 == 0:
                progress_callback(idx, data_len)
        
        results = []
        for state in states:
            total_trades = state.winning_trades + state.losing_trades
            win_rate = (state.winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            equity_series = pd.Series(state.equity_curve)
            returns = equity_series.pct_change().dropna()
            
            sharpe = 0
            if len(returns) > 0 and returns.std() > 0:
                sharpe = (returns.mean() * 252) / (returns.std() * (252 ** 0.5))
            
            peak = equity_series.expanding(min_periods=1).max()
            drawdown = (equity_series - peak) / peak * 100
            max_dd = drawdown.min()
            
            total_return = (state.equity - initial_capital) / initial_capital * 100
            
            score = sharpe
            
            results.append({
                "params": state.params,
                "score": score,
                "total_return_pct": total_return,
                "max_drawdown_pct": max_dd,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "sharpe_ratio": sharpe,
            })
        
        if progress_callback:
            progress_callback(data_len, data_len)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"并行评估完成: {n_params}组参数, 耗时{execution_time:.2f}秒")
        
        return results
    
    def random_search_parallel(
        self,
        param_ranges: list[ParameterRange],
        n_iterations: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
        result_callback: Callable[[dict], None] | None = None,
    ) -> OptimizationResult:
        """并行随机搜索"""
        start_time = datetime.now()
        self._stop_flag.clear()
        
        param_combinations = []
        for _ in range(n_iterations):
            params = {pr.name: pr.get_random_value() for pr in param_ranges}
            param_combinations.append(params)
        
        results = self.evaluate_params_parallel(
            param_combinations,
            progress_callback,
        )
        
        best_score = float('-inf')
        best_params = {}
        all_results = []
        
        for result in results:
            all_results.append(result)
            if result["score"] > best_score:
                best_score = result["score"]
                best_params = result["params"].copy()
            
            if result_callback:
                result_callback(result)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_result=None,
            all_results=all_results,
            optimization_method="parallel_random_search",
            total_iterations=n_iterations,
            execution_time=execution_time,
        )
    
    def grid_search_parallel(
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
        
        param_combinations = []
        for combo in itertools.product(*all_values):
            if len(param_combinations) >= max_iterations:
                break
            params = dict(zip(param_names, combo))
            param_combinations.append(params)
        
        results = self.evaluate_params_parallel(
            param_combinations,
            progress_callback,
        )
        
        best_score = float('-inf')
        best_params = {}
        all_results = []
        
        for result in results:
            all_results.append(result)
            if result["score"] > best_score:
                best_score = result["score"]
                best_params = result["params"].copy()
            
            if result_callback:
                result_callback(result)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_result=None,
            all_results=all_results,
            optimization_method="parallel_grid_search",
            total_iterations=len(param_combinations),
            execution_time=execution_time,
        )
