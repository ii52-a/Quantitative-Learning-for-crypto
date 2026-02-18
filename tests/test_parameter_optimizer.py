"""
参数优化器测试
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from Strategy.parameter_optimizer import (
    ParameterOptimizer,
    ParameterRange,
    OptimizationResult,
    MultiObjectiveOptimizer,
    get_parameter_ranges_from_strategy,
)
from Strategy.multi_indicator_strategy import MultiIndicatorStrategy
from core.config import BacktestConfig


class TestParameterRange:
    """参数范围测试"""
    
    def test_get_values_with_step(self):
        pr = ParameterRange("fast_period", 5, 20, 5)
        values = pr.get_values()
        assert values == [5, 10, 15, 20]
    
    def test_get_values_with_float_step(self):
        pr = ParameterRange("weight", 0.0, 1.0, 0.25)
        values = pr.get_values()
        assert len(values) == 5
        assert values[0] == 0.0
        assert values[-1] == 1.0
    
    def test_get_values_with_explicit_values(self):
        pr = ParameterRange("type", 0, 0, values=["A", "B", "C"])
        values = pr.get_values()
        assert values == ["A", "B", "C"]
    
    def test_get_values_zero_step(self):
        pr = ParameterRange("const", 10, 20, 0)
        values = pr.get_values()
        assert values == [10]


class TestOptimizationResult:
    """优化结果测试"""
    
    def test_to_dict(self):
        result = OptimizationResult(
            best_params={"fast": 10, "slow": 30},
            best_score=1.5,
            best_result=None,
            all_results=[
                {"params": {"fast": 10, "slow": 30}, "score": 1.5, "total_return_pct": 10.0, "max_drawdown_pct": 5.0, "win_rate": 60.0},
                {"params": {"fast": 5, "slow": 20}, "score": 1.2, "total_return_pct": 8.0, "max_drawdown_pct": 6.0, "win_rate": 55.0},
            ],
            optimization_method="random_search",
            total_iterations=10,
            execution_time=5.5,
        )
        
        d = result.to_dict()
        
        assert d["best_params"] == {"fast": 10, "slow": 30}
        assert d["best_score"] == 1.5
        assert d["total_iterations"] == 10
        assert len(d["top_results"]) == 2


class TestParameterOptimizer:
    """参数优化器测试"""
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='30min')
        data = pd.DataFrame({
            'open': np.random.uniform(49000, 51000, 100),
            'high': np.random.uniform(50000, 52000, 100),
            'low': np.random.uniform(48000, 50000, 100),
            'close': np.random.uniform(49000, 51000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }, index=dates)
        return data
    
    @pytest.fixture
    def config(self):
        return BacktestConfig(
            symbol="BTCUSDT",
            interval="30min",
            initial_capital=10000,
            leverage=1,
        )
    
    def test_optimizer_creation(self, sample_data, config):
        optimizer = ParameterOptimizer(
            strategy_class=MultiIndicatorStrategy,
            data=sample_data,
            base_config=config,
        )
        assert optimizer.optimization_metric == "sharpe_ratio"
    
    def test_random_search(self, sample_data, config):
        optimizer = ParameterOptimizer(
            strategy_class=MultiIndicatorStrategy,
            data=sample_data,
            base_config=config,
        )
        
        param_ranges = [
            ParameterRange("vote_threshold", 1.0, 3.0, 1.0),
        ]
        
        result = optimizer.random_search(
            param_ranges=param_ranges,
            n_iterations=3,
        )
        
        assert result.total_iterations == 3
        assert result.optimization_method == "random_search"
        assert len(result.all_results) == 3
    
    def test_grid_search(self, sample_data, config):
        optimizer = ParameterOptimizer(
            strategy_class=MultiIndicatorStrategy,
            data=sample_data,
            base_config=config,
        )
        
        param_ranges = [
            ParameterRange("vote_threshold", 1.0, 2.0, 1.0),
        ]
        
        result = optimizer.grid_search(
            param_ranges=param_ranges,
            max_iterations=2,
        )
        
        assert result.total_iterations == 2
        assert result.optimization_method == "grid_search"
    
    def test_bayesian_optimization(self, sample_data, config):
        optimizer = ParameterOptimizer(
            strategy_class=MultiIndicatorStrategy,
            data=sample_data,
            base_config=config,
        )
        
        param_ranges = [
            ParameterRange("vote_threshold", 1.0, 3.0, 0.5),
        ]
        
        result = optimizer.bayesian_optimization(
            param_ranges=param_ranges,
            n_iterations=5,
            n_initial=3,
        )
        
        assert result.total_iterations == 5
        assert result.optimization_method == "bayesian_optimization"
    
    def test_calculate_score_sharpe(self, sample_data, config):
        from backtest.engine import BacktestResult
        
        optimizer = ParameterOptimizer(
            strategy_class=MultiIndicatorStrategy,
            data=sample_data,
            base_config=config,
            optimization_metric="sharpe_ratio",
        )
        
        result = BacktestResult(
            initial_capital=10000,
            final_capital=12000,
            total_return_pct=20.0,
            sharpe_ratio=1.5,
            max_drawdown_pct=10.0,
            win_rate=60.0,
        )
        
        score = optimizer._calculate_score(result)
        assert score == 1.5
    
    def test_calculate_score_composite(self, sample_data, config):
        from backtest.engine import BacktestResult
        
        optimizer = ParameterOptimizer(
            strategy_class=MultiIndicatorStrategy,
            data=sample_data,
            base_config=config,
            optimization_metric="composite",
        )
        
        result = BacktestResult(
            initial_capital=10000,
            final_capital=12000,
            total_return_pct=20.0,
            sharpe_ratio=1.5,
            max_drawdown_pct=10.0,
            win_rate=60.0,
        )
        
        score = optimizer._calculate_score(result)
        assert score > 0


class TestMultiObjectiveOptimizer:
    """多目标优化器测试"""
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2024-01-01', periods=50, freq='30min')
        data = pd.DataFrame({
            'open': np.random.uniform(49000, 51000, 50),
            'high': np.random.uniform(50000, 52000, 50),
            'low': np.random.uniform(48000, 50000, 50),
            'close': np.random.uniform(49000, 51000, 50),
            'volume': np.random.uniform(100, 1000, 50)
        }, index=dates)
        return data
    
    def test_pareto_optimize(self, sample_data):
        config = BacktestConfig(symbol="BTCUSDT", interval="30min")
        
        optimizer = MultiObjectiveOptimizer(
            strategy_class=MultiIndicatorStrategy,
            data=sample_data,
            base_config=config,
        )
        
        param_ranges = [
            ParameterRange("vote_threshold", 1.0, 3.0, 1.0),
        ]
        
        result = optimizer.pareto_optimize(
            param_ranges=param_ranges,
            n_iterations=5,
        )
        
        assert result.total_iterations == 5
        assert result.optimization_method == "pareto_optimization"


class TestGetParameterRangesFromStrategy:
    """从策略获取参数范围测试"""
    
    def test_get_ranges(self):
        strategy = MultiIndicatorStrategy()
        ranges = get_parameter_ranges_from_strategy(strategy)
        
        assert len(ranges) > 0
        
        range_names = [r.name for r in ranges]
        assert "vote_threshold" in range_names
