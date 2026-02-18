"""
策略验证模块测试
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from backtest.validation import (
    ParameterValidator,
    BacktestConsistencyChecker,
    StrategyValidator,
    ValidationResult,
    ConsistencyCheckResult,
)
from Strategy.multi_indicator_strategy import MultiIndicatorStrategy
from core.config import BacktestConfig


class TestValidationResult:
    """验证结果测试"""
    
    def test_valid_result(self):
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_add_error(self):
        result = ValidationResult(is_valid=True)
        result.add_error("测试错误")
        assert result.is_valid is False
        assert "测试错误" in result.errors
    
    def test_add_warning(self):
        result = ValidationResult(is_valid=True)
        result.add_warning("测试警告")
        assert result.is_valid is True
        assert "测试警告" in result.warnings
    
    def test_to_dict(self):
        result = ValidationResult(is_valid=True)
        result.add_warning("警告")
        d = result.to_dict()
        assert d["is_valid"] is True
        assert len(d["warnings"]) == 1


class TestParameterValidator:
    """参数验证器测试"""
    
    def test_valid_risk_params(self):
        result = ParameterValidator.validate_risk_params(
            leverage=5,
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        assert result.is_valid is True
    
    def test_invalid_leverage(self):
        result = ParameterValidator.validate_risk_params(
            leverage=200,
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        assert result.is_valid is False
        assert any("杠杆" in e for e in result.errors)
    
    def test_invalid_stop_loss(self):
        result = ParameterValidator.validate_risk_params(
            leverage=5,
            stop_loss_pct=-5.0,
            take_profit_pct=10.0,
        )
        assert result.is_valid is False
    
    def test_high_leverage_warning(self):
        result = ParameterValidator.validate_risk_params(
            leverage=50,
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        assert len(result.warnings) > 0
    
    def test_no_stop_loss_with_leverage_warning(self):
        result = ParameterValidator.validate_risk_params(
            leverage=10,
            stop_loss_pct=0,
            take_profit_pct=10.0,
        )
        assert any("止损" in w for w in result.warnings)
    
    def test_validate_strategy_params(self):
        strategy = MultiIndicatorStrategy()
        result = ParameterValidator.validate_strategy_params(
            strategy,
            {"vote_threshold": 2.0}
        )
        assert result.is_valid is True
    
    def test_validate_backtest_config(self):
        config = BacktestConfig(
            symbol="BTCUSDT",
            initial_capital=10000,
            leverage=5,
        )
        result = ParameterValidator.validate_backtest_config(config)
        assert result.is_valid is True


class TestBacktestConsistencyChecker:
    """回测一致性检查器测试"""
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2024-01-01', periods=200, freq='30min')
        data = pd.DataFrame({
            'open': np.random.uniform(49000, 51000, 200),
            'high': np.random.uniform(50000, 52000, 200),
            'low': np.random.uniform(48000, 50000, 200),
            'close': np.random.uniform(49000, 51000, 200),
            'volume': np.random.uniform(100, 1000, 200)
        }, index=dates)
        return data
    
    def test_check_reproducibility(self, sample_data):
        strategy = MultiIndicatorStrategy()
        config = BacktestConfig(symbol="BTCUSDT", interval="30min", leverage=1)
        
        result = BacktestConsistencyChecker.check_reproducibility(
            strategy, config, sample_data, runs=2
        )
        
        assert result.is_consistent is True
    
    def test_check_cross_validation(self, sample_data):
        strategy = MultiIndicatorStrategy()
        config = BacktestConfig(symbol="BTCUSDT", interval="30min", leverage=1)
        
        result = BacktestConsistencyChecker.check_cross_validation(
            strategy, config, sample_data, folds=2
        )
        
        assert "returns_per_fold" in result.metrics_diff
    
    def test_check_boundary_conditions(self, sample_data):
        strategy = MultiIndicatorStrategy()
        config = BacktestConfig(symbol="BTCUSDT", interval="30min", leverage=1)
        
        result = BacktestConsistencyChecker.check_boundary_conditions(
            strategy, config, sample_data
        )
        
        assert isinstance(result.issues, list)


class TestStrategyValidator:
    """策略验证器测试"""
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2024-01-01', periods=200, freq='30min')
        data = pd.DataFrame({
            'open': np.random.uniform(49000, 51000, 200),
            'high': np.random.uniform(50000, 52000, 200),
            'low': np.random.uniform(48000, 50000, 200),
            'close': np.random.uniform(49000, 51000, 200),
            'volume': np.random.uniform(100, 1000, 200)
        }, index=dates)
        return data
    
    def test_validate_all(self, sample_data):
        strategy = MultiIndicatorStrategy()
        config = BacktestConfig(
            symbol="BTCUSDT",
            interval="30min",
            initial_capital=10000,
            leverage=5,
        )
        
        validator = StrategyValidator(strategy, config)
        results = validator.validate_all(sample_data)
        
        assert "risk_params" in results
        assert "strategy_params" in results
        assert "overall_valid" in results
    
    def test_generate_validation_report(self, sample_data):
        strategy = MultiIndicatorStrategy()
        config = BacktestConfig(
            symbol="BTCUSDT",
            interval="30min",
            initial_capital=10000,
            leverage=5,
        )
        
        validator = StrategyValidator(strategy, config)
        report = validator.generate_validation_report(sample_data)
        
        assert "策略验证报告" in report
        assert "风险参数验证" in report
