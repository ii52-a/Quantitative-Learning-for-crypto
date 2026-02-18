"""
配置模块测试
"""

import pytest
from core.config import Config, BacktestConfig, TradingConfig, RiskConfig


class TestRiskConfig:
    """风险配置测试"""
    
    def test_default_values(self):
        config = RiskConfig()
        assert config.stop_loss_pct == 0.0
        assert config.take_profit_pct == 0.0
        assert config.leverage == 1
    
    def test_custom_values(self):
        config = RiskConfig(stop_loss_pct=5.5, take_profit_pct=10.0, leverage=10)
        assert config.stop_loss_pct == 5.5
        assert config.take_profit_pct == 10.0
        assert config.leverage == 10
    
    def test_decimal_rounding(self):
        config = RiskConfig(stop_loss_pct=5.555, take_profit_pct=10.999)
        assert config.stop_loss_pct == 5.55
        assert config.take_profit_pct == 11.0
    
    def test_validate_valid_config(self):
        config = RiskConfig(stop_loss_pct=5.0, take_profit_pct=10.0, leverage=5)
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validate_invalid_leverage(self):
        config = RiskConfig(leverage=0)
        errors = config.validate()
        assert len(errors) > 0
        assert "杠杆" in errors[0]
        
        config = RiskConfig(leverage=200)
        errors = config.validate()
        assert len(errors) > 0
    
    def test_validate_invalid_stop_loss(self):
        config = RiskConfig(stop_loss_pct=-1)
        errors = config.validate()
        assert len(errors) > 0
        
        config = RiskConfig(stop_loss_pct=200)
        errors = config.validate()
        assert len(errors) > 0
    
    def test_get_risk_warning_high_leverage(self):
        config = RiskConfig(leverage=25)
        warning = config.get_risk_warning()
        assert warning is not None
        assert "高杠杆" in warning
    
    def test_get_risk_warning_medium_leverage(self):
        config = RiskConfig(leverage=15)
        warning = config.get_risk_warning()
        assert warning is not None
        assert "中高杠杆" in warning
    
    def test_get_risk_warning_no_stop_loss_with_leverage(self):
        config = RiskConfig(leverage=5, stop_loss_pct=0)
        warning = config.get_risk_warning()
        assert warning is not None
        assert "止损" in warning
    
    def test_get_risk_warning_no_warning(self):
        config = RiskConfig(leverage=1, stop_loss_pct=5.0, take_profit_pct=10.0)
        warning = config.get_risk_warning()
        assert warning is None
    
    def test_calculate_stop_loss_price_long(self):
        config = RiskConfig(stop_loss_pct=5.0)
        sl_price = config.calculate_stop_loss_price(100.0, is_long=True)
        assert sl_price == 95.0
    
    def test_calculate_stop_loss_price_short(self):
        config = RiskConfig(stop_loss_pct=5.0)
        sl_price = config.calculate_stop_loss_price(100.0, is_long=False)
        assert sl_price == 105.0
    
    def test_calculate_stop_loss_price_zero(self):
        config = RiskConfig(stop_loss_pct=0)
        sl_price = config.calculate_stop_loss_price(100.0)
        assert sl_price is None
    
    def test_calculate_take_profit_price_long(self):
        config = RiskConfig(take_profit_pct=10.0)
        tp_price = config.calculate_take_profit_price(100.0, is_long=True)
        assert abs(tp_price - 110.0) < 0.0001
    
    def test_calculate_take_profit_price_short(self):
        config = RiskConfig(take_profit_pct=10.0)
        tp_price = config.calculate_take_profit_price(100.0, is_long=False)
        assert tp_price == 90.0
    
    def test_to_dict(self):
        config = RiskConfig(stop_loss_pct=5.0, take_profit_pct=10.0, leverage=5)
        d = config.to_dict()
        assert d["stop_loss_pct"] == 5.0
        assert d["take_profit_pct"] == 10.0
        assert d["leverage"] == 5


class TestBacktestConfig:
    """回测配置测试"""
    
    def test_default_values(self):
        config = BacktestConfig()
        assert config.symbol == "BTCUSDT"
        assert config.interval == "30min"
        assert config.initial_capital == 10000.0
        assert config.leverage == 5
        assert config.stop_loss_pct == 0.0
        assert config.take_profit_pct == 0.0
    
    def test_validate_valid_config(self):
        config = BacktestConfig(
            initial_capital=10000,
            leverage=10,
            stop_loss_pct=5.0,
            take_profit_pct=10.0
        )
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validate_invalid_capital(self):
        config = BacktestConfig(initial_capital=-100)
        errors = config.validate()
        assert len(errors) > 0
    
    def test_validate_invalid_leverage(self):
        config = BacktestConfig(leverage=200)
        errors = config.validate()
        assert len(errors) > 0
    
    def test_get_risk_config(self):
        config = BacktestConfig(
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
            leverage=10
        )
        risk_config = config.get_risk_config()
        assert isinstance(risk_config, RiskConfig)
        assert risk_config.stop_loss_pct == 5.0
        assert risk_config.take_profit_pct == 10.0
        assert risk_config.leverage == 10


class TestTradingConfig:
    """交易配置测试"""
    
    def test_default_values(self):
        config = TradingConfig()
        assert config.symbol == "BTCUSDT"
        assert config.testnet is True
        assert config.dry_run is True
        assert config.leverage == 5
    
    def test_validate_missing_api_key(self):
        config = TradingConfig()
        errors = config.validate()
        assert len(errors) > 0
        assert "API Key" in errors[0]
    
    def test_validate_invalid_risk_per_trade(self):
        config = TradingConfig(
            api_key="test",
            api_secret="test",
            risk_per_trade=2.0
        )
        errors = config.validate()
        assert len(errors) > 0
    
    def test_get_risk_config(self):
        config = TradingConfig(
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            leverage=10
        )
        risk_config = config.get_risk_config()
        assert risk_config.stop_loss_pct == 5.0
        assert risk_config.take_profit_pct == 10.0
        assert risk_config.leverage == 10


class TestConfig:
    """全局配置测试"""
    
    def test_default_values(self):
        config = Config()
        assert config.project_name == "Quantitative Trading System"
        assert config.version == "2.0.0"
        assert config.debug is False
    
    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        config = Config.from_env()
        assert config.debug is True
        assert config.log_level == "DEBUG"
