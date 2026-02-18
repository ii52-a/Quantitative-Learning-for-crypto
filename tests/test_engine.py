"""
回测引擎测试
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from backtest.engine import BacktestEngine, BacktestResult, Trade, CompletedTrade
from core.config import BacktestConfig
from core.constants import SignalType, PositionSide
from Strategy.base import BaseStrategy, Bar, Signal, StrategyContext, StrategyResult


class SimpleTestStrategy(BaseStrategy):
    """简单测试策略"""
    
    name = "SimpleTestStrategy"
    display_name = "简单测试策略"
    description = "用于测试的简单策略"
    
    def __init__(self, params=None):
        super().__init__(params)
        self._bar_count = 0
    
    def initialize(self, context: StrategyContext) -> None:
        self._bar_count = 0
    
    def on_bar(self, bar: Bar, context: StrategyContext) -> StrategyResult:
        self._bar_count += 1
        
        if self._bar_count == 10:
            return StrategyResult(
                signal=Signal(
                    type=SignalType.OPEN_LONG,
                    price=bar.close,
                    reason="测试开仓"
                )
            )
        
        if self._bar_count == 20 and context.has_position:
            return StrategyResult(
                signal=Signal(
                    type=SignalType.CLOSE_LONG,
                    price=bar.close,
                    reason="测试平仓"
                )
            )
        
        return StrategyResult()


class TestTrade:
    """交易记录测试"""
    
    def test_trade_creation(self):
        trade = Trade(
            trade_id=1,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side="OPEN_LONG",
            quantity=0.1,
            price=50000.0,
            commission=2.0
        )
        assert trade.trade_id == 1
        assert trade.side == "OPEN_LONG"
        assert trade.price == 50000.0
    
    def test_trade_with_stop_loss_take_profit(self):
        trade = Trade(
            trade_id=1,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side="OPEN_LONG",
            quantity=0.1,
            price=50000.0,
            commission=2.0,
            stop_loss=48000.0,
            take_profit=52000.0
        )
        assert trade.stop_loss == 48000.0
        assert trade.take_profit == 52000.0


class TestCompletedTrade:
    """完成交易测试"""
    
    def test_completed_trade_creation(self):
        trade = CompletedTrade(
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=0.1,
            pnl=100.0,
            commission=4.0,
            reason="测试",
            exit_type="signal",
            leverage=5
        )
        assert trade.pnl == 100.0
        assert trade.exit_type == "signal"
        assert trade.leverage == 5


class TestBacktestResult:
    """回测结果测试"""
    
    def test_empty_result(self):
        result = BacktestResult(initial_capital=10000)
        assert result.initial_capital == 10000
        assert result.total_trades == 0
    
    def test_to_dict(self):
        result = BacktestResult(
            initial_capital=10000,
            final_capital=12000,
            total_return=2000,
            total_return_pct=20.0,
            total_trades=10,
            win_rate=60.0,
            leverage=5,
            stop_loss_hits=2,
            take_profit_hits=1
        )
        d = result.to_dict()
        assert d["initial_capital"] == 10000
        assert d["final_capital"] == 12000
        assert d["total_return_pct"] == 20.0
        assert d["leverage"] == 5
        assert d["stop_loss_hits"] == 2


class TestBacktestEngine:
    """回测引擎测试"""
    
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
            leverage=5,
            position_size=0.1,
            commission_rate=0.0004
        )
    
    def test_engine_creation(self, config):
        strategy = SimpleTestStrategy()
        engine = BacktestEngine(strategy, config)
        assert engine.config == config
        assert engine.risk_config is not None
    
    def test_run_with_empty_data(self, config):
        strategy = SimpleTestStrategy()
        engine = BacktestEngine(strategy, config)
        result = engine.run(pd.DataFrame())
        assert result.initial_capital == config.initial_capital
        assert result.total_trades == 0
    
    def test_run_with_sample_data(self, sample_data, config):
        strategy = SimpleTestStrategy()
        engine = BacktestEngine(strategy, config)
        result = engine.run(sample_data)
        
        assert result.initial_capital == config.initial_capital
        assert result.start_time is not None
        assert result.end_time is not None
        assert len(result.equity_curve) > 0
    
    def test_stop_loss_trigger(self, sample_data, config):
        config_with_sl = BacktestConfig(
            symbol="BTCUSDT",
            interval="30min",
            initial_capital=10000,
            leverage=5,
            position_size=0.1,
            commission_rate=0.0004,
            stop_loss_pct=50.0
        )
        
        strategy = SimpleTestStrategy()
        engine = BacktestEngine(strategy, config_with_sl)
        result = engine.run(sample_data)
        
        assert result.leverage == 5
    
    def test_take_profit_trigger(self, sample_data, config):
        config_with_tp = BacktestConfig(
            symbol="BTCUSDT",
            interval="30min",
            initial_capital=10000,
            leverage=5,
            position_size=0.1,
            commission_rate=0.0004,
            take_profit_pct=100.0
        )
        
        strategy = SimpleTestStrategy()
        engine = BacktestEngine(strategy, config_with_tp)
        result = engine.run(sample_data)
        
        assert result.leverage == 5
    
    def test_leverage_affects_pnl(self, sample_data):
        config_low_leverage = BacktestConfig(
            symbol="BTCUSDT",
            interval="30min",
            initial_capital=10000,
            leverage=1,
            position_size=0.1,
            commission_rate=0.0004
        )
        
        config_high_leverage = BacktestConfig(
            symbol="BTCUSDT",
            interval="30min",
            initial_capital=10000,
            leverage=10,
            position_size=0.1,
            commission_rate=0.0004
        )
        
        strategy = SimpleTestStrategy()
        
        engine_low = BacktestEngine(strategy, config_low_leverage)
        result_low = engine_low.run(sample_data)
        
        strategy2 = SimpleTestStrategy()
        engine_high = BacktestEngine(strategy2, config_high_leverage)
        result_high = engine_high.run(sample_data)
        
        assert result_high.leverage == 10
        assert result_low.leverage == 1
    
    def test_calculate_position_size(self, config):
        strategy = SimpleTestStrategy()
        engine = BacktestEngine(strategy, config)
        
        position_size = engine._calculate_position_size(50000.0)
        assert position_size > 0
    
    def test_calculate_equity_no_position(self, config):
        strategy = SimpleTestStrategy()
        engine = BacktestEngine(strategy, config)
        
        equity = engine._calculate_equity(50000.0)
        assert equity == config.initial_capital


class TestBacktestEngineRiskManagement:
    """回测引擎风险管理测试"""
    
    @pytest.fixture
    def trending_up_data(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='30min')
        base_price = 50000
        prices = [base_price + i * 100 for i in range(100)]
        data = pd.DataFrame({
            'open': prices,
            'high': [p + 100 for p in prices],
            'low': [p - 100 for p in prices],
            'close': prices,
            'volume': [1000] * 100
        }, index=dates)
        return data
    
    def test_stop_loss_calculation(self):
        from core.config import RiskConfig
        
        risk_config = RiskConfig(stop_loss_pct=5.0)
        
        sl_long = risk_config.calculate_stop_loss_price(100.0, is_long=True)
        assert sl_long == 95.0
        
        sl_short = risk_config.calculate_stop_loss_price(100.0, is_long=False)
        assert sl_short == 105.0
    
    def test_take_profit_calculation(self):
        from core.config import RiskConfig
        
        risk_config = RiskConfig(take_profit_pct=10.0)
        
        tp_long = risk_config.calculate_take_profit_price(100.0, is_long=True)
        assert abs(tp_long - 110.0) < 0.0001
        
        tp_short = risk_config.calculate_take_profit_price(100.0, is_long=False)
        assert abs(tp_short - 90.0) < 0.0001
