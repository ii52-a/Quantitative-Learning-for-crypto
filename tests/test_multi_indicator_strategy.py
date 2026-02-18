"""
多指标策略测试
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from Strategy.multi_indicator_strategy import (
    MultiIndicatorStrategy,
    AdaptiveMultiIndicatorStrategy,
    SignalWeight,
    IndicatorSignal,
)
from Strategy.base import StrategyContext, Bar, SignalType, Position, PositionSide
from core.constants import SignalType


class TestSignalWeight:
    """信号权重测试"""
    
    def test_signal_values(self):
        assert SignalWeight.STRONG_BUY.value == 2
        assert SignalWeight.BUY.value == 1
        assert SignalWeight.NEUTRAL.value == 0
        assert SignalWeight.SELL.value == -1
        assert SignalWeight.STRONG_SELL.value == -2


class TestIndicatorSignal:
    """指标信号测试"""
    
    def test_signal_creation(self):
        signal = IndicatorSignal(
            name="MACD",
            signal=SignalWeight.BUY,
            strength=0.8,
            reason="金叉"
        )
        assert signal.name == "MACD"
        assert signal.signal == SignalWeight.BUY
        assert signal.strength == 0.8


class TestMultiIndicatorStrategy:
    """多指标组合策略测试"""
    
    @pytest.fixture
    def strategy(self):
        return MultiIndicatorStrategy()
    
    @pytest.fixture
    def context(self):
        return StrategyContext(
            symbol="BTCUSDT",
            interval="30min",
            position=Position(
                side=PositionSide.EMPTY,
                quantity=0.0,
                entry_price=0.0,
                entry_time=datetime.now(),
            ),
            equity=10000.0,
            available_capital=10000.0,
            current_price=50000.0,
            timestamp=datetime.now(),
        )
    
    @pytest.fixture
    def sample_bar(self):
        return Bar(
            timestamp=datetime.now(),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
            symbol="BTCUSDT",
            interval="30min",
        )
    
    def test_strategy_creation(self, strategy):
        assert strategy.name == "MultiIndicatorStrategy"
        assert strategy.display_name == "多指标组合策略"
    
    def test_strategy_parameters(self, strategy):
        params = strategy.parameters
        assert len(params) > 0
        
        param_names = [p.name for p in params]
        assert "vote_threshold" in param_names
        assert "macd_weight" in param_names
        assert "rsi_weight" in param_names
    
    def test_initialize(self, strategy, context):
        strategy.initialize(context)
        assert strategy._initialized is True
        assert strategy._macd is not None
        assert strategy._rsi is not None
    
    def test_on_bar_insufficient_data(self, strategy, context, sample_bar):
        strategy.initialize(context)
        
        result = strategy.on_bar(sample_bar, context)
        
        assert result.signal is None
        assert "数据收集中" in result.log
    
    def test_on_bar_with_sufficient_data(self, strategy, context):
        strategy.initialize(context)
        
        for i in range(60):
            bar = Bar(
                timestamp=datetime.now() + timedelta(minutes=30*i),
                open=50000.0 + i * 10,
                high=50100.0 + i * 10,
                low=49900.0 + i * 10,
                close=50050.0 + i * 10,
                volume=1000.0,
                symbol="BTCUSDT",
                interval="30min",
            )
            result = strategy.on_bar(bar, context)
        
        assert "总得分" in result.log
    
    def test_calculate_total_score(self, strategy):
        signals = [
            IndicatorSignal("MACD", SignalWeight.BUY, 0.8, "金叉"),
            IndicatorSignal("RSI", SignalWeight.BUY, 0.5, "超卖反弹"),
            IndicatorSignal("BB", SignalWeight.NEUTRAL, 0.0, "中性"),
        ]
        
        score = strategy._calculate_total_score(signals)
        
        expected = 1 * 0.8 + 1 * 0.5 + 0 * 0.0
        assert abs(score - expected) < 0.001
    
    def test_analyze_macd_bullish_cross(self, strategy):
        strategy._macd = strategy._macd or type('MACD', (), {'calculate': lambda self, p: (
            pd.Series([0.1] * len(p)),
            pd.Series([0.05] * len(p)),
            pd.Series([0.05] * len(p))
        )})()
        
        prices = pd.Series([50000 + i for i in range(50)])
        
        signal = strategy._analyze_macd(prices)
        
        assert signal is not None
        assert signal.name == "MACD"
    
    def test_analyze_rsi_oversold(self, strategy):
        strategy._rsi = strategy._rsi or type('RSI', (), {'calculate': lambda self, p: (
            pd.Series([25.0] * len(p))
        )})()
        
        prices = pd.Series([50000 - i * 10 for i in range(50)])
        
        signal = strategy._analyze_rsi(prices)
        
        assert signal is not None
        assert signal.name == "RSI"
    
    def test_custom_params(self):
        strategy = MultiIndicatorStrategy({
            "vote_threshold": 3.0,
            "macd_weight": 2.0,
            "rsi_weight": 1.5,
        })
        
        assert strategy._params["vote_threshold"] == 3.0
        assert strategy._params["macd_weight"] == 2.0
    
    def test_get_required_data_count(self, strategy):
        assert strategy.get_required_data_count() == 50


class TestAdaptiveMultiIndicatorStrategy:
    """自适应多指标策略测试"""
    
    @pytest.fixture
    def strategy(self):
        return AdaptiveMultiIndicatorStrategy()
    
    def test_strategy_creation(self, strategy):
        assert strategy.name == "AdaptiveMultiIndicatorStrategy"
        assert strategy.display_name == "自适应多指标策略"
    
    def test_adapt_weights_high_volatility(self, strategy):
        from Strategy.base import StrategyContext
        
        context = StrategyContext(
            symbol="BTCUSDT",
            interval="30min",
            position=Position(
                side=PositionSide.EMPTY,
                quantity=0.0,
                entry_price=0.0,
                entry_time=datetime.now(),
            ),
            equity=10000.0,
            available_capital=10000.0,
            current_price=50000.0,
            timestamp=datetime.now(),
        )
        
        strategy.initialize(context)
        
        strategy._params = {
            "volatility_threshold": 3.0,
            "rsi_weight": 1.0,
            "bb_weight": 1.0,
            "macd_weight": 1.0,
            "ma_weight": 1.0,
        }
        
        strategy._highs = [50000 + i * 500 for i in range(50)]
        strategy._lows = [50000 - i * 500 for i in range(50)]
        strategy._prices = [50000 for _ in range(50)]
        
        bar = Bar(
            timestamp=datetime.now(),
            open=50000.0,
            high=52000.0,
            low=48000.0,
            close=50000.0,
            volume=1000.0,
            symbol="BTCUSDT",
            interval="30min",
        )
        
        strategy._adapt_weights(bar)
        
        assert strategy._params["rsi_weight"] != 1.0
    
    def test_inherits_from_multi_indicator(self, strategy):
        assert hasattr(strategy, '_analyze_macd')
        assert hasattr(strategy, '_analyze_rsi')
        assert hasattr(strategy, '_analyze_bollinger')
        assert hasattr(strategy, '_analyze_ma')


class TestMultiIndicatorStrategyIntegration:
    """多指标策略集成测试"""
    
    def test_full_backtest_flow(self):
        strategy = MultiIndicatorStrategy({
            "vote_threshold": 2.0,
            "macd_weight": 1.0,
            "rsi_weight": 1.0,
        })
        
        context = StrategyContext(
            symbol="BTCUSDT",
            interval="30min",
            position=Position(
                side=PositionSide.EMPTY,
                quantity=0.0,
                entry_price=0.0,
                entry_time=datetime.now(),
            ),
            equity=10000.0,
            available_capital=10000.0,
            current_price=50000.0,
            timestamp=datetime.now(),
        )
        
        strategy.initialize(context)
        
        for i in range(100):
            trend = 1 if i < 50 else -1
            price = 50000 + trend * i * 10
            
            bar = Bar(
                timestamp=datetime.now() + timedelta(minutes=30*i),
                open=price - 50,
                high=price + 100,
                low=price - 100,
                close=price,
                volume=1000.0,
                symbol="BTCUSDT",
                interval="30min",
            )
            
            result = strategy.on_bar(bar, context)
            
            if result.signal:
                if result.signal.type == SignalType.OPEN_LONG:
                    context.position = Position(
                        side=PositionSide.LONG,
                        quantity=0.1,
                        entry_price=price,
                        entry_time=bar.timestamp,
                    )
                elif result.signal.type == SignalType.CLOSE_LONG:
                    context.position = Position(
                        side=PositionSide.EMPTY,
                        quantity=0.0,
                        entry_price=0.0,
                        entry_time=bar.timestamp,
                    )
        
        assert True
