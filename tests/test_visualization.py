"""
可视化模块测试
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import tempfile
import os

from backtest.visualization import (
    ChartConfig,
    TradeMarker,
    BacktestVisualizer,
)
from backtest.engine import BacktestResult, CompletedTrade, Trade


class TestChartConfig:
    """图表配置测试"""
    
    def test_default_values(self):
        config = ChartConfig()
        assert config.width == 1200
        assert config.height == 800
        assert config.candle_bull_color == "#0ecb81"
        assert config.candle_bear_color == "#f6465d"
        assert config.show_volume is True
        assert config.show_equity is True
    
    def test_supported_intervals(self):
        config = ChartConfig()
        assert "1min" in config.supported_intervals
        assert "5min" in config.supported_intervals
        assert "15min" in config.supported_intervals
        assert "1h" in config.supported_intervals
        assert "1d" in config.supported_intervals
    
    def test_custom_values(self):
        config = ChartConfig(
            width=1600,
            height=900,
            candle_bull_color="#00ff00",
            show_volume=False
        )
        assert config.width == 1600
        assert config.height == 900
        assert config.candle_bull_color == "#00ff00"
        assert config.show_volume is False


class TestTradeMarker:
    """交易标记测试"""
    
    def test_trade_marker_creation(self):
        marker = TradeMarker(
            timestamp=datetime.now(),
            price=50000.0,
            side="OPEN_LONG",
            trade_type="entry",
            quantity=0.1,
            pnl=0.0,
            reason="测试"
        )
        assert marker.price == 50000.0
        assert marker.side == "OPEN_LONG"
        assert marker.trade_type == "entry"


class TestBacktestVisualizer:
    """回测可视化器测试"""
    
    @pytest.fixture
    def visualizer(self):
        return BacktestVisualizer()
    
    @pytest.fixture
    def sample_ohlcv(self):
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
    def sample_result(self):
        dates = pd.date_range(start='2024-01-01', periods=100, freq='30min')
        equity_curve = pd.Series(
            [10000 + i * 10 for i in range(100)],
            index=dates
        )
        
        trades = [
            Trade(
                trade_id=1,
                timestamp=dates[10],
                symbol="BTCUSDT",
                side="OPEN_LONG",
                quantity=0.1,
                price=50000.0,
                commission=2.0,
                reason="测试开仓"
            ),
            Trade(
                trade_id=2,
                timestamp=dates[20],
                symbol="BTCUSDT",
                side="CLOSE_LONG",
                quantity=0.1,
                price=51000.0,
                commission=2.0,
                pnl=100.0,
                reason="测试平仓"
            )
        ]
        
        completed_trades = [
            CompletedTrade(
                entry_time=dates[10],
                exit_time=dates[20],
                entry_price=50000.0,
                exit_price=51000.0,
                quantity=0.1,
                pnl=100.0,
                commission=4.0,
                reason="测试交易",
                exit_type="signal",
                leverage=5
            )
        ]
        
        result = BacktestResult(
            trades=trades,
            completed_trades=completed_trades,
            equity_curve=equity_curve,
            initial_capital=10000,
            final_capital=11000,
            total_return=1000,
            total_return_pct=10.0,
            max_drawdown=500,
            max_drawdown_pct=5.0,
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            win_rate=100.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=2.0,
            avg_win=100.0,
            avg_loss=0.0,
            profit_factor=10.0,
            start_time=dates[0],
            end_time=dates[-1],
            leverage=5,
            stop_loss_hits=0,
            take_profit_hits=0
        )
        
        return result
    
    def test_visualizer_creation(self, visualizer):
        assert visualizer.config is not None
    
    def test_generate_chart_data(self, visualizer, sample_ohlcv, sample_result):
        chart_data = visualizer.generate_chart_data(
            sample_ohlcv,
            sample_result,
            "30min"
        )
        
        assert "config" in chart_data
        assert "candles" in chart_data
        assert "trades" in chart_data
        assert "equity" in chart_data
        assert "metrics" in chart_data
        assert "completed_trades" in chart_data
    
    def test_format_candles(self, visualizer, sample_ohlcv):
        candles = visualizer._format_candles(sample_ohlcv)
        
        assert len(candles) == 100
        assert "time" in candles[0]
        assert "open" in candles[0]
        assert "high" in candles[0]
        assert "low" in candles[0]
        assert "close" in candles[0]
        assert "volume" in candles[0]
    
    def test_format_trades(self, visualizer, sample_result, sample_ohlcv):
        trades = visualizer._format_trades(sample_result.trades, sample_ohlcv)
        
        assert len(trades) == 2
        assert trades[0]["side"] == "OPEN_LONG"
        assert trades[1]["side"] == "CLOSE_LONG"
    
    def test_format_equity(self, visualizer, sample_result):
        equity = visualizer._format_equity(sample_result.equity_curve)
        
        assert len(equity) == 100
        assert "time" in equity[0]
        assert "value" in equity[0]
    
    def test_format_metrics(self, visualizer, sample_result):
        metrics = visualizer._format_metrics(sample_result)
        
        assert "summary" in metrics
        assert "trades" in metrics
        assert "risk" in metrics
        assert "averages" in metrics
        assert "risk_management" in metrics
        
        assert metrics["summary"]["total_return_pct"] == 10.0
        assert metrics["trades"]["total"] == 1
        assert metrics["trades"]["win_rate"] == 100.0
    
    def test_format_completed_trades(self, visualizer, sample_result):
        trades = visualizer._format_completed_trades(sample_result.completed_trades)
        
        assert len(trades) == 1
        assert trades[0]["entry_price"] == 50000.0
        assert trades[0]["exit_price"] == 51000.0
        assert trades[0]["pnl"] == 100.0
        assert trades[0]["is_winner"] is True
    
    def test_generate_html_report(self, visualizer, sample_ohlcv, sample_result):
        html = visualizer.generate_html_report(
            sample_ohlcv,
            sample_result,
            "30min",
            "测试报告"
        )
        
        assert "<!DOCTYPE html>" in html
        assert "测试报告" in html
        assert "echarts" in html
        assert "K线图" in html
        assert "权益曲线" in html
    
    def test_generate_trades_table_rows(self, visualizer, sample_result):
        rows = visualizer._generate_trades_table_rows(sample_result.completed_trades)
        
        assert "<tr>" in rows
        assert "</tr>" in rows
        assert "50000" in rows
        assert "51000" in rows
    
    def test_to_json(self, visualizer):
        data = {
            "time": datetime(2024, 1, 1, 12, 0),
            "value": 100.0
        }
        
        json_str = visualizer._to_json(data)
        
        assert '"time":' in json_str
        assert '"value": 100.0' in json_str
    
    def test_save_html_report(self, visualizer, sample_ohlcv, sample_result):
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            output_path = f.name
        
        try:
            saved_path = visualizer.save_html_report(
                sample_ohlcv,
                sample_result,
                output_path,
                "30min",
                "测试报告"
            )
            
            assert saved_path == output_path
            assert os.path.exists(output_path)
            
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "<!DOCTYPE html>" in content
            assert "测试报告" in content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestBacktestVisualizerCustomConfig:
    """自定义配置的可视化器测试"""
    
    def test_custom_config(self):
        config = ChartConfig(
            width=1600,
            height=900,
            candle_bull_color="#00ff00",
            candle_bear_color="#ff0000",
            background_color="#ffffff"
        )
        
        visualizer = BacktestVisualizer(config)
        
        assert visualizer.config.width == 1600
        assert visualizer.config.height == 900
        assert visualizer.config.candle_bull_color == "#00ff00"
    
    def test_custom_colors_in_chart_data(self):
        config = ChartConfig(
            candle_bull_color="#00ff00",
            candle_bear_color="#ff0000"
        )
        
        visualizer = BacktestVisualizer(config)
        
        dates = pd.date_range(start='2024-01-01', periods=10, freq='30min')
        sample_ohlcv = pd.DataFrame({
            'open': [100] * 10,
            'high': [110] * 10,
            'low': [90] * 10,
            'close': [105] * 10,
            'volume': [1000] * 10
        }, index=dates)
        
        equity_curve = pd.Series([10000 + i for i in range(10)], index=dates)
        result = BacktestResult(
            equity_curve=equity_curve,
            initial_capital=10000,
            final_capital=10010,
            total_return=10,
            total_return_pct=0.1,
            total_trades=0,
            start_time=dates[0],
            end_time=dates[-1]
        )
        
        chart_data = visualizer.generate_chart_data(sample_ohlcv, result, "30min")
        
        assert chart_data["config"]["colors"]["bull"] == "#00ff00"
        assert chart_data["config"]["colors"]["bear"] == "#ff0000"
