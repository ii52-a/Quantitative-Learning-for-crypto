"""
回测可视化模块

提供专业级回测可视化功能：
- K线图表（支持多时间周期）
- 交易标记（开仓/平仓点位）
- 策略指标线（MACD、RSI、布林带、均线等）
- 性能指标展示
- 交互功能（缩放、平移、悬停查看）
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import math

import pandas as pd
import numpy as np

from backtest.engine import BacktestResult, CompletedTrade
from Strategy.indicators import MACD, RSI, BollingerBands, MovingAverage, ATR


@dataclass
class IndicatorLineConfig:
    """指标线配置"""
    name: str
    display_name: str
    color: str
    line_width: int = 2
    visible: bool = True
    params: dict[str, Any] = field(default_factory=dict)


INDICATOR_PRESETS = {
    "MACD": [
        IndicatorLineConfig("macd", "MACD线", "#2962ff", 2, True, {}),
        IndicatorLineConfig("signal", "信号线", "#ff6d00", 2, True, {}),
    ],
    "RSI": [
        IndicatorLineConfig("rsi", "RSI", "#7c4dff", 2, True, {"period": 14}),
        IndicatorLineConfig("rsi_oversold", "超卖线", "#f6465d", 1, True, {"value": 30}),
        IndicatorLineConfig("rsi_overbought", "超买线", "#f6465d", 1, True, {"value": 70}),
    ],
    "BollingerBands": [
        IndicatorLineConfig("bb_upper", "上轨", "#00bcd4", 1, True, {"period": 20, "std_dev": 2.0}),
        IndicatorLineConfig("bb_middle", "中轨", "#00bcd4", 1, True, {"period": 20}),
        IndicatorLineConfig("bb_lower", "下轨", "#00bcd4", 1, True, {"period": 20, "std_dev": 2.0}),
    ],
    "MA": [
        IndicatorLineConfig("ma_fast", "快线MA", "#4caf50", 2, True, {"period": 10, "type": "EMA"}),
        IndicatorLineConfig("ma_slow", "慢线MA", "#ff9800", 2, True, {"period": 30, "type": "EMA"}),
    ],
}


@dataclass
class ChartConfig:
    """图表配置"""
    
    width: int = 1200
    height: int = 800
    
    candle_bull_color: str = "#0ecb81"
    candle_bear_color: str = "#f6465d"
    
    long_entry_color: str = "#0ecb81"
    long_exit_color: str = "#f6465d"
    short_entry_color: str = "#f6465d"
    short_exit_color: str = "#0ecb81"
    
    equity_line_color: str = "#f0b90b"
    drawdown_fill_color: str = "rgba(246, 70, 93, 0.3)"
    
    background_color: str = "#0b0e11"
    grid_color: str = "#2a2e39"
    text_color: str = "#eaecef"
    
    show_volume: bool = True
    show_equity: bool = True
    show_trades: bool = True
    
    supported_intervals: list[str] = field(default_factory=lambda: [
        "1min", "5min", "15min", "1h", "1d"
    ])
    
    enabled_indicators: list[str] = field(default_factory=lambda: ["MACD", "MA"])


@dataclass
class TradeMarker:
    """交易标记"""
    timestamp: datetime
    price: float
    side: str
    trade_type: str
    quantity: float
    pnl: float = 0.0
    reason: str = ""


class BacktestVisualizer:
    """回测可视化器"""
    
    def __init__(self, config: ChartConfig | None = None):
        self.config = config or ChartConfig()
    
    def generate_chart_data(
        self,
        ohlcv_data: pd.DataFrame,
        result: BacktestResult,
        interval: str = "30min",
        indicators: list[str] | None = None,
    ) -> dict[str, Any]:
        """生成图表数据"""
        enabled_indicators = indicators or self.config.enabled_indicators
        
        chart_data = {
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "colors": {
                    "bull": self.config.candle_bull_color,
                    "bear": self.config.candle_bear_color,
                    "long_entry": self.config.long_entry_color,
                    "long_exit": self.config.long_exit_color,
                    "short_entry": self.config.short_entry_color,
                    "short_exit": self.config.short_exit_color,
                    "equity": self.config.equity_line_color,
                    "drawdown": self.config.drawdown_fill_color,
                    "background": self.config.background_color,
                    "grid": self.config.grid_color,
                    "text": self.config.text_color,
                },
                "interval": interval,
                "enabled_indicators": enabled_indicators,
            },
            "candles": self._format_candles(ohlcv_data),
            "trades": self._format_trades(result.trades, ohlcv_data),
            "equity": self._format_equity(result.equity_curve),
            "metrics": self._format_metrics(result),
            "completed_trades": self._format_completed_trades(result.completed_trades),
            "indicators": self._calculate_indicators(ohlcv_data, enabled_indicators),
            "indicator_options": self._get_indicator_options(),
        }
        
        return chart_data
    
    def _calculate_indicators(
        self,
        ohlcv_data: pd.DataFrame,
        enabled_indicators: list[str],
    ) -> dict[str, Any]:
        """计算指标数据"""
        indicators_data = {}
        
        if ohlcv_data.empty:
            return indicators_data
        
        prices = ohlcv_data['close']
        high = ohlcv_data.get('high', prices)
        low = ohlcv_data.get('low', prices)
        
        for indicator_name in enabled_indicators:
            if indicator_name == "MACD":
                indicators_data["MACD"] = self._calculate_macd(prices)
            elif indicator_name == "RSI":
                indicators_data["RSI"] = self._calculate_rsi(prices)
            elif indicator_name == "BollingerBands":
                indicators_data["BollingerBands"] = self._calculate_bollinger(prices)
            elif indicator_name == "MA":
                indicators_data["MA"] = self._calculate_ma(prices)
        
        return indicators_data
    
    def _calculate_macd(self, prices: pd.Series) -> dict[str, Any]:
        """计算MACD指标"""
        macd = MACD()
        macd_line, signal_line, hist = macd.calculate(prices)
        
        times = [idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx) 
                 for idx in prices.index]
        
        return {
            "macd": [{"time": t, "value": float(v)} for t, v in zip(times, macd_line)],
            "signal": [{"time": t, "value": float(v)} for t, v in zip(times, signal_line)],
            "hist": [{"time": t, "value": float(v)} for t, v in zip(times, hist)],
        }
    
    def _calculate_rsi(self, prices: pd.Series) -> dict[str, Any]:
        """计算RSI指标"""
        rsi = RSI()
        rsi_values = rsi.calculate(prices)
        
        times = [idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx) 
                 for idx in prices.index]
        
        return {
            "rsi": [{"time": t, "value": float(v)} for t, v in zip(times, rsi_values)],
            "oversold": 30,
            "overbought": 70,
        }
    
    def _calculate_bollinger(self, prices: pd.Series) -> dict[str, Any]:
        """计算布林带"""
        bb = BollingerBands()
        upper, middle, lower = bb.calculate(prices)
        
        times = [idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx) 
                 for idx in prices.index]
        
        return {
            "upper": [{"time": t, "value": float(v)} for t, v in zip(times, upper)],
            "middle": [{"time": t, "value": float(v)} for t, v in zip(times, middle)],
            "lower": [{"time": t, "value": float(v)} for t, v in zip(times, lower)],
        }
    
    def _calculate_ma(self, prices: pd.Series) -> dict[str, Any]:
        """计算移动平均线"""
        ma_fast = MovingAverage(period=10, ma_type="EMA")
        ma_slow = MovingAverage(period=30, ma_type="EMA")
        
        ma_fast_values = ma_fast.calculate(prices)
        ma_slow_values = ma_slow.calculate(prices)
        
        times = [idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx) 
                 for idx in prices.index]
        
        return {
            "ma_fast": [{"time": t, "value": float(v)} for t, v in zip(times, ma_fast_values)],
            "ma_slow": [{"time": t, "value": float(v)} for t, v in zip(times, ma_slow_values)],
        }
    
    def _get_indicator_options(self) -> list[dict[str, Any]]:
        """获取可选指标列表"""
        return [
            {"name": "MACD", "display_name": "MACD指标", "description": "趋势跟踪指标"},
            {"name": "RSI", "display_name": "RSI指标", "description": "相对强弱指标"},
            {"name": "BollingerBands", "display_name": "布林带", "description": "波动率指标"},
            {"name": "MA", "display_name": "移动平均线", "description": "趋势指标"},
        ]
    
    def _format_candles(self, ohlcv_data: pd.DataFrame) -> list[dict]:
        """格式化K线数据"""
        candles = []
        
        for idx, row in ohlcv_data.iterrows():
            timestamp = idx
            if isinstance(idx, pd.Timestamp):
                timestamp = idx.isoformat()
            
            candle = {
                "time": timestamp,
                "open": float(row.get('open', 0)),
                "high": float(row.get('high', 0)),
                "low": float(row.get('low', 0)),
                "close": float(row.get('close', 0)),
                "volume": float(row.get('volume', 0)),
            }
            
            if self.config.show_volume and 'volume' in row:
                candle["volume"] = float(row['volume'])
            
            candles.append(candle)
        
        return candles
    
    def _format_trades(
        self,
        trades: list[Any],
        ohlcv_data: pd.DataFrame,
    ) -> list[dict]:
        """格式化交易数据"""
        formatted_trades = []
        
        for trade in trades:
            timestamp = trade.timestamp
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            
            trade_data = {
                "time": timestamp,
                "price": float(trade.price),
                "side": trade.side,
                "quantity": float(trade.quantity),
                "commission": float(trade.commission),
                "pnl": float(trade.pnl) if hasattr(trade, 'pnl') else 0,
                "reason": trade.reason,
                "type": "entry" if "OPEN" in trade.side else "exit",
                "stop_loss": trade.stop_loss if hasattr(trade, 'stop_loss') else None,
                "take_profit": trade.take_profit if hasattr(trade, 'take_profit') else None,
            }
            
            formatted_trades.append(trade_data)
        
        return formatted_trades
    
    def _format_equity(self, equity_curve: pd.Series) -> list[dict]:
        """格式化权益曲线数据"""
        if equity_curve.empty:
            return []
        
        equity_data = []
        for idx, value in equity_curve.items():
            timestamp = idx
            if isinstance(idx, pd.Timestamp):
                timestamp = idx.isoformat()
            
            equity_data.append({
                "time": timestamp,
                "value": float(value),
            })
        
        return equity_data
    
    def _format_metrics(self, result: BacktestResult) -> dict[str, Any]:
        """格式化绩效指标"""
        return {
            "summary": {
                "initial_capital": float(result.initial_capital),
                "final_capital": float(result.final_capital),
                "total_return": float(result.total_return),
                "total_return_pct": round(result.total_return_pct, 2),
                "max_drawdown": float(result.max_drawdown),
                "max_drawdown_pct": round(result.max_drawdown_pct, 2),
            },
            "trades": {
                "total": result.total_trades,
                "winning": result.winning_trades,
                "losing": result.losing_trades,
                "win_rate": round(result.win_rate, 2),
                "profit_factor": round(result.profit_factor, 2) if result.profit_factor != float('inf') else "∞",
            },
            "risk": {
                "sharpe_ratio": round(result.sharpe_ratio, 2),
                "sortino_ratio": round(result.sortino_ratio, 2),
                "calmar_ratio": round(result.calmar_ratio, 2),
            },
            "averages": {
                "avg_win": round(result.avg_win, 2),
                "avg_loss": round(result.avg_loss, 2),
            },
            "risk_management": {
                "stop_loss_hits": result.stop_loss_hits,
                "take_profit_hits": result.take_profit_hits,
                "leverage": result.leverage,
            },
        }
    
    def _format_completed_trades(
        self,
        completed_trades: list[CompletedTrade],
    ) -> list[dict]:
        """格式化完成的交易对"""
        trades = []
        
        for t in completed_trades:
            entry_time = t.entry_time
            exit_time = t.exit_time
            
            if isinstance(entry_time, datetime):
                entry_time = entry_time.isoformat()
            if isinstance(exit_time, datetime):
                exit_time = exit_time.isoformat()
            
            trade_data = {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": float(t.entry_price),
                "exit_price": float(t.exit_price),
                "quantity": float(t.quantity),
                "pnl": float(t.pnl),
                "commission": float(t.commission),
                "reason": t.reason,
                "exit_type": t.exit_type if hasattr(t, 'exit_type') else "signal",
                "leverage": t.leverage if hasattr(t, 'leverage') else 1,
                "is_winner": t.pnl > 0,
            }
            
            trades.append(trade_data)
        
        return trades
    
    def generate_html_report(
        self,
        ohlcv_data: pd.DataFrame,
        result: BacktestResult,
        interval: str = "30min",
        title: str = "回测报告",
    ) -> str:
        """生成HTML报告"""
        chart_data = self.generate_chart_data(ohlcv_data, result, interval)
        
        html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: {self.config.background_color};
            color: {self.config.text_color};
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 20px 0;
            border-bottom: 1px solid {self.config.grid_color};
            margin-bottom: 20px;
        }}
        .header h1 {{
            color: #f0b90b;
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background-color: #1e222d;
            border-radius: 8px;
            padding: 16px;
            border: 1px solid {self.config.grid_color};
        }}
        .metric-card .title {{
            color: #848e9c;
            font-size: 12px;
            margin-bottom: 8px;
        }}
        .metric-card .value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .metric-card .value.positive {{
            color: {self.config.candle_bull_color};
        }}
        .metric-card .value.negative {{
            color: {self.config.candle_bear_color};
        }}
        .chart-container {{
            background-color: #1e222d;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
            border: 1px solid {self.config.grid_color};
        }}
        .chart-title {{
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 12px;
            color: #f0b90b;
        }}
        #kline-chart {{
            width: 100%;
            height: 500px;
        }}
        #equity-chart {{
            width: 100%;
            height: 300px;
        }}
        #drawdown-chart {{
            width: 100%;
            height: 200px;
        }}
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            background-color: #1e222d;
            border-radius: 8px;
            overflow: hidden;
        }}
        .trades-table th, .trades-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid {self.config.grid_color};
        }}
        .trades-table th {{
            background-color: #2a2e39;
            color: #848e9c;
            font-weight: bold;
        }}
        .trades-table tr:hover {{
            background-color: #2a2e39;
        }}
        .profit {{
            color: {self.config.candle_bull_color};
        }}
        .loss {{
            color: {self.config.candle_bear_color};
        }}
        .risk-warning {{
            background-color: rgba(246, 70, 93, 0.1);
            border: 1px solid {self.config.candle_bear_color};
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
        }}
        .risk-warning h3 {{
            color: {self.config.candle_bear_color};
            margin-bottom: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {title}</h1>
            <p>回测周期: {result.start_time} 至 {result.end_time}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="title">总收益率</div>
                <div class="value {"positive" if result.total_return_pct > 0 else "negative"}">{result.total_return_pct:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="title">最大回撤</div>
                <div class="value negative">{result.max_drawdown_pct:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="title">夏普比率</div>
                <div class="value {"positive" if result.sharpe_ratio > 0 else ""}">{result.sharpe_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="title">胜率</div>
                <div class="value {"positive" if result.win_rate > 50 else ""}">{result.win_rate:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="title">盈亏比</div>
                <div class="value {"positive" if result.profit_factor > 1 else ""}">{result.profit_factor:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="title">交易次数</div>
                <div class="value">{result.total_trades}</div>
            </div>
            <div class="metric-card">
                <div class="title">杠杆倍数</div>
                <div class="value">{result.leverage}x</div>
            </div>
            <div class="metric-card">
                <div class="title">止损触发</div>
                <div class="value">{result.stop_loss_hits}</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📈 K线图与交易标记</div>
            <div id="kline-chart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">💰 权益曲线</div>
            <div id="equity-chart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📉 回撤曲线</div>
            <div id="drawdown-chart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📋 交易记录</div>
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>开仓时间</th>
                        <th>平仓时间</th>
                        <th>开仓价</th>
                        <th>平仓价</th>
                        <th>数量</th>
                        <th>盈亏</th>
                        <th>平仓原因</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_trades_table_rows(result.completed_trades)}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        const chartData = {self._to_json(chart_data)};
        
        // K线图
        const klineChart = echarts.init(document.getElementById('kline-chart'));
        const klineOption = {{
            backgroundColor: '#1e222d',
            tooltip: {{
                trigger: 'axis',
                axisPointer: {{ type: 'cross' }},
                backgroundColor: '#2a2e39',
                borderColor: '#2a2e39',
                textStyle: {{ color: '#eaecef' }}
            }},
            legend: {{
                data: ['K线', '买入', '卖出'],
                textStyle: {{ color: '#848e9c' }},
                top: 10
            }},
            grid: [
                {{ left: '10%', right: '8%', height: '60%' }},
                {{ left: '10%', right: '8%', top: '75%', height: '15%' }}
            ],
            xAxis: [
                {{ type: 'category', data: chartData.candles.map(c => c.time), axisLine: {{ lineStyle: {{ color: '#2a2e39' }} }}, axisLabel: {{ color: '#848e9c' }} }},
                {{ type: 'category', gridIndex: 1, axisLine: {{ lineStyle: {{ color: '#2a2e39' }} }}, axisLabel: {{ show: false }} }}
            ],
            yAxis: [
                {{ scale: true, axisLine: {{ lineStyle: {{ color: '#2a2e39' }} }}, splitLine: {{ lineStyle: {{ color: '#2a2e39' }} }}, axisLabel: {{ color: '#848e9c' }} }},
                {{ scale: true, gridIndex: 1, axisLine: {{ lineStyle: {{ color: '#2a2e39' }} }}, splitLine: {{ show: false }}, axisLabel: {{ color: '#848e9c' }} }}
            ],
            dataZoom: [
                {{ type: 'inside', xAxisIndex: [0, 1], start: 80, end: 100 }},
                {{ type: 'slider', xAxisIndex: [0, 1], bottom: 10, start: 80, end: 100, borderColor: '#2a2e39', backgroundColor: '#1e222d', fillerColor: 'rgba(240, 185, 11, 0.2)', handleStyle: {{ color: '#f0b90b' }}, textStyle: {{ color: '#848e9c' }} }}
            ],
            series: [
                {{
                    name: 'K线',
                    type: 'candlestick',
                    data: chartData.candles.map(c => [c.open, c.close, c.low, c.high]),
                    itemStyle: {{
                        color: chartData.config.colors.bull,
                        color0: chartData.config.colors.bear,
                        borderColor: chartData.config.colors.bull,
                        borderColor0: chartData.config.colors.bear
                    }}
                }},
                {{
                    name: '成交量',
                    type: 'bar',
                    xAxisIndex: 1,
                    yAxisIndex: 1,
                    data: chartData.candles.map(c => c.volume),
                    itemStyle: {{ color: '#f0b90b' }}
                }},
                {{
                    name: '买入',
                    type: 'scatter',
                    symbol: 'triangle',
                    symbolSize: 12,
                    data: chartData.trades.filter(t => t.type === 'entry').map(t => [t.time, t.price]),
                    itemStyle: {{ color: chartData.config.colors.long_entry }}
                }},
                {{
                    name: '卖出',
                    type: 'scatter',
                    symbol: 'triangle',
                    symbolRotate: 180,
                    symbolSize: 12,
                    data: chartData.trades.filter(t => t.type === 'exit').map(t => [t.time, t.price]),
                    itemStyle: {{ color: chartData.config.colors.long_exit }}
                }}
            ]
        }};
        klineChart.setOption(klineOption);
        
        // 权益曲线
        const equityChart = echarts.init(document.getElementById('equity-chart'));
        const equityOption = {{
            backgroundColor: '#1e222d',
            tooltip: {{
                trigger: 'axis',
                backgroundColor: '#2a2e39',
                borderColor: '#2a2e39',
                textStyle: {{ color: '#eaecef' }}
            }},
            grid: {{ left: '10%', right: '8%', top: '10%', bottom: '15%' }},
            xAxis: {{
                type: 'category',
                data: chartData.equity.map(e => e.time),
                axisLine: {{ lineStyle: {{ color: '#2a2e39' }} }},
                axisLabel: {{ show: false }}
            }},
            yAxis: {{
                type: 'value',
                axisLine: {{ lineStyle: {{ color: '#2a2e39' }} }},
                splitLine: {{ lineStyle: {{ color: '#2a2e39' }} }},
                axisLabel: {{ color: '#848e9c' }}
            }},
            dataZoom: [
                {{ type: 'inside', start: 0, end: 100 }},
                {{ type: 'slider', bottom: 10, borderColor: '#2a2e39', backgroundColor: '#1e222d', fillerColor: 'rgba(240, 185, 11, 0.2)', handleStyle: {{ color: '#f0b90b' }}, textStyle: {{ color: '#848e9c' }} }}
            ],
            series: [{{
                name: '权益',
                type: 'line',
                data: chartData.equity.map(e => e.value),
                lineStyle: {{ color: chartData.config.colors.equity, width: 2 }},
                areaStyle: {{ color: 'rgba(240, 185, 11, 0.1)' }},
                symbol: 'none'
            }}]
        }};
        equityChart.setOption(equityOption);
        
        // 回撤曲线
        const drawdownData = this._calculate_drawdown_data(chartData.equity);
        const drawdownChart = echarts.init(document.getElementById('drawdown-chart'));
        const drawdownOption = {{
            backgroundColor: '#1e222d',
            tooltip: {{
                trigger: 'axis',
                backgroundColor: '#2a2e39',
                borderColor: '#2a2e39',
                textStyle: {{ color: '#eaecef' }},
                formatter: function(params) {{
                    return '回撤: ' + Math.abs(params[0].value).toFixed(2) + '%';
                }}
            }},
            grid: {{ left: '10%', right: '8%', top: '10%', bottom: '15%' }},
            xAxis: {{
                type: 'category',
                data: chartData.equity.map(e => e.time),
                axisLine: {{ lineStyle: {{ color: '#2a2e39' }} }},
                axisLabel: {{ show: false }}
            }},
            yAxis: {{
                type: 'value',
                max: 0,
                axisLine: {{ lineStyle: {{ color: '#2a2e39' }} }},
                splitLine: {{ lineStyle: {{ color: '#2a2e39' }} }},
                axisLabel: {{ color: '#848e9c', formatter: '{{value}}%' }}
            }},
            series: [{{
                name: '回撤',
                type: 'line',
                data: drawdownData,
                lineStyle: {{ color: '#f6465d', width: 2 }},
                areaStyle: {{ color: 'rgba(246, 70, 93, 0.3)' }},
                symbol: 'none'
            }}]
        }};
        drawdownChart.setOption(drawdownOption);
        
        window.addEventListener('resize', function() {{
            klineChart.resize();
            equityChart.resize();
            drawdownChart.resize();
        }});
        
        function _calculate_drawdown_data(equityData) {{
            const result = [];
            let peak = equityData[0].value;
            for (let i = 0; i < equityData.length; i++) {{
                if (equityData[i].value > peak) {{
                    peak = equityData[i].value;
                }}
                const drawdown = ((equityData[i].value - peak) / peak) * 100;
                result.push(drawdown);
            }}
            return result;
        }}
    </script>
</body>
</html>'''
        
        return html
    
    def _generate_trades_table_rows(
        self,
        completed_trades: list[CompletedTrade],
    ) -> str:
        """生成交易记录表格行"""
        rows = []
        
        for t in completed_trades:
            pnl_class = "profit" if t.pnl > 0 else "loss"
            exit_type = t.exit_type if hasattr(t, 'exit_type') else "signal"
            
            row = f'''<tr>
                <td>{t.entry_time}</td>
                <td>{t.exit_time}</td>
                <td>{t.entry_price:.2f}</td>
                <td>{t.exit_price:.2f}</td>
                <td>{t.quantity:.4f}</td>
                <td class="{pnl_class}">{t.pnl:.2f}</td>
                <td>{exit_type}</td>
            </tr>'''
            rows.append(row)
        
        return "\n".join(rows)
    
    def _to_json(self, data: Any) -> str:
        """转换为JSON字符串"""
        import json
        
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if pd.isna(obj):
                return None
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(data, default=json_serializer, ensure_ascii=False)
    
    def save_html_report(
        self,
        ohlcv_data: pd.DataFrame,
        result: BacktestResult,
        output_path: str,
        interval: str = "30min",
        title: str = "回测报告",
    ) -> str:
        """保存HTML报告到文件"""
        html = self.generate_html_report(ohlcv_data, result, interval, title)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
