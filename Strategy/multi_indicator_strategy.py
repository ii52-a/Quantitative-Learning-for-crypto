"""
多指标组合策略模块

提供多种指标组合的交易策略：
- 投票策略：多个指标投票决定
- 加权策略：根据指标权重计算
- 信号强度策略：根据信号强度决定
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum

import pandas as pd
import numpy as np

from Strategy.base import (
    BaseStrategy,
    StrategyContext,
    StrategyResult,
    Signal,
    Bar,
    StrategyParameter,
)
from Strategy.indicators import MACD, RSI, BollingerBands, MovingAverage, ATR
from core.constants import SignalType, PositionSide


class SignalWeight(Enum):
    """信号权重"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class IndicatorSignal:
    """指标信号"""
    name: str
    signal: SignalWeight
    strength: float
    reason: str


class MultiIndicatorStrategy(BaseStrategy):
    """多指标组合策略
    
    策略逻辑：
    1. 多个指标分别产生信号
    2. 根据投票或加权方式决定最终信号
    3. 支持自定义指标权重
    """
    
    name = "MultiIndicatorStrategy"
    display_name = "多指标组合策略"
    description = "结合MACD、RSI、布林带、均线的多指标投票策略"
    strategy_type = "multi_indicator"
    risk_level = "medium"
    
    parameters = [
        StrategyParameter(
            name="vote_threshold",
            display_name="投票阈值",
            description="开仓所需的信号强度阈值",
            value_type=float,
            default_value=2.0,
            min_value=1.0,
            max_value=5.0,
        ),
        StrategyParameter(
            name="macd_weight",
            display_name="MACD权重",
            description="MACD指标的投票权重",
            value_type=float,
            default_value=1.0,
            min_value=0.0,
            max_value=3.0,
        ),
        StrategyParameter(
            name="rsi_weight",
            display_name="RSI权重",
            description="RSI指标的投票权重",
            value_type=float,
            default_value=1.0,
            min_value=0.0,
            max_value=3.0,
        ),
        StrategyParameter(
            name="bb_weight",
            display_name="布林带权重",
            description="布林带指标的投票权重",
            value_type=float,
            default_value=0.8,
            min_value=0.0,
            max_value=3.0,
        ),
        StrategyParameter(
            name="ma_weight",
            display_name="均线权重",
            description="均线指标的投票权重",
            value_type=float,
            default_value=0.8,
            min_value=0.0,
            max_value=3.0,
        ),
        StrategyParameter(
            name="rsi_oversold",
            display_name="RSI超卖",
            description="RSI超卖阈值",
            value_type=float,
            default_value=30.0,
            min_value=10.0,
            max_value=40.0,
        ),
        StrategyParameter(
            name="rsi_overbought",
            display_name="RSI超买",
            description="RSI超买阈值",
            value_type=float,
            default_value=70.0,
            min_value=60.0,
            max_value=90.0,
        ),
    ]
    
    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self._macd: MACD | None = None
        self._rsi: RSI | None = None
        self._bb: BollingerBands | None = None
        self._ma_fast: MovingAverage | None = None
        self._ma_slow: MovingAverage | None = None
        self._prices: list[float] = []
        self._prev_signals: dict[str, IndicatorSignal] = {}
    
    def initialize(self, context: StrategyContext) -> None:
        self._macd = MACD()
        self._rsi = RSI(period=14)
        self._bb = BollingerBands(period=20, std_dev=2.0)
        self._ma_fast = MovingAverage(period=10, ma_type="EMA")
        self._ma_slow = MovingAverage(period=30, ma_type="EMA")
        self._prices = []
        self._prev_signals = {}
        self._initialized = True
    
    def on_bar(self, bar: Bar, context: StrategyContext) -> StrategyResult:
        self._prices.append(bar.close)
        
        min_data = 50
        if len(self._prices) < min_data:
            return StrategyResult(log=f"数据收集中: {len(self._prices)}/{min_data}")
        
        prices = pd.Series(self._prices[-200:])
        
        signals = []
        
        macd_signal = self._analyze_macd(prices)
        if macd_signal:
            signals.append(macd_signal)
        
        rsi_signal = self._analyze_rsi(prices)
        if rsi_signal:
            signals.append(rsi_signal)
        
        bb_signal = self._analyze_bollinger(prices, bar.close)
        if bb_signal:
            signals.append(bb_signal)
        
        ma_signal = self._analyze_ma(prices, bar.close)
        if ma_signal:
            signals.append(ma_signal)
        
        total_score = self._calculate_total_score(signals)
        
        signal = None
        vote_threshold = self._params.get("vote_threshold", 2.0)
        
        if total_score >= vote_threshold and not context.has_position:
            signal = Signal(
                type=SignalType.OPEN_LONG,
                price=bar.close,
                reason=f"多指标看多 (得分: {total_score:.1f})",
            )
        elif total_score <= -vote_threshold and context.position.side == PositionSide.LONG:
            signal = Signal(
                type=SignalType.CLOSE_LONG,
                price=bar.close,
                reason=f"多指标看空 (得分: {total_score:.1f})",
            )
        
        indicator_values = {
            "total_score": total_score,
            "signals": {s.name: {"signal": s.signal.value, "strength": s.strength} for s in signals},
        }
        
        log_parts = [f"总得分: {total_score:.1f}"]
        for s in signals:
            log_parts.append(f"{s.name}: {s.signal.name}({s.strength:.1f})")
        
        return StrategyResult(
            signal=signal,
            indicators=indicator_values,
            log=" | ".join(log_parts),
        )
    
    def _analyze_macd(self, prices: pd.Series) -> IndicatorSignal | None:
        """分析MACD信号"""
        macd_line, signal_line, hist = self._macd.calculate(prices)
        
        curr_hist = float(hist.iloc[-1])
        prev_hist = float(hist.iloc[-2]) if len(hist) > 1 else 0
        
        weight = self._params.get("macd_weight", 1.0)
        
        if prev_hist < 0 and curr_hist > 0:
            strength = min(abs(curr_hist) * 10, 1.0)
            return IndicatorSignal(
                name="MACD",
                signal=SignalWeight.BUY,
                strength=strength * weight,
                reason=f"金叉 (HIST: {prev_hist:.2f} -> {curr_hist:.2f})",
            )
        elif prev_hist > 0 and curr_hist < 0:
            strength = min(abs(curr_hist) * 10, 1.0)
            return IndicatorSignal(
                name="MACD",
                signal=SignalWeight.SELL,
                strength=strength * weight,
                reason=f"死叉 (HIST: {prev_hist:.2f} -> {curr_hist:.2f})",
            )
        elif curr_hist > 0:
            return IndicatorSignal(
                name="MACD",
                signal=SignalWeight.BUY,
                strength=0.3 * weight,
                reason=f"多头趋势 (HIST: {curr_hist:.2f})",
            )
        elif curr_hist < 0:
            return IndicatorSignal(
                name="MACD",
                signal=SignalWeight.SELL,
                strength=0.3 * weight,
                reason=f"空头趋势 (HIST: {curr_hist:.2f})",
            )
        
        return IndicatorSignal(
            name="MACD",
            signal=SignalWeight.NEUTRAL,
            strength=0,
            reason="中性",
        )
    
    def _analyze_rsi(self, prices: pd.Series) -> IndicatorSignal | None:
        """分析RSI信号"""
        rsi_values = self._rsi.calculate(prices)
        
        curr_rsi = float(rsi_values.iloc[-1])
        prev_rsi = float(rsi_values.iloc[-2]) if len(rsi_values) > 1 else 50
        
        weight = self._params.get("rsi_weight", 1.0)
        oversold = self._params.get("rsi_oversold", 30)
        overbought = self._params.get("rsi_overbought", 70)
        
        if prev_rsi < oversold and curr_rsi > oversold:
            strength = min((oversold - prev_rsi) / oversold, 1.0)
            return IndicatorSignal(
                name="RSI",
                signal=SignalWeight.BUY,
                strength=strength * weight,
                reason=f"超卖反弹 (RSI: {prev_rsi:.1f} -> {curr_rsi:.1f})",
            )
        elif prev_rsi > overbought and curr_rsi < overbought:
            strength = min((prev_rsi - overbought) / (100 - overbought), 1.0)
            return IndicatorSignal(
                name="RSI",
                signal=SignalWeight.SELL,
                strength=strength * weight,
                reason=f"超买回落 (RSI: {prev_rsi:.1f} -> {curr_rsi:.1f})",
            )
        elif curr_rsi < oversold:
            return IndicatorSignal(
                name="RSI",
                signal=SignalWeight.BUY,
                strength=0.5 * weight,
                reason=f"超卖区域 (RSI: {curr_rsi:.1f})",
            )
        elif curr_rsi > overbought:
            return IndicatorSignal(
                name="RSI",
                signal=SignalWeight.SELL,
                strength=0.5 * weight,
                reason=f"超买区域 (RSI: {curr_rsi:.1f})",
            )
        
        return IndicatorSignal(
            name="RSI",
            signal=SignalWeight.NEUTRAL,
            strength=0,
            reason=f"中性 (RSI: {curr_rsi:.1f})",
        )
    
    def _analyze_bollinger(self, prices: pd.Series, current_price: float) -> IndicatorSignal | None:
        """分析布林带信号"""
        upper, middle, lower = self._bb.calculate(prices)
        
        bb_upper = float(upper.iloc[-1])
        bb_lower = float(lower.iloc[-1])
        bb_middle = float(middle.iloc[-1])
        
        weight = self._params.get("bb_weight", 0.8)
        
        if current_price <= bb_lower:
            strength = min((bb_lower - current_price) / bb_lower * 10, 1.0)
            return IndicatorSignal(
                name="BB",
                signal=SignalWeight.BUY,
                strength=strength * weight,
                reason=f"触及下轨 (价格: {current_price:.2f}, 下轨: {bb_lower:.2f})",
            )
        elif current_price >= bb_upper:
            strength = min((current_price - bb_upper) / bb_upper * 10, 1.0)
            return IndicatorSignal(
                name="BB",
                signal=SignalWeight.SELL,
                strength=strength * weight,
                reason=f"触及上轨 (价格: {current_price:.2f}, 上轨: {bb_upper:.2f})",
            )
        elif current_price > bb_middle:
            return IndicatorSignal(
                name="BB",
                signal=SignalWeight.BUY,
                strength=0.2 * weight,
                reason=f"中轨上方 (价格: {current_price:.2f})",
            )
        else:
            return IndicatorSignal(
                name="BB",
                signal=SignalWeight.SELL,
                strength=0.2 * weight,
                reason=f"中轨下方 (价格: {current_price:.2f})",
            )
    
    def _analyze_ma(self, prices: pd.Series, current_price: float) -> IndicatorSignal | None:
        """分析均线信号"""
        ma_fast = self._ma_fast.calculate(prices)
        ma_slow = self._ma_slow.calculate(prices)
        
        ma_fast_val = float(ma_fast.iloc[-1])
        ma_slow_val = float(ma_slow.iloc[-1])
        ma_fast_prev = float(ma_fast.iloc[-2]) if len(ma_fast) > 1 else ma_fast_val
        ma_slow_prev = float(ma_slow.iloc[-2]) if len(ma_slow) > 1 else ma_slow_val
        
        weight = self._params.get("ma_weight", 0.8)
        
        if ma_fast_prev < ma_slow_prev and ma_fast_val > ma_slow_val:
            strength = min(abs(ma_fast_val - ma_slow_val) / ma_slow_val * 100, 1.0)
            return IndicatorSignal(
                name="MA",
                signal=SignalWeight.BUY,
                strength=strength * weight,
                reason=f"金叉 (快线: {ma_fast_val:.2f}, 慢线: {ma_slow_val:.2f})",
            )
        elif ma_fast_prev > ma_slow_prev and ma_fast_val < ma_slow_val:
            strength = min(abs(ma_fast_val - ma_slow_val) / ma_slow_val * 100, 1.0)
            return IndicatorSignal(
                name="MA",
                signal=SignalWeight.SELL,
                strength=strength * weight,
                reason=f"死叉 (快线: {ma_fast_val:.2f}, 慢线: {ma_slow_val:.2f})",
            )
        elif current_price > ma_fast_val > ma_slow_val:
            return IndicatorSignal(
                name="MA",
                signal=SignalWeight.BUY,
                strength=0.3 * weight,
                reason=f"多头排列 (价格 > 快线 > 慢线)",
            )
        elif current_price < ma_fast_val < ma_slow_val:
            return IndicatorSignal(
                name="MA",
                signal=SignalWeight.SELL,
                strength=0.3 * weight,
                reason=f"空头排列 (价格 < 快线 < 慢线)",
            )
        
        return IndicatorSignal(
            name="MA",
            signal=SignalWeight.NEUTRAL,
            strength=0,
            reason="中性",
        )
    
    def _calculate_total_score(self, signals: list[IndicatorSignal]) -> float:
        """计算总得分"""
        total = 0.0
        for s in signals:
            total += s.signal.value * s.strength
        return total
    
    def get_required_data_count(self) -> int:
        return 50


class AdaptiveMultiIndicatorStrategy(MultiIndicatorStrategy):
    """自适应多指标策略
    
    根据市场状态动态调整指标权重
    """
    
    name = "AdaptiveMultiIndicatorStrategy"
    display_name = "自适应多指标策略"
    description = "根据市场波动率自适应调整指标权重的策略"
    strategy_type = "adaptive"
    risk_level = "medium"
    
    parameters = MultiIndicatorStrategy.parameters + [
        StrategyParameter(
            name="volatility_threshold",
            display_name="波动率阈值",
            description="高波动率判断阈值(%)",
            value_type=float,
            default_value=3.0,
            min_value=1.0,
            max_value=10.0,
        ),
    ]
    
    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self._atr: ATR | None = None
        self._highs: list[float] = []
        self._lows: list[float] = []
    
    def initialize(self, context: StrategyContext) -> None:
        super().initialize(context)
        self._atr = ATR(period=14)
        self._highs = []
        self._lows = []
    
    def on_bar(self, bar: Bar, context: StrategyContext) -> StrategyResult:
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        
        result = super().on_bar(bar, context)
        
        if len(self._prices) >= 20:
            self._adapt_weights(bar)
        
        return result
    
    def _adapt_weights(self, bar: Bar) -> None:
        """根据市场状态调整权重"""
        if len(self._highs) < 14 or len(self._lows) < 14:
            return
        
        high = pd.Series(self._highs[-50:])
        low = pd.Series(self._lows[-50:])
        close = pd.Series(self._prices[-50:])
        
        atr_values = self._atr.calculate(high, low, close)
        atr_pct = float(atr_values.iloc[-1]) / bar.close * 100
        
        volatility_threshold = self._params.get("volatility_threshold", 3.0)
        
        if atr_pct > volatility_threshold:
            self._params["rsi_weight"] = 1.5
            self._params["bb_weight"] = 1.2
            self._params["macd_weight"] = 0.8
            self._params["ma_weight"] = 0.5
        else:
            self._params["rsi_weight"] = 0.8
            self._params["bb_weight"] = 0.6
            self._params["macd_weight"] = 1.2
            self._params["ma_weight"] = 1.0
