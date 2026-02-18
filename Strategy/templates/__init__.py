"""
策略模板实现

包含多种主流交易策略
"""

import pandas as pd
from typing import Any

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


class MACDStrategy(BaseStrategy):
    """MACD趋势跟踪策略
    
    策略逻辑：
    1. HIST从负变正（金叉）-> 开多
    2. HIST从正变负（死叉）-> 平多
    
    适用场景：趋势行情
    风险等级：中等
    """
    
    name = "MACDStrategy"
    display_name = "MACD趋势策略"
    description = "基于MACD指标的趋势跟踪策略，适合趋势行情"
    strategy_type = "trend_following"
    risk_level = "medium"
    
    parameters = [
        StrategyParameter(
            name="hist_filter",
            display_name="HIST过滤",
            description="MACD柱状图过滤阈值",
            value_type=float,
            default_value=0.0,
            min_value=0.0,
            max_value=10.0,
        ),
        StrategyParameter(
            name="fast_period",
            display_name="快线周期",
            description="MACD快线EMA周期",
            value_type=int,
            default_value=12,
            min_value=5,
            max_value=50,
        ),
        StrategyParameter(
            name="slow_period",
            display_name="慢线周期",
            description="MACD慢线EMA周期",
            value_type=int,
            default_value=26,
            min_value=10,
            max_value=100,
        ),
        StrategyParameter(
            name="signal_period",
            display_name="信号线周期",
            description="MACD信号线周期",
            value_type=int,
            default_value=9,
            min_value=3,
            max_value=20,
        ),
    ]
    
    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self._macd: MACD | None = None
        self._prices: list[float] = []
        self._prev_hist: float | None = None
    
    def initialize(self, context: StrategyContext) -> None:
        self._macd = MACD(
            fast_period=self._params.get("fast_period", 12),
            slow_period=self._params.get("slow_period", 26),
            signal_period=self._params.get("signal_period", 9),
        )
        self._prices = []
        self._prev_hist = None
        self._initialized = True
    
    def on_bar(self, bar: Bar, context: StrategyContext) -> StrategyResult:
        self._prices.append(bar.close)
        
        min_data = self._params.get("slow_period", 26) + self._params.get("signal_period", 9) + 5
        
        if len(self._prices) < min_data:
            return StrategyResult(log=f"数据收集中: {len(self._prices)}/{min_data}")
        
        prices = pd.Series(self._prices[-200:])
        macd_line, signal_line, hist = self._macd.calculate(prices)
        
        curr_hist = float(hist.iloc[-1])
        curr_macd = float(macd_line.iloc[-1])
        curr_signal = float(signal_line.iloc[-1])
        
        signal = None
        hist_filter = self._params.get("hist_filter", 0.0)
        
        if self._prev_hist is not None:
            if self._prev_hist < -hist_filter and curr_hist > hist_filter and not context.has_position:
                signal = Signal(
                    type=SignalType.OPEN_LONG,
                    price=bar.close,
                    reason=f"MACD金叉 (HIST: {self._prev_hist:.2f} -> {curr_hist:.2f})",
                )
            elif self._prev_hist > hist_filter and curr_hist < -hist_filter and context.position.side == PositionSide.LONG:
                signal = Signal(
                    type=SignalType.CLOSE_LONG,
                    price=bar.close,
                    reason=f"MACD死叉 (HIST: {self._prev_hist:.2f} -> {curr_hist:.2f})",
                )
        
        self._prev_hist = curr_hist
        
        return StrategyResult(
            signal=signal,
            indicators={
                "macd": curr_macd,
                "signal": curr_signal,
                "hist": curr_hist,
            },
            log=f"MACD={curr_macd:.2f}, Signal={curr_signal:.2f}, HIST={curr_hist:.2f}"
        )
    
    def get_required_data_count(self) -> int:
        return self._params.get("slow_period", 26) + self._params.get("signal_period", 9) + 10


class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略（均线+MACD组合）
    
    策略逻辑：
    1. 价格在均线上方 + MACD金叉 -> 开多
    2. 价格跌破均线 或 MACD死叉 -> 平多
    """
    
    name = "TrendFollowingStrategy"
    display_name = "趋势跟踪策略"
    description = "均线与MACD组合的趋势跟踪策略"
    strategy_type = "trend_following"
    risk_level = "medium"
    
    parameters = [
        StrategyParameter(
            name="ma_period",
            display_name="均线周期",
            description="趋势判断均线周期",
            value_type=int,
            default_value=20,
            min_value=5,
            max_value=100,
        ),
        StrategyParameter(
            name="hist_filter",
            display_name="HIST过滤",
            description="MACD柱状图过滤阈值",
            value_type=float,
            default_value=0.0,
            min_value=0.0,
            max_value=10.0,
        ),
    ]
    
    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self._ma: MovingAverage | None = None
        self._macd: MACD | None = None
        self._prices: list[float] = []
        self._prev_hist: float | None = None
    
    def initialize(self, context: StrategyContext) -> None:
        self._ma = MovingAverage(period=self._params.get("ma_period", 20), ma_type="EMA")
        self._macd = MACD()
        self._prices = []
        self._prev_hist = None
        self._initialized = True
    
    def on_bar(self, bar: Bar, context: StrategyContext) -> StrategyResult:
        self._prices.append(bar.close)
        
        min_data = max(self._params.get("ma_period", 20) + 5, 40)
        
        if len(self._prices) < min_data:
            return StrategyResult(log=f"数据收集中: {len(self._prices)}/{min_data}")
        
        prices = pd.Series(self._prices[-200:])
        
        ma_series = self._ma.calculate(prices)
        ma_value = float(ma_series.iloc[-1])
        
        _, _, hist = self._macd.calculate(prices)
        curr_hist = float(hist.iloc[-1])
        
        above_ma = bar.close > ma_value
        hist_filter = self._params.get("hist_filter", 0.0)
        
        signal = None
        
        if self._prev_hist is not None:
            if above_ma and self._prev_hist < -hist_filter and curr_hist > hist_filter and not context.has_position:
                signal = Signal(
                    type=SignalType.OPEN_LONG,
                    price=bar.close,
                    reason=f"趋势向上+MACD金叉",
                )
            elif context.position.side == PositionSide.LONG:
                if not above_ma or (self._prev_hist > hist_filter and curr_hist < -hist_filter):
                    signal = Signal(
                        type=SignalType.CLOSE_LONG,
                        price=bar.close,
                        reason="趋势反转" if not above_ma else "MACD死叉",
                    )
        
        self._prev_hist = curr_hist
        
        return StrategyResult(
            signal=signal,
            indicators={"ma": ma_value, "hist": curr_hist, "above_ma": above_ma},
            log=f"MA={ma_value:.2f}, 价格{'>' if above_ma else '<'}MA, HIST={curr_hist:.2f}"
        )
    
    def get_required_data_count(self) -> int:
        return max(self._params.get("ma_period", 20) + 5, 40)


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略（RSI+布林带）
    
    策略逻辑：
    1. RSI超卖 + 价格触及布林带下轨 -> 开多
    2. RSI超买 或 价格触及布林带上轨 -> 平多
    """
    
    name = "MeanReversionStrategy"
    display_name = "均值回归策略"
    description = "RSI与布林带组合的均值回归策略，适合震荡行情"
    strategy_type = "mean_reversion"
    risk_level = "medium"
    
    parameters = [
        StrategyParameter(
            name="rsi_period",
            display_name="RSI周期",
            description="RSI计算周期",
            value_type=int,
            default_value=14,
            min_value=5,
            max_value=30,
        ),
        StrategyParameter(
            name="rsi_oversold",
            display_name="超卖阈值",
            description="RSI超卖阈值",
            value_type=float,
            default_value=30.0,
            min_value=10.0,
            max_value=40.0,
        ),
        StrategyParameter(
            name="rsi_overbought",
            display_name="超买阈值",
            description="RSI超买阈值",
            value_type=float,
            default_value=70.0,
            min_value=60.0,
            max_value=90.0,
        ),
        StrategyParameter(
            name="bb_period",
            display_name="布林带周期",
            description="布林带计算周期",
            value_type=int,
            default_value=20,
            min_value=10,
            max_value=50,
        ),
        StrategyParameter(
            name="bb_std",
            display_name="布林带标准差",
            description="布林带标准差倍数",
            value_type=float,
            default_value=2.0,
            min_value=1.0,
            max_value=3.0,
        ),
    ]
    
    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self._rsi: RSI | None = None
        self._bb: BollingerBands | None = None
        self._prices: list[float] = []
        self._prev_rsi: float | None = None
    
    def initialize(self, context: StrategyContext) -> None:
        self._rsi = RSI(period=self._params.get("rsi_period", 14))
        self._bb = BollingerBands(
            period=self._params.get("bb_period", 20),
            std_dev=self._params.get("bb_std", 2.0)
        )
        self._prices = []
        self._prev_rsi = None
        self._initialized = True
    
    def on_bar(self, bar: Bar, context: StrategyContext) -> StrategyResult:
        self._prices.append(bar.close)
        
        min_data = max(self._params.get("rsi_period", 14), self._params.get("bb_period", 20)) + 5
        
        if len(self._prices) < min_data:
            return StrategyResult(log=f"数据收集中: {len(self._prices)}/{min_data}")
        
        prices = pd.Series(self._prices[-200:])
        
        rsi_series = self._rsi.calculate(prices)
        curr_rsi = float(rsi_series.iloc[-1])
        
        upper, middle, lower = self._bb.calculate(prices)
        bb_upper = float(upper.iloc[-1])
        bb_lower = float(lower.iloc[-1])
        
        oversold = self._params.get("rsi_oversold", 30)
        overbought = self._params.get("rsi_overbought", 70)
        
        signal = None
        
        if self._prev_rsi is not None:
            if self._prev_rsi < oversold and curr_rsi > oversold and bar.close <= bb_lower and not context.has_position:
                signal = Signal(
                    type=SignalType.OPEN_LONG,
                    price=bar.close,
                    reason=f"RSI超卖反弹+触及布林带下轨 (RSI: {curr_rsi:.1f})",
                )
            elif context.position.side == PositionSide.LONG:
                if curr_rsi > overbought or bar.close >= bb_upper:
                    signal = Signal(
                        type=SignalType.CLOSE_LONG,
                        price=bar.close,
                        reason=f"RSI超买" if curr_rsi > overbought else "触及布林带上轨",
                    )
        
        self._prev_rsi = curr_rsi
        
        return StrategyResult(
            signal=signal,
            indicators={"rsi": curr_rsi, "bb_upper": bb_upper, "bb_lower": bb_lower},
            log=f"RSI={curr_rsi:.1f}, BB下轨={bb_lower:.2f}, BB上轨={bb_upper:.2f}"
        )
    
    def get_required_data_count(self) -> int:
        return max(self._params.get("rsi_period", 14), self._params.get("bb_period", 20)) + 5


class BollingerBandsStrategy(BaseStrategy):
    """布林带突破策略
    
    策略逻辑：
    1. 价格突破布林带下轨后回归 -> 开多
    2. 价格突破布林带上轨后回归 -> 平多
    """
    
    name = "BollingerBandsStrategy"
    display_name = "布林带策略"
    description = "基于布林带的突破回归策略"
    strategy_type = "mean_reversion"
    risk_level = "medium"
    
    parameters = [
        StrategyParameter(
            name="period",
            display_name="计算周期",
            description="布林带计算周期",
            value_type=int,
            default_value=20,
            min_value=10,
            max_value=50,
        ),
        StrategyParameter(
            name="std_dev",
            display_name="标准差倍数",
            description="布林带标准差倍数",
            value_type=float,
            default_value=2.0,
            min_value=1.0,
            max_value=3.0,
        ),
    ]
    
    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self._bb: BollingerBands | None = None
        self._prices: list[float] = []
        self._was_below_lower: bool = False
        self._was_above_upper: bool = False
    
    def initialize(self, context: StrategyContext) -> None:
        self._bb = BollingerBands(
            period=self._params.get("period", 20),
            std_dev=self._params.get("std_dev", 2.0)
        )
        self._prices = []
        self._was_below_lower = False
        self._was_above_upper = False
        self._initialized = True
    
    def on_bar(self, bar: Bar, context: StrategyContext) -> StrategyResult:
        self._prices.append(bar.close)
        
        min_data = self._params.get("period", 20) + 5
        
        if len(self._prices) < min_data:
            return StrategyResult(log=f"数据收集中: {len(self._prices)}/{min_data}")
        
        prices = pd.Series(self._prices[-200:])
        upper, middle, lower = self._bb.calculate(prices)
        
        bb_upper = float(upper.iloc[-1])
        bb_lower = float(lower.iloc[-1])
        bb_middle = float(middle.iloc[-1])
        
        below_lower = bar.close < bb_lower
        above_upper = bar.close > bb_upper
        
        signal = None
        
        if self._was_below_lower and not below_lower and not context.has_position:
            signal = Signal(
                type=SignalType.OPEN_LONG,
                price=bar.close,
                reason=f"布林带下轨反弹",
            )
        elif self._was_above_upper and not above_upper and context.position.side == PositionSide.LONG:
            signal = Signal(
                type=SignalType.CLOSE_LONG,
                price=bar.close,
                reason=f"布林带上轨回落",
            )
        
        self._was_below_lower = below_lower
        self._was_above_upper = above_upper
        
        return StrategyResult(
            signal=signal,
            indicators={"upper": bb_upper, "middle": bb_middle, "lower": bb_lower},
            log=f"上轨={bb_upper:.2f}, 中轨={bb_middle:.2f}, 下轨={bb_lower:.2f}"
        )
    
    def get_required_data_count(self) -> int:
        return self._params.get("period", 20) + 5


STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "MACDStrategy": MACDStrategy,
    "TrendFollowingStrategy": TrendFollowingStrategy,
    "MeanReversionStrategy": MeanReversionStrategy,
    "BollingerBandsStrategy": BollingerBandsStrategy,
}


def get_strategy(name: str, params: dict[str, Any] | None = None) -> BaseStrategy:
    if name not in STRATEGY_REGISTRY:
        from Strategy.multi_indicator_strategy import MultiIndicatorStrategy, AdaptiveMultiIndicatorStrategy
        if name == "MultiIndicatorStrategy":
            return MultiIndicatorStrategy(params)
        elif name == "AdaptiveMultiIndicatorStrategy":
            return AdaptiveMultiIndicatorStrategy(params)
        raise ValueError(f"未知的策略: {name}")
    return STRATEGY_REGISTRY[name](params)


def list_strategies() -> list[dict[str, Any]]:
    strategies = [get_strategy(name).get_info() for name in STRATEGY_REGISTRY]
    
    try:
        from Strategy.multi_indicator_strategy import MultiIndicatorStrategy, AdaptiveMultiIndicatorStrategy
        strategies.append(MultiIndicatorStrategy().get_info())
        strategies.append(AdaptiveMultiIndicatorStrategy().get_info())
    except ImportError:
        pass
    
    return strategies
