"""
策略模板实现

包含多种主流交易策略
"""

import pandas as pd
from typing import Any

from strategy.base import (
    BaseStrategy,
    StrategyContext,
    StrategyResult,
    Signal,
    Bar,
    StrategyParameter,
)
from strategy.indicators import MACD, RSI, BollingerBands, MovingAverage, ATR
from core.constants import SignalType, PositionSide


class MACDStrategy(BaseStrategy):
    """MACD趋势跟踪策略
    
    策略逻辑：
    1. HIST从负变正且绝对值超过阈值 -> 金叉开多
    2. HIST从正变负且绝对值超过阈值 -> 死叉平多
    
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
            description="MACD柱状图过滤阈值，只有超过此值才触发信号",
            value_type=float,
            default_value=0.5,
            min_value=0.1,
            max_value=5.0,
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
    
    def initialize(self, context: StrategyContext) -> None:
        self._macd = MACD(
            fast_period=self._params.get("fast_period", 12),
            slow_period=self._params.get("slow_period", 26),
            signal_period=self._params.get("signal_period", 9),
        )
        self._prices = []
        self._initialized = True
    
    def on_bar(self, bar: Bar, context: StrategyContext) -> StrategyResult:
        self._prices.append(bar.close)
        
        min_data = max(
            self._params.get("slow_period", 26) + self._params.get("signal_period", 9),
            35
        )
        
        if len(self._prices) < min_data:
            return StrategyResult(
                log=f"数据收集中: {len(self._prices)}/{min_data}"
            )
        
        prices = pd.Series(self._prices[-200:])
        indicators = self._macd.get_latest(prices)
        
        hist_filter = self._params.get("hist_filter", 0.5)
        hist = indicators["hist"]
        hist_prev = indicators["hist_prev"]
        
        signal = None
        
        if abs(hist_prev) >= hist_filter:
            if hist_prev < 0 < hist and not context.has_position:
                signal = Signal(
                    type=SignalType.OPEN_LONG,
                    price=bar.close,
                    reason="MACD金叉开多",
                )
            elif hist_prev > 0 > hist and context.position.side == PositionSide.LONG:
                signal = Signal(
                    type=SignalType.CLOSE_LONG,
                    price=bar.close,
                    reason="MACD死叉平多",
                )
        
        return StrategyResult(
            signal=signal,
            indicators={
                "macd": indicators["macd"],
                "signal": indicators["signal"],
                "hist": hist,
                "hist_prev": hist_prev,
            },
            log=f"MACD={indicators['macd']:.2f}, Signal={indicators['signal']:.2f}, HIST={hist:.2f}"
        )
    
    def get_required_data_count(self) -> int:
        return self._params.get("slow_period", 26) + self._params.get("signal_period", 9) + 10


class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略（均线+MACD组合）
    
    策略逻辑：
    1. 价格在均线上方 + MACD金叉 -> 开多
    2. 价格跌破均线 或 MACD死叉 -> 平多
    
    适用场景：强趋势行情
    风险等级：中等
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
            default_value=0.3,
            min_value=0.1,
            max_value=5.0,
        ),
    ]
    
    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self._ma: MovingAverage | None = None
        self._macd: MACD | None = None
        self._prices: list[float] = []
    
    def initialize(self, context: StrategyContext) -> None:
        self._ma = MovingAverage(
            period=self._params.get("ma_period", 20),
            ma_type="EMA"
        )
        self._macd = MACD()
        self._prices = []
        self._initialized = True
    
    def on_bar(self, bar: Bar, context: StrategyContext) -> StrategyResult:
        self._prices.append(bar.close)
        
        min_data = max(self._params.get("ma_period", 20) + 10, 35)
        
        if len(self._prices) < min_data:
            return StrategyResult(log=f"数据收集中: {len(self._prices)}/{min_data}")
        
        prices = pd.Series(self._prices[-200:])
        
        ma_result = self._ma.get_latest(prices)
        macd_result = self._macd.get_latest(prices)
        
        ma_value = ma_result["ma"]
        price = bar.close
        above_ma = price > ma_value
        
        hist = macd_result["hist"]
        hist_prev = macd_result["hist_prev"]
        hist_filter = self._params.get("hist_filter", 0.3)
        
        signal = None
        
        if abs(hist_prev) >= hist_filter:
            if above_ma and hist_prev < 0 < hist and not context.has_position:
                signal = Signal(
                    type=SignalType.OPEN_LONG,
                    price=price,
                    reason="趋势向上+MACD金叉",
                )
            elif (not above_ma or hist_prev > 0 > hist) and context.position.side == PositionSide.LONG:
                signal = Signal(
                    type=SignalType.CLOSE_LONG,
                    price=price,
                    reason="趋势反转" if not above_ma else "MACD死叉",
                )
        
        return StrategyResult(
            signal=signal,
            indicators={
                "ma": ma_value,
                "above_ma": above_ma,
                "hist": hist,
            },
            log=f"MA={ma_value:.2f}, 价格{'>' if above_ma else '<'}MA, HIST={hist:.2f}"
        )
    
    def get_required_data_count(self) -> int:
        return self._params.get("ma_period", 20) + 35


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略（RSI+布林带）
    
    策略逻辑：
    1. RSI超卖 + 价格触及布林带下轨 -> 开多
    2. RSI超买 或 价格触及布林带上轨 -> 平多
    
    适用场景：震荡行情
    风险等级：中等
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
    
    def initialize(self, context: StrategyContext) -> None:
        self._rsi = RSI(period=self._params.get("rsi_period", 14))
        self._bb = BollingerBands(
            period=self._params.get("bb_period", 20),
            std_dev=self._params.get("bb_std", 2.0)
        )
        self._prices = []
        self._initialized = True
    
    def on_bar(self, bar: Bar, context: StrategyContext) -> StrategyResult:
        self._prices.append(bar.close)
        
        min_data = max(self._params.get("rsi_period", 14), self._params.get("bb_period", 20)) + 10
        
        if len(self._prices) < min_data:
            return StrategyResult(log=f"数据收集中: {len(self._prices)}/{min_data}")
        
        prices = pd.Series(self._prices[-200:])
        
        rsi_result = self._rsi.get_latest(prices)
        bb_result = self._bb.get_latest(prices)
        
        rsi = rsi_result["rsi"]
        oversold = self._params.get("rsi_oversold", 30)
        overbought = self._params.get("rsi_overbought", 70)
        
        bb_position = bb_result["position"]
        price = bar.close
        
        signal = None
        
        if rsi < oversold and bb_position < 0.1 and not context.has_position:
            signal = Signal(
                type=SignalType.OPEN_LONG,
                price=price,
                reason="RSI超卖+触及布林带下轨",
            )
        elif context.position.side == PositionSide.LONG:
            if rsi > overbought or bb_position > 0.9:
                signal = Signal(
                    type=SignalType.CLOSE_LONG,
                    price=price,
                    reason="RSI超买" if rsi > overbought else "触及布林带上轨",
                )
        
        return StrategyResult(
            signal=signal,
            indicators={
                "rsi": rsi,
                "bb_upper": bb_result["upper"],
                "bb_lower": bb_result["lower"],
                "bb_position": bb_position,
            },
            log=f"RSI={rsi:.1f}, BB位置={bb_position:.2f}"
        )
    
    def get_required_data_count(self) -> int:
        return max(self._params.get("rsi_period", 14), self._params.get("bb_period", 20)) + 10


class BollingerBandsStrategy(BaseStrategy):
    """布林带突破策略
    
    策略逻辑：
    1. 价格突破布林带下轨后回归 -> 开多
    2. 价格突破布林带上轨后回归 -> 平多
    
    适用场景：波动市场
    风险等级：中等
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
        self._prev_below_lower = False
        self._prev_above_upper = False
    
    def initialize(self, context: StrategyContext) -> None:
        self._bb = BollingerBands(
            period=self._params.get("period", 20),
            std_dev=self._params.get("std_dev", 2.0)
        )
        self._prices = []
        self._prev_below_lower = False
        self._prev_above_upper = False
        self._initialized = True
    
    def on_bar(self, bar: Bar, context: StrategyContext) -> StrategyResult:
        self._prices.append(bar.close)
        
        min_data = self._params.get("period", 20) + 5
        
        if len(self._prices) < min_data:
            return StrategyResult(log=f"数据收集中: {len(self._prices)}/{min_data}")
        
        prices = pd.Series(self._prices[-200:])
        bb_result = self._bb.get_latest(prices)
        
        price = bar.close
        upper = bb_result["upper"]
        lower = bb_result["lower"]
        middle = bb_result["middle"]
        
        below_lower = price < lower
        above_upper = price > upper
        
        signal = None
        
        if self._prev_below_lower and not below_lower and not context.has_position:
            signal = Signal(
                type=SignalType.OPEN_LONG,
                price=price,
                reason="布林带下轨反弹",
            )
        elif self._prev_above_upper and not above_upper and context.position.side == PositionSide.LONG:
            signal = Signal(
                type=SignalType.CLOSE_LONG,
                price=price,
                reason="布林带上轨回落",
            )
        
        self._prev_below_lower = below_lower
        self._prev_above_upper = above_upper
        
        return StrategyResult(
            signal=signal,
            indicators={
                "upper": upper,
                "middle": middle,
                "lower": lower,
                "position": bb_result["position"],
            },
            log=f"上轨={upper:.2f}, 中轨={middle:.2f}, 下轨={lower:.2f}"
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
    """获取策略实例"""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知的策略: {name}，可用策略: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](params)


def list_strategies() -> list[dict[str, Any]]:
    """列出所有可用策略"""
    return [
        get_strategy(name).get_info()
        for name in STRATEGY_REGISTRY
    ]
