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
    """MACD趋势跟踪策略（支持双向交易）
    
    策略逻辑：
    做多模式：
    1. HIST从负变正（金叉）-> 开多
    2. HIST从正变负（死叉）-> 平多
    
    做空模式：
    1. HIST从正变负（死叉）-> 开空
    2. HIST从负变正（金叉）-> 平空
    
    双向模式：
    1. 金叉 -> 开多/平空
    2. 死叉 -> 开空/平多
    
    适用场景：趋势行情
    风险等级：中等
    """
    
    name = "MACDStrategy"
    display_name = "MACD趋势策略"
    description = "基于MACD指标的趋势跟踪策略，支持做多/做空/双向交易"
    strategy_type = "trend_following"
    risk_level = "medium"
    
    parameters = [
        StrategyParameter(
            name="trade_mode",
            display_name="交易模式",
            description="交易方向：long_only=仅做多, short_only=仅做空, both=双向",
            value_type=str,
            default_value="long_only",
            options=["long_only", "short_only", "both"],
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
        trade_mode = self._params.get("trade_mode", "long_only")
        
        if self._prev_hist is not None:
            golden_cross = self._prev_hist <= 0 and curr_hist > 0
            death_cross = self._prev_hist >= 0 and curr_hist < 0
            
            has_long = context.position.side == PositionSide.LONG
            has_short = context.position.side == PositionSide.SHORT
            has_position = context.has_position
            
            if trade_mode == "long_only":
                if golden_cross and abs(curr_hist) > hist_filter and not has_position:
                    signal = Signal(
                        type=SignalType.OPEN_LONG,
                        price=bar.close,
                        reason=f"MACD金叉做多 (HIST: {self._prev_hist:.4f} -> {curr_hist:.4f})",
                    )
                elif death_cross and has_long:
                    signal = Signal(
                        type=SignalType.CLOSE_LONG,
                        price=bar.close,
                        reason=f"MACD死叉平多 (HIST: {self._prev_hist:.4f} -> {curr_hist:.4f})",
                    )
            
            elif trade_mode == "short_only":
                if death_cross and abs(curr_hist) > hist_filter and not has_position:
                    signal = Signal(
                        type=SignalType.OPEN_SHORT,
                        price=bar.close,
                        reason=f"MACD死叉做空 (HIST: {self._prev_hist:.4f} -> {curr_hist:.4f})",
                    )
                elif golden_cross and has_short:
                    signal = Signal(
                        type=SignalType.CLOSE_SHORT,
                        price=bar.close,
                        reason=f"MACD金叉平空 (HIST: {self._prev_hist:.4f} -> {curr_hist:.4f})",
                    )
            
            elif trade_mode == "both":
                if golden_cross and abs(curr_hist) > hist_filter:
                    if has_short:
                        signal = Signal(
                            type=SignalType.CLOSE_SHORT,
                            price=bar.close,
                            reason=f"MACD金叉平空 (HIST: {self._prev_hist:.4f} -> {curr_hist:.4f})",
                        )
                        self._pending_open_long = True
                    elif not has_position:
                        signal = Signal(
                            type=SignalType.OPEN_LONG,
                            price=bar.close,
                            reason=f"MACD金叉做多 (HIST: {self._prev_hist:.4f} -> {curr_hist:.4f})",
                        )
                
                elif death_cross and abs(curr_hist) > hist_filter:
                    if has_long:
                        signal = Signal(
                            type=SignalType.CLOSE_LONG,
                            price=bar.close,
                            reason=f"MACD死叉平多 (HIST: {self._prev_hist:.4f} -> {curr_hist:.4f})",
                        )
                        self._pending_open_short = True
                    elif not has_position:
                        signal = Signal(
                            type=SignalType.OPEN_SHORT,
                            price=bar.close,
                            reason=f"MACD死叉做空 (HIST: {self._prev_hist:.4f} -> {curr_hist:.4f})",
                        )
                
                if not has_position:
                    if getattr(self, '_pending_open_long', False):
                        signal = Signal(
                            type=SignalType.OPEN_LONG,
                            price=bar.close,
                            reason=f"MACD金叉做多 (延迟)",
                        )
                        self._pending_open_long = False
                    elif getattr(self, '_pending_open_short', False):
                        signal = Signal(
                            type=SignalType.OPEN_SHORT,
                            price=bar.close,
                            reason=f"MACD死叉做空 (延迟)",
                        )
                        self._pending_open_short = False
        
        self._prev_hist = curr_hist
        
        return StrategyResult(
            signal=signal,
            indicators={
                "macd": curr_macd,
                "signal": curr_signal,
                "hist": curr_hist,
            },
            log=f"MACD={curr_macd:.2f}, Signal={curr_signal:.2f}, HIST={curr_hist:.4f}"
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


class OrderFlowPullbackStrategy(BaseStrategy):
    """订单流超买加仓回撤策略

    逻辑：
    1. 在上涨榜中筛选量能适中的高动量币（实盘由交易器选币）
    2. 进入超买区后，按动量动态调整加仓价差，小额连续加仓抬升持仓成本
    3. 暴涨后出现回撤即分段止盈，回撤越深收益空间越大
    """

    name = "OrderFlowPullbackStrategy"
    display_name = "订单流-追涨策略"
    description = "追涨高动量币，超买区顺势加仓并在回调时止盈"
    strategy_type = "order_flow"
    risk_level = "high"

    parameters = [
        StrategyParameter(
            name="auto_select_symbol",
            display_name="自动选币",
            description="实盘时自动切换至高波动USDT交易对",
            value_type=str,
            default_value="true",
            options=["true", "false"],
        ),
        StrategyParameter(
            name="momentum_bars",
            display_name="动量K线数",
            description="动量判断窗口",
            value_type=int,
            default_value=4,
            min_value=2,
            max_value=10,
        ),
        StrategyParameter(
            name="overbought_rsi",
            display_name="超买RSI阈值",
            description="进入超买区才允许持续加仓",
            value_type=float,
            default_value=68.0,
            min_value=55.0,
            max_value=90.0,
        ),
        StrategyParameter(
            name="min_volume_ratio",
            display_name="最低量比",
            description="过滤量能不足币对",
            value_type=float,
            default_value=1.1,
            min_value=0.8,
            max_value=3.0,
        ),
        StrategyParameter(
            name="max_volume_ratio",
            display_name="最高量比",
            description="过滤过热币对，选择量能适中标的",
            value_type=float,
            default_value=3.0,
            min_value=1.2,
            max_value=8.0,
        ),
        StrategyParameter(
            name="base_add_position_pct",
            display_name="基础加仓比例%",
            description="低动量时每次小额加仓比例",
            value_type=float,
            default_value=8.0,
            min_value=1.0,
            max_value=30.0,
        ),
        StrategyParameter(
            name="base_price_gap_pct",
            display_name="基础加仓价差%",
            description="低动量小价差，持续跟进成本",
            value_type=float,
            default_value=0.25,
            min_value=0.05,
            max_value=2.0,
        ),
        StrategyParameter(
            name="momentum_gap_boost",
            display_name="动量价差增益",
            description="动量越大，加仓价差放大以防插针",
            value_type=float,
            default_value=0.9,
            min_value=0.1,
            max_value=3.0,
        ),
        StrategyParameter(
            name="pullback_take_profit_pct",
            display_name="回调止盈%",
            description="从阶段高点回撤达到阈值平仓",
            value_type=float,
            default_value=1.0,
            min_value=0.2,
            max_value=8.0,
        ),
        StrategyParameter(
            name="trail_activation_pct",
            display_name="启动追踪止盈%",
            description="盈利达到该比例后，启用追踪止盈",
            value_type=float,
            default_value=1.5,
            min_value=0.2,
            max_value=20.0,
        ),
        StrategyParameter(
            name="trailing_stop_pct",
            display_name="追踪回撤阈值%",
            description="已启动追踪后，从高点回撤达到阈值平仓",
            value_type=float,
            default_value=0.8,
            min_value=0.1,
            max_value=10.0,
        ),
        StrategyParameter(
            name="hard_stop_loss_pct",
            display_name="硬止损%",
            description="相对持仓均价的硬止损，防止急跌",
            value_type=float,
            default_value=4.0,
            min_value=0.5,
            max_value=20.0,
        ),
        StrategyParameter(
            name="max_add_count",
            display_name="最大加仓次数",
            description="单次持仓允许连续加仓次数上限",
            value_type=int,
            default_value=4,
            min_value=1,
            max_value=12,
        ),
        StrategyParameter(
            name="add_size_decay",
            display_name="加仓衰减系数",
            description="每次加仓后按该系数缩小后续加仓比例",
            value_type=float,
            default_value=0.9,
            min_value=0.5,
            max_value=1.2,
        ),
        StrategyParameter(
            name="momentum_add_boost",
            display_name="动量加仓增益",
            description="动量越强时放大加仓比例，增强顺势能力",
            value_type=float,
            default_value=0.25,
            min_value=0.0,
            max_value=2.0,
        ),
        StrategyParameter(
            name="cooldown_bars",
            display_name="平仓冷却K线",
            description="平仓后等待指定K线数再允许开仓，避免来回打脸",
            value_type=int,
            default_value=2,
            min_value=0,
            max_value=20,
        ),
    ]

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self._closes: list[float] = []
        self._volumes: list[float] = []
        self._peak_price: float = 0.0
        self._last_add_price: float = 0.0
        self._add_count: int = 0
        self._cooldown_left: int = 0
        self._rsi = RSI(period=14)

    def initialize(self, context: StrategyContext) -> None:
        self._closes = []
        self._volumes = []
        self._peak_price = 0.0
        self._last_add_price = 0.0
        self._add_count = 0
        self._cooldown_left = 0
        self._initialized = True

    def on_bar(self, bar: Bar, context: StrategyContext) -> StrategyResult:
        self._closes.append(bar.close)
        self._volumes.append(bar.volume)

        if len(self._closes) < 25:
            return StrategyResult(log=f"数据收集中: {len(self._closes)}/25")

        momentum_bars = int(self._params.get("momentum_bars", 4))
        overbought_rsi = float(self._params.get("overbought_rsi", 68.0))
        min_volume_ratio = float(self._params.get("min_volume_ratio", 1.1))
        max_volume_ratio = float(self._params.get("max_volume_ratio", 3.0))
        base_add_pct = float(self._params.get("base_add_position_pct", 8.0)) / 100
        base_gap_pct = float(self._params.get("base_price_gap_pct", 0.25))
        gap_boost = float(self._params.get("momentum_gap_boost", 0.9))
        pullback_take_profit = float(self._params.get("pullback_take_profit_pct", 1.0))
        trail_activation_pct = float(self._params.get("trail_activation_pct", 1.5))
        trailing_stop_pct = float(self._params.get("trailing_stop_pct", 0.8))
        hard_stop_loss_pct = float(self._params.get("hard_stop_loss_pct", 4.0))
        max_add_count = int(self._params.get("max_add_count", 4))
        add_size_decay = float(self._params.get("add_size_decay", 0.9))
        momentum_add_boost = float(self._params.get("momentum_add_boost", 0.25))
        cooldown_bars = int(self._params.get("cooldown_bars", 2))

        recent = pd.Series(self._closes[-(momentum_bars + 1):])
        momentum_pct = max(0.0, (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0] * 100)
        recent_closes = self._closes[-momentum_bars:]
        consecutive_up = all(recent_closes[i] > recent_closes[i - 1] for i in range(1, len(recent_closes)))

        avg_volume = float(pd.Series(self._volumes[-20:]).mean())
        curr_volume = self._volumes[-1]
        volume_boost = (curr_volume / avg_volume) if avg_volume > 0 else 0
        rsi_value = float(self._rsi.calculate(pd.Series(self._closes[-60:])).iloc[-1])
        in_overbought = rsi_value >= overbought_rsi

        signal = None

        if not context.has_position:
            if self._cooldown_left > 0:
                self._cooldown_left -= 1
                return StrategyResult(
                    signal=None,
                    indicators={
                        "volume_ratio": volume_boost,
                        "momentum_pct": momentum_pct,
                        "rsi": rsi_value,
                        "peak_price": self._peak_price,
                    },
                    log=f"冷却中({self._cooldown_left}), 量比={volume_boost:.2f}, RSI={rsi_value:.1f}"
                )
            if consecutive_up and in_overbought and min_volume_ratio <= volume_boost <= max_volume_ratio:
                signal = Signal(
                    type=SignalType.OPEN_LONG,
                    price=bar.close,
                    reason=(
                        f"订单流追涨开仓: RSI{rsi_value:.1f}超买 + 量比{volume_boost:.2f} + 动量{momentum_pct:.2f}%"
                    ),
                )
                self._peak_price = bar.close
                self._last_add_price = bar.close
                self._add_count = 0
        elif context.position.side == PositionSide.LONG:
            self._peak_price = max(self._peak_price, bar.close)

            dynamic_gap = base_gap_pct + momentum_pct * gap_boost
            add_gap_hit = self._last_add_price <= 0 or bar.close >= self._last_add_price * (1 + dynamic_gap / 100)
            dynamic_boost = 1 + momentum_pct * momentum_add_boost / 10
            decay_factor = add_size_decay ** self._add_count
            add_pct = min(0.35, base_add_pct * dynamic_boost * decay_factor)

            entry_drawdown_pct = (context.position.entry_price - bar.close) / context.position.entry_price * 100
            if entry_drawdown_pct >= hard_stop_loss_pct:
                signal = Signal(
                    type=SignalType.CLOSE_LONG,
                    price=bar.close,
                    reason=f"订单流追涨硬止损: 回撤{entry_drawdown_pct:.2f}%",
                )
                self._peak_price = 0.0
                self._last_add_price = 0.0
                self._add_count = 0
                self._cooldown_left = cooldown_bars
                return StrategyResult(
                    signal=signal,
                    indicators={"volume_ratio": volume_boost, "momentum_pct": momentum_pct, "rsi": rsi_value, "peak_price": self._peak_price},
                    log=f"硬止损触发, 回撤={entry_drawdown_pct:.2f}%"
                )

            if in_overbought and add_gap_hit and self._add_count < max_add_count:
                signal = Signal(
                    type=SignalType.OPEN_LONG,
                    price=bar.close,
                    reason=f"订单流追涨加仓: 动量{momentum_pct:.2f}% 价差{dynamic_gap:.2f}% 次数{self._add_count + 1}/{max_add_count}",
                    extra={"add_position_pct": add_pct},
                )
                self._last_add_price = bar.close
                self._add_count += 1
            else:
                pullback_pct = (self._peak_price - bar.close) / self._peak_price * 100 if self._peak_price > 0 else 0
                entry_gain_pct = (bar.close - context.position.entry_price) / context.position.entry_price * 100
                use_trailing = entry_gain_pct >= trail_activation_pct
                target_pullback = trailing_stop_pct if use_trailing else pullback_take_profit
                if pullback_pct >= target_pullback:
                    signal = Signal(
                        type=SignalType.CLOSE_LONG,
                        price=bar.close,
                        reason=(
                            f"订单流追涨{'追踪' if use_trailing else '回调'}止盈: "
                            f"回调{pullback_pct:.2f}% / 阈值{target_pullback:.2f}%"
                        ),
                    )
                    self._peak_price = 0.0
                    self._last_add_price = 0.0
                    self._add_count = 0
                    self._cooldown_left = cooldown_bars

        return StrategyResult(
            signal=signal,
            indicators={
                "volume_ratio": volume_boost,
                "momentum_pct": momentum_pct,
                "rsi": rsi_value,
                "peak_price": self._peak_price,
            },
            log=f"量比={volume_boost:.2f}, RSI={rsi_value:.1f}, 动量={momentum_pct:.2f}%, 峰值={self._peak_price:.2f}"
        )

    def get_required_data_count(self) -> int:
        momentum_bars = int(self._params.get("momentum_bars", 4))
        return max(25, momentum_bars + 20)


class OrderFlowWoolStrategy(BaseStrategy):
    """订单流-薅羊毛策略

    逻辑：
    1. 自动寻找涨幅榜高波动、量能适中的交易对（可关闭自动过滤）
    2. 在上涨超买区不断小额分批做空，动量越强则加仓价差越大，防止插针
    3. 暴涨后等待回撤，触发回调/追踪止盈后平仓“薅羊毛”
    """

    name = "OrderFlowWoolStrategy"
    display_name = "订单流-薅羊毛策略"
    description = "上涨超买区分批小额做空，按动量动态价差加仓，捕捉暴涨后的回撤"
    strategy_type = "order_flow"
    risk_level = "high"

    parameters = [
        StrategyParameter(name="auto_select_symbol", display_name="自动选币", description="实盘时自动切换至高波动USDT交易对", value_type=str, default_value="true", options=["true", "false"]),
        StrategyParameter(name="auto_filter_symbol", display_name="自动筛量", description="是否自动筛选量能适中的标的", value_type=str, default_value="true", options=["true", "false"]),
        StrategyParameter(name="momentum_bars", display_name="动量K线数", description="动量判断窗口", value_type=int, default_value=4, min_value=2, max_value=10),
        StrategyParameter(name="overbought_rsi", display_name="超买RSI阈值", description="进入超买区才允许持续加仓", value_type=float, default_value=68.0, min_value=55.0, max_value=90.0),
        StrategyParameter(name="min_volume_ratio", display_name="最低量比", description="过滤量能不足币对", value_type=float, default_value=1.1, min_value=0.8, max_value=3.0),
        StrategyParameter(name="max_volume_ratio", display_name="最高量比", description="过滤过热币对，选择量能适中标的", value_type=float, default_value=3.0, min_value=1.2, max_value=8.0),
        StrategyParameter(name="base_add_position_pct", display_name="基础加仓比例%", description="低动量时每次小额加仓比例", value_type=float, default_value=8.0, min_value=1.0, max_value=30.0),
        StrategyParameter(name="base_price_gap_pct", display_name="基础加仓价差%", description="低动量小价差，高动量拉大价差", value_type=float, default_value=0.25, min_value=0.05, max_value=2.0),
        StrategyParameter(name="momentum_gap_boost", display_name="动量价差增益", description="动量越大，加仓价差放大以防插针", value_type=float, default_value=0.9, min_value=0.1, max_value=3.0),
        StrategyParameter(name="pullback_take_profit_pct", display_name="回调止盈%", description="从阶段高点回撤达到阈值平仓", value_type=float, default_value=1.0, min_value=0.2, max_value=8.0),
        StrategyParameter(name="trail_activation_pct", display_name="启动追踪止盈%", description="盈利达到该比例后，启用追踪止盈", value_type=float, default_value=1.5, min_value=0.2, max_value=20.0),
        StrategyParameter(name="trailing_stop_pct", display_name="追踪回撤阈值%", description="已启动追踪后，从高点回撤达到阈值平仓", value_type=float, default_value=0.8, min_value=0.1, max_value=10.0),
        StrategyParameter(name="hard_stop_loss_pct", display_name="硬止损%", description="相对持仓均价的硬止损，防止继续暴涨", value_type=float, default_value=4.0, min_value=0.5, max_value=20.0),
        StrategyParameter(name="max_add_count", display_name="最大加仓次数", description="单次持仓允许连续加仓次数上限", value_type=int, default_value=4, min_value=1, max_value=12),
        StrategyParameter(name="add_size_decay", display_name="加仓衰减系数", description="每次加仓后按该系数缩小后续加仓比例", value_type=float, default_value=0.9, min_value=0.5, max_value=1.2),
        StrategyParameter(name="momentum_add_boost", display_name="动量加仓增益", description="动量越强时放大加仓比例，增强贴顶能力", value_type=float, default_value=0.25, min_value=0.0, max_value=2.0),
        StrategyParameter(name="cooldown_bars", display_name="平仓冷却K线", description="平仓后等待指定K线数再允许开仓", value_type=int, default_value=2, min_value=0, max_value=20),
    ]

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self._closes: list[float] = []
        self._volumes: list[float] = []
        self._peak_price: float = 0.0
        self._last_add_price: float = 0.0
        self._add_count: int = 0
        self._cooldown_left: int = 0
        self._rsi = RSI(period=14)

    def initialize(self, context: StrategyContext) -> None:
        self._closes.clear()
        self._volumes.clear()
        self._peak_price = 0.0
        self._last_add_price = 0.0
        self._add_count = 0
        self._cooldown_left = 0

    def on_bar(self, bar: Bar, context: StrategyContext) -> StrategyResult:
        self._closes.append(bar.close)
        self._volumes.append(bar.volume)

        momentum_bars = int(self._params.get("momentum_bars", 4))
        if len(self._closes) < max(30, momentum_bars + 2):
            return StrategyResult(log="数据不足")

        overbought_rsi = float(self._params.get("overbought_rsi", 68.0))
        min_volume_ratio = float(self._params.get("min_volume_ratio", 1.1))
        max_volume_ratio = float(self._params.get("max_volume_ratio", 3.0))
        auto_filter_symbol = str(self._params.get("auto_filter_symbol", "true")).lower() == "true"
        base_add_pct = float(self._params.get("base_add_position_pct", 8.0)) / 100
        base_gap_pct = float(self._params.get("base_price_gap_pct", 0.25))
        gap_boost = float(self._params.get("momentum_gap_boost", 0.9))
        pullback_take_profit = float(self._params.get("pullback_take_profit_pct", 1.0))
        trail_activation_pct = float(self._params.get("trail_activation_pct", 1.5))
        trailing_stop_pct = float(self._params.get("trailing_stop_pct", 0.8))
        hard_stop_loss_pct = float(self._params.get("hard_stop_loss_pct", 4.0))
        max_add_count = int(self._params.get("max_add_count", 4))
        add_size_decay = float(self._params.get("add_size_decay", 0.9))
        momentum_add_boost = float(self._params.get("momentum_add_boost", 0.25))
        cooldown_bars = int(self._params.get("cooldown_bars", 2))

        recent = pd.Series(self._closes[-(momentum_bars + 1):])
        momentum_pct = max(0.0, (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0] * 100)
        recent_closes = self._closes[-momentum_bars:]
        consecutive_up = all(recent_closes[i] > recent_closes[i - 1] for i in range(1, len(recent_closes)))

        avg_volume = float(pd.Series(self._volumes[-20:]).mean())
        curr_volume = self._volumes[-1]
        volume_boost = (curr_volume / avg_volume) if avg_volume > 0 else 0
        rsi_value = float(self._rsi.calculate(pd.Series(self._closes[-60:])).iloc[-1])
        in_overbought = rsi_value >= overbought_rsi
        volume_ok = (min_volume_ratio <= volume_boost <= max_volume_ratio) if auto_filter_symbol else (volume_boost >= min_volume_ratio)

        signal = None

        if not context.has_position:
            if self._cooldown_left > 0:
                self._cooldown_left -= 1
                return StrategyResult(signal=None, indicators={"volume_ratio": volume_boost, "momentum_pct": momentum_pct, "rsi": rsi_value, "peak_price": self._peak_price}, log=f"冷却中({self._cooldown_left}), 量比={volume_boost:.2f}, RSI={rsi_value:.1f}")

            if consecutive_up and in_overbought and volume_ok:
                signal = Signal(type=SignalType.OPEN_SHORT, price=bar.close, reason=(f"订单流薅羊毛开仓: RSI{rsi_value:.1f}超买 + 量比{volume_boost:.2f} + 动量{momentum_pct:.2f}%"))
                self._peak_price = bar.close
                self._last_add_price = bar.close
                self._add_count = 0

        elif context.position.side == PositionSide.SHORT:
            self._peak_price = max(self._peak_price, bar.close)

            dynamic_gap = base_gap_pct + momentum_pct * gap_boost
            add_gap_hit = self._last_add_price <= 0 or bar.close >= self._last_add_price * (1 + dynamic_gap / 100)
            dynamic_boost = 1 + momentum_pct * momentum_add_boost / 10
            decay_factor = add_size_decay ** self._add_count
            add_pct = min(0.35, base_add_pct * dynamic_boost * decay_factor)

            adverse_move_pct = (bar.close - context.position.entry_price) / context.position.entry_price * 100
            if adverse_move_pct >= hard_stop_loss_pct:
                signal = Signal(type=SignalType.CLOSE_SHORT, price=bar.close, reason=f"订单流薅羊毛硬止损: 上涨{adverse_move_pct:.2f}%")
                self._peak_price = 0.0
                self._last_add_price = 0.0
                self._add_count = 0
                self._cooldown_left = cooldown_bars
                return StrategyResult(signal=signal, indicators={"volume_ratio": volume_boost, "momentum_pct": momentum_pct, "rsi": rsi_value, "peak_price": self._peak_price}, log=f"硬止损触发, 上涨={adverse_move_pct:.2f}%")

            if in_overbought and add_gap_hit and self._add_count < max_add_count:
                signal = Signal(type=SignalType.OPEN_SHORT, price=bar.close, reason=f"订单流薅羊毛加仓: 动量{momentum_pct:.2f}% 价差{dynamic_gap:.2f}% 次数{self._add_count + 1}/{max_add_count}", extra={"add_position_pct": add_pct})
                self._last_add_price = bar.close
                self._add_count += 1
            else:
                pullback_pct = (self._peak_price - bar.close) / self._peak_price * 100 if self._peak_price > 0 else 0
                entry_gain_pct = (context.position.entry_price - bar.close) / context.position.entry_price * 100
                use_trailing = entry_gain_pct >= trail_activation_pct
                target_pullback = trailing_stop_pct if use_trailing else pullback_take_profit
                if pullback_pct >= target_pullback:
                    signal = Signal(type=SignalType.CLOSE_SHORT, price=bar.close, reason=(f"订单流薅羊毛{'追踪' if use_trailing else '回调'}止盈: 回调{pullback_pct:.2f}% / 阈值{target_pullback:.2f}%"))
                    self._peak_price = 0.0
                    self._last_add_price = 0.0
                    self._add_count = 0
                    self._cooldown_left = cooldown_bars

        return StrategyResult(signal=signal, indicators={"volume_ratio": volume_boost, "momentum_pct": momentum_pct, "rsi": rsi_value, "peak_price": self._peak_price}, log=f"量比={volume_boost:.2f}, RSI={rsi_value:.1f}, 动量={momentum_pct:.2f}%, 峰值={self._peak_price:.2f}")

    def get_required_data_count(self) -> int:
        momentum_bars = int(self._params.get("momentum_bars", 4))
        return max(25, momentum_bars + 20)


STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "MACDStrategy": MACDStrategy,
    "TrendFollowingStrategy": TrendFollowingStrategy,
    "MeanReversionStrategy": MeanReversionStrategy,
    "BollingerBandsStrategy": BollingerBandsStrategy,
    "OrderFlowPullbackStrategy": OrderFlowPullbackStrategy,
    "OrderFlowWoolStrategy": OrderFlowWoolStrategy,
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
