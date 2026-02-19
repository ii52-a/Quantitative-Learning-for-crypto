"""复合策略模块 - 支持多策略组合运行"""
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from Strategy.base import BaseStrategy, StrategyResult, Signal, StrategyParameter
from core.constants import SignalType


class CombineMethod(Enum):
    """策略组合方法"""
    VOTE = "vote"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"
    ANY = "any"


@dataclass
class StrategyWeight:
    """策略权重配置"""
    strategy: BaseStrategy
    weight: float = 1.0
    enabled: bool = True


@dataclass
class CompositeConfig:
    """复合策略配置"""
    combine_method: CombineMethod = CombineMethod.WEIGHTED
    min_vote_ratio: float = 0.5
    conflict_resolution: str = "none"


class CompositeStrategy(BaseStrategy):
    """复合策略 - 组合多个策略进行协同决策"""
    
    name = "CompositeStrategy"
    display_name = "复合策略"
    description = "组合多个策略进行协同决策"
    parameters = [
        StrategyParameter(
            name="combine_method",
            display_name="组合方法",
            description="策略信号组合方法",
            value_type=str,
            default_value="weighted",
            options=["vote", "weighted", "unanimous", "any"],
        ),
        StrategyParameter(
            name="min_vote_ratio",
            display_name="最小投票比例",
            description="触发信号所需的最小投票比例",
            value_type=float,
            default_value=0.5,
            min_value=0.1,
            max_value=1.0,
        ),
    ]
    
    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self._strategies: list[StrategyWeight] = []
        self._config = CompositeConfig()
        
        if params:
            method = params.get("combine_method", "weighted")
            self._config.combine_method = CombineMethod(method)
            self._config.min_vote_ratio = params.get("min_vote_ratio", 0.5)
    
    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0) -> None:
        """添加子策略"""
        self._strategies.append(StrategyWeight(
            strategy=strategy,
            weight=weight,
            enabled=True
        ))
    
    def remove_strategy(self, strategy_name: str) -> None:
        """移除子策略"""
        self._strategies = [s for s in self._strategies if s.strategy.name != strategy_name]
    
    def set_weight(self, strategy_name: str, weight: float) -> None:
        """设置策略权重"""
        for sw in self._strategies:
            if sw.strategy.name == strategy_name:
                sw.weight = weight
                break
    
    def initialize(self, context) -> None:
        """初始化所有子策略"""
        for sw in self._strategies:
            if sw.enabled:
                sw.strategy.initialize(context)
        self._initialized = True
    
    def on_bar(self, bar, context) -> StrategyResult:
        """处理K线数据，组合各策略信号"""
        signals = []
        weights = []
        
        for sw in self._strategies:
            if not sw.enabled:
                continue
            
            result = sw.strategy.on_bar(bar, context)
            if result and result.signal:
                signals.append(result.signal)
                weights.append(sw.weight)
        
        if not signals:
            return StrategyResult()
        
        combined_signal = self._combine_signals(signals, weights)
        
        return StrategyResult(signal=combined_signal)
    
    def _combine_signals(self, signals: list[Signal], weights: list[float]) -> Signal | None:
        """组合多个信号"""
        if not signals:
            return None
        
        open_long_signals = []
        open_short_signals = []
        close_signals = []
        
        for sig, weight in zip(signals, weights):
            if sig.type == SignalType.OPEN_LONG:
                open_long_signals.append((sig, weight))
            elif sig.type == SignalType.OPEN_SHORT:
                open_short_signals.append((sig, weight))
            elif sig.type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                close_signals.append((sig, weight))
        
        method = self._config.combine_method
        
        if method == CombineMethod.VOTE:
            return self._vote_combine(open_long_signals, open_short_signals, close_signals)
        elif method == CombineMethod.WEIGHTED:
            return self._weighted_combine(open_long_signals, open_short_signals, close_signals, weights)
        elif method == CombineMethod.UNANIMOUS:
            return self._unanimous_combine(open_long_signals, open_short_signals, close_signals, signals)
        else:
            return self._any_combine(open_long_signals, open_short_signals, close_signals)
    
    def _vote_combine(self, long_sigs, short_sigs, close_sigs) -> Signal | None:
        """投票组合"""
        total = len(long_sigs) + len(short_sigs) + len(close_sigs)
        if total == 0:
            return None
        
        threshold = self._config.min_vote_ratio * total
        
        if len(close_sigs) >= threshold:
            return close_sigs[0][0]
        if len(long_sigs) >= threshold:
            return long_sigs[0][0]
        if len(short_sigs) >= threshold:
            return short_sigs[0][0]
        
        return None
    
    def _weighted_combine(self, long_sigs, short_sigs, close_sigs, all_weights) -> Signal | None:
        """加权组合"""
        total_weight = sum(all_weights)
        if total_weight == 0:
            return None
        
        long_weight = sum(w for _, w in long_sigs)
        short_weight = sum(w for _, w in short_sigs)
        close_weight = sum(w for _, w in close_sigs)
        
        threshold = self._config.min_vote_ratio * total_weight
        
        if close_weight >= threshold:
            return close_sigs[0][0]
        if long_weight >= threshold and long_weight > short_weight:
            return long_sigs[0][0]
        if short_weight >= threshold and short_weight > long_weight:
            return short_sigs[0][0]
        
        return None
    
    def _unanimous_combine(self, long_sigs, short_sigs, close_sigs, all_signals) -> Signal | None:
        """一致通过组合"""
        if len(close_sigs) == len(all_signals):
            return close_sigs[0][0]
        if len(long_sigs) == len(all_signals):
            return long_sigs[0][0]
        if len(short_sigs) == len(all_signals):
            return short_sigs[0][0]
        
        return None
    
    def _any_combine(self, long_sigs, short_sigs, close_sigs) -> Signal | None:
        """任一信号触发"""
        if close_sigs:
            return close_sigs[0][0]
        if long_sigs:
            return long_sigs[0][0]
        if short_sigs:
            return short_sigs[0][0]
        
        return None
    
    def get_info(self) -> dict[str, Any]:
        """获取策略信息"""
        info = super().get_info()
        info["sub_strategies"] = [
            {
                "name": sw.strategy.name,
                "weight": sw.weight,
                "enabled": sw.enabled,
            }
            for sw in self._strategies
        ]
        return info


def create_composite_strategy(
    strategies: list[tuple[BaseStrategy, float]],
    combine_method: str = "weighted",
    min_vote_ratio: float = 0.5,
) -> CompositeStrategy:
    """创建复合策略的便捷函数"""
    composite = CompositeStrategy({
        "combine_method": combine_method,
        "min_vote_ratio": min_vote_ratio,
    })
    
    for strategy, weight in strategies:
        composite.add_strategy(strategy, weight)
    
    return composite
