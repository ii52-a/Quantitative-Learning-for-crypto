
"""
策略返回信号
Strategy策略返回接口
"""
from dataclasses import dataclass
from typing import Hashable

"""
策略判断信号
"""
class PositionSignal:
    #开仓
    OPEN=1
    CLOSE=-1
    EMPTY=0


class Position:
    MORE=1
    LESS=-1



@dataclass
class StrategyResult:
    """

    :param signal:信号
    :param size:仓位比重
    :param execution_price:开仓价格
    :param execution_time:开仓时间
    :param more_less:多空方向
    :param comment:开仓原因
    """
    signal:int | None
    size: float        # 开仓权重(%资金)
    execution_price: float | None   #确定开仓价格
    execution_time: Hashable | None   #确定开仓日期
    more_less: int | None = PositionSignal.EMPTY
    comment: str = ""
