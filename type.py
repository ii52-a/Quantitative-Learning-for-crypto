from dataclasses import dataclass
from typing import Hashable

import Config


from dataclasses import dataclass, field

from Config import BackConfig

"""
策略判断信号
"""
class PositionSignal:
    #开仓
    OPEN=1
    CLOSE=-1
    EMPTY=0
    #多头，空头
    MORE=2
    LESS=-2



"""
策略返回信号
Strategy策略返回接口
"""
@dataclass
class StrategyResult:
    signal:int | None
    size: float        # 开仓权重(%资金)
    leverage: int      # 杠杆调整
    execution_price: float | None   #确定开仓价格
    execution_time: Hashable | None   #确定开仓日期
    more_less: int | None = PositionSignal.EMPTY
    comment: str = ""


@dataclass
class BackTestSetting:
    k_line: str
    trading_pair: str
    strategy: str
    origin_usdt: float=BackConfig.ORIGIN_USDT
    leverage:int=BackConfig.SET_LEVERAGE