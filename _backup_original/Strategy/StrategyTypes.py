from enum import Enum
from dataclasses import dataclass
from typing import Hashable


class PositionSignal(Enum):
    LONG = 1
    SHORT = -1
    RESERVED = 0
    FULL = 4
    PARTIAL =2



class PositionChange(Enum):
    PARTIAL = "partial_close"  # 部分平仓
    FULL = "full_close"  # 全部平仓
    RESERVED = "reserved"  # 反向开仓
    OPEN = "open"  # 开仓
    ERROR = "error"




@dataclass
class PositionStatus:
    """
        margin_used:仓位占用保险金   \n
        true_margin_used:<UNK>   \n
        leverage:杠杆  \n
        average_price:平均开仓价格  \n
        open_count:开仓次数  \n
        close_count:平仓次数  \n
    """

    margin_used: float
    true_margin_used: float
    nominal_value: float
    avg_price: float
    leverage: int
    open_count: int
    close_count: int

@dataclass
class PositionSet:
    """
    signal<PositionSignal>: 多 / 空   \n
    margin_used:变动金额   \n
    price:价格   \n

    """
    signal: PositionChange
    changed_usdt: float
    price: float
    open_time:Hashable


@dataclass
class PositionResult:
    """
       pnl:平仓盈利   \n
    """
    if_full:bool
    win: bool
    pnl: float=0



@dataclass
class StrategyResult:
    """
    symbol:期货   \n
    size:仓位比重  \n
    execution_price:开仓价格  \n
    execution_time:开仓时间   \n
    more_less:多空方向    \n
    comment:开仓原因   \n
    """
    symbol:str
    size: float        # 开仓权重(%资金)
    execution_price: float | None   #确定开仓价格
    execution_time: Hashable | None   #确定开仓日期
    direction: PositionSignal
    comment: str = ""
