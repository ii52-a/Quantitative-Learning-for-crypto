from dataclasses import dataclass, field


"""策略判断信号"""
class PositionSignal:
    #开仓
    OPEN=1
    CLOSE=-1
    EMPTY=0
    #多头，空头
    MORE=2
    LESS=-2

"""策略返回信号"""
@dataclass
class StrategyResult:
    signal:int | None
    size: float        # 开仓权重(%资金)
    leverage: int      # 杠杆调整
    comment: str = ""  # TODO
    more_less: int | None = PositionSignal.EMPTY