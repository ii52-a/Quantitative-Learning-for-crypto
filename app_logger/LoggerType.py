from dataclasses import dataclass

from Strategy.StrategyTypes import PositionChange


@dataclass
class PositionHistory:
    close_type:PositionChange
    symbol: str
    leverage: float
    open: float
    close: float
    open_time:any
    close_time:any
    pnl: float
