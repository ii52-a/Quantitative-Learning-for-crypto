from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Side(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EMPTY = "EMPTY"


@dataclass(slots=True)
class LivePosition:
    symbol: str
    side: Side = Side.EMPTY
    quantity: float = 0.0
    entry_price: float = 0.0
    margin: float = 0.0
    leverage: int = 1
    unrealized_pnl: float = 0.0
    entry_time: datetime | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def notional_value(self) -> float:
        return self.quantity * self.entry_price

    @property
    def is_open(self) -> bool:
        return self.side != Side.EMPTY and self.quantity > 0

    def calc_pnl(self, current_price: float) -> float:
        if not self.is_open or self.entry_price <= 0:
            return 0.0
        if self.side == Side.LONG:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def calc_pnl_percent(self, current_price: float) -> float:
        if not self.is_open or self.entry_price <= 0:
            return 0.0
        if self.side == Side.LONG:
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price

    def check_stop_loss(self, current_price: float) -> bool:
        if not self.is_open or self.stop_loss is None:
            return False
        if self.side == Side.LONG:
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss

    def check_take_profit(self, current_price: float) -> bool:
        if not self.is_open or self.take_profit is None:
            return False
        if self.side == Side.LONG:
            return current_price >= self.take_profit
        else:
            return current_price <= self.take_profit

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "margin": self.margin,
            "leverage": self.leverage,
            "unrealized_pnl": self.unrealized_pnl,
            "entry_time": str(self.entry_time) if self.entry_time else None,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }

    def __str__(self) -> str:
        if not self.is_open:
            return f"[{self.symbol}] 空仓"
        pnl_str = f"PnL: {self.unrealized_pnl:.4f}"
        return f"[{self.symbol}] {self.side.value} | 数量: {self.quantity} | 入场: {self.entry_price} | {pnl_str}"
