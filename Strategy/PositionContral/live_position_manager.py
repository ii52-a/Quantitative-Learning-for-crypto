from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Any, Callable

from Strategy.PositionContral.live_position import LivePosition, Side
from app_logger.logger_setup import Logger

logger = Logger(__name__)


@dataclass
class TradeRecord:
    symbol: str
    side: Side
    action: str
    quantity: float
    price: float
    margin: float
    pnl: float
    timestamp: datetime
    reason: str = ""


@dataclass
class RiskConfig:
    max_position_ratio: float = 0.3
    max_single_loss_ratio: float = 0.05
    default_leverage: int = 5
    default_stop_loss_pct: float = 0.02
    default_take_profit_pct: float = 0.05


class LivePositionManager:
    def __init__(
        self,
        initial_balance: float = 1000.0,
        risk_config: RiskConfig | None = None,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_config = risk_config or RiskConfig()
        self.positions: dict[str, LivePosition] = {}
        self.trade_history: list[TradeRecord] = []
        self._order_callback: Callable[[TradeRecord], None] | None = None

    @property
    def total_margin(self) -> float:
        return sum(p.margin for p in self.positions.values() if p.is_open)

    @property
    def available_balance(self) -> float:
        return self.balance - self.total_margin

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values() if p.is_open)

    @property
    def total_equity(self) -> float:
        return self.balance + self.total_unrealized_pnl

    @property
    def open_positions(self) -> list[LivePosition]:
        return [p for p in self.positions.values() if p.is_open]

    def set_order_callback(self, callback: Callable[[TradeRecord], None]) -> None:
        self._order_callback = callback

    def get_position(self, symbol: str) -> LivePosition:
        if symbol not in self.positions:
            self.positions[symbol] = LivePosition(symbol=symbol)
        return self.positions[symbol]

    def calc_quantity(self, symbol: str, price: float, risk_usdt: float | None = None) -> float:
        if risk_usdt is None:
            risk_usdt = self.available_balance * self.risk_config.max_position_ratio
        if price <= 0:
            return 0.0
        qty = risk_usdt * self.risk_config.default_leverage / price
        return math.floor(qty * 1000) / 1000

    def open_position(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        reason: str = "",
    ) -> LivePosition | None:
        if side == Side.EMPTY:
            logger.warning(f"[{symbol}] 无效方向: {side}")
            return None

        pos = self.get_position(symbol)
        if pos.is_open:
            if pos.side != side:
                logger.warning(f"[{symbol}] 已有反向持仓，请先平仓")
                return None
            return self._add_position(pos, side, quantity, price, stop_loss, take_profit, reason)

        margin = quantity * price / self.risk_config.default_leverage
        if margin > self.available_balance:
            logger.warning(f"[{symbol}] 保证金不足: 需要 {margin:.2f}, 可用 {self.available_balance:.2f}")
            return None

        if stop_loss is None:
            if side == Side.LONG:
                stop_loss = price * (1 - self.risk_config.default_stop_loss_pct)
            else:
                stop_loss = price * (1 + self.risk_config.default_stop_loss_pct)

        if take_profit is None:
            if side == Side.LONG:
                take_profit = price * (1 + self.risk_config.default_take_profit_pct)
            else:
                take_profit = price * (1 - self.risk_config.default_take_profit_pct)

        pos.side = side
        pos.quantity = quantity
        pos.entry_price = price
        pos.margin = margin
        pos.leverage = self.risk_config.default_leverage
        pos.entry_time = datetime.now()
        pos.stop_loss = stop_loss
        pos.take_profit = take_profit
        pos.unrealized_pnl = 0.0

        record = TradeRecord(
            symbol=symbol,
            side=side,
            action="OPEN",
            quantity=quantity,
            price=price,
            margin=margin,
            pnl=0.0,
            timestamp=datetime.now(),
            reason=reason,
        )
        self.trade_history.append(record)
        if self._order_callback:
            self._order_callback(record)

        logger.info(f"[{symbol}] 开仓成功: {side.value} {quantity} @ {price}")
        return pos

    def _add_position(
        self,
        pos: LivePosition,
        side: Side,
        quantity: float,
        price: float,
        stop_loss: float | None,
        take_profit: float | None,
        reason: str,
    ) -> LivePosition | None:
        add_margin = quantity * price / pos.leverage
        if add_margin > self.available_balance:
            logger.warning(f"[{pos.symbol}] 加仓保证金不足")
            return None

        total_qty = pos.quantity + quantity
        total_value = pos.quantity * pos.entry_price + quantity * price
        new_avg_price = total_value / total_qty

        pos.quantity = total_qty
        pos.entry_price = new_avg_price
        pos.margin += add_margin

        if stop_loss is not None:
            pos.stop_loss = stop_loss
        if take_profit is not None:
            pos.take_profit = take_profit

        record = TradeRecord(
            symbol=pos.symbol,
            side=side,
            action="ADD",
            quantity=quantity,
            price=price,
            margin=add_margin,
            pnl=0.0,
            timestamp=datetime.now(),
            reason=reason,
        )
        self.trade_history.append(record)
        if self._order_callback:
            self._order_callback(record)

        logger.info(f"[{pos.symbol}] 加仓成功: +{quantity} @ {price}, 均价: {new_avg_price:.4f}")
        return pos

    def close_position(
        self,
        symbol: str,
        quantity: float | None = None,
        price: float | None = None,
        reason: str = "",
    ) -> TradeRecord | None:
        pos = self.get_position(symbol)
        if not pos.is_open:
            logger.warning(f"[{symbol}] 无持仓可平")
            return None

        if price is None:
            logger.warning(f"[{symbol}] 平仓价格不能为空")
            return None

        close_qty = quantity if quantity is not None else pos.quantity
        close_qty = min(close_qty, pos.quantity)

        pnl = pos.calc_pnl(price) * (close_qty / pos.quantity) if pos.quantity > 0 else 0.0
        close_margin = pos.margin * (close_qty / pos.quantity) if pos.quantity > 0 else 0.0

        record = TradeRecord(
            symbol=symbol,
            side=pos.side,
            action="CLOSE",
            quantity=close_qty,
            price=price,
            margin=close_margin,
            pnl=pnl,
            timestamp=datetime.now(),
            reason=reason,
        )

        if close_qty >= pos.quantity:
            pos.side = Side.EMPTY
            pos.quantity = 0.0
            pos.entry_price = 0.0
            pos.margin = 0.0
            pos.unrealized_pnl = 0.0
            pos.stop_loss = None
            pos.take_profit = None
        else:
            pos.quantity -= close_qty
            pos.margin -= close_margin
            pos.unrealized_pnl = pos.calc_pnl(price)

        self.balance += pnl
        self.trade_history.append(record)
        if self._order_callback:
            self._order_callback(record)

        logger.info(f"[{symbol}] 平仓成功: {close_qty} @ {price}, PnL: {pnl:.4f}")
        return record

    def close_all_positions(self, prices: dict[str, float], reason: str = "全部平仓") -> list[TradeRecord]:
        records = []
        for symbol, pos in list(self.positions.items()):
            if pos.is_open and symbol in prices:
                record = self.close_position(symbol, price=prices[symbol], reason=reason)
                if record:
                    records.append(record)
        return records

    def update_pnl(self, prices: dict[str, float]) -> None:
        for symbol, price in prices.items():
            pos = self.get_position(symbol)
            if pos.is_open:
                pos.unrealized_pnl = pos.calc_pnl(price)

    def check_risk_events(self, prices: dict[str, float]) -> list[dict[str, Any]]:
        events = []
        for symbol, price in prices.items():
            pos = self.get_position(symbol)
            if not pos.is_open:
                continue

            if pos.check_stop_loss(price):
                events.append({
                    "symbol": symbol,
                    "type": "STOP_LOSS",
                    "price": price,
                    "position": pos.to_dict(),
                })
            elif pos.check_take_profit(price):
                events.append({
                    "symbol": symbol,
                    "type": "TAKE_PROFIT",
                    "price": price,
                    "position": pos.to_dict(),
                })
        return events

    def handle_risk_events(self, prices: dict[str, float]) -> list[TradeRecord]:
        events = self.check_risk_events(prices)
        records = []
        for event in events:
            symbol = event["symbol"]
            price = event["price"]
            reason = f"触发{event['type']}"
            record = self.close_position(symbol, price=price, reason=reason)
            if record:
                records.append(record)
        return records

    def get_statistics(self) -> dict[str, Any]:
        closed_trades = [t for t in self.trade_history if t.action == "CLOSE"]
        total_trades = len(closed_trades)
        win_trades = len([t for t in closed_trades if t.pnl > 0])
        lose_trades = len([t for t in closed_trades if t.pnl < 0])
        total_pnl = sum(t.pnl for t in closed_trades)

        return {
            "initial_balance": self.initial_balance,
            "current_balance": self.balance,
            "total_equity": self.total_equity,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_margin": self.total_margin,
            "available_balance": self.available_balance,
            "total_trades": total_trades,
            "win_trades": win_trades,
            "lose_trades": lose_trades,
            "win_rate": win_trades / total_trades * 100 if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "return_pct": (self.balance - self.initial_balance) / self.initial_balance * 100,
        }

    def summary(self) -> str:
        stats = self.get_statistics()
        lines = [
            "=" * 50,
            "账户概览",
            "=" * 50,
            f"初始资金: {stats['initial_balance']:.2f} USDT",
            f"当前余额: {stats['current_balance']:.2f} USDT",
            f"总权益: {stats['total_equity']:.2f} USDT",
            f"未实现盈亏: {stats['total_unrealized_pnl']:.4f} USDT",
            f"占用保证金: {stats['total_margin']:.2f} USDT",
            f"可用余额: {stats['available_balance']:.2f} USDT",
            "-" * 50,
            f"总交易次数: {stats['total_trades']}",
            f"盈利次数: {stats['win_trades']} | 亏损次数: {stats['lose_trades']}",
            f"胜率: {stats['win_rate']:.2f}%",
            f"累计盈亏: {stats['total_pnl']:.4f} USDT",
            f"收益率: {stats['return_pct']:.2f}%",
            "=" * 50,
        ]
        return "\n".join(lines)
