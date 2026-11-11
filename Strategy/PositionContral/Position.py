
import math

from Strategy.StrategyTypes import (
    PositionChange,
    PositionResult,
    PositionSet,
    PositionStatus,
)
from app_logger.LoggerType import PositionHistory
from app_logger.logger_setup import Logger

logger = Logger(__name__)


class Position:
    """单个交易对的仓位。

    核心状态是**带符号名义价值** nominal：
      nominal > 0 多头，nominal < 0 空头，nominal == 0 空仓。
    均价、保证金、已实现盈亏都由 nominal 推导，避免“方向 + 正数保证金”两套符号互相打架。
    """

    def __init__(self, symbol: str, leverage: int, fee_rate: float = 0.0004):
        self.symbol = symbol
        self.leverage = int(leverage)
        self.fee_rate = float(fee_rate)

        self.nominal: float = 0.0
        self.avg_price: float = 0.0
        self.margin_used: float = 0.0

        # 本笔交易累计
        self.d_pnl: float = 0.0          # 累计净盈亏（含手续费）
        self.fees: float = 0.0           # 本笔累计手续费
        self.open_all: float = 0.0       # 本笔累计投入保证金
        self._initial_margin: float = 0.0

        self._open_price: float | None = None
        self._open_time = None
        self._change_time = None
        self._price: float | None = None
        self._signal: PositionChange | None = None
        self._changed_used: float | None = None

    # ---------------- 兼容旧接口 ----------------
    @property
    def direction(self) -> int:
        if self.nominal > 0:
            return 1
        if self.nominal < 0:
            return -1
        return 0

    @property
    def get_nominal_value(self) -> float:
        return self.nominal

    @property
    def get_avg_price(self) -> float:
        return self.avg_price

    @property
    def margin_usdt(self) -> float:
        return self.margin_used

    def true_margin(self, price: float) -> float:
        """按当前价格估算真实占用保证金（含浮动盈亏），仅用于展示。"""
        if self.avg_price == 0:
            return float(self.margin_used)
        unrealized = (price / self.avg_price - 1.0) * self.nominal
        return round(self.margin_used + unrealized, 6)

    def status(self) -> PositionStatus:
        return PositionStatus(
            margin_used=self.margin_used,
            true_margin_used=self.margin_used,
            nominal_value=self.nominal,
            avg_price=self.avg_price,
            leverage=self.leverage,
            open_count=1 if self.nominal else 0,
            close_count=0,
        )

    # ---------------- 执行 ----------------
    def execute(self, position_set: PositionSet) -> PositionResult:
        price = position_set.price
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            raise ValueError(f"{self.symbol} 执行价格非法: {price}")
        price = float(price)
        self._price = price
        self._signal = position_set.signal
        self._changed_used = float(position_set.changed_usdt)
        self._change_time = position_set.open_time

        signal = position_set.signal
        if signal == PositionChange.OPEN:
            direction = position_set.direction
            if direction == 0:
                raise ValueError("开仓信号缺少方向")
            delta = direction * self._changed_used * self.leverage
            fee_pnl = self._apply_open(delta, price)
            return PositionResult(if_full=False, win=False, pnl=fee_pnl, fee=abs(delta) * self.fee_rate)

        if signal == PositionChange.FULL:
            return self._full_close(price, PositionChange.FULL)

        if signal == PositionChange.PARTIAL:
            delta = -self.direction * min(self._changed_used * self.leverage, abs(self.nominal))
            return self._partial_close(delta, price)

        if signal == PositionChange.RESERVED:
            new_direction = position_set.direction or -self.direction
            close_margin = self.margin_used
            result = self._full_close(price, PositionChange.RESERVED, reset=False)
            # 用等额保证金反手，新一笔只带新开仓的手续费
            open_delta = new_direction * close_margin * self.leverage
            open_cash = self._apply_open(open_delta, price)
            open_fee = abs(open_delta) * self.fee_rate
            self.d_pnl = -open_fee
            self.fees = open_fee
            self._initial_margin = abs(self.nominal) / self.leverage
            self.open_all = self._initial_margin
            self._open_price = price
            self._open_time = self._change_time
            return PositionResult(
                if_full=True, win=result.win, pnl=result.pnl + open_cash,
                trade_pnl=result.trade_pnl, fee=result.fee + open_fee,
                trade_fee=result.trade_fee,
                open_price=result.open_price, close_price=result.close_price,
                open_time=result.open_time, close_time=result.close_time,
                pnl_percent=result.pnl_percent,
            )

        logger.error(f"{self.symbol}仓位：execute处理失败，信号无法处理")
        return PositionResult(if_full=False, win=False, pnl=0.0)

    # ---------------- 内部 ----------------
    def _apply_open(self, delta: float, price: float) -> float:
        """开仓/加仓，返回本次现金流（手续费为负）。"""
        if delta == 0:
            return 0.0
        fee = abs(delta) * self.fee_rate
        if self.nominal == 0:
            self.avg_price = price
            self._open_price = price
            self._open_time = self._change_time
            self._initial_margin = abs(delta) / self.leverage
            self.open_all = self._initial_margin
            self.nominal = delta
        else:
            if self.nominal * delta < 0:
                raise ValueError("_apply_open 不能用于减仓")
            a, b = abs(self.nominal), abs(delta)
            self.avg_price = (self.avg_price * a + price * b) / (a + b)
            self.nominal += delta
            self.open_all += b / self.leverage
        self.margin_used = abs(self.nominal) / self.leverage
        self.d_pnl -= fee
        self.fees += fee
        return -fee

    def _apply_reduce(self, delta: float, price: float) -> tuple[float, float]:
        """减仓，返回 (本次现金流, 手续费)。delta 必须与持仓反向。"""
        if self.nominal == 0 or delta == 0:
            return 0.0, 0.0
        closed = min(abs(delta), abs(self.nominal)) * self.direction
        gross = (price / self.avg_price - 1.0) * closed
        fee = abs(closed) * self.fee_rate
        self.nominal -= closed
        self.margin_used = abs(self.nominal) / self.leverage
        self.d_pnl += gross - fee
        self.fees += fee
        if self.nominal == 0:
            self.avg_price = 0.0
        return gross - fee, fee

    def _partial_close(self, delta: float, price: float) -> PositionResult:
        cash, fee = self._apply_reduce(delta, price)
        return PositionResult(
            if_full=False,
            win=self.d_pnl > 0,
            pnl=cash,
            trade_pnl=self.d_pnl,
            fee=fee,
            close_price=price,
            close_time=self._change_time,
            pnl_percent=self._pnl_percent(self.d_pnl),
        )

    def _full_close(self, price: float, change_type: PositionChange, reset: bool = True) -> PositionResult:
        cash, fee = self._apply_reduce(-self.nominal, price)
        trade_pnl, pnl_percent = self.d_pnl, self._pnl_percent(self.d_pnl)
        win = trade_pnl > 0
        log: PositionHistory = self.get_log_standard(change_type, trade_pnl)
        logger.log_position_history(log)

        result = PositionResult(
            if_full=True, win=win, pnl=cash, trade_pnl=trade_pnl, fee=fee,
            trade_fee=self.fees,
            open_price=self._open_price, close_price=price,
            open_time=self._open_time, close_time=self._change_time,
            pnl_percent=pnl_percent,
        )
        if reset:
            self._reset_trade()
        return result

    def _reset_trade(self) -> None:
        self.nominal = 0.0
        self.avg_price = 0.0
        self.margin_used = 0.0
        self.d_pnl = 0.0
        self.fees = 0.0
        self.open_all = 0.0
        self._initial_margin = 0.0
        self._open_price = None
        self._open_time = None

    def _pnl_percent(self, pnl: float) -> float:
        if not self._initial_margin:
            return 0.0
        return pnl * 100.0 / self._initial_margin

    def get_log_standard(self, close_type: PositionChange, pnl: float) -> PositionHistory:
        return PositionHistory(
            close_type=close_type,
            symbol=self.symbol,
            leverage=self.leverage,
            open=self._open_price or 0.0,
            open_time=self._open_time,
            close=self._price or 0.0,
            close_time=self._change_time,
            pnl=round(pnl, 6),
            pnl_percent=round(self._pnl_percent(pnl), 6),
        )

    def __str__(self):
        return (f"{self.symbol} nominal={self.nominal:.4f} avg={self.avg_price:.4f} "
                f"margin={self.margin_used:.4f} d_pnl={self.d_pnl:.4f}")
