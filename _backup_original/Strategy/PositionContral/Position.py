
from Strategy.StrategyTypes import *
from app_logger.LoggerType import PositionHistory
from app_logger.logger_setup import Logger

logger = Logger(__name__)


class Position:
    def __init__(self, symbol, leverage):
        self.symbol: str = symbol
        self._position_status: PositionStatus = PositionStatus(
            # 占用保证金
            margin_used=0,
            # 实际占用（预留扩展）
            true_margin_used=0,
            # 名义价值（含方向，正=多，负=空）
            nominal_value=0,
            # 平均开仓价格
            avg_price=0,
            # 杠杆
            leverage=leverage,
            # 开仓次数
            open_count=0,
            # 平仓次数
            close_count=0,
        )
        # 流动字段
        self._nominal_value = None
        self._price = None
        self._signal = None
        self._changed_used = None
        self._change_time = None

        # 单笔持仓窗口
        self._open_price = None
        self._open_time = None

        # 累计 pnl（当前持仓周期）
        self.open_all = 0
        self.d_pnl = 0

        self._d_log = None

    def true_margin(self, price) -> float:
        """
        提供调用以模拟过程真实保证金比例计算。
        """
        if self._position_status.avg_price == 0:
            return float(self._position_status.margin_used)
        cur_pnl = (price - self._position_status.avg_price) / self._position_status.avg_price * self._position_status.nominal_value
        return round(self._position_status.margin_used - cur_pnl, 6)

    @property
    def margin_usdt(self) -> float:
        return self._position_status.margin_used

    @property
    def direction(self) -> int:
        # === [FIX-1] 明确零仓位方向 ===
        # 历史实现中，nominal=0 时返回 1，会让部分边界场景被当成“多仓”处理。
        # 这里显式返回 0，避免方向判定被污染。
        nominal = self.get_nominal_value
        if nominal > 0:
            return 1
        if nominal < 0:
            return -1
        return 0

    def execute(self, position_set: PositionSet) -> PositionResult:
        """仓位变动处理入口。"""
        self._update_context(position_set)
        if position_set.signal == PositionChange.OPEN:
            return self._open()
        if position_set.signal == PositionChange.RESERVED:
            return self._close_and_reverse()
        if position_set.signal == PositionChange.PARTIAL:
            return self._close()
        if position_set.signal == PositionChange.FULL:
            return self._full_close()

        logger.error(f"{self.symbol}仓位：execute处理失败，信号无法处理")
        return PositionResult(if_full=False, win=False, pnl=0)

    def _update_context(self, position_set):
        """处理仓位变动参数。"""
        self._changed_used = max(float(position_set.changed_usdt), 0.0)
        self._signal = position_set.signal
        self._price = float(position_set.price)

        # === [FIX-2] 开仓/平仓名义价值方向独立计算 ===
        # 旧逻辑统一依赖 self.direction；在 0 仓、反手、部分平仓时容易出现方向混乱。
        # 新逻辑：
        # - OPEN：方向由信号语义决定（默认沿当前方向/空仓按多）
        # - CLOSE/FULL：方向与当前仓位相反（减少仓位）
        cur_dir = self.direction
        if self._signal == PositionChange.OPEN:
            signal_dir = 1 if cur_dir == 0 else cur_dir
        else:
            signal_dir = -cur_dir
        self._nominal_value = self._changed_used * self._position_status.leverage * signal_dir
        self._change_time = position_set.open_time

    def _close(self) -> PositionResult:
        """处理部分平仓逻辑。"""
        if self._position_status.margin_used <= 0 or self._position_status.avg_price <= 0:
            logger.warning(f"{self.symbol} 收到部分平仓信号但当前无持仓，忽略")
            return PositionResult(if_full=False, win=False, pnl=0)

        # === [FIX-3] 平仓金额上限保护 ===
        # 旧逻辑允许 changed_usdt > 当前保证金，可能导致 margin<0。
        changed = min(self._changed_used, self._position_status.margin_used)
        close_nominal = changed * self._position_status.leverage * self.direction
        pnl_delta = (self._price - self._position_status.avg_price) / self._position_status.avg_price * close_nominal

        self.d_pnl += pnl_delta
        self._position_status.margin_used -= changed
        self._position_status.nominal_value -= close_nominal

        # 浮点收敛：非常接近 0 时归零，避免残留导致“幽灵仓位”。
        if abs(self._position_status.margin_used) < 1e-8:
            self._position_status.margin_used = 0.0
        if abs(self._position_status.nominal_value) < 1e-8:
            self._position_status.nominal_value = 0.0
            self._position_status.avg_price = 0.0

        return PositionResult(if_full=False, win=self.d_pnl > 0, pnl=round(self.d_pnl, 6))

    def _full_close(self) -> PositionResult:
        if self._position_status.margin_used <= 0 or self._position_status.avg_price <= 0:
            return PositionResult(if_full=True, win=False, pnl=0)

        # === [FIX-4] 全平盈亏使用“当前持仓名义价值”，而非上下文临时值 ===
        current_nominal = self._position_status.nominal_value
        pnl: float = (self._price - self._position_status.avg_price) / self._position_status.avg_price * current_nominal

        self._position_status.margin_used = 0
        self._position_status.nominal_value = 0
        self._position_status.avg_price = 0
        self.d_pnl += pnl

        log: PositionHistory = self.get_log_standard(PositionChange.FULL, self.d_pnl)
        self._d_log = log
        logger.log_position_history(log)

        win: bool = self.d_pnl >= 0
        pnl_s = self.d_pnl
        self.d_pnl = 0
        self.open_all = 0
        return PositionResult(if_full=True, win=win, pnl=pnl_s)

    def _open(self) -> PositionResult:
        """处理开仓逻辑。"""
        # === [FIX-5] 防止 new_nominal_value=0 引发除零 ===
        old_nominal = self._position_status.nominal_value
        new_nominal_value = old_nominal + self._nominal_value
        if abs(new_nominal_value) < 1e-8:
            logger.warning(f"{self.symbol} 开仓后名义价值趋近0，拒绝本次更新")
            return PositionResult(if_full=False, win=False, pnl=0)

        cur_percent = 0.0 if abs(old_nominal) < 1e-8 else old_nominal / new_nominal_value
        add_percent = 1 - cur_percent
        self._position_status.avg_price = self._position_status.avg_price * cur_percent + self._price * add_percent

        self._position_status.margin_used += self._changed_used
        self._position_status.nominal_value = new_nominal_value
        self.open_all += self._changed_used

        if self._open_price is None:
            self._open_price = self._price
            self._open_time = self._change_time

        return PositionResult(if_full=False, win=False, pnl=0)

    def _close_and_reverse(self) -> PositionResult:
        # === [FIX-6] 反手逻辑重构 ===
        # 旧实现先计算 other_nominal 再 _full_close + _open，会混用旧上下文的 changed_usdt。
        # 新实现：
        # 1) 先完整平仓并结算
        # 2) 再用“相同保证金额度”按反向开新仓
        old_changed = self._changed_used
        close_result = self._full_close()

        self._changed_used = old_changed
        self._nominal_value = self._changed_used * self._position_status.leverage
        self._open_price = self._price
        self._open_time = self._change_time
        self._open()
        return close_result

    @property
    def get_nominal_value(self) -> float:
        return self._position_status.nominal_value

    @property
    def get_avg_price(self) -> float:
        return self._position_status.avg_price

    def get_d_log(self):
        if self._d_log is None:
            return None
        log = self._d_log
        self._d_log = None
        if hasattr(log.open_time, "tz_localize"):
            log.open_time = log.open_time.tz_localize(None)
        if hasattr(log.close_time, "tz_localize"):
            log.close_time = log.close_time.tz_localize(None)
        return log

    def get_log_standard(self, close_type: PositionChange, pnl: float) -> PositionHistory:
        pnl_percent = 0.0 if self.open_all == 0 else round(pnl * 100 / self.open_all, 6)
        return PositionHistory(
            close_type=close_type,
            symbol=self.symbol,
            leverage=self._position_status.leverage,
            open=self._open_price,
            open_time=self._open_time,
            close=self._price,
            close_time=self._change_time,
            pnl=round(pnl, 6),
            pnl_percent=pnl_percent,
        )

    def __str__(self):
        return f"{self._position_status}"
