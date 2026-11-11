from Config import BackConfig
from Strategy.PositionContral.Position import Position
from Strategy.StrategyTypes import (
    PositionChange,
    PositionResult,
    PositionSet,
    PositionSignal,
    StrategyResult,
)
from app_logger.logger_setup import Logger

logger = Logger(__name__)


class PositionControl:
    """把策略信号翻译成仓位动作，并维护账户余额、手续费与交易统计。"""

    def __init__(self, usdt: float = 1000, leverage: int | None = None, fee_rate: float | None = None):
        self.position: dict[str, Position] = {}
        self.init_usdt: float = float(usdt)
        self.all_usdt: float = float(usdt)   # 已实现余额（含手续费）
        self.leverage: int = int(BackConfig.SET_LEVERAGE if leverage is None else leverage)
        self.fee_rate: float = float(BackConfig.OPEN_FEE_RADIO if fee_rate is None else fee_rate)

        self.total = 0
        self.win = 0
        self.lose = 0
        self.breakeven = 0
        self.fees_total: float = 0.0
        self.trades: list[dict] = []

    @property
    def _usdt(self) -> float:
        # 可用资金 = 当前已实现余额 - 仓位占用保证金
        return self.all_usdt - self.margin_usdt

    @property
    def margin_usdt(self) -> float:
        return sum(p.margin_usdt for p in self.position.values())

    @property
    def status(self) -> bool:
        return len(self.position) > 0

    def _sign_transform(self, symbol: str, strategy_result: StrategyResult):
        """返回 (变动保证金, 动作, 方向)；动作为 None 表示忽略该信号。"""
        pos = self.position[symbol]
        signal = strategy_result.direction
        size = max(float(strategy_result.size), 0.0)
        flat = pos.get_avg_price == 0

        if flat:
            # 空仓只接受开仓信号，平仓/部分平仓直接忽略
            if signal == PositionSignal.LONG:
                return size * max(self._usdt, 0.0), PositionChange.OPEN, 1
            if signal == PositionSignal.SHORT:
                return size * max(self._usdt, 0.0), PositionChange.OPEN, -1
            return 0.0, None, 0

        if signal == PositionSignal.FULL:
            return pos.margin_usdt, PositionChange.FULL, 0
        if signal == PositionSignal.PARTIAL:
            return size * pos.margin_usdt, PositionChange.PARTIAL, 0
        if signal == PositionSignal.RESERVED:
            return pos.margin_usdt, PositionChange.RESERVED, -pos.direction

        if pos.direction == 1 and signal == PositionSignal.LONG:
            return size * max(self._usdt, 0.0), PositionChange.OPEN, 1
        if pos.direction == -1 and signal == PositionSignal.SHORT:
            return size * max(self._usdt, 0.0), PositionChange.OPEN, -1
        if pos.direction == 1 and signal == PositionSignal.SHORT:
            return pos.margin_usdt, PositionChange.RESERVED, -1
        if pos.direction == -1 and signal == PositionSignal.LONG:
            return pos.margin_usdt, PositionChange.RESERVED, 1
        return 0.0, None, 0

    def main(self, strategy_result: StrategyResult) -> PositionResult | None:
        symbol = strategy_result.symbol
        if strategy_result.execution_price is None or strategy_result.execution_time is None:
            logger.warning(f"[{symbol}] 信号缺少价格或时间，忽略")
            return None

        if symbol not in self.position:
            self.position[symbol] = Position(symbol, self.leverage, self.fee_rate)

        changed_usdt, change_type, direction = self._sign_transform(symbol, strategy_result)

        if change_type is None:
            self._cleanup(symbol)
            return None

        if change_type == PositionChange.OPEN:
            if changed_usdt <= 0:
                logger.warning(f"[{symbol}] 可用资金不足，拒绝开仓: {changed_usdt:.4f}")
                self._cleanup(symbol)
                return None
            if changed_usdt > self._usdt + 1e-9:
                logger.warning(f"[{symbol}] 资金不足拒绝开仓: {changed_usdt:.2f} > 剩{self._usdt:.2f}")
                self._cleanup(symbol)
                return None

        pos_set = PositionSet(
            signal=change_type,
            changed_usdt=changed_usdt,
            price=strategy_result.execution_price,
            open_time=strategy_result.execution_time,
            direction=direction,
        )
        result: PositionResult = self.position[symbol].execute(pos_set)

        # 余额与手续费
        self.all_usdt += result.pnl
        self.fees_total += result.fee

        if result.if_full:
            self.total += 1
            if result.trade_pnl > 0:
                self.win += 1
            elif result.trade_pnl < 0:
                self.lose += 1
            else:
                self.breakeven += 1
            self.trades.append({
                'close_type': str(change_type),
                'symbol': symbol,
                'leverage': self.leverage,
                'open': result.open_price or 0.0,
                'close': result.close_price or 0.0,
                'open_time': result.open_time,
                'close_time': result.close_time,
                'pnl': round(result.trade_pnl, 6),
                'pnl_percent': round(result.pnl_percent, 6),
                'fee': round(result.trade_fee, 6),
                'balance': round(self.all_usdt, 6),
            })

        self._cleanup(symbol)

        if self.all_usdt <= 0:
            logger.error("!!! 账户已穿仓 (Balance <= 0) !!! 请检查仓位与手续费模型")
        return result

    def _cleanup(self, symbol: str) -> None:
        pos = self.position.get(symbol)
        if pos is not None and pos.get_avg_price == 0:
            self.position.pop(symbol, None)
