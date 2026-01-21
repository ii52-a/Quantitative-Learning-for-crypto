from Config import BackConfig
from Strategy.PositionContral.Position import Position
from Strategy.StrategyTypes import *
from app_logger.logger_setup import Logger

logger = Logger(__name__)


class PositionControl:
    def __init__(self, usdt=1000):
        self.position: dict[str:Position] = {}
        self.all_usdt = usdt  # 账户余额（含已结算盈亏）
        self._true_margin_usdt = 0
        self.leverage = BackConfig.SET_LEVERAGE

    @property
    def _usdt(self):
        # 可用资金 = 当前总余额 - 仓位占用保证金
        return self.all_usdt - self.margin_usdt

    @property
    def margin_usdt(self):
        return sum(p.margin_usdt for p in self.position.values())

    def _sign_transform(self, symbol, strategy_result: StrategyResult):
        signal = strategy_result.direction
        pos = self.position[symbol]

        # 初始或同向
        if pos.get_avg_price == 0 or (pos.direction == 1 and signal == PositionSignal.LONG) or (
                pos.direction == -1 and signal == PositionSignal.SHORT):
            return strategy_result.size * self._usdt, PositionChange.OPEN

        # 全平
        if signal == PositionSignal.FULL:
            return pos.margin_usdt, PositionChange.FULL

        # 反手
        if (pos.direction == 1 and signal == PositionSignal.SHORT) or (
                pos.direction == -1 and signal == PositionSignal.LONG):
            return pos.margin_usdt, PositionChange.RESERVED

        return 0, PositionChange.ERROR

    def main(self, strategy_result: StrategyResult):
        symbol = strategy_result.symbol
        if symbol not in self.position:
            self.position[symbol] = Position(symbol, self.leverage)


        changed_usdt, change_type = self._sign_transform(symbol, strategy_result)

        # 风险拦截
        if change_type == PositionChange.OPEN and changed_usdt > self._usdt:
            logger.warning(f"[{symbol}] 资金不足拒绝开仓: 需{changed_usdt:.2f} > 剩{self._usdt:.2f}")
            return

        # 执行
        pos_set = PositionSet(signal=change_type,
                              changed_usdt=changed_usdt,
                              price=strategy_result.execution_price,
                              open_time=strategy_result.execution_time,
                              )
        result = self.position[symbol].execute(pos_set)

        # 更新余额
        self.all_usdt += result.pnl


        if self.position[symbol].get_avg_price == 0:
            self.position.pop(symbol)

        # 强平保护
        if self.all_usdt < 0:
            logger.error("!!! 账户已穿仓 (Balance < 0) !!! 强制停止回测")
            # 可以在此处抛出异常或停止循环