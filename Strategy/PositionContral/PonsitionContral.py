import os

import pandas as pd

from Config import BackConfig
from Strategy.PositionContral.Position import Position
from Strategy.StrategyTypes import *
from app_logger.logger_setup import Logger

logger = Logger(__name__)


class PositionControl:
    def __init__(self, usdt=1000,leverage=BackConfig.SET_LEVERAGE):
        self.history_records = []
        self.position: dict[str:Position] = {}
        self.init_usdt=usdt
        self.all_usdt = usdt  # 账户余额（含已结算盈亏）
        self._true_margin_usdt = 0
        self.leverage = leverage

        self.total=0
        self.win=0
        self.lose=0


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
            logger.warning(f"[{symbol}] 资金不足拒绝开仓: {changed_usdt:.2f} > 剩{self._usdt:.2f}")
            return

        # 执行
        pos_set = PositionSet(signal=change_type,
                              changed_usdt=changed_usdt,
                              price=strategy_result.execution_price,
                              open_time=strategy_result.execution_time,
                              )
        result:PositionResult = self.position[symbol].execute(pos_set)

        # 更新余额
        self.all_usdt += result.pnl

        if result.if_full:
            if result.win:
                self.win +=1
            elif not result.win:
                self.lose +=1
            self.total+=1

            history_obj = self.position[symbol].get_d_log()

            # 转换为字典
            record = history_obj.__dict__
            # logger.warning(type(record))
            record['current_balance'] = self.all_usdt
            self.history_records.append(record)




        if self.position[symbol].get_avg_price == 0:
            self.position.pop(symbol)

        # 强平保护
        if self.all_usdt < 0:
            logger.error("!!! 账户已穿仓 (Balance < 0) !!! 强制停止回测")
            #

    def data_to_csv(self,filename = "backtest_result"):
        i=0
        filename1=filename + f"_{self.leverage}X.csv"
        while True:
            if os.path.exists(filename1):
                i += 1
                filename1 = filename + "_" + str(i) + f"_{self.leverage}X.csv"
            else:
                break



        if not self.history_records:
            logger.warning("<UNK>没有交易记录")
            return
        df = pd.DataFrame(self.history_records)
        df.to_csv(filename1, index=False)
        logger.info(f" 过程数据已导出至: {filename1}")