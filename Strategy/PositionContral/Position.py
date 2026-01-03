
from Strategy.StrategyTypes import *


class Position:
    def __init__(self,symbol,leverage):
        self.symbol: str = symbol
        self._position_status:PositionStatus=PositionStatus(
            margin_used = 0,
            true_margin_used = 0,
            nominal_value = 0,
            avg_price = 0,
            leverage= leverage,
            open_count = 0,
            close_count = 0,
        )
        #流动字段,不可调用
        self._nominal_value = None
        self._price = None
        self._signal = None
        self._changed_used = None

    #返回真实保险金占用
    def true_margin(self,price) -> float:
        """
        提供调用以模拟过程真实保险金比例计算
        :param price:
        :return:
        """
        cur_pnl= (price - self._position_status.avg_price) * self._position_status.nominal_value
        self._position_status.true_margin_used =self._position_status.margin_used - cur_pnl
        return self._position_status.true_margin_used

    #仓位处理
    def execute(self,position_set:PositionSet) -> PositionResult:
        """
        仓位变动处理入口
        :param position_set:
        :return:
        """
        self._update_context(position_set)
        #判断为平仓
        if self._nominal_value * self._position_status.nominal_value < 0:
            return self._close()
        else:
            return self._open()

    def _update_context(self,position_set):
        """
        处理仓位变动参数
        :param position_set
        :return:
        """
        # 变动保险金
        self._changed_used = position_set.changed_usdt
        # 方向
        self._signal = position_set.signal

        # 仓位变动价格
        self._price = position_set.price

        # 变动名义价值   (U本位价值)  value=变动保险金 * 杠杆 * 仓位方向
        self._nominal_value = self._changed_used * self._position_status.leverage * self._signal.value

    def _close(self) -> PositionResult:
        """
        处理平仓逻辑
        :return: PositionResult
        """
        # 计算平仓状态
        close_status: PositionResultSignal = PositionResultSignal.PARTIAL \
            if self._nominal_value + self._position_status.nominal_value == 0 \
            else PositionResultSignal.PARTIAL
        # 计算pnl
        pnl: float = (self._price - self._position_status.avg_price) * self._nominal_value
        # 处理仓位 释放占用保险金，更改名义价值
        self._position_status.margin_used -= self._changed_used
        self._position_status.nominal_value += self._nominal_value
        return PositionResult(
            signal=close_status,
            pnl=pnl
        )

    def _open(self) -> PositionResult:
        """
        处理开仓逻辑
        :return: PositionResult
        """
        open_status: PositionResultSignal = PositionResultSignal.OPEN
        # 计算平均价格
        new_nominal_value = self._position_status.nominal_value + self._nominal_value
        cur_persent = self._position_status.nominal_value / new_nominal_value
        add_persent = 1 - cur_persent
        self._position_status.avg_price = self._position_status.avg_price * cur_persent + self._price * add_persent
        # 占用保险金
        self._position_status.margin_used += self._changed_used
        self._position_status.nominal_value = new_nominal_value

        return PositionResult(
            signal=open_status,
            pnl=0
        )







