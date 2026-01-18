
from Strategy.StrategyTypes import *
from app_logger.logger_setup import setup_logger

logger=setup_logger(__name__)
class Position:
    def __init__(self,symbol,leverage):
        self.symbol: str = symbol
        self._position_status:PositionStatus=PositionStatus(
            #占用保险金
            margin_used = 0,
            #实际占用，计算当前盈亏
            true_margin_used = 0,
            #名义价值
            nominal_value = 0,
            #平均开仓价格
            avg_price = 0,
            #杠杆
            leverage= leverage,
            #开仓次数
            open_count = 0,
            #平仓次数  -产生盈亏为平仓
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
        提供调用以模拟过程真实保险金比例计算。
        :param price:
        :return:
        """
        if self._position_status.avg_price == 0:
            return float(self._position_status.margin_used)
        cur_pnl= (price - self._position_status.avg_price)/self._position_status.avg_price * self._position_status.nominal_value
        return round(self._position_status.margin_used - cur_pnl,6)

    @property
    def margin_usdt(self) -> float:
        return self._position_status.margin_used

    @property
    def direction(self) -> int:
        return 1 if self.get_nominal_value() >=0 else -1


    #仓位处理
    def execute(self,position_set:PositionSet) -> PositionResult:
        """
        仓位变动处理入口
        :param position_set:
        :return:
        """
        self._update_context(position_set)
        #判断为开仓
        if position_set.signal==PositionChange.OPEN:
            return self._open()
        #反向开仓
        elif position_set.signal==PositionChange.RESERVED:
            return self._close_and_reverse()
        #部分平仓
        elif position_set.signal==PositionChange.PARTIAL:
            return self._close()
        elif position_set.signal==PositionChange.FULL:
            return self._full_close()
        elif position_set.signal==PositionChange.ERROR:
            logger.error(f"{self.symbol}仓位：execute处理失败，仓位管理错误")
            return PositionResult(
                pnl=0
            )
        else:
            logger.error(f"{self.symbol}仓位：execute处理失败，信号无法处理")
            return PositionResult(
                pnl=0
            )



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
        self._nominal_value = self._changed_used * self._position_status.leverage * self.direction

    def _close(self) -> PositionResult:
        """
        处理平仓逻辑
        :return: PositionResult
        """
        # 计算平仓状态

        # 计算pnl
        pnl: float = (self._price - self._position_status.avg_price)/self._position_status.avg_price * -self._nominal_value
        # 处理仓位 释放占用保险金，更改名义价值
        self._position_status.margin_used -= self._changed_used
        self._position_status.nominal_value += self._nominal_value
        return PositionResult(
            pnl=round(pnl,6)
        )

    def _full_close(self) -> PositionResult:
        pnl: float = (self._price - self._position_status.avg_price) / self._position_status.avg_price * -self._nominal_value
        self._position_status.margin_used = 0
        self._position_status.nominal_value =0
        self._position_status.avg_price = 0
        return PositionResult(
            pnl=round(pnl,6)
        )

    def _open(self) -> PositionResult:
        """
        处理开仓逻辑
        :return: PositionResult
        """
        # 计算平均价格
        new_nominal_value = self._position_status.nominal_value + self._nominal_value
        cur_persent = self._position_status.nominal_value / new_nominal_value
        add_persent = 1 - cur_persent
        self._position_status.avg_price = self._position_status.avg_price * cur_persent + self._price * add_persent
        # 占用保险金
        self._position_status.margin_used += self._changed_used
        self._position_status.nominal_value = new_nominal_value

        return PositionResult(

            pnl=0
        )

    def _close_and_reverse(self) -> PositionResult:
        #反转，平所有仓位计算盈亏
        pnl: float = (self._price - self._position_status.avg_price)/self._position_status.avg_price * self._position_status.nominal_value
        #剩余名义价值判断
        self._position_status.nominal_value = -self._nominal_value
        self._position_status.avg_price = self._price
        self._position_status.margin_used = self._changed_used
        return PositionResult(
            pnl=round(pnl,6)
        )

    def get_nominal_value(self) -> float:
        return self._position_status.nominal_value

    def get_avg_price(self) -> float:
        return self._position_status.avg_price

    def __str__(self):
        return f"{self._position_status}"






