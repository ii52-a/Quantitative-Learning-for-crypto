from Config import BackConfig
from Strategy.PositionContral.Position import Position
from Strategy.StrategyTypes import StrategyResult, PositionSet, PositionSignal


class PositionControl:
    def __init__(self,usdt):
        self.position:dict[str:Position]={}
        #全部持有usdt价值
        self.all_usdt=usdt
        #仓位占用usdt，只考虑当前仓位占用
        self.margin_usdt:float=0

        #实际占用usdt  考虑仓位的盈亏即过程盈亏  仅用来计算是否爆仓或调整风险
        self._true_margin_usdt:float=0
        #保险金比率
        self.margin_percent:float=0
        #默认杠杆
        self.leverage:int=BackConfig.SET_LEVERAGE

    @property
    def _usdt(self):
        """
        计算，当前阶段开仓可用usdt,为总价值-当前过程真实占用与仓位占用的最大值，不进行盈利复投
        """
        return self.all_usdt - max(self.margin_usdt,self._true_margin_usdt)

    @property
    def _liquidation(self):
        """
            TODO：精确爆仓需要秒级数据，仅使用k线极值法估算是否爆仓，其误差在于每个仓位的极值不是同时取得的。
        :return:
        """
        return self.margin_percent >=0.98

    def update(self,cur_price:float):
        new_margin=0
        for position in self.position.values():
            new_margin+=position.true_margin(cur_price)

        self._true_margin_usdt=new_margin
        self.margin_percent=min(self._true_margin_usdt / (self.all_usdt-self.margin_usdt+self._true_margin_usdt),0.98)
        if self._liquidation:
            """进入爆仓流程"""
            pass



    def main(self,strategy_result:StrategyResult):
        symbol=strategy_result.symbol
        signal=strategy_result.more_less
        changed_usdt=strategy_result.size*self._usdt
        price=strategy_result.execution_price
        if symbol not in self.position.keys():
            #创建仓位
            self.position.setdefault(symbol,Position(symbol,self.leverage))
        position_set:PositionSet=PositionSet(
            signal=signal,
            changed_usdt=changed_usdt,
            price=price
        )
        self.position[symbol].execute(position_set)


