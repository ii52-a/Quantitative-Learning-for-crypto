
from Strategy.PositionContral.PonsitionContral import PositionControl
from app_logger.logger_setup import Logger
from Strategy.CTA.CTA_MACD_CORE import *


from Strategy.StrategyTypes import StrategyResult, PositionSignal

logger=Logger(__name__)
class StrategyMacd:
    def __init__(self,symbol,k_line,number=2000):
        self.usdt = 1000
        self.positionControl:PositionControl=PositionControl(self.usdt)
        self.number:int=number
        self.core=BaseMacd()
        self.core.init_data(symbol,k_line,number)

    @property
    def status(self):
        return len(self.positionControl.position)>0

    def main(self):
        count=self.core.data_count
        for i in range(2,count):
            strategy_result:StrategyResult=self.core.run_step(i)
            if strategy_result:
                self.positionControl.main(strategy_result)

        if self.status:
            self.all_full()
            #TODO:策略结束一键平仓

        #TODO:日志调试和输出
        logger.info(self.positionControl.all_usdt)

    def all_full(self):
        for key,value in list(self.positionControl.position.items()):
            st=StrategyResult(
                symbol=key,
                size=1,
                execution_price=self.core.get_close_price,
                execution_time=self.core.get_last_time,
                direction=PositionSignal.FULL
            )
            self.positionControl.main(st)

if __name__ == '__main__':

    s30=StrategyMacd("ETHUSDT","30min",1000)
    s30.main()