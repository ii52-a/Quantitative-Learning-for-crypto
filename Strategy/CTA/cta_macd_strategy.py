from tqdm import tqdm

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
        with tqdm(total=count,desc="回测进度") as pbar:
            for i in range(2,count):
                strategy_result:StrategyResult=self.core.run_step(i)
                if strategy_result:
                    self.positionControl.main(strategy_result)
                if i%100:
                    pbar.n=i+1
                    pbar.refresh()

        if self.status:
            self.all_full()
            #TODO:策略结束一键平仓

        #TODO:日志调试和输出
        logger.info(f"{self.positionControl.all_usdt}({self.positionControl.all_usdt*100/ self.positionControl.init_usdt :.2f}%)\n"
                    f"total:{self.positionControl.total}\t win:{self.positionControl.win}\t per:{self.positionControl.win/self.positionControl.total*100:.2f}%")

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

    s30=StrategyMacd("ETHUSDT","30min",20000)
    s30.main()