
from binance import Client

import Config
from app_logger.logger_setup import setup_logger
from Strategy.CTA.CTA_MACD_CORE import *
from Strategy.data_service import Dataservice

from Strategy.position_contral import PositionControl
from Strategy.StrategyTypes import PositionSignal, StrategyResult

logger=setup_logger(__name__)
class StrategyMacd:
    def __init__(self,symbol,k_line,number=2000):
        self.usdt = 100
        self.symbol:str = symbol
        self.k_line:str = k_line if k_line in Config.TradeMapper.data_time else "30min"
        self.position:PositionControl=PositionControl(self.symbol)
        self.status=0   #0 无持仓 1有持仓
        self.number:int=number
        self.core=BaseMacd()
    def main(self):
        data=Dataservice.get_data(self.symbol,self.k_line,self.number)
        macd_data=Dataservice.macd_data(data)
        self.core.init_data(data,macd_data)

        for i in range(2,len(macd_data)):
            self.core.update_context(i)
            strategy_result:StrategyResult=self.core.step()
            """filter"""
            #TODO:根据数据修改result
            if strategy_result:
                if strategy_result.signal==PositionSignal.OPEN:
                    self.position.open_position(strategy_result.size,strategy_result.execution_price,strategy_result.execution_time)
                elif strategy_result.signal==PositionSignal.CLOSE:
                    self.position.close_position(strategy_result.size, strategy_result.execution_price,
                                                )
        cur_price=data.iloc[-1]['close']
        cur_time=data.iloc[-1]['close_time']

        if self.position.position != PositionSignal.EMPTY:
            self.position.close_position(price=float(cur_price), time=cur_time)
            self.status=0
        self.position.print(self.number,"30min/per",Client.KLINE_INTERVAL_30MINUTE )
        self.position = PositionControl(self.symbol, self.usdt)


if __name__ == '__main__':


    s30=StrategyMacd("ETHUSDT","30min",20000)
    s30.main()