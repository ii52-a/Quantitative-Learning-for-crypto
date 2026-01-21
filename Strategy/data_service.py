
from typing import List

import pandas as pd
import talib

import Config
from Data.sqlite_oper import SqliteOper
from app_logger.logger_setup import Logger
from decorator import catch_and_log

logger=Logger(__name__)
class Dataservice:
    data_list:List[str]=Config.TradeMapper.data_time


    @classmethod
    @catch_and_log(logger=logger,return_default=None)
    def _get_data_time(cls,symbol:str,
                       kline:str,
                       start:pd.Timestamp | None,
                       length:int):
        logger.info(f"begin get {symbol} {kline}")
        data_df=SqliteOper.with_open_and_func(symbol=symbol,kline_str=kline,func=SqliteOper.read_range_kline,kline=kline,start_time=start,number=length)
        return data_df

    @classmethod
    @catch_and_log(logger=logger,return_default=None)
    def get_data(cls,symbol,time:str,length:int,start_time:pd.Timestamp| None=None):
        """
        :param symbol: 类型
        :param time:  k线类型
        :param start_time: 开始时间
        :param length: 长度
        :return:  DataFrame形式数据
        """
        if time not in cls.data_list:
            logger.error(f"{time} not in data_list!")
        kline=f"kline_{time}"
        data_df=cls._get_data_time(symbol=symbol,kline=kline,start=start_time,length=length+50)
        if data_df is None or data_df.empty:
            logger.error("data获取失败")
        return data_df

    @staticmethod
    def macd_data(data:pd.DataFrame)->pd.DataFrame:
        logger.debug("计算 MACD 指标中...")
        macd, macd_signal, macd_hist = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['MACD'] = macd
        data['MACD_SIGNAL'] = macd_signal
        data['MACD_HIST'] = macd_hist
        data = data.dropna(subset=['MACD', 'MACD_SIGNAL', 'MACD_HIST'])
        logger.debug("MACD 计算完成")
        return data


if __name__ == '__main__':
    start=pd.Timestamp("2025-07-30")
    n=1000
    data=Dataservice.get_data("ETHUSDT","30min",start_time=start,length=n)
    print(Dataservice.macd_data(data))

