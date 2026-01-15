import sqlite3
from pathlib import Path

import pandas as pd

import Config
from app_logger.logger_setup import setup_logger
from urlibs import FormatUrlibs

logger=setup_logger('kline_process')
class KlineProcess:
    LOCAL_PATH:Path=Path('LocalData')

    db_path:Path | None=None
    base_db_path:Path | None=None
    symbol:str | None=None
    cursor:sqlite3.Cursor | None=None
    @classmethod
    def main(cls,symbol:str):
        cls.db_path=cls.LOCAL_PATH / Path(f'{symbol}_aggerate.db')
        cls.base_db_path=cls.LOCAL_PATH / Path(f'{symbol}_base.db')
        cls.symbol=symbol
        logger.debug("[2]<klineProcess-main>:kline聚合常数定义完成")
        with sqlite3.connect(cls.base_db_path) as conn:
            curb=conn.cursor()
            cur_data:sqlite3.Cursor=curb.execute("""
            SELECT * FROM kline ORDER BY timestamp ASC
                              """)
            fc_data=cur_data.fetchall()
        logger.debug("[2,1]<klineProcess-main>:Base基础数据获取成功")
        if not fc_data:
            logger.warning(f"[2,1error]<klineProcess-main>:{symbol}基础数据集获取失败")
            return
        df = pd.DataFrame(fc_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
             "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
        ])
        logger.debug(f"[2,2]<klineProcess-main>:df{symbol}基础数据集pandas完成")
        df=FormatUrlibs.standard_timestamp(df)
        logger.debug(f"[2,3]<klineProcess-main>:开始聚合k线数据")
        cls._agg_useful_kline(df=df)
        logger.debug(f"[2,end]<klineProcess-main>聚合成功")



    @classmethod
    def _agg_useful_kline(cls,df:pd.DataFrame):

        with sqlite3.connect(cls.db_path) as conn:
            cls.cursor = conn.cursor()
            data_time=Config.TradeMapper.data_time
            for i in data_time:
                logger.debug(f"[2,4]<klineProcess-main>:{i}级k线开始聚合")
                cls._agg_kline(df,i)
                logger.debug(f"[2,4success]<klineProcess-main>:{i}级k线聚合成功")

    @classmethod
    def _agg_kline(cls,df:pd.DataFrame,time:str)->None:
        kline:str=f'kline_{time}'
        logger.debug(f"[2,4,1start]<KlineProcess-main-_agg_kline_{time}>开始")
        logger.debug("="*30)
        data:pd.DataFrame=df.resample(rule=time,origin='start').agg(
            {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'close_time': 'last',
                'quote_asset_volume': 'sum',
                'number_of_trades': 'sum',
                'taker_buy_base_asset_volume': 'sum',
                'taker_buy_quote_asset_volume': 'sum',
            }
        )
        logger.debug(f"[2,4,1]<KlineProcess-main-_agg_kline>:{kline}数据聚合成功")


        logger.debug(f"[2,4,2]<KlineProcess-main-_agg_kline>:{kline}开始写入")

        SqliteBase.init_sqlite(cursor=cls.cursor,kline=kline)
        if not SqliteOper.has_table(cls.cursor,kline):
            logger.warning(f"[2,4,2error]<KlineProcess-main-_agg_kline>:{kline}表不存在!")
            return

        logger.debug(f"[2,4,2success]<KlineProcess-main-_agg_kline>:{kline}创建成功>")

        SqliteOper.insert_or_update_data(symbol=cls.symbol,data=data,kline=kline,cursor=cls.cursor)

        logger.debug(f"[2,4end]<KlineProcess-main-_agg_kline_{time}>结束")


















