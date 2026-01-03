import sqlite3
from pathlib import Path
from typing import cast

import pandas as pd

import urlibs
from Config import ApiConfig

from app_logger.logger_setup import setup_logger
from decorator import catch_and_log

logger = setup_logger("sqlite_oper")


class SqliteBase:
    base_path = Path(ApiConfig.LOCAL_DATA_SQLITE_DIR)


    # 初始化数据库base
    @classmethod
    def init_sqlite(cls,
                    cursor:sqlite3.Cursor,
                    kline:str='kline'
                    ) -> None:
        logger.debug(f"检查表结构")
        cursor.execute(f"""
                       CREATE TABLE IF NOT EXISTS {kline}
                       (
                           timestamp                    REAL PRIMARY KEY,
                           open                         REAL,
                           high                         REAL,
                           low                          REAL,
                           close                        REAL,
                           volume                       REAL,
                           close_time                   REAL,
                           quote_asset_volume           REAL,
                           number_of_trades             REAL,
                           taker_buy_base_asset_volume  REAL,
                           taker_buy_quote_asset_volume REAL
                       )
                       """)
        logger.debug(f"{cursor}路径：{kline}数据集初始化完成")

    # 更新一分钟基准线
    @classmethod
    def update_base_kline(
            cls,
            data: pd.DataFrame,
            symbol:str
    )->None:
        path = Path(cls.base_path / f"{symbol}_base.db")

        with sqlite3.connect(path) as conn:
            cursor = conn.cursor()
            logger.debug(f"{symbol}数据库不存在，进入初始化程序")
            cls.init_sqlite(cursor=cursor)
            logger.debug(f"进入数据插入或更新环节")
            SqliteOper.insert_or_update_data(symbol=symbol, data=data, cursor=cursor)
            logger.info(SqliteOper.read_time_span_str(cursor=cursor))




class SqliteOper:
    # 获取最新时间
    @classmethod
    def read_newest_timestamp(
            cls,
            cursor:sqlite3.Cursor,
            kline:str='kline'
    )-> None | str:
        cursor.execute(
            f"""
            SELECT timestamp
            FROM {kline}
            ORDER BY timestamp DESC
            LIMIT 1
            """
        )
        caback: tuple[str] = cursor.fetchone()
        if not caback:
            logger.exception(f"{path}无基础数据集，程序错误")
            return None

        logger.debug(f"最新时间戳:{caback[0]}")
        return caback[0]

    # 获取时间跨度
    @classmethod
    def read_time_span(cls,
                       cursor: sqlite3.Cursor,
                       kline:str='kline'
                       ) -> tuple[pd.Timestamp, pd.Timestamp, int] | None:
            start_t = cursor.execute(f"SELECT timestamp FROM {kline} LIMIT 1").fetchone()
            end_t = cursor.execute(f"SELECT timestamp FROM {kline} ORDER BY timestamp DESC LIMIT 1").fetchone()
            if start_t is None or end_t is None:
                logger.warning(f"无法获取时间跨度,判断是否为None:start_t:{start_t is None},end_t:{end_t is None}")
                return None
            start_time = start_t[0]
            end_time = end_t[0]
            count = cursor.execute("SELECT count(*) FROM kline").fetchone()[0]
            logger.debug(f"路径:{cursor} 数据集,最旧时间:{start_time},最新时间:{end_time},总量:{count}")
            return (
                pd.to_datetime(start_time, unit='ms', utc=True),
                pd.to_datetime(end_time, unit='ms', utc=True),
                count
            )

    # 时间跨度字符串,用于数据更新完的logging
    @classmethod
    def read_time_span_str(cls, cursor:sqlite3.Cursor,kline:str='kline') -> str:
        time_span: tuple[pd.Timestamp, pd.Timestamp, int] = cls.read_time_span(cursor=cursor,kline=kline)
        if time_span is None:
            logger.warning(f"数据集跨度获取失败,指向路径:{cursor}")
            logger.exception("time_span is None")
        return f"数据集跨度:[{time_span[0].tz_convert(ApiConfig.LOCAL_DATE_PLACE)} -> {time_span[1].tz_convert(ApiConfig.LOCAL_DATE_PLACE)}] 总量:<{time_span[2]}>"

    @classmethod
    def has_table(cls,
                  cursor:sqlite3.Cursor,
                  table_name:str
                  ) -> bool:
        has=cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()
        return has is not None

    @classmethod
    def insert_or_update_data(
            cls,
            symbol:str,
            data:pd.DataFrame,
            cursor:sqlite3.Cursor,
            kline: str = 'kline'
    ) -> None:
        logger.debug(f"[in,1]<SqliteOper-insert_or_update_data>:symbol={symbol}开始插入或更新")
        insert_values = []
        if not cls.has_table(cursor=cursor, table_name=kline):
            logger.warning(f"[in,1error]<SqliteOper-insert_or_update_data>:{kline}表不存在！")
            return
        logger.debug(f"[in,2]<SqliteOper-insert_or_update_data>:{symbol}_{kline}数据开始转换为列表")
        try:
            for _, row in data.iterrows():
                # cast断言函数,断言类型以避免宽泛编译器hashable类型的自判断
                # pandas.timestamp会默认使用纳秒时间戳，需要转换为毫秒时间戳同步数据库和接口返回数据
                timestamp_obj = cast(pd.Timestamp, row.name)
                insert_values.append((
                    int(timestamp_obj.value / 10 ** 6),
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume']),
                    float(row['close_time'].value / 10 ** 6),
                    float(row['quote_asset_volume']),
                    float(row['number_of_trades']),
                    float(row['taker_buy_base_asset_volume']),
                    float(row['taker_buy_quote_asset_volume']),
                ))
        except Exception as e:
            logger.exception(f"[in,2error]<<SqliteOper-insert_or_update_data>:数据处理失败!{e}")

        logger.debug(f"[in,3]<SqliteOper-insert_or_update_data>:{symbol}基础数据开始写入数据库")
        try:
            cursor.executemany(
                f"""
                INSERT OR REPLACE INTO {kline} Values(?,?,?,?,?,?,?,?,?,?,?)
                """,
                insert_values
            )
        except Exception as e:
            logger.Exception(f"[in,3error]<SqliteOper-insert_or_update_data>:数据写入错误:{e}]")
        logger.debug(f"[in,3success]<SqliteOper-insert_or_update_data>:{symbol}_{kline}写入成功！")
        # INSERT OR REPLACE 无法用pd.to_sql因为if_exists=append无法处理主键重复，会报错，而原生代码则会执行replace替代
        # Data.to_sql('kline',conn,if_exists='append',index=True)
        logger.debug(f"[in,4]<SqliteOper-insert_or_update_data>:检查最新数据")
        cursor.execute(f"SELECT * FROM {kline} ORDER BY timestamp DESC LIMIT 1")
        last_entry = cursor.fetchone()
        if last_entry is None or len(last_entry) == 0:
            logger.warning(f"[in,4error]<SqliteOper-insert_or_update_data>{symbol}_{kline}数据集返回最新失败!")
            logger.Exception(f"{last_entry} is None")
        logger.debug(f"[in,5]<SqliteOper-insert_or_update_data>:{symbol}检查时间差]")
        try:
            now = pd.to_datetime('now', utc=True).tz_convert(ApiConfig.LOCAL_DATE_PLACE)
            new = pd.to_datetime(last_entry[0], unit='ms', utc=True).tz_convert(ApiConfig.LOCAL_DATE_PLACE)

            logger.debug(f"当前时间戳:{now}")
            logger.debug(f"数据集最新时间:{new}")
            logger.debug(f"数据集时间差:{now - new},状态(True/False):{(now - new).total_seconds() < 120}")
        except Exception as e:
            logger.warning(f"[in,5error]<SqliteOper-insert_or_update_data>:最新数据集时间戳获取错误")
        logger.debug(f"[in,end]<SqliteOper-insert_or_update_data>{symbol}_{kline}数据库操作完毕")

    @classmethod
    @catch_and_log(logger=logger,return_default=None)
    def read_range_kline(cls,
                         cursor:sqlite3.Cursor,
                         number: int,
                         kline: str = 'kline',
                         start_time:pd.Timestamp = None,
                         endtime:pd.Timestamp=None,
                         ) -> pd.DataFrame | None :
        logger.info(f"[in,1]<read_kline_range>读取表结构{kline}")
        if not cls.has_table(cursor=cursor, table_name=kline):
            logger.exception(f"[in,1error]<read_kline_range>没有找到{kline}表结构!")
        if not start_time and not endtime:
            endstr=cls.read_newest_timestamp(cursor=cursor,kline=kline)
            endtime=pd.to_datetime(float(endstr),unit='ms', utc=True)
            logger.info(f"[in,2set]<read_kline_range>默认end为最新数据{endtime.tz_convert('Asia/Shanghai')}")
        if start_time:
            if endtime:
                start_time=int(start_time.value/10 ** 6)
                endtime=int(endtime.value/10 ** 6)
                logger.info(f"[in,2set]<read_kline_range>start_time:{start_time} end_time:{endtime}")
                cursor.execute(f"SELECT * FROM {kline} WHERE timestamp >= {start_time} and timestamp <= {endtime};")
                data_df = pd.DataFrame(cursor.fetchall(),
                                       columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                                'taker_buy_quote_asset_volume'])
            else:
                start_time=int(start_time.value/10 ** 6)
                cursor.execute(f"SELECT * FROM {kline} WHERE timestamp >= {start_time} ORDER BY timestamp DESC LIMIT {number};")
                data_df = pd.DataFrame(cursor.fetchall(),
                                       columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                                'taker_buy_quote_asset_volume'])

        else:
            endtime=int(endtime.value / 10 ** 6)
            cursor.execute(f"SELECT * FROM {kline}  WHERE TIMESTAMP <= ? ORDER BY timestamp DESC LIMIT ?", (endtime, number,))
            data_df=pd.DataFrame(cursor.fetchall(), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'])
            data_df= data_df.sort_values("timestamp").reset_index(drop=True)
        return urlibs.FormatUrlibs.standard_timestamp(data_df)

    @classmethod
    @catch_and_log(logger=logger,return_default=None)
    def read_all_kline(cls,symbol:str,cursor:sqlite3.Cursor):
        """

        :param symbol:
        :param cursor: 统一光标
        :return: 该类型的所有数据
        """
        logger.info(f"[in,1]<read_all_kline>尝试获取所有{symbol}数据",stacklevel=2)
        if not cls.has_table(cursor=cursor, table_name=symbol):
            logger.error(f"[in,2] {symbol}数据获取失败，表结构不存在！")

        cursor.execute(f"SELECT * FROM {symbol}")
        data_df = pd.DataFrame(cursor.fetchall(),
                               columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                        'taker_buy_quote_asset_volume'])
        return data_df



    @classmethod
    @catch_and_log(logger=logger,return_default=None)
    def with_open_and_func(cls,symbol,kline_str:str,func,*args,**kwargs) -> any:
        """
        使用统一的数据库操作光标，防止出现同时访问的lock问题
        :param symbol: 期货类型
        :param kline_str: 访问k线类型
        :param func:  使用统一访问下表的功能，提供cursor
        :param args:  函数参数
        :param kwargs: 函数参数
        :return:
        """
        patht=None
        ROOT_DIR = Path(__file__).resolve().parent.parent
        if kline_str=="1m":
            patht=Path(f"Data/LocalData/{symbol}_base.db")
            logger.info(f"[in,1base]<with_open_and_func>路径:{patht}")
        else:
            patht=Path(f"Data/LocalData/{symbol}_aggerate.db")
            # urlibs.FileUrlibs.check_local_path(patht)
            logger.info(f"[in,1aggerate]<with_open_and_func>路径:{patht}")
        patht = Path(ApiConfig.PROJECT_ROOT) / "Data" / "LocalData" / f"{symbol}_aggerate.db"
        with sqlite3.connect(patht) as conn:
            cursor = conn.cursor()
            return func(cursor,*args,**kwargs)



if __name__ == '__main__':
    sql = SqliteOper()
    path=Path("LocalData/ETHUSDT_aggerate.db")
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        data=sql.read_range_kline(cursor=cursor,number=20,kline='kline_30min')
        print(data)





