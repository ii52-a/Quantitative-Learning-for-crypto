import sqlite3
from pathlib import Path
from typing import cast

import pandas as pd


from Config import ApiConfig
from urlibs import FileUrlibs

from app_logger.logger_setup import setup_logger

logger=setup_logger("sqlite_oper")

class SqliteOper:
    base_path=Path(ApiConfig.LOCAL_DATA_SQLITE_DIR)
    @classmethod
    def _init_sqlite(cls,symbol) -> None:
        url_str=f"{symbol}_base.db"
        path=Path(cls.base_path / url_str)
        if not path.exists():
            logger.debug(f"检测{url_str}不存在,创建数据库中")
            FileUrlibs.check_local_path(path)
            
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            #primary key
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kline (
                timestamp REAL PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                close_time REAL,
                quote_asset_volume REAL,
                number_of_trades REAL,
                taker_buy_base_asset_volume REAL,
                taker_buy_quote_asset_volume REAL
                )
            """)
            logger.info(f"{symbol}数据库初始化完成")
            conn.commit()
            conn.close()

    @classmethod
    def update_base_kline(cls,data:pd.DataFrame,symbol):
        path=Path(cls.base_path / f"{symbol}_base.db")
        if not path.exists():
            logger.debug(f"{symbol}数据库不存在，进入初始化程序")
            cls._init_sqlite(symbol)
        conn=sqlite3.connect(path)
        cursor=conn.cursor()
        insert_values=[]
        logger.debug(f"{symbol}基础数据开始写入")
        for _,row in data.iterrows():
            #cast断言函数,断言类型以避免宽泛编译器hashable类型的自判断
            #pandas.timestamp会默认使用纳秒时间戳，需要转换为毫秒时间戳同步数据库和接口返回数据
            timestamp_obj = cast(pd.Timestamp, row.name)
            insert_values.append((
                int(timestamp_obj.value/10**6),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                float(row['close_time'].value/10**6),
                float(row['quote_asset_volume']),
                float(row['number_of_trades']),
                float(row['taker_buy_base_asset_volume']),
                float(row['taker_buy_quote_asset_volume']),
            ))
        cursor.executemany(
            """
            INSERT OR REPLACE INTO kline Values(?,?,?,?,?,?,?,?,?,?,?)
            """,
            insert_values
        )
        #INSERT OR REPLACE 无法用pd.to_sql因为if_exists=append无法处理主键重复，会报错，而原生代码则会执行replace替代
        # data.to_sql('kline',conn,if_exists='append',index=True)
        cursor.execute("SELECT * FROM kline ORDER BY timestamp DESC LIMIT 1")
        last_entry = cursor.fetchone()
        now=pd.to_datetime('now')
        new=pd.to_datetime(last_entry[0],unit='ms',utc=True).tz_convert(ApiConfig.LOCAL_DATE_PLACE)

        logger.debug("当前时间戳:",now)
        logger.debug("数据集最新时间:", new)
        logger.debug(f"数据集时间差:{now-new},状态(True/False):{now-new < 10000}")
        time_span:tuple[pd.Timestamp,pd.Timestamp,int]=cls.read_time_span(symbol=symbol)
        logger.info(f"{symbol}数据集跨度:{time_span[0]} -> {time_span[1]},总量:{time_span[2]}")
        conn.commit()
        conn.close()

    @classmethod
    def read_newest_timestamp(cls,symbol) -> None | str:
        path=Path(cls.base_path / f"{symbol}_base.db")
        if not path.exists():
            cls._init_sqlite(symbol)
        conn=sqlite3.connect(path)
        cursor=conn.cursor()
        cursor.execute(
            """
            SELECT timestamp FROM kline
            ORDER BY timestamp DESC
            LIMIT 1
            """
        )
        caback:tuple[str]=cursor.fetchone()
        if not caback:
            logger.warning(f"{symbol}无基础数据集，程序错误")
            return None
        #返回最新数据值
        # print(pd.to_datetime(caback[0],unit='ms').tz_convert(ApiConfig.LOCAL_DATE_PLACE))
        return caback[0]

    @classmethod
    def read_time_span(cls,symbol) -> tuple[pd.Timestamp,pd.Timestamp,int] | None:
        path=Path(cls.base_path / f"{symbol}_base.db")
        if not path.exists():
            raise FileNotFoundError
        try:
            conn=sqlite3.connect(path)
            cursor=conn.cursor()
        except Exception as e:
            logger.warning(f"Error:{symbol}")
            return None
        start_time=cursor.execute("SELECT timestamp FROM kline ORDER BY timestamp ASC LIMIT 1").fetchone()[0]
        end_time=cursor.execute("SELECT timestamp FROM kline ORDER BY timestamp DESC LIMIT 1").fetchone()[0]
        count=cursor.execute("SELECT count(*) FROM kline").fetchone()[0]
        conn.close()
        logger.debug(f"目标:{symbol}数据集,最旧时间:{start_time},最新时间:{end_time},总量:{count}")
        return (
            pd.to_datetime(start_time,unit='ms'),
            pd.to_datetime(end_time,unit='ms'),
            count
        )



if __name__ == '__main__':
    sql=SqliteOper()
    sql.read_newest_timestamp("BTCUSDT")
    print(sql.read_time_span("BTCUSDT"))



