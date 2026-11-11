import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd

import urlibs
from Config import ApiConfig

from app_logger.logger_setup import Logger
from decorator import catch_and_log

logger = Logger("sqlite_oper")

# 数据库列顺序，所有读函数共用
COLUMNS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
]
# 建表 / 写入需要的数值列
VALUE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
    'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
]


def to_ms(value) -> int:
    """把 Timestamp / 毫秒数值 / 时间字符串统一成 UTC 毫秒。"""
    if isinstance(value, (int, float)):
        return int(value)
    # tz-aware 用绝对时间；naive 沿用 pandas 的本地时间解释，保持与旧代码一致
    return int(pd.Timestamp(value).timestamp() * 1000)


def _index_ms(index) -> list[int]:
    """DatetimeIndex -> UTC 毫秒。

    pandas 3.x 的 DatetimeIndex 可能是 s/us/ns 分辨率，asi8 不保证是纳秒，
    必须先 as_unit('ns') 再除以 1e6。
    """
    return (pd.DatetimeIndex(index).as_unit('ns').asi8 // 10 ** 6).tolist()


class SqliteBase:
    base_path = Path(ApiConfig.LOCAL_DATA_SQLITE_DIR)

    # 初始化数据库base
    @classmethod
    def init_sqlite(cls,
                    cursor: sqlite3.Cursor,
                    kline: str = 'kline'
                    ) -> None:
        logger.debug(f"检查表结构 {kline}")
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
        logger.debug(f"{kline} 数据集初始化完成")

    # 更新一分钟基准线
    @classmethod
    def update_base_kline(
            cls,
            data: pd.DataFrame,
            symbol: str
    ) -> int:
        path = Path(cls.base_path) / f"{symbol}_base.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as conn:
            cursor = conn.cursor()
            cls.init_sqlite(cursor=cursor)
            count = SqliteOper.insert_or_update_data(symbol=symbol, data=data, cursor=cursor)
            conn.commit()
            logger.info(SqliteOper.read_time_span_str(cursor=cursor))
        return count


class SqliteOper:
    # 获取最新时间
    @classmethod
    def read_newest_timestamp(
            cls,
            cursor: sqlite3.Cursor,
            kline: str = 'kline'
    ) -> int | None:
        if not cls.has_table(cursor, kline):
            logger.error(f"{kline} 表不存在，无法读取最新时间戳")
            return None
        cursor.execute(
            f"""
            SELECT timestamp
            FROM {kline}
            ORDER BY timestamp DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        if not row or row[0] is None:
            logger.warning(f"{kline} 无数据，返回 None")
            return None

        newest = int(row[0])
        logger.debug(f"最新时间戳:{newest}")
        return newest

    # 获取时间跨度
    @classmethod
    def read_time_span(cls,
                       cursor: sqlite3.Cursor,
                       kline: str = 'kline'
                       ) -> tuple[pd.Timestamp, pd.Timestamp, int] | None:
            if not cls.has_table(cursor, kline):
                logger.warning(f"{kline} 表不存在，无法获取时间跨度")
                return None
            start_t = cursor.execute(f"SELECT timestamp FROM {kline} ORDER BY timestamp ASC LIMIT 1").fetchone()
            end_t = cursor.execute(f"SELECT timestamp FROM {kline} ORDER BY timestamp DESC LIMIT 1").fetchone()
            if start_t is None or end_t is None:
                logger.warning(f"无法获取时间跨度,判断是否为None:start_t:{start_t is None},end_t:{end_t is None}")
                return None
            count = cursor.execute(f"SELECT count(*) FROM {kline}").fetchone()[0]
            return (
                pd.to_datetime(float(start_t[0]), unit='ms', utc=True),
                pd.to_datetime(float(end_t[0]), unit='ms', utc=True),
                int(count)
            )

    # 时间跨度字符串,用于数据更新完的logging
    @classmethod
    def read_time_span_str(cls, cursor: sqlite3.Cursor, kline: str = 'kline') -> str:
        time_span = cls.read_time_span(cursor=cursor, kline=kline)
        if time_span is None:
            return f"{kline} 数据集为空"
        start, end, count = time_span
        return (f"数据集跨度:[{start.tz_convert(ApiConfig.LOCAL_DATE_PLACE)} -> "
                f"{end.tz_convert(ApiConfig.LOCAL_DATE_PLACE)}] 总量:<{count}>")

    #检查是否有表结构
    @classmethod
    def has_table(cls,
                  cursor: sqlite3.Cursor,
                  table_name: str
                  ) -> bool:
        has = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone()
        return has is not None

    #插入或更新数据
    @classmethod
    def insert_or_update_data(
            cls,
            symbol: str,
            data: pd.DataFrame,
            cursor: sqlite3.Cursor,
            kline: str = 'kline'
    ) -> int:
        if data is None or data.empty:
            logger.warning(f"[in,0]<insert_or_update_data>: {symbol}_{kline} 待写入数据为空，跳过")
            return 0
        if not cls.has_table(cursor=cursor, table_name=kline):
            logger.error(f"[in,1]<insert_or_update_data>:{kline}表不存在！")
            return 0

        missing = [c for c in VALUE_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(f"{symbol}_{kline} 缺少列: {missing}")

        df = data.sort_index()
        if df.index.name != 'timestamp':
            raise ValueError(f"{symbol}_{kline} 索引必须是 timestamp")
        if df[VALUE_COLUMNS].isna().any().any():
            raise ValueError(f"{symbol}_{kline} 存在空值，拒绝写入以免污染数据")

        # 一次构造所有行，避免 iterrows 逐行转换：既慢又可能在异常时丢数据
        rows = list(zip(
            _index_ms(df.index),
            df['open'].astype(float).tolist(),
            df['high'].astype(float).tolist(),
            df['low'].astype(float).tolist(),
            df['close'].astype(float).tolist(),
            df['volume'].astype(float).tolist(),
            _index_ms(df['close_time']),
            df['quote_asset_volume'].astype(float).tolist(),
            df['number_of_trades'].astype(float).tolist(),
            df['taker_buy_base_asset_volume'].astype(float).tolist(),
            df['taker_buy_quote_asset_volume'].astype(float).tolist(),
        ))

        logger.debug(f"[in,2]<insert_or_update_data>:{symbol}_{kline} 写入 {len(rows)} 行")
        try:
            cursor.executemany(
                f"INSERT OR REPLACE INTO {kline} VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                rows
            )
        except Exception as e:
            logger.exception(f"[in,3error]<insert_or_update_data>:数据写入错误:{e}")
            raise

        logger.debug(f"[in,3success]<insert_or_update_data>:{symbol}_{kline}写入成功！")
        return len(rows)

    #读取数据范围
    @classmethod
    def read_range_kline(cls,
                         cursor: sqlite3.Cursor,
                         number: int,
                         kline: str = 'kline',
                         start_time: pd.Timestamp = None,
                         endtime: pd.Timestamp = None,
                         ) -> pd.DataFrame | None:
        """
        统一按时间升序返回最近 number 根 K 线。
        - 只给 start_time：从 start_time 起向后取 number 根
        - 只给 endtime  ：向 endtime 之前回溯取 number 根
        - 两者都给      ：区间内取 number 根
        """
        if not cls.has_table(cursor=cursor, table_name=kline):
            raise ValueError(f"{kline} 表不存在")
        number = int(number)
        if number <= 0:
            raise ValueError(f"number 必须为正整数，收到 {number}")

        if start_time is not None and endtime is not None:
            sql = (f"SELECT * FROM {kline} WHERE timestamp >= ? AND timestamp <= ? "
                   f"ORDER BY timestamp ASC LIMIT ?")
            params = (to_ms(start_time), to_ms(endtime), number)
            rows = cursor.execute(sql, params).fetchall()
        elif start_time is not None:
            sql = f"SELECT * FROM {kline} WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT ?"
            rows = cursor.execute(sql, (to_ms(start_time), number)).fetchall()
        else:
            end_ms = to_ms(endtime) if endtime is not None else cls.read_newest_timestamp(cursor, kline)
            if end_ms is None:
                logger.warning(f"{kline} 无最新时间戳，返回空数据")
                return cls._empty_frame()
            sql = f"SELECT * FROM {kline} WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT ?"
            rows = cursor.execute(sql, (end_ms, number)).fetchall()

        if not rows:
            logger.warning(f"{kline} 查询区间无数据")
            return cls._empty_frame()

        data_df = pd.DataFrame(rows, columns=COLUMNS).sort_values('timestamp')
        return urlibs.FormatUrlibs.standard_timestamp(data_df)

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        df = pd.DataFrame({c: pd.Series(dtype='float64') for c in COLUMNS})
        return urlibs.FormatUrlibs.standard_timestamp(df)

    @classmethod
    @catch_and_log(logger=logger, return_default=None)
    def read_all_kline(cls, symbol: str, cursor: sqlite3.Cursor):
        """

        :param symbol:
        :param cursor: 统一光标
        :return: 该类型的所有数据
        """
        logger.info(f"[in,1]<read_all_kline>尝试获取所有{symbol}数据")
        if not cls.has_table(cursor=cursor, table_name=symbol):
            logger.error(f"[in,2] {symbol}数据获取失败，表结构不存在！")
            return None

        cursor.execute(f"SELECT * FROM {symbol} ORDER BY timestamp ASC")
        data_df = pd.DataFrame(cursor.fetchall(), columns=COLUMNS)
        return data_df

    @classmethod
    def with_open_and_func(cls, symbol, kline_str: str, func, *args, **kwargs) -> any:
        """
        使用统一的数据库操作光标，防止出现同时访问的lock问题
        :param symbol: 期货类型
        :param kline_str: 访问k线类型
        :param func:  使用统一访问下表的功能，提供cursor
        :param args:  函数参数
        :param kwargs: 函数参数
        :return:
        """
        db_dir = Path(ApiConfig.LOCAL_DATA_SQLITE_DIR)
        name = f"{symbol}_base.db" if kline_str in ('1m', '1min') else f"{symbol}_aggerate.db"
        path = db_dir / name
        if not path.exists():
            raise FileNotFoundError(f"本地数据库不存在: {path}")
        logger.info(f"[in,1]<with_open_and_func>路径:{path}")
        with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as conn:
            cursor = conn.cursor()
            return func(cursor, *args, **kwargs)


if __name__ == '__main__':
    path = Path(ApiConfig.LOCAL_DATA_SQLITE_DIR) / "ETHUSDT_aggerate.db"
    with closing(sqlite3.connect(path)) as conn:
        cursor = conn.cursor()
        data = SqliteOper.read_range_kline(cursor=cursor, number=20, kline='kline_30min')
        print(data)
