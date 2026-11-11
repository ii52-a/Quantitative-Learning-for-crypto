import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
from pandas.tseries.frequencies import to_offset

from Config import ApiConfig, TradeMapper
from Data.sqlite_oper import SqliteBase, SqliteOper
from app_logger.logger_setup import Logger
from urlibs import FormatUrlibs

logger = Logger('kline_process')

AGG_COLUMNS = {
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


class KlineProcess:
    """把 1min 基准库聚合成各周期 K 线库（读写同一目录）。"""

    LOCAL_PATH: Path = Path(ApiConfig.LOCAL_DATA_SQLITE_DIR)
    symbol: str | None = None

    @classmethod
    def main(cls, symbol: str) -> int:
        cls.symbol = symbol
        base_db = cls.LOCAL_PATH / f'{symbol}_base.db'
        if not base_db.exists():
            raise FileNotFoundError(f"基础数据库不存在: {base_db}")

        logger.debug("[2]<klineProcess-main>:kline聚合开始")
        with closing(sqlite3.connect(f"file:{base_db.as_posix()}?mode=ro", uri=True)) as conn:
            rows = conn.execute("SELECT * FROM kline ORDER BY timestamp ASC").fetchall()
        if not rows:
            logger.warning(f"[2,1error]<klineProcess-main>:{symbol}基础数据集为空")
            return 0

        df = pd.DataFrame(rows, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
        ])
        logger.debug(f"[2,2]<klineProcess-main>:df{symbol}基础数据集pandas完成")
        df = FormatUrlibs.standard_timestamp(df)
        logger.debug(f"[2,3]<klineProcess-main>:开始聚合k线数据")
        written = cls._agg_useful_kline(df=df)
        logger.debug(f"[2,end]<klineProcess-main>聚合成功，写入 {written} 行")
        return written

    @classmethod
    def _agg_useful_kline(cls, df: pd.DataFrame) -> int:
        agg_db = cls.LOCAL_PATH / f'{cls.symbol}_aggerate.db'
        agg_db.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with closing(sqlite3.connect(agg_db)) as conn:
            cursor = conn.cursor()
            for rule in TradeMapper.data_time:
                logger.debug(f"[2,4]<klineProcess-main>:{rule}级k线开始聚合")
                total += cls._agg_kline(df, rule, cursor)
                logger.info(f"[2,4success]<klineProcess-main>:{rule}级k线聚合成功")
            conn.commit()
        return total

    @classmethod
    def _agg_kline(cls, df: pd.DataFrame, rule: str, cursor: sqlite3.Cursor) -> int:
        kline = f'kline_{rule}'
        # 表结构先建好：即使本次没有完整桶，策略侧也能给出"数据为空"而不是"表不存在"
        SqliteBase.init_sqlite(cursor=cursor, kline=kline)

        data = df.resample(rule=rule, origin='start_day').agg(AGG_COLUMNS)
        # 空桶（数据缺口）不写入
        data = data.dropna(subset=['open', 'close'])
        if data.empty:
            logger.warning(f"{kline} 无有效聚合数据")
            return 0

        # 丢掉当前未完成的桶，避免回测读到半根 K 线
        offset = to_offset(rule)
        last_source = df.index.max()
        if data.index[-1] + offset > last_source + pd.Timedelta(minutes=1):
            data = data.iloc[:-1]
        if data.empty:
            logger.warning(f"{kline} 只有未完成的桶，跳过")
            return 0

        data = data.copy()
        data.index.name = 'timestamp'
        written = SqliteOper.insert_or_update_data(symbol=cls.symbol, data=data, kline=kline, cursor=cursor)
        logger.debug(f"[2,4,2]<klineProcess-main>:{kline} 写入 {written} 行")
        return written
