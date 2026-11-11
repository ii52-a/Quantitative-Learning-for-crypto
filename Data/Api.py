import os
import sqlite3
from contextlib import closing
from pathlib import Path
from time import sleep
from typing import List

import pandas as pd
from binance import Client
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from Config import ApiConfig as Config
from Data.kline_process import KlineProcess
from Data.sqlite_oper import SqliteBase, SqliteOper
from app_logger.logger_setup import Logger
from urlibs import FormatUrlibs, get_progress

logger = Logger(__name__)
load_dotenv()

KLINE_COLUMNS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore',
]
NUMERIC_COLUMNS = [c for c in KLINE_COLUMNS if c not in ('timestamp', 'close_time', 'ignore')]


class Api:
    """币安期货 K 线读取 / 本地 1min 基准库维护。

    ping=False：构造 Api 时不发网络请求，避免 GUI 线程被网络阻塞；
    需要验证连通性时显式传 ping=True。
    """

    def __init__(self, ping: bool = False):
        logger.debug("初始化 Api 类，正在加载 API 密钥和客户端配置...")
        self.api_key = os.getenv("API_KEY")
        self.api_secret = os.getenv("API_SECRET")
        requests_params = {
            'timeout': Config.CUSTOM_TIMEOUT
        }
        self.local_data_base: SqliteBase = SqliteBase()
        self.local_data_oper: SqliteOper = SqliteOper()
        self.client = Client(api_key=self.api_key, api_secret=self.api_secret,
                             requests_params=requests_params, ping=ping)
        logger.debug("Api 初始化完成，开始数据操作")

    @staticmethod
    def _to_ms(value) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        # tz-aware 用绝对时间；naive 按本地时间解释，与旧逻辑保持一致
        return int(pd.Timestamp(value).timestamp() * 1000)

    @retry(stop=stop_after_attempt(Config.MAX_RETRY), wait=wait_fixed(Config.WAITING_TIME),
           retry=retry_if_exception_type(Config.RETRY_ERROR_ACCEPT), reraise=True)
    def get_futures_data(self, start_time=None, end_time=None, symbol: str = 'BTCUSDT',
                         interval: str = Client.KLINE_INTERVAL_30MINUTE,
                         limit: int = Config.LIMIT) -> pd.DataFrame:
        logger.debug(f"获取期货数据 start_time={start_time}, end_time={end_time}, "
                     f"symbol={symbol}, interval={interval}, limit={limit}")
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if start_time is not None:
            params['startTime'] = self._to_ms(start_time)
        if end_time is not None:
            params['endTime'] = self._to_ms(end_time)

        klines = self.client.futures_klines(**params)
        data = pd.DataFrame(klines, columns=KLINE_COLUMNS)
        for column in NUMERIC_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors='coerce')
        data['timestamp'] = data['timestamp'].astype('int64')
        data['close_time'] = data['close_time'].astype('int64')
        return data

    def local_data_main(self, symbol: str) -> None:
        """本地数据主程序：缺失则初始化，之后增量更新并重新聚合。"""
        logger.info(f" [1]<Api-main>:启动本地数据主程序 symbol={symbol}")
        path = Path(Config.LOCAL_DATA_SQLITE_DIR) / f"{symbol}_base.db"
        if not path.exists():
            logger.info(f"[1,1]<Api-main>:未找到本地数据库，将初始化基础 K 线数据 symbol={symbol}")
            self._init_base_k_line_data(symbol=symbol)
        self._update_base_k_line_data(symbol=symbol)
        logger.info(f"[1,2]<Api-main>:开始聚合常用k线数据: symbol={symbol}")
        KlineProcess.main(symbol=symbol)

    def _init_base_k_line_data(self, symbol: str) -> None:
        logger.info(f"开始初始化基础 K 线数据 symbol={symbol}")
        all_data: List[pd.DataFrame] = []
        needed: int = Config.LOCAL_MAX_HISTORY_ALLOW * 24 * 60
        end_time: pd.Timestamp = pd.Timestamp.now(tz='UTC')
        count = 0
        failures = 0

        with get_progress(total=needed, desc=f"正在初始化 {symbol}", unit="bar") as pbar:
            while count < needed:
                try:
                    data_df = self.get_futures_data(
                        end_time=end_time,
                        interval=Client.KLINE_INTERVAL_1MINUTE,
                        symbol=symbol,
                    )
                except Exception as e:
                    failures += 1
                    logger.error(f"API>>_init_base_k_line_data>>基础数据初始化失败:{e}")
                    logger.info(f"等待3秒后重试({failures}/5)")
                    if failures >= 5:
                        raise
                    sleep(3)
                    continue

                if data_df is None or data_df.empty:
                    logger.info(f"{symbol} 已到最早数据，停止初始化 (已获取 {count} 根)")
                    break

                all_data.insert(0, data_df)
                count += len(data_df)
                end_time = pd.to_datetime(int(data_df['timestamp'].iloc[0]), unit='ms', utc=True) \
                    - pd.Timedelta(milliseconds=1)
                pbar.n = count
                pbar.refresh()
                sleep(Config.API_BASE_GET_INTERVAL)

        if not all_data:
            logger.warning(f"{symbol} 未获取到任何基础数据")
            return

        data = pd.concat(all_data, ignore_index=True)
        data = FormatUrlibs.standard_timestamp(data)
        logger.info(f"获取未保存数据:{len(data)}")
        self.local_data_base.update_base_kline(symbol=symbol, data=data)
        logger.debug("基础 K 线初始化完成并写入数据库")

    def _update_base_k_line_data(self, symbol: str) -> None:
        logger.info(f"开始更新本地 K 线 symbol={symbol}")
        path = Path(Config.LOCAL_DATA_SQLITE_DIR) / f"{symbol}_base.db"
        if not path.exists():
            logger.warning(f"{path} 不存在，转为初始化")
            self._init_base_k_line_data(symbol=symbol)
            return

        with closing(sqlite3.connect(path)) as conn:
            cursor = conn.cursor()
            SqliteBase.init_sqlite(cursor=cursor)
            newest = SqliteOper.read_newest_timestamp(cursor=cursor)
            conn.commit()

        if newest is None:
            logger.warning(f"{symbol} 基准库为空，转为初始化")
            self._init_base_k_line_data(symbol=symbol)
            return

        start_time: pd.Timestamp = pd.to_datetime(newest, unit='ms', utc=True)
        logger.info(f"本地最新时间戳:{start_time}")
        add_data: List[pd.DataFrame] = []

        while True:
            data_df = self.get_futures_data(
                start_time=start_time,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                symbol=symbol,
            )
            if data_df is None or data_df.empty:
                logger.info(f"{symbol} 数据集已是最新")
                break

            add_data.append(data_df)
            last_ms = int(data_df['timestamp'].iloc[-1])
            start_time = pd.to_datetime(last_ms, unit='ms', utc=True)
            logger.debug(f"更新中,本次 {len(data_df)} 根,最新时间戳:{start_time}")

            if last_ms >= self._to_ms(pd.Timestamp.now(tz='UTC')):
                break
            if len(data_df) < Config.LIMIT:
                logger.debug(f"检测到结束数据标识:{len(data_df)}")
                break
            sleep(Config.API_BASE_GET_INTERVAL)

        if not add_data:
            return
        data_t = pd.concat(add_data, ignore_index=True)
        data_t = FormatUrlibs.standard_timestamp(data_t)
        count = self.local_data_base.update_base_kline(symbol=symbol, data=data_t)
        logger.info(f"K 线更新完成 symbol={symbol}, 写入 {count} 行")


if __name__ == '__main__':
    api = Api()
    api.local_data_main(symbol='ETHUSDT')
