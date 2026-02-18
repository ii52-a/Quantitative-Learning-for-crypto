"""
统一数据服务模块

提供：
- 统一的数据获取接口（数据库优先，API备用）
- 代理支持（解决地区限制）
- 正确的interval映射
- 完善的错误处理
- 分批获取大数据量
- 自动初始化
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from binance import Client
from dotenv import load_dotenv

from Config import ApiConfig
from Data.database import DatabaseManager, KlineRepository, DatabaseConfig
from app_logger.logger_setup import Logger

load_dotenv()
logger = Logger(__name__)


INTERVAL_MAP = {
    "1min": Client.KLINE_INTERVAL_1MINUTE,
    "3min": Client.KLINE_INTERVAL_3MINUTE,
    "5min": Client.KLINE_INTERVAL_5MINUTE,
    "15min": Client.KLINE_INTERVAL_15MINUTE,
    "30min": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "2h": Client.KLINE_INTERVAL_2HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "6h": Client.KLINE_INTERVAL_6HOUR,
    "12h": Client.KLINE_INTERVAL_12HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
    "1w": Client.KLINE_INTERVAL_1WEEK,
    "1M": Client.KLINE_INTERVAL_1MONTH,
}

INTERVAL_TO_MINUTES = {
    "1min": 1, "3min": 3, "5min": 5, "15min": 15, "30min": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440, "1w": 10080, "1M": 43200,
}


@dataclass
class DataServiceConfig:
    use_proxy: bool = False
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 7890
    testnet: bool = True
    prefer_database: bool = True
    auto_init: bool = True
    init_days: int = 1000


class DataSourceError(Exception):
    """数据源错误"""
    pass


class RegionRestrictedError(DataSourceError):
    """地区限制错误"""
    pass


class UnifiedDataService:
    MAX_API_LIMIT = 1000

    def __init__(self, config: DataServiceConfig | None = None):
        self.config = config or DataServiceConfig()
        self.db_manager = DatabaseManager()
        self._client: Client | None = None
        self._api_key = os.getenv("BINANCE_API_KEY", "") or os.getenv("API_KEY", "")
        self._api_secret = os.getenv("BINANCE_API_SECRET", "") or os.getenv("API_SECRET", "")

    def _get_client(self) -> Client:
        if self._client is not None:
            return self._client

        requests_params = {"timeout": (12, 32)}

        if self.config.use_proxy:
            proxies = {
                "http": f"http://{self.config.proxy_host}:{self.config.proxy_port}",
                "https": f"http://{self.config.proxy_host}:{self.config.proxy_port}",
            }
            requests_params["proxies"] = proxies
            logger.info(f"使用代理: {self.config.proxy_host}:{self.config.proxy_port}")

        self._client = Client(
            api_key=self._api_key,
            api_secret=self._api_secret,
            requests_params=requests_params,
        )
        return self._client

    def _close_client(self):
        if self._client:
            self._client = None

    def get_interval(self, interval: str) -> str:
        if interval in INTERVAL_MAP:
            return INTERVAL_MAP[interval]
        if interval in INTERVAL_MAP.values():
            return interval
        raise ValueError(f"不支持的K线周期: {interval}")

    def get_klines_from_database(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        repo = self.db_manager.get_repository(symbol)
        table_name = f"kline_{interval}"

        if not repo.table_exists(table_name):
            logger.info(f"[{symbol}] 数据库表 {table_name} 不存在")
            return pd.DataFrame()

        df = repo.get_klines(table_name=table_name, limit=limit)
        logger.info(f"[{symbol}] 从数据库获取 {len(df)} 条数据")
        return df

    def get_klines_from_api(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        try:
            client = self._get_client()
            binance_interval = self.get_interval(interval)

            actual_limit = min(limit, self.MAX_API_LIMIT)
            logger.info(f"[{symbol}] 从API获取数据: interval={binance_interval}, limit={actual_limit}")

            klines = client.futures_klines(
                symbol=symbol,
                interval=binance_interval,
                limit=actual_limit,
            )

            if not klines:
                logger.warning(f"[{symbol}] API返回空数据")
                return pd.DataFrame()

            df = self._parse_klines(klines)
            logger.info(f"[{symbol}] API获取成功: {len(df)} 条")
            return df

        except Exception as e:
            error_msg = str(e).lower()

            if "restricted location" in error_msg or "eligibility" in error_msg:
                logger.error(f"[{symbol}] 地区限制: {e}")
                raise RegionRestrictedError(
                    f"API访问受限，请配置代理或使用VPN。错误: {e}"
                )

            if "invalid interval" in error_msg:
                logger.error(f"[{symbol}] 无效周期: interval={interval}, 错误: {e}")
                raise DataSourceError(
                    f"无效的K线周期 '{interval}'，支持的周期: {list(INTERVAL_MAP.keys())}"
                )

            logger.error(f"[{symbol}] API获取失败: {e}")
            raise DataSourceError(f"API数据获取失败: {e}")

    def get_klines_from_api_batch(
        self,
        symbol: str,
        interval: str,
        total_limit: int = 5000,
        progress_callback=None,
    ) -> pd.DataFrame:
        """分批获取大量数据"""
        if total_limit <= self.MAX_API_LIMIT:
            return self.get_klines_from_api(symbol, interval, total_limit)

        client = self._get_client()
        binance_interval = self.get_interval(interval)

        all_klines = []
        batches = (total_limit + self.MAX_API_LIMIT - 1) // self.MAX_API_LIMIT

        logger.info(f"[{symbol}] 分批获取数据: 总量={total_limit}, 批次={batches}")

        end_time = None

        for i in range(batches):
            batch_limit = min(self.MAX_API_LIMIT, total_limit - len(all_klines))

            params = {
                "symbol": symbol,
                "interval": binance_interval,
                "limit": batch_limit,
            }

            if end_time:
                params["endTime"] = end_time - 1

            try:
                klines = client.futures_klines(**params)
                if not klines:
                    break

                all_klines = klines + all_klines
                end_time = int(klines[0][0])

                if progress_callback:
                    progress_callback(len(all_klines), total_limit)

                logger.info(f"[{symbol}] 批次 {i+1}/{batches}: 获取 {len(klines)} 条, 累计 {len(all_klines)} 条")

                time.sleep(ApiConfig.API_BASE_GET_INTERVAL)

                if len(all_klines) >= total_limit:
                    break

            except Exception as e:
                logger.error(f"[{symbol}] 批次 {i+1} 获取失败: {e}")
                break

        if not all_klines:
            return pd.DataFrame()

        df = self._parse_klines(all_klines[-total_limit:])
        logger.info(f"[{symbol}] 分批获取完成: {len(df)} 条")
        return df

    def _parse_klines(self, klines: list) -> pd.DataFrame:
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Shanghai')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df

    def save_to_database(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> int:
        if df.empty:
            return 0

        repo = self.db_manager.get_repository(symbol)
        table_name = f"kline_{interval}"

        if not repo.table_exists(table_name):
            repo.init_tables([table_name])

        count = repo.insert_klines(df, table_name=table_name)
        logger.info(f"[{symbol}] 保存到数据库: {count} 条")
        return count

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        prefer_database: bool | None = None,
    ) -> pd.DataFrame:
        prefer_db = prefer_database if prefer_database is not None else self.config.prefer_database

        if prefer_db:
            df = self.get_klines_from_database(symbol, interval, limit)
            if len(df) >= limit:
                return df.tail(limit)
            logger.info(f"[{symbol}] 数据库数据不足 ({len(df)}/{limit})，尝试API获取")

        try:
            df = self.get_klines_from_api_batch(symbol, interval, limit)
            if not df.empty:
                self.save_to_database(df, symbol, interval)
            return df.tail(limit) if len(df) > limit else df
        except RegionRestrictedError:
            logger.warning(f"[{symbol}] API地区限制，尝试从数据库获取")
            return self.get_klines_from_database(symbol, interval, limit)
        except Exception as e:
            logger.error(f"[{symbol}] 数据获取失败: {e}")
            if prefer_db:
                return self.get_klines_from_database(symbol, interval, limit)
            raise

    def get_backtest_data(
        self,
        symbol: str,
        interval: str,
        data_num: int = 500,
    ) -> pd.DataFrame:
        total_needed = data_num + ApiConfig.GET_COUNT
        logger.info(f"[{symbol}] 回测数据请求: 用户设置={data_num}, 额外填充={ApiConfig.GET_COUNT}, 总计={total_needed}")
        return self.get_klines(symbol, interval, total_needed)

    def update_local_data(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        progress_callback=None,
    ) -> int:
        df = self.get_klines_from_api_batch(symbol, interval, limit, progress_callback)
        if df.empty:
            return 0
        return self.save_to_database(df, symbol, interval)

    def initialize_database(
        self,
        symbol: str,
        interval: str,
        days: int = 1000,
        progress_callback=None,
    ) -> int:
        """初始化数据库，获取指定天数的历史数据"""
        minutes_per_kline = INTERVAL_TO_MINUTES.get(interval, 30)
        total_klines = (days * 24 * 60) // minutes_per_kline

        logger.info(f"[{symbol}] 初始化数据库: {days}天 ≈ {total_klines}条K线")

        return self.update_local_data(symbol, interval, total_klines, progress_callback)

    def check_and_auto_init(
        self,
        symbol: str,
        interval: str,
        min_count: int = 500,
    ) -> bool:
        """检查数据库，如果数据不足则自动初始化"""
        if not self.config.auto_init:
            return False

        repo = self.db_manager.get_repository(symbol)
        table_name = f"kline_{interval}"

        if not repo.table_exists(table_name):
            logger.info(f"[{symbol}] 数据库表不存在，开始初始化...")
            self.initialize_database(symbol, interval, self.config.init_days)
            return True

        count = repo.get_count(table_name)
        if count < min_count:
            logger.info(f"[{symbol}] 数据不足 ({count}/{min_count})，开始初始化...")
            self.initialize_database(symbol, interval, self.config.init_days)
            return True

        return False

    def get_data_status(self, symbol: str) -> dict[str, Any]:
        repo = self.db_manager.get_repository(symbol)
        status = {"symbol": symbol, "intervals": {}}

        for interval in INTERVAL_MAP.keys():
            table_name = f"kline_{interval}"
            if repo.table_exists(table_name):
                span = repo.get_time_span(table_name)
                status["intervals"][interval] = span
            else:
                status["intervals"][interval] = {"count": 0, "start": None, "end": None}

        return status


_data_service: UnifiedDataService | None = None


def get_data_service(config: DataServiceConfig | None = None) -> UnifiedDataService:
    global _data_service
    if _data_service is None:
        _data_service = UnifiedDataService(config)
    return _data_service
