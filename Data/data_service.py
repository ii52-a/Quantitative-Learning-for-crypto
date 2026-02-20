"""
统一数据服务模块

提供：
- 统一的数据获取接口（数据库优先，API备用）
- 基于基础数据（1分钟）聚合生成多周期数据，减少API调用
- API请求限流保护，防止被封禁
- 代理支持（解决地区限制）
- 正确的interval映射
- 完善的错误处理
- 分批获取大数据量
- 自动初始化
- 数据聚合延迟控制在5秒以内
"""

import os
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from binance import Client
from dotenv import load_dotenv

from Config import ApiConfig
from Data.database import DatabaseManager, KlineRepository, DatabaseConfig
from Data.kline_aggregator import KlineAggregator
from Data.rate_limiter import RateLimiter, RateLimitConfig, get_global_rate_limiter
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

SECOND_INTERVAL_TO_SECONDS = {
    "1s": 1,
    "5s": 5,
    "15s": 15,
}

BASE_INTERVAL = "1min"
AGGREGATABLE_INTERVALS = ["3min", "5min", "15min", "30min", "1h", "2h", "4h", "6h", "12h", "1d"]
MAX_AGGREGATION_DELAY_SECONDS = 5


@dataclass
class DataServiceConfig:
    use_proxy: bool = False
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 7890
    testnet: bool = True
    prefer_database: bool = True
    auto_init: bool = True
    init_days: int = 1000
    
    enable_rate_limit: bool = True
    requests_per_second: float = 3.0
    requests_per_minute: int = 600
    min_request_interval: float = 0.5
    cache_ttl: int = 300


class DataSourceError(Exception):
    pass


class RegionRestrictedError(DataSourceError):
    pass


class UnifiedDataService:
    MAX_API_LIMIT = 1000

    def __init__(self, config: DataServiceConfig | None = None):
        self.config = config or DataServiceConfig()
        self.db_manager = DatabaseManager()
        self._client: Client | None = None
        self._api_key = os.getenv("BINANCE_API_KEY", "") or os.getenv("API_KEY", "")
        self._api_secret = os.getenv("BINANCE_API_SECRET", "") or os.getenv("API_SECRET", "")
        self._aggregation_cache: dict[str, dict[str, pd.DataFrame]] = {}
        self._cache_lock = threading.Lock()
        self._last_aggregation_time: dict[str, float] = {}
        
        if self.config.enable_rate_limit:
            rate_config = RateLimitConfig(
                requests_per_second=self.config.requests_per_second,
                requests_per_minute=self.config.requests_per_minute,
                min_request_interval=self.config.min_request_interval,
                cache_ttl=self.config.cache_ttl,
            )
            self._rate_limiter = RateLimiter(rate_config)
        else:
            self._rate_limiter = None

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
        if interval in SECOND_INTERVAL_TO_SECONDS:
            # Binance标准K线接口不支持秒级，秒级数据走聚合成交逻辑
            return interval
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
        try:
            repo = self.db_manager.get_repository(symbol)
            table_name = f"kline_{interval}"

            if not repo.table_exists(table_name):
                logger.info(f"[{symbol}] 数据库表 {table_name} 不存在")
                return pd.DataFrame()

            df = repo.get_klines(table_name=table_name, limit=limit)
            logger.info(f"[{symbol}] 从数据库获取 {len(df)} 条数据")
            return df
        except Exception as e:
            logger.error(f"[{symbol}] 数据库读取失败: {e}")
            return pd.DataFrame()

    def get_klines_from_api(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        if self._rate_limiter:
            cache_key = f"klines_{symbol}_{interval}_{limit}"
            cached, hit = self._rate_limiter.get_cached(cache_key)
            if hit:
                logger.info(f"[{symbol}] 使用缓存数据")
                return cached
        
        if self._rate_limiter:
            if not self._rate_limiter.acquire(f"klines_{symbol}"):
                logger.warning(f"[{symbol}] 请求被限流，尝试从数据库获取")
                return self.get_klines_from_database(symbol, interval, limit)
        
        try:
            client = self._get_client()
            binance_interval = self.get_interval(interval)

            actual_limit = min(limit, self.MAX_API_LIMIT)
            logger.info(f"[{symbol}] 从API获取数据: interval={binance_interval}, limit={actual_limit}")

            start_time = time.time()
            klines = client.futures_klines(
                symbol=symbol,
                interval=binance_interval,
                limit=actual_limit,
            )
            response_time = time.time() - start_time

            if not klines:
                logger.warning(f"[{symbol}] API返回空数据")
                return pd.DataFrame()

            df = self._parse_klines(klines)
            logger.info(f"[{symbol}] API获取成功: {len(df)} 条 (耗时: {response_time:.2f}s)")
            
            if self._rate_limiter:
                self._rate_limiter.record_request(f"klines_{symbol}", True, response_time)
                cache_key = f"klines_{symbol}_{interval}_{limit}"
                self._rate_limiter.set_cache(cache_key, df)
            
            return df

        except Exception as e:
            error_msg = str(e).lower()
            
            if self._rate_limiter:
                self._rate_limiter.record_request(f"klines_{symbol}", False)

            if "restricted location" in error_msg or "eligibility" in error_msg:
                logger.error(f"[{symbol}] 地区限制: {e}")
                raise RegionRestrictedError(f"API访问受限，请配置代理或使用VPN。错误: {e}")

            if "invalid interval" in error_msg:
                logger.error(f"[{symbol}] 无效周期: interval={interval}, 错误: {e}")
                raise DataSourceError(f"无效的K线周期 '{interval}'，支持的周期: {list(INTERVAL_MAP.keys())}")

            if "invalid symbol" in error_msg:
                logger.error(f"[{symbol}] 无效交易对: {e}")
                raise DataSourceError(f"无效的交易对 '{symbol}'")

            logger.error(f"[{symbol}] API获取失败: {e}")
            raise DataSourceError(f"API数据获取失败: {e}")

    def get_klines_from_api_batch(
        self,
        symbol: str,
        interval: str,
        total_limit: int = 5000,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        if total_limit <= self.MAX_API_LIMIT:
            try:
                return self.get_klines_from_api(symbol, interval, total_limit)
            except Exception as e:
                logger.error(f"[{symbol}] API获取失败: {e}")
                return pd.DataFrame()

        try:
            client = self._get_client()
            binance_interval = self.get_interval(interval)

            all_klines = []
            batches = (total_limit + self.MAX_API_LIMIT - 1) // self.MAX_API_LIMIT

            logger.info(f"[{symbol}] 分批获取数据: 总量={total_limit}, 批次={batches}")

            end_time = None
            batch_delay = self.config.min_request_interval if self.config.enable_rate_limit else 0.3

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
                    if self._rate_limiter:
                        if not self._rate_limiter.acquire(f"batch_{symbol}"):
                            time.sleep(batch_delay * 2)
                    
                    start_time = time.time()
                    klines = client.futures_klines(**params)
                    response_time = time.time() - start_time
                    
                    if not klines:
                        break

                    all_klines = klines + all_klines
                    end_time = int(klines[0][0])

                    if progress_callback:
                        progress_callback(len(all_klines), total_limit)

                    logger.info(f"[{symbol}] 批次 {i+1}/{batches}: 获取 {len(klines)} 条, 累计 {len(all_klines)} 条 (耗时: {response_time:.2f}s)")
                    
                    if self._rate_limiter:
                        self._rate_limiter.record_request(f"batch_{symbol}", True, response_time)
                    
                    if i < batches - 1:
                        time.sleep(batch_delay)

                    time.sleep(ApiConfig.API_BASE_GET_INTERVAL)

                    if len(all_klines) >= total_limit:
                        break

                except Exception as e:
                    logger.error(f"[{symbol}] 批次 {i+1} 获取失败: {e}")
                    if i == 0:
                        raise
                    break

            if not all_klines:
                return pd.DataFrame()

            df = self._parse_klines(all_klines[-total_limit:])
            logger.info(f"[{symbol}] 分批获取完成: {len(df)} 条")
            return df

        except RegionRestrictedError:
            raise
        except Exception as e:
            logger.error(f"[{symbol}] 分批获取失败: {e}")
            return pd.DataFrame()

    def _parse_klines(self, klines: list) -> pd.DataFrame:
        valid_klines = []
        for k in klines:
            if len(k) < 6:
                continue
            try:
                for i in range(1, 6):
                    float(k[i])
                valid_klines.append(k)
            except (ValueError, TypeError):
                continue
        
        if not valid_klines:
            return pd.DataFrame()
            
        df = pd.DataFrame(valid_klines, columns=[
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

        try:
            repo = self.db_manager.get_repository(symbol)
            table_name = f"kline_{interval}"

            if not repo.table_exists(table_name):
                repo.init_tables([table_name])

            count = repo.insert_klines(df, table_name=table_name)
            logger.info(f"[{symbol}] 保存到数据库: {count} 条")
            return count
        except Exception as e:
            logger.error(f"[{symbol}] 保存数据库失败: {e}")
            return 0

    def _can_aggregate_from_base(self, interval: str) -> bool:
        return interval in AGGREGATABLE_INTERVALS

    def _get_base_data_for_aggregation(
        self,
        symbol: str,
        target_interval: str,
        target_limit: int,
    ) -> pd.DataFrame | None:
        target_minutes = INTERVAL_TO_MINUTES.get(target_interval, 30)
        base_limit = target_limit * target_minutes
        
        base_df = self.get_klines_from_database(symbol, BASE_INTERVAL, base_limit)
        
        if base_df.empty:
            logger.debug(f"[{symbol}] 基础数据为空，无法聚合")
            return None
        
        return base_df

    def _aggregate_from_base(
        self,
        base_df: pd.DataFrame,
        target_interval: str,
        target_limit: int,
    ) -> pd.DataFrame | None:
        start_time = time.time()
        
        try:
            aggregated = KlineAggregator.aggregate(base_df, target_interval)
            
            if aggregated.empty:
                return None
            
            aggregation_time = time.time() - start_time
            if aggregation_time > MAX_AGGREGATION_DELAY_SECONDS:
                logger.warning(
                    f"聚合耗时 {aggregation_time:.2f}s 超过阈值 {MAX_AGGREGATION_DELAY_SECONDS}s"
                )
            
            return aggregated.tail(target_limit)
            
        except Exception as e:
            logger.error(f"聚合失败: {e}")
            return None

    def _get_cached_aggregation(
        self,
        symbol: str,
        interval: str,
    ) -> pd.DataFrame | None:
        with self._cache_lock:
            if symbol in self._aggregation_cache:
                return self._aggregation_cache[symbol].get(interval)
        return None

    def _set_cached_aggregation(
        self,
        symbol: str,
        interval: str,
        df: pd.DataFrame,
    ) -> None:
        with self._cache_lock:
            if symbol not in self._aggregation_cache:
                self._aggregation_cache[symbol] = {}
            self._aggregation_cache[symbol][interval] = df
            self._last_aggregation_time[f"{symbol}_{interval}"] = time.time()

    def get_klines_via_aggregation(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> pd.DataFrame | None:
        if not self._can_aggregate_from_base(interval):
            return None
        
        base_df = self._get_base_data_for_aggregation(symbol, interval, limit)
        if base_df is None or base_df.empty:
            return None
        
        aggregated = self._aggregate_from_base(base_df, interval, limit)
        if aggregated is not None and not aggregated.empty:
            logger.info(f"[{symbol}] 通过聚合获取 {interval} 数据: {len(aggregated)} 条")
            return aggregated
        
        return None

    def _fetch_agg_trades(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """获取聚合成交（用于秒级K线构造）"""
        client = self._get_client()
        trades = client.futures_aggregate_trades(
            symbol=symbol,
            startTime=start_time_ms,
            endTime=end_time_ms,
            limit=min(1000, max(1, limit)),
        )
        return trades or []

    def get_second_klines_from_api(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """从聚合成交构造秒级K线（1s/5s/15s）"""
        if interval not in SECOND_INTERVAL_TO_SECONDS:
            raise DataSourceError(f"不支持的秒级周期: {interval}")

        sec = SECOND_INTERVAL_TO_SECONDS[interval]
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        # 扩大窗口，降低无成交导致的样本不足
        window_seconds = max(limit * sec * 3, 300)
        start_ms = end_ms - window_seconds * 1000

        all_trades: list[dict[str, Any]] = []
        cursor = start_ms
        max_rounds = 20

        for _ in range(max_rounds):
            trades = self._fetch_agg_trades(symbol, cursor, end_ms, 1000)
            if not trades:
                break

            all_trades.extend(trades)
            last_ts = int(trades[-1].get("T", cursor))
            if len(trades) < 1000 or last_ts >= end_ms - 1:
                break
            cursor = last_ts + 1
            time.sleep(0.05)

        if not all_trades:
            return pd.DataFrame()

        trade_df = pd.DataFrame(all_trades)
        trade_df["timestamp"] = pd.to_datetime(trade_df["T"].astype("int64"), unit="ms", utc=True)
        trade_df["price"] = trade_df["p"].astype(float)
        trade_df["qty"] = trade_df["q"].astype(float)
        trade_df = trade_df.set_index("timestamp").sort_index()

        ohlcv_1s = trade_df.resample("1s").agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("qty", "sum"),
        ).dropna(subset=["open", "high", "low", "close"])

        if ohlcv_1s.empty:
            return pd.DataFrame()

        if sec > 1:
            rule = f"{sec}s"
            ohlcv = ohlcv_1s.resample(rule).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna(subset=["open", "high", "low", "close"])
        else:
            ohlcv = ohlcv_1s

        ohlcv.index = ohlcv.index.tz_convert("Asia/Shanghai")
        return ohlcv.tail(limit)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        prefer_database: bool | None = None,
    ) -> pd.DataFrame:
        prefer_db = prefer_database if prefer_database is not None else self.config.prefer_database

        if interval in SECOND_INTERVAL_TO_SECONDS:
            try:
                return self.get_second_klines_from_api(symbol, interval, limit)
            except Exception as e:
                logger.error(f"[{symbol}] 秒级数据获取失败: {e}")
                return pd.DataFrame()

        if prefer_db:
            df = self.get_klines_from_database(symbol, interval, limit)
            if len(df) >= limit:
                return df.tail(limit)
            
            if self._can_aggregate_from_base(interval):
                aggregated = self.get_klines_via_aggregation(symbol, interval, limit)
                if aggregated is not None and len(aggregated) >= limit:
                    return aggregated.tail(limit)
            
            missing_count = limit - len(df)
            logger.info(f"[{symbol}] 数据库数据不足 ({len(df)}/{limit})，需补充 {missing_count} 条")

        if interval == BASE_INTERVAL:
            try:
                df = self.get_klines_from_api_batch(symbol, interval, limit)
                if not df.empty:
                    self.save_to_database(df, symbol, interval)
                    return df.tail(limit) if len(df) > limit else df
            except RegionRestrictedError:
                logger.warning(f"[{symbol}] API地区限制，尝试从数据库获取")
                return self.get_klines_from_database(symbol, interval, limit)
            except DataSourceError as e:
                logger.error(f"[{symbol}] 数据源错误: {e}")
                return self.get_klines_from_database(symbol, interval, limit)
            except Exception as e:
                logger.error(f"[{symbol}] 数据获取失败: {e}")
                return self.get_klines_from_database(symbol, interval, limit)
        
        try:
            df = self.get_klines_from_api_batch(symbol, interval, limit)
            if not df.empty:
                self.save_to_database(df, symbol, interval)
                return df.tail(limit) if len(df) > limit else df
            logger.warning(f"[{symbol}] API返回空数据，尝试从数据库获取")
            return self.get_klines_from_database(symbol, interval, limit)
        except RegionRestrictedError:
            logger.warning(f"[{symbol}] API地区限制，尝试从数据库获取")
            return self.get_klines_from_database(symbol, interval, limit)
        except DataSourceError as e:
            logger.error(f"[{symbol}] 数据源错误: {e}")
            return self.get_klines_from_database(symbol, interval, limit)
        except Exception as e:
            logger.error(f"[{symbol}] 数据获取失败: {e}")
            return self.get_klines_from_database(symbol, interval, limit)

    def get_backtest_data(
        self,
        symbol: str,
        interval: str,
        data_num: int = 500,
    ) -> pd.DataFrame:
        total_needed = data_num + ApiConfig.GET_COUNT
        logger.info(f"[{symbol}] 回测数据请求: 用户设置={data_num}, 总计={total_needed}")
        return self.get_klines(symbol, interval, total_needed)

    def update_local_data(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        progress_callback: Callable[[int, int], None] | None = None,
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
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        minutes_per_kline = INTERVAL_TO_MINUTES.get(interval, 30)
        total_klines = (days * 24 * 60) // minutes_per_kline

        logger.info(f"[{symbol}] 初始化数据库: {days}天 ≈ {total_klines}条K线")

        if interval != BASE_INTERVAL and self._can_aggregate_from_base(interval):
            base_minutes = INTERVAL_TO_MINUTES.get(BASE_INTERVAL, 1)
            base_klines = (days * 24 * 60) // base_minutes
            
            logger.info(f"[{symbol}] 优先初始化基础数据({BASE_INTERVAL})...")
            
            base_count = self.update_local_data(symbol, BASE_INTERVAL, base_klines, progress_callback)
            
            if base_count > 0:
                repo = self.db_manager.get_repository(symbol)
                base_df = repo.get_klines(table_name=f"kline_{BASE_INTERVAL}", limit=base_klines)
                
                if not base_df.empty:
                    aggregated = self._aggregate_from_base(base_df, interval, total_klines)
                    if aggregated is not None and not aggregated.empty:
                        agg_count = self.save_to_database(aggregated, symbol, interval)
                        logger.info(f"[{symbol}] 聚合生成 {interval} 数据: {agg_count} 条")
                        return agg_count
            
            logger.warning(f"[{symbol}] 基础数据初始化失败，直接获取 {interval} 数据")

        return self.update_local_data(symbol, interval, total_klines, progress_callback)

    def initialize_base_data(
        self,
        symbol: str,
        days: int = 1000,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        base_minutes = INTERVAL_TO_MINUTES.get(BASE_INTERVAL, 1)
        base_klines = (days * 24 * 60) // base_minutes
        
        logger.info(f"[{symbol}] 初始化基础数据({BASE_INTERVAL}): {days}天 ≈ {base_klines}条")
        
        return self.update_local_data(symbol, BASE_INTERVAL, base_klines, progress_callback)

    def check_and_auto_init(
        self,
        symbol: str,
        interval: str,
        min_count: int = 500,
    ) -> bool:
        if not self.config.auto_init:
            return False

        repo = self.db_manager.get_repository(symbol)
        table_name = f"kline_{interval}"

        if not repo.table_exists(table_name):
            logger.info(f"[{symbol}] 数据库表不存在，开始初始化...")
            try:
                self.initialize_database(symbol, interval, self.config.init_days)
                return True
            except Exception as e:
                logger.error(f"[{symbol}] 自动初始化失败: {e}")
                return False

        count = repo.get_count(table_name)
        if count < min_count:
            logger.info(f"[{symbol}] 数据不足 ({count}/{min_count})，开始初始化...")
            try:
                self.initialize_database(symbol, interval, self.config.init_days)
                return True
            except Exception as e:
                logger.error(f"[{symbol}] 自动初始化失败: {e}")
                return False

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
    
    def get_rate_limit_stats(self) -> dict[str, Any]:
        """获取限流统计信息"""
        if self._rate_limiter:
            return self._rate_limiter.get_stats()
        return {
            "rate_limiting_enabled": False,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
        }
    
    def clear_cache(self) -> None:
        """清空请求缓存"""
        if self._rate_limiter:
            self._rate_limiter.clear_cache()
            logger.info("数据服务缓存已清空")


_data_service: UnifiedDataService | None = None


def get_data_service(config: DataServiceConfig | None = None) -> UnifiedDataService:
    global _data_service
    if _data_service is None:
        _data_service = UnifiedDataService(config)
    return _data_service
