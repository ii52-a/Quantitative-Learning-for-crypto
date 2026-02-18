"""
币安API客户端模块

提供清晰的API接口设计，包含：
- 请求验证
- 错误处理
- 重试机制
- 类型注解
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from binance import AsyncClient, Client
from binance.enums import KLINE_INTERVAL_1MINUTE
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from Config import ApiConfig
from app_logger.logger_setup import Logger

load_dotenv()
logger = Logger(__name__)


class BinanceAPIError(Exception):
    """币安API错误基类"""
    pass


class RateLimitError(BinanceAPIError):
    """请求频率限制错误"""
    pass


class NetworkError(BinanceAPIError):
    """网络连接错误"""
    pass


@dataclass
class KlineData:
    """K线数据结构"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float
    trades: int
    taker_buy_base: float
    taker_buy_quote: float

    @classmethod
    def from_list(cls, data: list) -> "KlineData":
        return cls(
            timestamp=int(data[0]),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]),
            close_time=int(data[6]),
            quote_volume=float(data[7]),
            trades=int(data[8]),
            taker_buy_base=float(data[9]),
            taker_buy_quote=float(data[10]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "close_time": self.close_time,
            "quote_volume": self.quote_volume,
            "trades": self.trades,
            "taker_buy_base": self.taker_buy_base,
            "taker_buy_quote": self.taker_buy_quote,
        }


class BinanceClient:
    """币安API客户端封装"""

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

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        testnet: bool = True,
        timeout: tuple[int, int] = (12, 32),
    ):
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        self.testnet = testnet
        self.timeout = timeout
        self._client: AsyncClient | None = None
        self._sync_client: Client | None = None

    async def __aenter__(self) -> "BinanceClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._client is not None:
            return

        self._validate_credentials()
        self._client = await AsyncClient.create(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet,
        )
        logger.info(f"币安API连接成功 (测试网: {self.testnet})")

    async def close(self) -> None:
        if self._client:
            await self._client.close_connection()
            self._client = None

    def _validate_credentials(self) -> None:
        if not self.api_key or not self.api_secret:
            raise BinanceAPIError("缺少API密钥配置")

    def _get_interval(self, interval: str) -> str:
        if interval in self.INTERVAL_MAP:
            return self.INTERVAL_MAP[interval]
        return interval

    @retry(
        stop=stop_after_attempt(ApiConfig.MAX_RETRY),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((NetworkError, RateLimitError)),
    )
    async def get_klines(
        self,
        symbol: str,
        interval: str = "30min",
        limit: int = 500,
        start_time: datetime | int | None = None,
        end_time: datetime | int | None = None,
    ) -> list[KlineData]:
        if self._client is None:
            await self.connect()

        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": self._get_interval(interval),
            "limit": min(limit, ApiConfig.LIMIT),
        }

        if start_time is not None:
            if isinstance(start_time, datetime):
                params["startTime"] = int(start_time.timestamp() * 1000)
            else:
                params["startTime"] = start_time

        if end_time is not None:
            if isinstance(end_time, datetime):
                params["endTime"] = int(end_time.timestamp() * 1000)
            else:
                params["endTime"] = end_time

        try:
            klines = await self._client.futures_klines(**params)
            return [KlineData.from_list(k) for k in klines]
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "too many" in error_msg:
                raise RateLimitError(f"请求频率限制: {e}")
            if "connection" in error_msg or "timeout" in error_msg:
                raise NetworkError(f"网络错误: {e}")
            raise BinanceAPIError(f"API请求失败: {e}")

    async def get_klines_df(
        self,
        symbol: str,
        interval: str = "30min",
        limit: int = 500,
        start_time: datetime | int | None = None,
        end_time: datetime | int | None = None,
    ) -> pd.DataFrame:
        klines = await self.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
        )

        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame([k.to_dict() for k in klines])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime")
        return df

    async def get_latest_price(self, symbol: str) -> float:
        if self._client is None:
            await self.connect()

        try:
            ticker = await self._client.futures_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            raise BinanceAPIError(f"获取最新价格失败: {e}")

    async def get_exchange_info(self, symbol: str | None = None) -> dict[str, Any]:
        if self._client is None:
            await self.connect()

        try:
            if symbol:
                info = await self._client.futures_exchange_info(symbol=symbol)
            else:
                info = await self._client.futures_exchange_info()
            return info
        except Exception as e:
            raise BinanceAPIError(f"获取交易所信息失败: {e}")

    async def get_symbol_filters(self, symbol: str) -> dict[str, Any]:
        info = await self.get_exchange_info(symbol)
        for item in info.get("symbols", []):
            if item.get("symbol") == symbol:
                filters = {}
                for f in item.get("filters", []):
                    filters[f["filterType"]] = f
                return {
                    "price_precision": item.get("pricePrecision", 2),
                    "qty_precision": item.get("quantityPrecision", 3),
                    "filters": filters,
                }
        raise BinanceAPIError(f"未找到交易对: {symbol}")


class BinanceClientFactory:
    """客户端工厂类"""

    _instances: dict[str, BinanceClient] = {}

    @classmethod
    async def get_client(
        cls,
        api_key: str | None = None,
        api_secret: str | None = None,
        testnet: bool = True,
    ) -> BinanceClient:
        key = f"{api_key}:{testnet}"
        if key not in cls._instances:
            client = BinanceClient(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
            )
            await client.connect()
            cls._instances[key] = client
        return cls._instances[key]

    @classmethod
    async def close_all(cls) -> None:
        for client in cls._instances.values():
            await client.close()
        cls._instances.clear()
