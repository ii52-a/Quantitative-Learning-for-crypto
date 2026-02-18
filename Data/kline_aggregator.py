"""
多周期K线数据联动更新模块

功能：
1. 基于基础数据（1分钟K线）自动聚合多周期K线
2. 支持增量更新，避免全量重算
3. 数据一致性保证
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from Config import ApiConfig
from app_logger.logger_setup import Logger

logger = Logger(__name__)


@dataclass
class KlineInterval:
    name: str
    minutes: int
    resample_rule: str


class KlineAggregator:
    SUPPORTED_INTERVALS: list[KlineInterval] = [
        KlineInterval("1min", 1, "1min"),
        KlineInterval("3min", 3, "3min"),
        KlineInterval("5min", 5, "5min"),
        KlineInterval("15min", 15, "15min"),
        KlineInterval("30min", 30, "30min"),
        KlineInterval("1h", 60, "1h"),
        KlineInterval("2h", 120, "2h"),
        KlineInterval("4h", 240, "4h"),
        KlineInterval("6h", 360, "6h"),
        KlineInterval("12h", 720, "12h"),
        KlineInterval("1d", 1440, "1d"),
        KlineInterval("1w", 10080, "1W"),
        KlineInterval("1M", 43200, "1ME"),
    ]

    INTERVAL_MAP: dict[str, KlineInterval] = {i.name: i for i in SUPPORTED_INTERVALS}

    @classmethod
    def get_interval(cls, name: str) -> KlineInterval | None:
        return cls.INTERVAL_MAP.get(name)

    @classmethod
    def aggregate(
        cls,
        base_df: pd.DataFrame,
        interval: str | KlineInterval,
    ) -> pd.DataFrame:
        if isinstance(interval, str):
            interval = cls.get_interval(interval)
            if interval is None:
                raise ValueError(f"不支持的周期: {interval}")

        if base_df.empty:
            return base_df

        df = base_df.copy()

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "quote_asset_volume": "sum",
            "number_of_trades": "sum",
            "taker_buy_base_asset_volume": "sum",
            "taker_buy_quote_asset_volume": "sum",
        }

        if "close_time" in df.columns:
            agg_dict["close_time"] = "last"

        result = df.resample(rule=interval.resample_rule, origin="start_day").agg(
            agg_dict
        )
        result = result.dropna(subset=["open", "close"])

        return result


class MultiPeriodDataManager:
    DEFAULT_INTERVALS = ["3min", "5min", "15min", "30min", "1h", "4h", "1d"]

    def __init__(self, symbol: str, data_dir: Path | None = None):
        self.symbol = symbol
        self.data_dir = data_dir or Path(ApiConfig.LOCAL_DATA_SQLITE_DIR)
        self._base_data: pd.DataFrame | None = None
        self._aggregate_cache: dict[str, pd.DataFrame] = {}
        self._last_base_update: datetime | None = None

    def load_base_data(self, df: pd.DataFrame) -> None:
        self._base_data = df.copy()
        self._last_base_update = datetime.now(timezone.utc)
        self._aggregate_cache.clear()
        logger.info(f"[{self.symbol}] 基础数据加载完成: {len(df)}条")

    def update_base_data(self, new_data: pd.DataFrame) -> None:
        if self._base_data is None:
            self._base_data = new_data.copy()
        else:
            self._base_data = pd.concat([self._base_data, new_data])
            self._base_data = self._base_data[~self._base_data.index.duplicated(keep="last")]
            self._base_data = self._base_data.sort_index()

        self._last_base_update = datetime.now(timezone.utc)
        self._invalidate_cache()
        logger.debug(f"[{self.symbol}] 基础数据更新: +{len(new_data)}条, 总计{len(self._base_data)}条")

    def _invalidate_cache(self) -> None:
        self._aggregate_cache.clear()

    def get_aggregated(self, interval: str) -> pd.DataFrame | None:
        if self._base_data is None:
            return None

        if interval in self._aggregate_cache:
            return self._aggregate_cache[interval]

        try:
            result = KlineAggregator.aggregate(self._base_data, interval)
            self._aggregate_cache[interval] = result
            return result
        except Exception as e:
            logger.error(f"[{self.symbol}] 聚合{interval}失败: {e}")
            return None

    def update_all_aggregates(self, intervals: list[str] | None = None) -> dict[str, int]:
        intervals = intervals or self.DEFAULT_INTERVALS
        results = {}

        for interval in intervals:
            df = self.get_aggregated(interval)
            if df is not None:
                results[interval] = len(df)
                logger.info(f"[{self.symbol}] {interval} 聚合完成: {len(df)}条")
            else:
                results[interval] = 0
                logger.warning(f"[{self.symbol}] {interval} 聚合失败")

        return results

    @property
    def base_data_count(self) -> int:
        return len(self._base_data) if self._base_data is not None else 0

    @property
    def last_update_time(self) -> datetime | None:
        return self._last_base_update

    def get_status(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "base_count": self.base_data_count,
            "last_update": self._last_base_time_str,
            "cached_intervals": list(self._aggregate_cache.keys()),
        }

    @property
    def _last_base_time_str(self) -> str:
        if self._last_base_update is None:
            return "未更新"
        return self._last_base_update.strftime("%Y-%m-%d %H:%M:%S")


class DataUpdateCoordinator:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.manager = MultiPeriodDataManager(symbol)
        self._update_callbacks: list[callable] = []
        self._is_updating = False

    def register_callback(self, callback: callable) -> None:
        self._update_callbacks.append(callback)

    async def notify_callbacks(self, event: str, data: dict[str, Any]) -> None:
        for callback in self._update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, data)
                else:
                    callback(event, data)
            except Exception as e:
                logger.warning(f"回调执行失败: {e}")

    async def initialize(self, base_data: pd.DataFrame) -> None:
        logger.info(f"[{self.symbol}] 初始化多周期数据管理器...")
        self.manager.load_base_data(base_data)
        results = self.manager.update_all_aggregates()
        await self.notify_callbacks(
            "initialized",
            {"symbol": self.symbol, "intervals": results},
        )
        logger.info(f"[{self.symbol}] 初始化完成: {results}")

    async def incremental_update(self, new_data: pd.DataFrame) -> None:
        if self._is_updating:
            logger.debug(f"[{self.symbol}] 更新进行中，跳过本次更新")
            return

        self._is_updating = True
        try:
            self.manager.update_base_data(new_data)
            self.manager.update_all_aggregates()
            await self.notify_callbacks(
                "updated",
                {
                    "symbol": self.symbol,
                    "new_count": len(new_data),
                    "total_count": self.manager.base_data_count,
                },
            )
        finally:
            self._is_updating = False

    def get_data(self, interval: str) -> pd.DataFrame | None:
        if interval == "1min":
            return self.manager._base_data
        return self.manager.get_aggregated(interval)

    def get_status(self) -> dict[str, Any]:
        return self.manager.get_status()
