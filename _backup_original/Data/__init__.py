"""
Data模块

提供数据获取、存储和处理功能
"""

from Data.binance_client import (
    BinanceClient,
    BinanceClientFactory,
    BinanceAPIError,
    KlineData,
)
from Data.database import (
    DatabaseConnection,
    DatabaseConfig,
    DatabaseManager,
    KlineRepository,
)
from Data.kline_aggregator import (
    KlineAggregator,
    KlineInterval,
    MultiPeriodDataManager,
    DataUpdateCoordinator,
)
from Data.data_service import (
    UnifiedDataService,
    DataServiceConfig,
    DataSourceError,
    RegionRestrictedError,
    get_data_service,
)

__all__ = [
    "BinanceClient",
    "BinanceClientFactory",
    "BinanceAPIError",
    "KlineData",
    "DatabaseConnection",
    "DatabaseConfig",
    "DatabaseManager",
    "KlineRepository",
    "KlineAggregator",
    "KlineInterval",
    "MultiPeriodDataManager",
    "DataUpdateCoordinator",
    "UnifiedDataService",
    "DataServiceConfig",
    "DataSourceError",
    "RegionRestrictedError",
    "get_data_service",
]
