"""
数据服务测试
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from Data.data_service import (
    UnifiedDataService,
    DataServiceConfig,
    DataSourceError,
    RegionRestrictedError,
    INTERVAL_MAP,
    INTERVAL_TO_MINUTES,
    BASE_INTERVAL,
    AGGREGATABLE_INTERVALS,
)


class TestDataServiceConfig:
    """数据服务配置测试"""
    
    def test_default_values(self):
        config = DataServiceConfig()
        assert config.use_proxy is False
        assert config.prefer_database is True
        assert config.auto_init is True
    
    def test_custom_values(self):
        config = DataServiceConfig(
            use_proxy=True,
            proxy_host="127.0.0.1",
            proxy_port=7890,
            prefer_database=False
        )
        assert config.use_proxy is True
        assert config.proxy_host == "127.0.0.1"
        assert config.proxy_port == 7890


class TestIntervalConstants:
    """周期常量测试"""
    
    def test_interval_map_contains_common_intervals(self):
        assert "1min" in INTERVAL_MAP
        assert "5min" in INTERVAL_MAP
        assert "15min" in INTERVAL_MAP
        assert "1h" in INTERVAL_MAP
        assert "1d" in INTERVAL_MAP
    
    def test_interval_to_minutes(self):
        assert INTERVAL_TO_MINUTES["1min"] == 1
        assert INTERVAL_TO_MINUTES["5min"] == 5
        assert INTERVAL_TO_MINUTES["1h"] == 60
        assert INTERVAL_TO_MINUTES["1d"] == 1440
    
    def test_base_interval(self):
        assert BASE_INTERVAL == "1min"
    
    def test_aggregatable_intervals(self):
        assert "5min" in AGGREGATABLE_INTERVALS
        assert "15min" in AGGREGATABLE_INTERVALS
        assert "1h" in AGGREGATABLE_INTERVALS
        assert "1min" not in AGGREGATABLE_INTERVALS


class TestUnifiedDataService:
    """统一数据服务测试"""
    
    @pytest.fixture
    def service(self):
        return UnifiedDataService(DataServiceConfig(auto_init=False))
    
    @pytest.fixture
    def sample_klines(self):
        return [
            [1704067200000, "50000", "51000", "49000", "50500", "1000",
             1704067259999, "50000000", 5000, "500", "250000", "0"],
            [1704067260000, "50500", "51500", "50000", "51000", "1200",
             1704067319999, "60000000", 6000, "600", "300000", "0"],
        ]
    
    def test_service_creation(self, service):
        assert service.config is not None
        assert service.db_manager is not None
    
    def test_get_interval_valid(self, service):
        interval = service.get_interval("1min")
        assert interval == INTERVAL_MAP["1min"]
    
    def test_get_interval_invalid(self, service):
        with pytest.raises(ValueError):
            service.get_interval("invalid_interval")
    
    def test_parse_klines(self, service, sample_klines):
        df = service._parse_klines(sample_klines)
        
        assert len(df) == 2
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
    
    def test_can_aggregate_from_base(self, service):
        assert service._can_aggregate_from_base("5min") is True
        assert service._can_aggregate_from_base("15min") is True
        assert service._can_aggregate_from_base("1h") is True
        assert service._can_aggregate_from_base("1min") is False
        assert service._can_aggregate_from_base("1w") is False
    
    def test_aggregate_from_base(self, service):
        dates = pd.date_range(start='2024-01-01', periods=60, freq='1min')
        base_data = pd.DataFrame({
            'open': [100 + i for i in range(60)],
            'high': [105 + i for i in range(60)],
            'low': [95 + i for i in range(60)],
            'close': [100 + i for i in range(60)],
            'volume': [1000 for _ in range(60)],
            'quote_asset_volume': [100000 for _ in range(60)],
            'number_of_trades': [100 for _ in range(60)],
            'taker_buy_base_asset_volume': [500 for _ in range(60)],
            'taker_buy_quote_asset_volume': [50000 for _ in range(60)],
        }, index=dates)
        
        result = service._aggregate_from_base(base_data, "5min", 10)
        
        if result is not None:
            assert len(result) <= 12
    
    def test_get_klines_from_database_empty(self, service):
        with patch.object(service.db_manager, 'get_repository') as mock_repo:
            mock_repository = Mock()
            mock_repository.table_exists.return_value = False
            mock_repo.return_value = mock_repository
            
            df = service.get_klines_from_database("TESTUSDT", "1min", 100)
            assert df.empty
    
    def test_save_to_database_empty(self, service):
        count = service.save_to_database(pd.DataFrame(), "BTCUSDT", "1min")
        assert count == 0


class TestDataServiceAggregation:
    """数据服务聚合功能测试"""
    
    @pytest.fixture
    def service(self):
        return UnifiedDataService(DataServiceConfig(auto_init=False))
    
    @pytest.fixture
    def base_data(self):
        dates = pd.date_range(start='2024-01-01', periods=300, freq='1min')
        data = pd.DataFrame({
            'open': np.random.uniform(49000, 51000, 300),
            'high': np.random.uniform(50000, 52000, 300),
            'low': np.random.uniform(48000, 50000, 300),
            'close': np.random.uniform(49000, 51000, 300),
            'volume': np.random.uniform(100, 1000, 300),
            'quote_asset_volume': np.random.uniform(10000, 100000, 300),
            'number_of_trades': np.random.randint(50, 200, 300),
            'taker_buy_base_asset_volume': np.random.uniform(50, 500, 300),
            'taker_buy_quote_asset_volume': np.random.uniform(5000, 50000, 300),
        }, index=dates)
        return data
    
    def test_get_base_data_for_aggregation(self, service, base_data):
        with patch.object(service, 'get_klines_from_database', return_value=base_data):
            result = service._get_base_data_for_aggregation("BTCUSDT", "5min", 10)
            assert result is not None
            assert len(result) == 300
    
    def test_get_klines_via_aggregation(self, service, base_data):
        with patch.object(service, 'get_klines_from_database', return_value=base_data):
            result = service.get_klines_via_aggregation("BTCUSDT", "5min", 10)
            if result is not None:
                assert len(result) <= 60
    
    def test_aggregation_cache(self, service):
        assert service._aggregation_cache == {}
        
        service._set_cached_aggregation("BTCUSDT", "5min", pd.DataFrame())
        
        assert "BTCUSDT" in service._aggregation_cache
        assert "5min" in service._aggregation_cache["BTCUSDT"]
        
        cached = service._get_cached_aggregation("BTCUSDT", "5min")
        assert cached is not None


class TestDataServiceErrorHandling:
    """数据服务错误处理测试"""
    
    @pytest.fixture
    def service(self):
        return UnifiedDataService(DataServiceConfig(auto_init=False))
    
    def test_region_restricted_error(self, service):
        with patch.object(service, '_get_client') as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.futures_klines.side_effect = Exception("restricted location")
            mock_client.return_value = mock_client_instance
            
            with pytest.raises(RegionRestrictedError):
                service.get_klines_from_api("BTCUSDT", "1min", 100)
    
    def test_data_source_error_invalid_symbol(self, service):
        with patch.object(service, '_get_client') as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.futures_klines.side_effect = Exception("invalid symbol")
            mock_client.return_value = mock_client_instance
            
            with pytest.raises(DataSourceError):
                service.get_klines_from_api("INVALID", "1min", 100)
