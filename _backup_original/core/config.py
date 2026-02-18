"""
统一配置模块
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Config:
    """全局配置"""
    
    project_name: str = "Quantitative Trading System"
    version: str = "2.0.0"
    debug: bool = False
    
    log_level: str = "INFO"
    log_file: str = "quantitative_trading.log"
    
    data_dir: str = "data"
    cache_dir: str = "cache"
    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量加载配置"""
        import os
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


@dataclass
class BacktestConfig:
    """回测配置"""
    
    symbol: str = "BTCUSDT"
    interval: str = "30min"
    start_date: datetime | None = None
    end_date: datetime | None = None
    
    initial_capital: float = 10000.0
    commission_rate: float = 0.0004
    slippage: float = 0.0001
    
    leverage: int = 5
    position_size: float = 0.1
    
    data_limit: int = 1000
    
    def validate(self) -> list[str]:
        """验证配置"""
        errors = []
        if self.initial_capital <= 0:
            errors.append("初始资金必须大于0")
        if self.commission_rate < 0:
            errors.append("手续费率不能为负")
        if self.leverage < 1:
            errors.append("杠杆必须大于等于1")
        return errors


@dataclass
class TradingConfig:
    """交易配置"""
    
    symbol: str = "BTCUSDT"
    interval: str = "30min"
    
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    dry_run: bool = True
    
    risk_per_trade: float = 0.02
    max_position_ratio: float = 0.3
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    
    def validate(self) -> list[str]:
        """验证配置"""
        errors = []
        if not self.api_key:
            errors.append("API Key未配置")
        if not self.api_secret:
            errors.append("API Secret未配置")
        if self.risk_per_trade <= 0 or self.risk_per_trade > 1:
            errors.append("单笔风险比例必须在0-1之间")
        return errors
