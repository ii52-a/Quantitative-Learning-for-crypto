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
class RiskConfig:
    """风险控制配置"""
    
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    leverage: int = 1
    
    MIN_LEVERAGE: int = 1
    MAX_LEVERAGE: int = 125
    MIN_STOP_LOSS: float = 0.0
    MAX_STOP_LOSS: float = 100.0
    MIN_TAKE_PROFIT: float = 0.0
    MAX_TAKE_PROFIT: float = 1000.0
    
    def __post_init__(self):
        self.stop_loss_pct = round(self.stop_loss_pct, 2)
        self.take_profit_pct = round(self.take_profit_pct, 2)
    
    def validate(self) -> list[str]:
        """验证配置"""
        errors = []
        
        if not self.MIN_LEVERAGE <= self.leverage <= self.MAX_LEVERAGE:
            errors.append(f"杠杆倍数必须在 {self.MIN_LEVERAGE}-{self.MAX_LEVERAGE} 之间")
        
        if not self.MIN_STOP_LOSS <= self.stop_loss_pct <= self.MAX_STOP_LOSS:
            errors.append(f"止损率必须在 {self.MIN_STOP_LOSS}-{self.MAX_STOP_LOSS}% 之间")
        
        if not self.MIN_TAKE_PROFIT <= self.take_profit_pct <= self.MAX_TAKE_PROFIT:
            errors.append(f"止盈率必须在 {self.MIN_TAKE_PROFIT}-{self.MAX_TAKE_PROFIT}% 之间")
        
        return errors
    
    def get_risk_warning(self) -> str | None:
        """获取风险提示"""
        warnings = []
        
        if self.leverage > 20:
            warnings.append(f"⚠️ 高杠杆警告：当前杠杆 {self.leverage}x，风险极高！")
        elif self.leverage > 10:
            warnings.append(f"⚡ 中高杠杆提示：当前杠杆 {self.leverage}x，请谨慎操作。")
        
        if self.stop_loss_pct == 0 and self.leverage > 1:
            warnings.append("⚠️ 风险警告：使用杠杆但未设置止损，可能导致爆仓！")
        
        if self.take_profit_pct == 0 and self.leverage > 1:
            warnings.append("💡 提示：未设置止盈，建议设置合理的止盈目标。")
        
        if self.stop_loss_pct > 50:
            warnings.append(f"⚠️ 止损设置过宽：{self.stop_loss_pct}%，可能导致较大亏损。")
        
        return "\n".join(warnings) if warnings else None
    
    def calculate_stop_loss_price(self, entry_price: float, is_long: bool = True) -> float | None:
        """计算止损价格"""
        if self.stop_loss_pct <= 0:
            return None
        
        if is_long:
            return entry_price * (1 - self.stop_loss_pct / 100)
        else:
            return entry_price * (1 + self.stop_loss_pct / 100)
    
    def calculate_take_profit_price(self, entry_price: float, is_long: bool = True) -> float | None:
        """计算止盈价格"""
        if self.take_profit_pct <= 0:
            return None
        
        if is_long:
            return entry_price * (1 + self.take_profit_pct / 100)
        else:
            return entry_price * (1 - self.take_profit_pct / 100)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "leverage": self.leverage,
        }


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
    
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    
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
        if self.leverage > 125:
            errors.append("杠杆不能超过125倍")
        if self.stop_loss_pct < 0 or self.stop_loss_pct > 100:
            errors.append("止损率必须在0-100%之间")
        if self.take_profit_pct < 0 or self.take_profit_pct > 1000:
            errors.append("止盈率必须在0-1000%之间")
        return errors
    
    def get_risk_config(self) -> RiskConfig:
        """获取风险配置"""
        return RiskConfig(
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            leverage=self.leverage,
        )


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
    
    leverage: int = 5
    
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
    
    def get_risk_config(self) -> RiskConfig:
        """获取风险配置"""
        return RiskConfig(
            stop_loss_pct=self.stop_loss_pct * 100,
            take_profit_pct=self.take_profit_pct * 100,
            leverage=self.leverage,
        )
