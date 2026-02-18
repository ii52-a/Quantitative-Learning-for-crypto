"""
常量定义模块
"""

from enum import Enum, auto


class Interval(Enum):
    """K线周期"""
    MIN_1 = "1min"
    MIN_3 = "3min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_30 = "30min"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"
    
    @classmethod
    def from_string(cls, value: str) -> "Interval":
        """从字符串转换"""
        mapping = {
            "1min": cls.MIN_1, "1m": cls.MIN_1,
            "3min": cls.MIN_3, "3m": cls.MIN_3,
            "5min": cls.MIN_5, "5m": cls.MIN_5,
            "15min": cls.MIN_15, "15m": cls.MIN_15,
            "30min": cls.MIN_30, "30m": cls.MIN_30,
            "1h": cls.HOUR_1,
            "2h": cls.HOUR_2,
            "4h": cls.HOUR_4,
            "6h": cls.HOUR_6,
            "12h": cls.HOUR_12,
            "1d": cls.DAY_1,
            "1w": cls.WEEK_1,
            "1M": cls.MONTH_1,
        }
        if value not in mapping:
            raise ValueError(f"不支持的K线周期: {value}")
        return mapping[value]
    
    def to_minutes(self) -> int:
        """转换为分钟数"""
        minutes_map = {
            Interval.MIN_1: 1,
            Interval.MIN_3: 3,
            Interval.MIN_5: 5,
            Interval.MIN_15: 15,
            Interval.MIN_30: 30,
            Interval.HOUR_1: 60,
            Interval.HOUR_2: 120,
            Interval.HOUR_4: 240,
            Interval.HOUR_6: 360,
            Interval.HOUR_12: 720,
            Interval.DAY_1: 1440,
            Interval.WEEK_1: 10080,
            Interval.MONTH_1: 43200,
        }
        return minutes_map[self]


class OrderSide(Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"
    
    def opposite(self) -> "OrderSide":
        """获取相反方向"""
        return OrderSide.SELL if self == OrderSide.BUY else OrderSide.BUY


class OrderType(Enum):
    """订单类型"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"


class PositionSide(Enum):
    """持仓方向"""
    LONG = "LONG"
    SHORT = "SHORT"
    EMPTY = "EMPTY"
    
    def opposite(self) -> "PositionSide":
        """获取相反方向"""
        if self == PositionSide.LONG:
            return PositionSide.SHORT
        elif self == PositionSide.SHORT:
            return PositionSide.LONG
        return PositionSide.EMPTY


class SignalType(Enum):
    """信号类型"""
    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    HOLD = "HOLD"
    
    def is_open(self) -> bool:
        """是否为开仓信号"""
        return self in (SignalType.OPEN_LONG, SignalType.OPEN_SHORT)
    
    def is_close(self) -> bool:
        """是否为平仓信号"""
        return self in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT)
    
    def is_long(self) -> bool:
        """是否为多头信号"""
        return self in (SignalType.OPEN_LONG, SignalType.CLOSE_LONG)


class StrategyType(Enum):
    """策略类型"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    CUSTOM = "custom"


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"
