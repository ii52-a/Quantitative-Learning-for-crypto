"""
策略基类模块

定义策略的标准接口和数据结构
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from core.constants import SignalType, PositionSide


@dataclass
class Bar:
    """K线数据"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    interval: str = ""
    
    @property
    def typical_price(self) -> float:
        """典型价格"""
        return (self.high + self.low + self.close) / 3
    
    @property
    def range(self) -> float:
        """K线振幅"""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """实体大小"""
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        """是否为阳线"""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """是否为阴线"""
        return self.close < self.open


@dataclass
class Tick:
    """Tick数据"""
    timestamp: datetime
    price: float
    volume: float
    symbol: str = ""
    bid: float = 0.0
    ask: float = 0.0
    
    @property
    def spread(self) -> float:
        """买卖价差"""
        return self.ask - self.bid if self.bid and self.ask else 0.0


@dataclass
class Signal:
    """交易信号"""
    type: SignalType
    price: float
    quantity: float = 0.0
    timestamp: datetime | None = None
    reason: str = ""
    stop_loss: float | None = None
    take_profit: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """信号是否有效"""
        return self.type != SignalType.HOLD and self.price > 0


@dataclass
class Position:
    """持仓信息"""
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: float | None = None
    take_profit: float | None = None
    unrealized_pnl: float = 0.0
    
    @property
    def is_open(self) -> bool:
        """是否有持仓"""
        return self.side != PositionSide.EMPTY and self.quantity > 0
    
    def update_pnl(self, current_price: float) -> float:
        """更新未实现盈亏"""
        if not self.is_open:
            self.unrealized_pnl = 0.0
            return 0.0
        
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
        
        return self.unrealized_pnl


@dataclass
class StrategyContext:
    """策略上下文"""
    symbol: str
    interval: str
    position: Position
    equity: float
    available_capital: float
    current_price: float
    timestamp: datetime
    data_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_position(self) -> bool:
        """是否有持仓"""
        return self.position.is_open


@dataclass
class StrategyResult:
    """策略执行结果"""
    signal: Signal | None = None
    indicators: dict[str, Any] = field(default_factory=dict)
    log: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyParameter:
    """策略参数定义"""
    name: str
    display_name: str
    description: str
    value_type: type
    default_value: Any
    min_value: Any | None = None
    max_value: Any | None = None
    options: list[Any] | None = None
    
    def validate(self, value: Any) -> bool:
        """验证参数值"""
        if not isinstance(value, self.value_type):
            try:
                value = self.value_type(value)
            except (ValueError, TypeError):
                return False
        
        if self.min_value is not None and value < self.min_value:
            return False
        
        if self.max_value is not None and value > self.max_value:
            return False
        
        if self.options is not None and value not in self.options:
            return False
        
        return True


class BaseStrategy(ABC):
    """策略基类"""
    
    name: str = "BaseStrategy"
    display_name: str = "基础策略"
    description: str = "策略基类，所有策略必须继承此类"
    strategy_type: str = "custom"
    risk_level: str = "medium"
    
    parameters: list[StrategyParameter] = []
    
    def __init__(self, params: dict[str, Any] | None = None):
        self._params: dict[str, Any] = {}
        self._initialized = False
        
        if params:
            self.set_parameters(params)
        else:
            self._load_default_parameters()
    
    def _load_default_parameters(self) -> None:
        """加载默认参数"""
        for param in self.parameters:
            self._params[param.name] = param.default_value
    
    def get_parameters(self) -> dict[str, Any]:
        """获取当前参数"""
        return self._params.copy()
    
    def set_parameters(self, params: dict[str, Any]) -> list[str]:
        """设置参数，返回错误列表"""
        errors = []
        
        for param in self.parameters:
            if param.name in params:
                value = params[param.name]
                if param.validate(value):
                    self._params[param.name] = param.value_type(value)
                else:
                    errors.append(f"参数 {param.display_name} 值无效: {value}")
        
        return errors
    
    def get_parameter_definitions(self) -> list[dict[str, Any]]:
        """获取参数定义列表（用于UI显示）"""
        return [
            {
                "name": p.name,
                "display_name": p.display_name,
                "description": p.description,
                "type": p.value_type.__name__,
                "default": p.default_value,
                "min": p.min_value,
                "max": p.max_value,
                "options": p.options,
            }
            for p in self.parameters
        ]
    
    @abstractmethod
    def initialize(self, context: StrategyContext) -> None:
        """初始化策略
        
        在回测或实盘开始时调用一次
        用于设置指标、验证参数等
        """
        pass
    
    @abstractmethod
    def on_bar(self, bar: Bar, context: StrategyContext) -> StrategyResult:
        """K线事件处理
        
        每根K线收盘时调用
        返回交易信号和指标数据
        """
        pass
    
    def on_tick(self, tick: Tick, context: StrategyContext) -> StrategyResult:
        """Tick事件处理（可选实现）
        
        每个Tick更新时调用
        默认返回空结果
        """
        return StrategyResult()
    
    def on_order_filled(self, order: dict[str, Any], context: StrategyContext) -> None:
        """订单成交回调（可选实现）
        
        订单成交后调用
        用于更新策略内部状态
        """
        pass
    
    def on_stop(self, context: StrategyContext) -> None:
        """策略停止回调（可选实现）
        
        回测或实盘结束时调用
        用于清理资源、保存状态等
        """
        pass
    
    def get_required_data_count(self) -> int:
        """获取策略所需的最小数据量
        
        用于确保有足够的历史数据计算指标
        """
        return 50
    
    def get_info(self) -> dict[str, Any]:
        """获取策略信息"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "type": self.strategy_type,
            "risk_level": self.risk_level,
            "parameters": self.get_parameter_definitions(),
            "required_data_count": self.get_required_data_count(),
        }
