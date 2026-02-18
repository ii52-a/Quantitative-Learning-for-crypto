"""
自定义异常模块
"""


class QuantitativeError(Exception):
    """量化交易系统基础异常"""
    
    def __init__(self, message: str, code: str = "UNKNOWN"):
        self.message = message
        self.code = code
        super().__init__(self.message)
    
    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class DataError(QuantitativeError):
    """数据相关异常"""
    
    def __init__(self, message: str, code: str = "DATA_ERROR"):
        super().__init__(message, code)


class DataSourceError(DataError):
    """数据源异常"""
    
    def __init__(self, message: str):
        super().__init__(message, "DATA_SOURCE_ERROR")


class DataValidationError(DataError):
    """数据验证异常"""
    
    def __init__(self, message: str):
        super().__init__(message, "DATA_VALIDATION_ERROR")


class RegionRestrictedError(DataError):
    """地区限制异常"""
    
    def __init__(self, message: str = "API访问受限，请配置代理或使用VPN"):
        super().__init__(message, "REGION_RESTRICTED")


class StrategyError(QuantitativeError):
    """策略相关异常"""
    
    def __init__(self, message: str, code: str = "STRATEGY_ERROR"):
        super().__init__(message, code)


class StrategyConfigError(StrategyError):
    """策略配置异常"""
    
    def __init__(self, message: str):
        super().__init__(message, "STRATEGY_CONFIG_ERROR")


class StrategyExecutionError(StrategyError):
    """策略执行异常"""
    
    def __init__(self, message: str):
        super().__init__(message, "STRATEGY_EXECUTION_ERROR")


class BacktestError(QuantitativeError):
    """回测相关异常"""
    
    def __init__(self, message: str, code: str = "BACKTEST_ERROR"):
        super().__init__(message, code)


class BacktestConfigError(BacktestError):
    """回测配置异常"""
    
    def __init__(self, message: str):
        super().__init__(message, "BACKTEST_CONFIG_ERROR")


class BacktestDataError(BacktestError):
    """回测数据异常"""
    
    def __init__(self, message: str):
        super().__init__(message, "BACKTEST_DATA_ERROR")


class TradingError(QuantitativeError):
    """交易相关异常"""
    
    def __init__(self, message: str, code: str = "TRADING_ERROR"):
        super().__init__(message, code)


class OrderError(TradingError):
    """订单异常"""
    
    def __init__(self, message: str):
        super().__init__(message, "ORDER_ERROR")


class PositionError(TradingError):
    """仓位异常"""
    
    def __init__(self, message: str):
        super().__init__(message, "POSITION_ERROR")


class RiskLimitError(TradingError):
    """风控限制异常"""
    
    def __init__(self, message: str):
        super().__init__(message, "RISK_LIMIT_ERROR")
