"""
技术指标模块

提供常用技术指标计算
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass
class IndicatorResult:
    """指标计算结果"""
    values: pd.Series | pd.DataFrame
    name: str
    params: dict[str, Any]


class MACD:
    """MACD指标"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(self, prices: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标
        
        返回: (macd, signal, hist)
        """
        fast_ema = prices.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=self.slow_period, adjust=False).mean()
        
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        hist = macd - signal
        
        return macd, signal, hist
    
    def get_latest(self, prices: pd.Series) -> dict[str, float]:
        """获取最新指标值"""
        macd, signal, hist = self.calculate(prices)
        
        return {
            "macd": float(macd.iloc[-1]),
            "signal": float(signal.iloc[-1]),
            "hist": float(hist.iloc[-1]),
            "hist_prev": float(hist.iloc[-2]) if len(hist) > 1 else 0.0,
        }
    
    def detect_crossover(self, prices: pd.Series) -> str:
        """检测交叉信号
        
        返回: 'golden_cross', 'dead_cross', 'none'
        """
        macd, signal, hist = self.calculate(prices)
        
        if len(hist) < 2:
            return "none"
        
        hist_prev = hist.iloc[-2]
        hist_curr = hist.iloc[-1]
        
        if hist_prev < 0 < hist_curr:
            return "golden_cross"
        elif hist_prev > 0 > hist_curr:
            return "dead_cross"
        
        return "none"


class RSI:
    """RSI指标"""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=self.period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_latest(self, prices: pd.Series) -> dict[str, float]:
        """获取最新RSI值"""
        rsi = self.calculate(prices)
        
        return {
            "rsi": float(rsi.iloc[-1]),
            "rsi_prev": float(rsi.iloc[-2]) if len(rsi) > 1 else 50.0,
        }
    
    def detect_signal(self, prices: pd.Series, oversold: float = 30, overbought: float = 70) -> str:
        """检测RSI信号"""
        rsi = self.calculate(prices)
        
        if len(rsi) < 2:
            return "none"
        
        rsi_curr = rsi.iloc[-1]
        rsi_prev = rsi.iloc[-2]
        
        if rsi_prev < oversold < rsi_curr:
            return "oversold_bounce"
        elif rsi_prev > overbought > rsi_curr:
            return "overbought_decline"
        
        return "none"


class BollingerBands:
    """布林带指标"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
    
    def calculate(self, prices: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带
        
        返回: (upper, middle, lower)
        """
        middle = prices.rolling(window=self.period).mean()
        std = prices.rolling(window=self.period).std()
        
        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std
        
        return upper, middle, lower
    
    def get_latest(self, prices: pd.Series) -> dict[str, float]:
        """获取最新布林带值"""
        upper, middle, lower = self.calculate(prices)
        price = prices.iloc[-1]
        
        return {
            "upper": float(upper.iloc[-1]),
            "middle": float(middle.iloc[-1]),
            "lower": float(lower.iloc[-1]),
            "price": float(price),
            "position": self._get_position(price, upper.iloc[-1], lower.iloc[-1]),
        }
    
    def _get_position(self, price: float, upper: float, lower: float) -> float:
        """计算价格在布林带中的位置（0-1）"""
        if upper == lower:
            return 0.5
        return (price - lower) / (upper - lower)
    
    def detect_signal(self, prices: pd.Series) -> str:
        """检测布林带信号"""
        upper, middle, lower = self.calculate(prices)
        
        if len(prices) < 2:
            return "none"
        
        price_curr = prices.iloc[-1]
        price_prev = prices.iloc[-2]
        lower_curr = lower.iloc[-1]
        upper_curr = upper.iloc[-1]
        
        if price_prev < lower_curr < price_curr:
            return "lower_band_break"
        elif price_prev > upper_curr > price_curr:
            return "upper_band_break"
        
        return "none"


class MovingAverage:
    """移动平均线"""
    
    def __init__(self, period: int = 20, ma_type: str = "EMA"):
        self.period = period
        self.ma_type = ma_type.upper()
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """计算移动平均线"""
        if self.ma_type == "SMA":
            return prices.rolling(window=self.period).mean()
        elif self.ma_type == "EMA":
            return prices.ewm(span=self.period, adjust=False).mean()
        elif self.ma_type == "WMA":
            weights = np.arange(1, self.period + 1)
            return prices.rolling(window=self.period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        else:
            return prices.rolling(window=self.period).mean()
    
    def get_latest(self, prices: pd.Series) -> dict[str, float]:
        """获取最新均线值"""
        ma = self.calculate(prices)
        
        return {
            "ma": float(ma.iloc[-1]),
            "price": float(prices.iloc[-1]),
            "deviation": float((prices.iloc[-1] - ma.iloc[-1]) / ma.iloc[-1] * 100),
        }


class ATR:
    """平均真实波幅"""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """计算ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        
        return atr
    
    def get_latest(self, high: pd.Series, low: pd.Series, close: pd.Series) -> dict[str, float]:
        """获取最新ATR值"""
        atr = self.calculate(high, low, close)
        
        return {
            "atr": float(atr.iloc[-1]),
            "atr_pct": float(atr.iloc[-1] / close.iloc[-1] * 100),
        }
