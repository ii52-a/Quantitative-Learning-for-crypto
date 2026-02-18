"""
绩效指标计算模块

提供专业的回测绩效指标计算
"""

import numpy as np
import pandas as pd
from typing import Any


class PerformanceMetrics:
    """绩效指标计算器"""
    
    @staticmethod
    def total_return(equity_curve: pd.Series) -> float:
        """总收益率"""
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    
    @staticmethod
    def annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
        """年化收益率"""
        if len(equity_curve) < 2:
            return 0.0
        
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        n_periods = len(equity_curve)
        
        if total_return < 0:
            return -((1 + abs(total_return)) ** (periods_per_year / n_periods) - 1) * 100
        
        return ((1 + total_return) ** (periods_per_year / n_periods) - 1) * 100
    
    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> tuple[float, float]:
        """最大回撤
        
        返回: (最大回撤金额, 最大回撤百分比)
        """
        if len(equity_curve) < 2:
            return 0.0, 0.0
        
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        max_dd_pct = abs(drawdown.min()) * 100
        min_idx = drawdown.idxmin()
        if isinstance(min_idx, int):
            max_dd_amount = rolling_max.iloc[min_idx] * max_dd_pct / 100
        else:
            max_dd_amount = rolling_max.loc[min_idx] * max_dd_pct / 100
        
        return max_dd_amount, max_dd_pct
    
    @staticmethod
    def max_drawdown_duration(equity_curve: pd.Series) -> int:
        """最大回撤持续时间（周期数）"""
        if len(equity_curve) < 2:
            return 0
        
        rolling_max = equity_curve.cummax()
        drawdown = equity_curve < rolling_max
        
        max_duration = 0
        current_duration = 0
        
        for is_drawdown in drawdown:
            if is_drawdown:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
        """夏普比率"""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        rf_per_period = risk_free_rate / periods_per_year
        excess_returns = returns - rf_per_period
        
        return (excess_returns.mean() * periods_per_year) / (returns.std() * np.sqrt(periods_per_year))
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
        """索提诺比率"""
        if len(returns) < 2:
            return 0.0
        
        rf_per_period = risk_free_rate / periods_per_year
        excess_returns = returns - rf_per_period
        
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        return (excess_returns.mean() * periods_per_year) / (downside_std * np.sqrt(periods_per_year))
    
    @staticmethod
    def calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
        """卡玛比率"""
        ann_return = PerformanceMetrics.annualized_return(equity_curve, periods_per_year)
        _, max_dd_pct = PerformanceMetrics.max_drawdown(equity_curve)
        
        if max_dd_pct == 0:
            return float('inf') if ann_return > 0 else 0.0
        
        return ann_return / max_dd_pct
    
    @staticmethod
    def win_rate(trades: list[dict]) -> float:
        """胜率"""
        if not trades:
            return 0.0
        
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        total = len([t for t in trades if t.get('pnl') is not None])
        
        return (wins / total * 100) if total > 0 else 0.0
    
    @staticmethod
    def profit_factor(trades: list[dict]) -> float:
        """盈亏比"""
        if not trades:
            return 0.0
        
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = sum(abs(t.get('pnl', 0)) for t in trades if t.get('pnl', 0) < 0)
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def avg_trade(trades: list[dict]) -> tuple[float, float]:
        """平均盈利/亏损
        
        返回: (平均盈利, 平均亏损)
        """
        if not trades:
            return 0.0, 0.0
        
        wins = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
        losses = [abs(t.get('pnl', 0)) for t in trades if t.get('pnl', 0) < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        return avg_win, avg_loss
    
    @staticmethod
    def expectancy(trades: list[dict]) -> float:
        """期望值"""
        if not trades:
            return 0.0
        
        win_rate = PerformanceMetrics.win_rate(trades) / 100
        avg_win, avg_loss = PerformanceMetrics.avg_trade(trades)
        
        if avg_loss == 0:
            return avg_win * win_rate
        
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    @staticmethod
    def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """年化波动率"""
        if len(returns) < 2:
            return 0.0
        
        return returns.std() * np.sqrt(periods_per_year) * 100
    
    @staticmethod
    def var(returns: pd.Series, confidence: float = 0.95) -> float:
        """风险价值"""
        if len(returns) < 2:
            return 0.0
        
        return np.percentile(returns, (1 - confidence) * 100) * 100
    
    @staticmethod
    def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """条件风险价值"""
        if len(returns) < 2:
            return 0.0
        
        var = PerformanceMetrics.var(returns, confidence)
        return returns[returns <= var / 100].mean() * 100
    
    @staticmethod
    def calculate_all(equity_curve: pd.Series, trades: list[dict], risk_free_rate: float = 0.02) -> dict[str, Any]:
        """计算所有绩效指标"""
        returns = equity_curve.pct_change().dropna() if len(equity_curve) > 1 else pd.Series()
        
        max_dd_amount, max_dd_pct = PerformanceMetrics.max_drawdown(equity_curve)
        avg_win, avg_loss = PerformanceMetrics.avg_trade(trades)
        
        return {
            "total_return": PerformanceMetrics.total_return(equity_curve),
            "annualized_return": PerformanceMetrics.annualized_return(equity_curve),
            "max_drawdown": max_dd_amount,
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_duration": PerformanceMetrics.max_drawdown_duration(equity_curve),
            "sharpe_ratio": PerformanceMetrics.sharpe_ratio(returns, risk_free_rate),
            "sortino_ratio": PerformanceMetrics.sortino_ratio(returns, risk_free_rate),
            "calmar_ratio": PerformanceMetrics.calmar_ratio(equity_curve),
            "win_rate": PerformanceMetrics.win_rate(trades),
            "profit_factor": PerformanceMetrics.profit_factor(trades),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": PerformanceMetrics.expectancy(trades),
            "volatility": PerformanceMetrics.volatility(returns),
            "var_95": PerformanceMetrics.var(returns, 0.95),
            "cvar_95": PerformanceMetrics.cvar(returns, 0.95),
            "total_trades": len([t for t in trades if t.get('pnl') is not None]),
            "winning_trades": len([t for t in trades if t.get('pnl', 0) > 0]),
            "losing_trades": len([t for t in trades if t.get('pnl', 0) < 0]),
        }
