"""
回测引擎模块

提供完整的回测功能
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from core.config import BacktestConfig
from core.constants import SignalType, PositionSide
from strategy.base import BaseStrategy, Bar, Signal, Position, StrategyContext


@dataclass
class Trade:
    """交易记录"""
    trade_id: int
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    pnl: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    """回测结果"""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    start_time: datetime | None = None
    end_time: datetime | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "start_time": str(self.start_time) if self.start_time else None,
            "end_time": str(self.end_time) if self.end_time else None,
        }


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, strategy: BaseStrategy, config: BacktestConfig):
        self.strategy = strategy
        self.config = config
        
        self._capital: float = config.initial_capital
        self._position: Position = Position(
            side=PositionSide.EMPTY,
            quantity=0.0,
            entry_price=0.0,
            entry_time=datetime.now(),
        )
        self._trades: list[Trade] = []
        self._equity_history: list[tuple[datetime, float]] = []
        self._trade_id: int = 0
        
        self._initialized = False
    
    def run(self, data: pd.DataFrame) -> BacktestResult:
        """运行回测"""
        if data.empty:
            return BacktestResult(initial_capital=self.config.initial_capital)
        
        context = StrategyContext(
            symbol=self.config.symbol,
            interval=self.config.interval,
            position=self._position,
            equity=self._capital,
            available_capital=self._capital,
            current_price=0.0,
            timestamp=datetime.now(),
        )
        
        self.strategy.initialize(context)
        self._initialized = True
        
        for idx, row in data.iterrows():
            timestamp = idx if isinstance(idx, datetime) else pd.to_datetime(idx, utc=True)
            
            bar = Bar(
                timestamp=timestamp,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
                symbol=self.config.symbol,
                interval=self.config.interval,
            )
            
            context.current_price = bar.close
            context.timestamp = timestamp
            context.position = self._position
            context.equity = self._calculate_equity(bar.close)
            
            result = self.strategy.on_bar(bar, context)
            
            if result.signal and result.signal.is_valid():
                self._process_signal(result.signal, bar, timestamp)
            
            self._equity_history.append((timestamp, context.equity))
        
        return self._generate_result()
    
    def _process_signal(self, signal: Signal, bar: Bar, timestamp: datetime) -> None:
        """处理交易信号"""
        if signal.type == SignalType.OPEN_LONG:
            if self._position.is_open:
                return
            
            quantity = self._calculate_position_size(bar.close)
            if quantity <= 0:
                return
            
            commission = quantity * bar.close * self.config.commission_rate
            
            self._position = Position(
                side=PositionSide.LONG,
                quantity=quantity,
                entry_price=bar.close,
                entry_time=timestamp,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
            
            self._capital -= commission
            
            self._trade_id += 1
            trade = Trade(
                trade_id=self._trade_id,
                timestamp=timestamp,
                symbol=self.config.symbol,
                side="OPEN_LONG",
                quantity=quantity,
                price=bar.close,
                commission=commission,
                reason=signal.reason,
            )
            self._trades.append(trade)
        
        elif signal.type == SignalType.CLOSE_LONG:
            if not self._position.is_open or self._position.side != PositionSide.LONG:
                return
            
            quantity = self._position.quantity
            entry_price = self._position.entry_price
            
            pnl = (bar.close - entry_price) * quantity
            commission = quantity * bar.close * self.config.commission_rate
            pnl -= commission
            
            self._capital += pnl
            
            self._trade_id += 1
            trade = Trade(
                trade_id=self._trade_id,
                timestamp=timestamp,
                symbol=self.config.symbol,
                side="CLOSE_LONG",
                quantity=quantity,
                price=bar.close,
                commission=commission,
                pnl=pnl,
                reason=signal.reason,
            )
            self._trades.append(trade)
            
            self._position = Position(
                side=PositionSide.EMPTY,
                quantity=0.0,
                entry_price=0.0,
                entry_time=timestamp,
            )
    
    def _calculate_position_size(self, price: float) -> float:
        """计算仓位大小"""
        position_value = self._capital * self.config.position_size * self.config.leverage
        return position_value / price if price > 0 else 0.0
    
    def _calculate_equity(self, current_price: float) -> float:
        """计算当前权益"""
        equity = self._capital
        
        if self._position.is_open:
            if self._position.side == PositionSide.LONG:
                equity += (current_price - self._position.entry_price) * self._position.quantity
        
        return equity
    
    def _generate_result(self) -> BacktestResult:
        """生成回测结果"""
        result = BacktestResult(
            trades=self._trades,
            initial_capital=self.config.initial_capital,
            final_capital=self._calculate_equity(
                self._equity_history[-1][1] if self._equity_history else self._capital
            ),
        )
        
        if self._equity_history:
            timestamps = [e[0] for e in self._equity_history]
            equities = [e[1] for e in self._equity_history]
            result.equity_curve = pd.Series(equities, index=timestamps)
            
            result.start_time = timestamps[0]
            result.end_time = timestamps[-1]
        
        result.total_return = result.final_capital - result.initial_capital
        result.total_return_pct = (result.final_capital / result.initial_capital - 1) * 100
        
        if len(result.equity_curve) > 1:
            result.returns = result.equity_curve.pct_change().dropna()
            
            rolling_max = result.equity_curve.cummax()
            result.drawdown_curve = (result.equity_curve - rolling_max) / rolling_max
            result.max_drawdown_pct = abs(result.drawdown_curve.min()) * 100
            result.max_drawdown = result.initial_capital * result.max_drawdown_pct / 100
        
        result.total_trades = len([t for t in self._trades if t.side.startswith("CLOSE")])
        
        closed_trades = [t for t in self._trades if t.pnl != 0]
        result.winning_trades = len([t for t in closed_trades if t.pnl > 0])
        result.losing_trades = len([t for t in closed_trades if t.pnl < 0])
        result.win_rate = result.winning_trades / result.total_trades * 100 if result.total_trades > 0 else 0
        
        wins = [t.pnl for t in closed_trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in closed_trades if t.pnl < 0]
        
        result.avg_win = sum(wins) / len(wins) if wins else 0
        result.avg_loss = sum(losses) / len(losses) if losses else 0
        result.profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else float('inf') if sum(wins) > 0 else 0
        
        if len(result.returns) > 0:
            risk_free_rate = 0.02 / 252
            excess_returns = result.returns - risk_free_rate
            
            if result.returns.std() > 0:
                result.sharpe_ratio = (result.returns.mean() * 252) / (result.returns.std() * (252 ** 0.5))
            
            downside_returns = result.returns[result.returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                result.sortino_ratio = (result.returns.mean() * 252) / (downside_returns.std() * (252 ** 0.5))
            
            if result.max_drawdown_pct > 0:
                result.calmar_ratio = result.total_return_pct / result.max_drawdown_pct
        
        return result
