"""
回测引擎模块

提供完整的回测功能
支持止损止盈、杠杆设置
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from core.config import BacktestConfig, RiskConfig
from core.constants import SignalType, PositionSide
from Strategy.base import BaseStrategy, Bar, Signal, Position, StrategyContext


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
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class CompletedTrade:
    """完成的交易对"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float
    reason: str
    exit_type: str = "signal"
    leverage: int = 1
    side: str = "long"
    position_value: float = 0.0
    margin_used: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    trades: list[Trade] = field(default_factory=list)
    completed_trades: list[CompletedTrade] = field(default_factory=list)
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
    
    stop_loss_hits: int = 0
    take_profit_hits: int = 0
    liquidation_hits: int = 0
    leverage: int = 1
    
    symbol: str = ""
    interval: str = ""
    strategy_name: str = ""
    strategy_params: dict = field(default_factory=dict)
    commission_rate: float = 0.0004
    position_size: float = 0.1
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
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
            "stop_loss_hits": self.stop_loss_hits,
            "take_profit_hits": self.take_profit_hits,
            "liquidation_hits": self.liquidation_hits,
            "leverage": self.leverage,
            "symbol": self.symbol,
            "interval": self.interval,
            "strategy_name": self.strategy_name,
            "strategy_params": self.strategy_params,
            "commission_rate": self.commission_rate,
            "position_size": self.position_size,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
        }


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, strategy: BaseStrategy, config: BacktestConfig):
        self.strategy = strategy
        self.config = config
        self.risk_config = config.get_risk_config()
        
        self._capital: float = config.initial_capital
        self._initial_capital: float = config.initial_capital
        self._position: Position = Position(
            side=PositionSide.EMPTY,
            quantity=0.0,
            entry_price=0.0,
            entry_time=datetime.now(),
        )
        self._trades: list[Trade] = []
        self._completed_trades: list[CompletedTrade] = []
        self._equity_history: list[tuple[datetime, float]] = []
        self._trade_id: int = 0
        
        self._stop_loss_hits: int = 0
        self._take_profit_hits: int = 0
        self._liquidation_hits: int = 0
        
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
                volume=float(row['volume']) if 'volume' in row else 0.0,
                symbol=self.config.symbol,
                interval=self.config.interval,
            )
            
            context.current_price = bar.close
            context.timestamp = timestamp
            context.position = self._position
            context.equity = self._calculate_equity(bar.close)
            
            if self._position.is_open:
                sl_hit, tp_hit = self._check_stop_loss_take_profit(bar, timestamp)
                if sl_hit or tp_hit:
                    self._equity_history.append((timestamp, context.equity))
                    continue
                
                if self._check_liquidation(bar, timestamp):
                    self._equity_history.append((timestamp, 0))
                    continue
            
            result = self.strategy.on_bar(bar, context)
            
            if result.signal and result.signal.is_valid():
                self._process_signal(result.signal, bar, timestamp)
            
            self._equity_history.append((timestamp, context.equity))
        
        return self._generate_result()
    
    def _check_stop_loss_take_profit(self, bar: Bar, timestamp: datetime) -> tuple[bool, bool]:
        """检查止损止盈触发"""
        if not self._position.is_open:
            return False, False
        
        sl_hit = False
        tp_hit = False
        
        if self._position.side == PositionSide.LONG:
            if self._position.stop_loss and bar.low <= self._position.stop_loss:
                sl_hit = True
                self._close_position(
                    price=self._position.stop_loss,
                    timestamp=timestamp,
                    reason="止损触发",
                    exit_type="stop_loss"
                )
                self._stop_loss_hits += 1
            
            elif self._position.take_profit and bar.high >= self._position.take_profit:
                tp_hit = True
                self._close_position(
                    price=self._position.take_profit,
                    timestamp=timestamp,
                    reason="止盈触发",
                    exit_type="take_profit"
                )
                self._take_profit_hits += 1
        
        elif self._position.side == PositionSide.SHORT:
            if self._position.stop_loss and bar.high >= self._position.stop_loss:
                sl_hit = True
                self._close_position(
                    price=self._position.stop_loss,
                    timestamp=timestamp,
                    reason="止损触发",
                    exit_type="stop_loss"
                )
                self._stop_loss_hits += 1
            
            elif self._position.take_profit and bar.low <= self._position.take_profit:
                tp_hit = True
                self._close_position(
                    price=self._position.take_profit,
                    timestamp=timestamp,
                    reason="止盈触发",
                    exit_type="take_profit"
                )
                self._take_profit_hits += 1
        
        return sl_hit, tp_hit
    
    def _check_liquidation(self, bar: Bar, timestamp: datetime) -> bool:
        """检查爆仓
        
        爆仓条件：
        - 当权益 <= 0 时爆仓
        - 权益 = 资金 + 未实现盈亏
        """
        if not self._position.is_open:
            return False
        
        equity = self._calculate_equity(bar.close)
        
        if equity <= 0:
            entry_price = self._position.entry_price
            quantity = self._position.quantity
            leverage = self.config.leverage
            
            if self._position.side == PositionSide.LONG:
                max_loss_pct = 1.0 / leverage
                liquidation_price = entry_price * (1 - max_loss_pct)
            else:
                max_loss_pct = 1.0 / leverage
                liquidation_price = entry_price * (1 + max_loss_pct)
            
            entry_price_val = self._position.entry_price
            entry_time = self._position.entry_time
            position_side = self._position.side
            quantity = self._position.quantity
            
            exit_commission = quantity * liquidation_price * self.config.commission_rate
            
            self._capital = 0
            
            close_side = "CLOSE_LONG" if position_side == PositionSide.LONG else "CLOSE_SHORT"
            
            self._trade_id += 1
            trade = Trade(
                trade_id=self._trade_id,
                timestamp=timestamp,
                symbol=self.config.symbol,
                side=close_side,
                quantity=quantity,
                price=liquidation_price,
                commission=exit_commission,
                pnl=-self._initial_capital,
                reason=f"爆仓 (权益归零)",
            )
            self._trades.append(trade)
            
            entry_commission = quantity * entry_price_val * self.config.commission_rate
            total_commission = entry_commission + exit_commission
            
            position_value = quantity * entry_price_val
            margin_used = position_value / self.config.leverage
            
            completed = CompletedTrade(
                entry_time=entry_time,
                exit_time=timestamp,
                entry_price=entry_price_val,
                exit_price=liquidation_price,
                quantity=quantity,
                pnl=-self._initial_capital,
                commission=total_commission,
                reason=f"爆仓 (权益归零)",
                exit_type="liquidation",
                leverage=self.config.leverage,
                side="long",
                position_value=position_value,
                margin_used=margin_used,
            )
            self._completed_trades.append(completed)
            
            self._position = Position(
                side=PositionSide.EMPTY,
                quantity=0.0,
                entry_price=0.0,
                entry_time=timestamp,
            )
            
            self._liquidation_hits += 1
            
            return True
        
        return False
    
    def _close_position(
        self,
        price: float,
        timestamp: datetime,
        reason: str = "",
        exit_type: str = "signal"
    ) -> None:
        """平仓"""
        if not self._position.is_open:
            return
        
        quantity = self._position.quantity
        entry_price = self._position.entry_price
        entry_time = self._position.entry_time
        position_side = self._position.side
        
        if position_side == PositionSide.LONG:
            price_diff = price - entry_price
        else:
            price_diff = entry_price - price
        
        gross_pnl = price_diff * quantity * self.config.leverage
        exit_commission = quantity * price * self.config.commission_rate
        net_pnl = gross_pnl - exit_commission
        
        self._capital += gross_pnl - exit_commission
        
        close_side = "CLOSE_LONG" if position_side == PositionSide.LONG else "CLOSE_SHORT"
        
        self._trade_id += 1
        trade = Trade(
            trade_id=self._trade_id,
            timestamp=timestamp,
            symbol=self.config.symbol,
            side=close_side,
            quantity=quantity,
            price=price,
            commission=exit_commission,
            pnl=net_pnl,
            reason=reason,
        )
        self._trades.append(trade)
        
        entry_commission = quantity * entry_price * self.config.commission_rate
        total_commission = entry_commission + exit_commission
        
        position_value = quantity * entry_price
        margin_used = position_value / self.config.leverage
        
        completed = CompletedTrade(
            entry_time=entry_time,
            exit_time=timestamp,
            entry_price=entry_price,
            exit_price=price,
            quantity=quantity,
            pnl=net_pnl,
            commission=total_commission,
            reason=reason,
            exit_type=exit_type,
            leverage=self.config.leverage,
            side="long",
            position_value=position_value,
            margin_used=margin_used,
        )
        self._completed_trades.append(completed)
        
        self._position = Position(
            side=PositionSide.EMPTY,
            quantity=0.0,
            entry_price=0.0,
            entry_time=timestamp,
        )
    
    def _process_signal(self, signal: Signal, bar: Bar, timestamp: datetime) -> None:
        """处理交易信号"""
        if signal.type == SignalType.OPEN_LONG:
            if self._position.is_open:
                return
            
            quantity = self._calculate_position_size(bar.close)
            if quantity <= 0:
                return
            
            commission = quantity * bar.close * self.config.commission_rate
            
            stop_loss = signal.stop_loss
            take_profit = signal.take_profit
            
            if stop_loss is None and self.risk_config.stop_loss_pct > 0:
                stop_loss = self.risk_config.calculate_stop_loss_price(bar.close, is_long=True)
            
            if take_profit is None and self.risk_config.take_profit_pct > 0:
                take_profit = self.risk_config.calculate_take_profit_price(bar.close, is_long=True)
            
            self._position = Position(
                side=PositionSide.LONG,
                quantity=quantity,
                entry_price=bar.close,
                entry_time=timestamp,
                stop_loss=stop_loss,
                take_profit=take_profit,
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
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            self._trades.append(trade)
        
        elif signal.type == SignalType.CLOSE_LONG:
            if not self._position.is_open or self._position.side != PositionSide.LONG:
                return
            
            self._close_position(
                price=bar.close,
                timestamp=timestamp,
                reason=signal.reason,
                exit_type="signal"
            )
        
        elif signal.type == SignalType.OPEN_SHORT:
            if self._position.is_open:
                return
            
            quantity = self._calculate_position_size(bar.close)
            if quantity <= 0:
                return
            
            commission = quantity * bar.close * self.config.commission_rate
            
            stop_loss = signal.stop_loss
            take_profit = signal.take_profit
            
            if stop_loss is None and self.risk_config.stop_loss_pct > 0:
                stop_loss = self.risk_config.calculate_stop_loss_price(bar.close, is_long=False)
            
            if take_profit is None and self.risk_config.take_profit_pct > 0:
                take_profit = self.risk_config.calculate_take_profit_price(bar.close, is_long=False)
            
            self._position = Position(
                side=PositionSide.SHORT,
                quantity=quantity,
                entry_price=bar.close,
                entry_time=timestamp,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            
            self._capital -= commission
            
            self._trade_id += 1
            trade = Trade(
                trade_id=self._trade_id,
                timestamp=timestamp,
                symbol=self.config.symbol,
                side="OPEN_SHORT",
                quantity=quantity,
                price=bar.close,
                commission=commission,
                reason=signal.reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            self._trades.append(trade)
        
        elif signal.type == SignalType.CLOSE_SHORT:
            if not self._position.is_open or self._position.side != PositionSide.SHORT:
                return
            
            self._close_position(
                price=bar.close,
                timestamp=timestamp,
                reason=signal.reason,
                exit_type="signal"
            )
    
    def _calculate_position_size(self, price: float) -> float:
        """计算仓位大小
        
        仓位数量 = 资金 * 仓位比例 / 价格
        盈亏 = 价格变动 * 数量 * 杠杆
        """
        position_value = self._capital * self.config.position_size
        return position_value / price if price > 0 else 0.0
    
    def _calculate_equity(self, current_price: float) -> float:
        """计算当前权益"""
        equity = self._capital
        
        if self._position.is_open:
            if self._position.side == PositionSide.LONG:
                unrealized_pnl = (current_price - self._position.entry_price) * self._position.quantity * self.config.leverage
                equity += unrealized_pnl
            elif self._position.side == PositionSide.SHORT:
                unrealized_pnl = (self._position.entry_price - current_price) * self._position.quantity * self.config.leverage
                equity += unrealized_pnl
        
        return equity
    
    def _generate_result(self) -> BacktestResult:
        """生成回测结果"""
        result = BacktestResult(
            trades=self._trades,
            completed_trades=self._completed_trades,
            initial_capital=self._initial_capital,
            leverage=self.config.leverage,
            stop_loss_hits=self._stop_loss_hits,
            take_profit_hits=self._take_profit_hits,
            liquidation_hits=self._liquidation_hits,
            symbol=self.config.symbol,
            interval=self.config.interval,
            strategy_name=self.strategy.name,
            strategy_params=self.strategy.get_parameters(),
            commission_rate=self.config.commission_rate,
            position_size=self.config.position_size,
            stop_loss_pct=self.config.stop_loss_pct,
            take_profit_pct=self.config.take_profit_pct,
        )
        
        if self._equity_history:
            timestamps = [e[0] for e in self._equity_history]
            equities = [e[1] for e in self._equity_history]
            result.equity_curve = pd.Series(equities, index=timestamps)
            
            result.start_time = timestamps[0]
            result.end_time = timestamps[-1]
            
            result.final_capital = equities[-1]
        
        result.total_return = result.final_capital - result.initial_capital
        result.total_return_pct = (result.final_capital / result.initial_capital - 1) * 100
        
        if len(result.equity_curve) > 1:
            result.returns = result.equity_curve.pct_change().dropna()
            
            rolling_max = result.equity_curve.cummax()
            result.drawdown_curve = (result.equity_curve - rolling_max) / rolling_max * 100
            result.max_drawdown_pct = abs(result.drawdown_curve.min())
            result.max_drawdown = result.initial_capital * result.max_drawdown_pct / 100
        
        result.total_trades = len(self._completed_trades)
        
        wins = [t for t in self._completed_trades if t.pnl > 0]
        losses = [t for t in self._completed_trades if t.pnl < 0]
        
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        
        if result.total_trades > 0:
            result.win_rate = result.winning_trades / result.total_trades * 100
        
        win_pnls = [t.pnl for t in wins]
        loss_pnls = [abs(t.pnl) for t in losses]
        
        result.avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
        result.avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
        
        total_wins = sum(win_pnls)
        total_losses = sum(loss_pnls)
        
        if total_losses > 0:
            result.profit_factor = total_wins / total_losses
        elif total_wins > 0:
            result.profit_factor = float('inf')
        else:
            result.profit_factor = 0
        
        if len(result.returns) > 0 and result.returns.std() > 0:
            result.sharpe_ratio = (result.returns.mean() * 252) / (result.returns.std() * (252 ** 0.5))
            
            downside = result.returns[result.returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                result.sortino_ratio = (result.returns.mean() * 252) / (downside.std() * (252 ** 0.5))
            
            if result.max_drawdown_pct > 0:
                result.calmar_ratio = result.total_return_pct / result.max_drawdown_pct
        
        return result
