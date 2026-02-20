"""
回测报告模块

提供专业的回测报告生成
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from backtest.engine import BacktestResult, Trade, CompletedTrade
from backtest.metrics import PerformanceMetrics


@dataclass
class BacktestReport:
    """回测报告"""
    
    result: BacktestResult
    strategy_name: str = ""
    symbol: str = ""
    interval: str = ""
    generated_at: datetime = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()
    
    def get_summary(self) -> dict[str, Any]:
        """获取摘要信息"""
        trades_for_metrics = [
            {"pnl": t.pnl, "entry_price": t.entry_price, "exit_price": t.exit_price}
            for t in self.result.completed_trades
        ]
        metrics = PerformanceMetrics.calculate_all(self.result.equity_curve, trades_for_metrics)
        
        return {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "interval": self.interval,
            "period": {
                "start": str(self.result.start_time) if self.result.start_time else "N/A",
                "end": str(self.result.end_time) if self.result.end_time else "N/A",
            },
            "capital": {
                "initial": self.result.initial_capital,
                "final": self.result.final_capital,
            },
            "performance": {
                "total_return": self.result.total_return_pct,
                "annualized_return": metrics.get("annualized_return", 0),
                "max_drawdown": self.result.max_drawdown,
                "max_drawdown_pct": self.result.max_drawdown_pct,
                "sharpe_ratio": self.result.sharpe_ratio,
                "sortino_ratio": self.result.sortino_ratio,
                "calmar_ratio": self.result.calmar_ratio,
                "win_rate": self.result.win_rate,
                "profit_factor": self.result.profit_factor,
                "total_trades": self.result.total_trades,
                "winning_trades": self.result.winning_trades,
                "losing_trades": self.result.losing_trades,
                "avg_win": self.result.avg_win,
                "avg_loss": self.result.avg_loss,
                "volatility": metrics.get("volatility", 0),
            },
        }
    
    def format_text_report(self) -> str:
        """生成文本格式报告"""
        summary = self.get_summary()
        perf = summary["performance"]
        
        lines = [
            "=" * 70,
            f"                     回测报告 - {self.strategy_name}",
            "=" * 70,
            "",
            f" 交易对: {self.symbol}    周期: {self.interval}",
            f" 时间范围: {summary['period']['start']} 至 {summary['period']['end']}",
            "",
            "-" * 70,
            "                         绩效指标",
            "-" * 70,
            "",
            "【收益指标】",
            f"  初始资金:     {self.result.initial_capital:,.2f} USDT",
            f"  最终资金:     {self.result.final_capital:,.2f} USDT",
            f"  总收益:       {self.result.total_return:,.2f} USDT",
            f"  总收益率:     {perf['total_return']:.2f}%",
            "",
            "【风险指标】",
            f"  最大回撤:     {perf['max_drawdown']:,.2f} USDT ({perf['max_drawdown_pct']:.2f}%)",
            f"  年化波动率:   {perf['volatility']:.2f}%",
            "",
            "【风险调整收益】",
            f"  夏普比率:     {perf['sharpe_ratio']:.2f}",
            f"  索提诺比率:   {perf['sortino_ratio']:.2f}",
            f"  卡玛比率:     {perf['calmar_ratio']:.2f}",
            "",
            "-" * 70,
            "                         交易统计",
            "-" * 70,
            "",
            f"  总交易次数:   {perf['total_trades']}",
            f"  盈利次数:     {perf['winning_trades']}",
            f"  亏损次数:     {perf['losing_trades']}",
            f"  胜率:         {perf['win_rate']:.1f}%",
            f"  盈亏比:       {'∞' if perf['profit_factor'] == float('inf') else f"{perf['profit_factor']:.2f}"}",
            f"  平均盈利:     {perf['avg_win']:,.2f} USDT",
            f"  平均亏损:     {perf['avg_loss']:,.2f} USDT",
            "",
            "=" * 70,
            f" 报告生成时间: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
        ]
        
        return "\n".join(lines)
    
    def get_trade_dataframe(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        if not self.result.completed_trades:
            return pd.DataFrame()
        
        records = []
        for t in self.result.completed_trades:
            records.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "commission": t.commission,
                "reason": t.reason,
            })
        
        df = pd.DataFrame(records)
        if "entry_time" in df.columns:
            df["entry_time"] = pd.to_datetime(df["entry_time"])
            df = df.set_index("entry_time")
        
        return df


def generate_comparison_report(results: list[tuple[str, BacktestResult]], symbol: str = "") -> str:
    """生成策略对比报告"""
    lines = [
        "=" * 80,
        "                        策略对比报告",
        "=" * 80,
        "",
        f"交易对: {symbol}",
        "",
        "-" * 80,
        f"{'策略名称':<20} {'总收益率':>12} {'最大回撤':>12} {'夏普比率':>10} {'胜率':>10}",
        "-" * 80,
    ]
    
    for name, result in results:
        lines.append(
            f"{name:<20} {result.total_return_pct:>11.2f}% {result.max_drawdown_pct:>11.2f}% "
            f"{result.sharpe_ratio:>10.2f} {result.win_rate:>9.1f}%"
        )
    
    lines.extend([
        "-" * 80,
        "",
        "=" * 80,
    ])
    
    return "\n".join(lines)
