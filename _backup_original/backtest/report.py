"""
回测报告模块

提供专业的回测报告生成和可视化
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from backtest.engine import BacktestResult, Trade
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
        trades = [t.__dict__ if hasattr(t, '__dict__') else t for t in self.result.trades]
        metrics = PerformanceMetrics.calculate_all(self.result.equity_curve, trades)
        
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
            "performance": metrics,
        }
    
    def format_text_report(self) -> str:
        """生成文本格式报告"""
        summary = self.get_summary()
        perf = summary["performance"]
        
        lines = [
            "=" * 70,
            f"                    回测报告 - {self.strategy_name}",
            "=" * 70,
            "",
            f"交易对: {self.symbol}    周期: {self.interval}",
            f"时间范围: {summary['period']['start']} 至 {summary['period']['end']}",
            "",
            "-" * 70,
            "                        绩效指标",
            "-" * 70,
            "",
            "【收益指标】",
            f"  初始资金:     {self.result.initial_capital:,.2f} USDT",
            f"  最终资金:     {self.result.final_capital:,.2f} USDT",
            f"  总收益:       {self.result.total_return:,.2f} USDT",
            f"  总收益率:     {perf['total_return']:.2f}%",
            f"  年化收益率:   {perf['annualized_return']:.2f}%",
            "",
            "【风险指标】",
            f"  最大回撤:     {perf['max_drawdown']:,.2f} USDT ({perf['max_drawdown_pct']:.2f}%)",
            f"  年化波动率:   {perf['volatility']:.2f}%",
            f"  VaR(95%):     {perf['var_95']:.2f}%",
            f"  CVaR(95%):    {perf['cvar_95']:.2f}%",
            "",
            "【风险调整收益】",
            f"  夏普比率:     {perf['sharpe_ratio']:.2f}",
            f"  索提诺比率:   {perf['sortino_ratio']:.2f}",
            f"  卡玛比率:     {perf['calmar_ratio']:.2f}",
            "",
            "-" * 70,
            "                        交易统计",
            "-" * 70,
            "",
            f"  总交易次数:   {perf['total_trades']}",
            f"  盈利次数:     {perf['winning_trades']}",
            f"  亏损次数:     {perf['losing_trades']}",
            f"  胜率:         {perf['win_rate']:.1f}%",
            f"  盈亏比:       {perf['profit_factor']:.2f}",
            f"  平均盈利:     {perf['avg_win']:,.2f} USDT",
            f"  平均亏损:     {perf['avg_loss']:,.2f} USDT",
            f"  期望值:       {perf['expectancy']:,.2f} USDT",
            "",
            "=" * 70,
            f"报告生成时间: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
        ]
        
        return "\n".join(lines)
    
    def get_trade_dataframe(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        if not self.result.trades:
            return pd.DataFrame()
        
        records = []
        for t in self.result.trades:
            if hasattr(t, '__dict__'):
                records.append(t.__dict__)
            else:
                records.append(t)
        
        df = pd.DataFrame(records)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        return df
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """获取权益曲线DataFrame"""
        if self.result.equity_curve.empty:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'equity': self.result.equity_curve,
        })
        
        rolling_max = self.result.equity_curve.cummax()
        df['drawdown'] = (self.result.equity_curve - rolling_max) / rolling_max * 100
        df['drawdown_amount'] = self.result.equity_curve - rolling_max
        
        return df
    
    def get_monthly_returns(self) -> pd.Series:
        """获取月度收益"""
        if self.result.equity_curve.empty:
            return pd.Series()
        
        monthly = self.result.equity_curve.resample('M').last()
        monthly_returns = monthly.pct_change() * 100
        return monthly_returns.dropna()
    
    def get_yearly_returns(self) -> pd.Series:
        """获取年度收益"""
        if self.result.equity_curve.empty:
            return pd.Series()
        
        yearly = self.result.equity_curve.resample('Y').last()
        yearly_returns = yearly.pct_change() * 100
        return yearly_returns.dropna()


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
        trades = [t.__dict__ if hasattr(t, '__dict__') else t for t in result.trades]
        metrics = PerformanceMetrics.calculate_all(result.equity_curve, trades)
        
        lines.append(
            f"{name:<20} {metrics['total_return']:>11.2f}% {metrics['max_drawdown_pct']:>11.2f}% "
            f"{metrics['sharpe_ratio']:>10.2f} {metrics['win_rate']:>9.1f}%"
        )
    
    lines.extend([
        "-" * 80,
        "",
        "=" * 80,
    ])
    
    return "\n".join(lines)
