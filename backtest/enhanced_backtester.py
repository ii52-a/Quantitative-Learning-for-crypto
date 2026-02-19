"""强化回测模块 - 策略压力测试和样本外验证"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from enum import Enum
import numpy as np
import pandas as pd

from Strategy.base import BaseStrategy
from backtest.engine import BacktestEngine, BacktestResult
from core.config import BacktestConfig
from app_logger.logger_setup import Logger

logger = Logger(__name__)


class MarketScenario(Enum):
    """市场情景类型"""
    NORMAL = "normal"
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    FLASH_CRASH = "flash_crash"
    PUMP = "pump"
    TREND_REVERSAL = "trend_reversal"


@dataclass
class ScenarioConfig:
    """情景配置"""
    scenario_type: MarketScenario
    volatility_multiplier: float = 1.0
    trend_bias: float = 0.0
    gap_probability: float = 0.0
    gap_size_range: tuple = (-0.1, 0.1)
    duration_periods: int = 100


@dataclass
class StressTestResult:
    """压力测试结果"""
    scenario: MarketScenario
    total_return_pct: float
    max_drawdown_pct: float
    win_rate: float
    sharpe_ratio: float
    total_trades: int
    profit_factor: float
    recovery_factor: float
    avg_trade_duration: float
    worst_trade: float
    best_trade: float
    consecutive_losses: int
    details: dict = field(default_factory=dict)


@dataclass
class MonteCarloResult:
    """蒙特卡洛模拟结果"""
    simulations: int
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    var_95: float
    var_99: float
    cvar_95: float
    profit_probability: float
    max_drawdown_distribution: dict
    all_results: list = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """滚动窗口测试结果"""
    window_results: list
    avg_return: float
    avg_sharpe: float
    avg_max_drawdown: float
    return_stability: float
    performance_degradation: float
    consistency_score: float


class EnhancedBacktester:
    """强化回测器"""
    
    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        base_config: BacktestConfig,
    ):
        self.strategy_class = strategy_class
        self.base_config = base_config
    
    def run_scenario_test(
        self,
        data: pd.DataFrame,
        scenario: ScenarioConfig,
        strategy_params: dict | None = None,
    ) -> StressTestResult:
        """运行情景测试"""
        modified_data = self._apply_scenario(data, scenario)
        
        strategy = self.strategy_class(strategy_params or {})
        engine = BacktestEngine(strategy, self.base_config)
        result = engine.run(modified_data)
        
        worst_trade = 0
        best_trade = 0
        if result.completed_trades:
            trade_pnls = [t.pnl for t in result.completed_trades]
            worst_trade = min(trade_pnls) if trade_pnls else 0
            best_trade = max(trade_pnls) if trade_pnls else 0
        
        return StressTestResult(
            scenario=scenario.scenario_type,
            total_return_pct=result.total_return_pct,
            max_drawdown_pct=result.max_drawdown_pct,
            win_rate=result.win_rate,
            sharpe_ratio=result.sharpe_ratio,
            total_trades=result.total_trades,
            profit_factor=result.profit_factor,
            recovery_factor=abs(result.total_return / result.max_drawdown) if result.max_drawdown != 0 else 0,
            avg_trade_duration=0,
            worst_trade=worst_trade,
            best_trade=best_trade,
            consecutive_losses=0,
            details={
                "volatility_multiplier": scenario.volatility_multiplier,
                "trend_bias": scenario.trend_bias,
            }
        )
    
    def _apply_scenario(
        self,
        data: pd.DataFrame,
        scenario: ScenarioConfig,
    ) -> pd.DataFrame:
        """应用市场情景"""
        modified = data.copy()
        
        if scenario.scenario_type == MarketScenario.NORMAL:
            pass
        
        elif scenario.scenario_type == MarketScenario.BULL_MARKET:
            trend = np.linspace(0, scenario.trend_bias, len(modified))
            modified['close'] = modified['close'] * (1 + trend)
            modified['high'] = modified['high'] * (1 + trend * 1.1)
            modified['low'] = modified['low'] * (1 + trend * 0.9)
        
        elif scenario.scenario_type == MarketScenario.BEAR_MARKET:
            trend = np.linspace(0, -abs(scenario.trend_bias), len(modified))
            modified['close'] = modified['close'] * (1 + trend)
            modified['high'] = modified['high'] * (1 + trend * 0.9)
            modified['low'] = modified['low'] * (1 + trend * 1.1)
        
        elif scenario.scenario_type == MarketScenario.SIDEWAYS:
            mid_price = modified['close'].iloc[0]
            amplitude = modified['close'].std() * scenario.volatility_multiplier
            cycle = np.sin(np.linspace(0, 4 * np.pi, len(modified)))
            modified['close'] = mid_price + cycle * amplitude
            modified['high'] = modified['close'] * 1.005
            modified['low'] = modified['close'] * 0.995
        
        elif scenario.scenario_type == MarketScenario.HIGH_VOLATILITY:
            returns = modified['close'].pct_change()
            modified_returns = returns * scenario.volatility_multiplier
            modified_returns.iloc[0] = 0
            modified['close'] = modified['close'].iloc[0] * (1 + modified_returns).cumprod()
            modified['high'] = modified['close'] * (1 + 0.01 * scenario.volatility_multiplier)
            modified['low'] = modified['close'] * (1 - 0.01 * scenario.volatility_multiplier)
        
        elif scenario.scenario_type == MarketScenario.FLASH_CRASH:
            crash_idx = len(modified) // 2
            crash_depth = abs(scenario.trend_bias)
            
            pre_crash = np.ones(crash_idx)
            crash = np.linspace(1, 1 - crash_depth, 20)
            recovery = np.linspace(1 - crash_depth, 1 - crash_depth * 0.3, len(modified) - crash_idx - 20)
            
            crash_factor = np.concatenate([pre_crash, crash, recovery])[:len(modified)]
            modified['close'] = modified['close'] * crash_factor
            modified['high'] = modified['close'] * 1.005
            modified['low'] = modified['close'] * 0.995
        
        elif scenario.scenario_type == MarketScenario.PUMP:
            pump_idx = len(modified) // 3
            pump_height = abs(scenario.trend_bias)
            
            pre_pump = np.ones(pump_idx)
            pump = np.linspace(1, 1 + pump_height, 30)
            post_pump = np.linspace(1 + pump_height, 1 + pump_height * 0.5, len(modified) - pump_idx - 30)
            
            pump_factor = np.concatenate([pre_pump, pump, post_pump])[:len(modified)]
            modified['close'] = modified['close'] * pump_factor
            modified['high'] = modified['close'] * 1.01
            modified['low'] = modified['close'] * 0.99
        
        elif scenario.scenario_type == MarketScenario.TREND_REVERSAL:
            half = len(modified) // 2
            trend1 = np.linspace(0, scenario.trend_bias, half)
            trend2 = np.linspace(scenario.trend_bias, -scenario.trend_bias, len(modified) - half)
            trend = np.concatenate([trend1, trend2])
            modified['close'] = modified['close'] * (1 + trend)
            modified['high'] = modified['close'] * 1.005
            modified['low'] = modified['close'] * 0.995
        
        if scenario.gap_probability > 0:
            gap_mask = np.random.random(len(modified)) < scenario.gap_probability
            gap_sizes = np.random.uniform(
                scenario.gap_size_range[0],
                scenario.gap_size_range[1],
                len(modified)
            )
            gap_multiplier = np.where(gap_mask, 1 + gap_sizes, 1)
            modified['close'] = modified['close'] * gap_multiplier
            modified['high'] = modified['high'] * gap_multiplier
            modified['low'] = modified['low'] * gap_multiplier
        
        modified['open'] = modified['close'].shift(1).fillna(modified['close'].iloc[0])
        
        return modified
    
    def run_all_scenarios(
        self,
        data: pd.DataFrame,
        strategy_params: dict | None = None,
    ) -> dict[MarketScenario, StressTestResult]:
        """运行所有预设情景测试"""
        scenarios = {
            MarketScenario.NORMAL: ScenarioConfig(MarketScenario.NORMAL),
            MarketScenario.BULL_MARKET: ScenarioConfig(
                MarketScenario.BULL_MARKET,
                trend_bias=0.3,
                volatility_multiplier=1.2
            ),
            MarketScenario.BEAR_MARKET: ScenarioConfig(
                MarketScenario.BEAR_MARKET,
                trend_bias=0.3,
                volatility_multiplier=1.5
            ),
            MarketScenario.SIDEWAYS: ScenarioConfig(
                MarketScenario.SIDEWAYS,
                volatility_multiplier=0.8
            ),
            MarketScenario.HIGH_VOLATILITY: ScenarioConfig(
                MarketScenario.HIGH_VOLATILITY,
                volatility_multiplier=3.0
            ),
            MarketScenario.FLASH_CRASH: ScenarioConfig(
                MarketScenario.FLASH_CRASH,
                trend_bias=0.3,
                volatility_multiplier=2.0
            ),
            MarketScenario.PUMP: ScenarioConfig(
                MarketScenario.PUMP,
                trend_bias=0.5,
                volatility_multiplier=1.5
            ),
            MarketScenario.TREND_REVERSAL: ScenarioConfig(
                MarketScenario.TREND_REVERSAL,
                trend_bias=0.2,
                volatility_multiplier=1.3
            ),
        }
        
        results = {}
        for scenario_type, config in scenarios.items():
            try:
                result = self.run_scenario_test(data, config, strategy_params)
                results[scenario_type] = result
            except Exception as e:
                logger.warning(f"情景测试失败 {scenario_type.value}: {e}")
        
        return results
    
    def run_monte_carlo(
        self,
        data: pd.DataFrame,
        n_simulations: int = 100,
        strategy_params: dict | None = None,
        random_seed: int | None = None,
    ) -> MonteCarloResult:
        """运行蒙特卡洛模拟"""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        returns = data['close'].pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        
        all_results = []
        all_returns = []
        all_drawdowns = []
        
        for i in range(n_simulations):
            try:
                simulated_returns = np.random.normal(mean_return, std_return, len(data))
                simulated_prices = data['close'].iloc[0] * (1 + pd.Series(simulated_returns)).cumprod()
                
                sim_data = data.copy()
                sim_data['close'] = simulated_prices.values
                sim_data['high'] = simulated_prices.values * 1.005
                sim_data['low'] = simulated_prices.values * 0.995
                sim_data['open'] = simulated_prices.shift(1).fillna(simulated_prices.iloc[0]).values
                
                strategy = self.strategy_class(strategy_params or {})
                engine = BacktestEngine(strategy, self.base_config)
                result = engine.run(sim_data)
                
                all_results.append({
                    'return': result.total_return_pct,
                    'max_dd': result.max_drawdown_pct,
                    'sharpe': result.sharpe_ratio,
                    'trades': result.total_trades,
                })
                all_returns.append(result.total_return_pct)
                all_drawdowns.append(result.max_drawdown_pct)
                
            except Exception as e:
                logger.debug(f"蒙特卡洛模拟 {i} 失败: {e}")
        
        if not all_returns:
            return MonteCarloResult(
                simulations=0,
                mean_return=0,
                std_return=0,
                min_return=0,
                max_return=0,
                var_95=0,
                var_99=0,
                cvar_95=0,
                profit_probability=0,
                max_drawdown_distribution={},
            )
        
        returns_array = np.array(all_returns)
        drawdowns_array = np.array(all_drawdowns)
        
        sorted_returns = np.sort(returns_array)
        var_95_idx = int(len(sorted_returns) * 0.05)
        var_99_idx = int(len(sorted_returns) * 0.01)
        
        var_95 = sorted_returns[var_95_idx] if var_95_idx < len(sorted_returns) else sorted_returns[0]
        var_99 = sorted_returns[var_99_idx] if var_99_idx < len(sorted_returns) else sorted_returns[0]
        
        cvar_95 = np.mean(sorted_returns[:max(1, var_95_idx + 1)])
        
        dd_bins = {
            "0-5%": 0,
            "5-10%": 0,
            "10-20%": 0,
            "20-30%": 0,
            "30%+": 0,
        }
        for dd in all_drawdowns:
            abs_dd = abs(dd)
            if abs_dd < 5:
                dd_bins["0-5%"] += 1
            elif abs_dd < 10:
                dd_bins["5-10%"] += 1
            elif abs_dd < 20:
                dd_bins["10-20%"] += 1
            elif abs_dd < 30:
                dd_bins["20-30%"] += 1
            else:
                dd_bins["30%+"] += 1
        
        return MonteCarloResult(
            simulations=len(all_results),
            mean_return=np.mean(returns_array),
            std_return=np.std(returns_array),
            min_return=np.min(returns_array),
            max_return=np.max(returns_array),
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            profit_probability=np.mean(np.array(returns_array) > 0) * 100,
            max_drawdown_distribution=dd_bins,
            all_results=all_results,
        )
    
    def run_walk_forward(
        self,
        data: pd.DataFrame,
        train_window: int = 200,
        test_window: int = 50,
        step: int = 25,
        strategy_params: dict | None = None,
    ) -> WalkForwardResult:
        """运行滚动窗口测试"""
        window_results = []
        n_windows = (len(data) - train_window - test_window) // step + 1
        
        for i in range(n_windows):
            train_start = i * step
            train_end = train_start + train_window
            test_end = train_end + test_window
            
            if test_end > len(data):
                break
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[train_end:test_end]
            
            try:
                strategy = self.strategy_class(strategy_params or {})
                engine = BacktestEngine(strategy, self.base_config)
                result = engine.run(test_data)
                
                window_results.append({
                    'window': i + 1,
                    'train_start': data.index[train_start],
                    'test_start': data.index[train_end],
                    'test_end': data.index[test_end - 1],
                    'return': result.total_return_pct,
                    'sharpe': result.sharpe_ratio,
                    'max_dd': result.max_drawdown_pct,
                    'trades': result.total_trades,
                    'win_rate': result.win_rate,
                })
            except Exception as e:
                logger.debug(f"窗口 {i + 1} 测试失败: {e}")
        
        if not window_results:
            return WalkForwardResult(
                window_results=[],
                avg_return=0,
                avg_sharpe=0,
                avg_max_drawdown=0,
                return_stability=0,
                performance_degradation=0,
                consistency_score=0,
            )
        
        returns = [r['return'] for r in window_results]
        sharpes = [r['sharpe'] for r in window_results]
        drawdowns = [r['max_dd'] for r in window_results]
        
        if len(returns) > 1:
            first_half = np.mean(returns[:len(returns)//2])
            second_half = np.mean(returns[len(returns)//2:])
            degradation = ((second_half - first_half) / abs(first_half)) * 100 if first_half != 0 else 0
        else:
            degradation = 0
        
        positive_returns = sum(1 for r in returns if r > 0)
        consistency = positive_returns / len(returns) * 100 if returns else 0
        
        return WalkForwardResult(
            window_results=window_results,
            avg_return=np.mean(returns),
            avg_sharpe=np.mean(sharpes),
            avg_max_drawdown=np.mean(drawdowns),
            return_stability=np.std(returns),
            performance_degradation=degradation,
            consistency_score=consistency,
        )
    
    def run_sensitivity_analysis(
        self,
        data: pd.DataFrame,
        param_name: str,
        param_range: list,
        base_params: dict | None = None,
    ) -> dict:
        """运行参数敏感性分析"""
        results = {}
        
        for value in param_range:
            params = (base_params or {}).copy()
            params[param_name] = value
            
            try:
                strategy = self.strategy_class(params)
                engine = BacktestEngine(strategy, self.base_config)
                result = engine.run(data)
                
                results[value] = {
                    'return': result.total_return_pct,
                    'sharpe': result.sharpe_ratio,
                    'max_dd': result.max_drawdown_pct,
                    'win_rate': result.win_rate,
                    'trades': result.total_trades,
                }
            except Exception as e:
                logger.debug(f"敏感性分析 {param_name}={value} 失败: {e}")
        
        return results
    
    def generate_report(
        self,
        scenario_results: dict | None = None,
        monte_carlo_result: MonteCarloResult | None = None,
        walk_forward_result: WalkForwardResult | None = None,
    ) -> str:
        """生成强化回测报告"""
        report = []
        report.append("=" * 60)
        report.append("强化回测报告")
        report.append("=" * 60)
        
        if scenario_results:
            report.append("\n【市场情景测试】")
            report.append("-" * 40)
            
            for scenario, result in scenario_results.items():
                report.append(f"\n{scenario.value.upper()}:")
                report.append(f"  收益率: {result.total_return_pct:.2f}%")
                report.append(f"  最大回撤: {result.max_drawdown_pct:.2f}%")
                report.append(f"  夏普比率: {result.sharpe_ratio:.2f}")
                report.append(f"  胜率: {result.win_rate:.1f}%")
                report.append(f"  交易次数: {result.total_trades}")
        
        if monte_carlo_result:
            report.append("\n【蒙特卡洛模拟】")
            report.append("-" * 40)
            report.append(f"模拟次数: {monte_carlo_result.simulations}")
            report.append(f"平均收益: {monte_carlo_result.mean_return:.2f}%")
            report.append(f"收益标准差: {monte_carlo_result.std_return:.2f}%")
            report.append(f"VaR(95%): {monte_carlo_result.var_95:.2f}%")
            report.append(f"VaR(99%): {monte_carlo_result.var_99:.2f}%")
            report.append(f"CVaR(95%): {monte_carlo_result.cvar_95:.2f}%")
            report.append(f"盈利概率: {monte_carlo_result.profit_probability:.1f}%")
            report.append(f"回撤分布: {monte_carlo_result.max_drawdown_distribution}")
        
        if walk_forward_result:
            report.append("\n【滚动窗口测试】")
            report.append("-" * 40)
            report.append(f"测试窗口数: {len(walk_forward_result.window_results)}")
            report.append(f"平均收益: {walk_forward_result.avg_return:.2f}%")
            report.append(f"平均夏普: {walk_forward_result.avg_sharpe:.2f}")
            report.append(f"平均最大回撤: {walk_forward_result.avg_max_drawdown:.2f}%")
            report.append(f"收益稳定性(标准差): {walk_forward_result.return_stability:.2f}")
            report.append(f"性能衰减: {walk_forward_result.performance_degradation:.2f}%")
            report.append(f"一致性得分: {walk_forward_result.consistency_score:.1f}%")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def run_enhanced_backtest(
    strategy_class: type[BaseStrategy],
    data: pd.DataFrame,
    config: BacktestConfig,
    run_scenarios: bool = True,
    run_monte_carlo: bool = True,
    run_walk_forward: bool = True,
    n_simulations: int = 50,
) -> dict:
    """运行完整的强化回测"""
    backtester = EnhancedBacktester(strategy_class, config)
    
    results = {}
    
    if run_scenarios:
        results['scenarios'] = backtester.run_all_scenarios(data)
    
    if run_monte_carlo:
        results['monte_carlo'] = backtester.run_monte_carlo(data, n_simulations)
    
    if run_walk_forward:
        results['walk_forward'] = backtester.run_walk_forward(data)
    
    results['report'] = backtester.generate_report(
        results.get('scenarios'),
        results.get('monte_carlo'),
        results.get('walk_forward'),
    )
    
    return results
