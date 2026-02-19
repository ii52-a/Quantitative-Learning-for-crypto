"""复合策略参数优化器"""
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
import pandas as pd
import numpy as np

from Strategy.composite import CompositeStrategy, create_composite_strategy
from Strategy.parameter_optimizer import ParameterOptimizer, ParameterRange, OptimizationResult
from Strategy.templates import get_strategy
from backtest.engine import BacktestEngine
from core.config import BacktestConfig


MIN_DATA_REQUIREMENT = 10000


@dataclass
class StrategyAnalysis:
    """策略分析结果"""
    strategy_name: str
    contribution_pct: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    signal_count: int = 0
    optimal_params: dict = field(default_factory=dict)


@dataclass
class CompositeOptimizationResult:
    """复合策略优化结果"""
    best_params: dict[str, Any]
    best_score: float
    strategy_analyses: list[StrategyAnalysis]
    correlation_matrix: pd.DataFrame | None = None
    execution_time: float = 0.0
    data_points: int = 0


class CompositeOptimizer:
    """复合策略优化器"""
    
    def __init__(
        self,
        strategy_configs: list[dict[str, Any]],
        data: pd.DataFrame,
        base_config: BacktestConfig,
    ):
        """
        初始化复合策略优化器
        
        Args:
            strategy_configs: 策略配置列表，每个配置包含:
                - name: 策略名称
                - weight: 初始权重
                - params: 策略参数
            data: K线数据
            base_config: 基础回测配置
        """
        self.strategy_configs = strategy_configs
        self.data = data
        self.base_config = base_config
        
        self._validate_data()
    
    def _validate_data(self) -> None:
        """验证数据量是否满足最低要求"""
        data_count = len(self.data)
        if data_count < MIN_DATA_REQUIREMENT:
            raise ValueError(
                f"数据量不足: 当前{data_count}条, 最低要求{MIN_DATA_REQUIREMENT}条。"
                f"数据量过小可能导致模型过拟合。"
            )
    
    def analyze_individual_strategies(self) -> list[StrategyAnalysis]:
        """分析各个策略的独立表现"""
        analyses = []
        
        for config in self.strategy_configs:
            strategy_name = config.get("name")
            params = config.get("params", {})
            
            strategy = get_strategy(strategy_name)
            strategy.set_params(params)
            
            engine = BacktestEngine(strategy, self.base_config)
            result = engine.run(self.data)
            
            analysis = StrategyAnalysis(
                strategy_name=strategy_name,
                win_rate=result.win_rate,
                avg_return=result.total_return_pct,
                signal_count=result.total_trades,
                optimal_params=params,
            )
            
            analyses.append(analysis)
        
        return analyses
    
    def optimize_weights(
        self,
        n_iterations: int = 50,
        optimization_metric: str = "sharpe_ratio",
    ) -> dict[str, float]:
        """优化各策略权重"""
        best_weights = {cfg["name"]: cfg.get("weight", 1.0) for cfg in self.strategy_configs}
        best_score = float('-inf')
        
        for _ in range(n_iterations):
            weights = {
                cfg["name"]: np.random.uniform(0.1, 2.0)
                for cfg in self.strategy_configs
            }
            
            strategies = []
            for cfg in self.strategy_configs:
                strategy = get_strategy(cfg["name"])
                strategy.set_params(cfg.get("params", {}))
                strategies.append((strategy, weights[cfg["name"]]))
            
            composite = create_composite_strategy(strategies, "weighted")
            
            engine = BacktestEngine(composite, self.base_config)
            result = engine.run(self.data)
            
            score = getattr(result, optimization_metric, result.sharpe_ratio)
            
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
        
        return best_weights
    
    def analyze_correlation(self) -> pd.DataFrame:
        """分析策略间的信号相关性"""
        signals_data = {}
        
        for config in self.strategy_configs:
            strategy_name = config.get("name")
            params = config.get("params", {})
            
            strategy = get_strategy(strategy_name)
            strategy.set_params(params)
            
            signals = self._extract_signals(strategy)
            signals_data[strategy_name] = signals
        
        df = pd.DataFrame(signals_data)
        correlation = df.corr()
        
        return correlation
    
    def _extract_signals(self, strategy: Any) -> list[int]:
        """提取策略信号序列"""
        from core.context import BacktestContext
        
        signals = []
        context = BacktestContext()
        strategy.initialize(context)
        
        for idx, row in self.data.iterrows():
            from core.data_types import Bar
            bar = Bar(
                timestamp=idx,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
            )
            
            result = strategy.on_bar(bar, context)
            if result and result.signal:
                from core.constants import SignalType
                if result.signal.type == SignalType.OPEN_LONG:
                    signals.append(1)
                elif result.signal.type == SignalType.OPEN_SHORT:
                    signals.append(-1)
                else:
                    signals.append(0)
            else:
                signals.append(0)
        
        return signals
    
    def run_full_optimization(
        self,
        param_ranges: dict[str, list[ParameterRange]] | None = None,
        n_iterations: int = 100,
    ) -> CompositeOptimizationResult:
        """运行完整优化流程"""
        start_time = datetime.now()
        
        individual_analyses = self.analyze_individual_strategies()
        
        optimal_weights = self.optimize_weights(n_iterations // 2)
        
        correlation_matrix = self.analyze_correlation()
        
        for analysis in individual_analyses:
            analysis.contribution_pct = optimal_weights.get(analysis.strategy_name, 1.0)
        
        strategies = []
        for config in self.strategy_configs:
            strategy = get_strategy(config["name"])
            strategy.set_params(config.get("params", {}))
            weight = optimal_weights.get(config["name"], 1.0)
            strategies.append((strategy, weight))
        
        composite = create_composite_strategy(strategies, "weighted")
        
        engine = BacktestEngine(composite, self.base_config)
        result = engine.run(self.data)
        
        best_params = {
            "weights": optimal_weights,
            "combine_method": "weighted",
        }
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return CompositeOptimizationResult(
            best_params=best_params,
            best_score=result.sharpe_ratio,
            strategy_analyses=individual_analyses,
            correlation_matrix=correlation_matrix,
            execution_time=execution_time,
            data_points=len(self.data),
        )
    
    def generate_report(self, result: CompositeOptimizationResult) -> str:
        """生成优化报告"""
        lines = [
            "=" * 60,
            "复合策略优化报告",
            "=" * 60,
            f"数据量: {result.data_points} 条",
            f"执行时间: {result.execution_time:.2f} 秒",
            f"最优得分: {result.best_score:.4f}",
            "",
            "策略权重配置:",
        ]
        
        weights = result.best_params.get("weights", {})
        for name, weight in weights.items():
            lines.append(f"  {name}: {weight:.3f}")
        
        lines.extend([
            "",
            "各策略独立表现:",
        ])
        
        for analysis in result.strategy_analyses:
            lines.extend([
                f"  {analysis.strategy_name}:",
                f"    胜率: {analysis.win_rate:.2f}%",
                f"    收益: {analysis.avg_return:.2f}%",
                f"    信号数: {analysis.signal_count}",
            ])
        
        if result.correlation_matrix is not None:
            lines.extend([
                "",
                "策略相关性矩阵:",
            ])
            lines.append(result.correlation_matrix.to_string())
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
