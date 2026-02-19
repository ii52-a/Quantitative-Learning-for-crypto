"""复合策略参数优化器（升级版）"""
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime

import numpy as np
import pandas as pd

from Strategy.base import Bar, Position, StrategyContext
from Strategy.composite import create_composite_strategy
from Strategy.parameter_optimizer import ParameterRange
from Strategy.templates import get_strategy
from backtest.engine import BacktestEngine
from core.config import BacktestConfig
from core.constants import PositionSide, SignalType


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

            analyses.append(
                StrategyAnalysis(
                    strategy_name=strategy_name,
                    win_rate=result.win_rate,
                    avg_return=result.total_return_pct,
                    signal_count=result.total_trades,
                    optimal_params=params,
                )
            )

        return analyses

    def _normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """归一化权重，保证总和为1，避免不同规模影响阈值。"""
        positive = {k: max(0.0, float(v)) for k, v in weights.items()}
        s = sum(positive.values())
        if s <= 1e-12:
            n = len(positive)
            return {k: 1.0 / n for k in positive} if n else {}
        return {k: v / s for k, v in positive.items()}

    def _score_result(self, result: Any, penalty: float = 0.0) -> float:
        """稳定的复合评分函数（收益-风险-稳定性）"""
        if result.total_trades <= 0:
            return float("-inf")

        # 回撤越大惩罚越高；交易太少也惩罚
        trade_penalty = max(0, 5 - result.total_trades) * 0.15
        dd_penalty = result.max_drawdown_pct * 0.03
        liq_penalty = result.liquidation_hits * 5 if hasattr(result, "liquidation_hits") else 0

        # 主体：收益、夏普、胜率综合
        base = (
            result.total_return_pct * 0.45
            + result.sharpe_ratio * 25
            + result.win_rate * 0.15
        )

        return base - dd_penalty - trade_penalty - liq_penalty - penalty

    def _evaluate_weight_set(
        self,
        weights: dict[str, float],
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        optimization_metric: str,
        correlation_penalty: float,
    ) -> float:
        """在训练+验证分段上评估一组权重，提升泛化能力。"""
        norm_weights = self._normalize_weights(weights)

        strategies = []
        for cfg in self.strategy_configs:
            strategy = get_strategy(cfg["name"])
            strategy.set_params(cfg.get("params", {}))
            strategies.append((strategy, norm_weights[cfg["name"]]))

        composite_train = create_composite_strategy(strategies, "weighted")
        train_res = BacktestEngine(composite_train, self.base_config).run(train_data)

        # 验证段重新实例化策略，避免状态污染
        strategies_valid = []
        for cfg in self.strategy_configs:
            strategy = get_strategy(cfg["name"])
            strategy.set_params(cfg.get("params", {}))
            strategies_valid.append((strategy, norm_weights[cfg["name"]]))
        composite_valid = create_composite_strategy(strategies_valid, "weighted")
        valid_res = BacktestEngine(composite_valid, self.base_config).run(valid_data)

        if optimization_metric in {"sharpe_ratio", "total_return_pct", "calmar_ratio", "win_rate"}:
            train_score = float(getattr(train_res, optimization_metric, train_res.sharpe_ratio))
            valid_score = float(getattr(valid_res, optimization_metric, valid_res.sharpe_ratio))
            stability_gap = abs(train_score - valid_score)
            return 0.45 * train_score + 0.55 * valid_score - 0.2 * stability_gap - correlation_penalty

        train_score = self._score_result(train_res, correlation_penalty)
        valid_score = self._score_result(valid_res, correlation_penalty)
        stability_gap = abs(train_score - valid_score)
        return 0.45 * train_score + 0.55 * valid_score - 0.15 * stability_gap

    def optimize_weights(
        self,
        n_iterations: int = 80,
        optimization_metric: str = "composite",
    ) -> dict[str, float]:
        """优化各策略权重（Dirichlet采样 + 分阶段收敛 + 验证集约束）"""
        names = [cfg["name"] for cfg in self.strategy_configs]
        n = len(names)
        if n == 0:
            return {}

        split_idx = max(int(len(self.data) * 0.7), 1)
        train_data = self.data.iloc[:split_idx]
        valid_data = self.data.iloc[split_idx:] if split_idx < len(self.data) else self.data.iloc[-1:]

        correlation = self.analyze_correlation()
        avg_abs_corr = float(correlation.abs().where(~np.eye(len(correlation), dtype=bool)).mean().mean()) if not correlation.empty else 0.0

        best_weights = self._normalize_weights({cfg["name"]: cfg.get("weight", 1.0) for cfg in self.strategy_configs})
        best_score = float("-inf")

        # 阶段1：全局搜索
        phase1 = max(10, int(n_iterations * 0.6))
        for _ in range(phase1):
            sample = np.random.dirichlet(np.ones(n))
            weights = {name: float(sample[i]) for i, name in enumerate(names)}
            score = self._evaluate_weight_set(
                weights,
                train_data,
                valid_data,
                optimization_metric,
                correlation_penalty=avg_abs_corr * 2.5,
            )
            if score > best_score:
                best_score = score
                best_weights = weights.copy()

        # 阶段2：局部精修（围绕最优权重采样）
        phase2 = max(5, n_iterations - phase1)
        alpha_base = np.array([max(0.05, best_weights[name]) for name in names]) * 80
        for _ in range(phase2):
            sample = np.random.dirichlet(alpha_base)
            weights = {name: float(sample[i]) for i, name in enumerate(names)}
            score = self._evaluate_weight_set(
                weights,
                train_data,
                valid_data,
                optimization_metric,
                correlation_penalty=avg_abs_corr * 2.5,
            )
            if score > best_score:
                best_score = score
                best_weights = weights.copy()

        return self._normalize_weights(best_weights)

    def analyze_correlation(self) -> pd.DataFrame:
        """分析策略间信号相关性"""
        signals_data = {}

        for config in self.strategy_configs:
            strategy_name = config.get("name")
            params = config.get("params", {})

            strategy = get_strategy(strategy_name)
            strategy.set_params(params)

            signals_data[strategy_name] = self._extract_signals(strategy)

        if not signals_data:
            return pd.DataFrame()

        return pd.DataFrame(signals_data).corr().fillna(0.0)

    def _extract_signals(self, strategy: Any) -> list[int]:
        """提取策略信号序列（与当前框架接口对齐）"""
        signals: list[int] = []
        context = StrategyContext(
            symbol=self.base_config.symbol,
            interval=self.base_config.interval,
            position=Position(
                side=PositionSide.EMPTY,
                quantity=0.0,
                entry_price=0.0,
                entry_time=datetime.now(),
            ),
            equity=self.base_config.initial_capital,
            available_capital=self.base_config.initial_capital,
            current_price=0.0,
            timestamp=datetime.now(),
        )
        strategy.initialize(context)

        for idx, row in self.data.iterrows():
            bar = Bar(
                timestamp=idx if isinstance(idx, datetime) else pd.to_datetime(idx, utc=True),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]) if "volume" in row else 0.0,
                symbol=self.base_config.symbol,
                interval=self.base_config.interval,
            )
            context.current_price = bar.close
            context.timestamp = bar.timestamp

            result = strategy.on_bar(bar, context)
            if not result or not result.signal:
                signals.append(0)
                continue

            if result.signal.type == SignalType.OPEN_LONG:
                signals.append(1)
            elif result.signal.type == SignalType.OPEN_SHORT:
                signals.append(-1)
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

        optimal_weights = self.optimize_weights(max(20, n_iterations // 2), optimization_metric="composite")

        correlation_matrix = self.analyze_correlation()

        for analysis in individual_analyses:
            analysis.contribution_pct = optimal_weights.get(analysis.strategy_name, 0.0)

        strategies = []
        for config in self.strategy_configs:
            strategy = get_strategy(config["name"])
            strategy.set_params(config.get("params", {}))
            strategies.append((strategy, optimal_weights.get(config["name"], 0.0)))

        composite = create_composite_strategy(strategies, "weighted")

        result = BacktestEngine(composite, self.base_config).run(self.data)

        best_params = {
            "weights": optimal_weights,
            "combine_method": "weighted",
            "iterations": n_iterations,
        }

        execution_time = (datetime.now() - start_time).total_seconds()

        return CompositeOptimizationResult(
            best_params=best_params,
            best_score=self._score_result(result),
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
        for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
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
                f"    权重贡献: {analysis.contribution_pct:.3f}",
            ])

        if result.correlation_matrix is not None and not result.correlation_matrix.empty:
            lines.extend([
                "",
                "策略相关性矩阵:",
            ])
            lines.append(result.correlation_matrix.to_string())

        lines.append("=" * 60)

        return "\n".join(lines)
