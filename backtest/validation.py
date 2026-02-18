"""
策略验证模块

提供全面的参数验证和回测一致性检查：
- 参数边界验证
- 异常值处理
- 回测结果一致性检查
- 交叉验证
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import math

import pandas as pd
import numpy as np

from Strategy.base import BaseStrategy
from backtest.engine import BacktestEngine, BacktestResult
from core.config import BacktestConfig, RiskConfig
from app_logger.logger_setup import Logger

logger = Logger(__name__)


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)
    
    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class ConsistencyCheckResult:
    """一致性检查结果"""
    is_consistent: bool
    metrics_diff: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "is_consistent": self.is_consistent,
            "metrics_diff": self.metrics_diff,
            "issues": self.issues,
        }


class ParameterValidator:
    """参数验证器"""
    
    @staticmethod
    def validate_risk_params(
        leverage: int,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> ValidationResult:
        """验证风险参数"""
        result = ValidationResult(is_valid=True)
        
        if not 1 <= leverage <= 125:
            result.add_error(f"杠杆倍数必须在1-125之间，当前值: {leverage}")
        elif leverage > 20:
            result.add_warning(f"⚠️ 高杠杆风险: {leverage}x")
        
        if stop_loss_pct < 0 or stop_loss_pct > 100:
            result.add_error(f"止损率必须在0-100%之间，当前值: {stop_loss_pct}")
        elif stop_loss_pct > 50:
            result.add_warning(f"止损设置过宽: {stop_loss_pct}%")
        
        if take_profit_pct < 0 or take_profit_pct > 1000:
            result.add_error(f"止盈率必须在0-1000%之间，当前值: {take_profit_pct}")
        
        if stop_loss_pct == 0 and leverage > 1:
            result.add_warning("⚠️ 使用杠杆但未设置止损，可能导致爆仓")
        
        if stop_loss_pct > 0 and take_profit_pct > 0:
            if take_profit_pct < stop_loss_pct:
                result.add_warning("止盈目标低于止损，风险收益比不佳")
        
        return result
    
    @staticmethod
    def validate_strategy_params(
        strategy: BaseStrategy,
        params: dict[str, Any],
    ) -> ValidationResult:
        """验证策略参数"""
        result = ValidationResult(is_valid=True)
        
        for param in strategy.parameters:
            name = param.name
            value = params.get(name, param.default_value)
            
            if param.min_value is not None and value < param.min_value:
                result.add_error(
                    f"参数 {param.display_name}({name}) 不能小于 {param.min_value}，当前值: {value}"
                )
            
            if param.max_value is not None and value > param.max_value:
                result.add_error(
                    f"参数 {param.display_name}({name}) 不能大于 {param.max_value}，当前值: {value}"
                )
            
            if param.options is not None and value not in param.options:
                result.add_error(
                    f"参数 {param.display_name}({name}) 必须是 {param.options} 之一，当前值: {value}"
                )
        
        return result
    
    @staticmethod
    def validate_backtest_config(config: BacktestConfig) -> ValidationResult:
        """验证回测配置"""
        result = ValidationResult(is_valid=True)
        
        errors = config.validate()
        for error in errors:
            result.add_error(error)
        
        risk_errors = config.get_risk_config().validate()
        for error in risk_errors:
            result.add_error(error)
        
        if config.initial_capital < 100:
            result.add_warning("初始资金较小，可能影响策略表现")
        
        if config.data_limit < 100:
            result.add_warning("数据量较少，回测结果可能不够准确")
        
        return result


class BacktestConsistencyChecker:
    """回测一致性检查器"""
    
    MAX_METRICS_DIFF = {
        "total_return_pct": 5.0,
        "max_drawdown_pct": 2.0,
        "win_rate": 5.0,
        "sharpe_ratio": 0.2,
    }
    
    @staticmethod
    def check_reproducibility(
        strategy: BaseStrategy,
        config: BacktestConfig,
        data: pd.DataFrame,
        runs: int = 3,
    ) -> ConsistencyCheckResult:
        """检查回测可重复性"""
        results = []
        
        for i in range(runs):
            strategy_copy = type(strategy)(strategy._params)
            engine = BacktestEngine(strategy_copy, config)
            result = engine.run(data)
            results.append(result)
        
        return BacktestConsistencyChecker._compare_results(results)
    
    @staticmethod
    def _compare_results(results: list[BacktestResult]) -> ConsistencyCheckResult:
        """比较多次回测结果"""
        check_result = ConsistencyCheckResult(is_consistent=True)
        
        if len(results) < 2:
            return check_result
        
        metrics = ["total_return_pct", "max_drawdown_pct", "win_rate", "sharpe_ratio"]
        
        for metric in metrics:
            values = [getattr(r, metric) for r in results]
            
            if len(set(values)) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if mean_val != 0:
                    cv = abs(std_val / mean_val) * 100
                else:
                    cv = std_val
                
                check_result.metrics_diff[metric] = cv
                
                if cv > BacktestConsistencyChecker.MAX_METRICS_DIFF.get(metric, 5.0):
                    check_result.is_consistent = False
                    check_result.issues.append(
                        f"{metric} 变异系数 {cv:.2f}% 超过阈值"
                    )
        
        return check_result
    
    @staticmethod
    def check_cross_validation(
        strategy: BaseStrategy,
        config: BacktestConfig,
        data: pd.DataFrame,
        folds: int = 3,
    ) -> ConsistencyCheckResult:
        """交叉验证检查"""
        check_result = ConsistencyCheckResult(is_consistent=True)
        
        if len(data) < folds * 100:
            check_result.issues.append("数据量不足以进行交叉验证")
            return check_result
        
        fold_size = len(data) // folds
        results = []
        
        for i in range(folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size
            
            fold_data = data.iloc[start_idx:end_idx]
            
            strategy_copy = type(strategy)(strategy._params)
            engine = BacktestEngine(strategy_copy, config)
            result = engine.run(fold_data)
            results.append(result)
        
        returns = [r.total_return_pct for r in results]
        win_rates = [r.win_rate for r in results]
        
        if len(set(returns)) > 1:
            return_std = np.std(returns)
            if return_std > 20:
                check_result.is_consistent = False
                check_result.issues.append(
                    f"不同时间段收益率差异过大 (标准差: {return_std:.2f}%)"
                )
        
        if len(set(win_rates)) > 1:
            wr_std = np.std(win_rates)
            if wr_std > 20:
                check_result.issues.append(
                    f"不同时间段胜率差异较大 (标准差: {wr_std:.2f}%)"
                )
        
        check_result.metrics_diff = {
            "returns_per_fold": returns,
            "win_rates_per_fold": win_rates,
        }
        
        return check_result
    
    @staticmethod
    def check_boundary_conditions(
        strategy: BaseStrategy,
        config: BacktestConfig,
        data: pd.DataFrame,
    ) -> ConsistencyCheckResult:
        """边界条件检查"""
        check_result = ConsistencyCheckResult(is_consistent=True)
        
        if len(data) < 50:
            check_result.issues.append("数据量过少，无法进行边界检查")
            return check_result
        
        extreme_data = data.copy()
        extreme_data['close'] = extreme_data['close'] * 1.5
        extreme_data['high'] = extreme_data['high'] * 1.5
        extreme_data['low'] = extreme_data['low'] * 1.5
        
        try:
            strategy_copy = type(strategy)(strategy._params)
            engine = BacktestEngine(strategy_copy, config)
            result = engine.run(extreme_data)
            
            if abs(result.total_return_pct) > 500:
                check_result.issues.append(
                    f"极端价格条件下收益率异常: {result.total_return_pct:.2f}%"
                )
        except Exception as e:
            check_result.issues.append(f"极端价格条件下策略异常: {str(e)}")
        
        flat_data = data.copy()
        base_price = flat_data['close'].iloc[0]
        flat_data['close'] = base_price
        flat_data['high'] = base_price * 1.001
        flat_data['low'] = base_price * 0.999
        
        try:
            strategy_copy = type(strategy)(strategy._params)
            engine = BacktestEngine(strategy_copy, config)
            result = engine.run(flat_data)
            
            if result.total_trades > 0:
                check_result.issues.append(
                    f"横盘市场产生过多交易: {result.total_trades}笔"
                )
        except Exception as e:
            check_result.issues.append(f"横盘市场条件下策略异常: {str(e)}")
        
        return check_result


class StrategyValidator:
    """策略验证器"""
    
    def __init__(self, strategy: BaseStrategy, config: BacktestConfig):
        self.strategy = strategy
        self.config = config
    
    def validate_all(self, data: pd.DataFrame) -> dict[str, Any]:
        """执行所有验证"""
        results = {}
        
        results["risk_params"] = ParameterValidator.validate_risk_params(
            self.config.leverage,
            self.config.stop_loss_pct,
            self.config.take_profit_pct,
        ).to_dict()
        
        results["strategy_params"] = ParameterValidator.validate_strategy_params(
            self.strategy,
            self.strategy._params,
        ).to_dict()
        
        results["backtest_config"] = ParameterValidator.validate_backtest_config(
            self.config
        ).to_dict()
        
        results["reproducibility"] = BacktestConsistencyChecker.check_reproducibility(
            self.strategy, self.config, data
        ).to_dict()
        
        results["cross_validation"] = BacktestConsistencyChecker.check_cross_validation(
            self.strategy, self.config, data
        ).to_dict()
        
        results["boundary_conditions"] = BacktestConsistencyChecker.check_boundary_conditions(
            self.strategy, self.config, data
        ).to_dict()
        
        overall_valid = all(
            r.get("is_valid", r.get("is_consistent", True))
            for r in results.values()
        )
        
        results["overall_valid"] = overall_valid
        
        return results
    
    def generate_validation_report(self, data: pd.DataFrame) -> str:
        """生成验证报告"""
        results = self.validate_all(data)
        
        report_lines = [
            "=" * 60,
            "策略验证报告",
            "=" * 60,
            f"策略: {self.strategy.name}",
            f"交易对: {self.config.symbol}",
            f"周期: {self.config.interval}",
            f"验证时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
            "",
        ]
        
        report_lines.append("-" * 40)
        report_lines.append("风险参数验证")
        report_lines.append("-" * 40)
        rp = results["risk_params"]
        report_lines.append(f"状态: {'✅ 通过' if rp['is_valid'] else '❌ 失败'}")
        for error in rp.get("errors", []):
            report_lines.append(f"  ❌ {error}")
        for warning in rp.get("warnings", []):
            report_lines.append(f"  ⚠️ {warning}")
        
        report_lines.append("")
        report_lines.append("-" * 40)
        report_lines.append("策略参数验证")
        report_lines.append("-" * 40)
        sp = results["strategy_params"]
        report_lines.append(f"状态: {'✅ 通过' if sp['is_valid'] else '❌ 失败'}")
        for error in sp.get("errors", []):
            report_lines.append(f"  ❌ {error}")
        
        report_lines.append("")
        report_lines.append("-" * 40)
        report_lines.append("可重复性检查")
        report_lines.append("-" * 40)
        rep = results["reproducibility"]
        report_lines.append(f"状态: {'✅ 一致' if rep['is_consistent'] else '❌ 不一致'}")
        for issue in rep.get("issues", []):
            report_lines.append(f"  ⚠️ {issue}")
        
        report_lines.append("")
        report_lines.append("-" * 40)
        report_lines.append("交叉验证")
        report_lines.append("-" * 40)
        cv = results["cross_validation"]
        report_lines.append(f"状态: {'✅ 通过' if cv['is_consistent'] else '⚠️ 注意'}")
        for issue in cv.get("issues", []):
            report_lines.append(f"  ⚠️ {issue}")
        
        report_lines.append("")
        report_lines.append("-" * 40)
        report_lines.append("边界条件检查")
        report_lines.append("-" * 40)
        bc = results["boundary_conditions"]
        report_lines.append(f"状态: {'✅ 通过' if bc['is_consistent'] else '⚠️ 注意'}")
        for issue in bc.get("issues", []):
            report_lines.append(f"  ⚠️ {issue}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append(f"总体评估: {'✅ 验证通过' if results['overall_valid'] else '❌ 验证失败'}")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
