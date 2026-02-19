"""
量化交易系统 - 专业回测界面

功能：
- 多交易对、多周期选择
- 策略参数动态配置
- 止损止盈、杠杆设置
- 风险提示机制
- 回测可视化
"""

import sys
import traceback
import webbrowser
import os
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QUrl
from PyQt5.QtGui import QColor, QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QPushButton, QTextEdit,
    QSpinBox, QDoubleSpinBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QMessageBox,
    QProgressBar, QFrame, QGridLayout, QScrollArea, QSizePolicy,
    QFileDialog, QRadioButton, QLineEdit, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QFont, QColor, QPalette, QPainter
from PyQt5.QtCore import Qt as QtCore

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LiveAccountWorker(QThread):
    """实盘账户信息后台刷新线程"""
    account_updated = pyqtSignal(dict)
    positions_updated = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, trader, interval_ms: int = 3000):
        super().__init__()
        self._trader = trader
        self._interval_ms = interval_ms
        self._running = False
        self._mutex = None
    
    def run(self):
        self._running = True
        while self._running and not self.isInterruptionRequested():
            try:
                if self._trader and hasattr(self._trader, '_update_positions'):
                    self._trader._update_positions()
                    
                    stats = self._trader.get_statistics()
                    self.account_updated.emit(stats)
                    
                    positions = []
                    for symbol, pos in self._trader.positions.items():
                        positions.append({
                            'symbol': symbol,
                            'side': pos.side,
                            'quantity': pos.quantity,
                            'entry_price': pos.entry_price,
                            'current_price': pos.current_price if hasattr(pos, 'current_price') else pos.entry_price,
                            'unrealized_pnl': pos.unrealized_pnl,
                            'leverage': pos.leverage if hasattr(pos, 'leverage') else 1,
                        })
                    self.positions_updated.emit(positions)
                    
            except Exception as e:
                self.error_occurred.emit(str(e))
            
            self.msleep(self._interval_ms)
    
    def stop(self, timeout_ms: int = 1500) -> bool:
        self._running = False
        self.requestInterruption()

        if self.currentThread() == self:
            return False

        if self.isRunning():
            return self.wait(timeout_ms)
        return True


class EnhancedBacktestWorker(QThread):
    """强化回测后台工作线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    
    def __init__(self, config: dict):
        super().__init__()
        self._config = config
    
    def run(self):
        try:
            from backtest.enhanced_backtester import EnhancedBacktester
            from Strategy.templates import get_strategy
            
            strategy_name = self._config.get('strategy_name', 'MACDStrategy')
            strategy_params = self._config.get('strategy_params', {})
            data = self._config.get('data')
            backtest_config = self._config.get('backtest_config')
            
            if data is None or backtest_config is None:
                self.finished.emit({'error': '缺少必要数据'})
                return
            
            strategy = get_strategy(strategy_name, strategy_params)
            backtester = EnhancedBacktester(type(strategy), backtest_config)
            
            results = {}
            
            if self._config.get('run_scenarios'):
                self.progress.emit(f"[{datetime.now():%H:%M:%S}] 运行市场情景测试...")
                try:
                    results['scenarios'] = backtester.run_all_scenarios(data, strategy_params)
                except Exception as e:
                    results['scenarios_error'] = str(e)
            
            if self._config.get('run_monte_carlo'):
                n_sim = self._config.get('n_simulations', 30)
                self.progress.emit(f"[{datetime.now():%H:%M:%S}] 运行蒙特卡洛模拟 ({n_sim}次)...")
                try:
                    results['monte_carlo'] = backtester.run_monte_carlo(data, n_simulations=n_sim, strategy_params=strategy_params)
                except Exception as e:
                    results['monte_carlo_error'] = str(e)
            
            if self._config.get('run_walk_forward'):
                self.progress.emit(f"[{datetime.now():%H:%M:%S}] 运行滚动窗口测试...")
                try:
                    results['walk_forward'] = backtester.run_walk_forward(data, strategy_params=strategy_params)
                except Exception as e:
                    results['walk_forward_error'] = str(e)
            
            self.finished.emit(results)
            
        except Exception as e:
            import traceback
            self.finished.emit({'error': f"{str(e)}\n{traceback.format_exc()}"})


class BacktestWorker(QThread):
    progress = pyqtSignal(str)
    trade_log = pyqtSignal(str)  # 交易日志
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
    
    def run(self):
        try:
            symbol = self.config["symbol"]
            interval = self.config["interval"]
            data_num = self.config["data_num"]
            strategy_name = self.config["strategy"]
            strategy_params = self.config["strategy_params"]
            initial_capital = self.config["initial_capital"]
            leverage = self.config.get("leverage", 5)
            stop_loss_pct = self.config.get("stop_loss_pct", 0.0)
            take_profit_pct = self.config.get("take_profit_pct", 0.0)
            position_size = self.config.get("position_size", 0.1)
            
            self.progress.emit(f"📊 初始化回测环境...")
            self.progress.emit(f"   交易对: {symbol}")
            self.progress.emit(f"   周期: {interval}")
            self.progress.emit(f"   数据量: {data_num}")
            self.progress.emit(f"   策略: {strategy_name}")
            self.progress.emit(f"   杠杆: {leverage}x")
            self.progress.emit(f"   仓位: {position_size*100:.0f}%")
            if stop_loss_pct > 0:
                self.progress.emit(f"   止损: {stop_loss_pct}%")
            if take_profit_pct > 0:
                self.progress.emit(f"   止盈: {take_profit_pct}%")
            
            from Data.data_service import get_data_service, DataServiceConfig, RegionRestrictedError, DataSourceError
            from Strategy.templates import get_strategy
            from backtest.engine import BacktestEngine
            from backtest.report import BacktestReport
            from backtest.visualization import BacktestVisualizer
            from core.config import BacktestConfig
            
            self.progress.emit(f"\n📥 加载 {symbol} 数据...")
            
            service = get_data_service(DataServiceConfig(prefer_database=True, auto_init=True))
            
            try:
                data = service.get_backtest_data(symbol, interval, data_num)
            except RegionRestrictedError as e:
                self.error.emit(f"API访问受限，请配置代理:\n{str(e)}")
                return
            except DataSourceError as e:
                self.error.emit(f"数据源错误:\n{str(e)}")
                return
            
            if data.empty:
                self.error.emit(f"无法获取 {symbol} 数据\n请检查网络连接或交易对是否正确")
                return
            
            self.progress.emit(f"   数据条数: {len(data)}")
            self.progress.emit(f"   时间范围: {data.index[0].strftime('%Y-%m-%d %H:%M')} ~ {data.index[-1].strftime('%Y-%m-%d %H:%M')}")
            self.progress.emit(f"   价格范围: {data['low'].min():.2f} ~ {data['high'].max():.2f}")
            
            strategy = get_strategy(strategy_name, strategy_params)
            self.progress.emit(f"\n⚙️ 策略参数:")
            for name, value in strategy_params.items():
                self.progress.emit(f"   {name}: {value}")
            
            config = BacktestConfig(
                symbol=symbol,
                interval=interval,
                initial_capital=initial_capital,
                data_limit=data_num,
                leverage=leverage,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                position_size=position_size,
            )
            
            self.progress.emit(f"\n🚀 开始回测...")
            
            engine = BacktestEngine(strategy, config)
            result = engine.run(data)
            
            self.progress.emit(f"\n📈 回测完成!")
            self.progress.emit(f"   总交易次数: {result.total_trades}")
            self.progress.emit(f"   盈利交易: {result.winning_trades}")
            self.progress.emit(f"   亏损交易: {result.losing_trades}")
            self.progress.emit(f"   胜率: {result.win_rate:.2f}%")
            self.progress.emit(f"   总收益率: {result.total_return_pct:.2f}%")
            self.progress.emit(f"   最大回撤: {result.max_drawdown_pct:.2f}%")
            self.progress.emit(f"   夏普比率: {result.sharpe_ratio:.2f}")
            
            if result.stop_loss_hits > 0:
                self.progress.emit(f"   止损触发: {result.stop_loss_hits}次")
            if result.take_profit_hits > 0:
                self.progress.emit(f"   止盈触发: {result.take_profit_hits}次")
            if result.liquidation_hits > 0:
                self.progress.emit(f"   ⚠️ 爆仓: {result.liquidation_hits}次")
            
            if result.completed_trades:
                self.trade_log.emit("\n📋 交易记录:")
                self.trade_log.emit("-" * 80)
                for i, trade in enumerate(result.completed_trades[:20]):
                    entry_time = trade.entry_time.strftime('%m-%d %H:%M')
                    exit_time = trade.exit_time.strftime('%m-%d %H:%M')
                    pnl_sign = "+" if trade.pnl >= 0 else ""
                    exit_type = trade.exit_type if hasattr(trade, 'exit_type') else "signal"
                    self.trade_log.emit(
                        f"  #{i+1:2d} | {entry_time} -> {exit_time} | "
                        f"{trade.entry_price:.2f} -> {trade.exit_price:.2f} | "
                        f"PnL: {pnl_sign}{trade.pnl:.2f} ({pnl_sign}{trade.pnl/trade.entry_price*100:.2f}%) | {exit_type}"
                    )
                if len(result.completed_trades) > 20:
                    self.trade_log.emit(f"  ... 共 {len(result.completed_trades)} 笔交易")
            
            report = BacktestReport(result, strategy_name, symbol, interval)
            visualizer = BacktestVisualizer()
            
            self.finished.emit({
                "result": result, 
                "report": report,
                "data": data,
                "visualizer": visualizer,
                "config": config,
            })
            
        except Exception as e:
            import traceback
            self.error.emit(f"回测失败: {str(e)}\n{traceback.format_exc()}")


class OptimizerWorker(QThread):
    """参数优化工作线程"""
    progress = pyqtSignal(int, int)  # current, total
    log_message = pyqtSignal(str)  # 实时日志
    iteration_log = pyqtSignal(str)  # 迭代详情日志
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._best_score = float('-inf')
        self._best_params = {}
        self._iteration_count = 0
        self._no_trade_count = 0
        self._optimizer = None
        self._composite_optimizer = None
        self._stopped = False
    
    def stop(self):
        """停止优化"""
        self._stopped = True
        if self._composite_optimizer:
            self._composite_optimizer.stop()
        if self._optimizer:
            self._optimizer.stop()
    
    def run(self):
        try:
            from Strategy.parameter_optimizer import (
                ParameterOptimizer,
                ParameterRange,
                get_all_optimizable_params,
                OptimizationResult,
            )
            from Strategy.templates import get_strategy
            from Data.data_service import get_data_service, DataServiceConfig
            from core.config import BacktestConfig
            
            strategy_name = self.config["strategy"]
            self.log_message.emit(f"📊 初始化参数探索...")
            self.log_message.emit(f"   策略: {strategy_name}")
            self.log_message.emit(f"   交易对: {self.config['symbol']}")
            self.log_message.emit(f"   周期: {self.config['interval']}")
            self.log_message.emit(f"   数据量: {self.config['data_num']}")
            self.log_message.emit(f"   优化方法: {self.config['opt_method']}")
            self.log_message.emit(f"   迭代次数: {self.config['iterations']}")
            self.log_message.emit(f"   优化目标: {self.config['optimization_metric']}")
            
            strategy = get_strategy(strategy_name)
            
            self.log_message.emit(f"\n📥 加载数据...")
            service = get_data_service(DataServiceConfig(prefer_database=True, auto_init=True))
            data = service.get_klines(
                self.config["symbol"],
                self.config["interval"],
                self.config["data_num"]
            )
            
            if data.empty:
                self.error.emit("无法获取数据")
                return
            
            data_count = len(data)
            self.log_message.emit(f"   数据条数: {data_count}")
            self.log_message.emit(f"   时间范围: {data.index[0].strftime('%Y-%m-%d %H:%M')} ~ {data.index[-1].strftime('%Y-%m-%d %H:%M')}")
            
            if data_count < 5000:
                self.log_message.emit(f"   ⚠️ 数据量不足5000条，可能影响优化可靠性")
            
            config = BacktestConfig(
                symbol=self.config["symbol"],
                interval=self.config["interval"],
                initial_capital=self.config["initial_capital"],
                leverage=self.config.get("leverage", 5),
                stop_loss_pct=self.config.get("stop_loss_pct", 0.0),
                take_profit_pct=self.config.get("take_profit_pct", 0.0),
            )
            
            optimizer = ParameterOptimizer(
                strategy_class=type(strategy),
                data=data,
                base_config=config,
                optimization_metric=self.config.get("optimization_metric", "sharpe_ratio"),
            )
            
            self._optimizer = optimizer
            
            param_ranges = self.config.get("param_ranges", [])
            
            if not param_ranges:
                all_params = get_all_optimizable_params(strategy, include_risk_params=True)
                param_ranges = all_params["strategy"] + all_params["risk"]
            
            self.log_message.emit("┌──────────────────────────────────────┐")
            self.log_message.emit("│ 📊 参数范围                          │")
            self.log_message.emit("├──────────────────────────────────────┤")
            for pr in param_ranges:
                if pr.values:
                    val_str = str(pr.values)[:25]
                    self.log_message.emit(f"│ {pr.name:12} = {val_str:<20}│")
                else:
                    range_str = f"[{pr.min_value}, {pr.max_value}] 步长{pr.step}"
                    self.log_message.emit(f"│ {pr.name:12} {range_str:<22}│")
            self.log_message.emit("└──────────────────────────────────────┘")
            
            opt_method = self.config.get("opt_method", "随机搜索")
            iterations = self.config.get("iterations", 50)
            
            if opt_method == "网格搜索":
                total_combinations = 1
                for pr in param_ranges:
                    total_combinations *= len(pr.get_values())
                warn = " ⚠️过多" if total_combinations > 10000 else ""
                self.log_message.emit(f"📊 组合数: {total_combinations}{warn}")
            
            self.log_message.emit(f"🚀 {opt_method} × {iterations}次 {'─'*20}")
            
            self._iteration_count = 0
            self._no_trade_count = 0
            
            def progress_callback(current, total):
                self.progress.emit(current, total)
            
            def result_callback(result):
                self._iteration_count += 1
                
                if result["total_trades"] == 0:
                    self._no_trade_count += 1
                
                if result["score"] > self._best_score:
                    self._best_score = result["score"]
                    self._best_params = result["params"]
                    
                    pnl_sign = "+" if result["total_return_pct"] >= 0 else ""
                    self.iteration_log.emit(
                        f"✅#{self._iteration_count:3d} 得分{result['score']:.3f} │ "
                        f"收益{pnl_sign}{result['total_return_pct']:.1f}% │ "
                        f"回撤{result['max_drawdown_pct']:.1f}%"
                    )
                elif self._iteration_count % max(1, iterations // 5) == 0:
                    self.log_message.emit(f"⏳ {self._iteration_count}/{iterations} 最优{self._best_score:.3f}")
            
            optimizer._on_result = result_callback
            
            if opt_method == "网格搜索":
                result = optimizer.grid_search(
                    param_ranges=param_ranges,
                    max_iterations=iterations,
                    progress_callback=progress_callback,
                    result_callback=result_callback,
                )
            elif opt_method == "随机搜索":
                result = optimizer.random_search(
                    param_ranges=param_ranges,
                    n_iterations=iterations,
                    progress_callback=progress_callback,
                    result_callback=result_callback,
                )
            elif opt_method == "遗传算法":
                result = optimizer.genetic_algorithm(
                    param_ranges=param_ranges,
                    n_generations=max(5, iterations // 20),
                    population_size=20,
                    progress_callback=progress_callback,
                    result_callback=result_callback,
                )
            elif opt_method == "模拟退火":
                result = optimizer.simulated_annealing(
                    param_ranges=param_ranges,
                    n_iterations=iterations,
                    progress_callback=progress_callback,
                    result_callback=result_callback,
                )
            elif opt_method == "粒子群优化":
                result = optimizer.particle_swarm_optimization(
                    param_ranges=param_ranges,
                    n_particles=20,
                    n_iterations=max(5, iterations // 20),
                    progress_callback=progress_callback,
                    result_callback=result_callback,
                )
            elif opt_method == "强化学习":
                result = optimizer.reinforcement_learning_optimize(
                    param_ranges=param_ranges,
                    n_episodes=iterations,
                    progress_callback=progress_callback,
                    result_callback=result_callback,
                )
            elif opt_method == "复合优化":
                from Strategy.composite_param_optimizer import CompositeParameterOptimizer
                
                self.log_message.emit(f"\n🏆 复合优化: 组合多种算法")
                
                composite_optimizer = CompositeParameterOptimizer(
                    strategy_class=type(strategy),
                    data=data,
                    base_config=config,
                    optimization_metric=self.config.get("optimization_metric", "sharpe_ratio"),
                )
                self._composite_optimizer = composite_optimizer
                
                def composite_progress(algo_name, current, total):
                    self.log_message.emit(f"  [{algo_name}] {current}/{total}")
                    self.progress.emit(current, total)
                
                def composite_result(algo_result):
                    self.log_message.emit(
                        f"  ✅ {algo_result.algorithm_name}: 得分{algo_result.best_score:.3f} "
                        f"耗时{algo_result.execution_time:.1f}s"
                    )
                
                composite_result_obj = composite_optimizer.run_composite(
                    param_ranges=param_ranges,
                    iterations_per_algorithm=max(10, iterations // 6),
                    progress_callback=composite_progress,
                    result_callback=composite_result,
                )
                
                ranking = composite_result_obj.get_ranking()
                self.log_message.emit(f"\n📊 算法排名:")
                for i, (name, score) in enumerate(ranking, 1):
                    self.log_message.emit(f"  {i}. {name}: {score:.3f}")
                
                result = OptimizationResult(
                    best_params=composite_result_obj.best_params,
                    best_score=composite_result_obj.best_score,
                    best_result=None,
                    all_results=[],
                    optimization_method="composite",
                    total_iterations=sum(r.iterations for r in composite_result_obj.all_results),
                    execution_time=composite_result_obj.execution_time,
                    parameter_importance={},
                    convergence_data=[],
                )
            else:
                result = optimizer.bayesian_optimization(
                    param_ranges=param_ranges,
                    n_iterations=iterations,
                    progress_callback=progress_callback,
                    result_callback=result_callback,
                )
            
            if result.parameter_importance:
                self.log_message.emit("📊 重要性: " + " ".join(
                    f"{p}={v:.2f}" for p, v in 
                    sorted(result.parameter_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                ))
            
            self.log_message.emit(f"✅ 完成 得分{result.best_score:.3f} 耗时{result.execution_time:.1f}s")
            self.finished.emit(result)
            
        except Exception as e:
            import traceback
            self.error.emit(f"优化失败: {str(e)}\n{traceback.format_exc()}")


class MetricCard(QFrame):
    """指标卡片"""
    
    def __init__(self, title: str, unit: str = "", parent=None):
        super().__init__(parent)
        self.unit = unit
        self.setFixedHeight(80)
        self.setStyleSheet("""
            QFrame {
                background-color: #1e222d;
                border-radius: 8px;
                border: 1px solid #2a2e39;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #848e9c; font-size: 12px; border: none;")
        layout.addWidget(self.title_label)
        
        self.value_label = QLabel("-")
        self.value_label.setStyleSheet("color: #eaecef; font-size: 22px; font-weight: bold; border: none;")
        layout.addWidget(self.value_label)
    
    def set_value(self, value: float, is_positive: bool = None):
        if isinstance(value, (int, float)):
            text = f"{value:.2f}{self.unit}"
        else:
            text = str(value)
        
        self.value_label.setText(text)
        
        if is_positive is not None:
            color = "#0ecb81" if is_positive else "#f6465d"
            self.value_label.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold; border: none;")


class TradingUI(QMainWindow):
    """专业交易界面"""
    
    STRATEGIES = {
        "MACD趋势策略": "MACDStrategy",
        "趋势跟踪策略": "TrendFollowingStrategy",
        "均值回归策略": "MeanReversionStrategy",
        "布林带策略": "BollingerBandsStrategy",
        "多指标组合策略": "MultiIndicatorStrategy",
        "自适应多指标策略": "AdaptiveMultiIndicatorStrategy",
    }
    
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"]
    
    INTERVALS = ["1min", "5min", "15min", "30min", "1h", "4h", "1d"]
    
    INDICATOR_OPTIONS = ["MACD", "RSI", "BollingerBands", "MA"]
    
    def __init__(self):
        super().__init__()
        self._worker = None
        self._optimizer_worker = None
        self._last_result = None
        self._last_data = None
        self._last_visualizer = None
        self._last_config = None
        self._selected_indicators = ["MACD", "MA"]
        self._init_ui()
    
    def _init_ui(self):
        self.setWindowTitle("量化交易系统 v2.0")
        self.setGeometry(80, 80, 1500, 900)
        self.setStyleSheet(self._stylesheet())
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        layout.addWidget(self._create_header())
        
        self.main_tabs = QTabWidget()
        self.main_tabs.addTab(self._create_backtest_tab(), "📊 回测")
        self.main_tabs.addTab(self._create_optimizer_tab(), "🔍 参数探索")
        self.main_tabs.addTab(self._create_live_trading_tab(), "💹 实盘交易")
        self.main_tabs.addTab(self._create_equity_tab(), "📈 资产曲线")
        self.main_tabs.addTab(self._create_version_tab(), "📝 版本更新")
        layout.addWidget(self.main_tabs, 1)
        
        self._show_env_load_result()
    
    def _show_env_load_result(self) -> None:
        """显示API密钥加载结果"""
        if hasattr(self, '_env_loaded') and self._env_loaded:
            if self.api_key.text() or self.api_secret.text():
                self.live_log.append(f"[{datetime.now():%H:%M:%S}] 📋 已从.env加载API密钥")
        elif hasattr(self, '_env_load_error'):
            self.live_log.append(f"[{datetime.now():%H:%M:%S}] ⚠️ 加载.env失败: {self._env_load_error}")
    
    def _create_backtest_tab(self) -> QWidget:
        """创建回测标签页"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        layout.addWidget(self._create_backtest_config_panel(), 1)
        layout.addWidget(self._create_result_panel(), 2)
        
        return widget
    
    def _create_optimizer_tab(self) -> QWidget:
        """创建参数探索标签页"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        layout.addWidget(self._create_optimizer_config_panel(), 1)
        
        return widget
    
    def _create_live_trading_tab(self) -> QWidget:
        """创建实盘交易标签页 - 专业级界面"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        top_layout = QHBoxLayout()
        top_layout.addWidget(self._create_live_account_panel(), 1)
        top_layout.addWidget(self._create_live_risk_panel(), 1)
        top_layout.addWidget(self._create_live_leverage_panel(), 1)
        layout.addLayout(top_layout)
        
        mid_layout = QHBoxLayout()
        mid_layout.addWidget(self._create_live_strategy_panel(), 1)
        mid_layout.addWidget(self._create_live_control_panel(), 1)
        layout.addLayout(mid_layout)
        
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self._create_live_position_panel(), 2)
        bottom_layout.addWidget(self._create_live_order_panel(), 1)
        layout.addLayout(bottom_layout)
        
        layout.addWidget(self._create_live_log_panel(), 1)
        
        return widget
    
    def _create_live_account_panel(self) -> QWidget:
        """创建账户信息面板"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame { background-color: #1e222d; border-radius: 8px; }
            QLabel { color: #eaecef; }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        title = QLabel("💰 账户信息")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f0b90b;")
        layout.addWidget(title)
        
        self.account_grid = QGridLayout()
        self.account_grid.setSpacing(4)
        
        self._account_labels = {}
        account_items = [
            ("total_balance", "总权益", "USDT"),
            ("available", "可用余额", "USDT"),
            ("unrealized_pnl", "未实现盈亏", "USDT"),
            ("realized_pnl", "已实现盈亏", "USDT"),
            ("margin_used", "已用保证金", "USDT"),
            ("margin_ratio", "保证金率", "%"),
        ]
        
        for i, (key, label, unit) in enumerate(account_items):
            label_widget = QLabel(label)
            label_widget.setStyleSheet("color: #848e9c; font-size: 12px;")
            self.account_grid.addWidget(label_widget, i // 2, (i % 2) * 2)
            
            value_widget = QLabel(f"0 {unit}")
            value_widget.setStyleSheet("color: #eaecef; font-size: 12px; font-weight: bold;")
            self._account_labels[key] = (value_widget, unit)
            self.account_grid.addWidget(value_widget, i // 2, (i % 2) * 2 + 1)
        
        layout.addLayout(self.account_grid)
        return panel
    
    def _create_live_risk_panel(self) -> QWidget:
        """创建风险控制面板"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame { background-color: #1e222d; border-radius: 8px; }
            QLabel { color: #eaecef; }
            QDoubleSpinBox, QSpinBox { 
                background-color: #0b0e11; 
                border: 1px solid #2a2e39; 
                border-radius: 4px; 
                padding: 4px;
                color: #eaecef;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        title = QLabel("🛡️ 风险控制")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f0b90b;")
        layout.addWidget(title)
        
        risk_layout = QGridLayout()
        risk_layout.setSpacing(4)
        
        risk_layout.addWidget(QLabel("止损%"), 0, 0)
        self.live_stop_loss = QDoubleSpinBox()
        self.live_stop_loss.setRange(0, 50)
        self.live_stop_loss.setValue(5)
        self.live_stop_loss.setSingleStep(0.5)
        risk_layout.addWidget(self.live_stop_loss, 0, 1)
        
        risk_layout.addWidget(QLabel("止盈%"), 0, 2)
        self.live_take_profit = QDoubleSpinBox()
        self.live_take_profit.setRange(0, 100)
        self.live_take_profit.setValue(10)
        self.live_take_profit.setSingleStep(1)
        risk_layout.addWidget(self.live_take_profit, 0, 3)
        
        risk_layout.addWidget(QLabel("仓位%"), 1, 0)
        self.live_position_size = QDoubleSpinBox()
        self.live_position_size.setRange(1, 100)
        self.live_position_size.setValue(10)
        risk_layout.addWidget(self.live_position_size, 1, 1)
        
        risk_layout.addWidget(QLabel("最大日交易"), 1, 2)
        self.live_max_trades = QSpinBox()
        self.live_max_trades.setRange(1, 100)
        self.live_max_trades.setValue(10)
        risk_layout.addWidget(self.live_max_trades, 1, 3)
        
        risk_layout.addWidget(QLabel("日亏损限额%"), 2, 0)
        self.live_max_daily_loss = QDoubleSpinBox()
        self.live_max_daily_loss.setRange(1, 100)
        self.live_max_daily_loss.setValue(20)
        risk_layout.addWidget(self.live_max_daily_loss, 2, 1)
        
        risk_layout.addWidget(QLabel("最大持仓数"), 2, 2)
        self.live_max_positions = QSpinBox()
        self.live_max_positions.setRange(1, 20)
        self.live_max_positions.setValue(3)
        risk_layout.addWidget(self.live_max_positions, 2, 3)
        
        layout.addLayout(risk_layout)
        return panel
    
    def _create_live_leverage_panel(self) -> QWidget:
        """创建杠杆信息面板"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame { background-color: #1e222d; border-radius: 8px; }
            QLabel { color: #eaecef; }
            QSpinBox { 
                background-color: #0b0e11; 
                border: 1px solid #2a2e39; 
                border-radius: 4px; 
                padding: 4px;
                color: #eaecef;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        title = QLabel("⚡ 杠杆设置")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f0b90b;")
        layout.addWidget(title)
        
        leverage_layout = QGridLayout()
        leverage_layout.setSpacing(4)
        
        leverage_layout.addWidget(QLabel("杠杆倍数"), 0, 0)
        self.live_leverage = QSpinBox()
        self.live_leverage.setRange(1, 125)
        self.live_leverage.setValue(5)
        leverage_layout.addWidget(self.live_leverage, 0, 1)
        
        self._leverage_warning = QLabel("")
        self._leverage_warning.setStyleSheet("color: #f6465d; font-size: 11px;")
        leverage_layout.addWidget(self._leverage_warning, 1, 0, 1, 2)
        
        self.live_leverage.valueChanged.connect(self._on_leverage_changed)
        
        self._leverage_labels = {}
        leverage_items = [
            ("used_margin", "已用保证金", "USDT"),
            ("available_margin", "可用保证金", "USDT"),
            ("max_position", "最大仓位", "USDT"),
        ]
        
        for i, (key, label, unit) in enumerate(leverage_items):
            label_widget = QLabel(label)
            label_widget.setStyleSheet("color: #848e9c; font-size: 11px;")
            leverage_layout.addWidget(label_widget, i + 2, 0)
            
            value_widget = QLabel(f"0 {unit}")
            value_widget.setStyleSheet("color: #eaecef; font-size: 11px;")
            self._leverage_labels[key] = (value_widget, unit)
            leverage_layout.addWidget(value_widget, i + 2, 1)
        
        layout.addLayout(leverage_layout)
        return panel
    
    def _on_leverage_changed(self, value: int):
        """杠杆变化时的警告"""
        if value > 20:
            self._leverage_warning.setText("⚠️ 高杠杆风险极大！")
        elif value > 10:
            self._leverage_warning.setText("⚠️ 杠杆较高，注意风险")
        else:
            self._leverage_warning.setText("")
    
    def _create_live_strategy_panel(self) -> QWidget:
        """创建策略配置面板"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame { background-color: #1e222d; border-radius: 8px; }
            QLabel { color: #eaecef; }
            QComboBox { 
                background-color: #0b0e11; 
                border: 1px solid #2a2e39; 
                border-radius: 4px; 
                padding: 4px;
                color: #eaecef;
            }
            QLineEdit { 
                background-color: #0b0e11; 
                border: 1px solid #2a2e39; 
                border-radius: 4px; 
                padding: 4px;
                color: #eaecef;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        title = QLabel("📊 策略配置")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f0b90b;")
        layout.addWidget(title)
        
        strategy_layout = QGridLayout()
        strategy_layout.setSpacing(4)
        
        strategy_layout.addWidget(QLabel("交易对"), 0, 0)
        self.live_symbol = QComboBox()
        self.live_symbol.addItems(self.SYMBOLS[:10])
        strategy_layout.addWidget(self.live_symbol, 0, 1)
        
        strategy_layout.addWidget(QLabel("策略"), 0, 2)
        self.live_strategy = QComboBox()
        self.live_strategy.addItems(list(self.STRATEGIES.keys()))
        strategy_layout.addWidget(self.live_strategy, 0, 3)
        
        strategy_layout.addWidget(QLabel("交易模式"), 1, 0)
        self.live_trade_mode = QComboBox()
        self.live_trade_mode.addItems(["long_only", "short_only", "both"])
        strategy_layout.addWidget(self.live_trade_mode, 1, 1)
        
        strategy_layout.addWidget(QLabel("API Key"), 2, 0)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("输入API Key")
        strategy_layout.addWidget(self.api_key, 2, 1, 1, 2)
        
        self.show_api_btn = QPushButton("👁")
        self.show_api_btn.setFixedWidth(30)
        self.show_api_btn.clicked.connect(self._toggle_api_visibility)
        strategy_layout.addWidget(self.show_api_btn, 2, 3)
        
        strategy_layout.addWidget(QLabel("API Secret"), 3, 0)
        self.api_secret = QLineEdit()
        self.api_secret.setEchoMode(QLineEdit.Password)
        self.api_secret.setPlaceholderText("输入API Secret")
        strategy_layout.addWidget(self.api_secret, 3, 1, 1, 2)
        
        self.show_secret_btn = QPushButton("👁")
        self.show_secret_btn.setFixedWidth(30)
        self.show_secret_btn.clicked.connect(self._toggle_secret_visibility)
        strategy_layout.addWidget(self.show_secret_btn, 3, 3)
        
        mode_layout = QHBoxLayout()
        self.live_mode_test = QRadioButton("测试网")
        self.live_mode_test.setChecked(True)
        self.live_mode_live = QRadioButton("实盘")
        mode_layout.addWidget(self.live_mode_test)
        mode_layout.addWidget(self.live_mode_live)
        mode_layout.addStretch()
        strategy_layout.addLayout(mode_layout, 4, 0, 1, 4)
        
        self._load_api_from_env()
        self.live_mode_live.toggled.connect(self._on_mode_changed)
        
        layout.addLayout(strategy_layout)
        return panel
    
    def _create_live_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame { background-color: #1e222d; border-radius: 8px; }
            QLabel { color: #eaecef; }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        title = QLabel("🎮 交易控制")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f0b90b;")
        layout.addWidget(title)
        
        self._status_label = QLabel("● 未连接")
        self._status_label.setStyleSheet("color: #848e9c; font-size: 12px;")
        layout.addWidget(self._status_label)
        
        btn_layout = QGridLayout()
        
        self.connect_btn = QPushButton("🔗 连接")
        self.connect_btn.clicked.connect(self._connect_exchange)
        btn_layout.addWidget(self.connect_btn, 0, 0)
        
        self.start_live_btn = QPushButton("▶ 启动")
        self.start_live_btn.setObjectName("primary")
        self.start_live_btn.clicked.connect(self._start_live_trading)
        self.start_live_btn.setEnabled(False)
        btn_layout.addWidget(self.start_live_btn, 0, 1)
        
        self.stop_live_btn = QPushButton("⏹ 停止")
        self.stop_live_btn.clicked.connect(self._stop_live_trading)
        self.stop_live_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_live_btn, 1, 0, 1, 2)
        
        layout.addLayout(btn_layout)
        
        self._daily_stats = QLabel("今日: 0笔交易 | 盈亏: 0 USDT")
        self._daily_stats.setStyleSheet("color: #848e9c; font-size: 11px;")
        layout.addWidget(self._daily_stats)
        
        self._auto_refresh_label = QLabel("自动刷新: 3秒/次")
        self._auto_refresh_label.setStyleSheet("color: #848e9c; font-size: 10px;")
        layout.addWidget(self._auto_refresh_label)
        
        self.strategy_config_btn = QPushButton("⚙️ 策略参数")
        self.strategy_config_btn.clicked.connect(self._show_strategy_config_dialog)
        layout.addWidget(self.strategy_config_btn)
        
        layout.addStretch()
        return panel
    
    def _show_strategy_config_dialog(self):
        """显示策略参数配置弹窗"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QScrollArea
        
        dialog = QDialog(self)
        dialog.setWindowTitle("策略参数配置")
        dialog.setMinimumWidth(450)
        dialog.setMinimumHeight(400)
        dialog.setStyleSheet("""
            QDialog { background-color: #1e222d; }
            QLabel { color: #eaecef; }
            QSpinBox, QDoubleSpinBox, QComboBox { 
                background-color: #0b0e11; 
                border: 1px solid #2a2e39; 
                border-radius: 4px; 
                padding: 4px;
                color: #eaecef;
                min-width: 80px;
            }
            QGroupBox { 
                color: #f0b90b; 
                border: 1px solid #2a2e39; 
                border-radius: 4px; 
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
            QScrollArea { border: none; background-color: transparent; }
        """)
        
        main_layout = QVBoxLayout(dialog)
        
        strategy_group = QGroupBox("策略选择")
        strategy_layout = QVBoxLayout(strategy_group)
        
        strategy_combo = QComboBox()
        strategy_combo.addItems(list(self.STRATEGIES.keys()))
        if hasattr(self, 'live_strategy'):
            idx = strategy_combo.findText(self.live_strategy.currentText())
            if idx >= 0:
                strategy_combo.setCurrentIndex(idx)
        strategy_layout.addWidget(strategy_combo)
        main_layout.addWidget(strategy_group)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        param_group = QGroupBox("策略参数")
        self._dialog_param_layout = QGridLayout(param_group)
        self._dialog_param_widgets = {}
        scroll_layout.addWidget(param_group)
        
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll, 1)
        
        def update_params():
            strategy_name = strategy_combo.currentText()
            strategy_class = self.STRATEGIES.get(strategy_name)
            
            while self._dialog_param_layout.count():
                item = self._dialog_param_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self._dialog_param_widgets.clear()
            
            if strategy_class:
                try:
                    temp_strategy = strategy_class({})
                    param_ranges = temp_strategy.get_param_ranges()
                    
                    for i, pr in enumerate(param_ranges):
                        label = QLabel(pr.name)
                        label.setStyleSheet("color: #eaecef;")
                        self._dialog_param_layout.addWidget(label, i, 0)
                        
                        if pr.values:
                            combo = QComboBox()
                            combo.addItems([str(v) for v in pr.values])
                            self._dialog_param_layout.addWidget(combo, i, 1)
                            self._dialog_param_widgets[pr.name] = combo
                        else:
                            spinbox = QDoubleSpinBox()
                            spinbox.setRange(pr.min_value, pr.max_value)
                            spinbox.setValue(pr.min_value)
                            spinbox.setSingleStep(pr.step if pr.step > 0 else 1)
                            self._dialog_param_layout.addWidget(spinbox, i, 1)
                            self._dialog_param_widgets[pr.name] = spinbox
                        
                        range_label = QLabel(f"[{pr.min_value} ~ {pr.max_value}]")
                        range_label.setStyleSheet("color: #848e9c; font-size: 10px;")
                        self._dialog_param_layout.addWidget(range_label, i, 2)
                        
                except Exception as e:
                    pass
        
        strategy_combo.currentTextChanged.connect(update_params)
        update_params()
        
        risk_group = QGroupBox("风险参数")
        risk_layout = QGridLayout(risk_group)
        
        stop_loss = QDoubleSpinBox()
        stop_loss.setRange(0, 50)
        stop_loss.setValue(self.live_stop_loss.value() if hasattr(self, 'live_stop_loss') else 5)
        risk_layout.addWidget(QLabel("止损%"), 0, 0)
        risk_layout.addWidget(stop_loss, 0, 1)
        
        take_profit = QDoubleSpinBox()
        take_profit.setRange(0, 100)
        take_profit.setValue(self.live_take_profit.value() if hasattr(self, 'live_take_profit') else 10)
        risk_layout.addWidget(QLabel("止盈%"), 0, 2)
        risk_layout.addWidget(take_profit, 0, 3)
        
        main_layout.addWidget(risk_group)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        main_layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            if hasattr(self, 'live_strategy'):
                idx = strategy_combo.findText(strategy_combo.currentText())
                if idx >= 0:
                    self.live_strategy.setCurrentIndex(idx)
            
            if hasattr(self, 'live_stop_loss'):
                self.live_stop_loss.setValue(stop_loss.value())
            if hasattr(self, 'live_take_profit'):
                self.live_take_profit.setValue(take_profit.value())
            
            params_str = ", ".join(f"{k}={v.currentText() if isinstance(v, QComboBox) else v.value()}" 
                                   for k, v in self._dialog_param_widgets.items())
            self.live_log.append(f"[{datetime.now():%H:%M:%S}] ⚙️ 策略参数已更新")
            self.live_log.append(f"   策略: {strategy_combo.currentText()}")
            self.live_log.append(f"   参数: {params_str}")
            self.live_log.append(f"   止损: {stop_loss.value()}% | 止盈: {take_profit.value()}%")
    
    def _create_live_position_panel(self) -> QWidget:
        """创建持仓面板"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame { background-color: #1e222d; border-radius: 8px; }
            QLabel { color: #eaecef; }
            QTableWidget { 
                background-color: #0b0e11; 
                border: none;
                gridline-color: #2a2e39;
            }
            QHeaderView::section { 
                background-color: #1e222d; 
                color: #848e9c; 
                border: none;
                padding: 4px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        header = QHBoxLayout()
        title = QLabel("📈 当前持仓")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f0b90b;")
        header.addWidget(title)
        
        self._position_count = QLabel("0个持仓")
        self._position_count.setStyleSheet("color: #848e9c; font-size: 12px;")
        header.addStretch()
        header.addWidget(self._position_count)
        layout.addLayout(header)
        
        self.position_table = QTableWidget()
        self.position_table.setColumnCount(8)
        self.position_table.setHorizontalHeaderLabels([
            "交易对", "方向", "数量", "入场价", "当前价", "浮动盈亏", "盈亏%", "杠杆"
        ])
        self.position_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.position_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.position_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.position_table.setMaximumHeight(180)
        layout.addWidget(self.position_table)
        
        return panel
    
    def _create_live_order_panel(self) -> QWidget:
        """创建订单面板"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame { background-color: #1e222d; border-radius: 8px; }
            QLabel { color: #eaecef; }
            QTableWidget { 
                background-color: #0b0e11; 
                border: none;
                gridline-color: #2a2e39;
            }
            QHeaderView::section { 
                background-color: #1e222d; 
                color: #848e9c; 
                border: none;
                padding: 4px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        title = QLabel("📋 最近订单")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f0b90b;")
        layout.addWidget(title)
        
        self.order_table = QTableWidget()
        self.order_table.setColumnCount(5)
        self.order_table.setHorizontalHeaderLabels(["时间", "交易对", "方向", "数量", "状态"])
        self.order_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.order_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.order_table.setMaximumHeight(180)
        layout.addWidget(self.order_table)
        
        return panel
    
    def _create_live_equity_chart(self) -> QWidget:
        """创建资产折线图面板"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame { background-color: #1e222d; border-radius: 8px; }
            QLabel { color: #eaecef; }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        header = QHBoxLayout()
        title = QLabel("📈 资产曲线")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f0b90b;")
        header.addWidget(title)
        
        self._total_equity_label = QLabel("总资产: 0 USDT")
        self._total_equity_label.setStyleSheet("color: #0ecb81; font-size: 12px; font-weight: bold;")
        header.addStretch()
        header.addWidget(self._total_equity_label)
        layout.addLayout(header)
        
        self._equity_chart_widget = QWidget()
        self._equity_chart_widget.setMinimumHeight(150)
        self._equity_chart_widget.setStyleSheet("background-color: #0b0e11; border-radius: 4px;")
        
        chart_layout = QVBoxLayout(self._equity_chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        
        self._equity_chart_label = QLabel()
        self._equity_chart_label.setAlignment(Qt.AlignCenter)
        self._equity_chart_label.setStyleSheet("color: #848e9c; font-size: 11px;")
        self._equity_chart_label.setText("连接后显示资产曲线")
        chart_layout.addWidget(self._equity_chart_label)
        
        layout.addWidget(self._equity_chart_widget)
        
        self._equity_history = []
        self._equity_times = []
        
        return panel
    
    def _update_equity_chart(self):
        """更新资产折线图"""
        if not hasattr(self, '_trader') or not self._trader:
            return
        
        from datetime import datetime
        
        stats = self._trader.get_statistics()
        total_equity = stats.get('balance', 0) + stats.get('unrealized_pnl', 0)
        
        self._equity_history.append(total_equity)
        self._equity_times.append(datetime.now())
        
        if len(self._equity_history) > 60:
            self._equity_history = self._equity_history[-60:]
            self._equity_times = self._equity_times[-60:]
        
        if hasattr(self, '_total_equity_label'):
            if total_equity >= 0:
                self._total_equity_label.setText(f"总资产: {total_equity:.2f} USDT")
                self._total_equity_label.setStyleSheet("color: #0ecb81; font-size: 12px; font-weight: bold;")
            else:
                self._total_equity_label.setText(f"总资产: {total_equity:.2f} USDT")
                self._total_equity_label.setStyleSheet("color: #f6465d; font-size: 12px; font-weight: bold;")
        
        if len(self._equity_history) >= 2:
            self._draw_equity_chart()
    
    def _draw_equity_chart(self):
        """绘制资产折线图"""
        try:
            from PyQt5.QtGui import QPainter, QPen, QColor, QFont
            from PyQt5.QtCore import Qt, QRectF
            
            width = self._equity_chart_widget.width()
            height = self._equity_chart_widget.height()
            
            if width < 50 or height < 50:
                return
            
            pixmap = self._equity_chart_widget.grab()
            painter = QPainter()
            
            min_equity = min(self._equity_history)
            max_equity = max(self._equity_history)
            
            if max_equity == min_equity:
                max_equity = min_equity + 1
            
            padding = 20
            chart_width = width - 2 * padding
            chart_height = height - 2 * padding
            
            points = []
            for i, equity in enumerate(self._equity_history):
                x = padding + (i / max(1, len(self._equity_history) - 1)) * chart_width
                y = padding + (1 - (equity - min_equity) / (max_equity - min_equity)) * chart_height
                points.append((x, y))
            
            if len(points) >= 2:
                first_equity = self._equity_history[0]
                last_equity = self._equity_history[-1]
                change_pct = ((last_equity - first_equity) / first_equity * 100) if first_equity != 0 else 0
                
                if change_pct >= 0:
                    color = "#0ecb81"
                    change_text = f"+{change_pct:.2f}%"
                else:
                    color = "#f6465d"
                    change_text = f"{change_pct:.2f}%"
                
                chart_text = f"""
                <div style='text-align: center;'>
                    <p style='color: {color}; font-size: 16px; font-weight: bold; margin: 10px;'>
                        {last_equity:.2f} USDT
                    </p>
                    <p style='color: {color}; font-size: 12px; margin: 5px;'>
                        {change_text}
                    </p>
                    <p style='color: #848e9c; font-size: 10px; margin: 5px;'>
                        最高: {max_equity:.2f} | 最低: {min_equity:.2f}
                    </p>
                </div>
                """
                self._equity_chart_label.setText(chart_text)
                
        except Exception as e:
            pass
    
    def _create_live_log_panel(self) -> QWidget:
        """创建日志面板"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame { background-color: #1e222d; border-radius: 8px; }
            QTextEdit { 
                background-color: #0b0e11; 
                border: none;
                color: #eaecef;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        header = QHBoxLayout()
        title = QLabel("📝 交易日志")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f0b90b;")
        header.addWidget(title)
        header.addStretch()
        
        clear_btn = QPushButton("清空")
        clear_btn.setFixedWidth(50)
        clear_btn.clicked.connect(lambda: self.live_log.clear())
        header.addWidget(clear_btn)
        layout.addLayout(header)
        
        self.live_log = QTextEdit()
        self.live_log.setReadOnly(True)
        self.live_log.setMaximumHeight(120)
        layout.addWidget(self.live_log)
        
        return panel
    
    def _create_live_config_panel(self) -> QWidget:
        """创建实盘配置面板"""
        panel = QFrame()
        panel.setStyleSheet("QFrame { background-color: #1e222d; border-radius: 8px; }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        title = QLabel("💹 实盘交易配置")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #f0b90b;")
        layout.addWidget(title)
        
        mode_group = QGroupBox("交易模式")
        mode_layout = QVBoxLayout(mode_group)
        
        mode_row = QHBoxLayout()
        self.live_mode_test = QRadioButton("测试网")
        self.live_mode_test.setChecked(True)
        self.live_mode_live = QRadioButton("实盘")
        mode_row.addWidget(self.live_mode_test)
        mode_row.addWidget(self.live_mode_live)
        mode_row.addStretch()
        mode_layout.addLayout(mode_row)
        
        layout.addWidget(mode_group)
        
        api_group = QGroupBox("API配置")
        api_layout = QGridLayout(api_group)
        
        api_layout.addWidget(QLabel("API Key"), 0, 0)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("输入API Key")
        api_layout.addWidget(self.api_key, 0, 1)
        
        self.show_api_btn = QPushButton("👁")
        self.show_api_btn.setFixedWidth(30)
        self.show_api_btn.clicked.connect(self._toggle_api_visibility)
        api_layout.addWidget(self.show_api_btn, 0, 2)
        
        api_layout.addWidget(QLabel("API Secret"), 1, 0)
        self.api_secret = QLineEdit()
        self.api_secret.setEchoMode(QLineEdit.Password)
        self.api_secret.setPlaceholderText("输入API Secret")
        api_layout.addWidget(self.api_secret, 1, 1)
        
        self.show_secret_btn = QPushButton("👁")
        self.show_secret_btn.setFixedWidth(30)
        self.show_secret_btn.clicked.connect(self._toggle_secret_visibility)
        api_layout.addWidget(self.show_secret_btn, 1, 2)
        
        self._load_api_from_env()
        
        self.live_mode_live.toggled.connect(self._on_mode_changed)
        
        layout.addWidget(api_group)
        
        trade_group = QGroupBox("交易配置")
        trade_layout = QGridLayout(trade_group)
        
        trade_layout.addWidget(QLabel("交易对"), 0, 0)
        self.live_symbol = QComboBox()
        self.live_symbol.addItems(self.SYMBOLS[:5])
        trade_layout.addWidget(self.live_symbol, 0, 1)
        
        trade_layout.addWidget(QLabel("杠杆"), 0, 2)
        self.live_leverage = QSpinBox()
        self.live_leverage.setRange(1, 125)
        self.live_leverage.setValue(5)
        trade_layout.addWidget(self.live_leverage, 0, 3)
        
        trade_layout.addWidget(QLabel("仓位比例(%)"), 1, 0)
        self.live_position_size = QDoubleSpinBox()
        self.live_position_size.setRange(1, 100)
        self.live_position_size.setValue(10)
        trade_layout.addWidget(self.live_position_size, 1, 1)
        
        trade_layout.addWidget(QLabel("止损(%)"), 1, 2)
        self.live_stop_loss = QDoubleSpinBox()
        self.live_stop_loss.setRange(0, 50)
        self.live_stop_loss.setValue(5)
        trade_layout.addWidget(self.live_stop_loss, 1, 3)
        
        trade_layout.addWidget(QLabel("止盈(%)"), 2, 0)
        self.live_take_profit = QDoubleSpinBox()
        self.live_take_profit.setRange(0, 100)
        self.live_take_profit.setValue(10)
        trade_layout.addWidget(self.live_take_profit, 2, 1)
        
        trade_layout.addWidget(QLabel("最大日交易"), 2, 2)
        self.live_max_trades = QSpinBox()
        self.live_max_trades.setRange(1, 100)
        self.live_max_trades.setValue(10)
        trade_layout.addWidget(self.live_max_trades, 2, 3)
        
        layout.addWidget(trade_group)
        
        strategy_group = QGroupBox("策略选择")
        strategy_layout = QVBoxLayout(strategy_group)
        
        self.live_strategy = QComboBox()
        self.live_strategy.addItems(list(self.STRATEGIES.keys()))
        strategy_layout.addWidget(self.live_strategy)
        
        layout.addWidget(strategy_group)
        
        btn_layout = QHBoxLayout()
        
        self.connect_btn = QPushButton("🔗 连接")
        self.connect_btn.clicked.connect(self._connect_exchange)
        btn_layout.addWidget(self.connect_btn)
        
        self.start_live_btn = QPushButton("▶ 启动交易")
        self.start_live_btn.setObjectName("primary")
        self.start_live_btn.clicked.connect(self._start_live_trading)
        btn_layout.addWidget(self.start_live_btn)
        
        self.stop_live_btn = QPushButton("⏹ 停止交易")
        self.stop_live_btn.clicked.connect(self._stop_live_trading)
        self.stop_live_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_live_btn)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        
        return panel
    
    def _create_live_status_panel(self) -> QWidget:
        """创建实盘状态面板"""
        panel = QFrame()
        panel.setStyleSheet("QFrame { background-color: #1e222d; border-radius: 8px; }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        title = QLabel("📊 交易状态")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #f0b90b;")
        layout.addWidget(title)
        
        status_group = QGroupBox("账户状态")
        status_layout = QGridLayout(status_group)
        
        self.status_labels = {}
        status_items = [
            ("balance", "账户余额", "USDT"),
            ("available", "可用余额", "USDT"),
            ("unrealized_pnl", "未实现盈亏", "USDT"),
            ("realized_pnl", "已实现盈亏", "USDT"),
            ("margin_used", "已用保证金", "USDT"),
            ("daily_trades", "今日交易", "次"),
            ("daily_pnl", "今日盈亏", "USDT"),
        ]
        
        for i, (key, label, unit) in enumerate(status_items):
            status_layout.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            value_label = QLabel(f"0 {unit}")
            value_label.setStyleSheet("color: #eaecef; font-weight: bold;")
            self.status_labels[key] = (value_label, unit)
            status_layout.addWidget(value_label, i // 2, (i % 2) * 2 + 1)
        
        layout.addWidget(status_group)
        
        position_group = QGroupBox("当前持仓")
        position_layout = QVBoxLayout(position_group)
        
        self.position_table = QTableWidget()
        self.position_table.setColumnCount(7)
        self.position_table.setHorizontalHeaderLabels(["交易对", "方向", "数量", "入场价", "当前价", "浮动盈亏", "盈亏%"])
        self.position_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.position_table.setMaximumHeight(150)
        self.position_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.position_table.setSelectionBehavior(QTableWidget.SelectRows)
        position_layout.addWidget(self.position_table)
        
        layout.addWidget(position_group)
        
        log_group = QGroupBox("交易日志")
        log_layout = QVBoxLayout(log_group)
        
        self.live_log = QTextEdit()
        self.live_log.setReadOnly(True)
        self.live_log.setMaximumHeight(200)
        log_layout.addWidget(self.live_log)
        
        layout.addWidget(log_group)
        
        return panel
    
    def _create_equity_tab(self) -> QWidget:
        """创建资产曲线标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        header = QHBoxLayout()
        title = QLabel("📈 资产曲线")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #f0b90b;")
        header.addWidget(title)
        
        self._equity_total_label = QLabel("总资产: -- USDT")
        self._equity_total_label.setStyleSheet("color: #0ecb81; font-size: 14px; font-weight: bold;")
        header.addStretch()
        header.addWidget(self._equity_total_label)
        
        self._equity_change_label = QLabel("")
        self._equity_change_label.setStyleSheet("color: #848e9c; font-size: 12px;")
        header.addWidget(self._equity_change_label)
        
        layout.addLayout(header)
        
        stats_layout = QHBoxLayout()
        
        self._equity_high_label = QLabel("最高: --")
        self._equity_high_label.setStyleSheet("color: #0ecb81; font-size: 12px;")
        stats_layout.addWidget(self._equity_high_label)
        
        self._equity_low_label = QLabel("最低: --")
        self._equity_low_label.setStyleSheet("color: #f6465d; font-size: 12px;")
        stats_layout.addWidget(self._equity_low_label)
        
        self._equity_avg_label = QLabel("平均: --")
        self._equity_avg_label.setStyleSheet("color: #848e9c; font-size: 12px;")
        stats_layout.addWidget(self._equity_avg_label)
        
        self._equity_time_label = QLabel("更新时间: --")
        self._equity_time_label.setStyleSheet("color: #848e9c; font-size: 12px;")
        stats_layout.addWidget(self._equity_time_label)
        
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        self._equity_figure = Figure(figsize=(8, 4), dpi=100)
        self._equity_figure.patch.set_facecolor('#1e222d')
        self._equity_canvas = FigureCanvas(self._equity_figure)
        self._equity_canvas.setMinimumHeight(350)
        self._equity_canvas.setStyleSheet("background-color: #1e222d; border-radius: 8px;")
        layout.addWidget(self._equity_canvas, 1)
        
        history_group = QGroupBox("资产历史记录")
        history_layout = QVBoxLayout(history_group)
        
        self._equity_history_table = QTableWidget()
        self._equity_history_table.setColumnCount(4)
        self._equity_history_table.setHorizontalHeaderLabels(["时间", "总资产", "变化", "变化%"])
        self._equity_history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._equity_history_table.setMaximumHeight(150)
        self._equity_history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        history_layout.addWidget(self._equity_history_table)
        
        layout.addWidget(history_group)
        
        return widget
    
    def _update_equity_tab(self):
        """更新资产曲线标签页"""
        if not hasattr(self, '_trader') or not self._trader:
            return
        
        from datetime import datetime
        
        stats = self._trader.get_statistics()
        total_equity = stats.get('balance', 0) + stats.get('unrealized_pnl', 0)
        
        if not hasattr(self, '_equity_full_history'):
            self._equity_full_history = []
            self._equity_full_times = []
        
        self._equity_full_history.append(total_equity)
        self._equity_full_times.append(datetime.now())
        
        if len(self._equity_full_history) > 300:
            self._equity_full_history = self._equity_full_history[-300:]
            self._equity_full_times = self._equity_full_times[-300:]
        
        if hasattr(self, '_equity_total_label'):
            self._equity_total_label.setText(f"总资产: {total_equity:.2f} USDT")
            if total_equity >= 0:
                self._equity_total_label.setStyleSheet("color: #0ecb81; font-size: 14px; font-weight: bold;")
            else:
                self._equity_total_label.setStyleSheet("color: #f6465d; font-size: 14px; font-weight: bold;")
        
        if len(self._equity_full_history) >= 2:
            first_equity = self._equity_full_history[0]
            last_equity = self._equity_full_history[-1]
            change = last_equity - first_equity
            change_pct = (change / first_equity * 100) if first_equity != 0 else 0
            
            if change >= 0:
                self._equity_change_label.setText(f"+{change:.2f} USDT (+{change_pct:.2f}%)")
                self._equity_change_label.setStyleSheet("color: #0ecb81; font-size: 12px;")
            else:
                self._equity_change_label.setText(f"{change:.2f} USDT ({change_pct:.2f}%)")
                self._equity_change_label.setStyleSheet("color: #f6465d; font-size: 12px;")
        
        if hasattr(self, '_equity_high_label') and self._equity_full_history:
            max_equity = max(self._equity_full_history)
            min_equity = min(self._equity_full_history)
            avg_equity = sum(self._equity_full_history) / len(self._equity_full_history)
            
            self._equity_high_label.setText(f"最高: {max_equity:.2f}")
            self._equity_low_label.setText(f"最低: {min_equity:.2f}")
            self._equity_avg_label.setText(f"平均: {avg_equity:.2f}")
            self._equity_time_label.setText(f"更新: {datetime.now():%H:%M:%S}")
        
        self._draw_equity_chart_full()
        self._update_equity_history_table()
    
    def _draw_equity_chart_full(self):
        """绘制完整资产曲线图"""
        if not hasattr(self, '_equity_full_history') or len(self._equity_full_history) < 2:
            return
        
        if not hasattr(self, '_equity_figure'):
            return
        
        try:
            self._equity_figure.clear()
            
            ax = self._equity_figure.add_subplot(111)
            ax.set_facecolor('#1e222d')
            
            x = range(len(self._equity_full_history))
            y = self._equity_full_history
            
            first_equity = self._equity_full_history[0]
            last_equity = self._equity_full_history[-1]
            change_pct = ((last_equity - first_equity) / first_equity * 100) if first_equity != 0 else 0
            
            if change_pct >= 0:
                line_color = '#0ecb81'
                fill_color = '#0ecb81'
            else:
                line_color = '#f6465d'
                fill_color = '#f6465d'
            
            ax.plot(x, y, color=line_color, linewidth=2, label='总资产')
            ax.fill_between(x, y, alpha=0.3, color=fill_color)
            
            ax.axhline(y=first_equity, color='#848e9c', linestyle='--', linewidth=1, alpha=0.5, label='初始资产')
            
            max_equity = max(self._equity_full_history)
            min_equity = min(self._equity_full_history)
            ax.axhline(y=max_equity, color='#0ecb81', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(y=min_equity, color='#f6465d', linestyle=':', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('时间', color='#848e9c', fontsize=10)
            ax.set_ylabel('资产 (USDT)', color='#848e9c', fontsize=10)
            ax.set_title(f'资产曲线 | 当前: {last_equity:.2f} USDT ({change_pct:+.2f}%)', 
                        color='#eaecef', fontsize=12, fontweight='bold')
            
            ax.tick_params(colors='#848e9c', labelsize=8)
            ax.spines['top'].set_color('#2a2e39')
            ax.spines['right'].set_color('#2a2e39')
            ax.spines['bottom'].set_color('#2a2e39')
            ax.spines['left'].set_color('#2a2e39')
            
            ax.grid(True, alpha=0.2, color='#2a2e39')
            
            legend = ax.legend(loc='upper left', fontsize=8, facecolor='#1e222d', edgecolor='#2a2e39')
            for text in legend.get_texts():
                text.set_color('#848e9c')
            
            self._equity_figure.tight_layout()
            self._equity_canvas.draw()
            
        except Exception as e:
            pass
    
    def _update_equity_history_table(self):
        """更新资产历史表格"""
        if not hasattr(self, '_equity_full_history') or len(self._equity_full_history) < 2:
            return
        
        recent_data = list(zip(self._equity_full_times[-20:], self._equity_full_history[-20:]))
        recent_data.reverse()
        
        self._equity_history_table.setRowCount(len(recent_data))
        
        for i, (time, equity) in enumerate(recent_data):
            self._equity_history_table.setItem(i, 0, QTableWidgetItem(time.strftime("%H:%M:%S")))
            self._equity_history_table.setItem(i, 1, QTableWidgetItem(f"{equity:.2f}"))
            
            if i < len(recent_data) - 1:
                prev_equity = recent_data[i + 1][1]
                change = equity - prev_equity
                change_pct = (change / prev_equity * 100) if prev_equity != 0 else 0
                
                change_item = QTableWidgetItem(f"{change:+.2f}")
                pct_item = QTableWidgetItem(f"{change_pct:+.2f}%")
                
                if change >= 0:
                    change_item.setForeground(QColor("#0ecb81"))
                    pct_item.setForeground(QColor("#0ecb81"))
                else:
                    change_item.setForeground(QColor("#f6465d"))
                    pct_item.setForeground(QColor("#f6465d"))
                
                self._equity_history_table.setItem(i, 2, change_item)
                self._equity_history_table.setItem(i, 3, pct_item)
            else:
                self._equity_history_table.setItem(i, 2, QTableWidgetItem("--"))
                self._equity_history_table.setItem(i, 3, QTableWidgetItem("--"))
    
    def _create_version_tab(self) -> QWidget:
        """创建版本更新标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        header = QHBoxLayout()
        
        title = QLabel("📝 版本更新记录")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #f0b90b;")
        header.addWidget(title)
        
        header.addStretch()
        
        version_label = QLabel(f"当前版本: v{self._get_current_version()}")
        version_label.setStyleSheet("color: #848e9c; font-size: 14px;")
        header.addWidget(version_label)
        
        layout.addLayout(header)
        
        self.version_list = QListWidget()
        self.version_list.setStyleSheet("""
            QListWidget {
                background-color: #1e222d;
                border: 1px solid #2a2e39;
                border-radius: 8px;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #2a2e39;
            }
            QListWidget::item:selected {
                background-color: #2a2e39;
            }
        """)
        self.version_list.itemClicked.connect(self._show_version_details)
        layout.addWidget(self.version_list, 1)
        
        self.version_details = QTextEdit()
        self.version_details.setReadOnly(True)
        self.version_details.setStyleSheet("""
            QTextEdit {
                background-color: #1e222d;
                border: 1px solid #2a2e39;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        self.version_details.setMaximumHeight(300)
        layout.addWidget(self.version_details, 1)
        
        self._load_version_history()
        
        return widget
    
    def _get_current_version(self) -> str:
        """获取当前版本"""
        try:
            from core.version import get_current_version
            return get_current_version()
        except:
            return "2.0.0"
    
    def _load_version_history(self) -> None:
        """加载版本历史"""
        try:
            from core.version import get_version_history
            
            history = get_version_history()
            self.version_list.clear()
            
            for info in history:
                item_text = f"v{info.version} ({info.release_date})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, info)
                self.version_list.addItem(item)
            
            if self.version_list.count() > 0:
                self.version_list.setCurrentRow(0)
                self._show_version_details(self.version_list.item(0))
                
        except Exception as e:
            self.version_details.setText(f"加载版本历史失败: {e}")
    
    def _show_version_details(self, item: QListWidgetItem) -> None:
        """显示版本详情"""
        try:
            from core.version import format_version_info
            
            info = item.data(Qt.UserRole)
            if info:
                self.version_details.setText(format_version_info(info))
        except Exception as e:
            self.version_details.setText(f"显示版本详情失败: {e}")
    
    def _connect_exchange(self) -> None:
        """连接交易所"""
        from Trading.live_trader import LiveTrader, TradingConfig, TradingMode
        
        if hasattr(self, '_live_worker') and self._live_worker:
            self._live_worker.stop()
            self._live_worker = None
        
        if hasattr(self, '_trader') and self._trader and self._trader.is_running:
            self._trader.stop()
        
        mode = TradingMode.TEST if self.live_mode_test.isChecked() else TradingMode.LIVE
        
        config = TradingConfig(
            mode=mode,
            api_key=self.api_key.text(),
            api_secret=self.api_secret.text(),
            testnet=self.live_mode_test.isChecked(),
            symbol=self.live_symbol.currentText(),
            interval="30m",
            leverage=self.live_leverage.value(),
            position_size=self.live_position_size.value() / 100,
            stop_loss_pct=self.live_stop_loss.value(),
            take_profit_pct=self.live_take_profit.value(),
            max_daily_trades=self.live_max_trades.value(),
        )
        
        self._trader = LiveTrader(config)
        
        if self._trader.connect():
            self.connect_btn.setText("🔄 重连")
            self.connect_btn.setEnabled(True)
            self.start_live_btn.setEnabled(True)
            
            if hasattr(self, '_status_label'):
                self._status_label.setText("● 已连接")
                self._status_label.setStyleSheet("color: #0ecb81; font-size: 12px;")
            
            self.live_log.append(f"[{datetime.now():%H:%M:%S}] ✅ 连接成功 ({mode.value}模式)")
            self._update_account_status()
            self._update_position_table()
            self._update_equity_tab()
            
            self._live_worker = LiveAccountWorker(self._trader, 3000)
            self._live_worker.account_updated.connect(self._on_account_updated)
            self._live_worker.positions_updated.connect(self._on_positions_updated)
            self._live_worker.error_occurred.connect(self._on_live_error)
            self._live_worker.start()
            
            positions = self._trader.positions
            if positions:
                self.live_log.append(f"[{datetime.now():%H:%M:%S}] 📊 当前持仓: {len(positions)}个")
                for symbol, pos in positions.items():
                    side_text = "做多" if pos.side == "long" else "做空"
                    self.live_log.append(f"   {symbol}: {side_text} {pos.quantity:.4f} @ {pos.entry_price:.4f}")
        else:
            self.live_log.append(f"[{datetime.now():%H:%M:%S}] ❌ 连接失败")
            if hasattr(self, '_status_label'):
                self._status_label.setText("● 连接失败")
                self._status_label.setStyleSheet("color: #f6465d; font-size: 12px;")
    
    def _on_account_updated(self, stats: dict):
        """账户信息更新回调（在主线程中执行）"""
        try:
            for key, (value_label, unit) in self._account_labels.items():
                value = stats.get(key, 0)
                if "pnl" in key.lower():
                    if value >= 0:
                        value_label.setText(f"+{value:.2f} {unit}")
                        value_label.setStyleSheet("color: #0ecb81; font-weight: bold;")
                    else:
                        value_label.setText(f"{value:.2f} {unit}")
                        value_label.setStyleSheet("color: #f6465d; font-weight: bold;")
                else:
                    if isinstance(value, float):
                        value_label.setText(f"{value:.2f} {unit}")
                    else:
                        value_label.setText(f"{value} {unit}")
            
            if hasattr(self, '_leverage_labels'):
                balance = stats.get('balance', 0)
                margin_used = stats.get('margin_used', 0)
                leverage = self.live_leverage.value()
                
                if 'used_margin' in self._leverage_labels:
                    self._leverage_labels['used_margin'][0].setText(f"{margin_used:.2f} USDT")
                if 'available_margin' in self._leverage_labels:
                    self._leverage_labels['available_margin'][0].setText(f"{balance - margin_used:.2f} USDT")
                if 'max_position' in self._leverage_labels:
                    max_pos = balance * leverage if balance > 0 else 0
                    self._leverage_labels['max_position'][0].setText(f"{max_pos:.2f} USDT")
            
            if hasattr(self, '_daily_stats'):
                daily_trades = stats.get('daily_trades', 0)
                daily_pnl = stats.get('daily_pnl', 0)
                self._daily_stats.setText(f"今日: {daily_trades}笔交易 | 盈亏: {daily_pnl:+.2f} USDT")
            
            self._update_equity_tab_from_stats(stats)
            
        except Exception as e:
            pass
    
    def _on_positions_updated(self, positions: list):
        """持仓更新回调（在主线程中执行）"""
        try:
            self.position_table.setRowCount(len(positions))
            
            for i, pos in enumerate(positions):
                item = QTableWidgetItem(pos['symbol'])
                item.setTextAlignment(Qt.AlignCenter)
                self.position_table.setItem(i, 0, item)
                
                side_text = "做多" if pos['side'] == "long" else "做空"
                side_item = QTableWidgetItem(side_text)
                side_item.setTextAlignment(Qt.AlignCenter)
                if pos['side'] == "long":
                    side_item.setForeground(QColor("#0ecb81"))
                else:
                    side_item.setForeground(QColor("#f6465d"))
                self.position_table.setItem(i, 1, side_item)
                
                qty_item = QTableWidgetItem(f"{pos['quantity']:.4f}")
                qty_item.setTextAlignment(Qt.AlignCenter)
                self.position_table.setItem(i, 2, qty_item)
                
                entry_item = QTableWidgetItem(f"{pos['entry_price']:.4f}")
                entry_item.setTextAlignment(Qt.AlignCenter)
                self.position_table.setItem(i, 3, entry_item)
                
                current_item = QTableWidgetItem(f"{pos['current_price']:.4f}")
                current_item.setTextAlignment(Qt.AlignCenter)
                self.position_table.setItem(i, 4, current_item)
                
                pnl = pos['unrealized_pnl']
                pnl_item = QTableWidgetItem(f"{pnl:+.4f}")
                pnl_item.setTextAlignment(Qt.AlignCenter)
                if pnl >= 0:
                    pnl_item.setForeground(QColor("#0ecb81"))
                else:
                    pnl_item.setForeground(QColor("#f6465d"))
                self.position_table.setItem(i, 5, pnl_item)
                
                if pos['entry_price'] > 0:
                    pnl_pct = (pnl / (pos['entry_price'] * pos['quantity'])) * 100 if pos['quantity'] > 0 else 0
                else:
                    pnl_pct = 0
                pnl_pct_item = QTableWidgetItem(f"{pnl_pct:+.2f}%")
                pnl_pct_item.setTextAlignment(Qt.AlignCenter)
                if pnl_pct >= 0:
                    pnl_pct_item.setForeground(QColor("#0ecb81"))
                else:
                    pnl_pct_item.setForeground(QColor("#f6465d"))
                self.position_table.setItem(i, 6, pnl_pct_item)
                
                leverage_item = QTableWidgetItem(f"{pos['leverage']}x")
                leverage_item.setTextAlignment(Qt.AlignCenter)
                self.position_table.setItem(i, 7, leverage_item)
            
            if hasattr(self, '_position_count'):
                self._position_count.setText(f"{len(positions)}个持仓")
                
        except Exception as e:
            pass
    
    def _on_live_error(self, error: str):
        """错误回调"""
        self.live_log.append(f"[{datetime.now():%H:%M:%S}] ⚠️ 刷新错误: {error}")
    
    def _update_equity_tab_from_stats(self, stats: dict):
        """从统计数据更新资产曲线"""
        if not hasattr(self, '_trader') or not self._trader:
            return
        
        total_equity = stats.get('balance', 0) + stats.get('unrealized_pnl', 0)
        
        if not hasattr(self, '_equity_full_history'):
            self._equity_full_history = []
            self._equity_full_times = []
        
        self._equity_full_history.append(total_equity)
        self._equity_full_times.append(datetime.now())
        
        if len(self._equity_full_history) > 300:
            self._equity_full_history = self._equity_full_history[-300:]
            self._equity_full_times = self._equity_full_times[-300:]
        
        if hasattr(self, '_equity_total_label'):
            self._equity_total_label.setText(f"总资产: {total_equity:.2f} USDT")
            if total_equity >= 0:
                self._equity_total_label.setStyleSheet("color: #0ecb81; font-size: 14px; font-weight: bold;")
            else:
                self._equity_total_label.setStyleSheet("color: #f6465d; font-size: 14px; font-weight: bold;")
        
        if len(self._equity_full_history) >= 2:
            first_equity = self._equity_full_history[0]
            last_equity = self._equity_full_history[-1]
            change = last_equity - first_equity
            change_pct = (change / first_equity * 100) if first_equity != 0 else 0
            
            if change >= 0:
                self._equity_change_label.setText(f"+{change:.2f} USDT (+{change_pct:.2f}%)")
                self._equity_change_label.setStyleSheet("color: #0ecb81; font-size: 12px;")
            else:
                self._equity_change_label.setText(f"{change:.2f} USDT ({change_pct:.2f}%)")
                self._equity_change_label.setStyleSheet("color: #f6465d; font-size: 12px;")
        
        if hasattr(self, '_equity_high_label') and self._equity_full_history:
            max_equity = max(self._equity_full_history)
            min_equity = min(self._equity_full_history)
            avg_equity = sum(self._equity_full_history) / len(self._equity_full_history)
            
            self._equity_high_label.setText(f"最高: {max_equity:.2f}")
            self._equity_low_label.setText(f"最低: {min_equity:.2f}")
            self._equity_avg_label.setText(f"平均: {avg_equity:.2f}")
            self._equity_time_label.setText(f"更新: {datetime.now():%H:%M:%S}")
        
        self._draw_equity_chart_full()
        self._update_equity_history_table()
    
    def _auto_refresh_account(self):
        """自动刷新账户信息"""
        if not hasattr(self, '_trader') or not self._trader:
            return
        
        try:
            self._trader._update_positions()
            self._update_account_status()
            self._update_position_table()
            self._update_equity_tab()
        except Exception as e:
            self.live_log.append(f"[{datetime.now():%H:%M:%S}] ⚠️ 刷新失败: {e}")
    
    def _start_live_trading(self) -> None:
        """启动实盘交易"""
        if hasattr(self, '_trader') and self._trader:
            from Strategy.templates import get_strategy
            
            strategy_name = self.live_strategy.currentText()
            strategy = get_strategy(strategy_name, {})
            
            self._trader.set_strategy(strategy)
            
            self._trader.set_callbacks(
                order_callback=self._on_order_update,
                error_callback=self._on_trading_error,
            )
            
            if self._trader.start():
                self.start_live_btn.setEnabled(False)
                self.stop_live_btn.setEnabled(True)
                
                if hasattr(self, '_status_label'):
                    self._status_label.setText("● 运行中")
                    self._status_label.setStyleSheet("color: #0ecb81; font-size: 12px;")
                
                self.live_log.append(f"[{datetime.now():%H:%M:%S}] 🚀 交易已启动")
                self.live_log.append(f"   策略: {strategy_name}")
                self.live_log.append(f"   交易对: {self.live_symbol.currentText()}")
                self.live_log.append(f"   杠杆: {self.live_leverage.value()}x")
                
                self._status_timer = QTimer()
                self._status_timer.timeout.connect(self._update_account_status)
                self._status_timer.start(1000)
    
    def _stop_live_trading(self) -> None:
        """停止实盘交易"""
        if hasattr(self, '_live_worker') and self._live_worker:
            self._shutdown_thread(self._live_worker, "LiveAccountWorker")
            self._live_worker = None
        
        if hasattr(self, '_trader') and self._trader:
            self._trader.stop()
            self.start_live_btn.setEnabled(True)
            self.stop_live_btn.setEnabled(False)
            
            if hasattr(self, '_status_label'):
                self._status_label.setText("● 已停止")
                self._status_label.setStyleSheet("color: #f0b90b; font-size: 12px;")
            
            self.live_log.append(f"[{datetime.now():%H:%M:%S}] ⏹ 交易已停止")
            
            if hasattr(self, '_status_timer'):
                self._status_timer.stop()
    
    def _update_account_status(self) -> None:
        """更新账户状态"""
        if not hasattr(self, '_trader') or not self._trader:
            return
        
        stats = self._trader.get_statistics()
        
        for key, (value_label, unit) in self._account_labels.items():
            value = stats.get(key, 0)
            if "pnl" in key.lower():
                if value >= 0:
                    value_label.setText(f"+{value:.2f} {unit}")
                    value_label.setStyleSheet("color: #0ecb81; font-weight: bold;")
                else:
                    value_label.setText(f"{value:.2f} {unit}")
                    value_label.setStyleSheet("color: #f6465d; font-weight: bold;")
            else:
                if isinstance(value, float):
                    value_label.setText(f"{value:.2f} {unit}")
                else:
                    value_label.setText(f"{value} {unit}")
        
        if hasattr(self, '_leverage_labels'):
            balance = stats.get('balance', 0)
            margin_used = stats.get('margin_used', 0)
            leverage = self.live_leverage.value()
            
            if 'used_margin' in self._leverage_labels:
                self._leverage_labels['used_margin'][0].setText(f"{margin_used:.2f} USDT")
            if 'available_margin' in self._leverage_labels:
                self._leverage_labels['available_margin'][0].setText(f"{balance - margin_used:.2f} USDT")
            if 'max_position' in self._leverage_labels:
                max_pos = balance * leverage if balance > 0 else 0
                self._leverage_labels['max_position'][0].setText(f"{max_pos:.2f} USDT")
        
        if hasattr(self, '_daily_stats'):
            daily_trades = stats.get('daily_trades', 0)
            daily_pnl = stats.get('daily_pnl', 0)
            self._daily_stats.setText(f"今日: {daily_trades}笔交易 | 盈亏: {daily_pnl:+.2f} USDT")
        
        self._update_position_table()
    
    def _update_position_table(self) -> None:
        """更新持仓表格"""
        if not hasattr(self, '_trader') or not self._trader:
            return
        
        positions = self._trader.positions
        self.position_table.setRowCount(len(positions))
        
        for i, (symbol, pos) in enumerate(positions.items()):
            item = QTableWidgetItem(symbol)
            item.setTextAlignment(Qt.AlignCenter)
            self.position_table.setItem(i, 0, item)
            
            side_text = "做多" if pos.side == "long" else "做空"
            side_item = QTableWidgetItem(side_text)
            side_item.setTextAlignment(Qt.AlignCenter)
            if pos.side == "long":
                side_item.setForeground(QColor("#0ecb81"))
            else:
                side_item.setForeground(QColor("#f6465d"))
            self.position_table.setItem(i, 1, side_item)
            
            qty_item = QTableWidgetItem(f"{pos.quantity:.4f}")
            qty_item.setTextAlignment(Qt.AlignCenter)
            self.position_table.setItem(i, 2, qty_item)
            
            entry_item = QTableWidgetItem(f"{pos.entry_price:.4f}")
            entry_item.setTextAlignment(Qt.AlignCenter)
            self.position_table.setItem(i, 3, entry_item)
            
            current_price = pos.current_price if hasattr(pos, 'current_price') and pos.current_price > 0 else pos.entry_price
            current_item = QTableWidgetItem(f"{current_price:.4f}")
            current_item.setTextAlignment(Qt.AlignCenter)
            self.position_table.setItem(i, 4, current_item)
            
            pnl = pos.unrealized_pnl
            pnl_item = QTableWidgetItem(f"{pnl:+.4f}")
            pnl_item.setTextAlignment(Qt.AlignCenter)
            if pnl >= 0:
                pnl_item.setForeground(QColor("#0ecb81"))
            else:
                pnl_item.setForeground(QColor("#f6465d"))
            self.position_table.setItem(i, 5, pnl_item)
            
            if pos.entry_price > 0:
                pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100 if pos.quantity > 0 else 0
            else:
                pnl_pct = 0
            pnl_pct_item = QTableWidgetItem(f"{pnl_pct:+.2f}%")
            pnl_pct_item.setTextAlignment(Qt.AlignCenter)
            if pnl_pct >= 0:
                pnl_pct_item.setForeground(QColor("#0ecb81"))
            else:
                pnl_pct_item.setForeground(QColor("#f6465d"))
            self.position_table.setItem(i, 6, pnl_pct_item)
            
            leverage = pos.leverage if hasattr(pos, 'leverage') else 1
            leverage_item = QTableWidgetItem(f"{leverage}x")
            leverage_item.setTextAlignment(Qt.AlignCenter)
            self.position_table.setItem(i, 7, leverage_item)
        
        if hasattr(self, '_position_count'):
            self._position_count.setText(f"{len(positions)}个持仓")
    
    def _on_order_update(self, order: dict) -> None:
        """订单更新回调"""
        order_type = order.get('type', 'unknown')
        symbol = order.get('symbol', '')
        side = order.get('side', '')
        quantity = order.get('quantity', 0)
        price = order.get('price', 0)
        
        if order_type == 'open':
            self.live_log.append(f"[{datetime.now():%H:%M:%S}] 📈 开仓: {symbol} {side} {quantity:.4f}")
        elif order_type == 'close':
            self.live_log.append(f"[{datetime.now():%H:%M:%S}] 📉 平仓: {symbol} {side} {quantity:.4f}")
        else:
            self.live_log.append(f"[{datetime.now():%H:%M:%S}] 📋 订单: {order}")
    
    def _on_trading_error(self, error: str) -> None:
        """交易错误回调"""
        self.live_log.append(f"[{datetime.now():%H:%M:%S}] ❌ 错误: {error}")
    
    def _load_api_from_env(self) -> None:
        """从.env文件加载API密钥"""
        import os
        from pathlib import Path
        
        env_path = Path(__file__).parent.parent / ".env"
        
        if env_path.exists():
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip("'\"")
                            
                            if key in ['BINANCE_API_KEY', 'API_KEY']:
                                self.api_key.setText(value)
                            elif key in ['BINANCE_API_SECRET', 'API_SECRET']:
                                self.api_secret.setText(value)
                
                self._env_loaded = True
                    
            except Exception as e:
                self._env_load_error = str(e)
                self._env_loaded = False
    
    def _toggle_api_visibility(self) -> None:
        """切换API Key可见性"""
        if self.api_key.echoMode() == QLineEdit.Password:
            self.api_key.setEchoMode(QLineEdit.Normal)
            self.show_api_btn.setText("🔒")
        else:
            self.api_key.setEchoMode(QLineEdit.Password)
            self.show_api_btn.setText("👁")
    
    def _toggle_secret_visibility(self) -> None:
        """切换API Secret可见性"""
        if self.api_secret.echoMode() == QLineEdit.Password:
            self.api_secret.setEchoMode(QLineEdit.Normal)
            self.show_secret_btn.setText("🔒")
        else:
            self.api_secret.setEchoMode(QLineEdit.Password)
            self.show_secret_btn.setText("👁")
    
    def _on_mode_changed(self, checked: bool) -> None:
        """模式切换时自动连接"""
        if checked and self.api_key.text() and self.api_secret.text():
            self._connect_exchange()
    
    def _create_backtest_config_panel(self) -> QWidget:
        """创建回测配置面板"""
        panel = QFrame()
        panel.setStyleSheet("QFrame { background-color: #1e222d; border-radius: 8px; }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        config_grid = QGridLayout()
        config_grid.setSpacing(10)
        
        config_grid.addWidget(QLabel("交易对"), 0, 0)
        self.symbol = QComboBox()
        self.symbol.addItems(self.SYMBOLS)
        config_grid.addWidget(self.symbol, 0, 1)
        
        config_grid.addWidget(QLabel("周期"), 0, 2)
        self.interval = QComboBox()
        self.interval.addItems(self.INTERVALS)
        self.interval.setCurrentIndex(3)
        config_grid.addWidget(self.interval, 0, 3)
        
        config_grid.addWidget(QLabel("数据量"), 1, 0)
        self.data_limit = QSpinBox()
        self.data_limit.setRange(100, 50000)
        self.data_limit.setValue(1000)
        config_grid.addWidget(self.data_limit, 1, 1)
        
        config_grid.addWidget(QLabel("初始资金"), 1, 2)
        self.capital = QDoubleSpinBox()
        self.capital.setRange(100, 1000000)
        self.capital.setValue(10000)
        config_grid.addWidget(self.capital, 1, 3)
        
        layout.addLayout(config_grid)
        
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("background-color: #2a2e39;")
        layout.addWidget(line1)
        
        risk_label = QLabel("⚙️ 风险控制")
        risk_label.setStyleSheet("font-weight: bold; color: #f0b90b;")
        layout.addWidget(risk_label)
        
        risk_grid = QGridLayout()
        risk_grid.setSpacing(10)
        
        risk_grid.addWidget(QLabel("杠杆倍数"), 0, 0)
        self.leverage = QSpinBox()
        self.leverage.setRange(1, 125)
        self.leverage.setValue(5)
        self.leverage.valueChanged.connect(self._on_leverage_changed)
        risk_grid.addWidget(self.leverage, 0, 1)
        
        risk_grid.addWidget(QLabel("止损率 (%)"), 0, 2)
        self.stop_loss = QDoubleSpinBox()
        self.stop_loss.setRange(0, 100)
        self.stop_loss.setDecimals(2)
        self.stop_loss.setSingleStep(0.5)
        self.stop_loss.setValue(0)
        risk_grid.addWidget(self.stop_loss, 0, 3)
        
        risk_grid.addWidget(QLabel("止盈率 (%)"), 1, 0)
        self.take_profit = QDoubleSpinBox()
        self.take_profit.setRange(0, 1000)
        self.take_profit.setDecimals(2)
        self.take_profit.setSingleStep(0.5)
        self.take_profit.setValue(0)
        risk_grid.addWidget(self.take_profit, 1, 1)
        
        risk_grid.addWidget(QLabel("仓位比例 (%)"), 1, 2)
        self.position_size = QDoubleSpinBox()
        self.position_size.setRange(1, 100)
        self.position_size.setDecimals(0)
        self.position_size.setSingleStep(5)
        self.position_size.setValue(100)
        risk_grid.addWidget(self.position_size, 1, 3)
        
        self.risk_warning = QLabel("")
        self.risk_warning.setStyleSheet("color: #f6465d; font-size: 11px;")
        self.risk_warning.setWordWrap(True)
        layout.addLayout(risk_grid)
        
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #2a2e39;")
        layout.addWidget(line2)
        
        layout.addWidget(QLabel("策略选择"))
        
        self.strategy = QComboBox()
        self.strategy.addItems(list(self.STRATEGIES.keys()))
        self.strategy.currentTextChanged.connect(self._on_strategy_changed)
        layout.addWidget(self.strategy)
        
        self.params_frame = QFrame()
        self.params_layout = QGridLayout(self.params_frame)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.params_frame)
        
        self._param_widgets = {}
        self._on_strategy_changed(self.strategy.currentText())
        
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        
        self.reset_backtest_btn = QPushButton("🔄 重置参数")
        self.reset_backtest_btn.clicked.connect(self._reset_backtest_params)
        btn_layout.addWidget(self.reset_backtest_btn)
        
        self.run_btn = QPushButton("▶ 开始回测")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._run)
        btn_layout.addWidget(self.run_btn)
        
        self.export_btn = QPushButton("📊 导出报告")
        self.export_btn.clicked.connect(self._export_report)
        self.export_btn.setEnabled(False)
        btn_layout.addWidget(self.export_btn)
        
        self.sync_to_live_btn = QPushButton("💹 同步到实盘")
        self.sync_to_live_btn.clicked.connect(self._sync_to_live_trading)
        self.sync_to_live_btn.setEnabled(False)
        btn_layout.addWidget(self.sync_to_live_btn)
        
        self.enhanced_btn = QPushButton("🔬 强化回测")
        self.enhanced_btn.clicked.connect(self._run_enhanced_backtest)
        self.enhanced_btn.setEnabled(False)
        btn_layout.addWidget(self.enhanced_btn)
        
        layout.addLayout(btn_layout)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        return panel
    
    def _create_optimizer_config_panel(self) -> QWidget:
        """创建参数探索配置面板"""
        widget = QWidget()
        main_layout = QHBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(12)
        
        left_panel = QFrame()
        left_panel.setStyleSheet("QFrame { background-color: #1e222d; border-radius: 8px; }")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(10)
        
        opt_label = QLabel("🔍 参数探索")
        opt_label.setStyleSheet("font-weight: bold; color: #f0b90b; font-size: 14px;")
        left_layout.addWidget(opt_label)
        
        config_grid = QGridLayout()
        config_grid.setSpacing(8)
        
        config_grid.addWidget(QLabel("交易对"), 0, 0)
        self.opt_symbol = QComboBox()
        self.opt_symbol.addItems(self.SYMBOLS)
        config_grid.addWidget(self.opt_symbol, 0, 1)
        
        config_grid.addWidget(QLabel("周期"), 0, 2)
        self.opt_interval = QComboBox()
        self.opt_interval.addItems(self.INTERVALS)
        self.opt_interval.setCurrentIndex(3)
        config_grid.addWidget(self.opt_interval, 0, 3)
        
        config_grid.addWidget(QLabel("数据量"), 0, 4)
        self.opt_data_limit = QSpinBox()
        self.opt_data_limit.setRange(5000, 50000)
        self.opt_data_limit.setValue(5000)
        self.opt_data_limit.setSingleStep(1000)
        config_grid.addWidget(self.opt_data_limit, 0, 5)
        
        config_grid.addWidget(QLabel("策略"), 1, 0)
        self.opt_strategy = QComboBox()
        self.opt_strategy.addItems(list(self.STRATEGIES.keys()))
        config_grid.addWidget(self.opt_strategy, 1, 1, 1, 3)
        
        config_grid.addWidget(QLabel("初始资金"), 1, 4)
        self.opt_capital = QDoubleSpinBox()
        self.opt_capital.setRange(100, 1000000)
        self.opt_capital.setValue(10000)
        config_grid.addWidget(self.opt_capital, 1, 5)
        
        data_warning = QLabel("⚠️ 数据量最低5000条 (30分钟×5000)")
        data_warning.setStyleSheet("color: #f0b90b; font-size: 10px;")
        config_grid.addWidget(data_warning, 2, 0, 1, 6)
        
        left_layout.addLayout(config_grid)
        
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("background-color: #2a2e39;")
        left_layout.addWidget(line1)
        
        opt_grid = QGridLayout()
        opt_grid.setSpacing(8)
        
        opt_grid.addWidget(QLabel("方法"), 0, 0)
        self.opt_method = QComboBox()
        self.opt_method.addItems([
            "复合优化", 
            "随机搜索", 
            "网格搜索", 
            "贝叶斯优化", 
            "遗传算法",
            "模拟退火",
            "粒子群优化",
            "强化学习"
        ])
        self.opt_method.currentTextChanged.connect(self._update_method_description)
        opt_grid.addWidget(self.opt_method, 0, 1)
        
        opt_grid.addWidget(QLabel("迭代"), 0, 2)
        self.opt_iterations = QSpinBox()
        self.opt_iterations.setRange(10, 1000)
        self.opt_iterations.setValue(30)
        opt_grid.addWidget(self.opt_iterations, 0, 3)
        
        opt_grid.addWidget(QLabel("目标"), 0, 4)
        self.opt_metric = QComboBox()
        self.opt_metric.addItems(["夏普比率", "总收益率", "综合得分"])
        opt_grid.addWidget(self.opt_metric, 0, 5)
        
        self.method_description = QLabel()
        self.method_description.setWordWrap(True)
        self.method_description.setStyleSheet("color: #848e9c; font-size: 11px; padding: 3px;")
        opt_grid.addWidget(self.method_description, 1, 0, 1, 6)
        
        self._update_method_description("随机搜索")
        
        left_layout.addLayout(opt_grid)
        
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #2a2e39;")
        left_layout.addWidget(line2)
        
        params_label = QLabel("📊 参数范围设置")
        params_label.setStyleSheet("font-weight: bold; color: #f0b90b;")
        left_layout.addWidget(params_label)
        
        self.param_table = QTableWidget()
        self.param_table.setColumnCount(4)
        self.param_table.setHorizontalHeaderLabels(["参数名", "最小值", "最大值", "步长"])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.param_table.verticalHeader().setDefaultSectionSize(42)
        self.param_table.verticalHeader().setVisible(False)
        self.param_table.setMinimumHeight(200)
        self.param_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e222d;
                border: 1px solid #2a2e39;
                gridline-color: #2a2e39;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 8px;
                text-align: center;
            }
            QTableWidget::item:selected {
                background-color: #2a2e39;
                color: #f0b90b;
            }
            QTableWidget::item:focus {
                background-color: #0b0e11;
                border: 1px solid #f0b90b;
            }
            QHeaderView::section {
                background-color: #2a2e39;
                color: #eaecef;
                padding: 10px;
                border: none;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        self.param_table.itemChanged.connect(self._on_param_table_changed)
        left_layout.addWidget(self.param_table, 2)
        
        btn_layout = QHBoxLayout()
        
        self.reset_params_btn = QPushButton("🔄 重置参数")
        self.reset_params_btn.clicked.connect(self._reset_param_ranges)
        btn_layout.addWidget(self.reset_params_btn)
        
        self.start_opt_btn = QPushButton("🚀 开始探索")
        self.start_opt_btn.setObjectName("primary")
        self.start_opt_btn.clicked.connect(self._run_optimization)
        btn_layout.addWidget(self.start_opt_btn)
        
        self.stop_opt_btn = QPushButton("⏹ 停止")
        self.stop_opt_btn.clicked.connect(self._stop_optimization)
        self.stop_opt_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_opt_btn)
        
        self.apply_params_btn = QPushButton("✅ 应用到回测")
        self.apply_params_btn.clicked.connect(self._apply_best_params_to_backtest)
        self.apply_params_btn.setEnabled(False)
        btn_layout.addWidget(self.apply_params_btn)
        
        self.save_params_btn = QPushButton("💾 保存参数")
        self.save_params_btn.clicked.connect(self._save_optimization_params)
        self.save_params_btn.setEnabled(False)
        btn_layout.addWidget(self.save_params_btn)
        
        left_layout.addLayout(btn_layout)
        
        self.opt_progress = QProgressBar()
        self.opt_progress.setVisible(False)
        left_layout.addWidget(self.opt_progress)
        
        main_layout.addWidget(left_panel, 2)
        
        right_panel = QFrame()
        right_panel.setStyleSheet("QFrame { background-color: #1e222d; border-radius: 8px; }")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(10)
        
        log_label = QLabel("📝 运行日志")
        log_label.setStyleSheet("font-weight: bold; color: #f0b90b; font-size: 14px;")
        right_layout.addWidget(log_label)
        
        self.opt_result_text = QTextEdit()
        self.opt_result_text.setReadOnly(True)
        self.opt_result_text.setStyleSheet("""
            QTextEdit {
                background-color: #0b0e11;
                border: 1px solid #2a2e39;
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        right_layout.addWidget(self.opt_result_text)
        
        main_layout.addWidget(right_panel, 1)
        
        self._param_spinboxes = {}
        self._update_param_ranges_display()
        self.opt_strategy.currentTextChanged.connect(self._update_param_ranges_display)
        
        return widget
    
    def _update_method_description(self, method: str):
        """更新优化方法描述"""
        descriptions = {
            "复合优化": "🏆 复合优化: 组合多种优化算法(随机搜索、遗传算法、模拟退火、粒子群、强化学习、贝叶斯)，比较得出相对最优解。",
            "随机搜索": "🎲 随机搜索: 在参数空间中随机采样，简单高效，适合快速探索大范围参数空间。",
            "网格搜索": "📊 网格搜索: 遍历所有参数组合，结果可靠但计算量大，适合小范围精确搜索。",
            "贝叶斯优化": "🧠 贝叶斯优化: 基于概率模型的智能优化，利用历史结果指导搜索，适合昂贵的评估函数。",
            "遗传算法": "🧬 遗传算法: 模拟自然选择进化，通过选择、交叉、变异寻找最优解，适合复杂多峰问题。",
            "模拟退火": "🔥 模拟退火: 模拟金属退火过程，初期接受劣解以跳出局部最优，逐渐收敛到全局最优。",
            "粒子群优化": "🐦 粒子群优化: 模拟鸟群觅食行为，群体协作寻找最优解，适合连续参数空间优化。",
            "强化学习": "🤖 强化学习: 使用Q-Learning学习最优策略，通过探索与利用平衡逐步优化参数。",
        }
        self.method_description.setText(descriptions.get(method, ""))
    
    def _update_param_ranges_display(self):
        """更新参数范围显示"""
        try:
            from Strategy.parameter_optimizer import get_all_optimizable_params, ParameterRange
            from Strategy.templates import get_strategy
            
            strategy_name = self.STRATEGIES.get(self.opt_strategy.currentText())
            if not strategy_name:
                return
            
            strategy = get_strategy(strategy_name)
            all_params = get_all_optimizable_params(strategy, include_risk_params=True)
            
            all_ranges = all_params["strategy"] + all_params["risk"]
            
            self.param_table.blockSignals(True)
            self.param_table.setRowCount(len(all_ranges))
            self._param_spinboxes = {}
            
            for row, pr in enumerate(all_ranges):
                name_item = QTableWidgetItem(pr.name)
                name_item.setTextAlignment(Qt.AlignCenter)
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                name_item.setBackground(QColor("#2a2e39"))
                self.param_table.setItem(row, 0, name_item)
                
                if pr.values:
                    combo = QComboBox()
                    combo.addItems([str(v) for v in pr.values])
                    combo.setStyleSheet("""
                        QComboBox {
                            background-color: #0b0e11;
                            border: 1px solid #2a2e39;
                            border-radius: 4px;
                            padding: 4px 8px;
                            color: #f0b90b;
                            font-weight: bold;
                        }
                        QComboBox:hover {
                            border: 1px solid #f0b90b;
                        }
                        QComboBox::drop-down {
                            border: none;
                            width: 20px;
                        }
                        QComboBox QAbstractItemView {
                            background-color: #1e222d;
                            color: #eaecef;
                            selection-background-color: #2a2e39;
                        }
                    """)
                    self.param_table.setCellWidget(row, 1, combo)
                    
                    max_item = QTableWidgetItem("-")
                    max_item.setTextAlignment(Qt.AlignCenter)
                    max_item.setFlags(max_item.flags() & ~Qt.ItemIsEditable)
                    max_item.setBackground(QColor("#1a2332"))
                    self.param_table.setItem(row, 2, max_item)
                    
                    step_item = QTableWidgetItem("-")
                    step_item.setTextAlignment(Qt.AlignCenter)
                    step_item.setFlags(step_item.flags() & ~Qt.ItemIsEditable)
                    step_item.setBackground(QColor("#1a2332"))
                    self.param_table.setItem(row, 3, step_item)
                    
                    self._param_spinboxes[pr.name] = {
                        "combo": combo,
                        "values": pr.values,
                        "is_discrete": True,
                    }
                else:
                    min_item = QTableWidgetItem(str(pr.min_value))
                    min_item.setTextAlignment(Qt.AlignCenter)
                    self.param_table.setItem(row, 1, min_item)
                    
                    max_item = QTableWidgetItem(str(pr.max_value))
                    max_item.setTextAlignment(Qt.AlignCenter)
                    self.param_table.setItem(row, 2, max_item)
                    
                    step_item = QTableWidgetItem(str(pr.step))
                    step_item.setTextAlignment(Qt.AlignCenter)
                    self.param_table.setItem(row, 3, step_item)
                    
                    self._param_spinboxes[pr.name] = {
                        "min": min_item,
                        "max": max_item,
                        "step": step_item,
                        "is_discrete": False,
                    }
            
            self.param_table.resizeRowsToContents()
            self.param_table.blockSignals(False)
            
        except Exception as e:
            print(f"获取参数失败: {e}")
    
    def _get_param_ranges_from_table(self) -> list:
        """从表格获取参数范围"""
        from Strategy.parameter_optimizer import ParameterRange
        
        ranges = []
        for row in range(self.param_table.rowCount()):
            name = self.param_table.item(row, 0).text()
            spinboxes = self._param_spinboxes.get(name, {})
            
            if spinboxes.get("is_discrete"):
                combo = spinboxes.get("combo")
                if combo:
                    selected_value = combo.currentText()
                    values = spinboxes.get("values", [])
                    try:
                        idx = [str(v) for v in values].index(selected_value)
                        selected_actual = values[idx]
                    except ValueError:
                        selected_actual = values[0] if values else selected_value
                    
                    ranges.append(ParameterRange(
                        name=name,
                        min_value=0,
                        max_value=0,
                        values=values,
                    ))
            else:
                try:
                    min_val = float(spinboxes["min"].text()) if "min" in spinboxes else 0
                    max_val = float(spinboxes["max"].text()) if "max" in spinboxes else 100
                    step_val = float(spinboxes["step"].text()) if "step" in spinboxes else 1
                except (ValueError, AttributeError):
                    min_val = 0
                    max_val = 100
                    step_val = 1
                
                if min_val == int(min_val) and max_val == int(max_val) and step_val == int(step_val):
                    ranges.append(ParameterRange(
                        name=name,
                        min_value=int(min_val),
                        max_value=int(max_val),
                        step=int(step_val),
                    ))
                else:
                    ranges.append(ParameterRange(
                        name=name,
                        min_value=min_val,
                        max_value=max_val,
                        step=step_val,
                    ))
        
        return ranges
    
    def _reset_param_ranges(self):
        """重置参数范围到默认值"""
        self._update_param_ranges_display()
        self.opt_result_text.append("🔄 参数已重置为默认值")
    
    def _on_param_table_changed(self, item):
        """参数表格内容变化时的回调"""
        if item.column() == 0:
            return
        try:
            value = float(item.text())
            item.setBackground(QColor("#1a2332"))
        except ValueError:
            item.setText("0")
            item.setBackground(QColor("#3d2b2b"))
    
    def _reset_backtest_params(self):
        """重置回测参数到默认值"""
        self.data_limit.setValue(1000)
        self.capital.setValue(10000)
        self.leverage.setValue(5)
        self.stop_loss.setValue(0)
        self.take_profit.setValue(0)
        self.position_size.setValue(100)
        self._on_strategy_changed(self.strategy.currentText())
        self.log_output.append("🔄 回测参数已重置为默认值")
    
    def _stylesheet(self):
        return """
            QMainWindow, QWidget { background-color: #0b0e11; color: #eaecef; font-family: 'Segoe UI', sans-serif; }
            QGroupBox { border: 1px solid #2a2e39; border-radius: 8px; margin-top: 12px; padding-top: 12px; font-weight: bold; color: #eaecef; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
            QPushButton { background-color: #2a2e39; border: none; border-radius: 6px; padding: 10px 20px; color: #eaecef; font-weight: bold; font-size: 13px; }
            QPushButton:hover { background-color: #363a45; }
            QPushButton:pressed { background-color: #1e222d; }
            QPushButton#primary { background-color: #f0b90b; color: #0b0e11; }
            QPushButton#primary:hover { background-color: #d4a50a; }
            QComboBox, QSpinBox, QDoubleSpinBox { background-color: #1e222d; border: 1px solid #2a2e39; border-radius: 6px; padding: 8px; color: #eaecef; }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 5px solid #f0b90b; }
            QTextEdit { background-color: #1e222d; border: 1px solid #2a2e39; border-radius: 6px; color: #eaecef; font-family: 'Consolas', monospace; font-size: 12px; }
            QTableWidget { background-color: #1e222d; border: 1px solid #2a2e39; border-radius: 6px; gridline-color: #2a2e39; color: #eaecef; }
            QTableWidget::item { padding: 6px; }
            QTableWidget::item:selected { background-color: #2a2e39; }
            QHeaderView::section { background-color: #1e222d; border: none; padding: 8px; color: #848e9c; font-weight: bold; }
            QTabWidget::pane { border: 1px solid #2a2e39; border-radius: 6px; }
            QTabBar::tab { background-color: #1e222d; border: none; padding: 10px 24px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; color: #848e9c; }
            QTabBar::tab:selected { background-color: #2a2e39; color: #f0b90b; }
            QLabel { color: #eaecef; }
            QScrollBar:vertical { background-color: #1e222d; width: 8px; border-radius: 4px; }
            QScrollBar::handle:vertical { background-color: #2a2e39; border-radius: 4px; }
            QProgressBar { border: none; border-radius: 4px; background-color: #1e222d; text-align: center; color: #eaecef; }
            QProgressBar::chunk { background-color: #f0b90b; border-radius: 4px; }
        """
    
    def _create_header(self) -> QWidget:
        header = QFrame()
        header.setFixedHeight(50)
        header.setStyleSheet("QFrame { background-color: #1e222d; border-radius: 8px; }")
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 0, 16, 0)
        
        title = QLabel("⚡ 量化交易系统")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #f0b90b;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        self.status = QLabel("● 就绪")
        self.status.setStyleSheet("color: #0ecb81; font-size: 13px;")
        layout.addWidget(self.status)
        
        return header
    
    def _stylesheet(self):
        return """
            QMainWindow, QWidget { background-color: #0b0e11; color: #eaecef; font-family: 'Segoe UI', sans-serif; }
            QGroupBox { border: 1px solid #2a2e39; border-radius: 8px; margin-top: 12px; padding-top: 12px; font-weight: bold; color: #eaecef; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
            QPushButton { background-color: #2a2e39; border: none; border-radius: 6px; padding: 10px 20px; color: #eaecef; font-weight: bold; font-size: 13px; }
            QPushButton:hover { background-color: #363a45; }
            QPushButton:pressed { background-color: #1e222d; }
            QPushButton#primary { background-color: #f0b90b; color: #0b0e11; }
            QPushButton#primary:hover { background-color: #d4a50a; }
            QComboBox, QSpinBox, QDoubleSpinBox { background-color: #1e222d; border: 1px solid #2a2e39; border-radius: 6px; padding: 8px; color: #eaecef; }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 5px solid #f0b90b; }
            QTextEdit { background-color: #1e222d; border: 1px solid #2a2e39; border-radius: 6px; color: #eaecef; font-family: 'Consolas', monospace; font-size: 12px; }
            QTableWidget { background-color: #1e222d; border: 1px solid #2a2e39; border-radius: 6px; gridline-color: #2a2e39; color: #eaecef; }
            QTableWidget::item { padding: 6px; }
            QTableWidget::item:selected { background-color: #2a2e39; }
            QHeaderView::section { background-color: #1e222d; border: none; padding: 8px; color: #848e9c; font-weight: bold; }
            QTabWidget::pane { border: 1px solid #2a2e39; border-radius: 6px; }
            QTabBar::tab { background-color: #1e222d; border: none; padding: 10px 24px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; color: #848e9c; }
            QTabBar::tab:selected { background-color: #2a2e39; color: #f0b90b; }
            QLabel { color: #eaecef; }
            QScrollBar:vertical { background-color: #1e222d; width: 8px; border-radius: 4px; }
            QScrollBar::handle:vertical { background-color: #2a2e39; border-radius: 4px; }
            QProgressBar { border: none; border-radius: 4px; background-color: #1e222d; text-align: center; color: #eaecef; }
            QProgressBar::chunk { background-color: #f0b90b; border-radius: 4px; }
        """
    
    def _create_header(self) -> QWidget:
        header = QFrame()
        header.setFixedHeight(50)
        header.setStyleSheet("QFrame { background-color: #1e222d; border-radius: 8px; }")
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 0, 16, 0)
        
        title = QLabel("⚡ 量化交易系统")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #f0b90b;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        self.status = QLabel("● 就绪")
        self.status.setStyleSheet("color: #0ecb81;")
        layout.addWidget(self.status)
        
        return header
    
    def _on_leverage_changed(self, value: int):
        if value > 20:
            self.risk_warning.setText("⚠️ 高杠杆风险极高！")
        elif value > 10:
            self.risk_warning.setText("⚡ 中高杠杆，谨慎操作")
        else:
            self.risk_warning.setText("")
    
    def _export_report(self):
        if self._last_result is None or self._last_data is None:
            QMessageBox.warning(self, "提示", "请先运行回测")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存回测报告",
            f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "HTML文件 (*.html)"
        )
        
        if file_path:
            try:
                self._last_visualizer.save_html_report(
                    self._last_data,
                    self._last_result,
                    file_path,
                    self._last_config.interval,
                    f"回测报告 - {self._last_config.symbol}"
                )
                QMessageBox.information(self, "成功", f"报告已保存到:\n{file_path}")
                
                reply = QMessageBox.question(
                    self,
                    "打开报告",
                    "是否立即打开报告？",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    webbrowser.open(f"file://{os.path.abspath(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    
    def _run_enhanced_backtest(self):
        """运行强化回测"""
        if self._last_result is None:
            QMessageBox.warning(self, "提示", "请先运行基础回测")
            return
        
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QCheckBox, QSpinBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("强化回测配置")
        dialog.setMinimumWidth(350)
        dialog.setStyleSheet("""
            QDialog { background-color: #1e222d; }
            QLabel { color: #eaecef; }
            QCheckBox { color: #eaecef; }
            QSpinBox { background-color: #0b0e11; border: 1px solid #2a2e39; border-radius: 4px; padding: 4px; color: #eaecef; }
            QGroupBox { color: #f0b90b; border: 1px solid #2a2e39; border-radius: 4px; margin-top: 10px; padding-top: 10px; }
        """)
        
        layout = QVBoxLayout(dialog)
        
        scenario_check = QCheckBox("市场情景测试")
        scenario_check.setChecked(True)
        layout.addWidget(scenario_check)
        
        mc_check = QCheckBox("蒙特卡洛模拟")
        mc_check.setChecked(True)
        layout.addWidget(mc_check)
        
        mc_layout = QHBoxLayout()
        mc_layout.addWidget(QLabel("模拟次数:"))
        mc_simulations = QSpinBox()
        mc_simulations.setRange(10, 200)
        mc_simulations.setValue(30)
        mc_layout.addWidget(mc_simulations)
        layout.addLayout(mc_layout)
        
        wf_check = QCheckBox("滚动窗口测试")
        wf_check.setChecked(True)
        layout.addWidget(wf_check)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        self.log_output.append(f"[{datetime.now():%H:%M:%S}] 🔬 开始强化回测...")
        self.enhanced_btn.setEnabled(False)
        
        config = {
            'run_scenarios': scenario_check.isChecked(),
            'run_monte_carlo': mc_check.isChecked(),
            'run_walk_forward': wf_check.isChecked(),
            'n_simulations': mc_simulations.value(),
            'strategy_name': self._last_result.strategy_name,
            'strategy_params': self._last_result.strategy_params.copy() if self._last_result.strategy_params else {},
            'data': self._last_data.copy() if self._last_data is not None else None,
            'backtest_config': self._last_config,
        }
        
        self._enhanced_worker = EnhancedBacktestWorker(config)
        self._enhanced_worker.progress.connect(self._on_enhanced_progress)
        self._enhanced_worker.finished.connect(self._on_enhanced_finished)
        self._enhanced_worker.start()
    
    def _on_enhanced_progress(self, msg: str):
        """强化回测进度回调"""
        self.log_output.append(msg)
    
    def _on_enhanced_finished(self, results: dict):
        """强化回测完成回调"""
        self.enhanced_btn.setEnabled(True)
        
        if 'error' in results:
            self.log_output.append(f"[{datetime.now():%H:%M:%S}] ❌ 强化回测失败: {results['error']}")
            return
        
        if 'scenarios' in results:
            self.log_output.append(f"\n【市场情景测试结果】")
            for scenario, result in results['scenarios'].items():
                status = "✅" if result.total_return_pct > 0 else "❌"
                self.log_output.append(
                    f"  {status} {scenario.value}: 收益={result.total_return_pct:.2f}%, "
                    f"回撤={result.max_drawdown_pct:.2f}%, 夏普={result.sharpe_ratio:.2f}"
                )
        elif 'scenarios_error' in results:
            self.log_output.append(f"\n【市场情景测试失败】: {results['scenarios_error']}")
        
        if 'monte_carlo' in results:
            mc = results['monte_carlo']
            self.log_output.append(f"\n【蒙特卡洛模拟结果】")
            self.log_output.append(f"  平均收益: {mc.mean_return:.2f}%")
            self.log_output.append(f"  收益标准差: {mc.std_return:.2f}%")
            self.log_output.append(f"  VaR(95%): {mc.var_95:.2f}%")
            self.log_output.append(f"  盈利概率: {mc.profit_probability:.1f}%")
        elif 'monte_carlo_error' in results:
            self.log_output.append(f"\n【蒙特卡洛模拟失败】: {results['monte_carlo_error']}")
        
        if 'walk_forward' in results:
            wf = results['walk_forward']
            self.log_output.append(f"\n【滚动窗口测试结果】")
            self.log_output.append(f"  测试窗口数: {len(wf.window_results)}")
            self.log_output.append(f"  平均收益: {wf.avg_return:.2f}%")
            self.log_output.append(f"  一致性得分: {wf.consistency_score:.1f}%")
        elif 'walk_forward_error' in results:
            self.log_output.append(f"\n【滚动窗口测试失败】: {results['walk_forward_error']}")
        
        self.log_output.append(f"\n[{datetime.now():%H:%M:%S}] ✅ 强化回测完成")
        QMessageBox.information(self, "完成", "强化回测已完成，结果已显示在日志中")
    
    def _sync_to_live_trading(self):
        """同步回测参数到实盘交易"""
        if self._last_result is None:
            QMessageBox.warning(self, "提示", "请先运行回测")
            return
        
        result = self._last_result
        config = self._last_config
        
        self.main_tabs.setCurrentIndex(2)
        
        symbol = config.symbol
        idx = self.live_symbol.findText(symbol)
        if idx >= 0:
            self.live_symbol.setCurrentIndex(idx)
        
        self.live_leverage.setValue(int(config.leverage))
        self.live_position_size.setValue(float(config.position_size * 100))
        self.live_stop_loss.setValue(float(config.stop_loss_pct))
        self.live_take_profit.setValue(float(config.take_profit_pct))
        
        strategy_name = result.strategy_name
        idx = self.live_strategy.findText(strategy_name)
        if idx >= 0:
            self.live_strategy.setCurrentIndex(idx)
        
        if hasattr(self, '_param_widgets'):
            for name, widget in self._param_widgets.items():
                if name in result.strategy_params:
                    value = result.strategy_params[name]
                    if isinstance(widget, QDoubleSpinBox):
                        widget.setValue(float(value))
                    elif isinstance(widget, QSpinBox):
                        widget.setValue(int(value))
                    elif isinstance(widget, QComboBox):
                        idx = widget.findText(str(value))
                        if idx >= 0:
                            widget.setCurrentIndex(idx)
        
        self.live_log.append(f"[{datetime.now():%H:%M:%S}] 📋 已同步回测参数到实盘交易")
        self.live_log.append(f"   交易对: {symbol}")
        self.live_log.append(f"   杠杆: {config.leverage}x")
        self.live_log.append(f"   仓位: {config.position_size*100:.0f}%")
        self.live_log.append(f"   止损: {config.stop_loss_pct}%")
        self.live_log.append(f"   止盈: {config.take_profit_pct}%")
        
        issues = self._check_live_trading_issues(result)
        if issues:
            self.live_log.append(f"\n⚠️ 风险提示:")
            for issue in issues:
                self.live_log.append(f"   • {issue}")
            QMessageBox.warning(self, "风险提示", "检测到以下问题:\n\n" + "\n".join(f"• {i}" for i in issues))
        else:
            QMessageBox.information(self, "成功", "参数已同步到实盘交易\n请在实盘页面配置API后启动")
    
    def _check_live_trading_issues(self, result) -> list[str]:
        """检测实盘交易潜在问题"""
        issues = []
        
        if result.win_rate < 30:
            issues.append(f"胜率过低 ({result.win_rate:.1f}%)，建议优化策略")
        
        if result.max_drawdown_pct < -30:
            issues.append(f"最大回撤过大 ({result.max_drawdown_pct:.1f}%)，风险较高")
        
        if result.total_trades < 10:
            issues.append(f"交易次数过少 ({result.total_trades}次)，样本不足")
        
        if result.profit_factor < 1:
            issues.append(f"盈亏比小于1 ({result.profit_factor:.2f})，策略亏损")
        
        if result.sharpe_ratio < 0:
            issues.append(f"夏普比率为负 ({result.sharpe_ratio:.2f})，收益不佳")
        
        if self.live_leverage.value() > 20:
            issues.append(f"杠杆过高 ({self.live_leverage.value()}x)，爆仓风险大")
        
        if self.live_stop_loss.value() == 0:
            issues.append("未设置止损，风险不可控")
        
        return issues
    
    def _run_optimization(self):
        """运行参数优化"""
        metric_map = {
            "夏普比率": "sharpe_ratio",
            "总收益率": "total_return",
            "综合得分": "composite",
        }
        
        param_ranges = self._get_param_ranges_from_table()
        
        if not param_ranges:
            QMessageBox.warning(self, "提示", "请设置有效的参数范围")
            return
        
        config = {
            "symbol": self.opt_symbol.currentText(),
            "interval": self.opt_interval.currentText(),
            "data_num": self.opt_data_limit.value(),
            "strategy": self.STRATEGIES.get(self.opt_strategy.currentText()),
            "initial_capital": self.opt_capital.value(),
            "leverage": 5,
            "stop_loss_pct": 0,
            "take_profit_pct": 0,
            "opt_method": self.opt_method.currentText(),
            "iterations": self.opt_iterations.value(),
            "optimization_metric": metric_map.get(self.opt_metric.currentText(), "sharpe_ratio"),
            "param_ranges": param_ranges,
        }
        
        self.opt_result_text.clear()
        self.opt_result_text.append(f"{'='*60}")
        self.opt_result_text.append(f"[{datetime.now():%H:%M:%S}] 🔍 启动参数探索任务")
        self.opt_result_text.append(f"{'='*60}")
        
        self.start_opt_btn.setEnabled(False)
        self.stop_opt_btn.setEnabled(True)
        self.opt_progress.setVisible(True)
        self.opt_progress.setRange(0, 0)
        self.apply_params_btn.setEnabled(False)
        self.save_params_btn.setEnabled(False)
        
        self._optimizer_worker = OptimizerWorker(config)
        self._optimizer_worker.progress.connect(self._on_optimization_progress)
        self._optimizer_worker.log_message.connect(lambda m: self.opt_result_text.append(m))
        self._optimizer_worker.iteration_log.connect(lambda m: self.opt_result_text.append(m))
        self._optimizer_worker.finished.connect(self._on_optimization_finished)
        self._optimizer_worker.error.connect(self._on_optimization_error)
        self._optimizer_worker.start()
    
    def _on_optimization_progress(self, current: int, total: int):
        """优化进度更新"""
        self.opt_progress.setRange(0, total)
        self.opt_progress.setValue(current)
    
    def _stop_optimization(self):
        """停止参数优化"""
        if hasattr(self, '_optimizer_worker') and self._optimizer_worker.isRunning():
            self._optimizer_worker.stop()
            self.opt_result_text.append(f"\n⏹ 用户停止优化...")
            self.start_opt_btn.setEnabled(True)
            self.stop_opt_btn.setEnabled(False)
    
    def _on_optimization_finished(self, result):
        """优化完成"""
        self._last_optimization_result = result
        
        self.opt_result_text.append(f"\n{'═'*38}")
        self.opt_result_text.append(f"📋 {result.optimization_method} × {result.total_iterations}次 {result.execution_time:.1f}s")
        self.opt_result_text.append(f"{'═'*38}")
        
        params_str = " ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in result.best_params.items()
        )
        self.opt_result_text.append(f"✅ {params_str}")
        self.opt_result_text.append(f"   得分{result.best_score:.3f}")
        
        self.opt_result_text.append(f"\nTop 5:")
        for i, r in enumerate(result.to_dict()["top_results"][:5], 1):
            self.opt_result_text.append(
                f" {i}. 得分{r['score']:.3f} │ 收益{r['total_return_pct']:.1f}% │ 回撤{r['max_drawdown_pct']:.1f}%"
            )
        
        self.start_opt_btn.setEnabled(True)
        self.stop_opt_btn.setEnabled(False)
        self.opt_progress.setVisible(False)
        self.apply_params_btn.setEnabled(True)
        self.save_params_btn.setEnabled(True)
    
    def _save_optimization_params(self):
        """保存优化参数到本地"""
        if not hasattr(self, '_last_optimization_result') or not self._last_optimization_result:
            QMessageBox.warning(self, "提示", "没有可保存的优化结果")
            return
        
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QCheckBox, QGroupBox
        
        result = self._last_optimization_result
        top_results = result.to_dict().get("top_results", [])[:10]
        
        if not top_results:
            QMessageBox.warning(self, "提示", "没有可保存的参数")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("保存参数")
        dialog.setMinimumWidth(500)
        dialog.setStyleSheet("""
            QDialog { background-color: #1e222d; }
            QLabel { color: #eaecef; }
            QCheckBox { color: #eaecef; }
            QGroupBox { color: #f0b90b; border: 1px solid #2a2e39; border-radius: 4px; margin-top: 10px; padding-top: 10px; }
            QScrollArea { border: none; }
        """)
        
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("选择要保存的参数组合:"))
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        checkboxes = []
        for i, r in enumerate(top_results):
            params = r.get('params', {})
            score = r.get('score', 0)
            ret = r.get('total_return_pct', 0)
            dd = r.get('max_drawdown_pct', 0)
            
            params_str = ", ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in params.items()
            )
            
            cb = QCheckBox(f"#{i+1} 得分{score:.3f} │ 收益{ret:.1f}% │ 回撤{dd:.1f}%")
            cb.setChecked(i < 3)
            cb.setStyleSheet("QCheckBox { spacing: 8px; }")
            
            params_label = QLabel(f"   {params_str}")
            params_label.setStyleSheet("color: #848e9c; font-size: 11px; margin-left: 20px;")
            
            checkbox_widget = QWidget()
            cb_layout = QVBoxLayout(checkbox_widget)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            cb_layout.setSpacing(2)
            cb_layout.addWidget(cb)
            cb_layout.addWidget(params_label)
            
            scroll_layout.addWidget(checkbox_widget)
            checkboxes.append((cb, params, r))
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("保存名称:"))
        name_edit = QLineEdit(f"optimization_{datetime.now():%Y%m%d_%H%M%S}")
        name_edit.setStyleSheet("background-color: #0b0e11; border: 1px solid #2a2e39; border-radius: 4px; padding: 4px; color: #eaecef;")
        name_layout.addWidget(name_edit)
        layout.addLayout(name_layout)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        selected_params = []
        for cb, params, r in checkboxes:
            if cb.isChecked():
                selected_params.append({
                    'params': params,
                    'score': r.get('score', 0),
                    'total_return_pct': r.get('total_return_pct', 0),
                    'max_drawdown_pct': r.get('max_drawdown_pct', 0),
                    'win_rate': r.get('win_rate', 0),
                    'total_trades': r.get('total_trades', 0),
                })
        
        if not selected_params:
            QMessageBox.warning(self, "提示", "请至少选择一组参数")
            return
        
        import json
        from pathlib import Path
        
        save_dir = Path(__file__).parent.parent / "saved_params"
        save_dir.mkdir(exist_ok=True)
        
        save_name = name_edit.text()
        save_file = save_dir / f"{save_name}.json"
        
        save_data = {
            'name': save_name,
            'strategy': self.opt_strategy_combo.currentText(),
            'symbol': self.symbol_combo.currentText(),
            'optimization_method': result.optimization_method,
            'created_at': datetime.now().isoformat(),
            'params': selected_params,
        }
        
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        self.opt_result_text.append(f"\n[{datetime.now():%H:%M:%S}] 💾 已保存 {len(selected_params)} 组参数到:")
        self.opt_result_text.append(f"   {save_file}")
        QMessageBox.information(self, "成功", f"参数已保存到:\n{save_file}")
    
    def _on_optimization_error(self, msg: str):
        """优化错误"""
        self.opt_result_text.append(f"[{datetime.now():%H:%M:%S}] 错误: {msg}")
        QMessageBox.critical(self, "错误", msg)
        self.start_opt_btn.setEnabled(True)
        self.opt_progress.setVisible(False)
    
    def _apply_best_params_to_backtest(self):
        """将最优参数应用到回测"""
        if not hasattr(self, '_last_optimization_result') or self._last_optimization_result is None:
            QMessageBox.warning(self, "提示", "请先运行参数探索")
            return
        
        result = self._last_optimization_result
        params = result.best_params
        
        self.main_tabs.setCurrentIndex(0)
        
        if hasattr(self, 'opt_symbol'):
            symbol = self.opt_symbol.currentText()
            idx = self.symbol.findText(symbol)
            if idx >= 0:
                self.symbol.setCurrentIndex(idx)
        
        if hasattr(self, 'opt_interval'):
            interval = self.opt_interval.currentText()
            idx = self.interval.findText(interval)
            if idx >= 0:
                self.interval.setCurrentIndex(idx)
        
        if hasattr(self, 'opt_data_limit'):
            self.data_limit.setValue(self.opt_data_limit.value())
        
        if hasattr(self, 'opt_capital'):
            self.capital.setValue(self.opt_capital.value())
        
        if hasattr(self, 'opt_strategy'):
            strategy_name = self.opt_strategy.currentText()
            idx = self.strategy.findText(strategy_name)
            if idx >= 0:
                self.strategy.setCurrentIndex(idx)
        
        if "leverage" in params:
            self.leverage.setValue(int(params["leverage"]))
        if "stop_loss_pct" in params:
            self.stop_loss.setValue(float(params["stop_loss_pct"]))
        if "take_profit_pct" in params:
            self.take_profit.setValue(float(params["take_profit_pct"]))
        if "position_size" in params:
            self.position_size.setValue(float(params["position_size"]) * 100)
        
        for name, value in params.items():
            if name in self._param_widgets:
                widget = self._param_widgets[name]
                if isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(value))
                elif isinstance(widget, QSpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, QComboBox):
                    idx = widget.findText(str(value))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                    else:
                        idx = widget.findData(str(value))
                        if idx >= 0:
                            widget.setCurrentIndex(idx)
        
        self.log_output.append(f"[{datetime.now():%H:%M:%S}] 已应用最优参数到回测配置")
        QMessageBox.information(self, "成功", "最优参数已应用到回测配置\n请在回测页面开始回测")
    
    def _create_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("QFrame { background-color: #1e222d; border-radius: 8px; }")
        metrics_layout = QHBoxLayout(metrics_frame)
        metrics_layout.setContentsMargins(12, 12, 12, 12)
        metrics_layout.setSpacing(12)
        
        self.metrics = {}
        for key, title, unit in [
            ("return", "总收益率", "%"),
            ("drawdown", "最大回撤", "%"),
            ("sharpe", "夏普比率", ""),
            ("winrate", "胜率", "%"),
            ("profit", "盈亏比", ""),
            ("trades", "交易次数", ""),
        ]:
            card = MetricCard(title, unit)
            self.metrics[key] = card
            metrics_layout.addWidget(card)
        
        layout.addWidget(metrics_frame)
        
        tabs = QTabWidget()
        
        report_tab = QWidget()
        report_layout = QVBoxLayout(report_tab)
        self.report_output = QTextEdit()
        self.report_output.setReadOnly(True)
        report_layout.addWidget(self.report_output)
        tabs.addTab(report_tab, "回测报告")
        
        trades_tab = QWidget()
        trades_layout = QVBoxLayout(trades_tab)
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(5)
        self.trades_table.setHorizontalHeaderLabels(["开仓时间", "平仓时间", "开仓价", "平仓价", "盈亏"])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        trades_layout.addWidget(self.trades_table)
        tabs.addTab(trades_tab, "交易记录")
        
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        tabs.addTab(log_tab, "日志")
        
        layout.addWidget(tabs, 1)
        
        return panel
    
    def _on_strategy_changed(self, name: str):
        from Strategy.templates import get_strategy
        
        strategy = get_strategy(self.STRATEGIES.get(name))
        info = strategy.get_info()
        
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self._param_widgets = {}
        row_idx = 0
        for p in info.get("parameters", []):
            row, col = row_idx // 2, (row_idx % 2) * 2
            
            if p.get("options"):
                self.params_layout.addWidget(QLabel(p["display_name"]), row, col)
                
                combo = QComboBox()
                combo.addItems(p["options"])
                combo.setCurrentText(str(p.get("default", p["options"][0])))
                self._param_widgets[p["name"]] = combo
                self.params_layout.addWidget(combo, row, col + 1)
                row_idx += 1
                continue
            
            self.params_layout.addWidget(QLabel(p["display_name"]), row, col)
            
            min_val = p.get("min")
            max_val = p.get("max")
            default_val = p.get("default", 0)
            
            if min_val is None:
                min_val = 0
            if max_val is None:
                max_val = 1000
            
            if p["type"] == "float":
                w = QDoubleSpinBox()
                w.setRange(float(min_val), float(max_val))
                w.setValue(float(default_val))
                w.setSingleStep(0.1)
            else:
                w = QSpinBox()
                w.setRange(int(min_val), int(max_val))
                w.setValue(int(default_val))
            
            self._param_widgets[p["name"]] = w
            self.params_layout.addWidget(w, row, col + 1)
            row_idx += 1
    
    def _run(self):
        params = {}
        for n, w in self._param_widgets.items():
            if isinstance(w, QComboBox):
                params[n] = w.currentText()
            else:
                params[n] = w.value()
        
        config = {
            "symbol": self.symbol.currentText(),
            "interval": self.interval.currentText(),
            "data_num": self.data_limit.value(),
            "strategy": self.STRATEGIES.get(self.strategy.currentText()),
            "strategy_params": params,
            "initial_capital": self.capital.value(),
            "leverage": self.leverage.value(),
            "stop_loss_pct": self.stop_loss.value(),
            "take_profit_pct": self.take_profit.value(),
            "position_size": self.position_size.value() / 100,
        }
        
        self.log_output.clear()
        self.log_output.append(f"{'='*60}")
        self.log_output.append(f"[{datetime.now():%H:%M:%S}] 🚀 启动回测任务")
        self.log_output.append(f"{'='*60}")
        
        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("● 运行中")
        self.status.setStyleSheet("color: #f0b90b;")
        
        self._worker = BacktestWorker(config)
        self._worker.progress.connect(lambda m: self.log_output.append(m))
        self._worker.trade_log.connect(lambda m: self.log_output.append(m))
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
    
    def _on_finished(self, data):
        result = data["result"]
        report = data["report"]
        
        self._last_result = result
        self._last_data = data.get("data")
        self._last_visualizer = data.get("visualizer")
        self._last_config = data.get("config")
        
        self.metrics["return"].set_value(result.total_return_pct, result.total_return_pct > 0)
        self.metrics["drawdown"].set_value(result.max_drawdown_pct, False)
        self.metrics["sharpe"].set_value(result.sharpe_ratio, result.sharpe_ratio > 0)
        self.metrics["winrate"].set_value(result.win_rate, result.win_rate > 50)
        self.metrics["profit"].set_value(result.profit_factor, result.profit_factor > 1)
        self.metrics["trades"].set_value(result.total_trades)
        
        self.report_output.setText(report.format_text_report())
        
        trades = result.completed_trades
        self.trades_table.setRowCount(len(trades))
        for i, t in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(t.entry_time)))
            self.trades_table.setItem(i, 1, QTableWidgetItem(str(t.exit_time)))
            self.trades_table.setItem(i, 2, QTableWidgetItem(f"{t.entry_price:.2f}"))
            self.trades_table.setItem(i, 3, QTableWidgetItem(f"{t.exit_price:.2f}"))
            
            pnl_item = QTableWidgetItem(f"{t.pnl:.2f}")
            pnl_item.setForeground(QColor("#0ecb81") if t.pnl > 0 else QColor("#f6465d"))
            self.trades_table.setItem(i, 4, pnl_item)
        
        self.log_output.append(f"[{datetime.now():%H:%M:%S}] 完成: {result.total_trades}笔交易, 胜率{result.win_rate:.1f}%")
        
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.sync_to_live_btn.setEnabled(True)
        self.enhanced_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status.setText("● 完成")
        self.status.setStyleSheet("color: #0ecb81;")
    
    def _on_error(self, msg):
        self.log_output.append(f"[{datetime.now():%H:%M:%S}] 错误: {msg}")
        QMessageBox.critical(self, "错误", msg)
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.sync_to_live_btn.setEnabled(False)
        self.enhanced_btn.setEnabled(False)
        self.progress.setVisible(False)
        self.status.setText("● 错误")
        self.status.setStyleSheet("color: #f6465d;")

    def _shutdown_thread(self, worker: QThread | None, name: str, timeout_ms: int = 1200) -> None:
        """安全停止后台线程，避免UI关闭时卡死。"""
        if worker is None:
            return

        try:
            if hasattr(worker, "stop"):
                stopped = worker.stop(timeout_ms)  # type: ignore[misc]
                if stopped is True:
                    return
            else:
                worker.requestInterruption()

            if worker.isRunning() and not worker.wait(timeout_ms):
                logger.warning(f"线程 {name} 在 {timeout_ms}ms 内未退出，执行强制终止")
                worker.terminate()
                worker.wait(300)
        except Exception as e:
            logger.warning(f"停止线程 {name} 失败: {e}")
    
    def closeEvent(self, event):
        if hasattr(self, '_status_timer') and self._status_timer:
            self._status_timer.stop()

        if hasattr(self, '_optimizer_worker') and self._optimizer_worker:
            self._shutdown_thread(self._optimizer_worker, "OptimizerWorker")

        if hasattr(self, '_enhanced_worker') and self._enhanced_worker:
            self._shutdown_thread(self._enhanced_worker, "EnhancedBacktestWorker")

        if hasattr(self, '_worker') and self._worker:
            self._shutdown_thread(self._worker, "BacktestWorker")

        if hasattr(self, '_live_worker') and self._live_worker:
            self._shutdown_thread(self._live_worker, "LiveAccountWorker")

        if hasattr(self, '_trader') and self._trader:
            try:
                self._trader.stop()
            except Exception as e:
                logger.warning(f"停止交易器失败: {e}")

        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = TradingUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
