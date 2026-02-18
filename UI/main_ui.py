"""
量化交易系统 - 专业回测界面

功能：
- 多交易对、多周期选择
- 策略参数动态配置
- 止损止盈、杠杆设置
- 风险提示机制
- 回测可视化
- 异步参数优化
- UI悬停提示
"""

import sys
import traceback
import webbrowser
import os
from datetime import datetime
from typing import Any

import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QUrl
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QPushButton, QTextEdit,
    QSpinBox, QDoubleSpinBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QMessageBox,
    QProgressBar, QFrame, QGridLayout, QScrollArea, QSizePolicy,
    QFileDialog, QCheckBox
)
from PyQt5.QtGui import QFont, QColor, QPalette, QPainter
from PyQt5.QtCore import Qt as QtCore

from UI.widgets import (
    TooltipLabel, TooltipComboBox, TooltipSpinBox, 
    TooltipDoubleSpinBox, TooltipButton, TOOLTIP_TEXTS,
    OptimizerWorker, ValidationWorker, OptimizedTextEdit,
    OptimizedTableWidget, WorkerManager
)

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BacktestWorker(QThread):
    progress = pyqtSignal(str)
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
            
            self.progress.emit(f"正在初始化数据服务...")
            
            from Data.data_service import get_data_service, DataServiceConfig, RegionRestrictedError, DataSourceError
            from Strategy.templates import get_strategy
            from backtest.engine import BacktestEngine
            from backtest.report import BacktestReport
            from backtest.visualization import BacktestVisualizer
            from core.config import BacktestConfig
            
            service = get_data_service(DataServiceConfig(prefer_database=True, auto_init=False))
            
            self.progress.emit(f"正在加载 {symbol} 数据...")
            
            try:
                # UI回测优先保证响应速度，避免高周期触发超大基础K线聚合
                data = service.get_klines(symbol, interval, max(100, int(data_num)))
            except RegionRestrictedError as e:
                self.error.emit(f"API访问受限，请配置代理:\n{str(e)}")
                return
            except DataSourceError as e:
                self.error.emit(f"数据源错误:\n{str(e)}")
                return
            except Exception as e:
                self.error.emit(f"数据加载失败:\n{str(e)}")
                return
            
            if data.empty:
                self.error.emit(f"无法获取 {symbol} 数据\n请检查网络连接或交易对是否正确")
                return
            
            self.progress.emit(f"数据加载完成: {len(data)}条")
            
            self.progress.emit("正在初始化策略...")
            strategy = get_strategy(strategy_name, strategy_params)
            config = BacktestConfig(
                symbol=symbol,
                interval=interval,
                initial_capital=initial_capital,
                data_limit=data_num,
                leverage=leverage,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )
            
            self.progress.emit("正在执行回测...")
            engine = BacktestEngine(strategy, config)
            result = engine.run(data)
            
            self.progress.emit("正在生成报告...")
            report = BacktestReport(result, strategy_name, symbol, interval)
            
            visualizer = BacktestVisualizer()
            
            self.progress.emit("回测完成!")
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
        self._worker_manager = WorkerManager()
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
        
        content = QHBoxLayout()
        content.addWidget(self._create_config_panel(), 1)
        content.addWidget(self._create_result_panel(), 2)
        layout.addLayout(content, 1)
    
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
    
    def _create_config_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet("QFrame { background-color: #1e222d; border-radius: 8px; }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        config_grid = QGridLayout()
        config_grid.setSpacing(10)
        
        config_grid.addWidget(TooltipLabel("交易对", TOOLTIP_TEXTS["symbol"]), 0, 0)
        self.symbol = TooltipComboBox(TOOLTIP_TEXTS["symbol"])
        self.symbol.addItems(self.SYMBOLS)
        config_grid.addWidget(self.symbol, 0, 1)
        
        config_grid.addWidget(TooltipLabel("周期", TOOLTIP_TEXTS["interval"]), 0, 2)
        self.interval = TooltipComboBox(TOOLTIP_TEXTS["interval"])
        self.interval.addItems(self.INTERVALS)
        self.interval.setCurrentIndex(3)
        config_grid.addWidget(self.interval, 0, 3)
        
        config_grid.addWidget(TooltipLabel("数据量", TOOLTIP_TEXTS["data_limit"]), 1, 0)
        self.data_limit = TooltipSpinBox(TOOLTIP_TEXTS["data_limit"])
        self.data_limit.setRange(100, 50000)
        self.data_limit.setValue(1000)
        config_grid.addWidget(self.data_limit, 1, 1)
        
        config_grid.addWidget(TooltipLabel("初始资金", TOOLTIP_TEXTS["initial_capital"]), 1, 2)
        self.capital = TooltipDoubleSpinBox(TOOLTIP_TEXTS["initial_capital"])
        self.capital.setRange(100, 1000000)
        self.capital.setValue(10000)
        config_grid.addWidget(self.capital, 1, 3)
        
        layout.addLayout(config_grid)
        
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("background-color: #2a2e39;")
        layout.addWidget(line1)
        
        risk_label = TooltipLabel("⚙️ 风险控制", "设置止损止盈和杠杆参数\n合理控制交易风险")
        risk_label.setStyleSheet("font-weight: bold; color: #f0b90b;")
        layout.addWidget(risk_label)
        
        risk_grid = QGridLayout()
        risk_grid.setSpacing(10)
        
        risk_grid.addWidget(TooltipLabel("杠杆倍数", TOOLTIP_TEXTS["leverage"]), 0, 0)
        self.leverage = TooltipSpinBox(TOOLTIP_TEXTS["leverage"])
        self.leverage.setRange(1, 125)
        self.leverage.setValue(5)
        self.leverage.valueChanged.connect(self._on_leverage_changed)
        risk_grid.addWidget(self.leverage, 0, 1)
        
        risk_grid.addWidget(TooltipLabel("止损率 (%)", TOOLTIP_TEXTS["stop_loss"]), 0, 2)
        self.stop_loss = TooltipDoubleSpinBox(TOOLTIP_TEXTS["stop_loss"])
        self.stop_loss.setRange(0, 100)
        self.stop_loss.setDecimals(2)
        self.stop_loss.setSingleStep(0.5)
        self.stop_loss.setValue(0)
        risk_grid.addWidget(self.stop_loss, 0, 3)
        
        risk_grid.addWidget(TooltipLabel("止盈率 (%)", TOOLTIP_TEXTS["take_profit"]), 1, 0)
        self.take_profit = TooltipDoubleSpinBox(TOOLTIP_TEXTS["take_profit"])
        self.take_profit.setRange(0, 1000)
        self.take_profit.setDecimals(2)
        self.take_profit.setSingleStep(0.5)
        self.take_profit.setValue(0)
        risk_grid.addWidget(self.take_profit, 1, 1)
        
        self.risk_warning = QLabel("")
        self.risk_warning.setStyleSheet("color: #f6465d; font-size: 11px;")
        self.risk_warning.setWordWrap(True)
        risk_grid.addWidget(self.risk_warning, 1, 2, 1, 2)
        
        layout.addLayout(risk_grid)
        
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #2a2e39;")
        layout.addWidget(line2)
        
        layout.addWidget(TooltipLabel("策略选择", TOOLTIP_TEXTS["strategy"]))
        
        self.strategy = TooltipComboBox(TOOLTIP_TEXTS["strategy"])
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
        
        line3 = QFrame()
        line3.setFrameShape(QFrame.HLine)
        line3.setStyleSheet("background-color: #2a2e39;")
        layout.addWidget(line3)
        
        opt_label = TooltipLabel("🤖 参数优化", "自动寻找最优策略参数\n提高策略盈利能力")
        opt_label.setStyleSheet("font-weight: bold; color: #f0b90b;")
        layout.addWidget(opt_label)
        
        opt_grid = QGridLayout()
        opt_grid.setSpacing(10)
        
        opt_grid.addWidget(TooltipLabel("优化方法", TOOLTIP_TEXTS["opt_method"]), 0, 0)
        self.opt_method = TooltipComboBox(TOOLTIP_TEXTS["opt_method"])
        self.opt_method.addItems(["网格搜索", "随机搜索", "贝叶斯优化"])
        opt_grid.addWidget(self.opt_method, 0, 1)
        
        opt_grid.addWidget(TooltipLabel("迭代次数", TOOLTIP_TEXTS["opt_iterations"]), 0, 2)
        self.opt_iterations = TooltipSpinBox(TOOLTIP_TEXTS["opt_iterations"])
        self.opt_iterations.setRange(10, 1000)
        self.opt_iterations.setValue(50)
        opt_grid.addWidget(self.opt_iterations, 0, 3)

        opt_grid.addWidget(TooltipLabel("优化广度", TOOLTIP_TEXTS["opt_breadth"]), 1, 0)
        self.opt_breadth = TooltipDoubleSpinBox(TOOLTIP_TEXTS["opt_breadth"])
        self.opt_breadth.setRange(0.2, 3.0)
        self.opt_breadth.setDecimals(1)
        self.opt_breadth.setSingleStep(0.1)
        self.opt_breadth.setValue(1.0)
        opt_grid.addWidget(self.opt_breadth, 1, 1)

        opt_grid.addWidget(TooltipLabel("优化深度", TOOLTIP_TEXTS["opt_depth"]), 1, 2)
        self.opt_depth = TooltipSpinBox(TOOLTIP_TEXTS["opt_depth"])
        self.opt_depth.setRange(1, 5)
        self.opt_depth.setValue(2)
        opt_grid.addWidget(self.opt_depth, 1, 3)
        
        opt_grid.addWidget(TooltipLabel("优化目标", TOOLTIP_TEXTS["opt_metric"]), 2, 0)
        self.opt_metric = TooltipComboBox(TOOLTIP_TEXTS["opt_metric"])
        self.opt_metric.addItems(["夏普比率", "总收益率", "综合得分"])
        opt_grid.addWidget(self.opt_metric, 2, 1)
        
        self.optimize_risk_params = QCheckBox("优化风险参数")
        self.optimize_risk_params.setToolTip("同时优化止损、止盈、杠杆参数")
        self.optimize_risk_params.setChecked(False)
        opt_grid.addWidget(self.optimize_risk_params, 2, 2, 1, 2)
        
        layout.addLayout(opt_grid)
        
        btn_layout = QHBoxLayout()
        
        self.run_btn = TooltipButton("▶ 开始回测", "运行策略回测\n查看策略表现")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._run)
        btn_layout.addWidget(self.run_btn)
        
        self.optimize_btn = TooltipButton("🔍 参数优化", "自动寻找最优参数\n可能需要较长时间")
        self.optimize_btn.clicked.connect(self._run_optimization)
        btn_layout.addWidget(self.optimize_btn)
        
        self.cancel_btn = TooltipButton("⏹ 取消", "取消正在进行的优化任务")
        self.cancel_btn.clicked.connect(self._cancel_optimization)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)
        
        self.export_btn = TooltipButton("📊 导出报告", "导出HTML回测报告\n可在浏览器中查看")
        self.export_btn.clicked.connect(self._export_report)
        self.export_btn.setEnabled(False)
        btn_layout.addWidget(self.export_btn)
        
        layout.addLayout(btn_layout)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        return panel
    
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
    
    def _run_optimization(self):
        """运行参数优化（异步）"""
        if self._optimizer_worker is not None and self._optimizer_worker.isRunning():
            QMessageBox.warning(self, "提示", "已有优化任务在运行中")
            return
        
        metric_map = {
            "夏普比率": "sharpe_ratio",
            "总收益率": "total_return",
            "综合得分": "composite",
        }
        
        opt_method_map = {
            "网格搜索": "grid_search",
            "随机搜索": "random_search",
            "贝叶斯优化": "bayesian_optimization",
        }
        
        config = {
            "symbol": self.symbol.currentText(),
            "interval": self.interval.currentText(),
            "data_limit": self.data_limit.value(),
            "strategy": self.STRATEGIES.get(self.strategy.currentText()),
            "strategy_params": {n: w.value() for n, w in self._param_widgets.items()},
            "initial_capital": self.capital.value(),
            "leverage": self.leverage.value(),
            "stop_loss_pct": self.stop_loss.value(),
            "take_profit_pct": self.take_profit.value(),
            "optimization_metric": metric_map.get(self.opt_metric.currentText(), "sharpe_ratio"),
            "opt_method": opt_method_map.get(self.opt_method.currentText(), "random_search"),
            "iterations": self.opt_iterations.value(),
            "opt_breadth": self.opt_breadth.value(),
            "opt_depth": self.opt_depth.value(),
            "optimize_risk_params": self.optimize_risk_params.isChecked(),
        }
        
        self.log_output.append_text(f"[{datetime.now():%H:%M:%S}] 开始参数优化...")
        self.optimize_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText("● 优化中...")
        self.status.setStyleSheet("color: #f0b90b;")
        
        self._optimizer_worker = OptimizerWorker(config)
        self._worker_manager.add_worker(self._optimizer_worker)
        self._optimizer_worker.progress.connect(self._on_optimization_progress)
        self._optimizer_worker.finished.connect(self._on_optimization_finished)
        self._optimizer_worker.error.connect(self._on_optimization_error)
        self._optimizer_worker.start()
    
    def _cancel_optimization(self):
        """取消优化"""
        if self._optimizer_worker is not None and self._optimizer_worker.isRunning():
            self._optimizer_worker.cancel()
            self.log_output.append_text(f"[{datetime.now():%H:%M:%S}] 正在取消优化...")
    
    def _on_optimization_progress(self, message: str, current: int, total: int):
        """优化进度回调"""
        safe_total = max(1, total)
        if total != 100:
            pct = int(current * 100 / safe_total)
            self.progress.setValue(max(0, min(100, pct)))
        else:
            self.progress.setValue(max(0, min(100, current)))
        self.log_output.append_text(f"[{datetime.now():%H:%M:%S}] {message}")
    
    def _on_optimization_finished(self, data: dict):
        """优化完成回调"""
        result = data["result"]
        
        self.log_output.append_text(f"[{datetime.now():%H:%M:%S}] 优化完成!")
        self.log_output.append_text(f"最优参数: {result.best_params}")
        self.log_output.append_text(f"最优得分: {result.best_score:.4f}")
        self.log_output.append_text(f"执行时间: {result.execution_time:.2f}秒")
        
        self.report_output.append_text("\n" + "="*50)
        self.report_output.append_text("参数优化结果")
        self.report_output.append_text("="*50)
        self.report_output.append_text(f"优化方法: {result.optimization_method}")
        self.report_output.append_text(f"总迭代次数: {result.total_iterations}")
        self.report_output.append_text(f"执行时间: {result.execution_time:.2f}秒")
        self.report_output.append_text(f"\n最优参数: {result.best_params}")
        self.report_output.append_text(f"最优得分: {result.best_score:.4f}")
        self.report_output.append_text("\nTop 5 结果:")
        for i, r in enumerate(result.to_dict()["top_results"], 1):
            self.report_output.append_text(
                f"  {i}. 参数: {r['params']}, 得分: {r['score']:.4f}, "
                f"收益率: {r['total_return_pct']:.2f}%, "
                f"回撤: {r['max_drawdown_pct']:.2f}%"
            )
        
        self._reset_optimization_ui()
        
        if result.best_params:
            reply = QMessageBox.question(
                self,
                "应用参数",
                f"是否应用最优参数到当前策略？\n\n{result.best_params}",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._apply_optimized_params(result.best_params)
    
    def _on_optimization_error(self, error_msg: str):
        """优化错误回调"""
        self.log_output.append_text(f"[{datetime.now():%H:%M:%S}] {error_msg}")
        QMessageBox.critical(self, "错误", error_msg)
        self._reset_optimization_ui()
    
    def _reset_optimization_ui(self):
        """重置优化UI状态"""
        self.optimize_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status.setText("● 就绪")
        self.status.setStyleSheet("color: #0ecb81;")
    
    def _apply_optimized_params(self, params: dict):
        """应用优化后的参数"""
        for name, value in params.items():
            if name in self._param_widgets:
                widget = self._param_widgets[name]
                if isinstance(widget, QDoubleSpinBox):
                    widget.setValue(float(value))
                elif isinstance(widget, QSpinBox):
                    widget.setValue(int(value))
        
        self.log_output.append_text(f"[{datetime.now():%H:%M:%S}] 已应用优化参数")
    
    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        self._worker_manager.cleanup()
        
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(1000)
        
        event.accept()
    
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
        self.report_output = OptimizedTextEdit(max_lines=500)
        self.report_output.setReadOnly(True)
        report_layout.addWidget(self.report_output)
        tabs.addTab(report_tab, "回测报告")
        
        trades_tab = QWidget()
        trades_layout = QVBoxLayout(trades_tab)
        self.trades_table = OptimizedTableWidget()
        self.trades_table.setColumnCount(5)
        self.trades_table.setHorizontalHeaderLabels(["开仓时间", "平仓时间", "开仓价", "平仓价", "盈亏"])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        trades_layout.addWidget(self.trades_table)
        tabs.addTab(trades_tab, "交易记录")
        
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_output = OptimizedTextEdit(max_lines=500)
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
        for i, p in enumerate(info.get("parameters", [])):
            row, col = i // 2, (i % 2) * 2
            self.params_layout.addWidget(QLabel(p["display_name"]), row, col)
            
            if p["type"] == "float":
                w = QDoubleSpinBox()
                w.setRange(p.get("min", 0), p.get("max", 100))
                w.setValue(p["default"])
                w.setSingleStep(0.1)
            else:
                w = QSpinBox()
                w.setRange(p.get("min", 0), p.get("max", 1000))
                w.setValue(p["default"])
            
            self._param_widgets[p["name"]] = w
            self.params_layout.addWidget(w, row, col + 1)
    
    def _run(self):
        params = {n: w.value() for n, w in self._param_widgets.items()}
        
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
        }
        
        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("● 运行中")
        self.status.setStyleSheet("color: #f0b90b;")
        
        self._worker = BacktestWorker(config)
        self._worker.progress.connect(lambda m: self.log_output.append_text(f"[{datetime.now():%H:%M:%S}] {m}"))
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
        
        self.report_output.append_text(report.format_text_report())
        
        trades = result.completed_trades
        trade_data = []
        for t in trades:
            trade_data.append([
                str(t.entry_time),
                str(t.exit_time),
                f"{t.entry_price:.2f}",
                f"{t.exit_price:.2f}",
                f"{t.pnl:.2f}"
            ])
        self.trades_table.set_data_async(trade_data)
        
        self.log_output.append_text(f"[{datetime.now():%H:%M:%S}] 完成: {result.total_trades}笔交易, 胜率{result.win_rate:.1f}%")
        
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status.setText("● 完成")
        self.status.setStyleSheet("color: #0ecb81;")
    
    def _on_error(self, msg):
        self.log_output.append_text(f"[{datetime.now():%H:%M:%S}] 错误: {msg}")
        QMessageBox.critical(self, "错误", msg)
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.progress.setVisible(False)
        self.status.setText("● 错误")
        self.status.setStyleSheet("color: #f6465d;")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = TradingUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
