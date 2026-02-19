"""
量化交易系统 - 用户友好界面

面向非专业用户的引导式回测界面
"""

import sys
import traceback
from datetime import datetime
from typing import Any

import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QPushButton, QTextEdit,
    QSpinBox, QDoubleSpinBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QMessageBox,
    QStackedWidget, QFrame, QScrollArea, QProgressBar
)

from core.config import BacktestConfig
from Strategy.templates import list_strategies, get_strategy, STRATEGY_REGISTRY
from backtest.engine import BacktestEngine, BacktestResult
from backtest.report import BacktestReport

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestWorker(QThread):
    """回测工作线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, strategy, data, config):
        super().__init__()
        self.strategy = strategy
        self.data = data
        self.config = config
    
    def run(self):
        try:
            self.progress.emit("正在初始化策略...")
            engine = BacktestEngine(self.strategy, self.config)
            
            self.progress.emit("正在执行回测...")
            result = engine.run(self.data)
            
            self.progress.emit("回测完成！")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"回测失败: {str(e)}")
            logger.error(f"回测失败: {e}\n{traceback.format_exc()}")


class DataLoadWorker(QThread):
    """数据加载工作线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, symbol, interval, limit):
        super().__init__()
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
    
    def run(self):
        try:
            self.progress.emit("正在连接数据源...")
            
            from Data.data_service import get_data_service, DataServiceConfig
            import os
            
            use_proxy = os.getenv("USE_PROXY", "false").lower() == "true"
            proxy_host = os.getenv("PROXY_HOST", "127.0.0.1")
            proxy_port = int(os.getenv("PROXY_PORT", "7890"))
            
            config = DataServiceConfig(
                use_proxy=use_proxy,
                proxy_host=proxy_host,
                proxy_port=proxy_port,
                prefer_database=True,
                auto_init=True,
            )
            
            service = get_data_service(config)
            
            self.progress.emit(f"正在获取 {self.symbol} {self.interval} 数据...")
            
            df = service.get_backtest_data(self.symbol, self.interval, self.limit)
            
            if df.empty:
                self.error.emit("数据获取失败，请检查网络连接和API配置")
            else:
                self.progress.emit(f"数据加载完成: {len(df)} 条")
                self.finished.emit(df)
        except Exception as e:
            self.error.emit(f"数据加载失败: {str(e)}")
            logger.error(f"数据加载失败: {e}\n{traceback.format_exc()}")


class GuideStepWidget(QFrame):
    """引导步骤组件"""
    
    def __init__(self, step_number: int, title: str, description: str, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
            }
            QLabel#step_number {
                font-size: 24px;
                font-weight: bold;
                color: #007bff;
            }
            QLabel#title {
                font-size: 16px;
                font-weight: bold;
                color: #212529;
            }
            QLabel#description {
                font-size: 12px;
                color: #6c757d;
            }
        """)
        
        layout = QHBoxLayout(self)
        
        step_label = QLabel(str(step_number))
        step_label.setObjectName("step_number")
        step_label.setFixedWidth(40)
        layout.addWidget(step_label)
        
        text_layout = QVBoxLayout()
        
        title_label = QLabel(title)
        title_label.setObjectName("title")
        text_layout.addWidget(title_label)
        
        desc_label = QLabel(description)
        desc_label.setObjectName("description")
        desc_label.setWordWrap(True)
        text_layout.addWidget(desc_label)
        
        layout.addLayout(text_layout, 1)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(40, 0, 0, 0)
        layout.addWidget(self.content_widget)


class StrategyCardWidget(QFrame):
    """策略卡片组件"""
    
    clicked = pyqtSignal(str)
    
    def __init__(self, strategy_info: dict, parent=None):
        super().__init__(parent)
        self.strategy_name = strategy_info.get("name", "")
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setFixedHeight(120)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #dee2e6;
                border-radius: 8px;
            }
            QFrame:hover {
                border-color: #007bff;
                background-color: #f0f7ff;
            }
            QFrame:selected {
                border-color: #007bff;
                background-color: #e7f1ff;
            }
            QLabel#name {
                font-size: 14px;
                font-weight: bold;
                color: #212529;
            }
            QLabel#desc {
                font-size: 11px;
                color: #6c757d;
            }
            QLabel#type {
                font-size: 10px;
                color: white;
                background-color: #28a745;
                border-radius: 3px;
                padding: 2px 6px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        header = QHBoxLayout()
        
        name_label = QLabel(strategy_info.get("display_name", self.strategy_name))
        name_label.setObjectName("name")
        header.addWidget(name_label)
        
        type_label = QLabel(strategy_info.get("type", "custom"))
        type_label.setObjectName("type")
        header.addWidget(type_label)
        header.addStretch()
        
        layout.addLayout(header)
        
        desc_label = QLabel(strategy_info.get("description", ""))
        desc_label.setObjectName("desc")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        risk = strategy_info.get("risk_level", "medium")
        risk_text = {"low": "低风险", "medium": "中等风险", "high": "高风险"}.get(risk, risk)
        risk_label = QLabel(f"风险等级: {risk_text}")
        risk_label.setStyleSheet("font-size: 10px; color: #6c757d;")
        layout.addWidget(risk_label)
        
        self._selected = False
    
    def mousePressEvent(self, event):
        self.clicked.emit(self.strategy_name)
        self.setSelected(True)
    
    def setSelected(self, selected: bool):
        self._selected = selected
        if selected:
            self.setStyleSheet(self.styleSheet().replace(
                "border: 2px solid #dee2e6;", "border: 2px solid #007bff;"
            ))
        else:
            self.setStyleSheet(self.styleSheet().replace(
                "border: 2px solid #007bff;", "border: 2px solid #dee2e6;"
            ))


class MetricsTableWidget(QTableWidget):
    """绩效指标表格"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["指标", "数值"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QTableWidget {
                gridline-color: #dee2e6;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)
    
    def set_metrics(self, metrics: dict[str, Any]):
        """设置指标数据"""
        display_names = {
            "total_return": "总收益率",
            "annualized_return": "年化收益率",
            "max_drawdown_pct": "最大回撤",
            "sharpe_ratio": "夏普比率",
            "sortino_ratio": "索提诺比率",
            "calmar_ratio": "卡玛比率",
            "win_rate": "胜率",
            "profit_factor": "盈亏比",
            "total_trades": "总交易次数",
            "winning_trades": "盈利次数",
            "losing_trades": "亏损次数",
            "avg_win": "平均盈利",
            "avg_loss": "平均亏损",
            "volatility": "年化波动率",
        }
        
        formatters = {
            "total_return": lambda x: f"{x:.2f}%",
            "annualized_return": lambda x: f"{x:.2f}%",
            "max_drawdown_pct": lambda x: f"{x:.2f}%",
            "sharpe_ratio": lambda x: f"{x:.2f}",
            "sortino_ratio": lambda x: f"{x:.2f}",
            "calmar_ratio": lambda x: f"{x:.2f}",
            "win_rate": lambda x: f"{x:.1f}%",
            "profit_factor": lambda x: f"{x:.2f}",
            "total_trades": lambda x: str(x),
            "winning_trades": lambda x: str(x),
            "losing_trades": lambda x: str(x),
            "avg_win": lambda x: f"{x:.2f} USDT",
            "avg_loss": lambda x: f"{x:.2f} USDT",
            "volatility": lambda x: f"{x:.2f}%",
        }
        
        display_keys = [
            "total_return", "annualized_return", "max_drawdown_pct",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "win_rate", "profit_factor", "total_trades",
            "winning_trades", "losing_trades", "avg_win", "avg_loss",
        ]
        
        self.setRowCount(len(display_keys))
        
        for i, key in enumerate(display_keys):
            if key in metrics:
                name = display_names.get(key, key)
                value = metrics[key]
                formatted = formatters.get(key, str)(value)
                
                self.setItem(i, 0, QTableWidgetItem(name))
                self.setItem(i, 1, QTableWidgetItem(formatted))


class UserFriendlyMainWindow(QMainWindow):
    """用户友好的主窗口"""
    
    def __init__(self):
        super().__init__()
        self._selected_strategy: str = ""
        self._strategy_params: dict[str, Any] = {}
        self._backtest_data: pd.DataFrame | None = None
        self._backtest_result: BacktestResult | None = None
        
        self._init_ui()
    
    def _init_ui(self):
        self.setWindowTitle("量化交易系统 - 回测助手")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        header = self._create_header()
        main_layout.addWidget(header)
        
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack, 1)
        
        self.page_strategy = self._create_strategy_page()
        self.page_config = self._create_config_page()
        self.page_result = self._create_result_page()
        
        self.content_stack.addWidget(self.page_strategy)
        self.content_stack.addWidget(self.page_config)
        self.content_stack.addWidget(self.page_result)
        
        self._load_strategies()
    
    def _create_header(self) -> QWidget:
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #007bff;
                border-radius: 0;
            }
            QLabel {
                color: white;
            }
        """)
        header.setFixedHeight(60)
        
        layout = QHBoxLayout(header)
        
        title = QLabel("📈 量化交易回测助手")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        help_btn = QPushButton("❓ 帮助")
        help_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: 1px solid white;
                border-radius: 4px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: rgba(255,255,255,0.1);
            }
        """)
        help_btn.clicked.connect(self._show_help)
        layout.addWidget(help_btn)
        
        return header
    
    def _create_strategy_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        
        guide = GuideStepWidget(
            1, "选择策略", 
            "请从下方选择一个适合您交易风格的策略模板。不同策略适合不同的市场环境。"
        )
        layout.addWidget(guide)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        strategy_container = QWidget()
        self.strategy_grid = QVBoxLayout(strategy_container)
        self.strategy_grid.setSpacing(10)
        
        scroll.setWidget(strategy_container)
        layout.addWidget(scroll, 1)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        next_btn = QPushButton("下一步 →")
        next_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 30px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        next_btn.clicked.connect(self._go_to_config)
        btn_layout.addWidget(next_btn)
        
        layout.addLayout(btn_layout)
        
        return page
    
    def _create_config_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        
        guide = GuideStepWidget(
            2, "配置参数",
            "设置交易对、数据量和策略参数。鼠标悬停在参数名称上可查看详细说明。"
        )
        layout.addWidget(guide)
        
        config_group = QGroupBox("基本配置")
        config_layout = QVBoxLayout(config_group)
        
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("交易对:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
        h1.addWidget(self.symbol_combo)
        
        h1.addWidget(QLabel("K线周期:"))
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["15min", "30min", "1h", "4h", "1d"])
        self.interval_combo.setCurrentIndex(1)
        h1.addWidget(self.interval_combo)
        
        h1.addWidget(QLabel("数据量:"))
        self.data_limit = QSpinBox()
        self.data_limit.setRange(100, 50000)
        self.data_limit.setValue(1000)
        self.data_limit.setSingleStep(100)
        h1.addWidget(self.data_limit)
        h1.addWidget(QLabel("条"))
        
        config_layout.addLayout(h1)
        
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("初始资金:"))
        self.initial_capital = QDoubleSpinBox()
        self.initial_capital.setRange(100, 1000000)
        self.initial_capital.setValue(10000)
        h2.addWidget(self.initial_capital)
        h2.addWidget(QLabel("USDT"))
        h2.addStretch()
        config_layout.addLayout(h2)
        
        layout.addWidget(config_group)
        
        self.params_group = QGroupBox("策略参数")
        self.params_layout = QVBoxLayout(self.params_group)
        layout.addWidget(self.params_group)
        
        btn_layout = QHBoxLayout()
        
        back_btn = QPushButton("← 上一步")
        back_btn.clicked.connect(lambda: self.content_stack.setCurrentIndex(0))
        btn_layout.addWidget(back_btn)
        
        btn_layout.addStretch()
        
        self.run_btn = QPushButton("🚀 开始回测")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 30px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.run_btn.clicked.connect(self._run_backtest)
        btn_layout.addWidget(self.run_btn)
        
        layout.addLayout(btn_layout)
        
        return page
    
    def _create_result_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        
        guide = GuideStepWidget(
            3, "查看结果",
            "回测完成！下方展示策略的绩效指标和交易记录。"
        )
        layout.addWidget(guide)
        
        self.metrics_table = MetricsTableWidget()
        self.metrics_table.setMaximumHeight(300)
        layout.addWidget(self.metrics_table)
        
        trades_group = QGroupBox("交易记录")
        trades_layout = QVBoxLayout(trades_group)
        
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(6)
        self.trades_table.setHorizontalHeaderLabels(["时间", "方向", "价格", "数量", "盈亏", "原因"])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.trades_table.setEditTriggers(QTableWidget.NoEditTriggers)
        trades_layout.addWidget(self.trades_table)
        
        layout.addWidget(trades_group, 1)
        
        btn_layout = QHBoxLayout()
        
        back_btn = QPushButton("← 重新配置")
        back_btn.clicked.connect(lambda: self.content_stack.setCurrentIndex(1))
        btn_layout.addWidget(back_btn)
        
        btn_layout.addStretch()
        
        export_btn = QPushButton("📄 导出报告")
        export_btn.clicked.connect(self._export_report)
        btn_layout.addWidget(export_btn)
        
        new_btn = QPushButton("🔄 新回测")
        new_btn.clicked.connect(self._new_backtest)
        btn_layout.addWidget(new_btn)
        
        layout.addLayout(btn_layout)
        
        return page
    
    def _load_strategies(self):
        """加载策略列表"""
        strategies = list_strategies()
        
        for i, info in enumerate(strategies):
            card = StrategyCardWidget(info)
            card.clicked.connect(self._on_strategy_selected)
            self.strategy_grid.addWidget(card)
        
        self.strategy_grid.addStretch()
    
    def _on_strategy_selected(self, strategy_name: str):
        """策略选择回调"""
        self._selected_strategy = strategy_name
        
        for i in range(self.strategy_grid.count()):
            widget = self.strategy_grid.itemAt(i).widget()
            if isinstance(widget, StrategyCardWidget):
                widget.setSelected(widget.strategy_name == strategy_name)
    
    def _go_to_config(self):
        """进入配置页面"""
        if not self._selected_strategy:
            QMessageBox.warning(self, "提示", "请先选择一个策略")
            return
        
        self._load_strategy_params()
        self.content_stack.setCurrentIndex(1)
    
    def _load_strategy_params(self):
        """加载策略参数配置"""
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        strategy = get_strategy(self._selected_strategy)
        params_def = strategy.get_parameter_definitions()
        
        self._param_widgets = {}
        
        for param in params_def:
            h = QHBoxLayout()
            
            label = QLabel(f"{param['display_name']}:")
            label.setToolTip(param['description'])
            h.addWidget(label)
            
            if param['type'] == 'float':
                widget = QDoubleSpinBox()
                widget.setRange(param.get('min', 0), param.get('max', 100))
                widget.setValue(param['default'])
                widget.setSingleStep(0.1)
            elif param['type'] == 'int':
                widget = QSpinBox()
                widget.setRange(param.get('min', 0), param.get('max', 1000))
                widget.setValue(param['default'])
            else:
                widget = QSpinBox()
                widget.setValue(param['default'])
            
            self._param_widgets[param['name']] = widget
            h.addWidget(widget)
            
            desc_label = QLabel(f"  ({param['description']})")
            desc_label.setStyleSheet("color: #6c757d; font-size: 10px;")
            h.addWidget(desc_label)
            
            h.addStretch()
            self.params_layout.addLayout(h)
    
    def _run_backtest(self):
        """运行回测"""
        symbol = self.symbol_combo.currentText()
        interval = self.interval_combo.currentText()
        limit = self.data_limit.value()
        
        self._strategy_params = {
            name: widget.value() 
            for name, widget in self._param_widgets.items()
        }
        
        self.run_btn.setEnabled(False)
        self.run_btn.setText("正在加载数据...")
        
        self._data_worker = DataLoadWorker(symbol, interval, limit)
        self._data_worker.progress.connect(self._on_data_progress)
        self._data_worker.finished.connect(self._on_data_loaded)
        self._data_worker.error.connect(self._on_data_error)
        self._data_worker.start()
    
    def _on_data_progress(self, msg):
        self.run_btn.setText(msg)
    
    def _on_data_loaded(self, data):
        self._backtest_data = data
        
        self.run_btn.setText("正在执行回测...")
        
        strategy = get_strategy(self._selected_strategy, self._strategy_params)
        
        config = BacktestConfig(
            symbol=self.symbol_combo.currentText(),
            interval=self.interval_combo.currentText(),
            initial_capital=self.initial_capital.value(),
            data_limit=self.data_limit.value(),
        )
        
        self._backtest_worker = BacktestWorker(strategy, data, config)
        self._backtest_worker.progress.connect(self._on_backtest_progress)
        self._backtest_worker.finished.connect(self._on_backtest_finished)
        self._backtest_worker.error.connect(self._on_backtest_error)
        self._backtest_worker.start()
    
    def _on_data_error(self, msg):
        QMessageBox.critical(self, "错误", msg)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("🚀 开始回测")
    
    def _on_backtest_progress(self, msg):
        self.run_btn.setText(msg)
    
    def _on_backtest_finished(self, result):
        self._backtest_result = result
        
        report = BacktestReport(
            result=result,
            strategy_name=self._selected_strategy,
            symbol=self.symbol_combo.currentText(),
            interval=self.interval_combo.currentText(),
        )
        
        summary = report.get_summary()
        self.metrics_table.set_metrics(summary['performance'])
        
        self._load_trades_table(result.trades)
        
        self.content_stack.setCurrentIndex(2)
        
        self.run_btn.setEnabled(True)
        self.run_btn.setText("🚀 开始回测")
    
    def _on_backtest_error(self, msg):
        QMessageBox.critical(self, "错误", msg)
        self.run_btn.setEnabled(True)
        self.run_btn.setText("🚀 开始回测")
    
    def _load_trades_table(self, trades):
        """加载交易记录表格"""
        self.trades_table.setRowCount(len(trades))
        
        for i, trade in enumerate(trades):
            t = trade if isinstance(trade, dict) else trade.__dict__
            
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(t.get('timestamp', ''))))
            self.trades_table.setItem(i, 1, QTableWidgetItem(t.get('side', '')))
            self.trades_table.setItem(i, 2, QTableWidgetItem(f"{t.get('price', 0):.2f}"))
            self.trades_table.setItem(i, 3, QTableWidgetItem(f"{t.get('quantity', 0):.6f}"))
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{t.get('pnl', 0):.2f}"))
            self.trades_table.setItem(i, 5, QTableWidgetItem(t.get('reason', '')))
    
    def _export_report(self):
        """导出报告"""
        if not self._backtest_result:
            return
        
        report = BacktestReport(
            result=self._backtest_result,
            strategy_name=self._selected_strategy,
            symbol=self.symbol_combo.currentText(),
            interval=self.interval_combo.currentText(),
        )
        
        text_report = report.format_text_report()
        
        filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        QMessageBox.information(self, "导出成功", f"报告已保存到: {filename}")
    
    def _new_backtest(self):
        """新回测"""
        self._selected_strategy = ""
        self._backtest_data = None
        self._backtest_result = None
        
        for i in range(self.strategy_grid.count()):
            widget = self.strategy_grid.itemAt(i).widget()
            if isinstance(widget, StrategyCardWidget):
                widget.setSelected(False)
        
        self.content_stack.setCurrentIndex(0)
    
    def _show_help(self):
        """显示帮助"""
        help_text = """
        <h2>量化交易回测助手 - 使用指南</h2>
        
        <h3>第一步：选择策略</h3>
        <p>系统提供多种策略模板：</p>
        <ul>
            <li><b>MACD趋势策略</b>：适合趋势行情，基于MACD金叉死叉信号</li>
            <li><b>趋势跟踪策略</b>：均线与MACD组合，适合强趋势市场</li>
            <li><b>均值回归策略</b>：RSI与布林带组合，适合震荡行情</li>
            <li><b>布林带策略</b>：基于布林带突破，适合波动市场</li>
        </ul>
        
        <h3>第二步：配置参数</h3>
        <p>设置交易对、K线周期、数据量和初始资金。</p>
        <p>每个策略参数都有详细说明，鼠标悬停可查看。</p>
        
        <h3>第三步：查看结果</h3>
        <p>回测完成后，您可以：</p>
        <ul>
            <li>查看绩效指标（收益率、夏普比率、最大回撤等）</li>
            <li>查看详细的交易记录</li>
            <li>导出回测报告</li>
        </ul>
        
        <h3>常见问题</h3>
        <p><b>Q: 数据加载失败怎么办？</b></p>
        <p>A: 请检查网络连接，或在.env文件中配置代理。</p>
        
        <p><b>Q: 如何选择合适的策略？</b></p>
        <p>A: 趋势行情选择趋势跟踪类策略，震荡行情选择均值回归类策略。</p>
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("帮助")
        msg.setTextFormat(Qt.RichText)
        msg.setText(help_text)
        msg.exec_()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = UserFriendlyMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
