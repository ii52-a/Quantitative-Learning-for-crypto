"""
量化交易统一UI界面

整合功能：
- K线时间检测
- 数据更新机制
- 回测系统
- 实盘开仓
- 状态监控
"""

import asyncio
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QMetaObject, Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QPushButton, QTextEdit,
    QCheckBox, QSpinBox, QDoubleSpinBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QMessageBox, QProgressBar
)
from dotenv import load_dotenv

from Config import ApiConfig
from Strategy.kline_validator import KlineTimeValidator, KlineTimeReporter

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quantitative_trading.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class DataUpdateWorker(QThread):
    """数据更新工作线程"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, symbol: str, interval: str, limit: int):
        super().__init__()
        self.symbol = symbol
        self.interval = interval
        self.limit = limit

    def run(self):
        try:
            self.progress_signal.emit(f"正在获取 {self.symbol} {self.interval} 数据...")

            from Data.data_service import get_data_service, DataServiceConfig

            use_proxy = os.getenv("USE_PROXY", "false").lower() == "true"
            proxy_host = os.getenv("PROXY_HOST", "127.0.0.1")
            proxy_port = int(os.getenv("PROXY_PORT", "7890"))

            config = DataServiceConfig(
                use_proxy=use_proxy,
                proxy_host=proxy_host,
                proxy_port=proxy_port,
                prefer_database=False,
            )

            service = get_data_service(config)

            def progress_cb(current, total):
                self.progress_signal.emit(f"获取进度: {current}/{total} 条")

            self.progress_signal.emit(f"正在从API获取数据 (总量: {self.limit} 条)...")

            count = service.update_local_data(self.symbol, self.interval, self.limit, progress_cb)

            if count > 0:
                self.finished_signal.emit(True, f"数据更新成功: {count} 条已保存到数据库")
            else:
                self.finished_signal.emit(False, "数据获取失败")

        except Exception as e:
            error_msg = f"更新失败: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.finished_signal.emit(False, error_msg)


class BacktestWorker(QThread):
    """回测数据加载线程"""
    data_ready = pyqtSignal(pd.DataFrame)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self):
        try:
            self.progress_signal.emit("开始加载回测数据...")

            symbol = self.config["symbol"]
            interval = self.config["interval"]
            data_num = self.config["data_num"]

            from Data.data_service import get_data_service, DataServiceConfig, RegionRestrictedError, DataSourceError

            use_proxy = os.getenv("USE_PROXY", "false").lower() == "true"
            proxy_host = os.getenv("PROXY_HOST", "127.0.0.1")
            proxy_port = int(os.getenv("PROXY_PORT", "7890"))

            config = DataServiceConfig(
                use_proxy=use_proxy,
                proxy_host=proxy_host,
                proxy_port=proxy_port,
                prefer_database=True,
                auto_init=True,
                init_days=ApiConfig.LOCAL_MAX_HISTORY_ALLOW,
            )

            service = get_data_service(config)

            self.progress_signal.emit(f"检查数据库状态...")
            service.check_and_auto_init(symbol, interval, data_num + ApiConfig.GET_COUNT)

            self.progress_signal.emit(f"正在获取数据: {symbol} {interval} (请求量: {data_num})")

            df = service.get_backtest_data(symbol, interval, data_num)

            if df.empty:
                self.error_signal.emit("数据加载失败，请检查:\n1. 网络连接\n2. API配置\n3. 代理设置(如需要)")
            else:
                self.progress_signal.emit(f"数据加载成功: {len(df)} 条")
                self.data_ready.emit(df)

        except RegionRestrictedError as e:
            error_msg = f"地区限制错误:\n{str(e)}\n\n解决方案:\n1. 配置代理: 在.env中设置 USE_PROXY=true\n2. 或使用VPN\n3. 或使用本地数据库"
            logger.error(error_msg)
            self.error_signal.emit(error_msg)
        except DataSourceError as e:
            error_msg = f"数据源错误: {str(e)}"
            logger.error(error_msg)
            self.error_signal.emit(error_msg)
        except Exception as e:
            error_msg = f"回测数据加载异常: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.error_signal.emit(error_msg)


class LiveTradingThread(QThread):
    """实盘交易线程"""
    status_signal = pyqtSignal(str)
    trade_signal = pyqtSignal(dict)
    indicator_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def run(self):
        self._running = True
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_trading())
        except Exception as e:
            error_msg = f"交易异常: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.error_signal.emit(error_msg)
        finally:
            try:
                if self._loop and not self._loop.is_closed():
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    self._loop.close()
            except Exception as e:
                logger.warning(f"清理事件循环时出错: {e}")
            self.finished_signal.emit()

    async def _run_trading(self):
        from Strategy.live_trading import run_live_trading
        self.status_signal.emit("实盘交易启动中...")
        self.status_signal.emit("正在连接币安API...")
        
        try:
            await run_live_trading(
                api_key=self.config.get("api_key", ""),
                api_secret=self.config.get("api_secret", ""),
                symbol=self.config.get("symbol", "BTCUSDT"),
                interval=self.config.get("interval", "30m"),
                initial_balance=self.config.get("initial_balance", 1000),
                risk_usdt=self.config.get("risk_usdt", 100),
                hist_filter=self.config.get("hist_filter", 0.5),
                testnet=self.config.get("testnet", True),
                dry_run=self.config.get("dry_run", True),
            )
        except asyncio.CancelledError:
            self.status_signal.emit("实盘交易已取消")
            raise

    def stop(self):
        self._running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)


class KlineValidationThread(QThread):
    """K线时间检测线程"""
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, symbol: str, interval: str):
        super().__init__()
        self.symbol = symbol
        self.interval = interval

    def run(self):
        try:
            from Data.data_service import get_data_service

            service = get_data_service()
            df = service.get_klines_from_database(self.symbol, self.interval, 100)

            if df.empty:
                self.error_signal.emit("数据库中无数据，请先更新数据")
                return

            result = KlineTimeValidator.validate_dataframe(df, self.interval)
            result["symbol"] = self.symbol
            result["interval"] = self.interval
            result["report"] = KlineTimeReporter.generate_report(df, self.interval, self.symbol)
            self.result_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(f"K线时间检测失败: {str(e)}")


class QuantitativeTradingUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.backtest_worker: BacktestWorker | None = None
        self.live_thread: LiveTradingThread | None = None
        self.validation_thread: KlineValidationThread | None = None
        self.data_update_worker: DataUpdateWorker | None = None

        self._live_indicators: dict[str, Any] = {}

        self._init_ui()
        self._init_status_timer()

        logger.info("量化交易UI初始化完成")

    def _init_ui(self):
        self.setWindowTitle("量化交易系统 - 统一控制台")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter()
        main_layout.addWidget(splitter)

        left_widget = self._create_left_panel()
        right_widget = self._create_right_panel()

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([900, 500])

    def _create_left_panel(self) -> QWidget:
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        tab_widget = QTabWidget()

        tab_widget.addTab(self._create_backtest_tab(), "回测系统")
        tab_widget.addTab(self._create_live_tab(), "实盘交易")
        tab_widget.addTab(self._create_data_tab(), "数据管理")

        left_layout.addWidget(tab_widget)
        return left_widget

    def _create_right_panel(self) -> QWidget:
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        status_group = QGroupBox("实时状态监控")
        status_layout = QVBoxLayout(status_group)

        self.status_table = QTableWidget()
        self.status_table.setColumnCount(2)
        self.status_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.status_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.status_table.setRowCount(12)

        status_items = [
            "交易对", "当前价格", "K线周期", "下次收盘",
            "MACD", "Signal", "HIST", "持仓状态",
            "账户余额", "收益率", "开仓次数", "胜率"
        ]
        for i, item in enumerate(status_items):
            self.status_table.setItem(i, 0, QTableWidgetItem(item))
            self.status_table.setItem(i, 1, QTableWidgetItem("-"))

        status_layout.addWidget(self.status_table)

        kline_group = QGroupBox("K线时间检测")
        kline_layout = QVBoxLayout(kline_group)

        self.kline_status_label = QLabel("等待检测...")
        self.kline_status_label.setStyleSheet("font-size: 14px; padding: 10px;")
        kline_layout.addWidget(self.kline_status_label)

        self.kline_validation_output = QTextEdit()
        self.kline_validation_output.setReadOnly(True)
        self.kline_validation_output.setMaximumHeight(150)
        kline_layout.addWidget(self.kline_validation_output)

        right_layout.addWidget(status_group)
        right_layout.addWidget(kline_group)

        return right_widget

    def _create_backtest_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        config_group = QGroupBox("回测配置")
        config_layout = QVBoxLayout(config_group)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("交易对:"))
        self.backtest_symbol = QComboBox()
        self.backtest_symbol.addItems(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        h1.addWidget(self.backtest_symbol)
        h1.addWidget(QLabel("K线周期:"))
        self.backtest_interval = QComboBox()
        self.backtest_interval.addItems(["15min", "30min", "1h"])
        self.backtest_interval.setCurrentIndex(1)
        h1.addWidget(self.backtest_interval)
        config_layout.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("数据量:"))
        self.backtest_limit = QSpinBox()
        self.backtest_limit.setRange(100, 50000)
        self.backtest_limit.setValue(500)
        self.backtest_limit.setSingleStep(100)
        h2.addWidget(self.backtest_limit)
        h2.addWidget(QLabel("条 (最大50000)"))
        h2.addStretch()
        config_layout.addLayout(h2)

        h3 = QHBoxLayout()
        self.start_backtest_btn = QPushButton("开始回测")
        self.start_backtest_btn.clicked.connect(self.start_backtest)
        h3.addWidget(self.start_backtest_btn)
        config_layout.addLayout(h3)

        layout.addWidget(config_group)

        output_group = QGroupBox("回测输出")
        output_layout = QVBoxLayout(output_group)
        self.backtest_output = QTextEdit()
        self.backtest_output.setReadOnly(True)
        output_layout.addWidget(self.backtest_output)

        result_group = QGroupBox("回测结果")
        result_layout = QVBoxLayout(result_group)
        self.backtest_result = QTextEdit()
        self.backtest_result.setReadOnly(True)
        result_layout.addWidget(self.backtest_result)

        layout.addWidget(output_group)
        layout.addWidget(result_group)

        return tab

    def _create_live_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        config_group = QGroupBox("实盘配置")
        config_layout = QVBoxLayout(config_group)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("交易对:"))
        self.live_symbol = QComboBox()
        self.live_symbol.addItems(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        h1.addWidget(self.live_symbol)
        h1.addWidget(QLabel("K线周期:"))
        self.live_interval = QComboBox()
        self.live_interval.addItems(["15min", "30min", "1h"])
        self.live_interval.setCurrentIndex(1)
        h1.addWidget(self.live_interval)
        config_layout.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("初始资金:"))
        self.live_balance = QDoubleSpinBox()
        self.live_balance.setRange(100, 100000)
        self.live_balance.setValue(1000)
        h2.addWidget(self.live_balance)
        h2.addWidget(QLabel("风险资金:"))
        self.live_risk = QDoubleSpinBox()
        self.live_risk.setRange(10, 10000)
        self.live_risk.setValue(100)
        h2.addWidget(self.live_risk)
        h2.addWidget(QLabel("HIST过滤:"))
        self.live_hist_filter = QDoubleSpinBox()
        self.live_hist_filter.setRange(0.1, 5.0)
        self.live_hist_filter.setValue(0.5)
        self.live_hist_filter.setSingleStep(0.1)
        h2.addWidget(self.live_hist_filter)
        config_layout.addLayout(h2)

        h3 = QHBoxLayout()
        self.live_testnet = QCheckBox("使用测试网")
        self.live_testnet.setChecked(True)
        self.live_dry_run = QCheckBox("模拟模式(不实际下单)")
        self.live_dry_run.setChecked(True)
        h3.addWidget(self.live_testnet)
        h3.addWidget(self.live_dry_run)
        h3.addStretch()
        config_layout.addLayout(h3)

        btn_layout = QHBoxLayout()
        self.start_live_btn = QPushButton("启动实盘")
        self.start_live_btn.clicked.connect(self.start_live_trading)
        self.stop_live_btn = QPushButton("停止实盘")
        self.stop_live_btn.clicked.connect(self.stop_live_trading)
        self.stop_live_btn.setEnabled(False)
        btn_layout.addWidget(self.start_live_btn)
        btn_layout.addWidget(self.stop_live_btn)
        config_layout.addLayout(btn_layout)

        layout.addWidget(config_group)

        output_group = QGroupBox("实盘日志")
        output_layout = QVBoxLayout(output_group)
        self.live_output = QTextEdit()
        self.live_output.setReadOnly(True)
        output_layout.addWidget(self.live_output)

        layout.addWidget(output_group)

        return tab

    def _create_data_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        config_group = QGroupBox("数据管理")
        config_layout = QVBoxLayout(config_group)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("交易对:"))
        self.data_symbol = QComboBox()
        self.data_symbol.addItems(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        h1.addWidget(self.data_symbol)
        h1.addWidget(QLabel("K线周期:"))
        self.data_interval = QComboBox()
        self.data_interval.addItems(["1min", "5min", "15min", "30min", "1h", "4h", "1d"])
        self.data_interval.setCurrentIndex(3)
        h1.addWidget(self.data_interval)
        config_layout.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("数据量:"))
        self.data_limit = QSpinBox()
        self.data_limit.setRange(100, 50000)
        self.data_limit.setValue(5000)
        self.data_limit.setSingleStep(500)
        h2.addWidget(self.data_limit)
        h2.addWidget(QLabel("条 (最大50000)"))
        config_layout.addLayout(h2)

        h3 = QHBoxLayout()
        h3.addWidget(QLabel("初始化天数:"))
        self.init_days = QSpinBox()
        self.init_days.setRange(100, 2000)
        self.init_days.setValue(ApiConfig.LOCAL_MAX_HISTORY_ALLOW)
        h3.addWidget(self.init_days)
        h3.addWidget(QLabel("天"))
        h3.addStretch()
        config_layout.addLayout(h3)

        btn_layout = QHBoxLayout()
        self.update_data_btn = QPushButton("更新本地数据")
        self.update_data_btn.clicked.connect(self.update_local_data)
        self.init_data_btn = QPushButton("初始化数据库")
        self.init_data_btn.clicked.connect(self.initialize_database)
        self.validate_kline_btn = QPushButton("检测K线时间")
        self.validate_kline_btn.clicked.connect(self.validate_kline_time)
        btn_layout.addWidget(self.update_data_btn)
        btn_layout.addWidget(self.init_data_btn)
        btn_layout.addWidget(self.validate_kline_btn)
        config_layout.addLayout(btn_layout)

        self.data_progress = QProgressBar()
        self.data_progress.setVisible(False)
        config_layout.addWidget(self.data_progress)

        layout.addWidget(config_group)

        output_group = QGroupBox("数据输出")
        output_layout = QVBoxLayout(output_group)
        self.data_output = QTextEdit()
        self.data_output.setReadOnly(True)
        output_layout.addWidget(self.data_output)

        layout.addWidget(output_group)

        return tab

    def _init_status_timer(self):
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_display)
        self.status_timer.start(10000)

    def update_status_display(self):
        try:
            interval = self.live_interval.currentText()
            symbol = self.live_symbol.currentText()

            remaining = KlineTimeValidator.get_remaining_seconds(interval)
            next_close = KlineTimeValidator.get_next_close_time(interval)

            self.status_table.item(0, 1).setText(symbol)
            self.status_table.item(2, 1).setText(interval)
            self.status_table.item(3, 1).setText(
                f"{next_close.strftime('%H:%M:%S')} ({remaining // 60}分{remaining % 60}秒后)"
            )

            if self._live_indicators:
                self.status_table.item(1, 1).setText(f"{self._live_indicators.get('price', 0):.2f}")
                self.status_table.item(4, 1).setText(f"{self._live_indicators.get('macd', 0):.2f}")
                self.status_table.item(5, 1).setText(f"{self._live_indicators.get('signal', 0):.2f}")
                self.status_table.item(6, 1).setText(f"{self._live_indicators.get('hist', 0):.2f}")

            self.kline_status_label.setText(
                f"[{symbol}] {interval} | 下次收盘: {next_close.strftime('%H:%M:%S')} UTC | "
                f"剩余: {remaining // 60}分{remaining % 60}秒"
            )
        except Exception as e:
            logger.warning(f"状态更新异常: {e}")
            self.kline_status_label.setText(f"状态更新异常: {e}")

    def start_backtest(self):
        try:
            if self.backtest_worker and self.backtest_worker.isRunning():
                self.backtest_output.append("已有回测任务在运行中...")
                return

            self.backtest_output.clear()
            self.backtest_result.clear()

            symbol = self.backtest_symbol.currentText()
            interval = self.backtest_interval.currentText()
            data_num = self.backtest_limit.value()

            self.backtest_output.append("--- 参数配置成功 ---")
            self.backtest_output.append(f"交易对: {symbol}")
            self.backtest_output.append(f"K线周期: {interval}")
            self.backtest_output.append(f"数据量: {data_num}")
            self.backtest_output.append("数据源: 数据库优先，API备用")

            self.start_backtest_btn.setDisabled(True)

            config = {
                "symbol": symbol,
                "interval": interval,
                "data_num": data_num,
            }

            self.backtest_worker = BacktestWorker(config)
            self.backtest_worker.data_ready.connect(self.on_backtest_data_ready)
            self.backtest_worker.error_signal.connect(self.on_backtest_error)
            self.backtest_worker.progress_signal.connect(self.on_backtest_progress)
            self.backtest_worker.start()

            logger.info(f"回测启动: {config}")

        except Exception as e:
            error_msg = f"回测启动失败: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.backtest_output.append(error_msg)
            self.start_backtest_btn.setDisabled(False)

    def on_backtest_data_ready(self, data: pd.DataFrame):
        try:
            self.backtest_output.append(f"数据加载完成: {len(data)} 条")
            self.run_backtest(data)
        except Exception as e:
            self.on_backtest_error(f"数据处理失败: {str(e)}")
        finally:
            self.start_backtest_btn.setDisabled(False)

    def on_backtest_error(self, error: str):
        self.backtest_output.append(f"错误: {error}")
        self.start_backtest_btn.setDisabled(False)
        logger.error(f"回测错误: {error}")

    def on_backtest_progress(self, progress: str):
        self.backtest_output.append(progress)

    def run_backtest(self, data: pd.DataFrame):
        try:
            self.backtest_result.append("--- 回测结果 ---")
            self.backtest_result.append(f"数据范围: {len(data)} 条")

            if not data.empty:
                start_time = data.index[0]
                end_time = data.index[-1]
                self.backtest_result.append(f"时间范围: {start_time} 至 {end_time}")

                close_col = 'close' if 'close' in data.columns else data.columns[3]
                prices = data[close_col].astype(float)

                self.backtest_result.append(f"价格范围: {prices.min():.2f} - {prices.max():.2f}")
                self.backtest_result.append(f"价格变动: {((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100):.2f}%")

            self.backtest_result.append("--- 回测完成 ---")

        except Exception as e:
            error_msg = f"回测执行错误: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.backtest_result.append(error_msg)

    def start_live_trading(self):
        try:
            if self.live_thread and self.live_thread.isRunning():
                self.live_output.append("实盘交易已在运行中...")
                return

            self.live_output.clear()
            self.live_output.append("正在启动实盘交易...")

            api_key = os.getenv("BINANCE_API_KEY", "")
            api_secret = os.getenv("BINANCE_API_SECRET", "")

            if not api_key or not api_secret:
                self.live_output.append("错误: 未配置API密钥，请检查.env文件")
                QMessageBox.warning(self, "配置错误", "未配置BINANCE_API_KEY或BINANCE_API_SECRET")
                return

            config = {
                "api_key": api_key,
                "api_secret": api_secret,
                "symbol": self.live_symbol.currentText(),
                "interval": self.live_interval.currentText(),
                "initial_balance": self.live_balance.value(),
                "risk_usdt": self.live_risk.value(),
                "hist_filter": self.live_hist_filter.value(),
                "testnet": self.live_testnet.isChecked(),
                "dry_run": self.live_dry_run.isChecked(),
            }

            self.live_thread = LiveTradingThread(config)
            self.live_thread.status_signal.connect(self.on_live_status)
            self.live_thread.error_signal.connect(self.on_live_error)
            self.live_thread.finished_signal.connect(self.on_live_finished)
            self.live_thread.indicator_signal.connect(self.on_live_indicator)
            self.live_thread.start()

            self.start_live_btn.setEnabled(False)
            self.stop_live_btn.setEnabled(True)

            self.live_output.append("实盘线程已启动，等待连接...")

            logger.info(f"实盘交易启动: symbol={config['symbol']}, interval={config['interval']}")

        except Exception as e:
            error_msg = f"实盘启动失败: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.live_output.append(error_msg)

    def stop_live_trading(self):
        if self.live_thread and self.live_thread.isRunning():
            self.live_output.append("正在停止实盘交易...")
            self.live_thread.stop()

    def on_live_status(self, status: str):
        self.live_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] {status}")

    def on_live_error(self, error: str):
        self.live_output.append(f"[错误] {error}")

    def on_live_indicator(self, indicators: dict):
        self._live_indicators = indicators
        self.update_status_display()

    def on_live_finished(self):
        self.live_output.append("实盘交易已停止")
        self.start_live_btn.setEnabled(True)
        self.stop_live_btn.setEnabled(False)

    def update_local_data(self):
        try:
            if self.data_update_worker and self.data_update_worker.isRunning():
                self.data_output.append("已有更新任务在运行中...")
                return

            self.data_output.clear()
            self.data_output.append("正在更新本地数据...")

            symbol = self.data_symbol.currentText()
            interval = self.data_interval.currentText()
            limit = self.data_limit.value()

            self.data_output.append(f"交易对: {symbol}")
            self.data_output.append(f"周期: {interval}")
            self.data_output.append(f"数量: {limit}")

            self.update_data_btn.setDisabled(True)
            self.init_data_btn.setDisabled(True)
            self.data_progress.setVisible(True)
            self.data_progress.setRange(0, 0)

            self.data_update_worker = DataUpdateWorker(symbol, interval, limit)
            self.data_update_worker.progress_signal.connect(self.on_data_progress)
            self.data_update_worker.finished_signal.connect(self.on_data_finished)
            self.data_update_worker.start()

        except Exception as e:
            error_msg = f"更新失败: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.data_output.append(error_msg)
            self.update_data_btn.setDisabled(False)
            self.init_data_btn.setDisabled(False)
            self.data_progress.setVisible(False)

    def initialize_database(self):
        try:
            if self.data_update_worker and self.data_update_worker.isRunning():
                self.data_output.append("已有更新任务在运行中...")
                return

            self.data_output.clear()

            symbol = self.data_symbol.currentText()
            interval = self.data_interval.currentText()
            days = self.init_days.value()

            minutes_per_kline = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "1h": 60, "4h": 240, "1d": 1440}
            total_klines = (days * 24 * 60) // minutes_per_kline.get(interval, 30)

            self.data_output.append(f"初始化数据库: {symbol} {interval}")
            self.data_output.append(f"天数: {days} 天")
            self.data_output.append(f"预计K线数: {total_klines} 条")

            self.update_data_btn.setDisabled(True)
            self.init_data_btn.setDisabled(True)
            self.data_progress.setVisible(True)
            self.data_progress.setRange(0, 0)

            self.data_update_worker = DataUpdateWorker(symbol, interval, total_klines)
            self.data_update_worker.progress_signal.connect(self.on_data_progress)
            self.data_update_worker.finished_signal.connect(self.on_data_finished)
            self.data_update_worker.start()

        except Exception as e:
            error_msg = f"初始化失败: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.data_output.append(error_msg)
            self.update_data_btn.setDisabled(False)
            self.init_data_btn.setDisabled(False)
            self.data_progress.setVisible(False)

    def on_data_progress(self, progress: str):
        self.data_output.append(progress)

    def on_data_finished(self, success: bool, message: str):
        self.data_output.append(message)
        self.update_data_btn.setDisabled(False)
        self.init_data_btn.setDisabled(False)
        self.data_progress.setVisible(False)

        if success:
            self.data_output.append("数据已保存到SQLite数据库")

    def validate_kline_time(self):
        try:
            if self.validation_thread and self.validation_thread.isRunning():
                self.data_output.append("已有检测任务在运行中...")
                return

            self.data_output.clear()
            self.data_output.append("正在检测K线时间...")

            symbol = self.data_symbol.currentText()
            interval = self.data_interval.currentText()

            self.validation_thread = KlineValidationThread(symbol, interval)
            self.validation_thread.result_signal.connect(self.on_validation_result)
            self.validation_thread.error_signal.connect(self.on_validation_error)
            self.validation_thread.start()

        except Exception as e:
            error_msg = f"检测失败: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.data_output.append(error_msg)

    def on_validation_result(self, result: dict):
        self.data_output.append(result.get("report", ""))

        self.kline_validation_output.clear()
        self.kline_validation_output.append(result.get("report", ""))

    def on_validation_error(self, error: str):
        self.data_output.append(f"错误: {error}")

    def closeEvent(self, event):
        if self.live_thread and self.live_thread.isRunning():
            self.live_thread.stop()
            self.live_thread.wait(3000)

        if self.backtest_worker and self.backtest_worker.isRunning():
            self.backtest_worker.wait(3000)

        if self.data_update_worker and self.data_update_worker.isRunning():
            self.data_update_worker.wait(3000)

        event.accept()


def exception_hook(exctype, value, tb):
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    logger.critical(f"未捕获的异常:\n{error_msg}")
    QMessageBox.critical(None, "程序异常", f"程序发生异常:\n{str(value)}")


if __name__ == "__main__":
    sys.excepthook = exception_hook

    try:
        app = QApplication(sys.argv)
        window = QuantitativeTradingUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"程序启动失败: {e}\n{traceback.format_exc()}")
        QMessageBox.critical(None, "启动失败", f"程序启动失败:\n{str(e)}")
