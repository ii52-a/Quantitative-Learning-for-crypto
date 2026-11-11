import sys
import traceback
from functools import wraps
from pathlib import Path

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QProgressBar, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from Config import ApiConfig
from UI.QtheaderWork import BacktestWorker, DataSyncWorker
from app_logger.logger_setup import Logger

logger = Logger(__name__)

USDT = 1000.0


def guard(fn):
    """槽函数兜底：异常绝不能逃逸到 Qt 事件循环（PyQt5 会直接 abort 进程）。"""

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            logger.exception("界面操作失败")
            self._set_running(False)
            self._append_log("界面操作异常，已在下方显示详情")
            self.text_resultput.setPlainText(traceback.format_exc(limit=6))
            return None

    return wrapper


class GetbackUi(QMainWindow):
    def __init__(self, csv_dir: str | Path | None = None):
        super().__init__()
        self._workers: set = set()
        self._backtest_worker: BacktestWorker | None = None
        self._sync_worker: DataSyncWorker | None = None
        self._running = False
        self.csv_dir = Path(csv_dir) if csv_dir is not None else \
            Path(ApiConfig.PROJECT_ROOT) / 'Strategy' / 'CTA'
        self._init_ui()

    # ---------------- UI ----------------
    def _init_ui(self) -> None:
        self.setWindowTitle("BackTest回测")
        self.setGeometry(100, 100, 1000, 620)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)

        self._init_left_layout()
        self._init_right_layout()

    def _init_left_layout(self) -> None:
        self.results_pandel = QWidget()
        result_layout = QVBoxLayout(self.results_pandel)

        h_tb = QHBoxLayout()
        h_tb.addWidget(QLabel("回测图表"))
        self.QLabel_line = QLabel("None")
        self.QLabel_line.setStyleSheet("border:1px solid black;")
        h_tb.addWidget(self.QLabel_line, 1)
        result_layout.addLayout(h_tb, 3)

        # 每个布局只挂一次，避免 QLayout 重复父级告警
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("回测进程"))
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        h_layout.addWidget(self.text_output)
        result_layout.addLayout(h_layout, 2)

        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(QLabel("回测结果"))
        self.text_resultput = QTextEdit()
        self.text_resultput.setReadOnly(True)
        h_layout1.addWidget(self.text_resultput)
        result_layout.addLayout(h_layout1, 1)

        progress_layout = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_label = QLabel("空闲")
        progress_layout.addWidget(self.progress, 1)
        progress_layout.addWidget(self.progress_label, 0)
        result_layout.addLayout(progress_layout)

        self.main_layout.addWidget(self.results_pandel, 3)

    def _init_right_layout(self) -> None:
        self.right_layout = QGroupBox("数据回测")
        self.control_layout = QVBoxLayout(self.right_layout)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(['CTA-MACD做多'])
        self.strategy_combo.setCurrentIndex(0)
        h_strategy = QHBoxLayout()
        h_strategy.addWidget(QLabel("选择策略"))
        h_strategy.addWidget(self.strategy_combo)
        self.control_layout.addLayout(h_strategy)

        self.setting_groupbox = QGroupBox("策略配置参数")
        in_setting_groupbox = QVBoxLayout(self.setting_groupbox)

        self.h_combobox_kline = QComboBox()
        self.h_combobox_kline.addItems(['15min', '30min', '1h'])
        self.h_combobox_kline.setCurrentIndex(1)

        self.h_combobox_kline_num = QComboBox()
        self.h_combobox_kline_num.addItems(['300', '500', '1000', '2000', '5000', '10000'])
        self.h_combobox_kline_num.setCurrentIndex(3)

        h_label = QHBoxLayout()
        h_label.addWidget(QLabel("k线周期:"))
        h_label.addWidget(self.h_combobox_kline)
        h_label.addWidget(QLabel("数量:"))
        h_label.addWidget(self.h_combobox_kline_num)

        self.h_combobox_trading_pair = QComboBox()
        self.h_combobox_trading_pair.addItems(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        self.h_combobox_trading_pair.setCurrentIndex(1)

        h_label_2 = QHBoxLayout()
        h_label_2.addWidget(QLabel("交易对:"))
        h_label_2.addWidget(self.h_combobox_trading_pair)

        # 勾选：只读本地数据库；取消勾选：先联网更新本地库（在子线程）
        self.q_checkbox = QCheckBox()
        self.q_checkbox.setChecked(True)

        h_label_3 = QHBoxLayout()
        h_label_3.addWidget(QLabel("仅用本地数据库:"))
        h_label_3.addWidget(self.q_checkbox)

        self.button = QPushButton("开始回测")
        self.button.clicked.connect(self.start)

        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)

        in_setting_groupbox.addLayout(h_label)
        in_setting_groupbox.addLayout(h_label_2)
        in_setting_groupbox.addLayout(h_label_3)
        in_setting_groupbox.addStretch(1)

        self.control_layout.addWidget(self.setting_groupbox)
        self.control_layout.addWidget(self.button)
        self.control_layout.addWidget(self.cancel_button)
        self.control_layout.addStretch(1)

        self.main_layout.addWidget(self.right_layout, 1)

    # ---------------- 状态 ----------------
    def _set_running(self, running: bool) -> None:
        self._running = running
        self.button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.q_checkbox.setEnabled(not running)
        self.h_combobox_kline.setEnabled(not running)
        self.h_combobox_kline_num.setEnabled(not running)
        self.h_combobox_trading_pair.setEnabled(not running)
        if not running:
            self.progress_label.setText("空闲")

    def _append_log(self, message: str) -> None:
        self.text_output.append(message)

    @staticmethod
    def _format_summary(summary: dict) -> str:
        return (
            f"交易对      : {summary['symbol']}  周期: {summary['k_line']}  K线: {summary['bars']}\n"
            f"初始资金    : {summary['initial']:.2f} USDT\n"
            f"最终资金    : {summary['final']:.2f} USDT\n"
            f"净盈亏      : {summary['net_pnl']:+.2f} USDT ({summary['return_pct']:+.2f}%)\n"
            f"毛利/手续费 : {summary['gross_pnl']:+.2f} / {summary['fees']:.2f} USDT\n"
            f"交易次数    : {summary['trades']}  胜/负/平: {summary['win']}/{summary['lose']}/{summary['breakeven']}\n"
            f"胜率        : {summary['win_rate']:.2f}%\n"
            f"平均每笔    : {summary['avg_trade']:+.4f} USDT\n"
        )

    # ---------------- 操作 ----------------
    @guard
    def start(self, checked: bool = False) -> None:
        if self._running:
            return
        symbol = self.h_combobox_trading_pair.currentText()
        k_line = self.h_combobox_kline.currentText()
        number = int(self.h_combobox_kline_num.currentText())
        local_only = self.q_checkbox.isChecked()

        self.text_output.clear()
        self.text_resultput.clear()
        self._append_log("--- 参数配置成功 ---")
        self._append_log(f"交易对: {symbol}\tK线周期: {k_line}\t回测k线数量: {number}条")
        self._set_running(True)
        self.progress.setValue(0)

        if local_only:
            self._start_backtest(symbol, k_line, number)
        else:
            self._start_sync(symbol)

    @guard
    def cancel(self, checked: bool = False) -> None:
        if self._backtest_worker is not None:
            self._backtest_worker.cancel()
            self._append_log("已请求取消回测，等待当前 K 线处理结束 ...")
        elif self._sync_worker is not None:
            self._append_log("数据更新无法中断，请等待本次更新结束")
        self.cancel_button.setEnabled(False)

    def _start_sync(self, symbol: str) -> None:
        worker = DataSyncWorker(symbol)
        worker.log.connect(self._append_log)
        worker.finished_ok.connect(self._on_sync_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_finished)
        self._workers.add(worker)
        self._sync_worker = worker
        self.progress_label.setText("更新数据中")
        worker.start()

    def _start_backtest(self, symbol: str, k_line: str, number: int) -> None:
        csv_path = self.csv_dir / f'backtest_result_{symbol}_{k_line}.csv'
        worker = BacktestWorker(symbol, k_line, number, usdt=USDT, csv_path=csv_path)
        worker.progress.connect(self._on_progress)
        worker.log.connect(self._append_log)
        worker.finished_ok.connect(self._on_backtest_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_finished)
        self._workers.add(worker)
        self._backtest_worker = worker
        self.progress_label.setText("回测中")
        worker.start()

    # ---------------- 回调 ----------------
    @pyqtSlot(int)
    def _on_progress(self, percent: int) -> None:
        self.progress.setValue(max(0, min(100, int(percent))))
        self.progress_label.setText(f"回测中 {percent}%")

    @pyqtSlot(dict)
    def _on_backtest_done(self, summary: dict) -> None:
        self._set_running(False)
        self.progress.setValue(100)
        self.progress_label.setText("完成")
        self.text_resultput.setPlainText(self._format_summary(summary))
        self._append_log(f"回测完成，逐笔结果: Strategy/CTA/backtest_result_"
                         f"{summary['symbol']}_{summary['k_line']}.csv")

    @pyqtSlot(bool)
    def _on_sync_done(self, ok: bool) -> None:
        symbol = self.h_combobox_trading_pair.currentText()
        k_line = self.h_combobox_kline.currentText()
        number = int(self.h_combobox_kline_num.currentText())
        self._append_log("数据已更新，开始回测")
        self._set_running(True)
        self._start_backtest(symbol, k_line, number)

    @pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self.progress_label.setText("失败")
        self._append_log(f"失败: {message}")
        self.text_resultput.setPlainText(f"执行失败:\n{message}")

    @pyqtSlot()
    def _on_worker_finished(self) -> None:
        worker = self.sender()
        if worker is not None:
            self._workers.discard(worker)
            if worker is self._backtest_worker:
                self._backtest_worker = None
            if worker is self._sync_worker:
                self._sync_worker = None

    def closeEvent(self, event) -> None:
        for worker in list(self._workers):
            if hasattr(worker, 'cancel'):
                worker.cancel()
            worker.wait(2000)
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GetbackUi()
    window.show()
    sys.exit(app.exec_())
