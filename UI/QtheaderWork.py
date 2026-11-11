from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from Strategy.CTA.cta_macd_strategy import StrategyMacd
from app_logger.logger_setup import Logger

logger = Logger(__name__)


class BacktestWorker(QThread):
    """后台回测线程：绝不阻塞 GUI 线程。"""

    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, symbol: str, k_line: str, number: int, usdt: float = 1000,
                 csv_path: str | Path | None = None):
        super().__init__()
        self.symbol = symbol
        self.k_line = k_line
        self.number = int(number)
        self.usdt = float(usdt)
        self.csv_path = csv_path
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self.log.emit(f"开始回测 {self.symbol} {self.k_line} {self.number} 根 ...")
            strategy = StrategyMacd(self.symbol, self.k_line, self.number, self.usdt)
            self.log.emit(f"有效数据 {strategy.core.data_count} 根，开始逐根撮合")
            summary = strategy.main(
                progress=self.progress.emit,
                should_stop=lambda: self._stop,
                output_csv=self.csv_path,
            )
            self.finished_ok.emit(summary)
        except Exception as exc:  # 线程内异常必须转成信号，不能逃逸
            logger.exception(f"回测失败: {exc}")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DataSyncWorker(QThread):
    """后台更新本地数据库（网络请求全部在子线程）。"""

    log = pyqtSignal(str)
    finished_ok = pyqtSignal(bool)
    failed = pyqtSignal(str)

    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = symbol

    def run(self) -> None:
        try:
            from Data.Api import Api

            self.log.emit(f"开始更新本地数据库: {self.symbol}")
            api = Api(ping=False)
            api.local_data_main(self.symbol)
            self.log.emit("本地数据库更新完成")
            self.finished_ok.emit(True)
        except Exception as exc:
            logger.exception(f"数据更新失败: {exc}")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
