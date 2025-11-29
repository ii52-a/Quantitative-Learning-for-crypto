import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

from urlibs import FileUrlibs

"""API线程"""
class ApiWorker(QThread):
    # 设置信号
    data_ready = pyqtSignal(pd.DataFrame)

    def __init__(self, api, data_number, interval, symbol):
        super().__init__()
        self.api = api
        self.data_number = data_number
        self.interval = interval
        self.symbol = symbol

    def run(self):
        try:
            data = self.api.get_backtest_data(symbol=self.symbol, number=self.data_number, interval=self.interval)
            self.data_ready.emit(data)  # 发出信号
        except Exception as e:
            print(f"API 获取数据失败: {e}")
            self.data_ready.emit(pd.DataFrame())  # 数据空


"""本地加载线程"""
class LoaclWorker(QThread):
    data_ready = pyqtSignal(pd.DataFrame)

    def __init__(self, api, data_number, interval, symbol, file_path):
        super().__init__()
        self.api = api
        self.data_number = data_number
        self.symbol = symbol
        self.interval = interval
        self.path = file_path

    def run(self):
        try:
            self.api.update_local_csv(symbol=self.symbol, number=self.data_number, interval=self.interval, file_path=self.path)
            data_qp=FileUrlibs.get_csv_data(number=self.data_number, file_path=self.path)
            self.data_ready.emit(data_qp)
        except Exception as e:
            print(f"API 本地数据更新失败: {e}")
            self.data_ready.emit(pd.DataFrame())

