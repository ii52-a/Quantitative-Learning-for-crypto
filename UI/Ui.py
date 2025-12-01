import logging
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QTextEdit, QCheckBox
)
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_fixed

from BackTest.BackTestPlat import BackTest
from UI.QtheaderWork import ApiWorker, LoaclWorker
from data.Api import Api
from type import BackTestSetting
from Config import *
from urlibs import *

logging.basicConfig(level=logging.DEBUG)
class GetbackUi(QMainWindow):
    def __init__(self):
        super().__init__()
        self.api: Api | None = None
        self.init_ui()

        # 将设置选项改为self对象属性
        self.setting_kline:str = '30min'  # 默认值
        self.setting_trading_pair:str = 'ETHUSDT'  # 默认值
        self.setting_kline_num:int = 500  # 默认值
        self.setting_use_local_data:bool = True  # 默认值

    @retry(stop=stop_after_attempt(ApiConfig.MAX_RETRY), wait=wait_fixed(ApiConfig.WAITING_TIME),
           retry=retry_if_exception_type(ApiConfig.RETRY_ERROR_ACCEPT))

    def init_api(self) -> None:
        if not self.api:
            self.text_output.append("正在进行api链接:")
            self.api = Api()
            self.text_output.append("biance API链接成功")


    def init_ui(self) -> None:
        self.setWindowTitle("BackTest回测")
        self.setGeometry(100, 100, 1000, 600)

        # 设置控制
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.main_layout = QHBoxLayout(central_widget)

        self._init_left_layout()
        self._init_right_layout()

    def _init_left_layout(self):
        self.results_pandel = QWidget()

        result_layout = QVBoxLayout(self.results_pandel)
        h_tb = QHBoxLayout()
        h_tb.addWidget(QLabel("回测图表"), 0)

        # TODO 图表 学习和使用
        self.QLabel_line = QLabel("None")
        self.QLabel_line.setStyleSheet("border:1px solid black;")
        h_tb.addWidget(self.QLabel_line, 1)
        result_layout.addLayout(h_tb, 3)

        # 组件
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        h_layout = QHBoxLayout(self.results_pandel)
        h_layout.addWidget(QLabel("回测进程"))
        h_layout.addWidget(self.text_output)

        self.text_resultput = QTextEdit()
        self.text_resultput.setReadOnly(True)
        h_layout1 = QHBoxLayout(self.results_pandel)
        h_layout1.addWidget(QLabel("回测结果"))
        h_layout1.addWidget(self.text_resultput)

        # 合并布局
        result_layout.addLayout(h_layout, 2)
        result_layout.addLayout(h_layout1, 1)

        self.main_layout.addWidget(self.results_pandel, 3)

    def _init_right_layout(self):
        # 右侧大group>策略配置参数>选择
        self.right_layout = QGroupBox("数据回测")
        self.control_layout = QVBoxLayout(self.right_layout)

        # 策略选择复选框
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(['CTA-MACD做多'])  # TODO:增加交互
        self.strategy_combo.setCurrentIndex(0)

        h_strategy = QHBoxLayout()
        h_strategy.addWidget(QLabel("选择策略"))
        h_strategy.addWidget(self.strategy_combo)
        self.control_layout.addLayout(h_strategy)

        # ---策略配置----
        self.setting_groupbox = QGroupBox("策略配置参数")
        in_setting_groupbox = QVBoxLayout(self.setting_groupbox)

        # -k线选择
        self.h_combobox_kline = QComboBox()
        self.h_combobox_kline.addItems(['15min', '30min', '1h'])
        self.h_combobox_kline.setCurrentIndex(1)

        self.h_combobox_kline_num = QComboBox()
        self.h_combobox_kline_num.addItems(['300', '500', '1000','2000', '5000', '10000'])
        self.h_combobox_kline_num.setCurrentIndex(1)

        h_label = QHBoxLayout()
        h_label.addWidget(QLabel("k线周期:"))
        h_label.addWidget(self.h_combobox_kline)
        h_label.addWidget(QLabel("数量:"))
        h_label.addWidget(self.h_combobox_kline_num)

        # -交易对选择
        self.h_combobox_trading_pair = QComboBox()
        self.h_combobox_trading_pair.addItems(['BTCUSDT', 'ETHUSDT','SOLUSDT'])
        self.h_combobox_trading_pair.setCurrentIndex(1)

        h_label_2 = QHBoxLayout()
        h_label_2.addWidget(QLabel("交易对:"))
        h_label_2.addWidget(self.h_combobox_trading_pair)

        # 使用本地数据快速回测
        self.q_checkbox = QCheckBox()
        self.q_checkbox.setChecked(True)

        h_label_3 = QHBoxLayout()
        h_label_3.addWidget(QLabel("本地csv回测:"))
        h_label_3.addWidget(self.q_checkbox)

        # 开始回测按钮
        self.button = QPushButton("开始回测")
        self.button.clicked.connect(self.start)

        # 策略配置--布局添加
        in_setting_groupbox.addLayout(h_label)
        in_setting_groupbox.addLayout(h_label_2)
        in_setting_groupbox.addLayout(h_label_3)
        in_setting_groupbox.addStretch(1)

        # 右侧布局
        self.control_layout.addWidget(self.setting_groupbox)
        self.control_layout.addWidget(self.button)
        self.control_layout.addStretch(1)

        self.main_layout.addWidget(self.right_layout, 1)

    def start(self):
        self.text_output.clear()
        self.text_resultput.clear()

        # 使用self存储的实例变量
        self.setting_kline = self.h_combobox_kline.currentText()
        self.setting_trading_pair = self.h_combobox_trading_pair.currentText()
        self.setting_kline_num = int(self.h_combobox_kline_num.currentText())
        self.setting_use_local_data = self.q_checkbox.isChecked()

        self.text_output.append("--- 参数配置成功 ---")
        self.text_output.append(f"交易对: {self.setting_trading_pair}")
        self.text_output.append(f"K线周期: {self.setting_kline}\t回测k线数量: {self.setting_kline_num}条")

        # 禁用按钮
        self.button.setDisabled(True)
        """本地数据加载"""
        if self.setting_use_local_data:
            self.load_local_data()
        """不建议:直接api加载,次数多了就给吃api封禁"""
        if not self.setting_use_local_data:
            self.api_worker_start()


    #TODO
    def load_local_data(self):
        self.text_output.append("--- 正在检测本地csv ---")
        local_path = ApiConfig.LOCAL_DATA_CSV_DIR
        path = f"{local_path}/{self.setting_trading_pair}_{self.setting_kline}.csv"
        """
        :study:pathlib,Path(),parent(),mkdier(),touch()
        """
        FileUrlibs.check_local_path(path)
        # c()
        data_df: pd.DataFrame = pd.DataFrame()
        try:
            # c()
            data_df = FileUrlibs.get_csv_data(number=self.setting_kline_num, file_path=path)
        except pd.errors.EmptyDataError:  #没有本地数据
            # c()
            self.text_output.append("--- 本地csv获取失败 ---")
            self.text_output.append(f"-- 准备更新本地csv --")
            self.init_api()
            self.local_worker_start(path=path)
            # c()
            self.run_backtest(data_df)
            # c()
            return

        else:
            self.text_output.append("--- 本地csv加载成功 ---")
            self.run_backtest(data_df)
            return
    # 本地数据进行切片获取


    # csv本地线程
    def local_worker_start(self,path):
        local_worker = LoaclWorker(api=self.api, data_number=ApiConfig.LOCAL_MAX_CSV_NUMBER,
                                   symbol=self.setting_trading_pair, file_path=path,
                                   interval=TradeMapper.K_LINE_TYPE[self.setting_kline])
        local_worker.data_ready.connect(self.on_data_ready)
        local_worker.start()

    #  api线程
    def api_worker_start(self):
        # ApiWorker线程
        self.init_api()
        self.api_worker = ApiWorker(api=self.api, data_number=self.setting_kline_num,
                                    interval=TradeMapper.K_LINE_TYPE[self.setting_kline], symbol=self.setting_trading_pair)
        # 关联线程信号
        self.api_worker.data_ready.connect(self.on_data_ready)
        self.api_worker.start()

    """
        1 apiwork线程->读取data发出信号
        2 apiwork信号激活->调用connect的方法
        3 on_data_ready检测数据情况
        4 数据不为空->运行回测
        5 按钮回归状态
    """
    # 判断数据情况
    def on_data_ready(self, data: pd.DataFrame) -> None:
        if not data.empty:
            self.text_output.append("API数据加载完成")
            self.run_backtest(data)
        else:
            self.text_output.append("API数据加载失败")
            self.button.setDisabled(False)

    # 数据完成后进行回测
    def run_backtest(self, data: pd.DataFrame | None) -> None:
        try:
            # 统筹配置
            back_test_setting: BackTestSetting = BackTestSetting(
                k_line=TradeMapper.K_LINE_TYPE[self.h_combobox_kline.currentText()],
                trading_pair=self.h_combobox_trading_pair.currentText(),
                strategy=self.strategy_combo.currentText(),
            )
            backtest = BackTest(api=self.api, back_test_setting=back_test_setting)

            backtest.strategy_backtest_loop(interval=back_test_setting.k_line, data=data)

            self.text_resultput.append("回测完成")
        except Exception as e:
            self.text_resultput.setText(f"回测错误: {e}")
        finally:
            self.button.setDisabled(False)






#TODO:学习使用logging替代插print来寻找错误
n=0
def c():
    global n
    n+=1
    print(f'第{n}个*')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GetbackUi()
    window.show()
    sys.exit(app.exec_())
