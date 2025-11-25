import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox,  # 核心：垂直、水平、分组框
    QLabel, QLineEdit, QComboBox, QPushButton, QTextEdit, QGridLayout, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIntValidator


class GetbackUi(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initui()
    def initui(self):
        self.setWindowTitle("BackTest回测")
        self.setGeometry(100, 100, 1000, 600)

        #设置控制
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.main_layout = QHBoxLayout(central_widget)

        self._init_left_layout()
        self._init_right_layout()

    def _init_left_layout(self):
        self.results_pandel=QWidget()


        result_layout = QVBoxLayout(self.results_pandel)
        h_tb = QHBoxLayout()
        h_tb.addWidget(QLabel("回测图表"),0)

        #TODO 图表d 学习和使用
        self.QLabel_line=QLabel("<UNK>")
        self.QLabel_line.setStyleSheet("border:1px solid black;")
        h_tb.addWidget(self.QLabel_line,1)
        result_layout.addLayout(h_tb,3)

        # 组件

        self.text_output=QTextEdit()
        self.text_output.setReadOnly(True)
        h_layout = QHBoxLayout(self.results_pandel)
        h_layout.addWidget(QLabel("回测进程"))
        h_layout.addWidget(self.text_output)


        self.text_resultput=QTextEdit()
        self.text_resultput.setReadOnly(True)
        h_layout1 = QHBoxLayout(self.results_pandel)
        h_layout1.addWidget(QLabel("回测结果"))
        h_layout1.addWidget(self.text_resultput)

        #合并布局
        result_layout.addLayout(h_layout,2)
        result_layout.addLayout(h_layout1,1)

        self.main_layout.addWidget(self.results_pandel,3)

    def _init_right_layout(self):
        #右侧大group>策略配置参数>选择
        #group>start_button || others
        self.right_layout = QGroupBox("数据回测")
        self.control_layout=QVBoxLayout(self.right_layout)

        h_strategy = QHBoxLayout()
        h_strategy.addWidget(QLabel("选择策略"))
        self.strategy_combo=QComboBox()
        self.strategy_combo.addItem(['CTA-MACD30min做多'])  #TODO:使用Config增加交互

        #---策略配置----
        self.setting_groupbox=QGroupBox("策略配置参数")
        in_setting_groupbox=QVBoxLayout(self.setting_groupbox)

        # k线选择
        self.h_combobox_kline=QComboBox()
        self.h_combobox_kline.addItems(['15min','30min','1h'])
        self.h_combobox_kline.setCurrentIndex(1)


        h_label = QHBoxLayout()
        h_label.addWidget(QLabel("k线周期:"))
        h_label.addWidget(self.h_combobox_kline)

        #交易对选择
        self.h_combobox_trading_pair=QComboBox()
        self.h_combobox_trading_pair.addItems(['BTCUSDT','ETHUSDT'])
        self.h_combobox_trading_pair.setCurrentIndex(1)

        h_label_2=QHBoxLayout()
        h_label_2.addWidget(QLabel("交易对:"))
        h_label_2.addWidget(self.h_combobox_trading_pair)

        #开始回测按钮
        self.button=QPushButton("开始回测")
        self.button.clicked.connect(self.start)

        #  策略配置--布局添加
        #k线
        in_setting_groupbox.addLayout(h_label)
        #交易对
        in_setting_groupbox.addLayout(h_label_2)
        #按钮

        in_setting_groupbox.addStretch(1)

        #右侧布局
        self.control_layout.addWidget(self.setting_groupbox)
        self.control_layout.addWidget(self.button)
        self.control_layout.addStretch(1)
        self.main_layout.addWidget(self.right_layout,1)

    def start(self):
        self.text_resultput.clear()

        setting_kline=self.h_combobox_kline.currentText()
        setting_trading_pair=self.h_combobox_trading_pair.currentText()
        self.text_output.append("--- 参数读取成功 ---")
        self.text_output.append(f"交易对: {setting_trading_pair}以太坊期货")
        self.text_output.append(f"K线周期: {setting_kline}")
        self.i = 0
        self.timer=QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(400)


    def update(self):
        if self.i<10:
            self.i+=1
            self.text_resultput.setText(f"进度:[{'-' * (self.i) +' '*(10-self.i)}]{self.i * 10}%")

        else:
            self.timer.stop()
            self.text_resultput.append("回测完成:总回报率:8880%")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GetbackUi()
    window.show()
    sys.exit(app.exec_())


