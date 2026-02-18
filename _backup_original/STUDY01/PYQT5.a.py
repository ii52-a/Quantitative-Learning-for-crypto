import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox,  # 核心：垂直、水平、分组框
    QLabel, QLineEdit, QComboBox, QPushButton, QTextEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator


class ComplexLayoutWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("复杂布局")
        self.setGeometry(100, 100, 1000, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.main_layout = QHBoxLayout(central_widget)
        self._init_control_panel()
        self._init_result_pandel()

    def _init_control_panel(self):
        self.control_panel = QGroupBox("策略回测参数配置")
        control_layout = QVBoxLayout(self.control_panel)
        h_label = QHBoxLayout()
        h_label.addWidget(QLabel("k线周期:"))
        h_chose=QComboBox()
        h_chose.addItems(['30min','15min'])

        control_layout.addLayout(h_label)
        control_layout.addWidget(h_chose)
        self.macd_group=QGroupBox("MACD 参数")
        macd_layout = QVBoxLayout(self.macd_group)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("参数选择(Fast,slow,):"))
        self.kline_chose=QComboBox()
        self.kline_chose.addItems(['(12,23,9)','(15,28,9)'])
        h_layout.addWidget(self.kline_chose)
        macd_layout.addLayout(h_layout)

        self.control_panel.setLayout(control_layout)
        control_layout.addWidget(self.macd_group)

        self.run_button = QPushButton("执行回测")
        self.run_button.clicked.connect(self.run)
        control_layout.addWidget(self.run_button)

        control_layout.addStretch(1)
        self.main_layout.addWidget(self.control_panel,1)

    def _init_result_pandel(self):
        self.rusults_panel=QWidget()
        result_layout = QVBoxLayout(self.rusults_panel)
        result_layout.addWidget(QLabel("<UNK>"))
        result_layout.addStretch(1)
        result_layout.addWidget(QLabel("<UNKdw"))
        self.text_output=QTextEdit()
        self.text_output.setReadOnly(True)
        result_layout.addWidget(self.text_output,3)
        self.main_layout.addWidget(self.rusults_panel,3)

    def run(self):
        self.text_output.clear()
        print("success")




# 应用程序入口
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ComplexLayoutWindow()
    window.show()
    sys.exit(app.exec_())