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

        self.setWindowTitle("复杂布局：回测配置与结果展示")
        # 初始设置较大尺寸，以容纳左右两边的内容
        self.setGeometry(100, 100, 1000, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 🌟 知识点 1：主布局 (QHBoxLayout) 🌟
        # 主布局采用水平布局，将窗口分成左右两大部分
        self.main_layout = QHBoxLayout(central_widget)

        # 调用方法，分别初始化左右两侧的面板
        self._init_control_panel()
        self._init_results_panel()

    def _init_control_panel(self):
        # 🌟 知识点 2：左侧控制面板使用 QGroupBox 🌟
        # 使用 QGroupBox 来给配置区域添加一个带标题的边框
        self.control_panel = QGroupBox("策略参数配置")

        # 控制面板的内部布局采用垂直布局
        control_layout = QVBoxLayout(self.control_panel)

        # --- 1. K线周期输入行 (内部嵌套 QHBoxLayout) ---

        # 使用 QHBoxLayout 确保 标签 和 输入框 始终并排
        h_layout_timeframe = QHBoxLayout()
        h_layout_timeframe.addWidget(QLabel("K线周期:"))
        self.input_timeframe = QLineEdit("30m")
        h_layout_timeframe.addWidget(self.input_timeframe)

        # 将输入行添加到控制面板的垂直布局中
        control_layout.addLayout(h_layout_timeframe)

        # --- 2. 交易对选择 (QComboBox) ---

        h_layout_symbol = QHBoxLayout()
        h_layout_symbol.addWidget(QLabel("交易对:"))
        self.combo_symbol = QComboBox()
        self.combo_symbol.addItems(["BTCUSDT", "ETHUSDT"])
        h_layout_symbol.addWidget(self.combo_symbol)
        control_layout.addLayout(h_layout_symbol)

        # --- 3. 策略参数分组 (嵌套 QGroupBox) ---

        # 🌟 知识点 2：再次使用 QGroupBox 来对 MACD 参数进行分组
        self.macd_group = QGroupBox("MACD 参数")
        macd_layout = QVBoxLayout(self.macd_group)

        # MACD Fast 周期输入行
        h_layout_fast = QHBoxLayout()
        h_layout_fast.addWidget(QLabel("快线 (Fast):"))
        self.input_fast = QLineEdit("12")
        self.input_fast.setValidator(QIntValidator())  # 仅允许输入整数
        h_layout_fast.addWidget(self.input_fast)
        macd_layout.addLayout(h_layout_fast)

        self.control_panel.setLayout(control_layout)  # 确保设置了布局

        # 将 MACD 策略参数组添加到主控制面板布局中
        control_layout.addWidget(self.macd_group)

        # --- 4. 执行按钮 ---
        self.run_button = QPushButton("🚀 执行回测")
        self.run_button.clicked.connect(self.run_backtest_simulation)
        control_layout.addWidget(self.run_button)

        # 🌟 知识点 3：添加伸展器 🌟
        # 确保所有控件紧贴顶部，下方留白
        control_layout.addStretch(1)

        # 将整个控制面板（QGroupBox）添加到主水平布局中
        # 🌟 知识点 4：分配空间比例 1 🌟
        self.main_layout.addWidget(self.control_panel, 1)  # 左侧占据 1 份空间

    def _init_results_panel(self):
        # 右侧面板是一个 QWidget，用于承载结果
        self.results_panel = QWidget()

        # 结果面板内部使用垂直布局
        results_layout = QVBoxLayout(self.results_panel)

        # --- 1. 图表区域占位 ---
        results_layout.addWidget(QLabel("【图表区域占位】 - 稍后集成 Matplotlib"))

        # 🌟 知识点 3：添加伸展器 🌟
        # 这里的伸展器会保证上方的标签占据尽量小的空间，把大部分空间留给日志
        results_layout.addStretch(1)

        # --- 2. 日志/摘要区域 ---
        results_layout.addWidget(QLabel("回测日志与摘要:"))
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        # 日志区域占据大部分剩余空间
        results_layout.addWidget(self.text_output, 3)

        # 将整个结果面板添加到主水平布局中
        # 🌟 知识点 4：分配空间比例 3 🌟
        # 右侧占据 3 份空间，这样右侧宽度是左侧的 3 倍 (1:3 比例)
        self.main_layout.addWidget(self.results_panel, 3)

    # 槽函数：模拟回测执行
    def run_backtest_simulation(self):
        self.text_output.clear()

        # 1. 从控件读取所有参数
        timeframe = self.input_timeframe.text()
        symbol = self.combo_symbol.currentText()
        fast_period = self.input_fast.text()

        self.text_output.append("--- 参数读取成功 ---")
        self.text_output.append(f"交易对: {symbol}")
        self.text_output.append(f"K线周期: {timeframe}")
        self.text_output.append(f"MACD 快线: {fast_period}")

        # 2. 模拟回测结果输出
        self.text_output.append("\n... 正在执行核心计算 ...")
        self.text_output.append("回测完成：总回报率 +50.00%")


# 应用程序入口
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ComplexLayoutWindow()
    window.show()
    sys.exit(app.exec_())