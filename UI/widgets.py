"""
优化版UI组件模块

解决UI卡顿问题：
- 批量更新日志，减少重绘
- 节流信号处理
- 内存管理优化
- 异步数据加载
"""

from collections import deque
import time

from PyQt5.QtWidgets import (
    QWidget, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QProgressBar, QToolTip, QTextEdit, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QMutex, QMutexLocker
from PyQt5.QtGui import QFont, QTextCursor


class OptimizedTextEdit(QTextEdit):
    """优化的文本编辑器 - 批量更新减少重绘"""
    
    def __init__(self, parent=None, max_lines: int = 1000):
        super().__init__(parent)
        self._buffer = deque(maxlen=max_lines)
        self._pending_update = False
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._flush_buffer)
        self._mutex = QMutex()
        self._max_lines = max_lines
    
    def append_text(self, text: str):
        """添加文本（延迟更新）"""
        with QMutexLocker(self._mutex):
            self._buffer.append(text)
        
        if not self._pending_update:
            self._pending_update = True
            self._update_timer.start(100)
    
    def _flush_buffer(self):
        """刷新缓冲区到显示"""
        with QMutexLocker(self._mutex):
            if not self._buffer:
                self._pending_update = False
                return
            
            texts = list(self._buffer)
            self._buffer.clear()
        
        self.setUpdatesEnabled(False)
        try:
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.End)
            
            for text in texts:
                cursor.insertText(text + "\n")
            
            if self.document().blockCount() > self._max_lines:
                cursor.movePosition(QTextCursor.Start)
                cursor.select(QTextCursor.BlockUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()
            
            self.setTextCursor(cursor)
            self.ensureCursorVisible()
        finally:
            self.setUpdatesEnabled(True)
        
        self._pending_update = False
    
    def clear_text(self):
        """清空文本"""
        with QMutexLocker(self._mutex):
            self._buffer.clear()
        self.clear()


class OptimizedTableWidget(QTableWidget):
    """优化的表格组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data_queue = []
        self._is_loading = False
    
    def set_data_async(self, data: list, batch_size: int = 50):
        """异步设置表格数据"""
        self._data_queue = data
        self.setRowCount(0)
        
        if not data:
            return
        
        self.setRowCount(len(data))
        self._is_loading = True
        self._load_batch(0, batch_size)
    
    def _load_batch(self, start: int, batch_size: int):
        """批量加载行"""
        if start >= len(self._data_queue):
            self._is_loading = False
            return
        
        end = min(start + batch_size, len(self._data_queue))
        
        self.setUpdatesEnabled(False)
        try:
            for i in range(start, end):
                row_data = self._data_queue[i]
                for j, value in enumerate(row_data):
                    item = QTableWidgetItem(str(value))
                    self.setItem(i, j, item)
        finally:
            self.setUpdatesEnabled(True)
        
        if end < len(self._data_queue):
            QTimer.singleShot(10, lambda: self._load_batch(end, batch_size))
        else:
            self._is_loading = False


class ThrottledSignal:
    """信号节流器"""
    
    def __init__(self, interval_ms: int = 100):
        self._interval = interval_ms
        self._last_emit = 0
        self._pending_data = None
        self._timer = QTimer()
        self._timer.setSingleShot(True)
    
    def should_emit(self) -> bool:
        """检查是否应该发送信号"""
        current = int(time.time() * 1000)
        if current - self._last_emit >= self._interval:
            self._last_emit = current
            return True
        return False


class TooltipLabel(QLabel):
    """带悬停提示的标签"""
    
    def __init__(self, text: str, tooltip: str = "", parent=None):
        super().__init__(text, parent)
        if tooltip:
            self.setToolTip(tooltip)
            self.setToolTipDuration(10000)


class TooltipComboBox(QComboBox):
    """带悬停提示的下拉框"""
    
    def __init__(self, tooltip: str = "", parent=None):
        super().__init__(parent)
        if tooltip:
            self.setToolTip(tooltip)
            self.setToolTipDuration(10000)


class TooltipSpinBox(QSpinBox):
    """带悬停提示的整数输入框"""
    
    def __init__(self, tooltip: str = "", parent=None):
        super().__init__(parent)
        if tooltip:
            self.setToolTip(tooltip)
            self.setToolTipDuration(10000)


class TooltipDoubleSpinBox(QDoubleSpinBox):
    """带悬停提示的小数输入框"""
    
    def __init__(self, tooltip: str = "", parent=None):
        super().__init__(parent)
        if tooltip:
            self.setToolTip(tooltip)
            self.setToolTipDuration(10000)


class TooltipButton(QPushButton):
    """带悬停提示的按钮"""
    
    def __init__(self, text: str, tooltip: str = "", parent=None):
        super().__init__(text, parent)
        if tooltip:
            self.setToolTip(tooltip)
            self.setToolTipDuration(10000)


TOOLTIP_TEXTS = {
    "symbol": "选择要回测的交易对\n例如：BTCUSDT 表示比特币对USDT",
    "interval": "K线时间周期\n1min=1分钟，1h=1小时，1d=1天",
    "data_limit": "回测使用的数据量\n更多数据=更准确的回测结果",
    "initial_capital": "初始资金金额\n用于计算收益率和仓位大小",
    "leverage": "杠杆倍数 (1-125)\n⚠️ 高杠杆意味着高风险\n建议：新手使用1-5倍",
    "stop_loss": "止损百分比\n当亏损达到此比例时自动平仓\n0表示不设置止损",
    "take_profit": "止盈百分比\n当盈利达到此比例时自动平仓\n0表示不设置止盈",
    "strategy": "选择交易策略\n不同策略有不同的交易逻辑",
    "vote_threshold": "多指标投票阈值\n信号强度达到此值时开仓",
    "opt_method": "参数优化方法\n网格搜索：遍历所有组合\n随机搜索：随机采样\n贝叶斯优化：智能搜索",
    "opt_iterations": "优化迭代次数\n更多迭代=更可能找到最优参数\n但耗时更长",
    "opt_metric": "优化目标\n夏普比率：风险调整后收益\n总收益率：绝对收益\n综合得分：多指标加权",
    "opt_breadth": "优化广度\n控制参数搜索区间范围\n<1 更保守，>1 更激进",
    "opt_depth": "优化深度\n控制参数搜索精细度\n数值越大，步长越小",
}


class OptimizerWorker(QThread):
    """异步参数优化工作线程"""
    
    progress = pyqtSignal(str, int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._is_cancelled = False
    
    def cancel(self):
        """取消优化"""
        self._is_cancelled = True
    
    def run(self):
        try:
            from Strategy.parameter_optimizer import (
                ParameterOptimizer,
                ParameterRange,
                get_parameter_ranges_from_strategy,
                tune_parameter_ranges,
            )
            from Strategy.templates import get_strategy
            from Data.data_service import get_data_service, DataServiceConfig, RegionRestrictedError, DataSourceError
            from core.config import BacktestConfig
            
            strategy_name = self.config["strategy"]
            strategy_params = self.config["strategy_params"]
            strategy = get_strategy(strategy_name, strategy_params)
            
            self.progress.emit("正在初始化数据服务...", 0, 100)
            
            service = get_data_service(DataServiceConfig(prefer_database=True, auto_init=False))
            
            self.progress.emit("正在加载数据...", 2, 100)
            
            try:
                data_limit = max(100, int(self.config["data_limit"]))
                # 优化阶段无需额外 padding，避免高周期聚合触发超大数据拉取导致UI看似卡死
                data = service.get_klines(
                    self.config["symbol"],
                    self.config["interval"],
                    data_limit,
                )
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
                self.error.emit("无法获取数据，请检查网络连接")
                return
            
            self.progress.emit("数据加载完成，初始化优化器...", 5, 100)
            
            config = BacktestConfig(
                symbol=self.config["symbol"],
                interval=self.config["interval"],
                initial_capital=self.config["initial_capital"],
                leverage=self.config.get("leverage", 5),
                stop_loss_pct=self.config.get("stop_loss_pct", 0),
                take_profit_pct=self.config.get("take_profit_pct", 0),
            )
            
            optimizer = ParameterOptimizer(
                strategy_class=type(strategy),
                data=data,
                base_config=config,
                optimization_metric=self.config.get("optimization_metric", "sharpe_ratio"),
            )
            
            param_ranges = get_parameter_ranges_from_strategy(strategy)
            
            if not param_ranges:
                param_ranges = [
                    ParameterRange("fast_period", 5, 30, 5),
                    ParameterRange("slow_period", 20, 60, 10),
                ]
            
            if self.config.get("optimize_risk_params"):
                param_ranges.extend([
                    ParameterRange("stop_loss_pct", 0, 20, 5),
                    ParameterRange("take_profit_pct", 0, 50, 10),
                    ParameterRange("leverage", 1, 20, 1),
                ])

            param_ranges = tune_parameter_ranges(
                param_ranges,
                breadth=self.config.get("opt_breadth", 1.0),
                depth=self.config.get("opt_depth", 1),
            )
            
            last_progress_time = [0]
            
            def progress_callback(current, total):
                if self._is_cancelled:
                    raise InterruptedError("优化已取消")
                
                import time
                current_time = time.time()
                if current_time - last_progress_time[0] < 0.2:
                    return
                last_progress_time[0] = current_time
                
                pct = int(5 + (current / total) * 90)
                self.progress.emit(f"优化中 {current}/{total}", pct, 100)
            
            opt_method = self.config.get("opt_method", "random_search")
            iterations = self.config.get("iterations", 50)
            
            self.progress.emit("开始优化...", 10, 100)
            
            if opt_method == "grid_search":
                result = optimizer.grid_search(
                    param_ranges=param_ranges,
                    max_iterations=iterations,
                    progress_callback=progress_callback,
                )
            elif opt_method == "random_search":
                result = optimizer.random_search(
                    param_ranges=param_ranges,
                    n_iterations=iterations,
                    progress_callback=progress_callback,
                )
            else:
                result = optimizer.bayesian_optimization(
                    param_ranges=param_ranges,
                    n_iterations=iterations,
                    progress_callback=progress_callback,
                )
            
            if self._is_cancelled:
                self.error.emit("优化已取消")
                return
            
            self.progress.emit("完成!", 100, 100)
            self.finished.emit({
                "result": result,
                "strategy_name": strategy_name,
            })
            
        except InterruptedError:
            self.error.emit("优化已取消")
        except Exception as e:
            import traceback
            self.error.emit(f"优化失败: {str(e)}\n{traceback.format_exc()}")


class ValidationWorker(QThread):
    """策略验证工作线程"""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
    
    def run(self):
        try:
            from Strategy.templates import get_strategy
            from backtest.engine import BacktestEngine
            from core.config import BacktestConfig
            from Data.data_service import get_data_service, DataServiceConfig, RegionRestrictedError, DataSourceError
            
            self.progress.emit("正在初始化数据服务...")
            
            service = get_data_service(DataServiceConfig(prefer_database=True, auto_init=False))
            
            symbol = self.config["symbol"]
            interval = self.config["interval"]
            strategy_name = self.config["strategy"]
            strategy_params = self.config["strategy_params"]
            
            self.progress.emit("正在加载验证数据...")
            
            try:
                data = service.get_klines(symbol, interval, self.config["data_limit"])
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
                self.error.emit("无法获取验证数据，请检查网络连接")
                return
            
            self.progress.emit("数据加载完成，执行验证回测...")
            
            config = BacktestConfig(
                symbol=symbol,
                interval=interval,
                initial_capital=self.config["initial_capital"],
                leverage=self.config.get("leverage", 5),
                stop_loss_pct=self.config.get("stop_loss_pct", 0),
                take_profit_pct=self.config.get("take_profit_pct", 0),
            )
            
            strategy = get_strategy(strategy_name, strategy_params)
            engine = BacktestEngine(strategy, config)
            result = engine.run(data)
            
            validation_result = {
                "result": result,
                "params": strategy_params,
                "data_count": len(data),
                "symbol": symbol,
                "interval": interval,
            }
            
            self.progress.emit("验证完成!")
            self.finished.emit(validation_result)
            
        except Exception as e:
            import traceback
            self.error.emit(f"验证失败: {str(e)}")


class WorkerManager:
    """工作线程管理器 - 防止内存泄漏"""
    
    def __init__(self):
        self._workers = []
    
    def add_worker(self, worker: QThread):
        """添加工作线程"""
        worker.finished.connect(lambda: self._remove_worker(worker))
        self._workers.append(worker)
    
    def _remove_worker(self, worker: QThread):
        """移除已完成的工作线程"""
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()
    
    def cancel_all(self):
        """取消所有工作线程"""
        for worker in self._workers:
            if hasattr(worker, 'cancel'):
                worker.cancel()
            worker.quit()
            worker.wait(1000)
        self._workers.clear()
    
    def cleanup(self):
        """清理所有工作线程"""
        self.cancel_all()
