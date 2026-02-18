import logging
from pathlib import Path
import logging.handlers
import sys
from app_logger.LoggerType import PositionHistory


def _init(logger_name, console_level=logging.INFO, file_level=logging.DEBUG):
    LOG_DIR = Path("Logs")
    LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    # 独立目录
    module_dir = LOG_DIR / logger_name
    module_dir.mkdir(parents=True, exist_ok=True)

    log_file = module_dir / f"{logger_name}.log"

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColoredFormatter())

    # 使用自定义颜色格式化器
    color_formatter = ColoredFormatter()
    console_handler.setFormatter(color_formatter)


    # 控制台输出



    # 文件输出
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when='midnight',
        interval=1,
        backupCount=7,
        encoding='utf-8',
        delay=False
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(' <%(levelname)s>(%(filename)s:%(lineno)d): %(message)s   %(asctime)s'))

    # 装载 handler
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


class Logger:
    def __init__(self,logger_name):
        self.logger = _init(logger_name, console_level=logging.INFO, file_level=logging.DEBUG)

    def info(self, msg): self.logger.info(msg)

    def debug(self, msg): self.logger.debug(msg)

    def error(self, msg): self.logger.error(msg)

    def warning(self, msg): self.logger.warning(msg)

    def exception(self, msg): self.logger.exception(msg)

    def log_position_history(self,h:PositionHistory):
        status = "WIN" if h.pnl >= 0 else "LOSS"
        msg = (
               f"[TRADE_{status}] {h.symbol} x{h.leverage} | "
               f"PnL: {h.pnl:+.4f}({h.pnl_percent :.2f}%) |  ({h.close_type})\n"
               f"Price: {h.open:.2f}->{h.close:.2f} | "
               f"Time: {h.open_time}->{h.close_time}\n"
               f"==========================================="
               )
        self.logger.debug(msg)


class ColoredFormatter(logging.Formatter):
    GREY = "\033[38;20m"
    GREEN = "\033[32;20m"
    YELLOW = "\033[33;20m"
    RED = "\033[31;20m"
    BOLD_RED = "\033[31;1m"
    RESET = "\033[0m"
    WHITE="\033[38;5;250m"
    VUG_WHITE= "\033[38;2;255;255;255m"

    FORMAT = ' (%(filename)s:%(lineno)d): %(message)s  <%(levelname)s> %(asctime)s'

    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: WHITE,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record):
        """
        每当有一条日志（record）需要打印时，系统都会自动调用这个方法。
        我们在这里给日志穿上“彩色马甲”。
        """

        # 1.
        # 如果以后你用了个没定义过的日志级别（比如 logging.CRITICAL），
        # 也会默认给它一个灰色（self.GREY），防止代码直接报错崩掉。
        color = self.LEVEL_COLORS.get(record.levelno, self.WHITE)

        # 2. 【暂存模板】记录下当前 Formatter 原始的显示格式（即不带颜色的那串文字）。
        # 因为这个类实例是共享的，我们得先记住“原样”，方便一会儿还原。
        original_fmt = self._style._fmt

        # 3. 【借梁换柱】动态合成带颜色的新模板。
        # 比如把 " (file:74): Hello <INFO> " 变成 "\033[32m (file:74): Hello <INFO> \033[0m"。
        # 我们直接修改 self._style._fmt，这是底层真正存格式的地方。
        self._style._fmt = f"{color}{self.FORMAT}{self.RESET}"

        # 4. 【核心执行】调用父类（logging.Formatter）原本的 format 方法。
        # 它已经写好了如何计算时间戳、提取文件名、计算行号等复杂逻辑。
        result = super().format(record)

        # 5
        # 否则，如果其他 Handler（比如写文件的 Handler）也共用了这个属性，
        # 你的日志文件里就会出现一堆乱七八糟的颜色代码（\033...）。
        self._style._fmt = original_fmt

        # 6. 返回已经渲染好的彩色字符串。
        return result