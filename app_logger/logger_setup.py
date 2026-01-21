import logging
from pathlib import Path
import logging.handlers

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

    formatter = logging.Formatter(
        ' (%(filename)s:%(lineno)d): %(message)s  <%(levelname)s> %(asctime)s'
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

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
    file_handler.setFormatter(formatter)

    # 装载 handler
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


class Logger:
    def __init__(self,logger_name, console_level=logging.INFO, file_level=logging.DEBUG):
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
               f"PnL: {h.pnl:+.4f} |  ({h.close_type})\n"
               f"Price: {h.open:.2f}->{h.close:.2f} | "
               f"Time: {h.open_time}->{h.close_time}\n"
               f"==========================================="
               )
        self.logger.debug(msg)

