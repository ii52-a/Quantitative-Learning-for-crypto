import logging
from pathlib import Path
import logging.handlers




def setup_logger(logger_name, console_level=logging.INFO, file_level=logging.DEBUG):
    LOG_DIR = Path("Logs")
    LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler
    if logger.handlers:
        logger.handlers.clear()

    # 独立目录
    module_dir = LOG_DIR / logger_name
    module_dir.mkdir(parents=True, exist_ok=True)

    log_file = module_dir / f"{logger_name}.log"

    formatter = logging.Formatter(
        '%(asctime)s (%(filename)s:%(lineno)d): %(message)s  <%(levelname)s> '
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
