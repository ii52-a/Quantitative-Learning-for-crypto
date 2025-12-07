import logging
import logging.handlers
from pathlib import Path
from datetime import datetime



# ---常量---


def setup_logging(logger_name,console_level=logging.INFO,file_level=logging.DEBUG):
    LOG_DIR = Path('logs')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    file_path=LOG_DIR / Path(logger_name)

    #避免重复添加 Handler
    if logger.handlers:
        return logger
    file_path.parent.mkdir(parents=True, exist_ok=True)
    #定义日志格式:时间asctime - Logger名称name - 级别 - (文件名:行号) - 消息
    """
    %(asctime)s	时间	日志记录的精确时间。
    %(name)s	Logger 名称	对应 app_logger，用于区分不同模块的日志。
    %(levelname)s	级别名称	如 INFO, DEBUG, ERROR 等。
    %(filename)s	文件名	发生日志事件的 Python 文件名。
    %(lineno)d	行号	发生日志事件的代码行数。
    %(message)s	消息主体	用户传入的日志消息内容。
    """
    formatter = logging.Formatter(
        '%(asctime)s (%(filename)s:%(lineno)d):  %(message)s  <%(levelname)s> '
    )
    #handlers设置输出
    console_handler = logging.StreamHandler()

    #设置输出等级
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=file_path,
        when='midnight',
        interval=1,
        backupCount=7,
        encoding='utf-8',
        delay=False
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    #为logger模块设置触发器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

if __name__ == '__main__':
    app_logger = setup_logging("log_text")
    app_logger.info("="*30)
    app_logger.info("应用启动成功，start")
    app_logger.info("="*30)
    app_logger.debug("<UNK>")


    # 业务逻辑
    def process_data(value):
        if value < 0:
            # 警告信息 (控制台和文件都会出现)
            app_logger.warning("输入值 %s 为负数，已自动修正为 0。", value)
            return 0
        try:
            result = 10 / value
            return result
        except ZeroDivisionError:
            # 错误信息，使用 .exception() 自动捕获 traceback
            app_logger.exception("计算错误：除数不能为零！")
            return None


    # 调用案例
    process_data(5)
    process_data(-1)
    process_data(0)
    app_logger.info("end")