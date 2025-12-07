import logging
import logging.handlers
from pathlib import Path


def logging_setup(name):
    main_path=Path('logs')
    main_path.mkdir(exist_ok=True)


    module_path= main_path / f"{name}.module"

    file_path =module_path / f"{name}.log"
    module_path.mkdir(parents=True, exist_ok=True)
    file_path.parent.mkdir(parents=True, exist_ok=True)


    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ConsoleHandler = logging.StreamHandler()
    ConsoleHandler.setFormatter(formatter)
    ConsoleHandler.setLevel(logging.INFO)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=file_path,
        when='midnight',
        interval=1,
        backupCount=7,
        encoding='utf-8',
        delay=False,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(ConsoleHandler)
    logger.addHandler(file_handler)

    return logger

