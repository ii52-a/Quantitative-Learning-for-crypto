import sys
import traceback

from PyQt5.QtWidgets import QApplication

from app_logger.logger_setup import Logger

logger = Logger('main')


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    """兜住非槽路径的未捕获异常，保证窗口不会因为后台异常直接退出。"""
    logger.exception("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
    sys.__excepthook__(exc_type, exc_value, exc_tb)


if __name__ == '__main__':
    sys.excepthook = _excepthook
    app = QApplication(sys.argv)
    from UI.Ui import GetbackUi

    window = GetbackUi()
    window.show()
    sys.exit(app.exec_())
