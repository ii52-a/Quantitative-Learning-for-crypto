import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication

from UI.Ui import GetbackUi
from UI.QtheaderWork import BacktestWorker

pytestmark = pytest.mark.skipif(not os.environ.get("QT_QPA_PLATFORM"),
                                reason="需要 offscreen Qt")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def wait_until_idle(app, window, timeout=60.0):
    deadline = time.time() + timeout
    while window._running and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    return not window._running


def test_ui_backtest_runs_in_worker(qapp, tmp_path):
    window = GetbackUi(csv_dir=tmp_path)
    window.show()
    window.h_combobox_kline_num.setCurrentText("500")

    window.start()
    assert not window.button.isEnabled()
    assert window.cancel_button.isEnabled()

    assert wait_until_idle(qapp, window), "回测未在超时内结束"
    text = window.text_resultput.toPlainText()
    assert "交易次数" in text
    assert "最终资金" in text
    assert window._workers == set()
    assert window.button.isEnabled()
    window.close()


def test_ui_cancel_returns_to_idle(qapp, tmp_path):
    window = GetbackUi(csv_dir=tmp_path)
    window.show()
    window.h_combobox_kline_num.setCurrentText("10000")

    window.start()
    window.cancel()
    assert wait_until_idle(qapp, window, timeout=90), "取消后未回到空闲状态"
    assert window.button.isEnabled()
    assert window._workers == set()
    window.close()


def test_ui_missing_database_reports_error_instead_of_crashing(qapp, tmp_path):
    window = GetbackUi(csv_dir=tmp_path)
    window.show()
    window.h_combobox_trading_pair.setCurrentText("BTCUSDT")   # 本地没有该交易对的聚合库
    window.h_combobox_kline_num.setCurrentText("300")

    window.start()
    assert wait_until_idle(qapp, window, timeout=30), "错误路径没有回到空闲状态"
    assert "失败" in window.text_resultput.toPlainText()
    assert window.button.isEnabled()
    assert window._workers == set()
    window.close()


def test_worker_exposes_no_blocking_api():
    worker = BacktestWorker("ETHUSDT", "30min", 100)
    assert hasattr(worker, "cancel")
    assert worker._stop is False
    worker.cancel()
    assert worker._stop is True
