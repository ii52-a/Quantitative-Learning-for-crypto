"""修复后验证：UI 不再卡死/崩溃，回测结果正确。

Run:  .venv\\Scripts\\python.exe audit\\verify_after_fix.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt5.QtCore import qInstallMessageHandler  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

messages = []
qInstallMessageHandler(lambda mode, ctx, msg: messages.append(msg))


def wait_idle(app, window, timeout=120.0):
    deadline = time.time() + timeout
    while window._running and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    return not window._running


def main() -> int:
    from Config import ApiConfig
    from UI.Ui import GetbackUi

    print("== 1. 导入 ==")
    print("  UI.Ui import OK；数据目录:", ApiConfig.LOCAL_DATA_SQLITE_DIR)

    app = QApplication([])

    print("\n== 2. 打开窗口并跑一次回测（500 根）==")
    window = GetbackUi(csv_dir=ROOT / "audit" / "ui_run")
    window.show()
    window.h_combobox_kline_num.setCurrentText("500")
    window.button.click()  # 走真实的 Qt 信号路径
    print("  点击后 button.isEnabled():", window.button.isEnabled(), "(应为 False)")
    ok = wait_idle(app, window)
    print("  回测是否结束:", ok, "| 剩余 worker:", len(window._workers))
    print("  结果面板:\n" + "\n".join("    " + line for line in
                                    window.text_resultput.toPlainText().splitlines()))
    shot = ROOT / "audit" / "ui_after_fix.png"
    window.grab().save(str(shot))
    print("  截图:", shot)

    print("\n== 3. 运行中关闭窗口（验证 QThread 生命周期）==")
    window2 = GetbackUi(csv_dir=ROOT / "audit" / "ui_run")
    window2.show()
    window2.h_combobox_kline_num.setCurrentText("10000")
    window2.start()
    window2.close()
    for _ in range(200):
        app.processEvents()
        time.sleep(0.01)

    print("\n== 4. Api 构造是否联网 ==")
    from Data.Api import Api
    start = time.perf_counter()
    Api(ping=False)
    print(f"  Api(ping=False) 构造耗时 {time.perf_counter() - start:.3f}s")

    layout_warnings = [m for m in messages if "QLayout" in m]
    print("\n== 5. Qt 警告 ==")
    print("  QLayout 警告条数:", len(layout_warnings))
    print("  总警告条数:", len(messages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
