"""Reproduce the UI freeze/crash cases (headless Qt, no network, no writes).

（修复前复现脚本：描述的是 2026-11-11 修复前的行为，保留作对比；
 修复后的验证见 tests/test_ui.py 与 audit/verify_after_fix.py。）

Run:  .venv\\Scripts\\python.exe audit\\repro_ui_freeze.py

Each case runs in its own subprocess because three of them terminate the
process (Qt aborts on unhandled exceptions in slots and on QThread destruction).
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

CASES = ("import", "layout", "click", "thread", "ping")


def _stub_api():
    """UI.Ui imports `data.Api` (lower-case) and the real Api needs network."""
    import types

    pkg = types.ModuleType("data")
    pkg.__path__ = []
    api_mod = types.ModuleType("data.Api")

    class _Api:
        def __init__(self):
            pass

    api_mod.Api = _Api
    pkg.Api = _Api
    sys.modules["data"] = pkg
    sys.modules["data.Api"] = api_mod


def case_import():
    import UI.Ui  # noqa: F401


def case_layout():
    _stub_api()
    from PyQt5.QtCore import qInstallMessageHandler
    from PyQt5.QtWidgets import QApplication

    seen = []
    qInstallMessageHandler(lambda mode, ctx, msg: seen.append(msg))
    from UI.Ui import GetbackUi

    app = QApplication([])
    win = GetbackUi()
    win.show()
    layout_msgs = [m for m in seen if "QLayout" in m]
    print("layout warnings:", len(layout_msgs))
    for m in layout_msgs:
        print("  ", m)
    del app, win


def case_click():
    _stub_api()
    from PyQt5.QtWidgets import QApplication
    from UI.Ui import GetbackUi

    app = QApplication([])
    win = GetbackUi()
    win.show()
    print("clicking 开始回测 ...", flush=True)
    win.button.click()  # exception escapes the slot -> Qt abort
    print("survived (should not happen)", flush=True)


def case_thread():
    _stub_api()
    import gc
    import time

    from PyQt5.QtWidgets import QApplication
    from UI.Ui import GetbackUi
    from UI.QtheaderWork import LoaclWorker

    app = QApplication([])
    win = GetbackUi()

    class SlowApi:
        def update_local_csv(self, **kw):
            time.sleep(1.5)

    def start_local():
        worker = LoaclWorker(api=SlowApi(), data_number=10, interval="30m",
                             symbol="ETHUSDT", file_path="x")
        worker.data_ready.connect(lambda d: None)
        worker.start()
        # `worker` is a local: the reference dies here (same bug as Ui.local_worker_start)

    start_local()
    gc.collect()
    print("worker reference dropped; sleeping", flush=True)
    time.sleep(2)
    print("survived", flush=True)


def case_ping():
    import inspect

    import binance.client as bc
    from Config import ApiConfig

    sig = inspect.signature(bc.Client.__init__)
    body = inspect.getsource(bc.Client.__init__)
    print("Client(ping=True) by default :", sig.parameters["ping"].default)
    print("calls self.ping() in __init__:", "self.ping()" in body)
    retry_on_api = getattr(ApiConfig, "MAX_RETRY")
    wait = getattr(ApiConfig, "WAITING_TIME")
    print(f"Ui.init_api retry worst case : {retry_on_api} x {wait}s = {retry_on_api * wait:.0f}s of blocked GUI thread")


def run_all():
    print(f"root: {ROOT}\n")
    for name in CASES:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--case", name]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip().splitlines()
        print(f"--- case {name}: exit={proc.returncode} (0x{proc.returncode & 0xFFFFFFFF:08X})")
        if out:
            print("\n".join("    " + line for line in out.splitlines()[-8:]))
        if err:
            print("    stderr:", err[-1] if err else "")
        print()


if __name__ == "__main__":
    if "--case" in sys.argv:
        globals()[f"case_{sys.argv[sys.argv.index('--case') + 1]}"]()
    else:
        run_all()
