"""Reproduce position/accounting bugs without touching the DB or Logs/.

（修复前复现脚本：注释描述的是 2026-11-11 修复前的行为，保留作对比；
 修复后的正确行为见 tests/test_position.py 与 audit/verify_after_fix.py。）

Run:  .venv\\Scripts\\python.exe audit\\repro_position_math.py

Expected (correct) behaviour is printed next to each case; the current code
deviates in cases 2-5.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The real logger writes into ./Logs which may be locked; stub it out.
_fake = types.ModuleType("app_logger.logger_setup")


class _Logger:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


_fake.Logger = _Logger
sys.modules["app_logger.logger_setup"] = _fake

from Strategy.PositionContral.PonsitionContral import PositionControl  # noqa: E402
from Strategy.StrategyTypes import (  # noqa: E402
    PositionChange,
    PositionSet,
    PositionSignal,
    StrategyResult,
)


def send(pc, signal, price, size=1.0):
    pc.main(StrategyResult(symbol="ETHUSDT", direction=signal, size=size,
                           execution_price=price, execution_time=price))


def dump(tag, pc):
    p = pc.position.get("ETHUSDT")
    print(f"  {tag:<28s} balance={pc.all_usdt:9.2f} margin={(p.margin_usdt if p else 0):8.2f} "
          f"nominal={(p.get_nominal_value if p else 0):9.2f} avg={(p.get_avg_price if p else 0):8.3f} "
          f"dir={(p.direction if p else 0):+d}")


def case1():
    print("[1] long open -> full close   (expect +500 on 1000 margin @5x, price 100->110)")
    pc = PositionControl(1000)
    send(pc, PositionSignal.LONG, 100)
    dump("open LONG@100", pc)
    send(pc, PositionSignal.FULL, 110)
    dump("full close@110", pc)
    print(f"  => balance={pc.all_usdt:.2f} (expected 1500.00), trades={pc.total}, win={pc.win}\n")


def case2():
    print("[2] reversal LONG->SHORT        (expect nominal -5000, dir -1; no fee)")
    pc = PositionControl(1000)
    send(pc, PositionSignal.LONG, 100)
    send(pc, PositionSignal.SHORT, 100)
    dump("after SHORT signal", pc)
    print("  => nominal doubled to +10000 and direction still long (+1): exposure 2x, no flip\n")


def case3():
    print("[3] FULL signal while flat      (expect: ignored)")
    pc = PositionControl(1000)
    send(pc, PositionSignal.FULL, 100)
    dump("after FULL while flat", pc)
    print("  => opened a 1000-margin long instead of doing nothing\n")


def case4():
    print("[4] SHORT signal while flat     (expect: short, nominal < 0)")
    pc = PositionControl(1000)
    send(pc, PositionSignal.SHORT, 100)
    dump("after SHORT while flat", pc)
    print("  => opened a long (nominal +5000): short side is unreachable\n")


def case5():
    print("[5] partial close of a long     (expect d_pnl +250, margin 500, nominal 2500)")
    pc = PositionControl(1000)
    send(pc, PositionSignal.LONG, 100)
    pos = pc.position["ETHUSDT"]
    pos.execute(PositionSet(signal=PositionChange.PARTIAL, changed_usdt=500, price=110, open_time=110))
    print(f"  => d_pnl={pos.d_pnl:+.2f} margin={pos.margin_usdt:.2f} nominal={pos.get_nominal_value:.2f}")
    print("  => sign inverted AND notional grows instead of shrinking\n")


def case6():
    print("[6] capital summary of a 0-trade run (cta_macd_strategy.main last line)")
    pc = PositionControl(1000)
    try:
        _ = pc.win / pc.total * 100
    except ZeroDivisionError as exc:
        print(f"  => {type(exc).__name__}: division by zero when total==0\n")


def case7():
    print("[7] open when available balance is 0 (account wiped out)")
    pc = PositionControl(0)
    try:
        send(pc, PositionSignal.LONG, 100)
        print("  => no error\n")
    except ZeroDivisionError as exc:
        print(f"  => {type(exc).__name__}: {exc}  (Position._open: cur_persent = 0/0)\n")


if __name__ == "__main__":
    for fn in (case1, case2, case3, case4, case5, case6, case7):
        fn()
