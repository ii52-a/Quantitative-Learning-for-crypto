import math

import pytest

from Strategy.PositionContral.PonsitionContral import PositionControl
from Strategy.StrategyTypes import (
    PositionChange,
    PositionSet,
    PositionSignal,
    StrategyResult,
)

FEE = 0.0004


def send(pc: PositionControl, signal: PositionSignal, price: float, size: float = 1.0):
    return pc.main(StrategyResult(
        symbol="ETHUSDT", direction=signal, size=size,
        execution_price=price, execution_time=price,
    ))


def test_long_round_trip_charges_fees():
    pc = PositionControl(1000, leverage=5, fee_rate=FEE)
    send(pc, PositionSignal.LONG, 100)
    assert pc.position["ETHUSDT"].get_nominal_value == pytest.approx(5000)
    assert pc.all_usdt == pytest.approx(1000 - 5000 * FEE)      # 开仓手续费已扣

    send(pc, PositionSignal.FULL, 110)
    gross = (110 / 100 - 1) * 5000
    fees = 5000 * FEE * 2
    assert pc.all_usdt == pytest.approx(1000 + gross - fees)
    assert pc.total == 1 and pc.win == 1
    assert pc.trades[-1]["pnl"] == pytest.approx(gross - fees)
    assert pc.trades[-1]["fee"] == pytest.approx(fees)


def test_short_round_trip_profits_when_price_falls():
    pc = PositionControl(1000, leverage=5, fee_rate=FEE)
    send(pc, PositionSignal.SHORT, 100)
    assert pc.position["ETHUSDT"].get_nominal_value == pytest.approx(-5000)
    send(pc, PositionSignal.FULL, 90)
    gross = (90 / 100 - 1) * -5000
    assert pc.all_usdt == pytest.approx(1000 + gross - 5000 * FEE * 2)
    assert pc.total == 1 and pc.win == 1


def test_full_signal_while_flat_is_ignored():
    pc = PositionControl(1000, leverage=5, fee_rate=FEE)
    send(pc, PositionSignal.FULL, 100)
    assert pc.position == {}
    assert pc.all_usdt == 1000
    assert pc.total == 0


def test_reversal_flips_side_without_extra_exposure():
    pc = PositionControl(1000, leverage=5, fee_rate=FEE)
    send(pc, PositionSignal.LONG, 100)
    send(pc, PositionSignal.SHORT, 100)
    pos = pc.position["ETHUSDT"]
    assert pos.direction == -1
    assert pos.get_nominal_value == pytest.approx(-5000)
    assert pos.margin_usdt == pytest.approx(1000)
    assert pc.total == 1


def test_partial_close_reduces_size_and_realizes_profit():
    pc = PositionControl(1000, leverage=5, fee_rate=FEE)
    send(pc, PositionSignal.LONG, 100)
    pos = pc.position["ETHUSDT"]
    pos.execute(PositionSet(signal=PositionChange.PARTIAL, changed_usdt=500,
                            price=110, open_time=110))
    assert pos.get_nominal_value == pytest.approx(2500)
    assert pos.margin_usdt == pytest.approx(500)
    # 开仓手续费 2 + 平仓 (2500*0.0004)=1 -> d_pnl = -2 + (250 - 1) = 247
    assert pos.d_pnl == pytest.approx(247)


def test_zero_balance_does_not_raise():
    pc = PositionControl(0, leverage=5, fee_rate=FEE)
    send(pc, PositionSignal.LONG, 100)
    assert pc.position == {}
    assert pc.all_usdt == 0


def test_accounting_invariants():
    pc = PositionControl(1000, leverage=5, fee_rate=FEE)
    for price, signal in [(100, PositionSignal.LONG), (105, PositionSignal.FULL),
                          (110, PositionSignal.SHORT), (100, PositionSignal.FULL)]:
        send(pc, signal, price)

    assert pc.total == pc.win + pc.lose + pc.breakeven
    assert pc.all_usdt == pytest.approx(1000 + sum(t["pnl"] for t in pc.trades))
    assert pc.fees_total == pytest.approx(sum(t["fee"] for t in pc.trades), rel=1e-6)
    assert all(math.isfinite(t["pnl"]) for t in pc.trades)


def test_none_price_is_ignored_and_negative_price_rejected():
    pc = PositionControl(1000)
    assert pc.main(StrategyResult(symbol="ETHUSDT", direction=PositionSignal.LONG,
                                  size=1.0, execution_price=None, execution_time=1)) is None
    with pytest.raises(ValueError):
        pc.main(StrategyResult(symbol="ETHUSDT", direction=PositionSignal.LONG,
                               size=1.0, execution_price=-1, execution_time=1))
