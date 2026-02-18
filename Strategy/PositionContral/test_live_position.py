import unittest
from datetime import datetime

from Strategy.PositionContral.live_position import LivePosition, Side
from Strategy.PositionContral.live_position_manager import (
    LivePositionManager,
    RiskConfig,
    TradeRecord,
)


class TestLivePosition(unittest.TestCase):
    def setUp(self):
        self.position = LivePosition(symbol="BTCUSDT")

    def test_initial_state(self):
        self.assertEqual(self.position.symbol, "BTCUSDT")
        self.assertEqual(self.position.side, Side.EMPTY)
        self.assertEqual(self.position.quantity, 0.0)
        self.assertFalse(self.position.is_open)

    def test_open_long_position(self):
        self.position.side = Side.LONG
        self.position.quantity = 0.1
        self.position.entry_price = 50000.0
        self.position.margin = 1000.0

        self.assertTrue(self.position.is_open)
        self.assertEqual(self.position.notional_value, 5000.0)

    def test_calc_pnl_long(self):
        self.position.side = Side.LONG
        self.position.quantity = 0.1
        self.position.entry_price = 50000.0

        pnl = self.position.calc_pnl(51000.0)
        self.assertEqual(pnl, 100.0)

        pnl = self.position.calc_pnl(49000.0)
        self.assertEqual(pnl, -100.0)

    def test_calc_pnl_short(self):
        self.position.side = Side.SHORT
        self.position.quantity = 0.1
        self.position.entry_price = 50000.0

        pnl = self.position.calc_pnl(49000.0)
        self.assertEqual(pnl, 100.0)

        pnl = self.position.calc_pnl(51000.0)
        self.assertEqual(pnl, -100.0)

    def test_calc_pnl_percent(self):
        self.position.side = Side.LONG
        self.position.quantity = 0.1
        self.position.entry_price = 50000.0

        pct = self.position.calc_pnl_percent(55000.0)
        self.assertEqual(pct, 0.1)

    def test_stop_loss_long(self):
        self.position.side = Side.LONG
        self.position.quantity = 0.1
        self.position.entry_price = 50000.0
        self.position.stop_loss = 49000.0

        self.assertTrue(self.position.check_stop_loss(48500.0))
        self.assertTrue(self.position.check_stop_loss(49000.0))
        self.assertFalse(self.position.check_stop_loss(49500.0))

    def test_stop_loss_short(self):
        self.position.side = Side.SHORT
        self.position.quantity = 0.1
        self.position.entry_price = 50000.0
        self.position.stop_loss = 51000.0

        self.assertTrue(self.position.check_stop_loss(51500.0))
        self.assertTrue(self.position.check_stop_loss(51000.0))
        self.assertFalse(self.position.check_stop_loss(50500.0))

    def test_take_profit_long(self):
        self.position.side = Side.LONG
        self.position.quantity = 0.1
        self.position.entry_price = 50000.0
        self.position.take_profit = 52000.0

        self.assertTrue(self.position.check_take_profit(52500.0))
        self.assertTrue(self.position.check_take_profit(52000.0))
        self.assertFalse(self.position.check_take_profit(51500.0))

    def test_take_profit_short(self):
        self.position.side = Side.SHORT
        self.position.quantity = 0.1
        self.position.entry_price = 50000.0
        self.position.take_profit = 48000.0

        self.assertTrue(self.position.check_take_profit(47500.0))
        self.assertTrue(self.position.check_take_profit(48000.0))
        self.assertFalse(self.position.check_take_profit(48500.0))

    def test_to_dict(self):
        self.position.side = Side.LONG
        self.position.quantity = 0.1
        self.position.entry_price = 50000.0

        d = self.position.to_dict()
        self.assertEqual(d["symbol"], "BTCUSDT")
        self.assertEqual(d["side"], "LONG")
        self.assertEqual(d["quantity"], 0.1)


class TestLivePositionManager(unittest.TestCase):
    def setUp(self):
        self.risk_config = RiskConfig(
            max_position_ratio=0.3,
            default_leverage=5,
            default_stop_loss_pct=0.02,
            default_take_profit_pct=0.05,
        )
        self.manager = LivePositionManager(
            initial_balance=10000.0,
            risk_config=self.risk_config,
        )

    def test_initial_state(self):
        self.assertEqual(self.manager.balance, 10000.0)
        self.assertEqual(self.manager.total_margin, 0.0)
        self.assertEqual(self.manager.available_balance, 10000.0)
        self.assertEqual(len(self.manager.open_positions), 0)

    def test_open_long_position(self):
        pos = self.manager.open_position(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=0.1,
            price=50000.0,
            reason="测试开多",
        )

        self.assertIsNotNone(pos)
        self.assertEqual(pos.side, Side.LONG)
        self.assertEqual(pos.quantity, 0.1)
        self.assertEqual(pos.entry_price, 50000.0)
        self.assertEqual(pos.margin, 1000.0)
        self.assertEqual(self.manager.total_margin, 1000.0)
        self.assertEqual(self.manager.available_balance, 9000.0)

    def test_open_short_position(self):
        pos = self.manager.open_position(
            symbol="ETHUSDT",
            side=Side.SHORT,
            quantity=1.0,
            price=3000.0,
            reason="测试开空",
        )

        self.assertIsNotNone(pos)
        self.assertEqual(pos.side, Side.SHORT)
        self.assertEqual(pos.margin, 600.0)

    def test_insufficient_balance(self):
        pos = self.manager.open_position(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=10.0,
            price=50000.0,
        )

        self.assertIsNone(pos)

    def test_close_position_profit(self):
        self.manager.open_position(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=0.1,
            price=50000.0,
        )

        record = self.manager.close_position(
            symbol="BTCUSDT",
            price=51000.0,
            reason="止盈平仓",
        )

        self.assertIsNotNone(record)
        self.assertEqual(record.pnl, 100.0)
        self.assertEqual(self.manager.balance, 10100.0)

        pos = self.manager.get_position("BTCUSDT")
        self.assertFalse(pos.is_open)

    def test_close_position_loss(self):
        self.manager.open_position(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=0.1,
            price=50000.0,
        )

        record = self.manager.close_position(
            symbol="BTCUSDT",
            price=49000.0,
            reason="止损平仓",
        )

        self.assertIsNotNone(record)
        self.assertEqual(record.pnl, -100.0)
        self.assertEqual(self.manager.balance, 9900.0)

    def test_partial_close(self):
        self.manager.open_position(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=0.1,
            price=50000.0,
        )

        record = self.manager.close_position(
            symbol="BTCUSDT",
            quantity=0.05,
            price=51000.0,
            reason="部分平仓",
        )

        self.assertIsNotNone(record)
        self.assertEqual(record.pnl, 50.0)

        pos = self.manager.get_position("BTCUSDT")
        self.assertTrue(pos.is_open)
        self.assertEqual(pos.quantity, 0.05)

    def test_update_pnl(self):
        self.manager.open_position(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=0.1,
            price=50000.0,
        )

        self.manager.update_pnl({"BTCUSDT": 51000.0})

        pos = self.manager.get_position("BTCUSDT")
        self.assertEqual(pos.unrealized_pnl, 100.0)

    def test_auto_stop_loss_take_profit(self):
        pos = self.manager.open_position(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=0.1,
            price=50000.0,
        )

        self.assertIsNotNone(pos.stop_loss)
        self.assertIsNotNone(pos.take_profit)
        self.assertAlmostEqual(pos.stop_loss, 49000.0, places=2)
        self.assertAlmostEqual(pos.take_profit, 52500.0, places=2)

    def test_check_risk_events_stop_loss(self):
        self.manager.open_position(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=0.1,
            price=50000.0,
            stop_loss=49000.0,
        )

        events = self.manager.check_risk_events({"BTCUSDT": 48500.0})

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "STOP_LOSS")

    def test_check_risk_events_take_profit(self):
        self.manager.open_position(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=0.1,
            price=50000.0,
            take_profit=52000.0,
        )

        events = self.manager.check_risk_events({"BTCUSDT": 52500.0})

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "TAKE_PROFIT")

    def test_handle_risk_events(self):
        self.manager.open_position(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=0.1,
            price=50000.0,
            stop_loss=49000.0,
        )

        records = self.manager.handle_risk_events({"BTCUSDT": 48000.0})

        self.assertEqual(len(records), 1)
        pos = self.manager.get_position("BTCUSDT")
        self.assertFalse(pos.is_open)

    def test_close_all_positions(self):
        self.manager.open_position("BTCUSDT", Side.LONG, 0.1, 50000.0)
        self.manager.open_position("ETHUSDT", Side.SHORT, 1.0, 3000.0)

        records = self.manager.close_all_positions(
            prices={"BTCUSDT": 51000.0, "ETHUSDT": 2900.0},
            reason="全部平仓",
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(len(self.manager.open_positions), 0)

    def test_get_statistics(self):
        self.manager.open_position("BTCUSDT", Side.LONG, 0.1, 50000.0)
        self.manager.close_position("BTCUSDT", price=51000.0)

        self.manager.open_position("BTCUSDT", Side.LONG, 0.1, 50000.0)
        self.manager.close_position("BTCUSDT", price=49000.0)

        stats = self.manager.get_statistics()

        self.assertEqual(stats["total_trades"], 2)
        self.assertEqual(stats["win_trades"], 1)
        self.assertEqual(stats["lose_trades"], 1)
        self.assertEqual(stats["win_rate"], 50.0)

    def test_calc_quantity(self):
        qty = self.manager.calc_quantity("BTCUSDT", 50000.0, risk_usdt=1000.0)
        self.assertEqual(qty, 0.1)

    def test_add_position(self):
        self.manager.open_position(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=0.1,
            price=50000.0,
        )

        pos = self.manager.open_position(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=0.1,
            price=51000.0,
        )

        self.assertIsNotNone(pos)
        self.assertEqual(pos.quantity, 0.2)
        self.assertAlmostEqual(pos.entry_price, 50500.0, places=2)

    def test_order_callback(self):
        callback_records = []

        def on_order(record: TradeRecord):
            callback_records.append(record)

        self.manager.set_order_callback(on_order)
        self.manager.open_position("BTCUSDT", Side.LONG, 0.1, 50000.0)

        self.assertEqual(len(callback_records), 1)
        self.assertEqual(callback_records[0].action, "OPEN")


class TestIntegration(unittest.TestCase):
    def test_full_trade_cycle(self):
        manager = LivePositionManager(initial_balance=10000.0)

        pos = manager.open_position(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=0.1,
            price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
        )

        self.assertTrue(pos.is_open)
        self.assertEqual(manager.total_margin, 1000.0)

        manager.update_pnl({"BTCUSDT": 51500.0})
        self.assertEqual(pos.unrealized_pnl, 150.0)

        record = manager.close_position("BTCUSDT", price=52000.0)
        self.assertEqual(record.pnl, 200.0)
        self.assertEqual(manager.balance, 10200.0)

        stats = manager.get_statistics()
        self.assertEqual(stats["win_trades"], 1)
        self.assertEqual(stats["win_rate"], 100.0)

    def test_stop_loss_triggered(self):
        manager = LivePositionManager(initial_balance=10000.0)

        manager.open_position(
            symbol="BTCUSDT",
            side=Side.LONG,
            quantity=0.1,
            price=50000.0,
            stop_loss=49000.0,
        )

        manager.update_pnl({"BTCUSDT": 48500.0})
        records = manager.handle_risk_events({"BTCUSDT": 48500.0})

        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].pnl < 0)


if __name__ == "__main__":
    unittest.main()
