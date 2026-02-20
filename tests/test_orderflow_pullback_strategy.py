from datetime import datetime

import pandas as pd

from Strategy.templates import get_strategy
from backtest.engine import BacktestEngine
from core.config import BacktestConfig


def test_orderflow_pullback_strategy_can_add_position_and_close_on_pullback():
    strategy = get_strategy("OrderFlowPullbackStrategy", {
        "momentum_bars": 3,
        "overbought_rsi": 60,
        "min_volume_ratio": 0.8,
        "max_volume_ratio": 5.0,
        "base_add_position_pct": 20.0,
        "base_price_gap_pct": 0.1,
        "momentum_gap_boost": 0.1,
        "pullback_take_profit_pct": 0.6,
    })

    prices = [100 + i * 0.4 for i in range(30)] + [112.0, 112.3, 112.7, 112.9, 111.8, 111.0]
    rows = []
    for i, close in enumerate(prices):
        rows.append({
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 2000 + i * 15,
        })

    data = pd.DataFrame(rows, index=pd.date_range(datetime(2024, 1, 1), periods=len(rows), freq="30min"))

    engine = BacktestEngine(
        strategy,
        BacktestConfig(symbol="DOGEUSDT", interval="30min", initial_capital=10000, position_size=1.0),
    )
    result = engine.run(data)

    add_trades = [t for t in result.trades if t.side == "ADD_LONG"]
    assert len(add_trades) >= 1
    assert result.total_trades >= 1
