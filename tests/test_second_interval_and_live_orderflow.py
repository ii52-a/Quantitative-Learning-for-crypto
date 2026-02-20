from datetime import datetime, timezone

import pandas as pd

from Data.data_service import UnifiedDataService, DataServiceConfig
from Trading.live_trader import LiveTrader, TradingConfig, TradingMode
from Strategy.templates import get_strategy


def test_second_interval_aggregation_from_trades():
    service = UnifiedDataService(DataServiceConfig(prefer_database=False, auto_init=False))

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # 构造2分钟成交，覆盖1s/5s/15s聚合
    fake_trades = []
    for i in range(120):
        t_ms = now_ms - (120 - i) * 1000
        fake_trades.append({"T": t_ms, "p": str(100 + i * 0.01), "q": "5"})

    service._fetch_agg_trades = lambda symbol, start_time_ms, end_time_ms, limit=1000: fake_trades

    df_1s = service.get_second_klines_from_api("BTCUSDT", "1s", limit=30)
    df_5s = service.get_second_klines_from_api("BTCUSDT", "5s", limit=30)
    df_15s = service.get_second_klines_from_api("BTCUSDT", "15s", limit=30)

    assert not df_1s.empty
    assert not df_5s.empty
    assert not df_15s.empty
    assert len(df_1s) >= len(df_5s) >= len(df_15s)


def test_orderflow_strategy_runs_in_realtime_simulation_mode():
    trader = LiveTrader(TradingConfig(mode=TradingMode.TEST, symbol="BTCUSDT", interval="1s"))
    strategy = get_strategy("OrderFlowWoolStrategy", {
        "momentum_bars": 3,
        "overbought_rsi": 55,
        "min_volume_ratio": 0.6,
        "max_volume_ratio": 10.0,
    })
    trader.set_strategy(strategy)

    for _ in range(80):
        trader._execute_orderflow_realtime()

    buf = trader._realtime_buffers.get("BTCUSDT")
    assert buf is not None
    assert len(buf) > 50
    assert isinstance(pd.DataFrame(list(buf)), pd.DataFrame)
