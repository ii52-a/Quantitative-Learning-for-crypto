from pathlib import Path

import pandas as pd
import pytest

from Config import ApiConfig
from Strategy.CTA.cta_macd_strategy import StrategyMacd
from Strategy.data_service import Dataservice

DB = Path(ApiConfig.LOCAL_DATA_SQLITE_DIR) / "ETHUSDT_aggerate.db"
needs_db = pytest.mark.skipif(not DB.exists(), reason="需要本地聚合库 Data/LocalData/ETHUSDT_aggerate.db")


@needs_db
def test_get_data_with_start_time_is_ascending():
    df = Dataservice.get_data("ETHUSDT", "30min", length=200,
                             start_time=pd.Timestamp("2025-01-01", tz="UTC"))
    assert not df.empty
    assert df.index.is_monotonic_increasing
    assert df.index[0] >= pd.Timestamp("2025-01-01", tz="UTC").tz_convert("Asia/Shanghai")


@needs_db
def test_backtest_summary_is_consistent(tmp_path):
    strategy = StrategyMacd("ETHUSDT", "30min", number=1000)
    out = tmp_path / "trades.csv"
    summary = strategy.main(output_csv=out)

    assert summary["bars"] == 1000
    assert summary["trades"] == summary["win"] + summary["lose"] + summary["breakeven"]
    assert summary["final"] == pytest.approx(summary["initial"] + summary["net_pnl"])
    assert summary["gross_pnl"] == pytest.approx(summary["net_pnl"] + summary["fees"])
    assert 0 <= summary["win_rate"] <= 100

    trades = pd.read_csv(out)
    assert len(trades) == summary["trades"]
    assert trades["fee"].sum() == pytest.approx(summary["fees"], rel=1e-4)
    assert (trades["fee"] >= 0).all()
    assert trades[["open", "close"]].gt(0).all().all()


@needs_db
def test_backtest_can_be_cancelled():
    strategy = StrategyMacd("ETHUSDT", "30min", number=2000)
    summary = strategy.main(should_stop=lambda: True)
    # 取消后仍会强制平掉当前仓位并给出完整汇总
    assert summary["stopped"] is True
    assert summary["final"] == pytest.approx(summary["initial"] + summary["net_pnl"])
