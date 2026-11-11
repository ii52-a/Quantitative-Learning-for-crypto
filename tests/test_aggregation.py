import sqlite3
from contextlib import closing

import pandas as pd
import pytest

from Data.kline_process import KlineProcess
from Data.sqlite_oper import SqliteBase, SqliteOper


def build_base_db(path, bars: int, start: str = "2024-01-01 00:00"):
    index = pd.date_range(start, periods=bars, freq="1min", tz="UTC").tz_convert("Asia/Shanghai")
    index.name = "timestamp"
    close = pd.Series(range(100, 100 + bars), index=index, dtype=float)
    df = pd.DataFrame({
        "open": close - 0.2, "high": close + 0.5, "low": close - 0.5, "close": close,
        "volume": 1.0, "quote_asset_volume": 100.0, "number_of_trades": 1.0,
        "taker_buy_base_asset_volume": 0.5, "taker_buy_quote_asset_volume": 50.0,
    }, index=index)
    df["close_time"] = index + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        cursor = conn.cursor()
        SqliteBase.init_sqlite(cursor=cursor, kline="kline")
        SqliteOper.insert_or_update_data(symbol="TESTUSDT", data=df, kline="kline", cursor=cursor)
        conn.commit()
    return df


def read_table(path, table):
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(f"SELECT timestamp, open, close, close_time FROM {table} ORDER BY timestamp").fetchall()


def test_aggregation_writes_complete_buckets_only(tmp_path, monkeypatch):
    monkeypatch.setattr(KlineProcess, "LOCAL_PATH", tmp_path)
    # 29 根 1min：00:00-00:28，30min 桶（00:00-00:30）未完成，应被丢弃
    build_base_db(tmp_path / "TESTUSDT_base.db", bars=29)

    written = KlineProcess.main("TESTUSDT")
    assert written > 0

    agg = tmp_path / "TESTUSDT_aggerate.db"
    assert agg.exists()
    rows_30 = read_table(agg, "kline_30min")
    assert rows_30 == []  # 未完成桶不写入

    rows_3 = read_table(agg, "kline_3min")
    assert len(rows_3) == 9  # 00:00-00:27 共 9 个完整 3min 桶


def test_aggregation_keeps_complete_bucket(tmp_path, monkeypatch):
    monkeypatch.setattr(KlineProcess, "LOCAL_PATH", tmp_path)
    build_base_db(tmp_path / "TESTUSDT_base.db", bars=30)  # 00:00-00:29，桶完整

    KlineProcess.main("TESTUSDT")
    rows_30 = read_table(tmp_path / "TESTUSDT_aggerate.db", "kline_30min")
    assert len(rows_30) == 1
    ts, open_, close, close_time = rows_30[0]
    assert close == 129.0
    assert open_ == 99.8
    assert close_time > ts


def test_aggregation_requires_base_db(tmp_path, monkeypatch):
    monkeypatch.setattr(KlineProcess, "LOCAL_PATH", tmp_path)
    with pytest.raises(FileNotFoundError):
        KlineProcess.main("MISSING")
