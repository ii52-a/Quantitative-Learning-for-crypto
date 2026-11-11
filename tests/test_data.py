import sqlite3

import pandas as pd
import pytest

from Data.sqlite_oper import SqliteBase, SqliteOper


def make_frame(start: str = "2024-01-01", bars: int = 10, freq: str = "30min") -> pd.DataFrame:
    index = pd.date_range(start, periods=bars, freq=freq, tz="UTC").tz_convert("Asia/Shanghai")
    index.name = "timestamp"
    close = pd.Series(range(100, 100 + bars), index=index, dtype=float)
    df = pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 10.0,
        "quote_asset_volume": 1000.0,
        "number_of_trades": 5.0,
        "taker_buy_base_asset_volume": 4.0,
        "taker_buy_quote_asset_volume": 400.0,
    }, index=index)
    df["close_time"] = index + pd.Timedelta(minutes=30) - pd.Timedelta(milliseconds=1)
    return df


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    SqliteBase.init_sqlite(cursor=cursor, kline="kline_30min")
    yield connection, cursor
    connection.close()


def test_insert_then_read_is_ascending(conn):
    connection, cursor = conn
    df = make_frame()
    written = SqliteOper.insert_or_update_data(symbol="ETHUSDT", data=df.sample(frac=1, random_state=1),
                                               kline="kline_30min", cursor=cursor)
    assert written == len(df)

    out = SqliteOper.read_range_kline(cursor=cursor, number=5, kline="kline_30min")
    assert len(out) == 5
    assert out.index.is_monotonic_increasing
    assert list(out["close"]) == list(df["close"].iloc[-5:])


def test_start_time_branch_is_ascending(conn):
    _connection, cursor = conn
    df = make_frame(bars=20)
    SqliteOper.insert_or_update_data(symbol="ETHUSDT", data=df, kline="kline_30min", cursor=cursor)

    start = df.index[3]
    out = SqliteOper.read_range_kline(cursor=cursor, number=5, kline="kline_30min", start_time=start)
    assert out.index.is_monotonic_increasing
    assert out.index[0] == start
    assert len(out) == 5


def test_start_and_end_honours_number(conn):
    _connection, cursor = conn
    df = make_frame(bars=20)
    SqliteOper.insert_or_update_data(symbol="ETHUSDT", data=df, kline="kline_30min", cursor=cursor)

    out = SqliteOper.read_range_kline(cursor=cursor, number=3, kline="kline_30min",
                                      start_time=df.index[2], endtime=df.index[15])
    assert len(out) == 3
    assert out.index.is_monotonic_increasing


def test_empty_table_returns_empty_frame(conn):
    _connection, cursor = conn
    out = SqliteOper.read_range_kline(cursor=cursor, number=5, kline="kline_30min")
    assert out.empty
    assert isinstance(out.index, pd.DatetimeIndex)


def test_missing_table_raises(conn):
    _connection, cursor = conn
    with pytest.raises(ValueError):
        SqliteOper.read_range_kline(cursor=cursor, number=5, kline="kline_not_exist")


def test_read_newest_timestamp(conn):
    _connection, cursor = conn
    df = make_frame(bars=4)
    SqliteOper.insert_or_update_data(symbol="ETHUSDT", data=df, kline="kline_30min", cursor=cursor)
    newest = SqliteOper.read_newest_timestamp(cursor=cursor, kline="kline_30min")
    assert newest == int(df.index[-1].timestamp() * 1000)


def test_time_span_on_empty_table_does_not_raise(conn):
    _connection, cursor = conn
    assert SqliteOper.read_time_span(cursor=cursor, kline="kline_30min") is None
    assert "为空" in SqliteOper.read_time_span_str(cursor=cursor, kline="kline_30min")


def test_invalid_number_rejected(conn):
    _connection, cursor = conn
    with pytest.raises(ValueError):
        SqliteOper.read_range_kline(cursor=cursor, number=0, kline="kline_30min")
