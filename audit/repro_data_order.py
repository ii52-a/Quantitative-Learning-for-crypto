"""Read-only checks of the data layer (no writes, no network).

（修复前复现脚本：描述的是 2026-11-11 修复前的行为，保留作对比；
 修复后的行为见 tests/test_data.py / tests/test_aggregation.py。）

Run:  .venv\\Scripts\\python.exe audit\\repro_data_order.py

Checks
  1. where each module reads/writes SQLite (CWD-relative vs PROJECT_ROOT)
  2. row order returned by SqliteOper.read_range_kline(start_time=...)
  3. bar alignment produced by KlineProcess (origin='start_day' in Asia/Shanghai)
  4. whether the newest aggregated bar is an unfinished (partial) bar
"""
import sqlite3
import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_fake = types.ModuleType("app_logger.logger_setup")


class _Logger:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


_fake.Logger = _Logger
sys.modules["app_logger.logger_setup"] = _fake

from Config import ApiConfig  # noqa: E402
from Data import sqlite_oper  # noqa: E402
from Data.kline_process import KlineProcess  # noqa: E402


def where():
    print("== 1. path resolution ==")
    print(f"  PROJECT_ROOT                 : {ApiConfig.PROJECT_ROOT}")
    print(f"  SqliteBase.base_path         : {sqlite_oper.SqliteBase.base_path!r}  (CWD-relative)")
    print(f"  KlineProcess.LOCAL_PATH      : {KlineProcess.LOCAL_PATH!r}  (CWD-relative)")
    print("  SqliteOper.with_open_and_func: PROJECT_ROOT/'Data'/'LocalData'/f'{symbol}_aggerate.db'")
    root_db = ROOT / "LocalData" / "ETHUSDT_aggerate.db"
    read_db = ROOT / "Data" / "LocalData" / "ETHUSDT_aggerate.db"
    print(f"  write-side aggregate exists  : {root_db.exists()}  {root_db}")
    print(f"  read-side  aggregate exists  : {read_db.exists()}  {read_db}")
    if read_db.exists():
        print(f"  read-side  mtime             : {pd.Timestamp(read_db.stat().st_mtime, unit='s')}")
    root_base = ROOT / "LocalData" / "ETHUSDT_base.db"
    if root_base.exists():
        print(f"  write-side base mtime        : {pd.Timestamp(root_base.stat().st_mtime, unit='s')}")
    print()


def order():
    print("== 2. read_range_kline(start_time=...) row order ==")
    db = ROOT / "Data" / "LocalData" / "ETHUSDT_aggerate.db"
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        cur = conn.cursor()
        start = pd.Timestamp("2023-05-09 15:00", tz="UTC")
        df = sqlite_oper.SqliteOper.read_range_kline(
            cursor=cur, number=5, kline="kline_30min", start_time=start
        )
        print("  first 5 index values:", [str(x) for x in df.index[:5]])
        print("  monotonic increasing:", df.index.is_monotonic_increasing,
              "  (False => MACD is computed on a time-reversed series)")
        n = sqlite_oper.SqliteOper.read_range_kline(
            cursor=cur, number=5, kline="kline_30min",
            start_time=start, endtime=start + pd.Timedelta(hours=3)
        )
        print(f"  start+end branch ignores `number`: asked 5, got {len(n)} rows, "
              f"no ORDER BY in SQL")
    print()


def alignment():
    print("== 3. aggregated bar alignment (resample origin='start_day' in Asia/Shanghai) ==")
    db = ROOT / "Data" / "LocalData" / "ETHUSDT_aggerate.db"
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        cur = conn.cursor()
        for table in ("kline_4h", "kline_6h", "kline_1d", "kline_3d", "kline_1W", "kline_1ME"):
            ts = [pd.to_datetime(r[0], unit="ms", utc=True)
                  for r in cur.execute(f"SELECT timestamp FROM {table} ORDER BY timestamp LIMIT 3")]
            print(f"  {table:9s} UTC starts={[str(x)[11:16] for x in ts]}  "
                  f"(Binance uses 00:00 UTC grid for 1d/1W/1ME)")
    print()


def partial_bar():
    print("== 4. newest aggregated bar vs newest 1m bar ==")
    base = ROOT / "Data" / "LocalData" / "ETHUSDT_base.db"
    agg = ROOT / "Data" / "LocalData" / "ETHUSDT_aggerate.db"
    stamps = {}
    for db, table in ((base, "kline"), (agg, "kline_30min")):
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            hi = conn.execute(f"SELECT max(timestamp) FROM {table}").fetchone()[0]
        stamps[table] = pd.to_datetime(hi, unit="ms", utc=True)
        print(f"  {table:11s} last bar opens at {stamps[table]}")
    lag = stamps["kline"] - stamps["kline_30min"]
    print(f"  => aggregate lags the 1m source by {lag} (aggregation is not incremental)")
    print("  => and _agg_kline keeps the current, unfinished resample bucket (no drop of the last row)")


if __name__ == "__main__":
    where()
    order()
    alignment()
    partial_bar()
