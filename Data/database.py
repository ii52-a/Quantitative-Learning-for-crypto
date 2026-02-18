"""
SQLite数据库管理模块

提供：
- 统一的数据库连接管理
- 高效的查询接口
- 事务处理
- 索引优化
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import pandas as pd

from Config import ApiConfig
from app_logger.logger_setup import Logger

logger = Logger(__name__)


class DatabaseError(Exception):
    """数据库错误基类"""
    pass


class ConnectionError(DatabaseError):
    """连接错误"""
    pass


@dataclass
class DatabaseConfig:
    """数据库配置"""
    base_dir: Path = Path(ApiConfig.LOCAL_DATA_SQLITE_DIR)
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    cache_size: int = -64000
    busy_timeout: int = 30000


class DatabaseConnection:
    """数据库连接管理器"""

    def __init__(self, db_path: Path, config: DatabaseConfig | None = None):
        self.db_path = db_path
        self.config = config or DatabaseConfig()
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,
        )
        try:
            self._configure_connection(conn)
            yield conn
        except Exception as e:
            logger.error(f"数据库操作失败: {e}")
            raise DatabaseError(f"数据库操作失败: {e}")
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level="DEFERRED",
        )
        try:
            self._configure_connection(conn)
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"事务执行失败: {e}")
            raise DatabaseError(f"事务执行失败: {e}")
        finally:
            conn.close()

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        conn.execute(f"PRAGMA journal_mode={self.config.journal_mode}")
        conn.execute(f"PRAGMA synchronous={self.config.synchronous}")
        conn.execute(f"PRAGMA cache_size={self.config.cache_size}")
        conn.execute(f"PRAGMA busy_timeout={self.config.busy_timeout}")


class KlineRepository:
    """K线数据仓库"""

    TABLE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS {table_name} (
        timestamp INTEGER PRIMARY KEY,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        close_time INTEGER,
        quote_volume REAL,
        trades INTEGER,
        taker_buy_base REAL,
        taker_buy_quote REAL
    )
    """

    INDEX_SCHEMA = """
    CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp
    ON {table_name} (timestamp DESC)
    """

    def __init__(self, symbol: str, config: DatabaseConfig | None = None):
        self.symbol = symbol
        self.config = config or DatabaseConfig()
        self.base_db = DatabaseConnection(
            self.config.base_dir / f"{symbol}_base.db",
            self.config,
        )
        self.aggregate_db = DatabaseConnection(
            self.config.base_dir / f"{symbol}_aggregate.db",
            self.config,
        )

    def init_tables(self, intervals: list[str] | None = None) -> None:
        intervals = intervals or ["kline"]

        with self.base_db.transaction() as conn:
            for interval in intervals:
                self._create_table(conn, interval)

        logger.info(f"[{self.symbol}] 数据表初始化完成")

    def _create_table(self, conn: sqlite3.Connection, table_name: str) -> None:
        conn.execute(self.TABLE_SCHEMA.format(table_name=table_name))
        conn.execute(self.INDEX_SCHEMA.format(table_name=table_name))

    def insert_klines(
        self,
        df: pd.DataFrame,
        table_name: str = "kline",
        use_base_db: bool = True,
    ) -> int:
        if df.empty:
            return 0

        db = self.base_db if use_base_db else self.aggregate_db
        records = self._df_to_records(df)

        with db.transaction() as conn:
            conn.executemany(
                f"""
                INSERT OR REPLACE INTO {table_name} VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                records,
            )

        logger.debug(f"[{self.symbol}] 插入{table_name}: {len(records)}条")
        return len(records)

    def _df_to_records(self, df: pd.DataFrame) -> list[tuple]:
        records = []
        for idx, row in df.iterrows():
            timestamp = idx
            if isinstance(idx, pd.Timestamp):
                timestamp = int(idx.value / 1_000_000)

            close_time = row.get("close_time", timestamp)
            if isinstance(close_time, pd.Timestamp):
                close_time = int(close_time.value / 1_000_000)

            records.append((
                timestamp,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                close_time,
                float(row.get("quote_volume", row.get("quote_asset_volume", 0))),
                int(row.get("trades", row.get("number_of_trades", 0))),
                float(row.get("taker_buy_base", row.get("taker_buy_base_asset_volume", 0))),
                float(row.get("taker_buy_quote", row.get("taker_buy_quote_asset_volume", 0))),
            ))
        return records

    def get_klines(
        self,
        table_name: str = "kline",
        limit: int = 500,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        use_base_db: bool = True,
    ) -> pd.DataFrame:
        db = self.base_db if use_base_db else self.aggregate_db

        with db.connection() as conn:
            sql, params = self._build_query(table_name, limit, start_time, end_time)
            cursor = conn.execute(sql, params)
            columns = [
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote",
            ]
            df = pd.DataFrame(cursor.fetchall(), columns=columns)

        if df.empty:
            return df

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime").sort_index()
        return df

    def _build_query(
        self,
        table_name: str,
        limit: int,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> tuple[str, list]:
        conditions = []
        params = []

        if start_time:
            start_ms = int(start_time.timestamp() * 1000)
            conditions.append("timestamp >= ?")
            params.append(start_ms)

        if end_time:
            end_ms = int(end_time.timestamp() * 1000)
            conditions.append("timestamp <= ?")
            params.append(end_ms)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT * FROM {table_name}
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)

        return sql, params

    def get_latest_timestamp(self, table_name: str = "kline", use_base_db: bool = True) -> datetime | None:
        db = self.base_db if use_base_db else self.aggregate_db

        with db.connection() as conn:
            cursor = conn.execute(
                f"SELECT MAX(timestamp) FROM {table_name}"
            )
            result = cursor.fetchone()

        if result and result[0]:
            return datetime.fromtimestamp(result[0] / 1000, tz=timezone.utc)
        return None

    def get_time_span(self, table_name: str = "kline", use_base_db: bool = True) -> dict[str, Any]:
        db = self.base_db if use_base_db else self.aggregate_db

        with db.connection() as conn:
            cursor = conn.execute(f"""
                SELECT
                    MIN(timestamp) as start_time,
                    MAX(timestamp) as end_time,
                    COUNT(*) as count
                FROM {table_name}
            """)
            result = cursor.fetchone()

        if result and result[0]:
            return {
                "start": datetime.fromtimestamp(result[0] / 1000, tz=timezone.utc),
                "end": datetime.fromtimestamp(result[1] / 1000, tz=timezone.utc),
                "count": result[2],
            }
        return {"start": None, "end": None, "count": 0}

    def get_count(self, table_name: str = "kline", use_base_db: bool = True) -> int:
        db = self.base_db if use_base_db else self.aggregate_db

        with db.connection() as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]

    def table_exists(self, table_name: str, use_base_db: bool = True) -> bool:
        db = self.base_db if use_base_db else self.aggregate_db

        with db.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return cursor.fetchone() is not None


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, config: DatabaseConfig | None = None):
        self.config = config or DatabaseConfig()
        self._repositories: dict[str, KlineRepository] = {}

    def get_repository(self, symbol: str) -> KlineRepository:
        if symbol not in self._repositories:
            self._repositories[symbol] = KlineRepository(symbol, self.config)
        return self._repositories[symbol]

    def get_status(self, symbol: str) -> dict[str, Any]:
        repo = self.get_repository(symbol)
        base_span = repo.get_time_span("kline", use_base_db=True)
        return {
            "symbol": symbol,
            "base_data": base_span,
        }
