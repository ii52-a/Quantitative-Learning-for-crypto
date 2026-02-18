"""
K线时间检测模块

检测K线数据是否符合交易所整点时间标准
支持不同交易所的时间规则
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd

from app_logger.logger_setup import Logger

logger = Logger(__name__)


@dataclass
class IntervalConfig:
    name: str
    minutes: int
    description: str


class KlineTimeValidator:
    INTERVALS: dict[str, IntervalConfig] = {
        "1min": IntervalConfig("1min", 1, "1分钟"),
        "3min": IntervalConfig("3min", 3, "3分钟"),
        "5min": IntervalConfig("5min", 5, "5分钟"),
        "15min": IntervalConfig("15min", 15, "15分钟"),
        "30min": IntervalConfig("30min", 30, "30分钟"),
        "1h": IntervalConfig("1h", 60, "1小时"),
        "2h": IntervalConfig("2h", 120, "2小时"),
        "4h": IntervalConfig("4h", 240, "4小时"),
        "6h": IntervalConfig("6h", 360, "6小时"),
        "12h": IntervalConfig("12h", 720, "12小时"),
        "1d": IntervalConfig("1d", 1440, "1天"),
        "1w": IntervalConfig("1w", 10080, "1周"),
    }

    # 周期别名映射
    INTERVAL_ALIASES: dict[str, str] = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "12h": "12h",
        "1d": "1d",
        "7d": "1w",
        "1w": "1w",
        "1M": "1d",
    }

    @classmethod
    def _normalize_interval(cls, interval: str) -> str:
        """标准化周期格式"""
        if interval in cls.INTERVALS:
            return interval
        if interval in cls.INTERVAL_ALIASES:
            return cls.INTERVAL_ALIASES[interval]
        return interval

    @classmethod
    def validate_timestamp(cls, timestamp: datetime, interval: str) -> dict[str, Any]:
        normalized_interval = cls._normalize_interval(interval)
        config = cls.INTERVALS.get(normalized_interval)
        if config is None:
            return {"valid": False, "error": f"不支持的周期: {interval}"}

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        total_minutes = timestamp.hour * 60 + timestamp.minute
        remainder = total_minutes % config.minutes

        is_valid = remainder == 0 and timestamp.second == 0 and timestamp.microsecond == 0

        expected_time = timestamp.replace(
            minute=total_minutes - remainder,
            second=0,
            microsecond=0
        ) if not is_valid else timestamp

        return {
            "valid": is_valid,
            "interval": normalized_interval,
            "interval_minutes": config.minutes,
            "timestamp": timestamp,
            "expected_time": expected_time if not is_valid else None,
            "deviation_minutes": remainder,
            "deviation_seconds": timestamp.second,
        }

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, interval: str) -> dict[str, Any]:
        if df.empty:
            return {"valid": False, "error": "数据为空"}

        normalized_interval = cls._normalize_interval(interval)
        results = []
        invalid_count = 0

        for idx, row in df.iterrows():
            timestamp = idx if isinstance(idx, datetime) else pd.to_datetime(idx, utc=True)
            result = cls.validate_timestamp(timestamp, normalized_interval)
            results.append(result)
            if not result["valid"]:
                invalid_count += 1

        return {
            "valid": invalid_count == 0,
            "total_count": len(df),
            "invalid_count": invalid_count,
            "valid_count": len(df) - invalid_count,
            "first_invalid": next((r for r in results if not r["valid"]), None),
            "details": results[:10] if invalid_count > 0 else [],
        }

    @classmethod
    def get_next_close_time(cls, interval: str, from_time: datetime | None = None) -> datetime:
        normalized_interval = cls._normalize_interval(interval)
        config = cls.INTERVALS.get(normalized_interval)
        if config is None:
            raise ValueError(f"不支持的周期: {interval}")

        if from_time is None:
            from_time = datetime.now(timezone.utc)
        elif from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)

        total_minutes = from_time.hour * 60 + from_time.minute
        remainder = total_minutes % config.minutes

        if remainder == 0 and from_time.second == 0:
            return from_time

        next_minutes = total_minutes + (config.minutes - remainder)
        next_hour = next_minutes // 60
        next_minute = next_minutes % 60

        next_close = from_time.replace(
            hour=next_hour % 24,
            minute=next_minute,
            second=0,
            microsecond=0
        )

        if next_hour >= 24:
            next_close = next_close + timedelta(days=1)

        return next_close

    @classmethod
    def get_remaining_seconds(cls, interval: str, from_time: datetime | None = None) -> int:
        next_close = cls.get_next_close_time(interval, from_time)
        current = from_time or datetime.now(timezone.utc)
        return int((next_close - current).total_seconds())


class KlineTimeReporter:
    @classmethod
    def generate_report(cls, df: pd.DataFrame, interval: str, symbol: str) -> str:
        validation = KlineTimeValidator.validate_dataframe(df, interval)

        lines = [
            "=" * 60,
            f"K线时间检测报告 - {symbol} {interval}",
            "=" * 60,
        ]

        if validation.get("error"):
            lines.append(f"错误: {validation['error']}")
            return "\n".join(lines)

        lines.extend([
            f"总数据量: {validation['total_count']} 条",
            f"有效数据: {validation['valid_count']} 条",
            f"无效数据: {validation['invalid_count']} 条",
            f"检测结果: {'✓ 通过' if validation['valid'] else '✗ 存在问题'}",
        ])

        if not validation["valid"] and validation.get("first_invalid"):
            first = validation["first_invalid"]
            lines.extend([
                "-" * 60,
                "首个无效时间点:",
                f"  时间: {first['timestamp']}",
                f"  偏差: {first['deviation_minutes']}分 {first['deviation_seconds']}秒",
                f"  期望: {first['expected_time']}",
            ])

        remaining = KlineTimeValidator.get_remaining_seconds(interval)
        next_close = KlineTimeValidator.get_next_close_time(interval)
        lines.extend([
            "-" * 60,
            f"下次K线收盘: {next_close.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"剩余时间: {remaining // 60}分 {remaining % 60}秒",
            "=" * 60,
        ])

        return "\n".join(lines)

    @classmethod
    def get_status_line(cls, interval: str, symbol: str, price: float | None = None) -> str:
        remaining = KlineTimeValidator.get_remaining_seconds(interval)
        next_close = KlineTimeValidator.get_next_close_time(interval)
        config = KlineTimeValidator.INTERVALS.get(interval)

        status = f"[{symbol}] {config.description if config else interval} | "
        if price:
            status += f"价格: {price:.2f} | "
        status += f"下次收盘: {next_close.strftime('%H:%M:%S')} ({remaining // 60}分{remaining % 60}秒后)"

        return status
