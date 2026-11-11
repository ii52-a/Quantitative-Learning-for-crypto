import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from Config import ApiConfig

logger = logging.getLogger(__name__)


class ProgressBar:
    """tqdm 缺失时的降级实现：只保证 n/refresh/上下文管理器接口可用。"""

    def __init__(self, total: int | None = None, desc: str = "", unit: str = "", **_kwargs):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.n = 0
        self._last_report = 0.0
        self._last_pct = -1

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def update(self, n: int = 1) -> None:
        self.n += n
        self.refresh()

    def refresh(self) -> None:
        if not self.total:
            return
        pct = int(self.n * 100 / self.total)
        now = time.monotonic()
        if pct // 10 > self._last_pct // 10 or now - self._last_report > 5:
            self._last_pct = pct
            self._last_report = now
            logger.info("%s %s/%s (%s%%)", self.desc, self.n, self.total, pct)

    def close(self) -> None:
        return None


def get_progress(total: int | None = None, desc: str = "", unit: str = ""):
    """优先使用 tqdm；未安装时退化为按 10% 打日志的进度条。"""
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit=unit)
    except ImportError:
        return ProgressBar(total=total, desc=desc, unit=unit)


class FormatUrlibs:

    """内容格式化区"""
    @staticmethod
    def get_bianstr_time(value) -> str:
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.strftime('%Y-%m-%d %H:%M:%S')

        # 若是字符串：必须是真的时间字符串
        if isinstance(value, str):
            try:
                value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                return value.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                raise ValueError(f"传入的字符串不是合法时间：{value}")

        raise TypeError(f"无法处理的时间类型：{type(value)}")

    @staticmethod
    def standard_timestamp(data: pd.DataFrame) -> pd.DataFrame:
        """
            标准时区转换,转换为北京时区，并设置时间为表头
            Data['?']=pd.to_datetime(Data['?'], unit='ms',utc=True)
            Data['?']=Data['?'].dt.tz_convert('Asia/Shanghai')
            Data.set_index('?', inplace=True)
        """
        if data is None:
            raise ValueError("standard_timestamp 收到 None")
        if data.empty and 'timestamp' not in data.columns:
            return data
        if data.index.name == 'timestamp':
            data = data.reset_index()
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
        if not pd.api.types.is_datetime64_any_dtype(data['close_time']):
            data['close_time'] = pd.to_datetime(data['close_time'], unit='ms', utc=True)
        data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Shanghai')  # 转换为北京时间
        data['close_time'] = data['close_time'].dt.tz_convert('Asia/Shanghai')  # 转换为北京时间
        data = data.set_index('timestamp')
        return data


class FileUrlibs:
    """文件调用区"""

    @staticmethod
    def check_local_path(path) -> Path:
        """接受 str 或 Path，返回 Path。"""
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
        return path


    #获取本地csv数据
    @staticmethod
    def get_csv_data( file_path:str | Path,number:int| None=None) -> pd.DataFrame | None:

        if os.path.exists(file_path):
            try:
                #csv
                pd_f = pd.read_csv(
                    file_path,
                    index_col='timestamp',
                    parse_dates=True  # 尝试将索引解析为日期时间类型
                )
                if number is not None:
                    number += ApiConfig.GET_COUNT
                    pd_f = pd_f.iloc[-number:]
                return pd_f
            except Exception as e:
                raise e
        return None
