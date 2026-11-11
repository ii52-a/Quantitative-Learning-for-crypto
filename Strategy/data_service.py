from typing import List

import numpy as np
import pandas as pd

from Config import TradeMapper
from Data.sqlite_oper import SqliteOper
from app_logger.logger_setup import Logger

logger = Logger(__name__)


class Dataservice:
    """策略侧数据入口：读本地 SQLite + 计算 MACD（纯 pandas，无 TA-Lib 依赖）。"""

    data_list: List[str] = TradeMapper.data_time
    # 指标预热额外多取的 K 线数量
    additional_data = 300

    # ---------- 数据 ----------
    @classmethod
    def _get_data_time(cls, symbol: str, kline: str, start: pd.Timestamp | None, length: int) -> pd.DataFrame | None:
        logger.info(f"begin get {symbol} {kline}")
        return SqliteOper.with_open_and_func(
            symbol=symbol,
            kline_str=kline,
            func=SqliteOper.read_range_kline,
            kline=kline,
            start_time=start,
            number=length,
        )

    @classmethod
    def get_data(cls, symbol: str, time: str, length: int, start_time: pd.Timestamp | None = None) -> pd.DataFrame:
        if time not in cls.data_list:
            raise ValueError(f"{time} not in {cls.data_list}")
        length = int(length)
        if length <= 0:
            raise ValueError(f"length 必须为正整数，收到 {length}")

        kline = f"kline_{time}"
        data_df = cls._get_data_time(symbol=symbol, kline=kline, start=start_time,
                                     length=length + cls.additional_data)
        if data_df is None or data_df.empty:
            raise ValueError(f"{symbol} {kline} 本地数据为空，请先初始化本地数据库")
        # 统一保证升序，指标计算依赖时间顺序
        if not data_df.index.is_monotonic_increasing:
            data_df = data_df.sort_index()
        return data_df

    # ---------- 指标 ----------
    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        """与 TA-Lib 一致的 EMA：前 period 个值取 SMA 作种子，之后递推。"""
        values = series.to_numpy(dtype=float)
        out = np.full(values.shape, np.nan, dtype=float)
        if values.size < period:
            return pd.Series(out, index=series.index)
        out[period - 1] = values[:period].mean()
        alpha = 2.0 / (period + 1)
        for i in range(period, values.size):
            out[i] = (values[i] - out[i - 1]) * alpha + out[i - 1]
        return pd.Series(out, index=series.index)

    @classmethod
    def macd(cls, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """返回 (DIF, DEA, HIST)，口径与 talib.MACD 一致。"""
        ema_fast = cls._ema(close, fast)
        ema_slow = cls._ema(close, slow)
        macd_line = ema_fast - ema_slow

        signal_line = pd.Series(np.nan, index=macd_line.index, dtype=float)
        first_valid = macd_line.first_valid_index()
        if first_valid is not None:
            pos = macd_line.index.get_loc(first_valid)
            signal_line.iloc[pos:] = cls._ema(macd_line.iloc[pos:], signal).to_numpy()

        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @classmethod
    def macd_data(cls, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        if data is None or data.empty:
            raise ValueError("macd_data 收到空数据")
        logger.debug("计算 MACD 指标中...")
        data = data.copy()
        macd, macd_signal, macd_hist = cls.macd(data['close'], fast=fast, slow=slow, signal=signal)
        data['MACD'] = macd
        data['MACD_SIGNAL'] = macd_signal
        data['MACD_HIST'] = macd_hist
        data = data.dropna(subset=['MACD', 'MACD_SIGNAL', 'MACD_HIST'])
        if data.empty:
            raise ValueError("MACD 有效数据为空，检查本地历史长度是否足够")
        logger.debug("MACD 计算完成")
        return data


if __name__ == '__main__':
    start = pd.Timestamp("2025-07-30", tz="UTC")
    data = Dataservice.get_data("ETHUSDT", "30min", length=1000, start_time=start)
    print(data.index[:3], data.index[-3:])
    print(Dataservice.macd_data(data).tail())
