from datetime import datetime

import pandas as pd


class urlibs:
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
            data['?']=pd.to_datetime(data['?'], unit='ms',utc=True)
            data['?']=data['?'].dt.tz_localize('Asia/Shanghai')
            data.set_index('?', inplace=True)
        """
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
        if not pd.api.types.is_datetime64_any_dtype(data['close_time']):
            data['close_time'] = pd.to_datetime(data['close_time'], unit='ms', utc=True)
        data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Shanghai')  # 转换为北京时间
        data['close_time'] = data['close_time'].dt.tz_convert('Asia/Shanghai')  # 转换为北京时间
        data.set_index('timestamp', inplace=True)
        return data