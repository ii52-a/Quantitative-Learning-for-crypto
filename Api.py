import os
from typing import List

import talib
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from dotenv import load_dotenv

from urlibs import *
# import plotly.graph_objects as go

from Config import *

load_dotenv()


class Api:
    def __init__(self):
        # 获得账户信息
        self.api_key = os.getenv("API_KEY")
        self.api_secret = os.getenv("API_SECRET")
        requests_params = {
            'timeout': Config.CUSTOM_TIMEOUT
        }
        self.client = Client(api_key=self.api_key, api_secret=self.api_secret,requests_params=requests_params)
        self.symbol = 'BTCUSDT'
        self.limit=100

    @retry(stop=stop_after_attempt(Config.MAX_RETRY), wait=wait_fixed(Config.WAITING_TIME),
           retry=retry_if_exception_type(Config.RETRY_ERROR_ACCEPT))
    def get_futures_data(self,start_time=None,end_time=None,symbol='BTCUSDT',interval=Client.KLINE_INTERVAL_30MINUTE,limit=100) -> pd.DataFrame | None:
        """
            美国标准时间期货数据获取，时间未转换，无macd数据
            @retry 最大尝试5,间隔等待1s,允许的异常类型()
            :symbol: 期货类型
            :interval: 获取的k线类型
            :limit: 获取的k线数量
            :return<DataFrame>:含有columns为列的k线信息
        """
        try:
             # 根据配置获得k线
             if start_time and end_time:
                 start_ms=int(pd.to_datetime(start_time).timestamp()*1000)
                 end_ms=int(pd.to_datetime(end_time).timestamp()*1000)
                 klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit,
                                                     startTime=start_ms, endTime=end_ms,timeout=2000)
             else:
                 klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
             data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                  'quote_asset_volume', 'number_of_trades',
                                                  'taker_buy_base_asset_volume',
                                                  'taker_buy_quote_asset_volume', 'ignore'])
             return data
        except Exception as e:
             print(f"Api>get_futures_data获取api数据失败:{e}")
             return None

    @retry(stop=stop_after_attempt(Config.MAX_RETRY), wait=wait_fixed(Config.WAITING_TIME),
           retry=retry_if_exception_type(Config.RETRY_ERROR_ACCEPT))
    def get_standard_futures_data(self,start_time=None,end_time=None,symbol='BTCUSDT',interval=Client.KLINE_INTERVAL_30MINUTE,limit=100) -> pd.DataFrame:
        """
            标准的集成的合约数据,时间为北京时间，含有macd数据
            @retry 最大尝试5,间隔等待1s,允许的异常类型()
            :symbol<string>: 期货类型
            :interval: 获取的k线类型
            :limit<int>: 获取的k线数量
            :return<DataFrame>:含有columns为列的k线信息,时间为北京时间,含有macd数据
        """
        try:
            data= self.get_futures_data(start_time=start_time,end_time=end_time,symbol=symbol, interval=interval, limit=limit)
            data = urlibs.standard_timestamp(data)
            data = self.get_standard_macd(data)
        except Exception as e:
            print(f"Api>get_standard_futures_data获取api数据失败:{e}")
            raise
        return data


    """
    macd指标获取
    :param data:k线数据集
    :return 含有macd指标的k线数据集
    """


    @staticmethod
    def get_standard_macd(data:pd.DataFrame) ->pd.DataFrame:
        macd, macd_signal, macd_hist = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        ##添加指标至data数据
        data['MACD'] = macd
        data['MACD_SIGNAL'] = macd_signal
        data['MACD_HIST'] = macd_hist
        data = data.dropna(subset=['MACD', 'MACD_SIGNAL', 'MACD_HIST'])
        return data

    @retry(stop=stop_after_attempt(Config.MAX_RETRY), wait=wait_fixed(Config.WAITING_TIME),
           retry=retry_if_exception_type(Config.RETRY_ERROR_ACCEPT))
    def get_backtest_data(self, number: int, interval: str = Client.KLINE_INTERVAL_30MINUTE, limit: int = 100) ->pd.DataFrame | None:
        """
            获取历史K线数据
            :param number: 需要的数据总条数
            :param interval: K线周期字符串
            :param limit: 单次请求最大条数
        """
        try:
            all_data: List[pd.DataFrame] = []

            # 计算总请求次数 (考虑MACD所需额外数据量)
            total_requests: int = ((number + Config.MACD_GET_COUNT) // limit) + (
                1 if (number + Config.MACD_GET_COUNT) % limit != 0 else 0)

            current_time: pd.Timestamp = pd.to_datetime('now')  # 当前时间

            # 周期对应的分钟数
            kline_interval: int = Config.ST_TIME_TRANSFORM[interval]

            for i in range(total_requests):
                # 计算请求时间段 (倒推)
                end_time: pd.Timestamp = current_time - pd.Timedelta(minutes=i * limit * kline_interval)
                start_time: pd.Timestamp = end_time - pd.Timedelta(minutes=limit * kline_interval)
                # print(end_time, start_time)

                # 调用API获取数据
                data: pd.DataFrame = self.get_futures_data(start_time=start_time, end_time=end_time,
                                                               interval=interval, limit=limit)
                all_data.append(data)

            # 合并所有数据
            all_data_df: pd.DataFrame = pd.concat(all_data, ignore_index=True)
            # 按时间戳排序
            all_data_df.sort_values(by='timestamp', ascending=True, inplace=True)

            # 计算MACD指标
            all_data_df = self.get_standard_macd(all_data_df)

            # 标准化时间戳并设为索引
            data: pd.DataFrame = urlibs.standard_timestamp(all_data_df)
            return data
        except Exception as e:
            print(f"API>BackTest>get_backtest_data>获取回测数据错误:{e}")
            return None

    def get_csv_data(self, number:int, file_path:str='data.csv') -> pd.DataFrame | None:
        number += Config.MACD_GET_COUNT
        if os.path.exists(file_path):
            try:
                #csv
                pd_f = pd.read_csv(
                    file_path,
                    index_col='timestamp',
                    parse_dates=True  # 尝试将索引解析为日期时间类型
                ).iloc[-number:]
                print(f"使用本地数据,本地数据最新时间:{pd_f.index[-1]}")
                return pd_f
            except Exception as e:
                print(f"Api>get_csv_data获取本地数据失败:{e},切换至api实时查询")
                return self.get_backtest_data(number=number)
        return None

    @retry(stop=stop_after_attempt(Config.MAX_RETRY), wait=wait_fixed(Config.WAITING_TIME),
           retry=retry_if_exception_type(Config.RETRY_ERROR_ACCEPT))
    def update_local_csv(self,number:int,file_path='data.csv')->None:
        data=self.get_backtest_data(number=number)
        data.to_csv(file_path)
        print(f"更新成功,数据量:{number},包含预留实际数据量:{len(data.index)},最新时间:{data.index[-1]}")

if __name__ == '__main__':
    api=Api()
    api.update_local_csv(number=int(input('选择更新数据量:')))