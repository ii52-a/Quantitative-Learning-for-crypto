import math
from pathlib import Path
from time import sleep
from typing import List


import talib
from binance import Client
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from dotenv import load_dotenv


from urlibs import *
# import plotly.graph_objects as go

from Config import ApiConfig as Config, ApiConfig
from Config import TradeMapper

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

        self.update_local_csv_count=0

    @retry(stop=stop_after_attempt(Config.MAX_RETRY), wait=wait_fixed(Config.WAITING_TIME),
           retry=retry_if_exception_type(Config.RETRY_ERROR_ACCEPT))
    def get_futures_data(self,end_time=None,symbol:str='BTCUSDT',interval:str=Client.KLINE_INTERVAL_30MINUTE,limit:int=ApiConfig.LIMIT) -> pd.DataFrame | None:
        """
            美国标准时间期货数据获取，时间未转换，无macd数据
            @retry 最大尝试,间隔等待,允许的异常类型()
            :param end_time: 需要使用pd.Timestamp格式
            :param symbol: 期货类型
            :param interval: 获取的k线类型
            :param limit: 获取的k线数量
            :return<DataFrame>:含有columns为列的k线信息
        """
        try:
             # 根据配置获得k线
             if end_time:
                 end_ms=int(pd.to_datetime(end_time).timestamp()*1000)
                 klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit,
                                                     endTime=end_ms,timeout=2000)
             else:
                 klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit,timeout=2000)
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
    def get_standard_futures_data(self,end_time=None,symbol='BTCUSDT',interval=Client.KLINE_INTERVAL_30MINUTE,limit=ApiConfig.LIMIT) -> pd.DataFrame:
        """
            标准的集成的合约数据,时间为北京时间，含有macd数据
            @retry 最大尝试5,间隔等待1s,允许的异常类型()
            :param symbol<string>: 期货类型
            :param interval: 获取的k线类型
            :param limit<int>: 获取的k线数量
            :return<DataFrame>:含有columns为列的k线信息,时间为北京时间,含有macd数据
        """
        try:
            data= self.get_futures_data(end_time=end_time,symbol=symbol, interval=interval, limit=limit)
            data = FormatUrlibs.standard_timestamp(data)
            data = self.get_standard_macd(data)
        except Exception as e:
            print(f"Api>get_standard_futures_data获取api数据失败:{e}")
            raise
        return data



    @staticmethod
    def get_standard_macd(data:pd.DataFrame) ->pd.DataFrame:
        """
        macd指标获取
        :param data:k线数据集
        :return 含有macd指标的k线数据集
        """
        macd, macd_signal, macd_hist = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        ##添加指标至data数据
        data['MACD'] = macd
        data['MACD_SIGNAL'] = macd_signal
        data['MACD_HIST'] = macd_hist
        data = data.dropna(subset=['MACD', 'MACD_SIGNAL', 'MACD_HIST'])
        return data



    @retry(stop=stop_after_attempt(Config.MAX_RETRY), wait=wait_fixed(Config.WAITING_TIME),
           retry=retry_if_exception_type(Config.RETRY_ERROR_ACCEPT))
    def get_backtest_data(self,symbol:str, number: int, interval: str, limit: int = ApiConfig.LIMIT) ->pd.DataFrame | None:
        """
            获取历史K线数据
            :param number: 需要的数据总条数
            :param interval: K线周期字符串
            :param limit: 单次请求最大条数
        """
        try:
            # TODO面临废弃[所有k线行为可能通过k线组合进行]
            all_data: List[pd.DataFrame] = []

            # 计算总请求次数 (考虑MACD所需额外数据量)
            total_requests: int = math.ceil((number + Config.GET_COUNT) / limit)

            current_time: pd.Timestamp = pd.to_datetime('now')  # 当前时间

            # 周期对应的分钟数
            kline_interval: int = TradeMapper.K_LINE_TO_MINUTE[interval]
            # print(kline_interval)

            for i in range(total_requests):
                # 计算请求时间段
                start_time: pd.Timestamp = current_time- pd.Timedelta(minutes=limit * kline_interval*(total_requests - i))
                # print(end_time, start_time)

                # 调用API获取数据
                data: pd.DataFrame = self.get_futures_data(symbol=symbol,start_time=start_time,
                                                               interval=interval, limit=limit)
                all_data.append(data)

            # 合并所有数据
            all_data_df: pd.DataFrame = pd.concat(all_data, ignore_index=True)

            # 按时间戳排序 out
            # all_data_df.sort_values(by='timestamp', ascending=True, inplace=True)

            # 计算MACD指标
            all_data_df = self.get_standard_macd(all_data_df)

            # 标准化时间戳并设为索引
            data: pd.DataFrame = FormatUrlibs.standard_timestamp(all_data_df)
            return data
        except Exception as e:
            print(f"API>get_backtest_data>获取回测数据错误:{e}")
            return None



    """最小数据获取程序"""
    @retry(stop=stop_after_attempt(Config.MAX_RETRY), wait=wait_fixed(Config.WAITING_TIME),
           retry=retry_if_exception_type(Config.RETRY_ERROR_ACCEPT))

    def update_local_csv(self,number:int,interval:str,symbol:str,file_path:str)->None:
        """废弃"""
        data=self.get_backtest_data(number=number,interval=interval,symbol=symbol)
        data.to_csv(file_path)


    def update_local_data(self,symbol:str)->None | bool:
        base_path=ApiConfig.LOCAL_DATA_CSV_DIR
        file_path = Path(f"{base_path}/{symbol}_base.csv")
        print("a")
        if not file_path.exists():
            print("<UNK>")
            self._init_base_k_line_data(symbol=symbol)
        return False

    def _init_base_k_line_data(self,symbol:str):
        base_path=ApiConfig.LOCAL_DATA_CSV_DIR
        file_path=Path(f"{base_path}/{symbol}_base.csv")
        setting_path=Path(f"{base_path}/settings.csv")

        #检测文件是否存在不存在就创建
        FileUrlibs.check_local_path(file_path)
        FileUrlibs.check_local_path(setting_path)


        #获取无指标最原始数据
        all_data: List[pd.DataFrame] = []
        #计算最大基础数据获取数量
        get_time_min:int=ApiConfig.LOCAL_MAX_HISTORY_ALLOW * 24 * 60
        end_time:pd.Timestamp=pd.to_datetime('now')
        #用动态循环向前推进数据，避免数据重复
        while self.update_local_csv_count < get_time_min:
            data_df=self.get_futures_data(end_time=end_time,interval=Client.KLINE_INTERVAL_1MINUTE,symbol=symbol)
            all_data.insert(0, data_df)

            #获取过程数量
            self.update_local_csv_count +=len(data_df)
            print(f"已获取数据:{self.update_local_csv_count}")
            end_time:pd.Timestamp=pd.to_datetime(data_df['timestamp'].iloc[0], unit='ms') - pd.Timedelta(milliseconds=1)
            if data_df is None or data_df.empty:
                break

            sleep(ApiConfig.API_BASE_GET_INTERVAL)
        data:pd.DataFrame=pd.concat(all_data, ignore_index=True)
        # data.sort_values(by='timestamp', ascending=True, inplace=True)

        data=FormatUrlibs.standard_timestamp(data)
        data.to_csv(file_path)

    # def get_base_csv_num(self)->int:
    #     return self.update_local_csv_count












if __name__ == '__main__':
    #TODO: 改为base数据更新
    try:
        api=Api()
        api.update_local_data("BTCUSDT")


    except Exception as e:
        print(f"<UNK>:{e}")