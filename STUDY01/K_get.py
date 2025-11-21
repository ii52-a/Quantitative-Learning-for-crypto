
import talib
from binance.client import Client
import pandas as pd
from requests import RequestException
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type


class Api:
    def __init__(self):
        # 获得账户信息
        self.api_key = 'nwiIxe5M8nnNspyJSrwI61WHyIq173KB7sHe8hId9tGehioPwznWwvCFIw4vWCyK'
        self.api_secret = 'f1bToOm4j4kOqip5oTnb6tBYc1h8VJM0yBddnNo8uwyfO0QXBhMKUV4rbK79VYVv'
        self.client = Client(api_key=self.api_key, api_secret=self.api_secret)
        self.symbol = 'BTCUSDT'

    """
    美国标准时间期货数据获取，时间未转换，无macd数据
    @retry 最大尝试5,间隔等待1s,允许的异常类型()
    :symbol: 期货类型
    :interval: 获取的k线类型
    :limit: 获取的k线数量
    :return<DataFrame>:含有columns为列的k线信息
    """
    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type((RequestException, ConnectionAbortedError)))
    def get_futures_data(self, symbol='BTCUSDT',interval=Client.KLINE_INTERVAL_30MINUTE,limit=100) -> pd.DataFrame:
        try:
             # 根据配置获得k线
             klines:dict=self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
             data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                             'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                             'taker_buy_quote_asset_volume', 'ignore'])
             return data
        except Exception as e:
             print(e)
             raise


    """
    标准的集成的合约数据,时间为北京时间，含有macd数据
    @retry 最大尝试5,间隔等待1s,允许的异常类型()
    :symbol<string>: 期货类型
    :interval: 获取的k线类型
    :limit<int>: 获取的k线数量
    :return<DataFrame>:含有columns为列的k线信息,时间为北京时间,含有macd数据
    """

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1),
           retry=retry_if_exception_type((RequestException, ConnectionAbortedError)))
    def get_standard_futures_data(self, symbol='BTCUSDT',interval=Client.KLINE_INTERVAL_30MINUTE,limit=100) -> pd.DataFrame:
        try:
            data=self.get_futures_data(symbol=symbol, interval=interval, limit=limit)
            data = self.standard_timestamp(data)
            data = self.get_standard_macd(data)
        except Exception as e:
            print(e)
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
        return data



    """
    标准时区转换,转换为北京时区，并设置时间为表头
    data['?']=pd.to_datetime(data['?'], unit='ms',utc=True)
    data['?']=data['?'].dt.tz_localize('Asia/Shanghai')
    data.set_index('?', inplace=True)
    """
    @staticmethod
    def standard_timestamp(data:pd.DataFrame) -> pd.DataFrame:
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
        data['close_time'] = pd.to_datetime(data['close_time'], unit='ms', utc=True)
        data['timestamp'] = data['timestamp'].dt.tz_convert('Asia/Shanghai')  # 转换为北京时间
        data['close_time'] = data['close_time'].dt.tz_convert('Asia/Shanghai')  # 转换为北京时间
        data.set_index('timestamp', inplace=True)
        return data

class CtaTrading:
    def __init__(self,api:Api):
        self.api = api
        self.data:pd.DataFrame= api.get_standard_futures_data()
        self.symbol:str=api.symbol

    def strategy_trading(self,data):
        cur_macd = data['MACD_HIST']
        if cur_macd.iloc[-1] * cur_macd.iloc[-2] < 0:  # 趋势反转捕捉信号
            # TODO:过滤假信号
            pass
            cur_price = data['open'].iloc[-1]
            if cur_macd.iloc[-1] > 0:

                print('='*40)
                print(f"币种:{self.symbol}\t\t方向:做多\t\t杠杆:5x")
                print(f"开仓数量:0.01btc\t\t开仓价格:{cur_price}")  #TODO创建开仓数量与杠杆
                print('=' *40)
            else:
                print('=' * 40)
                print(f"币种:{self.symbol}\t\t方向:做多\t\t杠杆:5x")
                print(f"开仓数量:0.01btc\t\t开仓价格:{cur_price}")  # TODO创建开仓数量与杠杆
                print('=' * 40)
        else:
            print(f"当前macd_hist:{data['MACD_HIST'].iloc[-1]}")


if __name__ == '__main__':
    cta=CtaTrading(Api())
