from dataclasses import dataclass


import talib
from binance.client import Client
from requests import RequestException
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from urlibs import *
# import plotly.graph_objects as go

@dataclass
class Config:
    ST_TIME_TRANSFORM={Client.KLINE_INTERVAL_30MINUTE:30}    #bian枚举对应值
    SET_LEVERAGE=5   #全局默认杠杆
    ORIGIN_USDT=100   #默认策略初始usdt


class Position:
    LONG=1
    SHORT=-1

@dataclass
class StrategyResult:
    signal: Position.LONG or Position.SHORT or None
    size: float        # 开仓权重(%资金)
    leverage: int      # 杠杆调整
    comment: str = ""  # 备注，用于调试


class Api:
    def __init__(self):
        # 获得账户信息
        self.api_key = 'nwiIxe5M8nnNspyJSrwI61WHyIq173KB7sHe8hId9tGehioPwznWwvCFIw4vWCyK'
        self.api_secret = 'f1bToOm4j4kOqip5oTnb6tBYc1h8VJM0yBddnNo8uwyfO0QXBhMKUV4rbK79VYVv'
        self.client = Client(api_key=self.api_key, api_secret=self.api_secret)
        self.symbol = 'BTCUSDT'
        self.limit=100

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type((RequestException, ConnectionAbortedError)))
    def get_futures_data(self,start_time=None,end_time=None,symbol='BTCUSDT',interval=Client.KLINE_INTERVAL_30MINUTE,limit=100) -> pd.DataFrame:
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
                                                     startTime=start_ms, endTime=end_ms)
             else:
                 klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
             data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                  'quote_asset_volume', 'number_of_trades',
                                                  'taker_buy_base_asset_volume',
                                                  'taker_buy_quote_asset_volume', 'ignore'])
             return data
        except Exception as e:
             print(e)
             raise




    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1),
           retry=retry_if_exception_type((RequestException, ConnectionAbortedError)))
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




class Strategy:

    @staticmethod
    def strategy_macd30min(setting:dict)-> StrategyResult:
        data= setting.get('data')
        leverage = setting.get('leverage')
        if 'MACD_HIST' not in data:
            raise Exception('缺少macd数据')
        cur_macd = data['MACD_HIST']
        if cur_macd.iloc[-1] * cur_macd.iloc[-2] < 0:  # 趋势反转捕捉信号
            # TODO:过滤假信号
            pass

            return StrategyResult(
                signal=Position.LONG if cur_macd.iloc[-1]>0 else Position.SHORT,
                size=1,
                leverage=leverage,
            )
        return StrategyResult(
            signal=None,
            size=0,
            leverage=leverage,

        )


class BackTest:
    def __init__(self,api:Api):
        self.api = api
        self.data:pd.DataFrame= api.get_standard_futures_data()
        self.symbol:str=api.symbol
        self.usdt=Config.ORIGIN_USDT
        self.occupy_usdt=0
        """
        :position_now<dict>: {
        symbol<str>,
        position<position>,
        open_position<dict>:{'usdt':,'-symbol':},
        open_price<float>,leverage<int>
        }
        """
        #动态仓位
        self.position_now:dict={'symbol':self.symbol,'position':None,'size':0,'open_price':None,'leverage':None}
        self.position_open_history:pd.DataFrame=pd.DataFrame(columns=['symbol','open_price','size','position','leverage','open_time'])   #仓位改变记录
        self.position_history:pd.DataFrame=pd.DataFrame(columns=['symbol', 'position', 'open_time', 'close_time', 'open_price', 'close_price', 'size', 'leverage', 'pnl'])

    def open_position(self, value, price,open_time,leverage=Config.SET_LEVERAGE):
        e_size = value / price * Config.SET_LEVERAGE
        #首次开仓修改open_time
        if self.position_now['position'] == 0 and e_size !=0:
            self.position_now['open_time'] = open_time

        self.position_now = {
            "symbol": self.symbol,
            "open_price": price,
            "size": self.position_now['size']-e_size,
            "position": Position.LONG if self.position_now['size'] > 0 else Position.SHORT,
            "leverage": leverage,
            "open_time": self.position_now['open_time'],
        }
        self.position_open_history.append({
            "symbol": self.symbol,
            "open_price": price,
            "size": e_size,
            "position": Position.LONG if e_size > 0 else Position.SHORT,
            "leverage": leverage,
            "open_time": open_time,
        })
        print(f"[开仓]: {self.position_now['symbol']}[{leverage}x] @ {price} size={self.position_now['size']:.4f}{self.symbol.replace('USDT','')}")

    def close_position(self, price):
        pos = self.position_now
        if pos is None:
            return

        pnl = pos["size"] * (price - pos["open_price"]) * pos["dir"]
        print(f"[平仓] pnl={pnl}")

        pos["close_price"] = price
        pos["pnl"] = pnl
        self.position_history.append(pos)

        self.position_now = {'symbol':self.symbol,'position':None,'size':0,'open_price':None,'leverage':None}

    def strategy_backtest_loop(self,symbol,leverage=Config.SET_LEVERAGE):
        setting={'symbol':symbol,'data':None,'leverage':leverage,'size':0}
        for i in range(len(self.data)):
            slice_data=self.data.iloc[:i]
            setting['data']=slice_data
            signals:StrategyResult=Strategy.strategy_macd30min(setting)
            if signals.signal!=Position.LONG:
                self.open_position(self.usdt*signals.size,slice_data['open_price'],slice_data['open_time'],leverage)



    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1),
           retry=retry_if_exception_type((RequestException, ConnectionAbortedError)))
    def get_backtest_data(self, number: int, interval=Client.KLINE_INTERVAL_30MINUTE, limit=100) -> None:
        """
            获取用于回测的大量历史数据
            :param number: 需要的数据条数
            :param interval: K线周期
            :param limit: 每次请求的最大数据条数（最多 100 条）
            :return: pd.DataFrame，包含历史数据的 DataFrame
        """
        try:
            all_data = []
            # 计算请求次数
            total_requests = (number // limit) + (1 if number % limit != 0 else 0)


            current_time = pd.to_datetime('now')  # 当前时间

            kline_interval=Config.ST_TIME_TRANSFORM[interval]  #对应分钟时间间隔
            for i in range(total_requests):
                # 计算每次请求的起始时间和结束时间
                end_time = current_time - pd.Timedelta(minutes=i * limit * kline_interval)
                start_time = end_time - pd.Timedelta(minutes=limit * kline_interval)
                # print(end_time, start_time)



                data = self.api.get_futures_data(start_time=start_time, end_time=end_time, interval=interval, limit=limit)
                all_data.append(data)

            # 合并所有获取的数据
            all_data_df = pd.concat(all_data,ignore_index=True)
            #排序
            all_data_df.sort_index(inplace=False)
            #macd
            all_data_df =self.api.get_standard_macd(all_data_df)




            self.data = urlibs.standard_timestamp(all_data_df)
        except Exception as e:
            print(e)
            raise

def main():
    try:
        bt = BackTest(Api())
        while True: #TODO
            try:
                print("1:")
                chose=int(input("回测数据量"))
                bt.get_backtest_data(number=chose)
                print(bt.data)
            except Exception as e:
                print(e)
                continue
    except Exception as e:
        print(e)
        main()

if __name__ == '__main__':
    main()
