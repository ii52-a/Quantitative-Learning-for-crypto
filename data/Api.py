import math
from time import sleep
from typing import List


import talib
from binance import Client
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from dotenv import load_dotenv

from urlibs import *
# import plotly.graph_objects as go

from Config import TradeMapper,ApiConfig as Config

from data.sqlite_oper import *
from kline_process import KlineProcess

from app_logger.logger_setup import setup_logger
logger = setup_logger(__name__)
load_dotenv()
class Api:
    def __init__(self):
        logger.debug("初始化 Api 类，正在加载 API 密钥和客户端配置...")
        self.api_key = os.getenv("API_KEY")
        self.api_secret = os.getenv("API_SECRET")
        requests_params = {
            'timeout': Config.CUSTOM_TIMEOUT
        }
        self.local_data_base:SqliteBase=SqliteBase()
        self.local_data_oper:SqliteOper=SqliteOper()
        self.client = Client(api_key=self.api_key, api_secret=self.api_secret,requests_params=requests_params)

        self.update_local_count=0
        logger.debug("Api 初始化完成，开始数据操作")


    @retry(stop=stop_after_attempt(Config.MAX_RETRY), wait=wait_fixed(Config.WAITING_TIME),
           retry=retry_if_exception_type(Config.RETRY_ERROR_ACCEPT))
    @catch_and_log(logger=logger)
    def get_futures_data(self,start_time=None,end_time=None,symbol:str='BTCUSDT',interval:str=Client.KLINE_INTERVAL_30MINUTE,limit:int=ApiConfig.LIMIT) -> pd.DataFrame | None:
         logger.debug(f"获取期货数据 start_time={start_time}, end_time={end_time}, symbol={symbol}, interval={interval}, limit={limit}")
         if start_time and not end_time:
             start_ms = int(pd.to_datetime(start_time).timestamp()*1000)
             klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit,
                                                 startTime=start_ms, timeout=2000)
         elif end_time and not start_time:
             end_ms=int(pd.to_datetime(end_time).timestamp()*1000)
             klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit,
                                                 endTime=end_ms,timeout=2000)
         else:
             start_ms = int(pd.to_datetime(start_time).timestamp() * 1000)
             end_ms = int(pd.to_datetime(end_time).timestamp() * 1000)
             klines = self.client.futures_klines(startTime=start_ms,endTime=end_ms,symbol=symbol, interval=interval, limit=limit,timeout=2000)
         data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                              'quote_asset_volume', 'number_of_trades',
                                              'taker_buy_base_asset_volume',
                                              'taker_buy_quote_asset_volume','ignore'])
         return data


    @retry(stop=stop_after_attempt(Config.MAX_RETRY), wait=wait_fixed(Config.WAITING_TIME),
           retry=retry_if_exception_type(Config.RETRY_ERROR_ACCEPT))
    def get_standard_futures_data(self,start_time=None,end_time=None,symbol='BTCUSDT',interval=Client.KLINE_INTERVAL_30MINUTE,limit=ApiConfig.LIMIT) -> pd.DataFrame:
        logger.info("开始获取标准合约数据（含 MACD + 北京时间）...")
        try:
            data= self.get_futures_data(start_time=start_time,end_time=end_time,symbol=symbol, interval=interval, limit=limit)
            data = FormatUrlibs.standard_timestamp(data)
            data = self.get_standard_macd(data)
        except Exception as e:
            logger.error(f"Api>get_standard_futures_data获取api数据失败:{e}")
            raise
        logger.info("标准合约数据处理完成")
        return data

    @staticmethod
    def get_standard_macd(data:pd.DataFrame) ->pd.DataFrame:
        logger.debug("计算 MACD 指标中...")
        macd, macd_signal, macd_hist = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['MACD'] = macd
        data['MACD_SIGNAL'] = macd_signal
        data['MACD_HIST'] = macd_hist
        data = data.dropna(subset=['MACD', 'MACD_SIGNAL', 'MACD_HIST'])
        logger.debug("MACD 计算完成")
        return data

    @retry(stop=stop_after_attempt(Config.MAX_RETRY), wait=wait_fixed(Config.WAITING_TIME),
           retry=retry_if_exception_type(Config.RETRY_ERROR_ACCEPT))
    def get_backtest_data(self,symbol:str, number: int, interval: str, limit: int = ApiConfig.LIMIT) ->pd.DataFrame | None:
        logger.info(f"开始获取回测数据 symbol={symbol}, number={number}, interval={interval}")
        try:
            all_data: List[pd.DataFrame] = []
            total_requests: int = math.ceil((number + Config.GET_COUNT) / limit)

            current_time: pd.Timestamp = pd.to_datetime('now')
            kline_interval: int = TradeMapper.K_LINE_TO_MINUTE[interval]

            for i in range(total_requests):
                start_time: pd.Timestamp = current_time- pd.Timedelta(minutes=limit * kline_interval*(total_requests - i))
                data: pd.DataFrame = self.get_futures_data(symbol=symbol,start_time=start_time,
                                                               interval=interval, limit=limit)
                all_data.append(data)
            all_data_df: pd.DataFrame = pd.concat(all_data, ignore_index=True)
            all_data_df = self.get_standard_macd(all_data_df)
            data: pd.DataFrame = FormatUrlibs.standard_timestamp(all_data_df)
            logger.info(f"回测数据获取完成，总条数={len(data)}")
            return data
        except Exception as e:
            logger.error(f"API>get_backtest_data>获取回测数据错误:{e}")
            return None

    def local_data_main(self,symbol):
        logger.info(f" [1]<Api-main>:启动本地数据主程序 symbol={symbol}")
        path=Path(f"LocalData/{symbol}_base.db")
        if not path.exists():

            logger.info(f"[1,1]<Api-main>:未找到本地数据库，将初始化基础 K 线数据 symbol={symbol}")
            self._init_base_k_line_data(symbol=symbol)
            logger.info(f"[1,2]<Api-main>:准备聚合常用k线数据: symbol={symbol}")
            KlineProcess.main(symbol=symbol)

        self._update_base_k_line_data(symbol=symbol)

    def _init_base_k_line_data(self,symbol:str) -> None:
        logger.debug(f"开始初始化基础 K 线数据 symbol={symbol}")
        all_data: List[pd.DataFrame] = []
        get_time_min:int=ApiConfig.LOCAL_MAX_HISTORY_ALLOW * 24 * 60
        end_time:pd.Timestamp=pd.to_datetime('now')
        link_count=0
        while self.update_local_count < get_time_min:
            try:
                if end_time is None:
                    logger.warning(f"{symbol}获取end_time为None!")
                    return

                data_df:pd.DataFrame=self.get_futures_data(end_time=end_time,interval=Client.KLINE_INTERVAL_1MINUTE,symbol=symbol)
                all_data.insert(0, data_df)
                ceil_percent:int=math.ceil(self.update_local_count / (get_time_min*0.1))
                percent:float=round(self.update_local_count*100 / get_time_min,2)

                if not data_df.empty:
                    self.update_local_count +=len(data_df)
                    if self.update_local_count%10000==0:
                        logger.info(f"已获得数据:{self.update_local_count}  [{'='*ceil_percent}{' '*(10-ceil_percent)}]{percent}%")

                end_time:pd.Timestamp=pd.to_datetime(data_df['timestamp'].iloc[0], unit='ms') - pd.Timedelta(milliseconds=1)
                if data_df is None or data_df.empty:
                    logger.debug(f"{symbol}已达最新数据,总添加数据量:{len(all_data)}")
                    break
                sleep(ApiConfig.API_BASE_GET_INTERVAL)

            except Exception as e:
                logger.error(f"API>>_init_base_k_line_data>>基础数据初始化失败:{e}")
                logger.info("等待3秒后重试")
                sleep(3)
                link_count+=1
                if link_count>=5:
                    raise Exception("API>>_init_base_k_line_data>>链接次数过多，请寻找错误或调高时间间隔")
                continue
        data:pd.DataFrame=pd.concat(all_data, ignore_index=True)
        data=FormatUrlibs.standard_timestamp(data)
        logger.info(f"获取未保存数据:{len(data)}")
        self.local_data_base.update_base_kline(symbol=symbol,data=data)
        logger.debug("基础 K 线初始化完成并写入数据库")

    def _update_base_k_line_data(self, symbol: str) -> None:
        logger.debug(f"开始更新本地 K 线 symbol={symbol}")
        add_data: List[pd.DataFrame] = []
        path=Path(f"LocalData/{symbol}_base.db")
        start_time: pd.Timestamp =pd.to_datetime(self.local_data_oper.read_newest_timestamp(path),unit='ms')
        while True:
            if start_time is None:
                logger.warning(f"请求初始时间start_time={start_time}")
            data_df:pd.DataFrame = self.get_futures_data(start_time=start_time,interval=Client.KLINE_INTERVAL_1MINUTE, symbol=symbol)
            if data_df is None or data_df.empty:
                logger.info(f"{symbol} 数据集已是最新")
                return
            start_time:pd.Timestamp= pd.to_datetime(data_df['timestamp'].iloc[-1], unit='ms')
            self.update_local_count += len(data_df)
            logger.info(f"更新中,数据量:{self.update_local_count}")
            logger.debug(f"最新时间戳:{start_time}")
            add_data.append(data_df)
            if start_time >= pd.to_datetime('now'):
                break
            if len(data_df) < ApiConfig.LIMIT:
                logger.debug(f"检测到结束数据标识:{len(data_df)}")
                break
            sleep(ApiConfig.API_BASE_GET_INTERVAL)
        data_t=pd.concat(add_data, ignore_index=True)
        data_t=FormatUrlibs.standard_timestamp(data_t)
        self.local_data_base.update_base_kline(symbol=symbol,data=data_t)

        logger.debug(f"K 线更新完成 symbol={symbol}")


if __name__ == '__main__':
        api=Api()
        api.local_data_main(symbol='ETHUSDT')