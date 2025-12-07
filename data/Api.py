# Updated code with print replaced by logger and extra logging. (User's code preserved except print->logger.* and added logs)

import math
from time import sleep
from typing import List

import talib
from binance import Client
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from dotenv import load_dotenv

from urlibs import *
# import plotly.graph_objects as go

from Config import ApiConfig as Config
from Config import TradeMapper

from data.sqlite_oper import *

load_dotenv()
from app_logger.logger_setup import setup_logger
logger = setup_logger(__name__)

class Api:
    def __init__(self):
        logger.info("初始化 Api 类，正在加载 API 密钥和客户端配置...")
        self.api_key = os.getenv("API_KEY")
        self.api_secret = os.getenv("API_SECRET")
        requests_params = {
            'timeout': Config.CUSTOM_TIMEOUT
        }
        self.local_data_api:SqliteOper | None=None
        self.client = Client(api_key=self.api_key, api_secret=self.api_secret,requests_params=requests_params)

        self.update_local_csv_count=0
        logger.info("Api 初始化完成，准备开始数据操作")

    def _init_local_data(self):
        logger.info("初始化本地数据库对象 SqliteOper ...")
        self.local_data_api=SqliteOper()

    @retry(stop=stop_after_attempt(Config.MAX_RETRY), wait=wait_fixed(Config.WAITING_TIME),
           retry=retry_if_exception_type(Config.RETRY_ERROR_ACCEPT))
    def get_futures_data(self,start_time=None,end_time=None,symbol:str='BTCUSDT',interval:str=Client.KLINE_INTERVAL_30MINUTE,limit:int=ApiConfig.LIMIT) -> pd.DataFrame | None:
        logger.debug(f"获取期货数据 start_time={start_time}, end_time={end_time}, symbol={symbol}, interval={interval}, limit={limit}")
        try:
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
        except Exception as e:
             logger.error(f"Api>get_futures_data获取api数据失败:{e}")
             return None

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
        logger.info(f"启动本地数据主程序 symbol={symbol}")
        path=Path(f"LocalData/{symbol}_base.db")
        if not self.local_data_api:
            self._init_local_data()
        if not path.exists():
            try:
                logger.warning(f"未找到本地数据库，将初始化基础 K 线数据 symbol={symbol}")
                self._init_base_k_line_data(symbol=symbol)
            except Exception as e:
                logger.error(f"Api>>local_data_main>>基础数据集初始化错误:{e}")
        self._update_base_k_line_data(symbol=symbol)

    def _init_base_k_line_data(self,symbol:str):
        logger.info(f"开始初始化基础 K 线数据 symbol={symbol}")
        all_data: List[pd.DataFrame] = []
        get_time_min:int=ApiConfig.LOCAL_MAX_HISTORY_ALLOW * 24 * 60
        end_time:pd.Timestamp=pd.to_datetime('now')
        link_count=0
        while self.update_local_csv_count < get_time_min:
            try:
                data_df=self.get_futures_data(end_time=end_time,interval=Client.KLINE_INTERVAL_1MINUTE,symbol=symbol)
                all_data.insert(0, data_df)
                if not data_df.empty:
                    self.update_local_csv_count +=len(data_df)
                    if self.update_local_csv_count%10000==0:
                        logger.info(f"已获取数据:{self.update_local_csv_count}")
                end_time:pd.Timestamp=pd.to_datetime(data_df['timestamp'].iloc[0], unit='ms') - pd.Timedelta(milliseconds=1)
                if data_df is None or data_df.empty:
                    logger.info("已是最新数据")
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
        self.local_data_api.update_base_kline(symbol=symbol,data=data)
        logger.info("基础 K 线初始化完成并写入数据库")

    def _update_base_k_line_data(self, symbol: str) -> None:
        logger.info(f"开始更新本地 K 线 symbol={symbol}")
        add_data: List[pd.DataFrame] = []

        start_time: pd.Timestamp =pd.to_datetime(self.local_data_api.read_newest_timestamp(symbol=symbol),unit='ms')
        while True:
            data_df:pd.DataFrame = self.get_futures_data(start_time=start_time, symbol=symbol)
            if data_df is None or data_df.empty:
                logger.info(f"{symbol} 数据集已是最新")
                return
            start_time:pd.Timestamp= pd.to_datetime(data_df['timestamp'].iloc[-1], unit='ms')
            self.update_local_csv_count += len(data_df)
            logger.info(f"更新中,数据量:{self.update_local_csv_count}")
            logger.debug(f"最新时间戳:{start_time}")
            add_data.append(data_df)
            if start_time >= pd.to_datetime('now'):
                break
            if len(data_df) < ApiConfig.LIMIT:
                break
            sleep(ApiConfig.API_BASE_GET_INTERVAL)
        data_t=pd.concat(add_data, ignore_index=True)
        data_t=FormatUrlibs.standard_timestamp(data_t)
        self.local_data_api.update_base_kline(symbol=symbol,data=data_t)
        logger.info(f"K 线更新完成 symbol={symbol}")


if __name__ == '__main__':
        api=Api()
        api.local_data_main(symbol='BTCUSDT')