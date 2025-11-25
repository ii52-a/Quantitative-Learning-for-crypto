from dataclasses import dataclass
from binance import Client
from requests import RequestException
from requests.exceptions import SSLError


@dataclass
class ApiConfig:
    """RETRY参数"""
    MAX_RETRY=20
    WAITING_TIME=1.5
    RETRY_ERROR_ACCEPT=(RequestException, ConnectionAbortedError, SSLError)
    CUSTOM_TIMEOUT=(10,30)
    LIMIT = 100  #最大单次请求

    PADDING_COUNT=300   #增加n条数据以平衡指数线偏差,减少前期指标的巨大误差
    CALCULATE_BASECOUNT=100  #计算基础数据量:50条数据为nan删除数据
    GET_COUNT=PADDING_COUNT+CALCULATE_BASECOUNT #请求数据总值



class BackConfig:
    """默认投入参数"""
    SET_LEVERAGE = 5  # 全局默认杠杆
    ORIGIN_USDT = 100  # 默认策略初始usdt

    OPEN_FEE_RADIO = 0.0004  # 手续费占比
    CLOSE_FEE_RADIO = 0.0004

    ROUND_RADIO = 5  # 近似小数位





class TradeConfig:
    ORIGIN_USDT = 100

class TradeMapper:
    K_LINE_TYPE:dict[str,str]={
    # '1s':Client.KLINE_INTERVAL_1SECOND,
    '1min':Client.KLINE_INTERVAL_1MINUTE,
    '3min':Client.KLINE_INTERVAL_3MINUTE,
    '5min':Client.KLINE_INTERVAL_5MINUTE,
    '15min':Client.KLINE_INTERVAL_15MINUTE,
    '30min':Client.KLINE_INTERVAL_30MINUTE,
    '1h':Client.KLINE_INTERVAL_1HOUR,
    '4h':Client.KLINE_INTERVAL_4HOUR,
    '6h':Client.KLINE_INTERVAL_6HOUR,
    '8h':Client.KLINE_INTERVAL_8HOUR,
    '12h':Client.KLINE_INTERVAL_12HOUR,
    '1d':Client.KLINE_INTERVAL_1DAY,
    '3d':Client.KLINE_INTERVAL_3DAY,
    '1w':Client.KLINE_INTERVAL_1WEEK,
    '1m':Client.KLINE_INTERVAL_1MONTH,
}
    K_LINE_TO_MINUTE:dict[str,int]={
        Client.KLINE_INTERVAL_1MINUTE : 1,
        Client.KLINE_INTERVAL_3MINUTE : 3,
        Client.KLINE_INTERVAL_5MINUTE : 5,
        Client.KLINE_INTERVAL_15MINUTE : 15,
        Client.KLINE_INTERVAL_30MINUTE : 30,
        Client.KLINE_INTERVAL_1HOUR : 60,
        Client.KLINE_INTERVAL_4HOUR : 240,
        Client.KLINE_INTERVAL_6HOUR : 360,
        Client.KLINE_INTERVAL_8HOUR : 480,
        Client.KLINE_INTERVAL_12HOUR : 720,
        Client.KLINE_INTERVAL_1DAY : 1440,
        Client.KLINE_INTERVAL_3DAY : 1440 * 3,
        Client.KLINE_INTERVAL_1WEEK : 1440 * 7,
        Client.KLINE_INTERVAL_1MONTH : 1440 * 30,
     }
    trade_pair:str=''

