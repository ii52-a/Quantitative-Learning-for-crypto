from dataclasses import dataclass

from binance import Client


from requests import RequestException
from requests.exceptions import SSLError


@dataclass
class Config:
    """RETRY参数"""
    MAX_RETRY=20
    WAITING_TIME=1.5
    RETRY_ERROR_ACCEPT=(RequestException, ConnectionAbortedError, SSLError)
    CUSTOM_TIMEOUT=(10,30)
    LIMIT = 100  #最大单次请求
    """数据误差值去除"""
    #macd策略
    MACD_PADDING_COUNT=300   #增加n条数据以平衡macd指数线偏差,减少前期macd指标的巨大误差
    MACD_CALCULATE_BASECOUNT=50  #macd计算基础数据量:50条数据为nan删除数据
    MACD_GET_COUNT=MACD_PADDING_COUNT+MACD_CALCULATE_BASECOUNT #请求数据总值


    """默认投入参数"""
    ST_TIME_TRANSFORM={Client.KLINE_INTERVAL_30MINUTE:30}    #bian枚举对应值
    SET_LEVERAGE=5   #全局默认杠杆
    ORIGIN_USDT=100   #默认策略初始usdt

    OPEN_FEE_RADIO=0.0004  #手续费占比
    CLOSE_FEE_RADIO=0.0004

    ROUND_RADIO=5  #近似小数位




