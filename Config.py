from dataclasses import dataclass
from typing import Type

from binance import Client
from requests import RequestException
from requests.exceptions import SSLError


@dataclass
class ApiConfig:
    """RETRY参数"""
    #最大尝试
    MAX_RETRY:int=20
    #最大等待时间
    WAITING_TIME:float=1.5
    # 允许错误类型
    RETRY_ERROR_ACCEPT:tuple[Type[BaseException]]=(RequestException, ConnectionAbortedError, SSLError)
    # 最大请求时长
    CUSTOM_TIMEOUT:tuple[int]=(10,30)
    # 最大单次请求,最大1000,不建议调低，将会增加api封禁可能
    LIMIT:int = 1000

    #请求间隔，调高将减少api封禁可能，但会加长请求时间
    API_BASE_GET_INTERVAL=0.2

    """LocalData模块参数"""
    # 本地数据存储的文件夹地址
    LOCAL_DATA_CSV_DIR: str = 'data_csv'

    # 最大本地csv保存值
    LOCAL_MAX_CSV_NUMBER: int = 10000

    #本地数据库更新频率 /天
    LOCAL_BASE_DATA_UPDATE_INTERVAL=3


    #增加n条数据以平衡指数线偏差,减少前期指标的巨大误差
    PADDING_COUNT:int=300
    # 计算基础数据量:50条数据为nan删除数据
    CALCULATE_BASECOUNT:int=50
    # 请求数据总值
    GET_COUNT:int=PADDING_COUNT+CALCULATE_BASECOUNT



    #允许最大初始化回测的历史跨度单位:日
    LOCAL_MAX_HISTORY_ALLOW:int=1000



class BackConfig:
    """默认投入参数"""
    # 全局默认杠杆
    SET_LEVERAGE:int = 5
    # 默认策略初始usdt
    ORIGIN_USDT:float = 100
    # 手续费占比
    OPEN_FEE_RADIO:float = 0.0004
    CLOSE_FEE_RADIO:float = 0.0004
    # 回测数据近似小数位
    ROUND_RADIO:int = 5





"""TODO:移除，懒得改"""
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
    strategy:list=['CTA-macd']


