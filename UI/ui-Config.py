from binance import Client
class TradeMapper:
    K_LINE_TYPE:dict[str,str]={
    '1s':Client.KLINE_INTERVAL_1SECOND,
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

    trade_pair:str=''