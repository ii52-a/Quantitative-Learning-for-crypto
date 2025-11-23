import talib

from BackTest import Api

if __name__ == '__main__':
    api = Api()
    data=api.get_futures_data()
    macd, macd_signal, macd_hist = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    ##添加指标至data数据
    data['MACD'] = macd
    data['MACD_SIGNSL'] = macd_signal
    data['MACD_HIST'] = macd_hist
    print(data)