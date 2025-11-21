import pandas as pd
import talib
from binance.client import Client

# 设置API密钥
api_key = 'your_api_key'
api_secret = 'your_api_secret'

# 初始化Binance客户端
client = Client(api_key, api_secret)


# 获取 30 分钟 K 线数据
def get_klines(symbol, interval, limit=100):
    klines = client.get_historical_klines(symbol, interval, limit=limit)
    # 将数据转为 DataFrame 格式，方便后续操作
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                         'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                         'taker_buy_quote_asset_volume', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    data = data[['close']].astype(float)  # 只保留 'close' 价格
    return data


# 计算 MACD 指标
def calculate_macd(data):
    macd, macd_signal, macd_hist = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD'] = macd
    data['MACD_Signal'] = macd_signal
    data['MACD_Hist'] = macd_hist
    return data


# 生成交易信号
def generate_signal(data):
    data['signal'] = 0  # 初始化信号列
    # 买入信号：MACD 线突破信号线
    data.loc[data['MACD'] > data['MACD_Signal'], 'signal'] = 1
    # 卖出信号：MACD 线跌破信号线
    data.loc[data['MACD'] < data['MACD_Signal'], 'signal'] = -1
    return data


# 执行交易（买入/卖出）
def execute_trade(symbol, signal, quantity):
    if signal == 1:
        # 买入信号
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
        print(f"买入成功: {order}")
    elif signal == -1:
        # 卖出信号
        order = client.order_market_sell(symbol=symbol, quantity=quantity)
        print(f"卖出成功: {order}")
    else:
        print("没有交易信号")


# 策略运行函数
def run_strategy(symbol, interval, quantity):
    # 获取市场数据
    data = get_klines(symbol, interval)

    # 计算 MACD 指标
    data = calculate_macd(data)

    # 生成交易信号
    data = generate_signal(data)

    # 输出最近几条数据
    print(data.tail())

    # 执行交易信号
    for index, row in data.iterrows():
        signal = row['signal']
        if signal != 0:  # 只在有信号时执行交易
            execute_trade(symbol, signal, quantity)


# 主函数
if __name__ == "__main__":
    # 设置交易对和数量
    symbol = 'BTCUSDT'  # 比特币与USDT的交易对
    interval = Client.KLINE_INTERVAL_30MINUTE  # 30 分钟 K 线
    quantity = 0.001  # 假设交易 0.001 BTC

    # 运行策略
    run_strategy(symbol, interval, quantity)
