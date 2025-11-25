

from data.Api import *
from position_contral import PositionControl
from strategy.macd30min_strategy import Strategy
from typing import Optional, Dict, Any, Hashable

from data.type import StrategyResult, PositionSignal


class BackTest:
    """交易回测主类"""

    def __init__(self, api: Api):
        self.api: Api = api
        # 回测K线数据
        self.data: Optional[pd.DataFrame] = None
        # 交易对
        self.symbol: str = api.symbol
        # 初始本金
        self.usdt: float = Config.ORIGIN_USDT
        # 仓位控制实例
        self.position: PositionControl = PositionControl(self.symbol, self.usdt)

    def strategy_backtest_loop(self, symbol: str, data_number: int, leverage: float = Config.SET_LEVERAGE) -> None:
        """
        策略回测循环
        :param symbol: 交易对
        :param data_number: K线数据量
        :param leverage: 杠杆
        """
        chose=input("输入x调用api获取实时数据,其他使用本地csv")
        if chose=='x':
            self.data=self.api.get_backtest_data(number=data_number)
        else:
            self.data=self.api.get_csv_data(number=data_number)


        # 检查数据
        if self.data is None:
            print("数据获取失败，回测终止。")
            return

        data_len: int = len(self.data)
        # 策略参数设置字典
        setting: Dict[str, Any] = {'symbol': symbol, 'data': None, 'leverage': leverage, 'size': 0}
        execution_price=0
        execution_time=0
        # 循环遍历K线数据进行回测
        # 从 MACD 预留数据量开始 (Config.MACD_PADDING_COUNT)
        for i in range(Config.MACD_PADDING_COUNT, data_len):
            # 1. 准备数据切片
            slice_data: pd.DataFrame = self.data.iloc[:i]
            setting['data'] = slice_data

            # 2. 获取当前 K 线数据用于执行价
            current_candle: pd.Series = self.data.iloc[i - 1]
            execution_price: float = float(current_candle['close']/2+current_candle['open']/2)  # 以当前K线(i-1)的开盘价作为执行价
            execution_time: Hashable = current_candle.name  # K线时间

            # 3. 策略产生信号
            try:
                signals: StrategyResult = Strategy.strategy_macd30min(setting)  # 假设返回 StrategyResult
            except Exception as e:
                print(f"strategy_loop>Strategy策略错误:{e}")
                return

            # 4. 执行交易操作
            if signals.signal == PositionSignal.OPEN:
                self.position.open_position(size_ratio=signals.size, price=execution_price, time=execution_time)
            elif signals.signal == PositionSignal.CLOSE:
                self.position.close_position(price=execution_price, time=execution_time)

        # 有持仓就平仓
        if self.position.position != PositionSignal.EMPTY:
            self.position.close_position(price=execution_price, time=execution_time)


        self.position.print(data_len - Config.MACD_PADDING_COUNT,cl_k_time="30min/per")

        # 重置仓位
        self.position = PositionControl(self.symbol, self.usdt)






def main() -> None:
    try:
        bt: BackTest = BackTest(Api())
        while True:
            try:
                #选择 TODO:使用UI界面
                chose: int = int(input("回测数据量"))

                try:
                    # 执行回测
                    bt.strategy_backtest_loop(symbol="BTCUSDT", data_number=chose)
                except Exception as e:
                    print("回测循环错误<strategy_backtest_loop_error>:" + str(e))

            except ValueError:
                print("输入数字")
            except Exception as e:
                print("主循环内部错误<main_loop_internal_error>:" + str(e))

    except Exception as e:
        print("程序初始化错误<main_setup_error>:" + str(e))
        # 异常时重新调用 main
        main()


if __name__ == '__main__':
    main()