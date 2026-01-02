

from data.Api import *
from BackTest.position_contral import PositionControl
from strategy.cta_macd_strategy import Strategy


from type import BackTestSetting, StrategyResult, PositionSignal, StaSetting


class BackTest:
    """交易回测主类"""

    def __init__(self,api:Api,back_test_setting:BackTestSetting):


        # 交易对
        self.symbol: str = back_test_setting.trading_pair
        # 初始本金
        self.usdt: float = back_test_setting.origin_usdt


        self.leverage=back_test_setting.leverage
        self.api: Api = api
        # 仓位控制实例
        self.position: PositionControl = PositionControl(self.symbol, self.usdt)
        # 回测K线数据
        self.data: pd.DataFrame | None = None

    def strategy_backtest_loop(self,
                               number: int,
                               end_time:pd.Timestamp=pd.to_datetime('now', unit='ms')
                               ) -> None:

        # 策略参数设置字典
        setting: StaSetting=StaSetting(
            symbol=self.symbol,
            leverage=self.leverage,
            size=0,
            end_time=end_time,
            number=number,
        )

        #     # 4. 执行交易操作
        #     if signals.signal == PositionSignal.OPEN:
        #         self.position.open_position(size_ratio=signals.size, price=signals.execution_price, time=signals.execution_time)
        #     elif signals.signal == PositionSignal.CLOSE:
        #         self.position.close_position(price=signals.execution_price, time=signals.execution_time)
        #
        # # 有持仓就平仓
        # if self.position.position != PositionSignal.EMPTY:
        #     self.position.close_position(price=float(cur_price), time=cur_time)
        #
        #
        # self.position.print(data_len - Config.PADDING_COUNT,cl_k_time=f"{interval}/per",interval=interval)
        #
        # # 重置仓位
        # self.position = PositionControl(self.symbol, self.usdt)






def main():
    # try:
    #     bt: BackTest = BackTest()
    #     while True:
    #         try:
    #             #选择 TODO:使用UI界面
    #             chose: int = int(input("回测数据量"))
    #
    #             try:
    #                 # 执行回测
    #                 bt.strategy_backtest_loop(symbol="BTCUSDT", data_number=chose)
    #             except Exception as e:
    #                 print("回测循环错误<strategy_backtest_loop_error>:" + str(e))
    #
    #         except ValueError:
    #             print("输入数字")
    #         except Exception as e:
    #             print("主循环内部错误<main_loop_internal_error>:" + str(e))
    #
    # except Exception as e:
    #     print("程序初始化错误<main_setup_error>:" + str(e))
    #     # 异常时重新调用 main
    #     main()
    pass
    #TODO:重写文件启动服务


if __name__ == '__main__':
    main()  #TODO