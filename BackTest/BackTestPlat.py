

from data.Api import *
from BackTest.position_contral import PositionControl
from strategy.cta_macd_strategy import Strategy
from typing import  Dict, Any


from type import BackTestSetting, StrategyResult, PositionSignal



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

    def strategy_backtest_loop(self, interval:str,data:pd.DataFrame) -> None:
        """
        策略回测循环
        :param interval<biance str>: 周期数,输入biance常量str，解耦self形参,变化的参数不要放在绑定的类属性上,容易与单个对象耦合
        :param data: 数据
        """
        #TODO 修改为UI控制
        # chose=input("输入x调用api获取实时数据,其他使用本地csv")
        # if chose=='x':
        #     self.data=self.api.get_backtest_data(number=data_number,interval=interval)
        # else:
        #     self.data=self.api.get_csv_data(number=data_number)


        # 检查数据
        if data is None:
            print("数据获取失败，回测终止。")
            return

        data_len: int = len(data)
        # 策略参数设置字典
        setting: Dict[str, Any] = {'symbol': self.symbol, 'data': None, 'leverage': self.leverage, 'size': 0}
        cur_price=data.iloc[-1]['close']
        cur_time=data.index[-1]
        # print(cur_price,cur_time)
        # 循环遍历K线数据进行回测
        # 从 预留数据量开始 (Config.PADDING_COUNT)
        for i in range(Config.PADDING_COUNT, data_len):
            # 1. 准备数据切片
            slice_data: pd.DataFrame = data.iloc[:i]
            setting['data'] = slice_data
            # 3. 策略产生信号
            try:
                signals: StrategyResult = Strategy.strategy_macd30min(setting)  #  StrategyResult
            except Exception as e:
                print(f"strategy_loop>Strategy策略错误:{e}")
                return

            # 4. 执行交易操作
            if signals.signal == PositionSignal.OPEN:
                self.position.open_position(size_ratio=signals.size, price=signals.execution_price, time=signals.execution_time)
            elif signals.signal == PositionSignal.CLOSE:
                self.position.close_position(price=signals.execution_price, time=signals.execution_time)

        # 有持仓就平仓
        if self.position.position != PositionSignal.EMPTY:
            self.position.close_position(price=float(cur_price), time=cur_time)


        self.position.print(data_len - Config.PADDING_COUNT,cl_k_time=f"{interval}/per",interval=interval)

        # 重置仓位
        self.position = PositionControl(self.symbol, self.usdt)






def main() -> None:
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