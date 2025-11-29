import pandas as pd

from type import PositionSignal, StrategyResult


class Strategy:

    @staticmethod
    def strategy_macd30min(setting:dict)-> StrategyResult:
        data= setting.get('data')
        current_candle: pd.Series = data.iloc[-1]
        execution_price= float(float(current_candle.get('open'))/5+float(current_candle.get('close'))*4/5)
        execution_time= current_candle.name
        leverage = setting.get('leverage')
        if 'MACD_HIST' not in data:
            raise Exception('缺少macd数据')
        cur_macd = data['MACD_HIST']
        # print(cur_macd)

        if cur_macd.iloc[-1] * cur_macd.iloc[-2] < 0:  # 趋势反转捕捉信号
            # TODO:过滤假信号
            pass
            return StrategyResult(
                signal=PositionSignal.OPEN if cur_macd.iloc[-1]>0 else PositionSignal.CLOSE,
                size=1,
                leverage=leverage,
                execution_price=float(execution_price),
                execution_time=execution_time,
            )
        return StrategyResult(
            signal=None,
            size=0,
            leverage=leverage,
            execution_price=None,
            execution_time=None,
        )