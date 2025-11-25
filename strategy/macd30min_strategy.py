from data.type import StrategyResult, PositionSignal


class Strategy:

    @staticmethod
    def strategy_macd30min(setting:dict)-> StrategyResult:
        data= setting.get('data')
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
            )
        return StrategyResult(
            signal=None,
            size=0,
            leverage=leverage,
        )