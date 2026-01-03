from strategy.CTA.CTA_MACD_CORE.cta_macd_core import CtaMacdCore
from strategy.types import StrategyResult, PositionSignal, Position


class BaseMacd(CtaMacdCore):
    def __init__(self):
        super().__init__()

    def init_data(self, data,macd_data) -> None:
        self.data = data
        self.macd_data = macd_data


    def update_context(self, i) -> None:
        self.macd_now = self.macd_data.iloc[i - 1]['MACD_HIST']
        self.macd_last = self.macd_data.iloc[i - 2]['MACD_HIST']
        self.price = self.macd_data.iloc[i - 1]['close']
        self.time = self.macd_data.iloc[i - 1]['close_time']

    def step(self) -> StrategyResult | None:


        # 金叉
        if self.macd_last < 0 < self.macd_now:
            return StrategyResult(
                signal=PositionSignal.OPEN,
                size=1.0,
                execution_price=self.price,
                execution_time=self.time,
                more_less=Position.MORE,
                comment="MACD golden cross"
            )

        # 死叉
        if self.macd_now < 0 < self.macd_last:
            return StrategyResult(
                signal=PositionSignal.CLOSE,
                size=1.0,
                execution_price=self.price,
                execution_time=self.time,
                comment="MACD dead cross"
            )

        return None