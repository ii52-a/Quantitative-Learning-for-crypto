from Strategy.CTA.CTA_MACD_CORE.cta_macd_core import CtaMacdCore
from Strategy.StrategyTypes import StrategyResult, PositionSignal
from Strategy.data_service import Dataservice


class BaseMacd(CtaMacdCore):
    def __init__(self):
        super().__init__()
        self.open=False

    def init_data(self,symbol,k_line,number) -> None:
        self.symbol=symbol
        self.k_line=k_line
        self.data=Dataservice.get_data(self.symbol,self.k_line,number)
        self.macd_data = Dataservice.macd_data(self.data)


    def update_context(self, i) -> None:
        self.macd_now = self.macd_data.iloc[i - 1]['MACD_HIST']
        self.macd_last = self.macd_data.iloc[i - 2]['MACD_HIST']
        self.price = self.macd_data.iloc[i - 1]['close']
        self.time = self.macd_data.iloc[i - 1]['close_time']

    def step(self,i) -> StrategyResult | None:

        # 金叉
        if self.macd_last < 0 < self.macd_now and not self.open:
            self.open=True
            return StrategyResult(
                symbol=self.symbol,
                direction=PositionSignal.LONG,
                size=1.0,
                execution_price=self.price,
                execution_time=self.time,
                comment="MACD golden cross"
            )


        # 死叉
        elif self.macd_now < 0 < self.macd_last and self.open:
            self.open=False
            return StrategyResult(
                symbol=self.symbol,
                direction=PositionSignal.FULL,
                size=1.0,
                execution_price=self.price,
                execution_time=self.time,
                comment="MACD dead cross"
            )

        return None