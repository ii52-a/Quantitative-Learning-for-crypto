from Strategy.CTA.CTA_MACD_CORE.cta_macd_core import CtaMacdCore
from Strategy.StrategyTypes import StrategyResult, PositionSignal
from Strategy.data_service import Dataservice


class BaseMacd(CtaMacdCore):
    """MACD 柱穿零：上穿做多，下穿平多（只做多，单笔全仓）。"""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__()
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.open = False
        self.number = 0

    def init_data(self, symbol, k_line, number) -> None:
        self.symbol = symbol
        self.k_line = k_line
        self.number = int(number)
        raw = Dataservice.get_data(self.symbol, self.k_line, self.number)
        macd = Dataservice.macd_data(raw, fast=self.fast, slow=self.slow, signal=self.signal)
        # 只保留最近 number 根，保证回测区间与请求一致
        self.macd_data = macd.iloc[-self.number:] if len(macd) > self.number else macd
        self.data = self.macd_data
        self.open = False

    def update_context(self, i) -> None:
        row = self.macd_data.iloc[i]
        prev = self.macd_data.iloc[i - 1]
        self.macd_now = row['MACD_HIST']
        self.macd_last = prev['MACD_HIST']
        self.price = float(row['close'])
        self.time = row['close_time']

    def step(self, i) -> StrategyResult | None:
        # 金叉：HIST 由负转正（等价于 DIF 上穿 DEA）
        if self.macd_last < 0 < self.macd_now and not self.open:
            self.open = True
            return StrategyResult(
                symbol=self.symbol,
                direction=PositionSignal.LONG,
                size=1.0,
                execution_price=self.price,
                execution_time=self.time,
                comment="MACD golden cross"
            )

        # 死叉：HIST 由正转负
        if self.macd_now < 0 < self.macd_last and self.open:
            self.open = False
            return StrategyResult(
                symbol=self.symbol,
                direction=PositionSignal.FULL,
                size=1.0,
                execution_price=self.price,
                execution_time=self.time,
                comment="MACD dead cross"
            )

        return None
