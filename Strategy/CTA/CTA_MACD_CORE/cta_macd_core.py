from abc import ABC, abstractmethod


from strategy.types import StrategyResult


class CtaMacdCore(ABC):
    def __init__(self):
        self.data = None
        self.macd_data = None
        self.macd_now = None
        self.macd_last = None
        self.price = None
        self.time = None

    @abstractmethod
    def init_data(self,**kwargs) ->None:
        """
        初始化：传入数据
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def update_context(self,i) -> None:
        """
        更新每次的数据次序
        :return:
        """


    @abstractmethod
    def step(self) -> StrategyResult | None:
        """
        在第 i 根 K 线上，给出策略判断
        """