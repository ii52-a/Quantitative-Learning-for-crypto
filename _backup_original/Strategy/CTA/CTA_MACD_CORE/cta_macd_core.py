from abc import ABC, abstractmethod

import pandas as pd

from Strategy.StrategyTypes import StrategyResult


class CtaMacdCore(ABC):
    def __init__(self):
        self.symbol:str | None = None
        self.data:pd.DataFrame | None = None
        self.macd_data:pd.DataFrame | None = None
        self.macd_now:pd.Series | None = None
        self.macd_last:pd.Series | None = None
        self.price:float | None = None
        self.time = None

    @abstractmethod
    def init_data(self,**kwargs) ->None:
        """
        初始化：传入数据
        :param kwargs:
        :return:
        """
        pass

    def run_step(self, i):
        """循环调用"""
        self.update_context(i)
        return self.step(i)

    @property
    def data_count(self):
        return len(self.macd_data)

    @property
    def get_last_time(self):
        return self.macd_data.iloc[-1]['close_time']

    @property
    def get_close_price(self):
        return self.macd_data.iloc[-1]['close']

    @abstractmethod
    def update_context(self,i) -> None:
        """
        更新每次的数据次序
        :return:
        """

    @abstractmethod
    def step(self, i) -> StrategyResult | None:
        """子类逻辑实现"""
        pass


