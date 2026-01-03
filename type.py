from dataclasses import dataclass
from typing import Hashable

import pandas as pd

import Config


from dataclasses import dataclass, field

from Config import BackConfig




"""策略输入setting"""
@dataclass
class StaSetting:
    symbol: str
    leverage: int
    size: int
    number:int
    end_time: pd.Timestamp=pd.Timestamp.now()




@dataclass
class BackTestSetting:
    k_line: str
    trading_pair: str
    strategy: str
    origin_usdt: float=BackConfig.ORIGIN_USDT
    leverage:int=BackConfig.SET_LEVERAGE