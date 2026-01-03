import logging
from app_logger.logger_setup import setup_logger

import pandas as pd


from Config import *
from type import PositionSignal

logger=setup_logger(__name__)
class PositionControl:
    def __init__(self, symbol: str, init_usdt: float = BackConfig.ORIGIN_USDT, leverage=BackConfig.SET_LEVERAGE):
        self.symbol: str = symbol
        self.init_usdt: float = init_usdt
        self.usdt: float = init_usdt

        # 追踪已占用保证金
        self.margin_used: float = 0

        self.position: int = PositionSignal.EMPTY
        self.size: float = 0.0
        self.open_price: float | None = None
        self.leverage: int = leverage

        self.position_history = []
        self.position_change_history = []

        self.open_first=None
        self.open_num=0
        self.open_gain=0
        self.fee:float = 0.0


    def open_position(self, size_ratio: float, price: float, time, leverage=BackConfig.SET_LEVERAGE)->None:
        """
        开仓
        :param time: 时间
        :param leverage:杠杆
        :param price: 价格
        :param size_ratio: 投入保证金占当前可用资金 (self.usdt) 的比例
        """
        try:
            # 1. 本次开仓需要的保证金
            margin_usdt = self.usdt * size_ratio

            if margin_usdt <= 0 or self.usdt < margin_usdt:
                print("资金不足，开仓失败。")
                return
            self.open_num+=1
            # 2. 计算合约数量
            # 合约数量 (e_size) = (保证金 * 杠杆) / 开仓价格
            e_size:float = round(margin_usdt * leverage / price, BackConfig.ROUND_RADIO)
            notional_value_open:float = e_size * price   #名义价值

            open_fee:float=round(notional_value_open*BackConfig.OPEN_FEE_RADIO,BackConfig.ROUND_RADIO)  #开仓手续费
            self.fee +=open_fee
            # 3. 更新资金和保证金占用
            self.margin_used += margin_usdt  # 占用保证金
            self.usdt -= margin_usdt  # 从可用资金中扣除保证金

            # 4. 首次开仓时记录
            if self.position is PositionSignal.EMPTY and e_size != 0:
                self.open_price = price
                self.position = PositionSignal.OPEN
                self.size = e_size
                self.leverage = leverage
                self.open_first = time
                logger.debug(f"\t{self.symbol}\t[{leverage}x]")
                logger.debug(
                    f"""\t[开仓]:\t开仓价:{round(price,2)}usdt\t持有数量:{self.size}{self.symbol.replace('USDT', '')} 开仓时间:{time}\t""")

                self.position_change_history.append({
                    "symbol": self.symbol,
                    "open_price": price,
                    "size": self.size,
                    "position": "more",
                    "leverage": leverage,
                    "open_time": time,
                    "open_fee": open_fee,
                })
        except Exception as e:
            print(f"PositionControl>open_position_Error开仓错误:{e}")

    def close_position(self, price: float, time)->None:
        """平仓"""
        try:
            if self.position == PositionSignal.EMPTY:
                print("当前没有仓位需要平仓")
                return

            # 计算盈亏 pnl
            national_value=self.size*price
            close_fee:float=BackConfig.CLOSE_FEE_RADIO*national_value
            self.fee += close_fee
            pnl_usdt = round(self.size * (price - self.open_price)-self.fee,BackConfig.ROUND_RADIO)

            # 计算回报率
            # 保证金
            margin_used_for_calc = self.margin_used

            pnl = round(pnl_usdt, 5)  # 盈亏金额

            pnl_percent = round((pnl_usdt / margin_used_for_calc) * 100, BackConfig.ROUND_RADIO)

            # 格式化输出：使用 pnl 盈亏金额
            logger.debug(f"\t[平仓] 平仓价格:{round(price,2)}usdt 回报:{pnl}USDT @ 回报率:{pnl_percent:.2f}%\t")
            logger.debug(f"平仓时间:{time} 手续费:{self.fee}")
            logger.debug('=' * 40)
            if pnl_percent >0:
                self.open_gain+=1

            # 更新资金
            self.usdt += margin_used_for_calc + pnl_usdt

            # 记录变动记录

            self.position_change_history.append({

                "symbol": self.symbol,

                "open_price": price,

                "size": self.size,

                "position": "more" if self.size < 0 else "less",

                "leverage": self.leverage,

                "close_time": time,

            })

            # 记录仓位历史

            self.position_history.append({

                "symbol": self.symbol,

                "open_price": round(self.open_price,2),

                "close_price": round(price,2),

                f"size({self.symbol.replace('USDT','')})": self.size,

                "position": PositionSignal.CLOSE,

                "leverage": self.leverage,

                "open_time": self.open_first,

                "close_time": time,

                "pnl": pnl,

                "pnl_percent": pnl_percent,
                "fee": -round(self.fee,BackConfig.ROUND_RADIO),

            })

            # 清空仓位
            self.position = PositionSignal.EMPTY
            self.size = 0
            self.open_price = None
            self.margin_used = 0 # 清空占用的保证金
            self.leverage = BackConfig.SET_LEVERAGE
            self.fee =0
            self.open_first = None

        except Exception as e:
            print(f"PositionControl>close_position_Error平仓错误:{e}")

    def print(self,k_num,cl_k_time:str,interval):
        if self.open_num<50:
            for i in self.position_history:
                logger.debug(i)

        npnl_percent = round((self.usdt - self.init_usdt) / self.init_usdt * 100, 3)
        logger.info('='*40)
        logger.info(f"回测k线数量:{k_num}({cl_k_time})\t回测时长:{pd.Timedelta(minutes=k_num*TradeMapper.K_LINE_TO_MINUTE[interval])}")
        logger.info(f"模拟投入(usd/usdt):{self.init_usdt}\t\t回测结果(usd/usdt):{self.usdt:.4f}")
        logger.info(f"共开仓{self.open_num}次,总盈利{self.open_gain}次，胜率:{self.open_gain/self.open_num*100:.2f}%")
        logger.info(f"策略总回报率:{npnl_percent}% ")
        logger.info('=' * 40)