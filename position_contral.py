from Config import *

class PositionControl:
    def __init__(self, symbol: str, init_usdt: float = Config.ORIGIN_USDT, leverage=Config.SET_LEVERAGE):
        self.symbol: str = symbol
        self.init_usdt: float = init_usdt
        self.usdt: float = init_usdt

        # 统一使用 self.margin_used 追踪已占用保证金
        self.margin_used: float = 0

        self.position: PositionSignal = PositionSignal.EMPTY
        # 统一使用 self.size 追踪合约数量 (BTC 数量)
        self.size: float = 0
        self.open_price: float | None = None
        self.leverage: int = leverage

        self.position_history = []
        self.position_change_history = []

        self.open_first=None


    def open_position(self, size_ratio: float, price: float, time, leverage=Config.SET_LEVERAGE):
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

            # 2. 计算合约数量
            # 合约数量 (e_size) = (保证金 * 杠杆) / 开仓价格
            e_size = round(margin_usdt * leverage / price, 4)

            # 3. 更新资金和保证金占用
            self.margin_used += margin_usdt  # 占用保证金
            self.usdt -= margin_usdt  # 从可用资金中扣除保证金 (最关键的修正)

            # 4. 首次开仓时记录
            if self.position is PositionSignal.EMPTY and e_size != 0:
                self.open_price = price
                self.position = PositionSignal.OPEN
                self.size = e_size
                self.leverage = leverage
                self.open_first = time

                print(
                    f"[开仓]:\t{self.symbol}\t[{leverage}x]\t开仓价:{price}usdt\t持有数量:{self.size}{self.symbol.replace('USDT', '')}")

                self.position_change_history.append({
                    "symbol": self.symbol,
                    "open_price": price,
                    "size": self.size,
                    "position": "more",
                    "leverage": leverage,
                    "open_time": time,
                })
        except Exception as e:
            print(f"PositionControl>open_position_Error开仓错误:{e}")

    def close_position(self, price: float, time):
        """平仓"""
        try:
            if self.position == PositionSignal.EMPTY:
                print("当前没有仓位需要平仓")
                return

            # 计算盈亏 pnl
            pnl_usdt = self.size * (price - self.open_price)

            # 计算回报率
            # 保证金
            margin_used_for_calc = self.margin_used

            pnl = round(pnl_usdt, 5)  # 盈亏金额
            pnl_percent = round((pnl_usdt / margin_used_for_calc) * 100, 5)

            # 格式化输出：使用 pnl 盈亏金额
            print(f"[平仓] 平仓价格:{price}usdt 回报:{pnl}USDT @ 回报率:{pnl_percent:.2f}%")
            print('=' * 40)

            # 更新资金
            self.usdt += margin_used_for_calc + pnl_usdt

            # 记录变动记录

            self.position_change_history.append({

                "symbol": self.symbol,

                "open_price": price,

                "size": self.size,

                "position": "more" if self.size < 0 else "less",

                "leverage": self.leverage,

                "open_time": time,

            })

            # 记录仓位历史

            self.position_history.append({

                "symbol": self.symbol,

                "open_price": self.open_price,

                "close_price": price,

                "size": self.size,

                "position": PositionSignal.CLOSE,

                "leverage": self.leverage,

                "open_time": self.open_first,

                "close_time": time,

                "pnl": pnl,

                "pnl_percent": pnl_percent

            })

            # 清空仓位
            self.position = PositionSignal.EMPTY
            self.size = 0
            self.open_price = None
            self.margin_used = 0 # 清空占用的保证金
            self.leverage = Config.SET_LEVERAGE

        except Exception as e:
            print(f"PositionControl>close_position_Error平仓错误:{e}")

    def print(self,k_num):
        for i in self.position_history:
            print(i)

        npnl_percent = round((self.usdt - self.init_usdt) / self.init_usdt * 100, 3)
        print('='*40)
        print(f"回测k线数量:{k_num}\t回测时长:{pd.Timedelta(minutes=k_num*30)}")
        print(f"模拟投入(usd/usdt):{self.init_usdt}\t\t回测结果(usd/usdt):{self.usdt:.4f}")
        print(f"策略总回报率:{npnl_percent}%")
        print('=' * 40)