from Config import BackConfig
from Strategy.PositionContral.Position import Position
from Strategy.StrategyTypes import StrategyResult, PositionSet, PositionSignal, PositionResult, \
    PositionChange
from app_logger.logger_setup import setup_logger

logger=setup_logger(__name__)
class PositionControl:
    def __init__(self,usdt):
        self.position:dict[str:Position]={}
        #全部持有usdt价值
        self.all_usdt=usdt

        self._true_margin_usdt:float=0


        #保险金比率
        self.margin_percent:float=0
        #默认杠杆
        self.leverage:int=BackConfig.SET_LEVERAGE

    # 仓位占用usdt，只考虑当前仓位占用
    @property
    def margin_usdt(self):
        mu = 0
        for i in self.position.values():
            mu += i.margin_usdt
        return mu

    @property
    def _usdt(self):
        """
        计算，当前阶段开仓可用usdt,为总价值-当前过程真实占用与仓位占用的最大值，不进行盈利复投
        """
        return self.all_usdt - max(self.margin_usdt,self._true_margin_usdt)


    def true_margin_usdt(self,cur_price:dict):
        new_margin = 0
        for symbol, position in self.position.items():
            if symbol not in cur_price.keys():
                logger.error(f"价格信息不足，保险金比率计算失效！")
            else:
                curr_price = cur_price[symbol]
                new_margin += position.true_margin(curr_price)

        self._true_margin_usdt=new_margin

    @property
    def _liquidated(self):
        """
            TODO：精确爆仓需要秒级数据，仅使用k线极值法估算是否爆仓，其误差在于每个仓位的极值不是同时取得的。
        :return:
        """
        return self.margin_percent >=0.98

    def update(self,cur_price:dict):


        self.true_margin_usdt(cur_price)
        self.margin_percent = min(self._true_margin_usdt / (self.all_usdt - self.margin_usdt + self._true_margin_usdt),
                                  0.98)
        if self._liquidated:
            """进入爆仓流程"""
            pass

    def close(self,symbol:str):
        self.position.pop(symbol)

    def _sign_transform(self,symbol,strategy_result:StrategyResult):
        signal=strategy_result.more_less
        print(signal)
        # 同向开仓
        if self.position[symbol].direction* signal.value>=0:
            position_change=PositionChange.OPEN
            changed_usdt = strategy_result.size * self._usdt
        #反转
        elif signal==PositionSignal.RESERVED:
            position_change=PositionChange.RESERVED
            changed_usdt = strategy_result.size * self._usdt
        #部分平仓
        elif signal==PositionSignal.PARTIAL:
            position_change=PositionChange.PARTIAL
            changed_usdt = strategy_result.size * self.position[symbol].margin_usdt
        #全部平仓
        elif signal==PositionSignal.FULL:
            position_change=PositionChange.FULL
            changed_usdt = self.position[symbol].margin_usdt
            self.close(symbol)
        else:
            logger.error(f"仓位管理：策略信号转化失败")
            changed_usdt = 0
            position_change = PositionChange.ERROR
        return changed_usdt,position_change


    def main(self,strategy_result:StrategyResult):
        symbol=strategy_result.symbol
        if symbol not in self.position.keys():
            #创建仓位
            self.position.setdefault(symbol,Position(symbol,self.leverage))
        price=strategy_result.execution_price

        #仓位执行信号

        #策略 -> 仓位 信号转化


        changed_usdt,position_change=self._sign_transform(symbol,strategy_result)


        if changed_usdt>self._usdt:
            logger.error(f"可用保险金不足！")
            position_change=PositionChange.ERROR
            changed_usdt=0


        position_set:PositionSet=PositionSet(
            signal=position_change,
            changed_usdt=changed_usdt,
            price=price
        )
        result:PositionResult=self.position[symbol].execute(position_set)
        pnl=result.pnl


if __name__ == "__main__":
    # 1. 初始化仓位控制器，初始资金 10000 USDT
    pc = PositionControl(usdt=10000.0)
    print(f"--- 初始状态: 总资金 {pc.all_usdt} USDT ---")

    # 模拟当前市场价格
    market_prices = {"BTC/USDT": 50000.0, "ETH/USDT": 3000.0}

    # ---------------------------------------------------------
    # 2. 场景一：开启 BTC 多头仓位 (占用可用资金的 10%)
    # ---------------------------------------------------------
    res1 = StrategyResult(
        symbol="BTC/USDT",
        size=0.1,
        execution_price=50000.0,
        execution_time="2026-01-17 12:00:00",
        more_less=PositionSignal.MORE,
        comment="突破信号"
    )
    print("\n[Action] 开启 BTC 多单 (10% 保证金)...")
    pc.main(res1)
    pc.update(market_prices)

    btc_pos = pc.position["BTC/USDT"]
    print(
        f"BTC 仓位: 均价={btc_pos.get_avg_price()}, 占用保证金={btc_pos.margin_usdt}, 名义价值={btc_pos.get_nominal_value()}")
    print(f"账户状态: 可用={pc._usdt}, 保证金占比={pc.margin_percent:.2%}")

    # ---------------------------------------------------------
    # 3. 场景二：BTC 价格上涨，进行加仓 (再追加可用资金的 10%)
    # ---------------------------------------------------------
    market_prices["BTC/USDT"] = 55000.0
    res2 = StrategyResult(
        symbol="BTC/USDT",
        size=0.1,
        execution_price=55000.0,
        execution_time="2026-01-17 13:00:00",
        more_less=PositionSignal.MORE,
        comment="趋势回调加仓"
    )
    print(f"\n[Action] BTC 上涨至 55000，加仓...")
    pc.main(res2)
    pc.update(market_prices)
    print(f"BTC 均价更新为: {btc_pos.get_avg_price():.2f}")

    # ---------------------------------------------------------
    # 4. 场景三：开启 ETH 空头仓位 (双向持仓模拟)
    # ---------------------------------------------------------
    res3 = StrategyResult(
        symbol="ETH/USDT",
        size=0.2,
        execution_price=3000.0,
        execution_time="2026-01-17 14:00:00",
        more_less=PositionSignal.LESS,
        comment="ETH 遇阻开空"
    )
    print(f"\n[Action] 开启 ETH 空单 (20% 保证金)...")
    pc.main(res3)
    pc.update(market_prices)
    print(f"当前总仓位数量: {len(pc.position)}")
    print(f"当前账户保证金比率: {pc.margin_percent:.2%}")

    # ---------------------------------------------------------
    # 5. 场景四：BTC 部分平仓 (平掉一半保证金)
    # ---------------------------------------------------------
    res4 = StrategyResult(
        symbol="BTC/USDT",
        size=0.5,  # 在 _sign_transform 中，PARTIAL 信号使用此比例乘以当前仓位保证金
        execution_price=60000.0,
        execution_time="2026-01-17 15:00:00",
        more_less=PositionSignal.PARTIAL,
        comment="BTC 止盈一半"
    )
    print(f"\n[Action] BTC 上涨至 60000，平掉一半仓位...")
    pc.main(res4)
    pc.update(market_prices)
    print(f"BTC 剩余保证金: {pc.position['BTC/USDT'].margin_usdt}")

    # ---------------------------------------------------------
    # 6. 场景五：BTC 反手 (由多转空)
    # ---------------------------------------------------------
    res5 = StrategyResult(
        symbol="BTC/USDT",
        size=0.15,
        execution_price=58000.0,
        execution_time="2026-01-17 16:00:00",
        more_less=PositionSignal.RESERVED,
        comment="趋势反转，反手做空"
    )
    print(f"\n[Action] BTC 信号反转，平多开空...")
    pc.main(res5)
    pc.update(market_prices)
    print(f"BTC 当前方向 (1多-1空): {pc.position['BTC/USDT'].direction}")
    print(f"BTC 当前名义价值: {pc.position['BTC/USDT'].get_nominal_value()}")

    # ---------------------------------------------------------
    # 7. 场景六：极端行情下的爆仓检查
    # ---------------------------------------------------------
    print(f"\n[Scenario] 模拟极端行情导致 ETH 爆仓...")
    market_prices["ETH/USDT"] = 10000.0  # ETH 暴涨，空单巨亏
    pc.update(market_prices)
    print(f"当前真实占用保证金: {pc._true_margin_usdt:.2f}")
    print(f"当前保证金比率: {pc.margin_percent:.2%}")
    if pc._liquidated:
        print("!!! 警报：账户已达到爆仓阈值 !!!")








