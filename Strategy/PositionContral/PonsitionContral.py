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
        signal=strategy_result.direction
        logger.info(f"{symbol}\t{signal} p:{self.position[symbol].direction}")
        # 同向开仓
        if  self.position[symbol].get_avg_price()==0 or abs(self.position[symbol].direction+ signal.value)==2:
            position_change=PositionChange.OPEN
            changed_usdt = strategy_result.size * self._usdt
            logger.info("同向开仓")
        # 部分平仓
        elif self.position[symbol].direction+ signal.value==0:
            position_change = PositionChange.PARTIAL
            changed_usdt = strategy_result.size * self.position[symbol].margin_usdt
            logger.info("部分平仓")
        #反转
        elif signal==PositionSignal.RESERVED:
            position_change=PositionChange.RESERVED
            changed_usdt = strategy_result.size * self.position[symbol].margin_usdt
            logger.info("反手")

        #全部平仓
        elif signal==PositionSignal.FULL:
            position_change=PositionChange.FULL
            changed_usdt = self.position[symbol].margin_usdt

            logger.info("全平")
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
        if self.position[symbol].get_avg_price()==0:
            self.close(symbol)


if __name__ == "__main__":
    # 1. 账户初始化：假设 10000 USDT 初始资金
    pc = PositionControl(usdt=10000.0)

    print(f"=== 账户初始化: {pc.all_usdt} USDT | 杠杆: {pc.leverage}x ===")

    # --- 场景 1: 同时开启两个仓位 ---
    # BTC 开多：使用可用资金的 10%
    btc_res1 = StrategyResult(symbol="BTC/USDT", size=0.1, execution_price=50000.0,
                              execution_time="T1", direction=PositionSignal.LONG)
    pc.main(btc_res1)

    # ETH 开多：使用此时剩余可用资金的 10%
    eth_res1 = StrategyResult(symbol="ETH/USDT", size=0.1, execution_price=3000.0,
                              direction=PositionSignal.LONG, execution_time="T1")
    pc.main(eth_res1)

    print(f"\n[Step 1] 初始双开仓完成:")
    for s, p in pc.position.items():
        print(f" - {s}: 保证金={p.margin_usdt:.2f}, 名义价值={p.get_nominal_value():.2f}")
    print(f"总计保证金占用: {pc.margin_usdt:.2f} | 账户剩余可用 (_usdt): {pc._usdt:.2f}")

    # --- 场景 2: 市场剧变 (BTC 暴跌，ETH 阴跌) ---
    # 此时 BTC 产生巨大浮亏，会拉高 _true_margin_usdt
    market_prices = {
        "BTC/USDT": 42000.0,  # 跌 16%，20倍杠杆下已经穿仓边缘
        "ETH/USDT": 2900.0  # 跌 3.3%
    }
    pc.update(market_prices)

    print(f"\n[Step 2] 市场暴跌更新:")
    print(f"账户真实占用 (True Margin): {pc._true_margin_usdt:.2f} (包含浮亏对保证金的挤占)")
    print(f"当前账户风险比率: {pc.margin_percent * 100:.2f}%")
    print(f"此时账户真正剩余可用资金: {pc._usdt:.2f} (此时可能已经没钱开新仓了)")

    # --- 场景 3: 尝试在风险极高时强行加仓 (测试系统安全性) ---
    print(f"\n[Step 3] 尝试在账户危机时为 ETH 加仓...")
    eth_res2 = StrategyResult(symbol="ETH/USDT", size=0.5, execution_price=2800.0,
                              direction=PositionSignal.LONG, execution_time="T2")
    # 如果 _usdt 已经因为 BTC 的浮亏变负数或极小，这里应该报错
    pc.main(eth_res2)

    # --- 场景 4: 强制全平 BTC 释放保证金 ---
    print(f"\n[Step 4] 斩仓 BTC 以释放保证金...")
    btc_close = StrategyResult(symbol="BTC/USDT", size=1.0, execution_price=42000.0,
                               direction=PositionSignal.FULL, execution_time="T3")
    pc.main(btc_close)

    print(f"BTC 平仓后，账户剩余可用 (_usdt): {pc._usdt:.2f}")
    print(f"此时 ETH 是否依然健在: {'是' if 'ETH/USDT' in pc.position else '否'}")