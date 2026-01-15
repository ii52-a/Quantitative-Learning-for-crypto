
from Strategy.StrategyTypes import *


class Position:
    def __init__(self,symbol,leverage):
        self.symbol: str = symbol
        self._position_status:PositionStatus=PositionStatus(
            #占用保险金
            margin_used = 0,
            #实际占用，计算当前盈亏
            true_margin_used = 0,
            #名义价值
            nominal_value = 0,
            #平均开仓价格
            avg_price = 0,
            #杠杆
            leverage= leverage,
            #开仓次数
            open_count = 0,
            #平仓次数  -产生盈亏为平仓
            close_count = 0,
        )
        #流动字段,不可调用
        self._nominal_value = None
        self._price = None
        self._signal = None
        self._changed_used = None

    #返回真实保险金占用
    def true_margin(self,price) -> float:
        """
        提供调用以模拟过程真实保险金比例计算。
        :param price:
        :return:
        """
        cur_pnl= (price - self._position_status.avg_price)/self._position_status.avg_price * self._position_status.nominal_value
        return self._position_status.margin_used - cur_pnl

    #仓位处理
    def execute(self,position_set:PositionSet) -> PositionResult:
        """
        仓位变动处理入口
        :param position_set:
        :return:
        """
        self._update_context(position_set)
        #判断为开仓
        if self._nominal_value * self._position_status.nominal_value >= 0:
            return self._open()
        #反向开仓
        elif abs(self._nominal_value) > abs(self._position_status.nominal_value):
            return self._close_and_reverse()
        #部分平仓
        else:
            return self._close()



    def _update_context(self,position_set):
        """
        处理仓位变动参数
        :param position_set
        :return:
        """
        # 变动保险金
        self._changed_used = position_set.changed_usdt
        # 方向
        self._signal = position_set.signal

        # 仓位变动价格
        self._price = position_set.price

        # 变动名义价值   (U本位价值)  value=变动保险金 * 杠杆 * 仓位方向
        self._nominal_value = self._changed_used * self._position_status.leverage * self._signal.value

    def _close(self) -> PositionResult:
        """
        处理平仓逻辑
        :return: PositionResult
        """
        # 计算平仓状态

        # 计算pnl
        pnl: float = (self._price - self._position_status.avg_price)/self._position_status.avg_price * -self._nominal_value
        # 处理仓位 释放占用保险金，更改名义价值
        self._position_status.margin_used -= self._changed_used
        if_all = self._nominal_value == -self._position_status.nominal_value
        self._position_status.nominal_value += self._nominal_value

        close_status: PositionResultSignal = PositionResultSignal.FULL if if_all else PositionResultSignal.PARTIAL
        self._position_status.avg_price=0 if if_all else self._position_status.avg_price
        return PositionResult(
            signal=close_status,
            pnl=pnl
        )

    def _open(self) -> PositionResult:
        """
        处理开仓逻辑
        :return: PositionResult
        """
        open_status: PositionResultSignal = PositionResultSignal.OPEN
        # 计算平均价格
        new_nominal_value = self._position_status.nominal_value + self._nominal_value
        cur_persent = self._position_status.nominal_value / new_nominal_value
        add_persent = 1 - cur_persent
        self._position_status.avg_price = self._position_status.avg_price * cur_persent + self._price * add_persent
        # 占用保险金
        self._position_status.margin_used += self._changed_used
        self._position_status.nominal_value = new_nominal_value

        return PositionResult(
            signal=open_status,
            pnl=0
        )

    def _close_and_reverse(self) -> PositionResult:
        reserve_status: PositionResultSignal = PositionResultSignal.RESERVED
        #反转，平所有仓位计算盈亏
        pnl: float = (self._price - self._position_status.avg_price)/self._position_status.avg_price * self._position_status.nominal_value
        #剩余名义价值判断
        self._position_status.nominal_value =self._position_status.nominal_value + self._nominal_value
        self._position_status.avg_price = self._price
        self._position_status.margin_used = self._changed_used -self._position_status.margin_used
        return PositionResult(
            signal=reserve_status,
            pnl=pnl
        )

    def get_nominal_value(self) -> float:
        return self._position_status.nominal_value

    def get_avg_price(self) -> float:
        return self._position_status.avg_price


if __name__ == "__main__":
    # 模拟外部定义的类型和枚举（根据你的代码结构补充）
    # 假设 PositionSet, PositionStatus, PositionResult 等已导入

    # 1. 初始化：交易 BTC，杠杆 10 倍
    print("--- 步骤 1: 初始化仓位 ---")
    pos = Position(symbol="BTCUSDT", leverage=10)

    # 2. 模拟开多：价格 100，投入 100U 保证金
    # 名义价值 = 100 * 10 * 1 = 1000
    print("\n--- 步骤 2: 开多仓 (100U @ 100) ---")
    pset1 = PositionSet(changed_usdt=100, price=100, signal=PositionSignal.MORE)
    res1 = pos.execute(pset1)
    print(f"结果: {res1.signal}, 盈亏: {res1.pnl}")
    print(f"仓位状态: 均价={pos.get_avg_price()}, 名义价值={pos.get_nominal_value()}")

    # 3. 模拟部分平仓：价格 110，平掉 50U 保证金对应的仓位
    # 预期盈亏: (110 - 100) * (50 * 10) = 500
    print("\n--- 步骤 3: 部分平仓 (50U @ 110) ---")
    # 注意：平仓时 signal 应与持仓方向相反，或者你的逻辑中定义了平仓信号
    pset2 = PositionSet(changed_usdt=50, price=110, signal=PositionSignal.LESS)
    res2 = pos.execute(pset2)
    print(f"结果: {res2.signal}, 盈亏: {res2.pnl}")
    print(f"仓位状态: 均价={pos.get_avg_price()}, 名义价值={pos.get_nominal_value()}")

    # 4. 模拟加仓：价格 120，再投入 50U 保证金
    # 此时原名义价值 500，新增 500，新均价应为 (110*500 + 120*500)/1000 = 115
    # (注意：你现在的逻辑是基于当前价格和旧均价加权)
    print("\n--- 步骤 4: 再次加仓 (50U @ 120) ---")
    pset3 = PositionSet(changed_usdt=50, price=120, signal=PositionSignal.MORE)
    res3 = pos.execute(pset3)
    print(f"结果: {res3.signal}, 盈亏: {res3.pnl}")
    print(f"仓位状态: 均价={pos.get_avg_price()}, 名义价值={pos.get_nominal_value()}")

    # 5. 模拟反手：价格 130，变动 200U 保证金（即先平掉当前的，再开反向）
    # 预期盈亏: (130 - 115) * 1000 = 1500
    print("\n--- 步骤 5: 反手做空 (200U @ 130) ---")
    print(f"{pos.get_avg_price()}")
    pset4 = PositionSet(changed_usdt=200, price=130, signal=PositionSignal.LESS)
    res4 = pos.execute(pset4)

    print(f"结果: {res4.signal}, 盈亏: {res4.pnl}")

    print(f"新仓位状态: 均价={pos.get_avg_price()}, 名义价值={pos.get_nominal_value()}")

    # 6. 检查实时保证金占用
    print("\n--- 步骤 6: 检查实时保证金 (价格跌到 120) ---")
    m = pos.true_margin(price=130)
    print(f"当前价格 130 时的真实保证金占用: {m}")
    # 如果现在做空，价格跌了，应该盈利，true_margin 应该变小（风险降低）
    m = pos.true_margin(price=120)
    print(f"当前价格 120 时的真实保证金占用: {m}")





