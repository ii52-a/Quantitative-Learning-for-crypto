from pathlib import Path

import pandas as pd

from Strategy.CTA.CTA_MACD_CORE import BaseMacd
from Strategy.PositionContral.PonsitionContral import PositionControl
from Strategy.StrategyTypes import PositionSignal, StrategyResult
from app_logger.logger_setup import Logger
from urlibs import get_progress

logger = Logger(__name__)

TRADE_COLUMNS = ['close_type', 'symbol', 'leverage', 'open', 'close', 'open_time',
                 'close_time', 'pnl', 'pnl_percent', 'fee', 'balance']


class StrategyMacd:
    def __init__(self, symbol: str, k_line: str, number: int = 2000, usdt: float = 1000,
                 leverage: int | None = None, fee_rate: float | None = None):
        self.symbol = symbol
        self.k_line = k_line
        self.number = int(number)
        self.usdt = float(usdt)
        self.positionControl: PositionControl = PositionControl(
            self.usdt, leverage=leverage, fee_rate=fee_rate)
        self.core = BaseMacd()
        self.core.init_data(symbol, k_line, self.number)

    @property
    def status(self):
        return self.positionControl.status

    def main(self, progress=None, should_stop=None, output_csv: str | Path | None = None) -> dict:
        """
        :param progress: 进度回调 progress(percent:int)
        :param should_stop: 取消回调，返回 True 时提前结束
        :param output_csv: 逐笔结果输出路径
        """
        count = self.core.data_count
        if count < 3:
            raise ValueError(f"{self.symbol} {self.k_line} 有效K线不足({count})，无法回测")

        if progress:
            progress(0)

        stopped = False
        with get_progress(total=count - 1, desc="回测进度") as pbar:
            for i in range(1, count):
                if should_stop is not None and should_stop():
                    logger.warning("回测被取消，提前结束并平掉当前仓位")
                    stopped = True
                    break

                strategy_result = self.core.run_step(i)
                if strategy_result is not None:
                    self.positionControl.main(strategy_result)

                # 策略状态与真实仓位对齐：订单被拒时不会出现“以为有仓”的漂移
                if not self.positionControl.status:
                    self.core.open = False

                if i % 100 == 0:
                    pbar.n = i
                    pbar.refresh()
                    if progress:
                        progress(int(i * 100 / (count - 1)))

        if self.status:
            self.all_full()

        summary = self.summary(bars=count, stopped=stopped)
        if output_csv:
            self.export_trades(output_csv)
        if progress:
            progress(100)
        return summary

    def all_full(self) -> None:
        """回测结束一键平仓（价格取最后一根 K 线收盘）。"""
        price = self.core.get_close_price
        last_time = self.core.get_last_time
        for symbol in list(self.positionControl.position):
            st = StrategyResult(
                symbol=symbol,
                size=1.0,
                execution_price=price,
                execution_time=last_time,
                direction=PositionSignal.FULL,
                comment="回测结束强制平仓",
            )
            self.positionControl.main(st)

    def summary(self, bars: int, stopped: bool = False) -> dict:
        pc = self.positionControl
        net = pc.all_usdt - pc.init_usdt
        fees = pc.fees_total
        return {
            'symbol': self.symbol,
            'k_line': self.k_line,
            'bars': bars,
            'initial': round(pc.init_usdt, 6),
            'final': round(pc.all_usdt, 6),
            'net_pnl': round(net, 6),
            'gross_pnl': round(net + fees, 6),
            'fees': round(fees, 6),
            'return_pct': round(net * 100 / pc.init_usdt, 4) if pc.init_usdt else 0.0,
            'trades': pc.total,
            'win': pc.win,
            'lose': pc.lose,
            'breakeven': pc.breakeven,
            'win_rate': round(pc.win * 100 / pc.total, 4) if pc.total else 0.0,
            'avg_trade': round(net / pc.total, 6) if pc.total else 0.0,
            'stopped': stopped,
        }

    def export_trades(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.positionControl.trades, columns=TRADE_COLUMNS)
        df.to_csv(path, index=False, encoding='utf-8-sig')
        logger.info(f"逐笔结果已写入 {path}")
        return path


if __name__ == '__main__':
    s30 = StrategyMacd("ETHUSDT", "30min", number=20000)
    result = s30.main(progress=lambda p: logger.info(f"进度 {p}%"))
    print(result)
