from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Any, AsyncIterator, Callable

import pandas as pd
from binance import AsyncClient
from binance.enums import SIDE_BUY, SIDE_SELL
from binance import Client

INTERVAL_MAP = {
    "1min": Client.KLINE_INTERVAL_1MINUTE,
    "3min": Client.KLINE_INTERVAL_3MINUTE,
    "5min": Client.KLINE_INTERVAL_5MINUTE,
    "15min": Client.KLINE_INTERVAL_15MINUTE,
    "30min": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "2h": Client.KLINE_INTERVAL_2HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "6h": Client.KLINE_INTERVAL_6HOUR,
    "12h": Client.KLINE_INTERVAL_12HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
    "1w": Client.KLINE_INTERVAL_1WEEK,
    "1M": Client.KLINE_INTERVAL_1MONTH,
}

from Strategy.PositionContral.live_position import LivePosition, Side
from Strategy.PositionContral.live_position_manager import (
    LivePositionManager,
    RiskConfig,
    TradeRecord,
)
from Strategy.kline_validator import KlineTimeValidator
from app_logger.logger_setup import Logger

logger = Logger(__name__)


@dataclass(slots=True)
class MarketEvent:
    symbol: str
    price: float
    timestamp: datetime
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderCommand:
    symbol: str
    side: str
    quantity: float
    order_type: str = "MARKET"
    strategy_name: str = ""
    reason: str = ""
    reduce_only: bool = False


@dataclass(slots=True)
class OrderAck:
    order_id: str
    symbol: str
    status: str
    executed_qty: float = 0.0
    avg_price: float = 0.0
    message: str = ""


class RealTimeMarketStream:
    async def events(self) -> AsyncIterator[MarketEvent]:
        raise NotImplementedError


class OrderGateway:
    async def send_order(self, order: OrderCommand) -> OrderAck:
        raise NotImplementedError

    async def get_position(self, symbol: str) -> dict[str, Any]:
        raise NotImplementedError


class InMemoryOrderGateway(OrderGateway):
    def __init__(self, position_manager: LivePositionManager):
        self.position_manager = position_manager
        self.commands: list[OrderCommand] = []
        self._order_id = 0

    async def send_order(self, order: OrderCommand) -> OrderAck:
        self.commands.append(order)
        self._order_id += 1

        side = Side.LONG if order.side == SIDE_BUY else Side.SHORT

        if order.reduce_only:
            pos = self.position_manager.get_position(order.symbol)
            record = self.position_manager.close_position(
                symbol=order.symbol,
                price=order.quantity * 100 if pos.entry_price == 0 else pos.entry_price,
                reason=order.reason,
            )
            pnl = record.pnl if record else 0.0
        else:
            self.position_manager.open_position(
                symbol=order.symbol,
                side=side,
                quantity=order.quantity,
                price=order.quantity * 100,
                reason=order.reason,
            )
            pnl = 0.0

        ack = OrderAck(
            order_id=f"SIM-{self._order_id:06d}",
            symbol=order.symbol,
            status="FILLED",
            executed_qty=order.quantity,
            message=f"simulated fill, pnl={pnl}",
        )
        logger.info(f"模拟下单: {ack}")
        return ack

    async def get_position(self, symbol: str) -> dict[str, Any]:
        pos = self.position_manager.get_position(symbol)
        return pos.to_dict()


class BinanceKlinePollingStream(RealTimeMarketStream):
    def __init__(
        self,
        client: AsyncClient,
        symbol: str,
        interval: str = "30m",
        poll_seconds: float = 5.0,
    ):
        self.client = client
        self.symbol = symbol
        self.interval = self._get_binance_interval(interval)
        self.poll_seconds = poll_seconds
        self._last_close_time_ms: int | None = None
        self._kline_count = 0

    def _get_binance_interval(self, interval: str) -> str:
        if interval in INTERVAL_MAP:
            return INTERVAL_MAP[interval]
        if interval in INTERVAL_MAP.values():
            return interval
        logger.warning(f"未知的interval: {interval}, 使用默认值 30m")
        return Client.KLINE_INTERVAL_30MINUTE

    async def events(self) -> AsyncIterator[MarketEvent]:
        logger.info(f"[{self.symbol}] 开始监听K线, 周期: {self.interval}")
        while True:
            try:
                klines = await self.client.futures_klines(
                    symbol=self.symbol, interval=self.interval, limit=2
                )
                if not klines:
                    continue

                last = klines[-1]
                close_time_ms = int(last[6])
                now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
                is_closed = close_time_ms <= now_ms

                if is_closed and close_time_ms != self._last_close_time_ms:
                    self._last_close_time_ms = close_time_ms
                    self._kline_count += 1
                    close_price = float(last[4])
                    event_time = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)

                    logger.info(
                        f"[{self.symbol}] K线#{self._kline_count} 收盘 @ {close_price} | "
                        f"时间: {event_time.strftime('%H:%M:%S')}"
                    )

                    yield MarketEvent(
                        symbol=self.symbol,
                        price=close_price,
                        timestamp=event_time,
                        extra={
                            "kline": last,
                            "open": float(last[1]),
                            "high": float(last[2]),
                            "low": float(last[3]),
                            "volume": float(last[5]),
                        },
                    )
                else:
                    next_close = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
                    remaining = (close_time_ms - now_ms) / 1000
                    if remaining > 0 and remaining < 60:
                        logger.debug(f"[{self.symbol}] 距离下次收盘: {remaining:.0f}秒")

            except Exception as e:
                logger.error(f"获取K线数据失败: {e}")

            await asyncio.sleep(self.poll_seconds)


class BinanceFuturesOrderGateway(OrderGateway):
    def __init__(self, client: AsyncClient, symbol: str):
        self.client = client
        self.symbol = symbol
        self._step_size: Decimal | None = None
        self._min_qty: Decimal | None = None
        self._price_precision: int = 2

    async def _load_symbol_filters(self) -> None:
        if self._step_size is not None:
            return

        info = await self.client.futures_exchange_info()
        for item in info.get("symbols", []):
            if item.get("symbol") != self.symbol:
                continue
            self._price_precision = item.get("pricePrecision", 2)
            for f in item.get("filters", []):
                if f.get("filterType") == "LOT_SIZE":
                    self._step_size = Decimal(str(f["stepSize"]))
                    self._min_qty = Decimal(str(f["minQty"]))
                    return
        raise ValueError(f"未找到 {self.symbol} 的 LOT_SIZE 规则")

    def _format_qty(self, qty: float) -> str:
        if self._step_size is None or self._min_qty is None:
            raise ValueError("symbol filters 未初始化")

        raw = Decimal(str(qty))
        quantized = raw.quantize(self._step_size, rounding=ROUND_DOWN)
        if quantized < self._min_qty:
            raise ValueError(f"下单量过小: {quantized} < minQty({self._min_qty})")
        return format(quantized.normalize(), "f")

    async def send_order(self, order: OrderCommand) -> OrderAck:
        await self._load_symbol_filters()
        try:
            qty = self._format_qty(order.quantity)
        except ValueError as e:
            logger.warning(f"下单量格式化失败: {e}")
            return OrderAck(order_id="", symbol=order.symbol, status="FAILED", message=str(e))

        params: dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side,
            "type": order.order_type,
            "quantity": qty,
        }
        if order.reduce_only:
            params["reduceOnly"] = "true"

        try:
            result = await self.client.futures_create_order(**params)
            ack = OrderAck(
                order_id=str(result.get("orderId", "")),
                symbol=result.get("symbol", order.symbol),
                status=result.get("status", "UNKNOWN"),
                executed_qty=float(result.get("executedQty", 0)),
                avg_price=float(result.get("avgPrice", 0)) if result.get("avgPrice") else 0.0,
                message=order.reason,
            )
            logger.info(f"实盘下单成功: {ack}")
            return ack
        except Exception as e:
            logger.error(f"实盘下单失败: {e}")
            return OrderAck(order_id="", symbol=order.symbol, status="FAILED", message=str(e))

    async def get_position(self, symbol: str) -> dict[str, Any]:
        try:
            positions = await self.client.futures_position_information(symbol=symbol)
            if positions:
                pos = positions[0]
                return {
                    "symbol": pos.get("symbol"),
                    "side": "LONG" if float(pos.get("positionAmt", 0)) > 0 else "SHORT" if float(pos.get("positionAmt", 0)) < 0 else "EMPTY",
                    "quantity": abs(float(pos.get("positionAmt", 0))),
                    "entry_price": float(pos.get("entryPrice", 0)),
                    "unrealized_pnl": float(pos.get("unRealizedProfit", 0)),
                    "leverage": int(pos.get("leverage", 1)),
                }
        except Exception as e:
            logger.error(f"获取仓位失败: {e}")
        return {}


class MACD30mCTAStrategy:
    HIST_FILTER = 0.5
    FAST_PERIOD = 12
    SLOW_PERIOD = 26
    SIGNAL_PERIOD = 9
    MIN_DATA_COUNT = 35
    HISTORY_LIMIT = 300

    def __init__(
        self,
        symbol: str,
        position_manager: LivePositionManager,
        risk_usdt: float = 100.0,
        hist_filter: float = 0.5,
    ):
        self.symbol = symbol
        self.position_manager = position_manager
        self.risk_usdt = risk_usdt
        self.hist_filter = hist_filter
        self._prices: list[float] = []
        self._event_count = 0
        self._ready = False

    async def load_history_data(self, client: AsyncClient, interval: str, limit: int = 300) -> int:
        logger.info(f"[{self.symbol}] 正在加载历史K线数据 (limit={limit})...")
        try:
            binance_interval = self._get_binance_interval(interval) if hasattr(self, '_get_binance_interval') else INTERVAL_MAP.get(interval, interval)
            
            klines = await client.futures_klines(
                symbol=self.symbol,
                interval=binance_interval,
                limit=limit,
            )
            if not klines:
                logger.warning(f"[{self.symbol}] 未获取到历史数据")
                return 0

            self._prices = [float(k[4]) for k in klines]
            self._ready = len(self._prices) >= self.MIN_DATA_COUNT

            if self._ready:
                series = pd.Series(self._prices[-200:])
                macd, signal, hist = self._build_macd(series)
                now_macd = float(macd.iloc[-1])
                now_signal = float(signal.iloc[-1])
                now_hist = float(hist.iloc[-1])
                last_hist = float(hist.iloc[-2])
                last_price = self._prices[-1]

                logger.info(
                    f"[{self.symbol}] 历史数据加载完成: {len(self._prices)}条 | "
                    f"最新价格: {last_price:.2f}"
                )
                logger.info(
                    f"[{self.symbol}] 当前指标状态: MACD={now_macd:.2f} | "
                    f"Signal={now_signal:.2f} | HIST={now_hist:.2f} (上一根: {last_hist:.2f})"
                )

                if abs(last_hist) >= self.hist_filter:
                    if last_hist < 0 < now_hist:
                        logger.info(f"[{self.symbol}] ⚠️ 历史数据已出现金叉信号，等待下一根K线确认")
                    elif now_hist < 0 < last_hist:
                        logger.info(f"[{self.symbol}] ⚠️ 历史数据已出现死叉信号，等待下一根K线确认")
            else:
                logger.warning(f"[{self.symbol}] 历史数据不足: {len(self._prices)}/{self.MIN_DATA_COUNT}")

            return len(self._prices)
        except Exception as e:
            logger.error(f"[{self.symbol}] 加载历史数据失败: {e}")
            return 0

    @property
    def is_ready(self) -> bool:
        return self._ready

    def get_current_indicators(self) -> dict[str, Any]:
        if not self._ready or len(self._prices) < 35:
            return {"ready": False}
        
        series = pd.Series(self._prices[-200:])
        macd, signal, hist = self._build_macd(series)
        
        return {
            "ready": True,
            "macd": float(macd.iloc[-1]),
            "signal": float(signal.iloc[-1]),
            "hist": float(hist.iloc[-1]),
            "last_hist": float(hist.iloc[-2]),
            "price": self._prices[-1] if self._prices else 0,
            "data_count": len(self._prices),
        }

    def _build_macd(self, prices: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        fast_ema = prices.ewm(span=self.FAST_PERIOD, adjust=False).mean()
        slow_ema = prices.ewm(span=self.SLOW_PERIOD, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=self.SIGNAL_PERIOD, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def _calc_qty(self, price: float, leverage: int = 5) -> float:
        qty = self.risk_usdt * leverage / price
        return math.floor(qty * 1000) / 1000

    async def on_market_event(self, event: MarketEvent) -> OrderCommand | None:
        self._event_count += 1
        current_price = event.price

        self._prices.append(current_price)
        if len(self._prices) > 500:
            self._prices = self._prices[-500:]

        self.position_manager.update_pnl({self.symbol: current_price})
        pos = self.position_manager.get_position(self.symbol)

        if not self._ready:
            self._ready = len(self._prices) >= self.MIN_DATA_COUNT
            if not self._ready:
                logger.info(f"[{self.symbol}] 数据收集中: {len(self._prices)}/{self.MIN_DATA_COUNT}")
                return None
            logger.info(f"[{self.symbol}] 数据收集完成，策略已就绪")

        series = pd.Series(self._prices[-200:])
        macd, signal, hist = self._build_macd(series)

        last_hist = float(hist.iloc[-2])
        now_hist = float(hist.iloc[-1])
        now_macd = float(macd.iloc[-1])
        now_signal = float(signal.iloc[-1])

        logger.info(
            f"[{self.symbol}] MACD: {now_macd:.2f} | Signal: {now_signal:.2f} | "
            f"HIST: {now_hist:.2f} (上一根: {last_hist:.2f})"
        )

        if abs(last_hist) < self.hist_filter:
            logger.debug(f"[{self.symbol}] HIST过滤: |{last_hist:.2f}| < {self.hist_filter}")
            return None

        if last_hist < 0 < now_hist and not pos.is_open:
            qty = self._calc_qty(current_price)
            if qty <= 0:
                return None
            logger.info(f"[{self.symbol}] 金叉开多信号 @ {current_price:.2f} | 数量: {qty}")
            return OrderCommand(
                symbol=self.symbol,
                side=SIDE_BUY,
                quantity=qty,
                reason="MACD_GOLDEN_CROSS",
                reduce_only=False,
            )

        if now_hist < 0 < last_hist and pos.is_open and pos.side == Side.LONG:
            logger.info(f"[{self.symbol}] 死叉平多信号 @ {current_price:.2f}")
            return OrderCommand(
                symbol=self.symbol,
                side=SIDE_SELL,
                quantity=pos.quantity,
                reason="MACD_DEAD_CROSS",
                reduce_only=True,
            )

        return None

    def on_fill(self, ack: OrderAck) -> None:
        if ack.status != "FILLED":
            logger.warning(f"订单未成交: {ack}")
            return

        if "MACD_DEAD_CROSS" in ack.message:
            self.position_manager.close_position(
                symbol=self.symbol,
                price=ack.avg_price if ack.avg_price > 0 else 0,
                reason=ack.message,
            )
            logger.info(f"[{self.symbol}] 死叉平仓成交 @ {ack.avg_price}")
        elif "MACD_GOLDEN_CROSS" in ack.message:
            self.position_manager.open_position(
                symbol=self.symbol,
                side=Side.LONG,
                quantity=ack.executed_qty,
                price=ack.avg_price,
                reason=ack.message,
            )
            logger.info(f"[{self.symbol}] 金叉开仓成交 @ {ack.avg_price}")


class LiveTradeEngine:
    STATUS_INTERVAL = 600

    def __init__(
        self,
        market_stream: RealTimeMarketStream,
        strategy: MACD30mCTAStrategy,
        order_gateway: OrderGateway,
        position_manager: LivePositionManager,
        interval: str = "30m",
    ):
        self.market_stream = market_stream
        self.strategy = strategy
        self.order_gateway = order_gateway
        self.position_manager = position_manager
        self.interval = interval
        self._running = False
        self._on_trade_callback: Callable[[TradeRecord], None] | None = None
        self._last_status_time: datetime | None = None
        self._last_kline_count: int = 0
        self._status_task: asyncio.Task | None = None
        self._current_price: float = 0.0

    def set_trade_callback(self, callback: Callable[[TradeRecord], None]) -> None:
        self._on_trade_callback = callback
        self.position_manager.set_order_callback(callback)

    async def _status_monitor(self) -> None:
        await asyncio.sleep(self.STATUS_INTERVAL)
        while self._running:
            try:
                if self._last_status_time is not None:
                    elapsed = datetime.now() - self._last_status_time
                    if elapsed.total_seconds() >= self.STATUS_INTERVAL:
                        self._periodic_status()
                        self._last_status_time = datetime.now()
            except Exception as e:
                logger.warning(f"状态监控异常: {e}")
            await asyncio.sleep(60)

    def _periodic_status(self) -> None:
        if not self.strategy.is_ready:
            return

        stats = self.position_manager.get_statistics()
        pos = self.position_manager.get_position(self.strategy.symbol)
        indicators = self.strategy.get_current_indicators()
        
        remaining = KlineTimeValidator.get_remaining_seconds(self.interval)
        next_close = KlineTimeValidator.get_next_close_time(self.interval)

        lines = [
            "",
            "=" * 60,
            f"[状态报告] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            f"交易对: {self.strategy.symbol} | 周期: {self.interval}",
            "-" * 60,
        ]

        if pos.is_open:
            lines.extend([
                f"当前持仓: {pos.side.value} {pos.quantity:.6f}",
                f"入场价格: {pos.entry_price:.2f} USDT",
                f"未实现盈亏: {pos.unrealized_pnl:.4f} USDT",
                f"持仓时长: {(datetime.now() - pos.entry_time).seconds // 60 if pos.entry_time else 0} 分钟" if pos.entry_time else "持仓时长: -",
            ])
        else:
            lines.append("当前持仓: 空仓")

        lines.extend([
            "-" * 60,
            f"当前价格: {indicators.get('price', 0):.2f} USDT",
            f"MACD: {indicators.get('macd', 0):.4f}",
            f"Signal: {indicators.get('signal', 0):.4f}",
            f"HIST: {indicators.get('hist', 0):.4f} (上一根: {indicators.get('last_hist', 0):.4f})",
            "-" * 60,
            f"账户余额: {stats['current_balance']:.2f} USDT",
            f"总权益: {stats['total_equity']:.2f} USDT",
            f"累计盈亏: {stats['total_pnl']:.4f} USDT",
            f"收益率: {stats['return_pct']:.2f}%",
            "-" * 60,
            f"开仓次数: {stats['total_trades']}",
            f"盈利次数: {stats['win_trades']} | 亏损次数: {stats['lose_trades']}",
            f"胜率: {stats['win_rate']:.1f}%",
            "-" * 60,
            f"下次K线收盘: {next_close.strftime('%H:%M:%S')} UTC ({remaining // 60}分{remaining % 60}秒后)",
            f"数据量: {indicators.get('data_count', 0)} 条",
            "=" * 60,
        ])

        logger.info("\n".join(lines))

    async def run(self) -> None:
        self._running = True
        self._last_status_time = datetime.now()
        logger.info("=" * 60)
        logger.info("实盘交易引擎启动")
        logger.info("=" * 60)

        self._status_task = asyncio.create_task(self._status_monitor())

        try:
            async for event in self.market_stream.events():
                if not self._running:
                    break

                order = await self.strategy.on_market_event(event)
                if order is None:
                    continue

                ack = await self.order_gateway.send_order(order)
                if ack.status == "FILLED":
                    self.strategy.on_fill(ack)

                self._log_status()

        except asyncio.CancelledError:
            logger.info("实盘引擎收到停止信号")
        except Exception as e:
            logger.error(f"实盘引擎异常: {e}")
            raise
        finally:
            if self._status_task:
                self._status_task.cancel()

    def stop(self) -> None:
        self._running = False
        logger.info("实盘引擎停止中...")

    def _log_status(self) -> None:
        stats = self.position_manager.get_statistics()
        logger.info(
            f"账户状态: 余额={stats['current_balance']:.2f} USDT | "
            f"未实现PnL={stats['total_unrealized_pnl']:.4f} | "
            f"持仓数={len(self.position_manager.open_positions)}"
        )


async def run_live_trading(
    api_key: str,
    api_secret: str,
    symbol: str = "BTCUSDT",
    interval: str = "30m",
    initial_balance: float = 1000.0,
    risk_usdt: float = 100.0,
    hist_filter: float = 0.5,
    testnet: bool = True,
    dry_run: bool = False,
) -> None:
    risk_config = RiskConfig(
        max_position_ratio=0.3,
        default_leverage=5,
    )
    position_manager = LivePositionManager(
        initial_balance=initial_balance,
        risk_config=risk_config,
    )

    logger.info("正在连接币安API...")
    client = await AsyncClient.create(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )
    logger.info("币安API连接成功")

    stream = BinanceKlinePollingStream(client=client, symbol=symbol, interval=interval)
    strategy = MACD30mCTAStrategy(
        symbol=symbol,
        position_manager=position_manager,
        risk_usdt=risk_usdt,
        hist_filter=hist_filter,
    )

    await strategy.load_history_data(client=client, interval=interval, limit=MACD30mCTAStrategy.HISTORY_LIMIT)

    if dry_run:
        gateway = InMemoryOrderGateway(position_manager=position_manager)
        logger.info("=== 模拟运行模式 ===")
    else:
        gateway = BinanceFuturesOrderGateway(client=client, symbol=symbol)
        logger.info("=== 实盘运行模式 ===")

    engine = LiveTradeEngine(
        market_stream=stream,
        strategy=strategy,
        order_gateway=gateway,
        position_manager=position_manager,
        interval=interval,
    )

    def on_trade(record: TradeRecord) -> None:
        logger.info(f"交易记录: {record.symbol} {record.side.value} {record.quantity} @ {record.price}")

    engine.set_trade_callback(on_trade)

    logger.info(f"策略参数: HIST过滤={hist_filter}")

    try:
        await engine.run()
    finally:
        await client.close_connection()
        print(position_manager.summary())


async def main() -> None:
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key or not api_secret:
        raise ValueError("请设置环境变量 BINANCE_API_KEY 和 BINANCE_API_SECRET")

    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    dry_run = os.getenv("BINANCE_DRY_RUN", "true").lower() == "true"

    await run_live_trading(
        api_key=api_key,
        api_secret=api_secret,
        symbol=os.getenv("BINANCE_SYMBOL", "BTCUSDT"),
        interval=os.getenv("BINANCE_INTERVAL", "30m"),
        initial_balance=float(os.getenv("BINANCE_INITIAL_BALANCE", "1000")),
        risk_usdt=float(os.getenv("BINANCE_RISK_USDT", "100")),
        hist_filter=float(os.getenv("BINANCE_HIST_FILTER", "0.5")),
        testnet=testnet,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    asyncio.run(main())
