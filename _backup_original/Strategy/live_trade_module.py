from __future__ import annotations

import asyncio
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, AsyncIterator, Protocol

import pandas as pd
from binance import AsyncClient
from binance.enums import SIDE_BUY, SIDE_SELL

from Strategy.StrategyTypes import PositionSignal, StrategyResult
from app_logger.logger_setup import Logger

logger = Logger(__name__)


@dataclass(slots=True)
class MarketEvent:
    """实时行情事件。"""

    symbol: str
    price: float
    timestamp: datetime
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderCommand:
    """统一下单指令。"""

    symbol: str
    side: str
    quantity: float
    order_type: str = "MARKET"
    strategy_name: str = ""
    reason: str = ""
    reduce_only: bool = False


@dataclass(slots=True)
class OrderAck:
    """下单回执。"""

    order_id: str
    symbol: str
    status: str
    message: str = ""


class RealTimeMarketStream(ABC):
    """行情流接口，方便接入 websocket / kafka / redis stream 等。"""

    @abstractmethod
    async def events(self) -> AsyncIterator[MarketEvent]:
        raise NotImplementedError


class StrategySignalProcessor(Protocol):
    """策略处理接口。"""

    async def on_market_event(self, event: MarketEvent) -> StrategyResult | None:
        ...


class OrderGateway(ABC):
    """交易网关接口。"""

    @abstractmethod
    async def send_order(self, order: OrderCommand) -> OrderAck:
        raise NotImplementedError


class InMemoryOrderGateway(OrderGateway):
    """演示网关：仅记录下单，不触发真实交易。"""

    def __init__(self):
        self.commands: list[OrderCommand] = []

    async def send_order(self, order: OrderCommand) -> OrderAck:
        self.commands.append(order)
        ack = OrderAck(
            order_id=f"SIM-{len(self.commands):06d}",
            symbol=order.symbol,
            status="FILLED",
            message="simulated fill",
        )
        logger.info(f"模拟下单成功: {ack}")
        return ack


class QueueMarketStream(RealTimeMarketStream):
    """基于 asyncio.Queue 的实时行情流，便于从其他协程推送行情。"""

    def __init__(self, queue: asyncio.Queue[MarketEvent], stop_event: asyncio.Event | None = None):
        self._queue = queue
        self._stop_event = stop_event or asyncio.Event()

    async def events(self) -> AsyncIterator[MarketEvent]:
        while not self._stop_event.is_set():
            event = await self._queue.get()
            yield event

    def stop(self) -> None:
        self._stop_event.set()


class BinanceKlinePollingStream(RealTimeMarketStream):
    """
    基于 Binance Futures REST 的轮询行情流。

    设计说明：
    1) 该实现只在 K 线“收盘”后发出事件，避免未收盘K线造成信号反复。
    2) 轮询实现比 websocket 更易维护，便于先跑稳定版本；后续可替换为 websocket 流。
    """

    def __init__(
        self,
        client: AsyncClient,
        symbol: str,
        interval: str = "1m",
        poll_seconds: float = 1.5,
    ):
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.poll_seconds = poll_seconds
        self._last_close_time_ms: int | None = None

    async def events(self) -> AsyncIterator[MarketEvent]:
        while True:
            klines = await self.client.futures_klines(symbol=self.symbol, interval=self.interval, limit=2)
            last = klines[-1]
            is_closed = bool(last[6] <= int(datetime.now(tz=timezone.utc).timestamp() * 1000))
            close_time_ms = int(last[6])

            # 只推送新收盘K线。
            if is_closed and close_time_ms != self._last_close_time_ms:
                self._last_close_time_ms = close_time_ms
                close_price = float(last[4])
                event_time = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
                yield MarketEvent(
                    symbol=self.symbol,
                    price=close_price,
                    timestamp=event_time,
                    extra={"kline": last},
                )

            await asyncio.sleep(self.poll_seconds)


class SimpleMACDLiveStrategy(StrategySignalProcessor):
    """
    与回测逻辑风格一致的“MACD金叉开多/死叉平仓”实时策略。

    注意：
    - 这是基础可用版，不包含止损、手续费模型、滑点模型。
    - 用 `quote_risk_usdt` 控制单次下单风险资金，再根据价格换算为下单数量。
    """

    def __init__(
        self,
        symbol: str,
        interval: str,
        quote_risk_usdt: float,
        quantity_precision: int = 3,
    ):
        self.symbol = symbol
        self.interval = interval
        self.quote_risk_usdt = quote_risk_usdt
        self.quantity_precision = quantity_precision
        self._hist: list[float] = []
        self._in_position = False

    @staticmethod
    def _build_macd_hist(prices: pd.Series) -> pd.Series:
        fast_ema = prices.ewm(span=12, adjust=False).mean()
        slow_ema = prices.ewm(span=26, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal

    def _calc_qty(self, price: float) -> float:
        qty = self.quote_risk_usdt / price
        factor = 10**self.quantity_precision
        return math.floor(qty * factor) / factor

    async def on_market_event(self, event: MarketEvent) -> StrategyResult | None:
        kline = event.extra.get("kline")
        if kline is None:
            return None

        close_price = float(kline[4])
        self._hist.append(close_price)
        if len(self._hist) < 35:
            return None

        series = pd.Series(self._hist[-200:])
        hist = self._build_macd_hist(series)
        last_hist = float(hist.iloc[-2])
        now_hist = float(hist.iloc[-1])

        # 金叉：开多
        if last_hist < 0 < now_hist and not self._in_position:
            self._in_position = True
            qty = self._calc_qty(close_price)
            if qty <= 0:
                logger.warning(f"{self.symbol} 计算下单数量<=0，忽略信号")
                return None
            return StrategyResult(
                symbol=self.symbol,
                size=qty,
                execution_price=close_price,
                execution_time=event.timestamp,
                direction=PositionSignal.LONG,
                comment="MACD golden cross",
            )

        # 死叉：平仓
        if now_hist < 0 < last_hist and self._in_position:
            self._in_position = False
            qty = self._calc_qty(close_price)
            if qty <= 0:
                return None
            return StrategyResult(
                symbol=self.symbol,
                size=qty,
                execution_price=close_price,
                execution_time=event.timestamp,
                direction=PositionSignal.FULL,
                comment="MACD dead cross",
            )

        return None


class BinanceFuturesOrderGateway(OrderGateway):
    """
    币安U本位合约下单网关。

    关键点：
    - 使用 `reduceOnly` 标记平仓单，避免误加仓。
    - 使用交易所 lot size 规则进行下单量截断，避免精度报错。
    """

    def __init__(self, client: AsyncClient, symbol: str):
        self.client = client
        self.symbol = symbol
        self._step_size: Decimal | None = None
        self._min_qty: Decimal | None = None

    async def _load_symbol_filters(self) -> None:
        if self._step_size is not None and self._min_qty is not None:
            return

        info = await self.client.futures_exchange_info()
        for item in info.get("symbols", []):
            if item.get("symbol") != self.symbol:
                continue
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
        qty = self._format_qty(order.quantity)

        params: dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side,
            "type": order.order_type,
            "quantity": qty,
        }
        if order.reduce_only:
            params["reduceOnly"] = "true"

        result = await self.client.futures_create_order(**params)
        ack = OrderAck(
            order_id=str(result.get("orderId", "")),
            symbol=result.get("symbol", order.symbol),
            status=result.get("status", "UNKNOWN"),
            message=order.reason,
        )
        logger.info(f"实盘下单回执: {ack}")
        return ack


class LiveTradeEngine:
    """实时策略执行引擎：行情 -> 策略 -> 下单。"""

    def __init__(self, market_stream: RealTimeMarketStream, strategy: StrategySignalProcessor, order_gateway: OrderGateway):
        self.market_stream = market_stream
        self.strategy = strategy
        self.order_gateway = order_gateway

    async def run(self) -> None:
        async for event in self.market_stream.events():
            logger.debug(f"接收行情: {event.symbol} {event.price} @ {event.timestamp}")
            strategy_result = await self.strategy.on_market_event(event)
            if strategy_result is None:
                continue

            order = self._convert_to_order(strategy_result)
            if order is None:
                continue

            await self.order_gateway.send_order(order)

    @staticmethod
    def _convert_to_order(result: StrategyResult) -> OrderCommand | None:
        if result.execution_price is None or result.execution_time is None:
            logger.warning("策略信号缺失 execution 字段，跳过下单")
            return None

        signal = result.direction
        if signal == PositionSignal.LONG:
            side = SIDE_BUY
            reduce_only = False
        elif signal == PositionSignal.SHORT:
            side = SIDE_SELL
            reduce_only = False
        elif signal == PositionSignal.FULL:
            # 这里默认用于平多仓。若做双向持仓模式可继续细分。
            side = SIDE_SELL
            reduce_only = True
        else:
            logger.debug(f"收到不处理信号: {signal}")
            return None

        quantity = abs(result.size)
        if quantity <= 0:
            logger.warning("策略下单数量<=0，跳过下单")
            return None

        return OrderCommand(
            symbol=result.symbol,
            side=side,
            quantity=quantity,
            strategy_name="SimpleMACDLiveStrategy",
            reason=result.comment,
            reduce_only=reduce_only,
        )


async def run_binance_live_demo() -> None:
    """
    可直接运行的实盘入口（默认 testnet）。

    使用方式：
    1) 设置环境变量：
       - BINANCE_API_KEY
       - BINANCE_API_SECRET
       - BINANCE_TESTNET=true/false
    2) 根据需要修改 symbol / interval / 风险资金。
    """

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        raise ValueError("缺少 BINANCE_API_KEY / BINANCE_API_SECRET 环境变量")

    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    client = await AsyncClient.create(api_key=api_key, api_secret=api_secret, testnet=testnet)

    symbol = os.getenv("BINANCE_SYMBOL", "BTCUSDT")
    interval = os.getenv("BINANCE_INTERVAL", "1m")
    risk_usdt = float(os.getenv("BINANCE_RISK_USDT", "50"))

    stream = BinanceKlinePollingStream(client=client, symbol=symbol, interval=interval)
    strategy = SimpleMACDLiveStrategy(symbol=symbol, interval=interval, quote_risk_usdt=risk_usdt)
    gateway = BinanceFuturesOrderGateway(client=client, symbol=symbol)
    engine = LiveTradeEngine(market_stream=stream, strategy=strategy, order_gateway=gateway)

    try:
        logger.info(f"启动实盘引擎: {symbol} {interval}, risk={risk_usdt}")
        await engine.run()
    finally:
        await client.close_connection()


if __name__ == "__main__":
    asyncio.run(run_binance_live_demo())
