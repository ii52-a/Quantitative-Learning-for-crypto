from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Protocol

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
            side = "BUY"
            reduce_only = False
        elif signal == PositionSignal.SHORT:
            side = "SELL"
            reduce_only = False
        elif signal == PositionSignal.FULL:
            side = "SELL"
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
            strategy_name="StrategyResultAdapter",
            reason=result.comment,
            reduce_only=reduce_only,
        )
