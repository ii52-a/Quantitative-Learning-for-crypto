"""实盘交易核心模块"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from enum import Enum
import threading
import time
import hashlib
import hmac
import requests

import pandas as pd

from app_logger.logger_setup import Logger
from core.constants import PositionSide, SignalType
from Strategy.base import Bar, Position as StrategyPosition, StrategyContext

logger = Logger(__name__)

BINANCE_SPOT_API = "https://api.binance.com"
BINANCE_FUTURES_API = "https://fapi.binance.com"
BINANCE_TESTNET_FUTURES_API = "https://testnet.binancefuture.com"


class TradingMode(Enum):
    """交易模式"""
    TEST = "test"
    LIVE = "live"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """订单"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: str
    quantity: float
    price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    create_time: datetime = field(default_factory=datetime.now)
    update_time: datetime = field(default_factory=datetime.now)
    commission: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "create_time": self.create_time.isoformat(),
            "update_time": self.update_time.isoformat(),
            "commission": self.commission,
        }


@dataclass
class Position:
    """持仓"""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    leverage: int = 1
    margin: float = 0.0
    
    def update_price(self, current_price: float):
        """更新当前价格和未实现盈亏"""
        self.current_price = current_price
        if self.side == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "leverage": self.leverage,
            "margin": self.margin,
        }


@dataclass
class TradingAccount:
    """交易账户"""
    balance: float = 0.0
    available: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin_used: float = 0.0
    margin_ratio: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "balance": self.balance,
            "available": self.available,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "margin_used": self.margin_used,
            "margin_ratio": self.margin_ratio,
        }


@dataclass
class TradingConfig:
    """交易配置"""
    mode: TradingMode = TradingMode.TEST
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    symbol: str = "BTCUSDT"
    interval: str = "30m"
    leverage: int = 5
    position_size: float = 0.1
    max_positions: int = 1
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 10.0
    max_daily_loss: float = 20.0
    max_daily_trades: int = 10


class LiveTrader:
    """实盘交易器"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self._account = TradingAccount()
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._trade_history: list[dict] = []
        
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        
        self._strategy = None
        self._data_callback: Callable | None = None
        self._order_callback: Callable | None = None
        self._error_callback: Callable | None = None
        
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._last_reset = datetime.now().date()
    
    @property
    def account(self) -> TradingAccount:
        return self._account
    
    @property
    def positions(self) -> dict[str, Position]:
        return self._positions
    
    @property
    def orders(self) -> dict[str, Order]:
        return self._orders
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def set_strategy(self, strategy) -> None:
        """设置策略"""
        self._strategy = strategy
    
    def set_callbacks(
        self,
        data_callback: Callable | None = None,
        order_callback: Callable | None = None,
        error_callback: Callable | None = None,
    ) -> None:
        """设置回调函数"""
        self._data_callback = data_callback
        self._order_callback = order_callback
        self._error_callback = error_callback
    
    def connect(self) -> bool:
        """连接交易所"""
        try:
            if self.config.mode == TradingMode.TEST:
                self._account.balance = 10000
                self._account.available = 10000
                logger.info("测试模式: 初始化模拟账户")
                return True
            
            if not self.config.api_key or not self.config.api_secret:
                logger.error("实盘模式需要API密钥")
                return False
            
            self._base_url = BINANCE_TESTNET_FUTURES_API if self.config.testnet else BINANCE_FUTURES_API
            
            account_info = self._get_account_info()
            
            if account_info:
                self._account.balance = float(account_info.get('totalWalletBalance', 0))
                self._account.available = float(account_info.get('availableBalance', 0))
                self._account.unrealized_pnl = float(account_info.get('totalUnrealizedProfit', 0))
                self._account.margin_used = float(account_info.get('totalInitialMargin', 0))
                self._account.margin_ratio = float(account_info.get('totalMaintMargin', 0)) / self._account.balance if self._account.balance > 0 else 0
                
                assets = account_info.get('assets', [])
                usdt_asset = next((a for a in assets if a.get('asset') == 'USDT'), None)
                if usdt_asset:
                    self._account.realized_pnl = float(usdt_asset.get('walletBalance', 0))
                
                self._update_positions()
                
                logger.info(f"实盘模式: 连接成功，余额={self._account.balance:.2f} USDT，可用={self._account.available:.2f} USDT，持仓={len(self._positions)}个")
                return True
            else:
                logger.error("获取账户信息失败")
                return False
            
        except Exception as e:
            logger.error(f"连接失败: {e}")
            if self._error_callback:
                self._error_callback(str(e))
            return False
    
    def _get_server_time(self) -> int:
        """获取服务器时间"""
        try:
            response = requests.get(f"{self._base_url}/fapi/v1/time", timeout=10)
            return response.json()['serverTime']
        except:
            return int(time.time() * 1000)
    
    def _sign_request(self, params: dict) -> str:
        """签名请求"""
        query_string = '&'.join(f"{k}={v}" for k, v in params.items())
        signature = hmac.new(
            self.config.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _get_account_info(self) -> dict | None:
        """获取账户信息"""
        try:
            params = {
                'timestamp': self._get_server_time(),
                'recvWindow': 5000
            }
            params['signature'] = self._sign_request(params)
            
            headers = {
                'X-MBX-APIKEY': self.config.api_key
            }
            
            response = requests.get(
                f"{self._base_url}/fapi/v2/account",
                params=params,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API错误: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
            return None
    
    def _get_positions(self) -> list:
        """获取当前持仓"""
        try:
            params = {
                'timestamp': self._get_server_time(),
                'recvWindow': 5000
            }
            params['signature'] = self._sign_request(params)
            
            headers = {
                'X-MBX-APIKEY': self.config.api_key
            }
            
            response = requests.get(
                f"{self._base_url}/fapi/v2/positionRisk",
                params=params,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                positions = response.json()
                return [p for p in positions if float(p.get('positionAmt', 0)) != 0]
            return []
                
        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
            return []
    
    def disconnect(self) -> None:
        """断开连接"""
        self.stop()
        logger.info("已断开交易所连接")
    
    def start(self) -> bool:
        """启动交易"""
        if self._running:
            return True
        
        if not self.connect():
            return False
        
        self._running = True
        self._stop_event.clear()
        
        self._thread = threading.Thread(target=self._trading_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"交易已启动 (模式: {self.config.mode.value})")
        return True
    
    def stop(self) -> None:
        """停止交易"""
        self._running = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        logger.info("交易已停止")
    
    def _trading_loop(self) -> None:
        """交易循环"""
        while self._running and not self._stop_event.is_set():
            try:
                self._check_daily_reset()
                
                if self._strategy:
                    self._execute_strategy()
                
                self._update_positions()
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"交易循环错误: {e}")
                if self._error_callback:
                    self._error_callback(str(e))
    
    def _check_daily_reset(self) -> None:
        """检查每日重置"""
        today = datetime.now().date()
        if today != self._last_reset:
            self._daily_trades = 0
            self._daily_pnl = 0.0
            self._last_reset = today
            logger.info("每日统计已重置")
    
    def _execute_strategy(self) -> None:
        """执行策略"""
        if not self._strategy:
            return
        
        try:
            strategy_params = self._strategy.get_parameters() if hasattr(self._strategy, "get_parameters") else {}
            auto_select = str(strategy_params.get("auto_select_symbol", "true")).lower() == "true"
            if self._strategy.__class__.__name__ == "OrderFlowPullbackStrategy" and auto_select:
                selected_symbol = self._pick_volatile_symbol()
                if selected_symbol and selected_symbol != self.config.symbol:
                    self.config.symbol = selected_symbol
                    logger.info(f"订单流策略自动选币: {selected_symbol}")

            data = self._get_klines(self.config.symbol, limit=200)
            
            if data is None or len(data) < 50:
                return
            
            if hasattr(self._strategy, "on_bar"):
                self._execute_base_strategy(data)
                return

            signal = self._strategy.generate_signal(data)

            if signal is None:
                return

            current_position = self._positions.get(self.config.symbol)

            if signal == 1:
                if current_position and current_position.side == "short":
                    self._close_position(self.config.symbol)
                if not current_position or current_position.side != "long":
                    self._open_position(self.config.symbol, "long")

            elif signal == -1:
                if current_position and current_position.side == "long":
                    self._close_position(self.config.symbol)
                if not current_position or current_position.side != "short":
                    self._open_position(self.config.symbol, "short")
                    
        except Exception as e:
            logger.error(f"策略执行错误: {e}")
            if self._error_callback:
                self._error_callback(f"策略执行错误: {e}")

    def _execute_base_strategy(self, data: pd.DataFrame) -> None:
        """执行 BaseStrategy 接口策略"""
        strategy = self._strategy
        symbol = self.config.symbol

        if not hasattr(strategy, "_live_initialized"):
            strategy._live_initialized = False

        current_position = self._positions.get(symbol)
        if current_position is None:
            strategy_position = StrategyPosition(
                side=PositionSide.EMPTY,
                quantity=0.0,
                entry_price=0.0,
                entry_time=datetime.now(),
            )
        else:
            side_map = {"long": PositionSide.LONG, "short": PositionSide.SHORT}
            strategy_position = StrategyPosition(
                side=side_map.get(current_position.side, PositionSide.EMPTY),
                quantity=current_position.quantity,
                entry_price=current_position.entry_price,
                entry_time=datetime.now(),
                unrealized_pnl=current_position.unrealized_pnl,
            )

        row = data.iloc[-1]
        bar = Bar(
            timestamp=data.index[-1].to_pydatetime() if hasattr(data.index[-1], "to_pydatetime") else datetime.now(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            symbol=symbol,
            interval=self.config.interval,
        )

        context = StrategyContext(
            symbol=symbol,
            interval=self.config.interval,
            position=strategy_position,
            equity=self._account.balance + self._account.unrealized_pnl,
            available_capital=self._account.available,
            current_price=bar.close,
            timestamp=bar.timestamp,
            data_count=len(data),
        )

        if not strategy._live_initialized:
            strategy.initialize(context)
            strategy._live_initialized = True

        result = strategy.on_bar(bar, context)
        signal = result.signal if result else None
        if not signal:
            return

        if signal.type == SignalType.OPEN_LONG:
            if current_position and current_position.side == "short":
                self._close_position(symbol)
            size_factor = float(signal.extra.get("size_factor", 1.0)) if getattr(signal, "extra", None) else 1.0
            self._open_position(symbol, "long", size_factor=size_factor)
        elif signal.type == SignalType.OPEN_SHORT:
            if current_position and current_position.side == "long":
                self._close_position(symbol)
            size_factor = float(signal.extra.get("size_factor", 1.0)) if getattr(signal, "extra", None) else 1.0
            self._open_position(symbol, "short", size_factor=size_factor)
        elif signal.type == SignalType.CLOSE_LONG and current_position and current_position.side == "long":
            self._close_position(symbol)
        elif signal.type == SignalType.CLOSE_SHORT and current_position and current_position.side == "short":
            self._close_position(symbol)
    
    def _pick_volatile_symbol(self) -> str | None:
        """自动选择高波动USDT交易对"""
        try:
            if self.config.mode == TradingMode.TEST:
                return self.config.symbol

            response = requests.get(f"{self._base_url}/fapi/v1/ticker/24hr", timeout=10)
            if response.status_code != 200:
                return None

            tickers = response.json()
            candidates = []
            for t in tickers:
                symbol = t.get("symbol", "")
                if not (symbol.endswith("USDT") or symbol.endswith("USDC")):
                    continue
                if any(x in symbol for x in ("UP", "DOWN", "BULL", "BEAR")):
                    continue
                try:
                    change_pct = abs(float(t.get("priceChangePercent", 0)))
                    quote_volume = float(t.get("quoteVolume", 0))
                except Exception:
                    continue
                if quote_volume < 5_000_000:
                    continue
                candidates.append((change_pct, quote_volume, symbol))

            if not candidates:
                return None

            candidates.sort(reverse=True)
            return candidates[0][2]
        except Exception as e:
            logger.warning(f"自动选币失败: {e}")
            return None

    def _get_klines(self, symbol: str, limit: int = 200) -> pd.DataFrame | None:
        """获取K线数据"""
        try:
            if self.config.mode == TradingMode.TEST:
                import numpy as np
                np.random.seed(int(time.time()) % 10000)
                n = limit
                prices = [100000]
                for i in range(n - 1):
                    change = np.random.uniform(-0.02, 0.02)
                    prices.append(prices[-1] * (1 + change))
                
                dates = pd.date_range(end=datetime.now(), periods=n, freq='30min')
                return pd.DataFrame({
                    'open': prices,
                    'high': [p * 1.005 for p in prices],
                    'low': [p * 0.995 for p in prices],
                    'close': prices,
                    'volume': [1000] * n
                }, index=dates)
            
            params = {
                'symbol': symbol,
                'interval': self.config.interval,
                'limit': limit
            }
            
            response = requests.get(
                f"{self._base_url}/fapi/v1/klines",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                klines = response.json()
                data = pd.DataFrame(klines, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    data[col] = pd.to_numeric(data[col])
                
                data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
                data.set_index('open_time', inplace=True)
                
                return data[['open', 'high', 'low', 'close', 'volume']]
            
            return None
            
        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return None
    
    def _open_position(self, symbol: str, side: str, size_factor: float = 1.0) -> bool:
        """开仓"""
        try:
            account_balance = self._account.balance + self._account.unrealized_pnl
            step_factor = max(0.01, min(float(size_factor), 0.5))
            position_value = account_balance * self.config.position_size * step_factor
            leverage = self.config.leverage

            quantity = position_value * leverage / 100
            
            if quantity <= 0:
                return False
            
            if self.config.mode == TradingMode.TEST:
                current_price = 100000
                existing = self._positions.get(symbol)
                if existing and existing.side == side:
                    total_qty = existing.quantity + quantity
                    weighted_entry = (existing.entry_price * existing.quantity + current_price * quantity) / total_qty
                    existing.quantity = total_qty
                    existing.entry_price = weighted_entry
                    existing.current_price = current_price
                    existing.margin += position_value
                else:
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        entry_price=current_price,
                        current_price=current_price,
                        unrealized_pnl=0,
                        leverage=leverage,
                        margin=position_value,
                    )

                self._daily_trades += 1
                logger.info(f"测试模式: 开仓/加仓 {side} {symbol} 数量={quantity:.4f} 步长={step_factor*100:.1f}%")
                
                if self._order_callback:
                    self._order_callback({
                        'type': 'open',
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'price': current_price,
                    })
                return True
            
            order_side = "BUY" if side == "long" else "SELL"
            
            params = {
                'symbol': symbol,
                'side': order_side,
                'type': 'MARKET',
                'quantity': round(quantity, 3),
                'timestamp': self._get_server_time(),
                'recvWindow': 5000
            }
            params['signature'] = self._sign_request(params)
            
            headers = {'X-MBX-APIKEY': self.config.api_key}
            
            response = requests.post(
                f"{self._base_url}/fapi/v1/order",
                params=params,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                order = response.json()
                self._daily_trades += 1
                logger.info(f"开仓成功: {side} {symbol} 数量={quantity:.4f}")
                
                if self._order_callback:
                    self._order_callback({
                        'type': 'open',
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'order_id': order.get('orderId'),
                    })
                return True
            else:
                logger.error(f"开仓失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"开仓错误: {e}")
            return False
    
    def _close_position(self, symbol: str) -> bool:
        """平仓"""
        try:
            position = self._positions.get(symbol)
            if not position:
                return False
            
            quantity = position.quantity
            
            if self.config.mode == TradingMode.TEST:
                del self._positions[symbol]
                self._daily_trades += 1
                logger.info(f"测试模式: 平仓 {symbol}")
                
                if self._order_callback:
                    self._order_callback({
                        'type': 'close',
                        'symbol': symbol,
                        'side': position.side,
                        'quantity': quantity,
                    })
                return True
            
            order_side = "SELL" if position.side == "long" else "BUY"
            
            params = {
                'symbol': symbol,
                'side': order_side,
                'type': 'MARKET',
                'quantity': round(quantity, 3),
                'timestamp': self._get_server_time(),
                'recvWindow': 5000
            }
            params['signature'] = self._sign_request(params)
            
            headers = {'X-MBX-APIKEY': self.config.api_key}
            
            response = requests.post(
                f"{self._base_url}/fapi/v1/order",
                params=params,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                order = response.json()
                if symbol in self._positions:
                    del self._positions[symbol]
                self._daily_trades += 1
                logger.info(f"平仓成功: {symbol}")
                
                if self._order_callback:
                    self._order_callback({
                        'type': 'close',
                        'symbol': symbol,
                        'side': position.side,
                        'quantity': quantity,
                        'order_id': order.get('orderId'),
                    })
                return True
            else:
                logger.error(f"平仓失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"平仓错误: {e}")
            return False
    
    def _update_positions(self) -> None:
        """更新持仓"""
        if self.config.mode == TradingMode.TEST:
            for symbol, position in self._positions.items():
                position.update_price(position.current_price)
        else:
            positions = self._get_positions()
            self._positions.clear()
            
            for p in positions:
                symbol = p.get('symbol')
                position_amt = float(p.get('positionAmt', 0))
                entry_price = float(p.get('entryPrice', 0))
                mark_price = float(p.get('markPrice', entry_price))
                unrealized_pnl = float(p.get('unRealizedProfit', 0))
                
                if position_amt != 0:
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        side="long" if position_amt > 0 else "short",
                        quantity=abs(position_amt),
                        entry_price=entry_price,
                        current_price=mark_price,
                        unrealized_pnl=unrealized_pnl,
                        leverage=int(p.get('leverage', 1)),
                        margin=float(p.get('initialMargin', 0)),
                    )
        
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        self._account.unrealized_pnl = total_unrealized
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float | None = None,
        order_type: str = "market",
    ) -> Order | None:
        """下单"""
        if self._daily_trades >= self.config.max_daily_trades:
            logger.warning("已达每日最大交易次数")
            return None
        
        if abs(self._daily_pnl) >= self.config.max_daily_loss:
            logger.warning("已达每日最大亏损")
            return None
        
        order_id = f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )
        
        self._orders[order_id] = order
        
        if self.config.mode == TradingMode.TEST:
            self._simulate_order(order)
        else:
            self._submit_order(order)
        
        self._daily_trades += 1
        
        if self._order_callback:
            self._order_callback(order)
        
        logger.info(f"下单: {order_id} {side.value} {quantity} {symbol}")
        return order
    
    def _simulate_order(self, order: Order) -> None:
        """模拟订单执行"""
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = order.price or 100000
        order.update_time = datetime.now()
        
        if order.side == OrderSide.BUY:
            self._open_position_from_order(order)
        else:
            self._close_position_from_order(order)
    
    def _submit_order(self, order: Order) -> None:
        """提交订单到交易所"""
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = order.price or 100000
        order.update_time = datetime.now()
        
        if order.side == OrderSide.BUY:
            self._open_position_from_order(order)
        else:
            self._close_position_from_order(order)
    
    def _open_position_from_order(self, order: Order) -> None:
        """开仓"""
        symbol = order.symbol
        
        if symbol in self._positions:
            logger.warning(f"已有 {symbol} 持仓")
            return
        
        position = Position(
            symbol=symbol,
            side="long",
            quantity=order.filled_quantity,
            entry_price=order.filled_price,
            current_price=order.filled_price,
            leverage=self.config.leverage,
            margin=order.filled_quantity * order.filled_price / self.config.leverage,
        )
        
        self._positions[symbol] = position
        self._account.margin_used += position.margin
        self._account.available -= position.margin
        
        logger.info(f"开仓: {symbol} {position.quantity} @ {position.entry_price}")
    
    def _close_position_from_order(self, order: Order) -> None:
        """平仓"""
        symbol = order.symbol
        
        if symbol not in self._positions:
            logger.warning(f"无 {symbol} 持仓")
            return
        
        position = self._positions[symbol]
        pnl = (order.filled_price - position.entry_price) * position.quantity
        
        self._account.realized_pnl += pnl
        self._account.balance += pnl
        self._account.margin_used -= position.margin
        self._account.available += position.margin
        self._daily_pnl += pnl
        
        trade_record = {
            "symbol": symbol,
            "side": "close",
            "quantity": order.filled_quantity,
            "entry_price": position.entry_price,
            "exit_price": order.filled_price,
            "pnl": pnl,
            "time": datetime.now().isoformat(),
        }
        self._trade_history.append(trade_record)
        
        del self._positions[symbol]
        
        logger.info(f"平仓: {symbol} 盈亏: {pnl:.2f}")
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self._orders:
            return False
        
        order = self._orders[order_id]
        if order.status != OrderStatus.PENDING:
            return False
        
        order.status = OrderStatus.CANCELLED
        order.update_time = datetime.now()
        
        logger.info(f"取消订单: {order_id}")
        return True
    
    def get_trade_history(self, limit: int = 100) -> list[dict]:
        """获取交易历史"""
        return self._trade_history[-limit:]
    
    def get_statistics(self) -> dict[str, Any]:
        """获取交易统计"""
        total_trades = len(self._trade_history)
        winning_trades = len([t for t in self._trade_history if t["pnl"] > 0])
        losing_trades = len([t for t in self._trade_history if t["pnl"] < 0])
        
        total_pnl = sum(t["pnl"] for t in self._trade_history)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            "balance": self._account.balance,
            "available": self._account.available,
            "unrealized_pnl": self._account.unrealized_pnl,
            "realized_pnl": self._account.realized_pnl,
            "margin_used": self._account.margin_used,
            "margin_ratio": self._account.margin_ratio,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "daily_trades": self._daily_trades,
            "daily_pnl": self._daily_pnl,
        }
