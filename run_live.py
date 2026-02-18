"""
实盘交易启动脚本 - MACD 30min CTA策略

使用方法:
1. 复制 .env.example 为 .env
2. 填写你的 API Key 和 Secret
3. 运行: python run_live.py
"""

import asyncio
import os
import signal
import sys
from datetime import datetime

from dotenv import load_dotenv

from Strategy.live_trading import run_live_trading
from app_logger.logger_setup import Logger

load_dotenv()
logger = Logger(__name__)

engine = None


def signal_handler(sig, frame):
    global engine
    logger.info("收到停止信号，正在安全退出...")
    if engine:
        engine.stop()
    sys.exit(0)


async def main():
    global engine

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key or not api_secret:
        print("=" * 60)
        print("错误: 未找到 API 密钥配置")
        print("=" * 60)
        print("请创建 .env 文件并配置以下内容:")
        print("")
        print("BINANCE_API_KEY=你的API_KEY")
        print("BINANCE_API_SECRET=你的API_SECRET")
        print("BINANCE_TESTNET=true")
        print("BINANCE_DRY_RUN=true")
        print("=" * 60)
        return

    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    dry_run = os.getenv("BINANCE_DRY_RUN", "true").lower() == "true"
    symbol = os.getenv("BINANCE_SYMBOL", "BTCUSDT")
    interval = os.getenv("BINANCE_INTERVAL", "30m")
    risk_usdt = float(os.getenv("BINANCE_RISK_USDT", "100"))
    initial_balance = float(os.getenv("BINANCE_INITIAL_BALANCE", "1000"))
    hist_filter = float(os.getenv("BINANCE_HIST_FILTER", "0.5"))

    print("\n" + "=" * 60)
    print("       MACD 30min CTA 实盘交易系统")
    print("=" * 60)
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    print(f"  交易对:     {symbol}")
    print(f"  K线周期:    {interval}")
    print(f"  初始资金:   {initial_balance} USDT")
    print(f"  风险资金:   {risk_usdt} USDT")
    print("-" * 60)
    print(f"  HIST过滤:   {hist_filter}")
    print(f"  历史数据:   300条 (启动时自动加载)")
    print("-" * 60)
    print(f"  测试网:     {'是' if testnet else '否'}")
    print(f"  模拟模式:   {'是' if dry_run else '否 (实盘!)'}")
    print("=" * 60 + "\n")

    if not dry_run and not testnet:
        confirm = input("确认要在主网实盘运行? (输入 YES 确认): ")
        if confirm != "YES":
            print("已取消")
            return

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await run_live_trading(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            interval=interval,
            initial_balance=initial_balance,
            risk_usdt=risk_usdt,
            hist_filter=hist_filter,
            testnet=testnet,
            dry_run=dry_run,
        )
    except Exception as e:
        logger.error(f"交易异常: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
