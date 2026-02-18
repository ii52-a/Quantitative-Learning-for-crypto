"""
量化交易系统

主程序入口
"""

import sys


def main():
    """主函数"""
    print("=" * 60)
    print("       量化交易系统 v2.0.0")
    print("=" * 60)
    print()
    print("请选择启动模式:")
    print("  1. 用户友好界面 (推荐新手)")
    print("  2. 完整功能界面")
    print("  3. 命令行回测")
    print("  4. 实盘交易")
    print()
    
    choice = input("请输入选项 (1-4): ").strip()
    
    if choice == "1":
        from ui.user_friendly import main as ui_main
        ui_main()
    elif choice == "2":
        from UI.main_ui import main as full_ui_main
        full_ui_main()
    elif choice == "3":
        _run_cli_backtest()
    elif choice == "4":
        _run_live_trading()
    else:
        print("无效选项，启动用户友好界面...")
        from ui.user_friendly import main as ui_main
        ui_main()


def _run_cli_backtest():
    """命令行回测"""
    print("\n命令行回测模式")
    print("-" * 40)
    
    symbol = input("交易对 (默认 BTCUSDT): ").strip() or "BTCUSDT"
    interval = input("K线周期 (默认 30min): ").strip() or "30min"
    limit = int(input("数据量 (默认 1000): ").strip() or "1000")
    strategy_name = input("策略名称 (默认 MACDStrategy): ").strip() or "MACDStrategy"
    
    print(f"\n正在加载 {symbol} {interval} 数据...")
    
    from Data.data_service import get_data_service, DataServiceConfig
    from strategy.templates import get_strategy
    from backtest.engine import BacktestEngine
    from backtest.report import BacktestReport
    from core.config import BacktestConfig
    
    service = get_data_service(DataServiceConfig(prefer_database=True, auto_init=True))
    data = service.get_backtest_data(symbol, interval, limit)
    
    if data.empty:
        print("数据加载失败")
        return
    
    print(f"数据加载完成: {len(data)} 条")
    
    strategy = get_strategy(strategy_name)
    config = BacktestConfig(symbol=symbol, interval=interval, data_limit=limit)
    
    print("正在执行回测...")
    engine = BacktestEngine(strategy, config)
    result = engine.run(data)
    
    report = BacktestReport(result, strategy_name, symbol, interval)
    print("\n" + report.format_text_report())


def _run_live_trading():
    """实盘交易"""
    print("\n实盘交易模式")
    print("-" * 40)
    
    import asyncio
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    
    if not api_key or not api_secret:
        print("错误: 未配置API密钥，请检查.env文件")
        return
    
    from Strategy.live_trading import run_live_trading
    
    symbol = input("交易对 (默认 BTCUSDT): ").strip() or "BTCUSDT"
    interval = input("K线周期 (默认 30min): ").strip() or "30min"
    testnet = input("使用测试网? (y/n, 默认 y): ").strip().lower() != "n"
    dry_run = input("模拟模式? (y/n, 默认 y): ").strip().lower() != "n"
    
    print(f"\n启动实盘交易: {symbol} {interval}")
    print(f"测试网: {testnet}, 模拟模式: {dry_run}")
    
    asyncio.run(run_live_trading(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        interval=interval,
        testnet=testnet,
        dry_run=dry_run,
    ))


if __name__ == "__main__":
    main()
