"""
API连接测试脚本
"""

import os
import sys
from datetime import datetime

print("=" * 50)
print("API连接测试")
print("=" * 50)

print("\n1. 检查环境变量...")
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("BINANCE_API_KEY") or os.getenv("API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv("API_SECRET")

print(f"   API Key: {'已配置' if api_key else '未配置'}")
print(f"   API Secret: {'已配置' if api_secret else '未配置'}")

print("\n2. 测试Binance客户端连接...")
try:
    from binance import Client
    from binance.helpers import round_step_size
    
    client = Client(api_key or "", api_secret or "")
    
    print("   测试获取服务器时间...")
    server_time = client.get_server_time()
    print(f"   服务器时间: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
    
    print("   测试获取交易所信息...")
    exchange_info = client.get_exchange_info()
    print(f"   交易所状态: {exchange_info.get('timezone', 'OK')}")
    
    print("\n3. 测试获取K线数据...")
    klines = client.get_klines(symbol="BTCUSDT", interval="1h", limit=5)
    print(f"   获取到 {len(klines)} 条K线数据")
    if klines:
        last_close = float(klines[-1][4])
        print(f"   最新收盘价: ${last_close:,.2f}")
    
    print("\n4. 测试期货API...")
    try:
        futures_klines = client.futures_klines(symbol="BTCUSDT", interval="1h", limit=5)
        print(f"   期货K线: 获取到 {len(futures_klines)} 条")
    except Exception as e:
        print(f"   期货API: {str(e)[:50]}...")
    
    print("\n" + "=" * 50)
    print("API连接状态: 正常")
    print("=" * 50)
    
except Exception as e:
    print(f"\n错误: {str(e)}")
    
    if "restricted location" in str(e).lower() or "access denied" in str(e).lower():
        print("\n检测到地区限制，尝试使用代理...")
        
        try:
            from Data.data_service import get_data_service, DataServiceConfig
            
            print("   尝试通过数据服务获取数据...")
            service = get_data_service(DataServiceConfig(
                use_proxy=True,
                proxy_host="127.0.0.1",
                proxy_port=7890,
            ))
            
            df = service.get_klines_from_database("BTCUSDT", "1h", 5)
            if not df.empty:
                print(f"   从数据库获取到 {len(df)} 条数据")
                print(f"   最新价格: ${df['close'].iloc[-1]:,.2f}")
                print("\n数据库状态: 正常")
            else:
                print("   数据库为空")
                
        except Exception as e2:
            print(f"   数据服务错误: {str(e2)}")
    
    print("\n" + "=" * 50)
    print("API连接状态: 受限")
    print("建议: 配置代理或使用数据库数据")
    print("=" * 50)

print("\n5. 测试数据库连接...")
try:
    from Data.database import DatabaseManager
    
    db = DatabaseManager()
    repo = db.get_repository("BTCUSDT")
    
    tables = repo.list_tables()
    print(f"   数据库表: {tables}")
    
    for table in tables:
        count = repo.get_count(table)
        print(f"   {table}: {count} 条记录")
    
    print("\n数据库状态: 正常")
    
except Exception as e:
    print(f"   数据库错误: {str(e)}")

print("\n测试完成!")
