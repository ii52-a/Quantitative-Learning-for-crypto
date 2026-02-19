"""测试Binance API连接和数据获取"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import requests
import hashlib
import hmac
import time
from pathlib import Path

# 加载.env文件
env_path = Path(project_root) / ".env"
api_key = ""
api_secret = ""

print(f"查找.env文件: {env_path}")

if env_path.exists():
    print("找到.env文件")
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                if key in ['BINANCE_API_KEY', 'API_KEY']:
                    api_key = value
                elif key in ['BINANCE_API_SECRET', 'API_SECRET']:
                    api_secret = value
else:
    print("未找到.env文件")

print(f"API Key: {api_key[:10]}..." if api_key else "未找到API Key")
print(f"API Secret: {api_secret[:10]}..." if api_secret else "未找到API Secret")

# 测试网和实盘API
BASE_URLS = {
    "testnet": "https://testnet.binancefuture.com",
    "live": "https://fapi.binance.com"
}

def get_server_time(base_url):
    """获取服务器时间"""
    response = requests.get(f"{base_url}/fapi/v1/time", timeout=10)
    return response.json()['serverTime']

def sign_request(params, secret):
    """签名请求"""
    query_string = '&'.join(f"{k}={v}" for k, v in params.items())
    signature = hmac.new(
        secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def get_account_info(base_url, api_key, api_secret):
    """获取账户信息"""
    params = {
        'timestamp': get_server_time(base_url),
        'recvWindow': 5000
    }
    params['signature'] = sign_request(params, api_secret)
    
    headers = {
        'X-MBX-APIKEY': api_key
    }
    
    response = requests.get(
        f"{base_url}/fapi/v2/account",
        params=params,
        headers=headers,
        timeout=10
    )
    
    return response.json() if response.status_code == 200 else None

def get_balance(base_url, api_key, api_secret):
    """获取余额"""
    params = {
        'timestamp': get_server_time(base_url),
        'recvWindow': 5000
    }
    params['signature'] = sign_request(params, api_secret)
    
    headers = {
        'X-MBX-APIKEY': api_key
    }
    
    response = requests.get(
        f"{base_url}/fapi/v2/balance",
        params=params,
        headers=headers,
        timeout=10
    )
    
    return response.json() if response.status_code == 200 else None

def get_positions(base_url, api_key, api_secret):
    """获取持仓"""
    params = {
        'timestamp': get_server_time(base_url),
        'recvWindow': 5000
    }
    params['signature'] = sign_request(params, api_secret)
    
    headers = {
        'X-MBX-APIKEY': api_key
    }
    
    response = requests.get(
        f"{base_url}/fapi/v2/positionRisk",
        params=params,
        headers=headers,
        timeout=10
    )
    
    return response.json() if response.status_code == 200 else None

# 测试实盘API
print("\n" + "="*60)
print("测试实盘API")
print("="*60)

base_url = BASE_URLS["live"]

print("\n[1] 账户信息 (fapi/v2/account)")
account = get_account_info(base_url, api_key, api_secret)
if account:
    print(f"  totalWalletBalance: {account.get('totalWalletBalance', 'N/A')}")
    print(f"  totalUnrealizedProfit: {account.get('totalUnrealizedProfit', 'N/A')}")
    print(f"  totalMarginBalance: {account.get('totalMarginBalance', 'N/A')}")
    print(f"  availableBalance: {account.get('availableBalance', 'N/A')}")
    print(f"  maxWithdrawAmount: {account.get('maxWithdrawAmount', 'N/A')}")
    
    # 打印所有资产
    assets = account.get('assets', [])
    for asset in assets:
        if float(asset.get('walletBalance', 0)) > 0:
            print(f"\n  资产 {asset.get('asset')}:")
            print(f"    walletBalance: {asset.get('walletBalance')}")
            print(f"    unrealizedProfit: {asset.get('unrealizedProfit')}")
            print(f"    marginBalance: {asset.get('marginBalance')}")
            print(f"    maintMargin: {asset.get('maintMargin')}")
            print(f"    initialMargin: {asset.get('initialMargin')}")
else:
    print("  获取失败")

print("\n[2] 余额信息 (fapi/v2/balance)")
balance = get_balance(base_url, api_key, api_secret)
if balance:
    for b in balance:
        if float(b.get('balance', 0)) > 0 or float(b.get('availableBalance', 0)) > 0:
            print(f"\n  资产 {b.get('asset')}:")
            print(f"    balance: {b.get('balance')}")
            print(f"    availableBalance: {b.get('availableBalance')}")
            print(f"    crossWalletBalance: {b.get('crossWalletBalance')}")
            print(f"    crossUnPnl: {b.get('crossUnPnl')}")
else:
    print("  获取失败")

print("\n[3] 持仓信息 (fapi/v2/positionRisk)")
positions = get_positions(base_url, api_key, api_secret)
if positions:
    has_position = False
    for p in positions:
        position_amt = float(p.get('positionAmt', 0))
        if position_amt != 0:
            has_position = True
            print(f"\n  持仓 {p.get('symbol')}:")
            print(f"    positionAmt: {p.get('positionAmt')}")
            print(f"    entryPrice: {p.get('entryPrice')}")
            print(f"    markPrice: {p.get('markPrice')}")
            print(f"    unRealizedProfit: {p.get('unRealizedProfit')}")
            print(f"    liquidationPrice: {p.get('liquidationPrice')}")
            print(f"    leverage: {p.get('leverage')}")
            print(f"    marginType: {p.get('marginType')}")
    
    if not has_position:
        print("  无持仓")
else:
    print("  获取失败")

print("\n" + "="*60)
print("API测试完成")
print("="*60)
