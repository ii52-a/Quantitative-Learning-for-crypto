"""API延迟测试脚本"""
import requests
import time

print('=' * 50)
print('币安API延迟测试')
print('=' * 50)

# 测试主网
print('\n【主网测试】')
base_url = 'https://fapi.binance.com'

start = time.time()
try:
    resp = requests.get(f'{base_url}/fapi/v1/ticker/24hr?symbol=BTCUSDT', timeout=10)
    latency = (time.time() - start) * 1000
    print(f'24hr行情接口: {latency:.0f}ms')
    if resp.status_code == 200:
        data = resp.json()
        print(f'  当前价格: {float(data.get("lastPrice", 0)):.2f}')
        print(f'  24h涨跌: {float(data.get("priceChangePercent", 0)):.2f}%')
except Exception as e:
    print(f'24hr行情接口: 失败 - {e}')

start = time.time()
try:
    resp = requests.get(f'{base_url}/fapi/v1/premiumIndex?symbol=BTCUSDT', timeout=10)
    latency = (time.time() - start) * 1000
    print(f'标记价格接口: {latency:.0f}ms')
    if resp.status_code == 200:
        data = resp.json()
        print(f'  标记价格: {float(data.get("markPrice", 0)):.2f}')
except Exception as e:
    print(f'标记价格接口: 失败 - {e}')

start = time.time()
try:
    resp = requests.get(f'{base_url}/fapi/v1/time', timeout=10)
    latency = (time.time() - start) * 1000
    print(f'服务器时间接口: {latency:.0f}ms')
except Exception as e:
    print(f'服务器时间接口: 失败 - {e}')

# 测试测试网
print('\n【测试网测试】')
testnet_url = 'https://testnet.binancefuture.com'

start = time.time()
try:
    resp = requests.get(f'{testnet_url}/fapi/v1/ticker/24hr?symbol=BTCUSDT', timeout=10)
    latency = (time.time() - start) * 1000
    print(f'24hr行情接口: {latency:.0f}ms')
    if resp.status_code == 200:
        data = resp.json()
        print(f'  当前价格: {float(data.get("lastPrice", 0)):.2f}')
except Exception as e:
    print(f'24hr行情接口: 失败 - {e}')

start = time.time()
try:
    resp = requests.get(f'{testnet_url}/fapi/v1/premiumIndex?symbol=BTCUSDT', timeout=10)
    latency = (time.time() - start) * 1000
    print(f'标记价格接口: {latency:.0f}ms')
    if resp.status_code == 200:
        data = resp.json()
        print(f'  标记价格: {float(data.get("markPrice", 0)):.2f}')
except Exception as e:
    print(f'标记价格接口: 失败 - {e}')

# 连续测试
print('\n【连续测试 - 10次请求】')
latencies = []
for i in range(10):
    start = time.time()
    try:
        resp = requests.get(f'{base_url}/fapi/v1/ticker/24hr?symbol=BTCUSDT', timeout=10)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        print(f'  第{i+1}次: {latency:.0f}ms')
    except:
        pass
    time.sleep(0.1)

if latencies:
    print(f'\n平均延迟: {sum(latencies)/len(latencies):.0f}ms')
    print(f'最小延迟: {min(latencies):.0f}ms')
    print(f'最大延迟: {max(latencies):.0f}ms')

print('\n' + '=' * 50)
print('结论:')
if latencies and sum(latencies)/len(latencies) < 500:
    print('API延迟正常，实时价格延迟可能是UI更新频率或程序逻辑导致')
else:
    print('API延迟较高，可能是网络问题')
