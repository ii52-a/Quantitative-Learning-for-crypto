"""
测试API限流功能
"""

import time
from Data.rate_limiter import RateLimiter, RateLimitConfig

print("=" * 50)
print("API限流功能测试")
print("=" * 50)

print("\n1. 测试限流器初始化...")
config = RateLimitConfig(
    requests_per_second=3.0,
    requests_per_minute=100,
    min_request_interval=0.3,
    cache_ttl=60,
)
limiter = RateLimiter(config)
print(f"   配置: {config.requests_per_second} req/s, {config.requests_per_minute} req/min")
print(f"   最小间隔: {config.min_request_interval}s")

print("\n2. 测试请求许可获取...")
start = time.time()
for i in range(5):
    acquired = limiter.acquire("test_endpoint")
    limiter.record_request("test_endpoint", True, 0.1)
    elapsed = time.time() - start
    print(f"   请求 {i+1}: 获取={'成功' if acquired else '失败'}, 累计耗时: {elapsed:.2f}s")

print("\n3. 测试缓存功能...")
limiter.set_cache("test_key", {"data": "test_value"})
cached, hit = limiter.get_cached("test_key")
print(f"   缓存命中: {hit}, 数据: {cached}")

print("\n4. 测试限流统计...")
stats = limiter.get_stats()
print(f"   总请求数: {stats['total_requests']}")
print(f"   成功请求: {stats['successful_requests']}")
print(f"   失败请求: {stats['failed_requests']}")
print(f"   成功率: {stats['success_rate']:.1f}%")
print(f"   平均响应时间: {stats['avg_response_time_ms']:.1f}ms")
print(f"   缓存大小: {stats['cache_size']}")

print("\n5. 测试数据服务限流集成...")
try:
    from Data.data_service import get_data_service, DataServiceConfig
    
    service_config = DataServiceConfig(
        prefer_database=True,
        enable_rate_limit=True,
        requests_per_second=2.0,
        requests_per_minute=100,
        min_request_interval=0.5,
        cache_ttl=300,
    )
    
    service = get_data_service(service_config)
    print(f"   数据服务已初始化")
    print(f"   限流启用: {service_config.enable_rate_limit}")
    
    print("\n   获取数据测试...")
    start = time.time()
    data = service.get_klines("BTCUSDT", "30min", 50)
    elapsed = time.time() - start
    print(f"   获取 {len(data)} 条数据, 耗时: {elapsed:.2f}s")
    
    print("\n   再次获取相同数据(测试缓存)...")
    start = time.time()
    data2 = service.get_klines("BTCUSDT", "30min", 50)
    elapsed = time.time() - start
    print(f"   获取 {len(data2)} 条数据, 耗时: {elapsed:.2f}s")
    
    print("\n   限流统计:")
    stats = service.get_rate_limit_stats()
    print(f"   总请求: {stats['total_requests']}")
    print(f"   成功率: {stats['success_rate']:.1f}%")
    print(f"   缓存大小: {stats['cache_size']}")
    
except Exception as e:
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("测试完成!")
print("=" * 50)
