"""
API限流模块

提供API请求限流功能，防止被封禁：
- 请求频率限制
- 请求间隔控制
- 指数退避重试
- 请求缓存
"""

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable
from collections import deque
import hashlib
import json

from app_logger.logger_setup import Logger

logger = Logger(__name__)


@dataclass
class RateLimitConfig:
    """限流配置"""
    
    requests_per_second: float = 5.0
    requests_per_minute: int = 1200
    requests_per_day: int = 100000
    
    min_request_interval: float = 0.2
    
    burst_limit: int = 10
    burst_window: float = 1.0
    
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    
    cache_ttl: int = 300
    cache_max_size: int = 1000


@dataclass
class RequestRecord:
    """请求记录"""
    timestamp: float
    endpoint: str
    success: bool
    response_time: float = 0.0


class RateLimiter:
    """API请求限流器"""
    
    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        
        self._request_history: deque[RequestRecord] = deque(maxlen=10000)
        self._last_request_time: float = 0.0
        self._lock = threading.Lock()
        
        self._cache: dict[str, tuple[Any, float]] = {}
        self._cache_lock = threading.Lock()
        
        self._consecutive_failures: int = 0
        self._backoff_until: float = 0.0
    
    def acquire(self, endpoint: str = "default") -> bool:
        """获取请求许可"""
        with self._lock:
            current_time = time.time()
            
            if current_time < self._backoff_until:
                wait_time = self._backoff_until - current_time
                logger.warning(f"限流退避中，等待 {wait_time:.1f}秒")
                return False
            
            if not self._check_rate_limits(current_time):
                return False
            
            interval = self.config.min_request_interval
            time_since_last = current_time - self._last_request_time
            if time_since_last < interval:
                time.sleep(interval - time_since_last)
            
            self._last_request_time = time.time()
            return True
    
    def _check_rate_limits(self, current_time: float) -> bool:
        """检查各项限流指标"""
        second_ago = current_time - 1.0
        minute_ago = current_time - 60.0
        day_ago = current_time - 86400.0
        
        requests_last_second = sum(
            1 for r in self._request_history 
            if r.timestamp > second_ago
        )
        if requests_last_second >= self.config.requests_per_second:
            logger.warning(f"秒级限流: {requests_last_second}/{self.config.requests_per_second}")
            return False
        
        requests_last_minute = sum(
            1 for r in self._request_history 
            if r.timestamp > minute_ago
        )
        if requests_last_minute >= self.config.requests_per_minute:
            logger.warning(f"分钟级限流: {requests_last_minute}/{self.config.requests_per_minute}")
            return False
        
        requests_last_day = sum(
            1 for r in self._request_history 
            if r.timestamp > day_ago
        )
        if requests_last_day >= self.config.requests_per_day:
            logger.warning(f"日级限流: {requests_last_day}/{self.config.requests_per_day}")
            return False
        
        burst_window_start = current_time - self.config.burst_window
        burst_requests = sum(
            1 for r in self._request_history 
            if r.timestamp > burst_window_start
        )
        if burst_requests >= self.config.burst_limit:
            logger.warning(f"突发限流: {burst_requests}/{self.config.burst_limit}")
            return False
        
        return True
    
    def record_request(
        self, 
        endpoint: str, 
        success: bool, 
        response_time: float = 0.0
    ) -> None:
        """记录请求结果"""
        with self._lock:
            record = RequestRecord(
                timestamp=time.time(),
                endpoint=endpoint,
                success=success,
                response_time=response_time,
            )
            self._request_history.append(record)
            
            if success:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3:
                    self._apply_backoff()
    
    def _apply_backoff(self) -> None:
        """应用指数退避"""
        delay = min(
            self.config.retry_base_delay * (2 ** self._consecutive_failures),
            self.config.retry_max_delay
        )
        self._backoff_until = time.time() + delay
        logger.warning(f"连续失败 {self._consecutive_failures} 次，退避 {delay:.1f}秒")
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached(self, cache_key: str) -> tuple[Any, bool]:
        """获取缓存"""
        with self._cache_lock:
            if cache_key in self._cache:
                data, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self.config.cache_ttl:
                    logger.debug(f"缓存命中: {cache_key[:8]}")
                    return data, True
                else:
                    del self._cache[cache_key]
        return None, False
    
    def set_cache(self, cache_key: str, data: Any) -> None:
        """设置缓存"""
        with self._cache_lock:
            if len(self._cache) >= self.config.cache_max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            
            self._cache[cache_key] = (data, time.time())
    
    def clear_cache(self) -> None:
        """清空缓存"""
        with self._cache_lock:
            self._cache.clear()
            logger.info("缓存已清空")
    
    def get_stats(self) -> dict[str, Any]:
        """获取限流统计"""
        current_time = time.time()
        
        with self._lock:
            total_requests = len(self._request_history)
            successful_requests = sum(1 for r in self._request_history if r.success)
            failed_requests = total_requests - successful_requests
            
            avg_response_time = 0.0
            if successful_requests > 0:
                avg_response_time = sum(
                    r.response_time for r in self._request_history if r.success
                ) / successful_requests
            
            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": successful_requests / total_requests * 100 if total_requests > 0 else 0,
                "avg_response_time_ms": avg_response_time * 1000,
                "consecutive_failures": self._consecutive_failures,
                "cache_size": len(self._cache),
                "is_backing_off": current_time < self._backoff_until,
            }


class ThrottledAPIClient:
    """带限流的API客户端包装器"""
    
    def __init__(
        self, 
        client: Any, 
        rate_limiter: RateLimiter | None = None,
        config: RateLimitConfig | None = None,
    ):
        self.client = client
        self.rate_limiter = rate_limiter or RateLimiter(config)
        self._enabled = True
    
    def enable_throttling(self, enabled: bool = True) -> None:
        """启用/禁用限流"""
        self._enabled = enabled
    
    def call(
        self, 
        method_name: str, 
        *args, 
        use_cache: bool = True,
        **kwargs
    ) -> Any:
        """执行带限流的API调用"""
        cache_key = ""
        if use_cache and self._enabled:
            cache_key = self.rate_limiter.get_cache_key(method_name, *args, **kwargs)
            cached, hit = self.rate_limiter.get_cached(cache_key)
            if hit:
                return cached
        
        if self._enabled:
            if not self.rate_limiter.acquire(method_name):
                raise Exception("请求被限流，请稍后重试")
        
        start_time = time.time()
        success = False
        result = None
        
        try:
            method = getattr(self.client, method_name)
            result = method(*args, **kwargs)
            success = True
            return result
        except Exception as e:
            logger.error(f"API调用失败 [{method_name}]: {e}")
            raise
        finally:
            response_time = time.time() - start_time
            if self._enabled:
                self.rate_limiter.record_request(method_name, success, response_time)
                
                if success and use_cache and cache_key:
                    self.rate_limiter.set_cache(cache_key, result)
    
    def call_with_retry(
        self,
        method_name: str,
        *args,
        max_retries: int | None = None,
        **kwargs
    ) -> Any:
        """带重试的API调用"""
        retries = max_retries or self.rate_limiter.config.retry_max_attempts
        last_error = None
        
        for attempt in range(retries):
            try:
                return self.call(method_name, *args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    delay = self.rate_limiter.config.retry_base_delay * (2 ** attempt)
                    logger.warning(f"重试 {attempt + 1}/{retries}, 等待 {delay}秒")
                    time.sleep(delay)
        
        raise last_error


_global_rate_limiter: RateLimiter | None = None
_global_limiter_lock = threading.Lock()


def get_global_rate_limiter(config: RateLimitConfig | None = None) -> RateLimiter:
    """获取全局限流器"""
    global _global_rate_limiter
    
    with _global_limiter_lock:
        if _global_rate_limiter is None:
            _global_rate_limiter = RateLimiter(config)
            logger.info("全局限流器已初始化")
        return _global_rate_limiter


def configure_rate_limits(
    requests_per_second: float = 5.0,
    requests_per_minute: int = 1200,
    min_request_interval: float = 0.2,
    cache_ttl: int = 300,
) -> None:
    """配置全局限流参数"""
    config = RateLimitConfig(
        requests_per_second=requests_per_second,
        requests_per_minute=requests_per_minute,
        min_request_interval=min_request_interval,
        cache_ttl=cache_ttl,
    )
    
    global _global_rate_limiter
    _global_rate_limiter = RateLimiter(config)
    logger.info(f"限流配置已更新: {requests_per_second} req/s, {requests_per_minute} req/min")
