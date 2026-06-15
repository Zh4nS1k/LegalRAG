from __future__ import annotations

import inspect
import time
from threading import Lock


class CircuitBreakerOpen(RuntimeError):
    pass


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, reset_timeout: int = 60):
        self.name = name
        self.failures = 0
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0.0
        self.state = "closed"
        self._lock = Lock()

    def _before_call(self) -> None:
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = "half-open"
                else:
                    raise CircuitBreakerOpen(f"Circuit breaker {self.name} is OPEN")

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.threshold:
                self.state = "open"

    def call(self, func, *args, **kwargs):
        self._before_call()
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    async def call_async(self, func, *args, **kwargs):
        self._before_call()
        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise


class CircuitBreakerProxy:
    def __init__(
        self,
        target,
        breaker: CircuitBreaker,
        *,
        sync_methods: set[str] | None = None,
        async_methods: set[str] | None = None,
        stream_methods: set[str] | None = None,
    ):
        self._target = target
        self._breaker = breaker
        self._sync_methods = sync_methods or set()
        self._async_methods = async_methods or set()
        self._stream_methods = stream_methods or set()

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    def __getattr__(self, name: str):
        if name == "describe_index_stats" and hasattr(self._target, "_index"):
            index = getattr(self._target, "_index")

            def describe_index_stats(*args, **kwargs):
                return self._breaker.call(index.describe_index_stats, *args, **kwargs)

            return describe_index_stats

        attr = getattr(self._target, name)
        if name in self._sync_methods and callable(attr):

            def sync_wrapper(*args, **kwargs):
                return self._breaker.call(attr, *args, **kwargs)

            return sync_wrapper

        if name in self._async_methods and callable(attr):

            async def async_wrapper(*args, **kwargs):
                return await self._breaker.call_async(attr, *args, **kwargs)

            return async_wrapper

        if name in self._stream_methods and callable(attr):

            async def stream_wrapper(*args, **kwargs):
                self._breaker._before_call()
                try:
                    async for item in attr(*args, **kwargs):
                        yield item
                    self._breaker.record_success()
                except Exception:
                    self._breaker.record_failure()
                    raise

            return stream_wrapper

        return attr
