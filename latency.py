import time
import contextvars
from functools import wraps
from pydantic import BaseModel
from typing import Dict, Any, Optional

# contextvar to hold latency metrics per request
# Using a dict to store {"embedding": 110, "vector_search": 65, ...}
metrics_ctx = contextvars.ContextVar("metrics_ctx", default=None)

def measure_latency(step_name: str):
    """
    Decorator to measure execution time of a function and store it in contextvars.
    Does not modify return value or original functionality.
    """
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics = metrics_ctx.get()
            if metrics is None:
                return func(*args, **kwargs)
                
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                # accumulate in case the same step is called multiple times (e.g. LLM streaming)
                if step_name in metrics:
                    metrics[step_name] += elapsed_ms
                else:
                    metrics[step_name] = elapsed_ms

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = metrics_ctx.get()
            if metrics is None:
                return await func(*args, **kwargs)
                
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                if step_name in metrics:
                    metrics[step_name] += elapsed_ms
                else:
                    metrics[step_name] = elapsed_ms

        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
