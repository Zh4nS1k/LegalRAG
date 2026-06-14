"""
Query caching for frequent legal queries to reduce retrieval and LLM costs.
Uses Redis if available, falls back to in-memory LRU cache.
"""

import hashlib
import json
import logging
import pickle
import time
from functools import lru_cache
from typing import Any, Optional, Tuple

from ai_service.core import config

logger = logging.getLogger("ai_service.cache")

# Try to import redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# In-memory cache as fallback
_IN_MEMORY_CACHE = {}
_CACHE_HITS = 0
_CACHE_MISSES = 0


def get_cache_client() -> Optional[redis.Redis]:
    """Get Redis client if configured, otherwise None."""
    if not REDIS_AVAILABLE:
        return None

    redis_url = config.REDIS_URL if hasattr(config, 'REDIS_URL') else None
    if not redis_url:
        return None

    try:
        client = redis.from_url(redis_url)
        client.ping()  # Test connection
        return client
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}, falling back to in-memory cache")
        return None


def generate_cache_key(query: str, intent: str, **kwargs) -> str:
    """Generate deterministic cache key from query and parameters."""
    data = {
        "query": query.strip().lower(),
        "intent": intent,
        **kwargs
    }
    # Sort keys for consistent hashing
    sorted_data = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return f"legalrag:{hashlib.sha256(sorted_data.encode()).hexdigest()}"


def cache_get(key: str, ttl_seconds: int = 3600) -> Optional[Any]:
    """Retrieve cached result if available and not expired."""
    global _CACHE_HITS, _CACHE_MISSES

    # Try Redis first
    redis_client = get_cache_client()
    if redis_client:
        try:
            cached = redis_client.get(key)
            if cached:
                _CACHE_HITS += 1
                return pickle.loads(cached)
        except Exception as e:
            logger.warning(f"Redis cache get failed: {e}")

    # Fallback to in-memory cache
    if key in _IN_MEMORY_CACHE:
        entry = _IN_MEMORY_CACHE[key]
        if time.time() - entry['timestamp'] < ttl_seconds:
            _CACHE_HITS += 1
            return entry['data']
        else:
            del _IN_MEMORY_CACHE[key]  # Expired

    _CACHE_MISSES += 1
    return None


def cache_set(key: str, data: Any, ttl_seconds: int = 3600) -> None:
    """Store result in cache with TTL."""
    # Try Redis first
    redis_client = get_cache_client()
    if redis_client:
        try:
            redis_client.setex(key, ttl_seconds, pickle.dumps(data))
            return
        except Exception as e:
            logger.warning(f"Redis cache set failed: {e}")

    # Fallback to in-memory cache (limited to 1000 entries)
    if len(_IN_MEMORY_CACHE) >= 1000:
        # Remove oldest entry
        oldest_key = min(_IN_MEMORY_CACHE.keys(),
                        key=lambda k: _IN_MEMORY_CACHE[k]['timestamp'])
        del _IN_MEMORY_CACHE[oldest_key]

    _IN_MEMORY_CACHE[key] = {
        'data': data,
        'timestamp': time.time()
    }


def cached_rag_response(func):
    """Decorator to cache RAG responses based on query and intent."""
    @lru_cache(maxsize=100)
    def memoized_key(query: str, intent: str, **kwargs):
        return generate_cache_key(query, intent, **kwargs)

    def wrapper(query: str, intent: str, **kwargs):
        # Skip cache for detective mode or streaming requests
        if intent == "detective" or kwargs.get('stream', False):
            return func(query, intent, **kwargs)

        cache_key = memoized_key(query, intent, **kwargs)

        # Check cache
        cached = cache_get(cache_key)
        if cached:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached

        # Execute and cache result
        result = func(query, intent, **kwargs)

        # Only cache successful results
        if result and result.get('result'):
            cache_set(cache_key, result)

        return result

    return wrapper


def get_cache_stats() -> dict:
    """Get cache statistics."""
    redis_client = get_cache_client()
    redis_info = {}
    if redis_client:
        try:
            redis_info = {
                'redis_connected': True,
                'memory_used': redis_client.info().get('used_memory_human', 'unknown'),
                'keys': redis_client.dbsize()
            }
        except:
            redis_info = {'redis_connected': False}

    return {
        'cache_hits': _CACHE_HITS,
        'cache_misses': _CACHE_MISSES,
        'hit_rate': _CACHE_HITS / (_CACHE_HITS + _CACHE_MISSES) if (_CACHE_HITS + _CACHE_MISSES) > 0 else 0,
        'in_memory_entries': len(_IN_MEMORY_CACHE),
        'redis': redis_info
    }


def clear_cache(pattern: str = "legalrag:*") -> int:
    """Clear cache entries matching pattern."""
    count = 0

    # Clear in-memory cache
    keys_to_delete = [k for k in _IN_MEMORY_CACHE.keys() if k.startswith("legalrag:")]
    count += len(keys_to_delete)
    for k in keys_to_delete:
        del _IN_MEMORY_CACHE[k]

    # Clear Redis cache
    redis_client = get_cache_client()
    if redis_client:
        try:
            # Use SCAN for large datasets
            cursor = 0
            while True:
                cursor, keys = redis_client.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    redis_client.delete(*keys)
                    count += len(keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")

    logger.info(f"Cleared {count} cache entries")
    return count