"""Cache management utilities for LLM applications.

This module provides comprehensive caching for LLM workflows:

Core Components:
    CacheEntry - Single cache entry with metadata
    CacheStats - Cache statistics and metrics
    BaseCache - Base class for all cache implementations
    cached - Decorator to cache function results

Cache Backends:
    MemoryCache - Fast in-memory LRU cache with TTL support
    DiskCache - Persistent disk-based cache with serialization
    TieredCache - Two-tier cache (memory + disk) for best performance
    LLMCache - High-level cache wrapper for LLM applications

Key Strategies:
    generate_cache_key - Generate cache key from arguments
    generate_prompt_key - Generate key for LLM prompts
    generate_embedding_key - Generate key for embeddings

Utilities:
    create_memory_cache - Create in-memory cache
    create_disk_cache - Create disk-based cache
    create_tiered_cache - Create two-tier cache
    create_llm_cache - Create LLM-specific cache
    invalidate_expired_entries - Clean up expired cache entries
    export_cache_stats - Export cache statistics
    estimate_cache_size - Estimate cache size in various units

The cache system provides:
- LLM response caching by prompt hash
- Embedding cache management
- Automatic cache invalidation (TTL)
- LRU eviction for memory management
- Cost and time savings tracking
- Multiple backend options (memory, disk, tiered)
- Persistent storage for cross-session caching

Example:
    >>> from kerb.cache import cached, MemoryCache
    >>> from kerb.cache.backends import LLMCache
    >>> from kerb.cache.strategies import generate_prompt_key
    >>>
    >>> # Use decorator
    >>> @cached(ttl=3600)
    >>> def expensive_function(x):
    ...     return x * 2
    >>>
    >>> # Use cache directly
    >>> cache = MemoryCache(max_size=100)
    >>> cache.set("key", "value")
    >>> cache.get("key")
"""

# Import submodules
from . import backends, strategies, utils
# Cache backend implementations
from .backends import DiskCache, LLMCache, MemoryCache, TieredCache
# Cache strategies and decorator
from .strategies import (cached, generate_cache_key, generate_embedding_key,
                         generate_prompt_key)
# Core cache models and interfaces
from .types import BaseCache, CacheEntry, CacheStats
# Most commonly used utilities
from .utils import (create_disk_cache, create_llm_cache, create_memory_cache,
                    create_tiered_cache, estimate_cache_size,
                    export_cache_stats, invalidate_expired_entries)

__all__ = [
    # Core models and interfaces
    "CacheEntry",
    "CacheStats",
    "BaseCache",
    # Submodules
    "models",
    "backends",
    "strategies",
    "utils",
    # Cache backends
    "MemoryCache",
    "DiskCache",
    "TieredCache",
    "LLMCache",
    # Cache strategies
    "generate_cache_key",
    "generate_prompt_key",
    "generate_embedding_key",
    "cached",
    # Cache utilities
    "create_memory_cache",
    "create_disk_cache",
    "create_tiered_cache",
    "create_llm_cache",
    "invalidate_expired_entries",
    "export_cache_stats",
    "estimate_cache_size",
]
