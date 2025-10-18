"""Cache utility functions.

This module provides helper functions for working with caches:
- create_memory_cache: Convenience factory for MemoryCache
- create_disk_cache: Convenience factory for DiskCache
- create_tiered_cache: Convenience factory for TieredCache
- create_llm_cache: Convenience factory for LLMCache
- invalidate_expired_entries: Clean up expired cache entries
- export_cache_stats: Export cache statistics in various formats
- estimate_cache_size: Estimate cache size in different units
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

from .backends import DiskCache, LLMCache, MemoryCache, TieredCache
from .types import BaseCache

if TYPE_CHECKING:
    from kerb.core.enums import ExportFormat, SizeUnit


def create_memory_cache(
    max_size: int = 1000, default_ttl: Optional[float] = None
) -> MemoryCache:
    """Create a new memory cache.

    Args:
        max_size: Maximum number of entries
        default_ttl: Default TTL in seconds

    Returns:
        MemoryCache: New memory cache instance

    Example:
        >>> cache = create_memory_cache(max_size=100, default_ttl=3600)
    """
    return MemoryCache(max_size=max_size, default_ttl=default_ttl)


def create_disk_cache(
    cache_dir: str = ".cache",
    max_size: Optional[int] = None,
    default_ttl: Optional[float] = None,
    serializer: str = "pickle",
) -> DiskCache:
    """Create a new disk cache.

    Args:
        cache_dir: Directory to store cache files
        max_size: Maximum number of entries
        default_ttl: Default TTL in seconds
        serializer: Serialization format ('pickle' or 'json')

    Returns:
        DiskCache: New disk cache instance

    Example:
        >>> cache = create_disk_cache(cache_dir=".cache/llm", serializer="json")
    """
    return DiskCache(
        cache_dir=cache_dir,
        max_size=max_size,
        default_ttl=default_ttl,
        serializer=serializer,
    )


def create_tiered_cache(
    memory_max_size: int = 100,
    disk_cache_dir: str = ".cache",
    disk_max_size: Optional[int] = None,
    default_ttl: Optional[float] = None,
) -> TieredCache:
    """Create a new tiered cache (memory + disk).

    Args:
        memory_max_size: Maximum entries in memory cache
        disk_cache_dir: Directory for disk cache
        disk_max_size: Maximum entries in disk cache
        default_ttl: Default TTL in seconds

    Returns:
        TieredCache: New tiered cache instance

    Example:
        >>> cache = create_tiered_cache(
        ...     memory_max_size=50,
        ...     disk_cache_dir=".cache/llm"
        ... )
    """
    return TieredCache(
        memory_max_size=memory_max_size,
        disk_cache_dir=disk_cache_dir,
        disk_max_size=disk_max_size,
        default_ttl=default_ttl,
    )


def create_llm_cache(
    backend: Optional[BaseCache] = None,
    cost_per_token: float = 0.00001,
    avg_tokens_per_request: int = 1000,
    avg_response_time: float = 2.0,
) -> LLMCache:
    """Create a new LLM-specific cache.

    Args:
        backend: Cache backend to use
        cost_per_token: Cost per token for tracking
        avg_tokens_per_request: Average tokens per request
        avg_response_time: Average response time in seconds

    Returns:
        LLMCache: New LLM cache instance

    Example:
        >>> cache = create_llm_cache(
        ...     backend=create_tiered_cache(),
        ...     cost_per_token=0.00002
        ... )
    """
    return LLMCache(
        backend=backend,
        cost_per_token=cost_per_token,
        avg_tokens_per_request=avg_tokens_per_request,
        avg_response_time=avg_response_time,
    )


def invalidate_expired_entries(cache: BaseCache) -> int:
    """Manually invalidate all expired entries in a cache.

    Args:
        cache: Cache to clean

    Returns:
        int: Number of entries invalidated

    Example:
        >>> count = invalidate_expired_entries(cache)
        >>> print(f"Removed {count} expired entries")
    """
    if isinstance(cache, DiskCache):
        cache._clean_expired()
        return 0  # DiskCache cleans internally

    if not isinstance(cache, MemoryCache):
        return 0

    expired_keys = []
    for key, entry in cache._cache.items():
        if entry.is_expired():
            expired_keys.append(key)

    for key in expired_keys:
        cache.delete(key)

    return len(expired_keys)


def export_cache_stats(
    cache: Union[BaseCache, LLMCache], format: Union["ExportFormat", str] = "dict"
) -> Union[Dict, str]:
    """Export cache statistics in various formats.

    Args:
        cache: Cache to export stats from
        format: Output format (ExportFormat enum or string: 'dict', 'json', 'csv', 'table')

    Returns:
        Union[Dict, str]: Statistics in requested format

    Examples:
        >>> from kerb.core.enums import ExportFormat
        >>> stats = export_cache_stats(cache, format=ExportFormat.JSON)
        >>> print(stats)
    """
    from kerb.core.enums import ExportFormat, validate_enum_or_string

    # Validate and normalize format
    format_val = validate_enum_or_string(format, ExportFormat, "format")
    if isinstance(format_val, ExportFormat):
        format_str = format_val.value
    else:
        format_str = format_val

    if isinstance(cache, LLMCache):
        stats = cache.get_stats()
    elif isinstance(cache, (MemoryCache, DiskCache)):
        stats = cache.get_stats()
    elif isinstance(cache, TieredCache):
        stats_dict = cache.get_stats()
        # Return both
        if format_str == "json":
            return json.dumps(
                {
                    "memory": stats_dict["memory"].to_dict(),
                    "disk": stats_dict["disk"].to_dict(),
                },
                indent=2,
            )
        return {
            "memory": stats_dict["memory"].to_dict(),
            "disk": stats_dict["disk"].to_dict(),
        }
    else:
        stats = cache.stats

    stats_dict = stats.to_dict()

    if format_str == "json":
        return json.dumps(stats_dict, indent=2)
    return stats_dict


def estimate_cache_size(
    cache: BaseCache, unit: Union["SizeUnit", str] = "entries"
) -> Union[int, str]:
    """Estimate cache size in various units.

    Args:
        cache: Cache to measure
        unit: Unit to measure in (SizeUnit enum or string: 'entries', 'bytes', 'kb', 'mb', 'gb')

    Returns:
        Union[int, str]: Size in requested unit

    Examples:
        >>> from kerb.core.enums import SizeUnit
        >>> size = estimate_cache_size(cache, unit=SizeUnit.MB)
        >>> print(f"Cache size: {size}")
    """
    from kerb.core.enums import SizeUnit, validate_enum_or_string

    # Validate and normalize unit
    unit_val = validate_enum_or_string(unit, SizeUnit, "unit")
    if isinstance(unit_val, SizeUnit):
        unit_str = unit_val.value
    else:
        unit_str = unit_val

    if unit_str == "entries":
        return cache.size()

    if isinstance(cache, DiskCache):
        # Calculate disk usage
        total_bytes = 0
        for key in cache.keys():
            cache_path = cache._get_cache_path(key)
            if cache_path.exists():
                total_bytes += cache_path.stat().st_size

        if unit_str == "bytes":
            return total_bytes
        elif unit_str == "kb":
            return total_bytes / 1024.0
        elif unit_str == "mb":
            return total_bytes / (1024.0 * 1024.0)
        elif unit_str == "gb":
            return total_bytes / (1024.0 * 1024.0 * 1024.0)
        elif unit_str == "human":
            # Convert to human readable
            for unit_name in ["B", "KB", "MB", "GB"]:
                if total_bytes < 1024.0:
                    return f"{total_bytes:.2f} {unit_name}"
                total_bytes /= 1024.0
            return f"{total_bytes:.2f} TB"

    return cache.size()
