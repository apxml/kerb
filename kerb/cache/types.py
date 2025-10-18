"""Cache data models and base interfaces.

This module provides the fundamental building blocks for caching:
- CacheEntry: Represents a single cache entry with metadata
- CacheStats: Tracks cache statistics and metrics
- BaseCache: Abstract base class for all cache implementations
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None  # Time to live in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary (excluding value for metadata)."""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "ttl": self.ttl,
            "metadata": self.metadata,
        }


@dataclass
class CacheStats:
    """Cache statistics and metrics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    total_requests: int = 0
    estimated_cost_saved: float = 0.0  # In dollars
    estimated_time_saved: float = 0.0  # In seconds

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "total_requests": self.total_requests,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "estimated_cost_saved": self.estimated_cost_saved,
            "estimated_time_saved": self.estimated_time_saved,
        }


# ============================================================================
# Base Cache Interface
# ============================================================================


class BaseCache:
    """Base cache interface with common operations."""

    def __init__(
        self, max_size: Optional[int] = None, default_ttl: Optional[float] = None
    ):
        """Initialize cache.

        Args:
            max_size: Maximum number of entries (None for unlimited)
            default_ttl: Default time-to-live in seconds (None for no expiration)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set value in cache."""
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """Delete key from cache. Returns True if key existed."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all cache entries."""
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        raise NotImplementedError

    def size(self) -> int:
        """Get current cache size."""
        raise NotImplementedError

    def keys(self) -> List[str]:
        """Get all cache keys."""
        raise NotImplementedError
