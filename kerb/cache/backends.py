"""Cache backend implementations.

This module provides concrete cache implementations:
- MemoryCache: Fast in-memory LRU cache
- DiskCache: Persistent disk-based cache
- TieredCache: Two-tier memory + disk cache
- LLMCache: High-level wrapper for LLM applications
"""

import hashlib
import json
import os
import pickle
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .strategies import generate_embedding_key, generate_prompt_key
from .types import BaseCache, CacheEntry, CacheStats

# ============================================================================
# In-Memory Cache
# ============================================================================


class MemoryCache(BaseCache):
    """In-memory cache with LRU eviction and TTL support."""

    def __init__(
        self, max_size: Optional[int] = 1000, default_ttl: Optional[float] = None
    ):
        """Initialize memory cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds

        Example:
            >>> cache = MemoryCache(max_size=100, default_ttl=3600)
            >>> cache.set("key", "value")
            >>> cache.get("key")
            'value'
        """
        super().__init__(max_size, default_ttl)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        self.stats.total_requests += 1

        if key not in self._cache:
            self.stats.misses += 1
            return None

        entry = self._cache[key]

        # Check if expired
        if entry.is_expired():
            self.delete(key)
            self.stats.misses += 1
            return None

        # Update access metadata
        entry.last_accessed = time.time()
        entry.access_count += 1

        # Move to end (LRU)
        self._cache.move_to_end(key)

        self.stats.hits += 1
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set value in cache."""
        # Use default TTL if not specified
        if ttl is None:
            ttl = self.default_ttl

        # Check if we need to evict
        if self.max_size is not None and key not in self._cache:
            if len(self._cache) >= self.max_size:
                self._evict_oldest()

        # Create entry
        now = time.time()
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            last_accessed=now,
            access_count=0,
            ttl=ttl,
            metadata=metadata or {},
        )

        self._cache[key] = entry
        self._cache.move_to_end(key)
        self.stats.size = len(self._cache)

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            self.stats.size = len(self._cache)
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self.stats.size = 0

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if key not in self._cache:
            return False

        entry = self._cache[key]
        if entry.is_expired():
            self.delete(key)
            return False

        return True

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def keys(self) -> List[str]:
        """Get all cache keys."""
        return list(self._cache.keys())

    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) entry."""
        if self._cache:
            self._cache.popitem(last=False)
            self.stats.evictions += 1
            self.stats.size = len(self._cache)

    def get_entry(self, key: str) -> Optional[CacheEntry]:
        """Get full cache entry with metadata."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if entry.is_expired():
            self.delete(key)
            return None

        return entry

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.stats = CacheStats(size=len(self._cache))


# ============================================================================
# Disk Cache
# ============================================================================


class DiskCache(BaseCache):
    """Persistent disk-based cache."""

    def __init__(
        self,
        cache_dir: str = ".cache",
        max_size: Optional[int] = None,
        default_ttl: Optional[float] = None,
        serializer: str = "pickle",
    ):
        """Initialize disk cache.

        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
            serializer: Serialization format ('pickle' or 'json')

        Example:
            >>> cache = DiskCache(cache_dir=".cache/llm")
            >>> cache.set("key", {"data": "value"})
            >>> cache.get("key")
            {'data': 'value'}
        """
        super().__init__(max_size, default_ttl)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.serializer = serializer

        # Metadata file
        self.metadata_file = self.cache_dir / "_metadata.json"
        self._metadata: Dict[str, Dict[str, Any]] = self._load_metadata()

        # Clean expired entries on init
        self._clean_expired()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self._metadata, f)

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        # Use hash of key as filename to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        self.stats.total_requests += 1

        if key not in self._metadata:
            self.stats.misses += 1
            return None

        entry_meta = self._metadata[key]

        # Check if expired
        ttl = entry_meta.get("ttl")
        if ttl is not None:
            created_at = entry_meta["created_at"]
            if time.time() - created_at > ttl:
                self.delete(key)
                self.stats.misses += 1
                return None

        # Load from disk
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            # Metadata exists but file doesn't - clean up
            del self._metadata[key]
            self._save_metadata()
            self.stats.misses += 1
            return None

        try:
            if self.serializer == "pickle":
                with open(cache_path, "rb") as f:
                    value = pickle.load(f)
            else:  # json
                with open(cache_path, "r") as f:
                    value = json.load(f)

            # Update access metadata
            entry_meta["last_accessed"] = time.time()
            entry_meta["access_count"] = entry_meta.get("access_count", 0) + 1
            self._save_metadata()

            self.stats.hits += 1
            return value

        except Exception:
            # Failed to load - clean up
            self.delete(key)
            self.stats.misses += 1
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set value in cache."""
        # Use default TTL if not specified
        if ttl is None:
            ttl = self.default_ttl

        # Check if we need to evict
        if self.max_size is not None and key not in self._metadata:
            if len(self._metadata) >= self.max_size:
                self._evict_oldest()

        # Save to disk
        cache_path = self._get_cache_path(key)
        try:
            if self.serializer == "pickle":
                with open(cache_path, "wb") as f:
                    pickle.dump(value, f)
            else:  # json
                with open(cache_path, "w") as f:
                    json.dump(value, f)

            # Update metadata
            now = time.time()
            self._metadata[key] = {
                "created_at": now,
                "last_accessed": now,
                "access_count": 0,
                "ttl": ttl,
                "metadata": metadata or {},
            }
            self._save_metadata()
            self.stats.size = len(self._metadata)

        except Exception as e:
            # Failed to save - clean up
            if cache_path.exists():
                cache_path.unlink()
            raise e

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._metadata:
            # Delete file
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()

            # Delete metadata
            del self._metadata[key]
            self._save_metadata()
            self.stats.size = len(self._metadata)
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        # Delete all cache files
        for key in list(self._metadata.keys()):
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()

        # Clear metadata
        self._metadata.clear()
        self._save_metadata()
        self.stats.size = 0

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if key not in self._metadata:
            return False

        entry_meta = self._metadata[key]
        ttl = entry_meta.get("ttl")
        if ttl is not None:
            created_at = entry_meta["created_at"]
            if time.time() - created_at > ttl:
                self.delete(key)
                return False

        return True

    def size(self) -> int:
        """Get current cache size."""
        return len(self._metadata)

    def keys(self) -> List[str]:
        """Get all cache keys."""
        return list(self._metadata.keys())

    def _evict_oldest(self) -> None:
        """Evict the oldest entry by last access time."""
        if not self._metadata:
            return

        oldest_key = min(
            self._metadata.keys(), key=lambda k: self._metadata[k]["last_accessed"]
        )
        self.delete(oldest_key)
        self.stats.evictions += 1

    def _clean_expired(self) -> None:
        """Remove all expired entries."""
        expired_keys = []
        now = time.time()

        for key, meta in self._metadata.items():
            ttl = meta.get("ttl")
            if ttl is not None:
                created_at = meta["created_at"]
                if now - created_at > ttl:
                    expired_keys.append(key)

        for key in expired_keys:
            self.delete(key)

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.stats = CacheStats(size=len(self._metadata))


# ============================================================================
# Tiered Cache (Memory + Disk)
# ============================================================================


class TieredCache(BaseCache):
    """Two-tier cache: fast memory cache backed by persistent disk cache."""

    def __init__(
        self,
        memory_max_size: int = 100,
        disk_cache_dir: str = ".cache",
        disk_max_size: Optional[int] = None,
        default_ttl: Optional[float] = None,
    ):
        """Initialize tiered cache.

        Args:
            memory_max_size: Maximum entries in memory cache
            disk_cache_dir: Directory for disk cache
            disk_max_size: Maximum entries in disk cache
            default_ttl: Default TTL in seconds

        Example:
            >>> cache = TieredCache(memory_max_size=50, disk_cache_dir=".cache")
            >>> cache.set("key", "value")
            >>> cache.get("key")  # Fast memory access
            'value'
        """
        super().__init__(max_size=None, default_ttl=default_ttl)
        self.memory_cache = MemoryCache(
            max_size=memory_max_size, default_ttl=default_ttl
        )
        self.disk_cache = DiskCache(
            cache_dir=disk_cache_dir, max_size=disk_max_size, default_ttl=default_ttl
        )

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then disk)."""
        # Try memory first
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        # Try disk
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value)
            return value

        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set value in both caches."""
        if ttl is None:
            ttl = self.default_ttl

        self.memory_cache.set(key, value, ttl, metadata)
        self.disk_cache.set(key, value, ttl, metadata)

    def delete(self, key: str) -> bool:
        """Delete key from both caches."""
        mem_deleted = self.memory_cache.delete(key)
        disk_deleted = self.disk_cache.delete(key)
        return mem_deleted or disk_deleted

    def clear(self) -> None:
        """Clear both caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()

    def exists(self, key: str) -> bool:
        """Check if key exists in either cache."""
        return self.memory_cache.exists(key) or self.disk_cache.exists(key)

    def size(self) -> int:
        """Get total unique keys across both caches."""
        mem_keys = set(self.memory_cache.keys())
        disk_keys = set(self.disk_cache.keys())
        return len(mem_keys | disk_keys)

    def keys(self) -> List[str]:
        """Get all unique cache keys."""
        mem_keys = set(self.memory_cache.keys())
        disk_keys = set(self.disk_cache.keys())
        return list(mem_keys | disk_keys)

    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for both caches."""
        return {
            "memory": self.memory_cache.get_stats(),
            "disk": self.disk_cache.get_stats(),
        }

    def reset_stats(self) -> None:
        """Reset statistics for both caches."""
        self.memory_cache.reset_stats()
        self.disk_cache.reset_stats()


# ============================================================================
# LLM-Specific Cache Wrapper
# ============================================================================


class LLMCache:
    """High-level cache wrapper for LLM applications."""

    def __init__(
        self,
        backend: Optional[BaseCache] = None,
        cost_per_token: float = 0.00001,  # Default: ~$0.01 per 1K tokens
        avg_tokens_per_request: int = 1000,
        avg_response_time: float = 2.0,  # seconds
    ):
        """Initialize LLM cache.

        Args:
            backend: Cache backend to use (defaults to MemoryCache)
            cost_per_token: Cost per token for cost tracking
            avg_tokens_per_request: Average tokens per request
            avg_response_time: Average response time in seconds

        Example:
            >>> cache = LLMCache()
            >>> response = cache.get_or_compute(
            ...     key="prompt:123",
            ...     compute_fn=lambda: call_llm("What is AI?"),
            ...     cost=0.001
            ... )
        """
        self.backend = backend or MemoryCache()
        self.cost_per_token = cost_per_token
        self.avg_tokens_per_request = avg_tokens_per_request
        self.avg_response_time = avg_response_time

    def cache_prompt(
        self,
        prompt: str,
        response: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        ttl: Optional[float] = None,
        cost: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Cache an LLM prompt and response.

        Args:
            prompt: The prompt text
            response: The LLM response
            model: Model name
            temperature: Temperature setting
            max_tokens: Max tokens setting
            ttl: Time to live in seconds
            cost: Actual cost of the request
            **kwargs: Additional parameters

        Returns:
            str: The cache key

        Example:
            >>> key = cache.cache_prompt(
            ...     prompt="What is AI?",
            ...     response="AI is...",
            ...     model="gpt-4",
            ...     cost=0.001
            ... )
        """
        key = generate_prompt_key(prompt, model, temperature, max_tokens, **kwargs)

        metadata = {}
        if cost is not None:
            metadata["cost"] = cost

        self.backend.set(key, response, ttl=ttl, metadata=metadata)
        return key

    def get_cached_prompt(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Optional[str]:
        """Get cached LLM response for a prompt.

        Args:
            prompt: The prompt text
            model: Model name
            temperature: Temperature setting
            max_tokens: Max tokens setting
            **kwargs: Additional parameters

        Returns:
            Optional[str]: Cached response or None

        Example:
            >>> response = cache.get_cached_prompt(
            ...     prompt="What is AI?",
            ...     model="gpt-4"
            ... )
        """
        key = generate_prompt_key(prompt, model, temperature, max_tokens, **kwargs)
        cached = self.backend.get(key)

        # Track cost savings
        if cached is not None and isinstance(self.backend, MemoryCache):
            entry = self.backend.get_entry(key)
            if entry:
                cost = entry.metadata.get(
                    "cost", self.cost_per_token * self.avg_tokens_per_request
                )
                self.backend.stats.estimated_cost_saved += cost
                self.backend.stats.estimated_time_saved += self.avg_response_time

        return cached

    def cache_embedding(
        self,
        text: str,
        embedding: List[float],
        model: Optional[str] = None,
        ttl: Optional[float] = None,
        cost: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Cache an embedding.

        Args:
            text: The text that was embedded
            embedding: The embedding vector
            model: Model name
            ttl: Time to live in seconds
            cost: Actual cost of the request
            **kwargs: Additional parameters

        Returns:
            str: The cache key

        Example:
            >>> key = cache.cache_embedding(
            ...     text="Hello world",
            ...     embedding=[0.1, 0.2, ...],
            ...     model="text-embedding-3-small",
            ...     cost=0.00001
            ... )
        """
        key = generate_embedding_key(text, model, **kwargs)

        metadata = {}
        if cost is not None:
            metadata["cost"] = cost

        self.backend.set(key, embedding, ttl=ttl, metadata=metadata)
        return key

    def get_cached_embedding(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> Optional[List[float]]:
        """Get cached embedding for text.

        Args:
            text: The text to get embedding for
            model: Model name
            **kwargs: Additional parameters

        Returns:
            Optional[List[float]]: Cached embedding or None

        Example:
            >>> embedding = cache.get_cached_embedding(
            ...     text="Hello world",
            ...     model="text-embedding-3-small"
            ... )
        """
        key = generate_embedding_key(text, model, **kwargs)
        cached = self.backend.get(key)

        # Track cost savings
        if cached is not None and isinstance(self.backend, MemoryCache):
            entry = self.backend.get_entry(key)
            if entry:
                cost = entry.metadata.get("cost", 0.0001)  # Default embedding cost
                self.backend.stats.estimated_cost_saved += cost
                self.backend.stats.estimated_time_saved += 0.5  # Embeddings are faster

        return cached

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl: Optional[float] = None,
        cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Get from cache or compute if not found.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl: Time to live in seconds
            cost: Cost of computing the value
            metadata: Additional metadata

        Returns:
            Any: Cached or computed value

        Example:
            >>> result = cache.get_or_compute(
            ...     key="expensive:computation",
            ...     compute_fn=lambda: expensive_api_call(),
            ...     ttl=3600,
            ...     cost=0.01
            ... )
        """
        # Try to get from cache
        value = self.backend.get(key)
        if value is not None:
            # Track savings
            if cost is not None and isinstance(self.backend, MemoryCache):
                self.backend.stats.estimated_cost_saved += cost
            return value

        # Compute value
        value = compute_fn()

        # Store in cache
        cache_metadata = metadata or {}
        if cost is not None:
            cache_metadata["cost"] = cost

        self.backend.set(key, value, ttl=ttl, metadata=cache_metadata)

        return value

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        if isinstance(self.backend, (MemoryCache, DiskCache)):
            return self.backend.get_stats()
        elif isinstance(self.backend, TieredCache):
            stats_dict = self.backend.get_stats()
            # Combine stats from both caches
            combined = CacheStats()
            combined.hits = stats_dict["memory"].hits + stats_dict["disk"].hits
            combined.misses = stats_dict["memory"].misses + stats_dict["disk"].misses
            combined.evictions = (
                stats_dict["memory"].evictions + stats_dict["disk"].evictions
            )
            combined.total_requests = stats_dict["memory"].total_requests
            combined.estimated_cost_saved = (
                stats_dict["memory"].estimated_cost_saved
                + stats_dict["disk"].estimated_cost_saved
            )
            combined.estimated_time_saved = (
                stats_dict["memory"].estimated_time_saved
                + stats_dict["disk"].estimated_time_saved
            )
            combined.size = self.backend.size()
            return combined
        return self.backend.stats

    def clear(self) -> None:
        """Clear all cache entries."""
        self.backend.clear()

    def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all keys with a given prefix.

        Args:
            prefix: Key prefix to invalidate

        Returns:
            int: Number of keys invalidated

        Example:
            >>> cache.invalidate_by_prefix("prompt:")
            42
        """
        keys = self.backend.keys()
        invalidated = 0

        for key in keys:
            if key.startswith(prefix):
                if self.backend.delete(key):
                    invalidated += 1

        return invalidated

    def invalidate_by_pattern(self, pattern: Callable[[str], bool]) -> int:
        """Invalidate keys matching a pattern function.

        Args:
            pattern: Function that returns True for keys to invalidate

        Returns:
            int: Number of keys invalidated

        Example:
            >>> cache.invalidate_by_pattern(lambda k: "gpt-3" in k)
            15
        """
        keys = self.backend.keys()
        invalidated = 0

        for key in keys:
            if pattern(key):
                if self.backend.delete(key):
                    invalidated += 1

        return invalidated
