"""Cache key generation strategies.

This module provides functions for generating cache keys:
- generate_cache_key: Generic key generation from arguments
- generate_prompt_key: LLM prompt-specific key generation
- generate_embedding_key: Embedding-specific key generation
- cached: Decorator to cache function results
"""

import hashlib
import json
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from kerb.cache.models import BaseCache


def generate_cache_key(
    *args, prefix: str = "", hash_algorithm: str = "sha256", **kwargs
) -> str:
    """Generate a cache key from arguments.

    Args:
        *args: Positional arguments to include in key
        prefix: Optional prefix for the key
        hash_algorithm: Hash algorithm to use (sha256, md5, sha1)
        **kwargs: Keyword arguments to include in key

    Returns:
        str: Generated cache key

    Example:
        >>> key = generate_cache_key("prompt text", model="gpt-4", temp=0.7)
        >>> key = generate_cache_key(prompt, prefix="llm", model=model)
    """
    # Create a deterministic representation
    key_data = {
        "args": args,
        "kwargs": kwargs,
    }

    # Serialize to JSON with sorted keys for consistency
    key_str = json.dumps(key_data, sort_keys=True, default=str)

    # Hash the key string
    if hash_algorithm == "sha256":
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
    elif hash_algorithm == "md5":
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
    elif hash_algorithm == "sha1":
        key_hash = hashlib.sha1(key_str.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")

    # Add prefix if provided
    if prefix:
        return f"{prefix}:{key_hash}"
    return key_hash


def generate_prompt_key(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> str:
    """Generate a cache key specifically for LLM prompts.

    Args:
        prompt: The prompt text
        model: Model name
        temperature: Temperature setting
        max_tokens: Max tokens setting
        **kwargs: Additional parameters

    Returns:
        str: Cache key for the prompt

    Example:
        >>> key = generate_prompt_key("What is AI?", model="gpt-4", temperature=0.7)
    """
    params = {"prompt": prompt}
    if model is not None:
        params["model"] = model
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    params.update(kwargs)

    return generate_cache_key(prefix="prompt", **params)


def generate_embedding_key(text: str, model: Optional[str] = None, **kwargs) -> str:
    """Generate a cache key specifically for embeddings.

    Args:
        text: The text to embed
        model: Model name
        **kwargs: Additional parameters

    Returns:
        str: Cache key for the embedding

    Example:
        >>> key = generate_embedding_key("Hello world", model="text-embedding-3-small")
    """
    params = {"text": text}
    if model is not None:
        params["model"] = model
    params.update(kwargs)

    return generate_cache_key(prefix="embedding", **params)


# ============================================================================
# Cache Decorator
# ============================================================================


def cached(
    cache: Optional["BaseCache"] = None,
    ttl: Optional[float] = None,
    key_fn: Optional[Callable[..., str]] = None,
    cost: Optional[float] = None,
):
    """Decorator to cache function results.

    Args:
        cache: Cache instance to use (creates LLMCache if None)
        ttl: Time to live in seconds
        key_fn: Function to generate cache key from args/kwargs
        cost: Cost of computing the function

    Example:
        >>> @cached(ttl=3600)
        ... def expensive_computation(x, y):
        ...     return x + y

        >>> from kerb.cache.strategies import generate_prompt_key
        >>> @cached(key_fn=lambda prompt, **kw: generate_prompt_key(prompt, **kw))
        ... def call_llm(prompt, model="gpt-4", **kwargs):
        ...     return make_api_call(prompt, model, **kwargs)
    """
    # Import here to avoid circular dependency
    if cache is None:
        from kerb.cache.backends import LLMCache

        cache = LLMCache()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                key = generate_cache_key(*args, **kwargs)

            # Check if the cache is an LLMCache (has get_or_compute)
            if hasattr(cache, "get_or_compute"):
                return cache.get_or_compute(
                    key=key,
                    compute_fn=lambda: func(*args, **kwargs),
                    ttl=ttl,
                    cost=cost,
                )
            else:
                # For other cache types, implement basic get/set logic
                result = cache.get(key)
                if result is not None:
                    return result
                result = func(*args, **kwargs)
                cache.set(key, result, ttl=ttl)
                return result

        return wrapper

    return decorator
