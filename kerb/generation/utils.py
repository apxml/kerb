"""Utility functions and classes for LLM generation.

This module provides helper utilities for generation, including rate limiting,
caching, cost tracking, retry logic, and response validation.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from kerb.core.types import Message, MessageRole
from kerb.parsing import ParseMode, extract_json

from .config import GenerationConfig, GenerationResponse, Usage
from .enums import MODEL_PRICING, LLMProvider, ModelName

# ============================================================================
# Rate Limiting
# ============================================================================


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(
        self, requests_per_minute: int = 60, tokens_per_minute: Optional[int] = None
    ):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute (optional)
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_times: List[float] = []
        self.token_counts: List[Tuple[float, int]] = []

    def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Wait if rate limit would be exceeded.

        Args:
            estimated_tokens: Estimated token count for this request
        """
        current_time = time.time()

        # Clean old request times (older than 1 minute)
        self.request_times = [t for t in self.request_times if current_time - t < 60]

        # Check request rate limit
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                current_time = time.time()
                self.request_times = [
                    t for t in self.request_times if current_time - t < 60
                ]

        # Check token rate limit if applicable
        if self.tokens_per_minute and estimated_tokens > 0:
            self.token_counts = [
                (t, c) for t, c in self.token_counts if current_time - t < 60
            ]
            total_tokens = sum(c for _, c in self.token_counts)

            if total_tokens + estimated_tokens > self.tokens_per_minute:
                sleep_time = 60 - (current_time - self.token_counts[0][0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    current_time = time.time()
                    self.token_counts = [
                        (t, c) for t, c in self.token_counts if current_time - t < 60
                    ]

        # Record this request
        self.request_times.append(current_time)
        if estimated_tokens > 0 and self.tokens_per_minute:
            self.token_counts.append((current_time, estimated_tokens))


# ============================================================================
# Response Cache
# ============================================================================


class ResponseCache:
    """Simple in-memory cache for LLM responses."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """Initialize response cache.

        Args:
            max_size: Maximum number of cached responses
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[GenerationResponse, float]] = {}

    def _make_key(self, messages: List[Message], config: GenerationConfig) -> str:
        """Create cache key from messages and config."""
        from dataclasses import asdict

        messages_str = json.dumps([m.to_dict() for m in messages], sort_keys=True)
        config_str = json.dumps(asdict(config), sort_keys=True)
        key_data = f"{messages_str}:{config_str}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(
        self, messages: List[Message], config: GenerationConfig
    ) -> Optional[GenerationResponse]:
        """Get cached response if available and not expired."""
        key = self._make_key(messages, config)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                # Mark as cached
                response.cached = True
                return response
            else:
                del self.cache[key]
        return None

    def set(
        self,
        messages: List[Message],
        config: GenerationConfig,
        response: GenerationResponse,
    ) -> None:
        """Cache a response."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        key = self._make_key(messages, config)
        self.cache[key] = (response, time.time())


# ============================================================================
# Cost Tracking
# ============================================================================


class CostTracker:
    """Track costs across LLM API calls."""

    def __init__(self):
        """Initialize cost tracker."""
        self.total_cost = 0.0
        self.total_tokens = 0
        self.requests_by_model: Dict[str, int] = {}
        self.cost_by_model: Dict[str, float] = {}
        self.tokens_by_model: Dict[str, int] = {}

    def add_request(self, model: str, usage: Usage, cost: float) -> None:
        """Record a request."""
        self.total_cost += cost
        self.total_tokens += usage.total_tokens
        self.requests_by_model[model] = self.requests_by_model.get(model, 0) + 1
        self.cost_by_model[model] = self.cost_by_model.get(model, 0.0) + cost
        self.tokens_by_model[model] = (
            self.tokens_by_model.get(model, 0) + usage.total_tokens
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        return {
            "total_cost": round(self.total_cost, 4),
            "total_tokens": self.total_tokens,
            "total_requests": sum(self.requests_by_model.values()),
            "by_model": {
                model: {
                    "requests": self.requests_by_model[model],
                    "cost": round(self.cost_by_model[model], 4),
                    "tokens": self.tokens_by_model[model],
                }
                for model in self.requests_by_model
            },
        }

    def reset(self) -> None:
        """Reset all tracking."""
        self.total_cost = 0.0
        self.total_tokens = 0
        self.requests_by_model.clear()
        self.cost_by_model.clear()
        self.tokens_by_model.clear()


# Global cost tracker instance
# Users can import and use this explicitly: from kerb.generation import global_cost_tracker
_global_cost_tracker = CostTracker()
global_cost_tracker = _global_cost_tracker


# ============================================================================
# Cost Calculation
# ============================================================================


def calculate_cost(model: Union[str, ModelName], usage: Usage) -> float:
    """Calculate cost for a request.

    Args:
        model: Model name (as string or ModelName enum)
        usage: Token usage

    Returns:
        float: Cost in USD
    """
    # Convert ModelName enum to string
    model_str = model.value if isinstance(model, ModelName) else model

    if model_str not in MODEL_PRICING:
        return 0.0

    input_price, output_price = MODEL_PRICING[model_str]
    input_cost = (usage.prompt_tokens / 1_000_000) * input_price
    output_cost = (usage.completion_tokens / 1_000_000) * output_price
    return input_cost + output_cost


# ============================================================================
# Retry Logic
# ============================================================================


def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
) -> Any:
    """Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter
        retryable_exceptions: Exceptions that trigger retry

    Returns:
        Result from function

    Raises:
        Last exception if all retries fail
    """
    import random

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt == max_retries:
                raise

            delay = initial_delay * (exponential_base**attempt)
            if jitter:
                delay *= 0.5 + random.random()

            time.sleep(delay)

    raise last_exception


async def async_retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
) -> Any:
    """Async version of retry with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter
        retryable_exceptions: Exceptions that trigger retry

    Returns:
        Result from function

    Raises:
        Last exception if all retries fail
    """
    import random

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt == max_retries:
                raise

            delay = initial_delay * (exponential_base**attempt)
            if jitter:
                delay *= 0.5 + random.random()

            await asyncio.sleep(delay)

    raise last_exception


# ============================================================================
# Response Validation
# ============================================================================


def parse_json_response(response: Union[GenerationResponse, str]) -> Dict[str, Any]:
    """Parse JSON from LLM response.

    Handles markdown code blocks and other formatting.

    Args:
        response: GenerationResponse or content string

    Returns:
        Dict[str, Any]: Parsed JSON

    Raises:
        ValueError: If JSON cannot be parsed

    Example:
        >>> response = generate("Return JSON", model="gpt-4o-mini",
        ...                     provider=LLMProvider.OPENAI,
        ...                     response_format={"type": "json_object"})
        >>> data = parse_json_response(response)
    """
    content = response.content if isinstance(response, GenerationResponse) else response

    result = extract_json(content, mode=ParseMode.BEST_EFFORT)

    if result.success:
        return result.data
    else:
        raise ValueError(
            f"Failed to parse JSON from response: {result.error}\nContent: {content[:200]}"
        )


def validate_response(
    response: GenerationResponse,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    must_contain: Optional[List[str]] = None,
    must_not_contain: Optional[List[str]] = None,
    pattern: Optional[str] = None,
) -> bool:
    """Validate LLM response against criteria.

    Args:
        response: Generation response
        min_length: Minimum content length
        max_length: Maximum content length
        must_contain: Strings that must be present
        must_not_contain: Strings that must not be present
        pattern: Regex pattern that must match

    Returns:
        bool: True if valid, False otherwise

    Example:
        >>> response = generate("List 3 programming languages",
        ...                     model="gpt-4o-mini", provider=LLMProvider.OPENAI)
        >>> is_valid = validate_response(response, min_length=20, must_contain=["Python"])
    """
    content = response.content

    if min_length and len(content) < min_length:
        return False

    if max_length and len(content) > max_length:
        return False

    if must_contain:
        for text in must_contain:
            if text not in content:
                return False

    if must_not_contain:
        for text in must_not_contain:
            if text in content:
                return False

    if pattern:
        import re

        if not re.search(pattern, content):
            return False

    return True


# ============================================================================
# Cost Tracking Helpers
# ============================================================================


def get_cost_summary(cost_tracker: Optional[CostTracker] = None) -> Dict[str, Any]:
    """Get cost tracking summary.

    Args:
        cost_tracker: CostTracker instance. If None, uses global tracker.

    Returns:
        Dict[str, Any]: Cost summary with totals and per-model breakdown

    Example:
        >>> generate("Hello", model="gpt-4o-mini", provider=LLMProvider.OPENAI, track_cost=True)
        >>> summary = get_cost_summary()
        >>> print(f"Total cost: ${summary['total_cost']}")
    """
    tracker = cost_tracker if cost_tracker else _global_cost_tracker
    return tracker.get_summary()


def reset_cost_tracking(cost_tracker: Optional[CostTracker] = None) -> None:
    """Reset cost tracking.

    Args:
        cost_tracker: CostTracker instance. If None, resets global tracker.

    Example:
        >>> reset_cost_tracking()
    """
    tracker = cost_tracker if cost_tracker else _global_cost_tracker
    tracker.reset()


# ============================================================================
# Message Formatting
# ============================================================================


def format_messages(
    system: Optional[str] = None,
    user: Optional[str] = None,
    assistant: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Message]:
    """Format messages for generation.

    Args:
        system: System message
        user: User message
        assistant: Assistant message (for few-shot examples)
        history: Conversation history as list of {"role": "...", "content": "..."}

    Returns:
        List[Message]: Formatted messages

    Example:
        >>> messages = format_messages(system="You are helpful", user="What is Python?")
        >>> response = generate(messages, model="gpt-4o-mini", provider=LLMProvider.OPENAI)
    """
    messages = []

    if history:
        for msg in history:
            messages.append(Message(role=msg["role"], content=msg["content"]))

    if system:
        messages.insert(0, Message(role=MessageRole.SYSTEM, content=system))

    if user:
        messages.append(Message(role=MessageRole.USER, content=user))

    if assistant:
        messages.append(Message(role=MessageRole.ASSISTANT, content=assistant))

    return messages


# ============================================================================
# Batch Generation Utility
# ============================================================================


def batch_generate(
    prompts: List[str], model: str = "gpt-4o-mini", **kwargs
) -> List[GenerationResponse]:
    """Batch generate responses for multiple prompts.

    Args:
        prompts: List of prompt strings
        model: Model to use
        **kwargs: Additional generation parameters (including provider)

    Returns:
        List of GenerationResponse objects

    Example:
        >>> prompts = ["Hello", "How are you?"]
        >>> responses = batch_generate(prompts, model="gpt-4o-mini",
        ...                            provider=LLMProvider.OPENAI)
    """
    from kerb.core.types import Message, MessageRole

    from .generator import generate_batch

    messages_list = [
        [Message(role=MessageRole.USER, content=prompt)] for prompt in prompts
    ]
    return generate_batch(messages_list, model=model, **kwargs)
