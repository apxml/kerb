"""Core context window management functions.

This module provides functions for creating and managing context windows,
including truncation strategies.
"""

from typing import List, Union, Optional, Callable

from kerb.tokenizer import count_tokens, Tokenizer
from .types import ContextItem, ContextWindow, TruncationStrategy


def create_context_window(
    items: Union[List[str], List[ContextItem]],
    max_tokens: Optional[int] = None,
    strategy: TruncationStrategy = TruncationStrategy.LAST,
    token_estimator: Optional[Callable[[str], int]] = None
) -> ContextWindow:
    """Create a managed context window from items.
    
    Args:
        items: List of strings or ContextItem objects
        max_tokens: Maximum tokens allowed in window
        strategy: Truncation strategy if limit exceeded
        token_estimator: Custom token estimation function (defaults to count_tokens from tokenizer)
        
    Returns:
        ContextWindow: Managed context window
        
    Example:
        >>> window = create_context_window(["Hello", "World"], max_tokens=1000)
        >>> print(window.current_tokens)
    """
    # Use count_tokens from tokenizer module as default
    estimator = token_estimator or (lambda text: count_tokens(text, Tokenizer.CL100K_BASE))
    
    # Convert strings to ContextItems if needed
    context_items = []
    for item in items:
        if isinstance(item, str):
            token_count = estimator(item)
            context_items.append(ContextItem(
                content=item,
                token_count=token_count
            ))
        else:
            if item.token_count is None:
                item.token_count = estimator(item.content)
            context_items.append(item)
    
    window = ContextWindow(
        items=context_items,
        max_tokens=max_tokens,
        strategy=strategy
    )
    
    # Calculate total tokens
    window.current_tokens = sum(item.token_count or 0 for item in context_items)
    
    # Apply truncation if needed
    if max_tokens and window.current_tokens > max_tokens:
        window = truncate_context_window(window, max_tokens, strategy)
    
    return window


def truncate_context_window(
    window: ContextWindow,
    max_tokens: int,
    strategy: TruncationStrategy = TruncationStrategy.LAST
) -> ContextWindow:
    """Truncate context window to fit within token limit.
    
    Args:
        window: Context window to truncate
        max_tokens: Maximum tokens allowed
        strategy: Truncation strategy to use
        
    Returns:
        ContextWindow: Truncated context window
        
    Example:
        >>> window = truncate_context_window(window, max_tokens=500)
    """
    if window.current_tokens <= max_tokens:
        return window
    
    if strategy == TruncationStrategy.FIRST:
        return _truncate_first(window, max_tokens)
    elif strategy == TruncationStrategy.LAST:
        return _truncate_last(window, max_tokens)
    elif strategy == TruncationStrategy.MIDDLE:
        return _truncate_middle(window, max_tokens)
    elif strategy == TruncationStrategy.PRIORITY:
        return _truncate_priority(window, max_tokens)
    else:
        return _truncate_last(window, max_tokens)


def _truncate_first(window: ContextWindow, max_tokens: int) -> ContextWindow:
    """Keep first items up to token limit."""
    kept_items = []
    current = 0
    
    for item in window.items:
        if current + (item.token_count or 0) <= max_tokens:
            kept_items.append(item)
            current += item.token_count or 0
        else:
            break
    
    return ContextWindow(
        items=kept_items,
        max_tokens=max_tokens,
        current_tokens=current,
        strategy=window.strategy,
        metadata=window.metadata
    )


def _truncate_last(window: ContextWindow, max_tokens: int) -> ContextWindow:
    """Keep last items up to token limit."""
    kept_items = []
    current = 0
    
    for item in reversed(window.items):
        if current + (item.token_count or 0) <= max_tokens:
            kept_items.insert(0, item)
            current += item.token_count or 0
        else:
            break
    
    return ContextWindow(
        items=kept_items,
        max_tokens=max_tokens,
        current_tokens=current,
        strategy=window.strategy,
        metadata=window.metadata
    )


def _truncate_middle(window: ContextWindow, max_tokens: int) -> ContextWindow:
    """Keep start and end items, remove middle."""
    if not window.items:
        return window
    
    # Allocate half to start, half to end
    start_tokens = max_tokens // 2
    end_tokens = max_tokens - start_tokens
    
    # Get start items
    start_items = []
    current = 0
    for item in window.items:
        if current + (item.token_count or 0) <= start_tokens:
            start_items.append(item)
            current += item.token_count or 0
        else:
            break
    
    # Get end items
    end_items = []
    current = 0
    for item in reversed(window.items):
        if current + (item.token_count or 0) <= end_tokens:
            end_items.insert(0, item)
            current += item.token_count or 0
        else:
            break
    
    # Combine
    kept_items = start_items + end_items
    total_tokens = sum(item.token_count or 0 for item in kept_items)
    
    return ContextWindow(
        items=kept_items,
        max_tokens=max_tokens,
        current_tokens=total_tokens,
        strategy=window.strategy,
        metadata=window.metadata
    )


def _truncate_priority(window: ContextWindow, max_tokens: int) -> ContextWindow:
    """Keep highest priority items up to token limit."""
    # Sort by priority (descending)
    sorted_items = sorted(window.items, key=lambda x: x.priority, reverse=True)
    
    kept_items = []
    current = 0
    
    for item in sorted_items:
        if current + (item.token_count or 0) <= max_tokens:
            kept_items.append(item)
            current += item.token_count or 0
    
    # Restore original order for kept items
    original_order = {id(item): i for i, item in enumerate(window.items)}
    kept_items.sort(key=lambda x: original_order.get(id(x), 0))
    
    return ContextWindow(
        items=kept_items,
        max_tokens=max_tokens,
        current_tokens=current,
        strategy=window.strategy,
        metadata=window.metadata
    )
