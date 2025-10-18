"""Sliding window utilities for context management.

This module provides functions for creating different types of sliding windows
over context items.
"""

from typing import List, Optional
from collections import deque

from .types import ContextItem, ContextWindow


def create_sliding_window(
    items: List[ContextItem],
    window_size: int,
    step_size: Optional[int] = None
) -> List[ContextWindow]:
    """Create sliding windows over context items.
    
    Args:
        items: List of context items
        window_size: Number of items per window
        step_size: Step size between windows (defaults to window_size)
        
    Returns:
        List[ContextWindow]: List of sliding windows
        
    Example:
        >>> windows = create_sliding_window(items, window_size=3, step_size=1)
    """
    if not items:
        return []
    
    step = step_size or window_size
    windows = []
    
    for i in range(0, len(items), step):
        window_items = items[i:i + window_size]
        if window_items:
            tokens = sum(item.token_count or 0 for item in window_items)
            windows.append(ContextWindow(
                items=window_items,
                current_tokens=tokens
            ))
    
    return windows


def create_token_sliding_window(
    items: List[ContextItem],
    max_tokens: int,
    overlap_tokens: int = 0
) -> List[ContextWindow]:
    """Create sliding windows based on token limits.
    
    Args:
        items: List of context items
        max_tokens: Maximum tokens per window
        overlap_tokens: Number of tokens to overlap between windows
        
    Returns:
        List[ContextWindow]: List of token-based sliding windows
        
    Example:
        >>> windows = create_token_sliding_window(items, max_tokens=500, overlap_tokens=50)
    """
    if not items:
        return []
    
    windows = []
    current_window = []
    current_tokens = 0
    overlap_buffer = deque()
    overlap_buffer_tokens = 0
    
    for item in items:
        item_tokens = item.token_count or 0
        
        # Check if item fits in current window
        if current_tokens + item_tokens <= max_tokens:
            current_window.append(item)
            current_tokens += item_tokens
            
            # Add to overlap buffer
            overlap_buffer.append(item)
            overlap_buffer_tokens += item_tokens
            
            # Trim overlap buffer
            while overlap_buffer_tokens > overlap_tokens and overlap_buffer:
                removed = overlap_buffer.popleft()
                overlap_buffer_tokens -= removed.token_count or 0
        else:
            # Save current window
            if current_window:
                windows.append(ContextWindow(
                    items=current_window.copy(),
                    max_tokens=max_tokens,
                    current_tokens=current_tokens
                ))
            
            # Start new window with overlap
            current_window = list(overlap_buffer) + [item]
            current_tokens = overlap_buffer_tokens + item_tokens
            
            # Reset overlap buffer
            overlap_buffer.clear()
            overlap_buffer.append(item)
            overlap_buffer_tokens = item_tokens
    
    # Add final window
    if current_window:
        windows.append(ContextWindow(
            items=current_window,
            max_tokens=max_tokens,
            current_tokens=current_tokens
        ))
    
    return windows


def create_adaptive_window(
    items: List[ContextItem],
    max_tokens: int,
    recency_weight: float = 0.5,
    priority_weight: float = 0.5
) -> ContextWindow:
    """Create adaptive window balancing recency and priority.
    
    Args:
        items: List of context items
        max_tokens: Maximum tokens allowed
        recency_weight: Weight for recency (0-1)
        priority_weight: Weight for priority (0-1)
        
    Returns:
        ContextWindow: Adaptively selected context window
        
    Example:
        >>> window = create_adaptive_window(items, max_tokens=1000)
    """
    if not items:
        return ContextWindow(max_tokens=max_tokens)
    
    # Calculate adaptive scores
    for i, item in enumerate(items):
        recency_score = (i + 1) / len(items)  # More recent = higher score
        priority_score = item.priority
        
        item.metadata["adaptive_score"] = (
            recency_weight * recency_score + 
            priority_weight * priority_score
        )
    
    # Sort by adaptive score
    sorted_items = sorted(
        items, 
        key=lambda x: x.metadata.get("adaptive_score", 0), 
        reverse=True
    )
    
    # Select items up to token limit
    selected = []
    current_tokens = 0
    
    for item in sorted_items:
        item_tokens = item.token_count or 0
        if current_tokens + item_tokens <= max_tokens:
            selected.append(item)
            current_tokens += item_tokens
    
    # Restore original order
    original_order = {id(item): i for i, item in enumerate(items)}
    selected.sort(key=lambda x: original_order.get(id(x), 0))
    
    return ContextWindow(
        items=selected,
        max_tokens=max_tokens,
        current_tokens=current_tokens,
        metadata={"adaptive_selection": True}
    )
