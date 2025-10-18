"""Context optimization utilities.

This module provides functions for optimizing context windows through
deduplication, reordering, merging, and query-specific optimization.
"""

from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Union

from .priority import priority_by_diversity, priority_by_relevance
from .types import ContextItem, ContextWindow
from .window import truncate_context_window

if TYPE_CHECKING:
    from kerb.core.enums import ReorderStrategy


def deduplicate_context(
    items: List[ContextItem], similarity_threshold: float = 0.9
) -> List[ContextItem]:
    """Remove duplicate or highly similar context items.

    Args:
        items: List of context items
        similarity_threshold: Threshold for considering items duplicates (0-1)

    Returns:
        List[ContextItem]: Deduplicated items

    Example:
        >>> unique_items = deduplicate_context(items, similarity_threshold=0.85)
    """
    if not items:
        return []

    unique_items = [items[0]]

    for item in items[1:]:
        is_duplicate = False

        # Check against all unique items
        for unique_item in unique_items:
            # Simple word-based similarity
            words1 = set(item.content.lower().split())
            words2 = set(unique_item.content.lower().split())

            if not words1 or not words2:
                continue

            overlap = len(words1 & words2)
            union = len(words1 | words2)
            similarity = overlap / union if union > 0 else 0.0

            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_items.append(item)

    return unique_items


def reorder_context(
    items: List[ContextItem], strategy: Union["ReorderStrategy", str] = "chronological"
) -> List[ContextItem]:
    """Reorder context items using specified strategy.

    Args:
        items: List of context items
        strategy: Reordering strategy (ReorderStrategy enum or string: "chronological", "priority", "relevance", "alternating")

    Returns:
        List[ContextItem]: Reordered items

    Examples:
        >>> # Using enum (recommended)
        >>> from kerb.core.enums import ReorderStrategy
        >>> reordered = reorder_context(items, strategy=ReorderStrategy.PRIORITY)

        >>> # Using string (for backward compatibility)
        >>> reordered = reorder_context(items, strategy="priority")
    """
    from kerb.core.enums import ReorderStrategy, validate_enum_or_string

    # Validate and normalize strategy
    strategy_val = validate_enum_or_string(strategy, ReorderStrategy, "strategy")
    if isinstance(strategy_val, ReorderStrategy):
        strategy_str = strategy_val.value
    else:
        strategy_str = strategy_val

    if strategy_str == "chronological":
        return sorted(items, key=lambda x: x.timestamp or 0)
    elif strategy_str == "priority":
        return sorted(items, key=lambda x: x.priority, reverse=True)
    elif strategy_str == "relevance":
        return sorted(items, key=lambda x: x.priority, reverse=True)
    elif strategy_str == "alternating":
        # Alternate between different item types
        type_groups = defaultdict(list)
        for item in items:
            type_groups[item.item_type].append(item)

        result = []
        max_len = max(len(group) for group in type_groups.values())

        for i in range(max_len):
            for item_type in sorted(type_groups.keys()):
                if i < len(type_groups[item_type]):
                    result.append(type_groups[item_type][i])

        return result
    else:
        return items


def merge_context_windows(
    windows: List[ContextWindow],
    max_tokens: Optional[int] = None,
    deduplication: bool = True,
) -> ContextWindow:
    """Merge multiple context windows into one.

    Args:
        windows: List of context windows to merge
        max_tokens: Maximum tokens for merged window
        deduplication: Whether to deduplicate items

    Returns:
        ContextWindow: Merged context window

    Example:
        >>> merged = merge_context_windows([window1, window2], max_tokens=2000)
    """
    if not windows:
        return ContextWindow(max_tokens=max_tokens)

    # Collect all items
    all_items = []
    for window in windows:
        all_items.extend(window.items)

    # Deduplicate if requested
    if deduplication:
        all_items = deduplicate_context(all_items)

    # Create merged window
    total_tokens = sum(item.token_count or 0 for item in all_items)

    merged = ContextWindow(
        items=all_items, max_tokens=max_tokens, current_tokens=total_tokens
    )

    # Truncate if needed
    if max_tokens and total_tokens > max_tokens:
        merged = truncate_context_window(merged, max_tokens)

    return merged


def optimize_context_for_query(
    window: ContextWindow,
    query: str,
    max_tokens: int,
    relevance_weight: float = 0.7,
    diversity_weight: float = 0.3,
) -> ContextWindow:
    """Optimize context window for a specific query.

    Args:
        window: Context window to optimize
        query: Query to optimize for
        max_tokens: Maximum tokens allowed
        relevance_weight: Weight for relevance scoring
        diversity_weight: Weight for diversity scoring

    Returns:
        ContextWindow: Optimized context window

    Example:
        >>> optimized = optimize_context_for_query(window, "What is AI?", max_tokens=1000)
    """
    items = window.items.copy()

    # Assign relevance scores
    items = priority_by_relevance(items, query)
    relevance_scores = {id(item): item.priority for item in items}

    # Assign diversity scores
    items = priority_by_diversity(items)
    diversity_scores = {id(item): item.priority for item in items}

    # Combine scores
    for item in items:
        relevance = relevance_scores.get(id(item), 0)
        diversity = diversity_scores.get(id(item), 0)
        item.priority = relevance_weight * relevance + diversity_weight * diversity

    # Sort by combined score
    items.sort(key=lambda x: x.priority, reverse=True)

    # Select items up to token limit
    selected = []
    current_tokens = 0

    for item in items:
        item_tokens = item.token_count or 0
        if current_tokens + item_tokens <= max_tokens:
            selected.append(item)
            current_tokens += item_tokens

    # Restore chronological order for selected items
    original_order = {id(item): i for i, item in enumerate(window.items)}
    selected.sort(key=lambda x: original_order.get(id(x), 0))

    return ContextWindow(
        items=selected,
        max_tokens=max_tokens,
        current_tokens=current_tokens,
        metadata={**window.metadata, "query_optimized": True},
    )
