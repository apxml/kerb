"""Context formatting utilities.

This module provides functions for formatting context windows for various
consumption patterns, including LLM inputs and chat message formats.
"""

from collections import defaultdict
from typing import Dict, List, Optional

from .types import ContextWindow


def format_context_window(
    window: ContextWindow,
    format_template: Optional[str] = None,
    include_metadata: bool = False,
) -> str:
    """Format context window for LLM consumption.

    Args:
        window: Context window to format
        format_template: Custom format template
        include_metadata: Whether to include item metadata

    Returns:
        str: Formatted context string

    Example:
        >>> formatted = format_context_window(window)
    """
    if format_template:
        # Use custom template
        result = []
        for i, item in enumerate(window.items):
            formatted_item = format_template.format(
                index=i,
                content=item.content,
                priority=item.priority,
                type=item.item_type,
                tokens=item.token_count or 0,
            )
            result.append(formatted_item)
        return "\n".join(result)

    # Default formatting
    result = []
    for item in window.items:
        if include_metadata:
            header = f"[{item.item_type.upper()}] (priority: {item.priority:.2f})"
            result.append(f"{header}\n{item.content}\n")
        else:
            result.append(item.content)

    return "\n\n".join(result)


def context_to_messages(
    window: ContextWindow, system_prefix: Optional[str] = None
) -> List[Dict[str, str]]:
    """Convert context window to chat message format.

    Args:
        window: Context window to convert
        system_prefix: Optional system message prefix

    Returns:
        List[Dict[str, str]]: List of message dictionaries

    Example:
        >>> messages = context_to_messages(window, system_prefix="You are a helpful assistant.")
    """
    messages = []

    if system_prefix:
        messages.append({"role": "system", "content": system_prefix})

    for item in window.items:
        # Determine role based on item type
        role = item.metadata.get("role", "user")
        messages.append({"role": role, "content": item.content})

    return messages


def extract_context_summary(window: ContextWindow) -> str:
    """Extract summary of context window contents.

    Args:
        window: Context window to summarize

    Returns:
        str: Summary of context window

    Example:
        >>> summary = extract_context_summary(window)
        >>> print(summary)
    """
    num_items = len(window.items)
    total_tokens = window.current_tokens

    # Count by type
    type_counts = defaultdict(int)
    for item in window.items:
        type_counts[item.item_type] += 1

    # Format summary
    summary_parts = [
        f"Context Window Summary:",
        f"  Total items: {num_items}",
        f"  Total tokens: {total_tokens}",
    ]

    if window.max_tokens:
        utilization = (total_tokens / window.max_tokens) * 100
        summary_parts.append(f"  Token utilization: {utilization:.1f}%")

    if type_counts:
        summary_parts.append("  Item types:")
        for item_type, count in sorted(type_counts.items()):
            summary_parts.append(f"    - {item_type}: {count}")

    return "\n".join(summary_parts)
