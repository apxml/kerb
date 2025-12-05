"""Utility functions for memory management.

This module provides utility functions for working with conversation buffers:
- format_messages: Format messages for display or export
- filter_messages: Filter messages by various criteria
- merge_conversations: Merge multiple conversation buffers
"""

import json
from typing import TYPE_CHECKING, List, Optional

from kerb.core.types import Message

from .classes import Entity
from .entities import merge_entities

if TYPE_CHECKING:
    from .buffers import ConversationBuffer


def format_messages(
    messages: List[Message],
    format_style: str = "simple",
    include_metadata: bool = False,
) -> str:
    """Format messages for display or export.

    Args:
        messages: Messages to format
        format_style: "simple", "detailed", "json", or "chat"
        include_metadata: Whether to include metadata

    Returns:
        str: Formatted messages

    Example:
        >>> formatted = format_messages(messages, format_style="chat")
    """
    if not messages:
        return ""

    if format_style == "simple":
        lines = [f"{m.role}: {m.content}" for m in messages]
        return "\n".join(lines)

    elif format_style == "detailed":
        lines = []
        for msg in messages:
            lines.append(f"[{msg.timestamp}] {msg.role}:")
            lines.append(f"  {msg.content}")
            if include_metadata and msg.metadata:
                lines.append(f"  Metadata: {msg.metadata}")
        return "\n".join(lines)

    elif format_style == "json":
        data = [m.to_dict() for m in messages]
        return json.dumps(data, indent=2)

    elif format_style == "chat":
        lines = []
        for msg in messages:
            if msg.role == "user":
                lines.append(f"👤 User: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"🤖 Assistant: {msg.content}")
            elif msg.role == "system":
                lines.append(f"⚙️  System: {msg.content}")
            else:
                lines.append(f"{msg.role}: {msg.content}")
        return "\n".join(lines)

    return "\n".join(f"{m.role}: {m.content}" for m in messages)


def filter_messages(
    messages: List[Message],
    role: Optional[str] = None,
    contains: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> List[Message]:
    """Filter messages by various criteria.

    Args:
        messages: Messages to filter
        role: Filter by role
        contains: Filter by content substring
        start_time: Filter by start time (ISO format)
        end_time: Filter by end time (ISO format)

    Returns:
        List[Message]: Filtered messages

    Example:
        >>> user_messages = filter_messages(messages, role="user")
    """
    filtered = messages

    if role:
        filtered = [m for m in filtered if m.role == role]

    if contains:
        contains_lower = contains.lower()
        filtered = [m for m in filtered if contains_lower in m.content.lower()]

    if start_time:
        filtered = [m for m in filtered if m.timestamp >= start_time]

    if end_time:
        filtered = [m for m in filtered if m.timestamp <= end_time]

    return filtered


def merge_conversations(
    *buffers: "ConversationBuffer", sort_by_time: bool = True
) -> "ConversationBuffer":
    """Merge multiple conversation buffers.

    Args:
        *buffers: Conversation buffers to merge
        sort_by_time: Whether to sort merged messages by timestamp

    Returns:
        ConversationBuffer: Merged buffer

    Example:
        >>> merged = merge_conversations(buffer1, buffer2)
    """
    from .buffers import ConversationBuffer

    merged = ConversationBuffer()

    # Merge messages
    all_messages = []
    for buffer in buffers:
        all_messages.extend(buffer.messages)

    if sort_by_time:
        all_messages.sort(key=lambda m: m.timestamp)

    merged.messages = all_messages

    # Merge summaries
    for buffer in buffers:
        merged.summaries.extend(buffer.summaries)

    # Merge entities
    for buffer in buffers:
        for key, entity in buffer.entities.items():
            if key in merged.entities:
                merged.entities[key] = merge_entities(merged.entities[key], entity)
            else:
                merged.entities[key] = entity

    return merged
