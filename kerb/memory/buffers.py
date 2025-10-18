"""Conversation buffer and sliding window implementations.

This module provides:
- ConversationBuffer: Main class for managing conversation memory
- Sliding window functions for recent context management
"""

import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from kerb.core.types import Message

from .classes import ConversationSummary, Entity
from .entities import extract_entities, merge_entities
from .summaries import summarize_conversation

if TYPE_CHECKING:
    from kerb.core.enums import PruneStrategy


def create_sliding_window(
    messages: List[Message], window_size: int = 10, include_system: bool = True
) -> List[Message]:
    """Create a sliding window of recent messages.

    Args:
        messages: List of conversation messages
        window_size: Number of recent messages to keep
        include_system: Whether to include system messages in the window

    Returns:
        List[Message]: Most recent messages within window

    Example:
        >>> messages = [Message("user", "Hello"), Message("assistant", "Hi")]
        >>> recent = create_sliding_window(messages, window_size=5)
    """
    if not messages:
        return []

    if include_system:
        return messages[-window_size:]
    else:
        # Filter out system messages
        non_system = [m for m in messages if m.role != "system"]
        return non_system[-window_size:]


def create_token_limited_window(
    messages: List[Message],
    max_tokens: int = 2000,
    token_estimator: Optional[Callable[[str], int]] = None,
) -> List[Message]:
    """Create a sliding window limited by token count.

    Args:
        messages: List of conversation messages
        max_tokens: Maximum total tokens
        token_estimator: Function to estimate tokens (defaults to word count / 0.75)

    Returns:
        List[Message]: Most recent messages that fit within token limit

    Example:
        >>> window = create_token_limited_window(messages, max_tokens=1000)
    """
    if not messages:
        return []

    if token_estimator is None:
        # Simple token estimation: roughly 1 token per 0.75 words
        token_estimator = lambda text: int(len(text.split()) / 0.75)

    result = []
    total_tokens = 0

    # Work backwards from most recent
    for message in reversed(messages):
        message_tokens = token_estimator(message.content)
        if total_tokens + message_tokens <= max_tokens:
            result.insert(0, message)
            total_tokens += message_tokens
        else:
            break

    return result


def create_alternating_window(messages: List[Message], pairs: int = 5) -> List[Message]:
    """Create a window with alternating user/assistant pairs.

    Args:
        messages: List of conversation messages
        pairs: Number of user-assistant pairs to keep

    Returns:
        List[Message]: Recent alternating message pairs

    Example:
        >>> window = create_alternating_window(messages, pairs=3)
    """
    if not messages:
        return []

    result = []
    pair_count = 0
    current_pair = []

    # Work backwards to get recent pairs
    for message in reversed(messages):
        if message.role == "system":
            continue

        current_pair.insert(0, message)

        # A pair is complete when we have user + assistant
        if len(current_pair) == 2:
            result = current_pair + result
            current_pair = []
            pair_count += 1

            if pair_count >= pairs:
                break

    # Add any incomplete pair
    if current_pair:
        result = current_pair + result

    return result


class ConversationBuffer:
    """Manages conversation history with multiple memory strategies."""

    def __init__(
        self,
        max_messages: int = 100,
        window_size: int = 10,
        enable_summaries: bool = True,
        enable_entity_tracking: bool = True,
    ):
        """Initialize conversation buffer.

        Args:
            max_messages: Maximum messages to store
            window_size: Size of sliding window
            enable_summaries: Whether to create summaries
            enable_entity_tracking: Whether to track entities
        """
        self.max_messages = max_messages
        self.window_size = window_size
        self.enable_summaries = enable_summaries
        self.enable_entity_tracking = enable_entity_tracking

        self.messages: List[Message] = []
        self.summaries: List[ConversationSummary] = []
        self.entities: Dict[str, Entity] = {}
        self.metadata: Dict[str, Any] = {}

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict] = None
    ) -> Message:
        """Add a message to the buffer.

        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content
            metadata: Optional metadata

        Returns:
            Message: The created message
        """
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)

        # Trim old messages if over limit
        if len(self.messages) > self.max_messages:
            # Create summary before removing
            if self.enable_summaries:
                old_messages = self.messages[: len(self.messages) - self.max_messages]
                summary = summarize_conversation(old_messages)
                self.summaries.append(summary)

            self.messages = self.messages[-self.max_messages :]

        # Update entity tracking
        if self.enable_entity_tracking:
            new_entities = extract_entities([content])
            for entity in new_entities:
                key = f"{entity.type}:{entity.name.lower()}"
                if key in self.entities:
                    self.entities[key].mentions += 1
                    self.entities[key].last_seen = entity.last_seen
                    self.entities[key].context.extend(entity.context)
                else:
                    self.entities[key] = entity

        return message

    def get_recent_messages(self, count: Optional[int] = None) -> List[Message]:
        """Get recent messages (sliding window).

        Args:
            count: Number of messages (defaults to window_size)

        Returns:
            List[Message]: Recent messages
        """
        count = count or self.window_size
        return create_sliding_window(self.messages, window_size=count)

    def get_context(
        self, max_tokens: Optional[int] = None, include_summary: bool = True
    ) -> str:
        """Get conversation context as formatted string.

        Args:
            max_tokens: Maximum tokens (if None, use all recent messages)
            include_summary: Whether to include summaries of old messages

        Returns:
            str: Formatted conversation context
        """
        parts = []

        # Add summaries of old conversations
        if include_summary and self.summaries:
            summary_text = "\n".join(s.summary for s in self.summaries[-3:])
            parts.append(f"Previous conversation summary:\n{summary_text}\n")

        # Get recent messages
        if max_tokens:
            recent = create_token_limited_window(self.messages, max_tokens=max_tokens)
        else:
            recent = self.get_recent_messages()

        # Format messages
        for msg in recent:
            parts.append(f"{msg.role}: {msg.content}")

        return "\n".join(parts)

    def get_entities(self, min_mentions: int = 1) -> List[Entity]:
        """Get tracked entities.

        Args:
            min_mentions: Minimum mentions to include

        Returns:
            List[Entity]: List of entities
        """
        entities = [e for e in self.entities.values() if e.mentions >= min_mentions]
        return sorted(entities, key=lambda e: e.mentions, reverse=True)

    def search_messages(self, query: str, max_results: int = 10) -> List[Message]:
        """Search messages by content.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List[Message]: Matching messages
        """
        query_lower = query.lower()
        matches = []

        for msg in self.messages:
            if query_lower in msg.content.lower():
                matches.append(msg)
                if len(matches) >= max_results:
                    break

        return matches

    def clear(self, keep_summaries: bool = True):
        """Clear the buffer.

        Args:
            keep_summaries: Whether to keep summaries
        """
        if keep_summaries and self.messages:
            summary = summarize_conversation(self.messages)
            self.summaries.append(summary)

        self.messages.clear()
        self.entities.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export buffer to dictionary."""
        return {
            "messages": [m.to_dict() for m in self.messages],
            "summaries": [s.to_dict() for s in self.summaries],
            "entities": {k: e.to_dict() for k, e in self.entities.items()},
            "metadata": self.metadata,
            "config": {
                "max_messages": self.max_messages,
                "window_size": self.window_size,
                "enable_summaries": self.enable_summaries,
                "enable_entity_tracking": self.enable_entity_tracking,
            },
        }

    def from_dict(self, data: Dict[str, Any]):
        """Load buffer from dictionary."""
        self.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        self.summaries = [
            ConversationSummary.from_dict(s) for s in data.get("summaries", [])
        ]
        self.entities = {
            k: Entity.from_dict(e) for k, e in data.get("entities", {}).items()
        }
        self.metadata = data.get("metadata", {})

        config = data.get("config", {})
        self.max_messages = config.get("max_messages", self.max_messages)
        self.window_size = config.get("window_size", self.window_size)
        self.enable_summaries = config.get("enable_summaries", self.enable_summaries)
        self.enable_entity_tracking = config.get(
            "enable_entity_tracking", self.enable_entity_tracking
        )

    def save(self, filepath: str) -> None:
        """Save conversation buffer to file.

        Args:
            filepath: Path to save to (JSON format)

        Example:
            >>> buffer.save("conversation.json")
        """
        data = self.to_dict()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ConversationBuffer":
        """Load conversation buffer from file.

        Args:
            filepath: Path to load from

        Returns:
            ConversationBuffer: Loaded buffer

        Example:
            >>> buffer = ConversationBuffer.load("conversation.json")
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        buffer = cls()
        buffer.from_dict(data)
        return buffer

    def prune(
        self,
        strategy: Union["PruneStrategy", str] = "oldest",
        keep_count: Optional[int] = None,
        keep_percentage: Optional[float] = None,
    ) -> "ConversationBuffer":
        """Prune messages from buffer using various strategies.

        Args:
            strategy: Pruning strategy (PruneStrategy enum or string: "oldest", "newest", "least_relevant", "most_relevant", "token_limit")
            keep_count: Number of messages to keep
            keep_percentage: Percentage of messages to keep (0-1)

        Returns:
            ConversationBuffer: Self (for method chaining, modifies in place)

        Examples:
            >>> # Using enum (recommended)
            >>> from kerb.core.enums import PruneStrategy
            >>> buffer.prune(strategy=PruneStrategy.OLDEST, keep_count=50)

            >>> # Using string (for backward compatibility)
            >>> buffer.prune(strategy="oldest", keep_count=50)
        """
        from kerb.core.enums import PruneStrategy, validate_enum_or_string

        # Validate strategy
        strategy_val = validate_enum_or_string(strategy, PruneStrategy, "strategy")
        if isinstance(strategy_val, PruneStrategy):
            strategy_str = strategy_val.value
        else:
            strategy_str = strategy_val

        # Determine how many to keep
        if keep_count is not None:
            target_count = min(keep_count, len(self.messages))
        elif keep_percentage is not None:
            target_count = max(1, int(len(self.messages) * keep_percentage))
        else:
            target_count = len(self.messages) // 2  # Default: keep half

        if target_count >= len(self.messages):
            return self

        # Apply strategy
        if strategy_str == "oldest":
            # Keep most recent messages
            self.messages = self.messages[-target_count:]
        elif strategy_str == "newest":
            # Keep oldest messages
            self.messages = self.messages[:target_count]
        elif strategy_str in ("least_relevant", "most_relevant"):
            # Placeholder - would need relevance scoring
            self.messages = self.messages[-target_count:]
        elif strategy_str == "token_limit":
            # Placeholder - would need token counting
            self.messages = self.messages[-target_count:]

        return self
