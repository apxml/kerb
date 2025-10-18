"""Advanced memory patterns for specialized use cases.

This module provides advanced memory management patterns:
- create_semantic_memory: Group similar messages together
- create_episodic_memory: Split conversation into episodes
- get_relevant_memory: Retrieve relevant memories for a query
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

from kerb.core.types import Message

if TYPE_CHECKING:
    from .buffers import ConversationBuffer


def create_semantic_memory(
    messages: List[Message], similarity_threshold: float = 0.7
) -> Dict[str, List[Message]]:
    """Group similar messages together (requires embeddings).

    Args:
        messages: Messages to group
        similarity_threshold: Minimum similarity for grouping

    Returns:
        Dict[str, List[Message]]: Groups of similar messages

    Note:
        This is a placeholder. For real semantic grouping, use the embedding module.

    Example:
        >>> groups = create_semantic_memory(messages)
    """
    # Simple keyword-based grouping as placeholder
    groups: Dict[str, List[Message]] = defaultdict(list)

    for msg in messages:
        # Extract key words
        words = set(w.lower() for w in msg.content.split() if len(w) > 4)

        # Find best matching group
        best_group = None
        best_overlap = 0

        for group_key in groups:
            group_words = set(group_key.split(","))
            overlap = len(words & group_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_group = group_key

        # Add to group or create new
        if best_group and best_overlap >= 2:
            groups[best_group].append(msg)
        else:
            # Create new group
            key_words = sorted(list(words))[:3]
            group_key = ",".join(key_words)
            groups[group_key].append(msg)

    return dict(groups)


def create_episodic_memory(
    messages: List[Message], episode_duration: int = 10
) -> List[List[Message]]:
    """Split conversation into episodic memories.

    Args:
        messages: Messages to split
        episode_duration: Messages per episode

    Returns:
        List[List[Message]]: List of episodes

    Example:
        >>> episodes = create_episodic_memory(messages, episode_duration=5)
    """
    episodes = []

    for i in range(0, len(messages), episode_duration):
        episode = messages[i : i + episode_duration]
        if episode:
            episodes.append(episode)

    return episodes


def get_relevant_memory(
    query: str, buffer: "ConversationBuffer", top_k: int = 5
) -> List[Message]:
    """Retrieve relevant memories for a query.

    Args:
        query: Query string
        buffer: Conversation buffer to search
        top_k: Number of relevant messages to return

    Returns:
        List[Message]: Most relevant messages

    Example:
        >>> relevant = get_relevant_memory("python async", buffer)
    """
    # Simple keyword-based relevance
    query_words = set(query.lower().split())

    scored_messages = []
    for msg in buffer.messages:
        msg_words = set(msg.content.lower().split())
        overlap = len(query_words & msg_words)
        if overlap > 0:
            scored_messages.append((overlap, msg))

    # Sort by relevance
    scored_messages.sort(key=lambda x: x[0], reverse=True)

    return [msg for _, msg in scored_messages[:top_k]]
