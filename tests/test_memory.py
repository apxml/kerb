"""Comprehensive tests for the memory module."""

import json
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from kerb.core.types import Message
from kerb.memory import (ConversationBuffer, ConversationSummary, Entity,
                         create_alternating_window,
                         create_hierarchical_summary,
                         create_progressive_summary, create_sliding_window,
                         create_token_limited_window, load_conversation,
                         save_conversation, summarize_conversation)
from kerb.memory.entities import (extract_entities,
                                  extract_entity_relationships, merge_entities,
                                  track_entity_mentions)
from kerb.memory.patterns import (create_episodic_memory,
                                  create_semantic_memory, get_relevant_memory)
from kerb.memory.utils import (filter_messages, format_messages,
                               merge_conversations, prune_buffer)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        Message("system", "You are helpful"),
        Message("user", "Hi, I'm Alice"),
        Message("assistant", "Hello Alice!"),
        Message("user", "Tell me about Python"),
        Message("assistant", "Python is a versatile language"),
    ]


@pytest.fixture
def conversation_buffer():
    """Sample conversation buffer."""
    buffer = ConversationBuffer()
    buffer.add_message("user", "Hello")
    buffer.add_message("assistant", "Hi there!")
    return buffer


# ============================================================================
# Data Class Tests
# ============================================================================


def test_message_creation():
    """Test Message creation."""
    msg = Message("user", "Hello", metadata={"key": "value"})
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.metadata["key"] == "value"
    assert msg.timestamp is not None


def test_message_serialization():
    """Test Message to_dict/from_dict."""
    msg = Message("user", "Hello", metadata={"test": True})
    data = msg.to_dict()

    assert data["role"] == "user"
    assert data["content"] == "Hello"
    assert data["metadata"]["test"] is True

    restored = Message.from_dict(data)
    # Compare role values (handles both string and enum)
    msg_role = msg.role.value if hasattr(msg.role, "value") else msg.role
    restored_role = (
        restored.role.value if hasattr(restored.role, "value") else restored.role
    )
    assert restored_role == msg_role
    assert restored.content == msg.content


def test_entity_creation():
    """Test Entity creation."""
    entity = Entity("Alice", "person", mentions=3, context=["Hi Alice"])
    assert entity.name == "Alice"
    assert entity.type == "person"
    assert entity.mentions == 3


def test_conversation_summary():
    """Test ConversationSummary."""
    now = datetime.now().isoformat()
    summary = ConversationSummary(
        summary="User asked about Python",
        key_points=["Python", "Programming"],
        entities=["Python"],
        message_count=5,
        start_time=now,
        end_time=now,
    )
    assert summary.summary == "User asked about Python"
    assert len(summary.key_points) == 2


# ============================================================================
# Sliding Window Tests
# ============================================================================


def test_create_sliding_window(sample_messages):
    """Test basic sliding window."""
    result = create_sliding_window(sample_messages, window_size=3)
    assert len(result) == 3
    assert result[-1].content == "Python is a versatile language"


def test_sliding_window_larger_than_messages(sample_messages):
    """Test sliding window larger than message list."""
    result = create_sliding_window(sample_messages, window_size=10)
    assert len(result) == len(sample_messages)


def test_create_token_limited_window(sample_messages):
    """Test token-limited window."""
    result = create_token_limited_window(sample_messages, max_tokens=50)
    assert len(result) <= len(sample_messages)
    # Should include at least one message
    assert len(result) > 0


def test_create_alternating_window(sample_messages):
    """Test alternating window."""
    result = create_alternating_window(sample_messages, pairs=2)
    # Should get 2 pairs = 4 messages (user + assistant each)
    assert len(result) == 4


# ============================================================================
# Summary Tests
# ============================================================================


def test_create_progressive_summary_short(sample_messages):
    """Test short progressive summary."""
    summary = create_progressive_summary(sample_messages[1:], summary_length="short")
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_create_progressive_summary_medium(sample_messages):
    """Test medium progressive summary."""
    summary = create_progressive_summary(sample_messages[1:], summary_length="medium")
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_summarize_conversation(sample_messages):
    """Test conversation summarization."""
    summary = summarize_conversation(sample_messages)
    assert summary.message_count == len(sample_messages)
    assert len(summary.summary) > 0


def test_create_hierarchical_summary(sample_messages):
    """Test hierarchical summary."""
    # Extend messages to have enough for chunks
    extended = sample_messages * 3
    summaries = create_hierarchical_summary(extended, chunk_size=5)
    assert len(summaries) > 0
    assert all(isinstance(s, ConversationSummary) for s in summaries)


# ============================================================================
# Entity Extraction Tests
# ============================================================================


def test_extract_entities():
    """Test entity extraction."""
    texts = ["My name is Alice and I work at TechCorp", "Email me at alice@example.com"]
    entities = extract_entities(texts)

    # Should extract some entities
    assert len(entities) > 0
    assert all(isinstance(e, Entity) for e in entities)


def test_extract_entities_with_types():
    """Test entity extraction with specific types."""
    texts = ["Contact Bob at bob@example.com"]
    entities = extract_entities(texts, entity_types=["person", "email"])

    types = [e.type for e in entities]
    assert "person" in types or "email" in types


def test_track_entity_mentions():
    """Test entity mention tracking."""
    messages = [
        Message("user", "My name is Alice"),
        Message("assistant", "Hello Alice!"),
        Message("user", "Alice likes Python"),
    ]
    entity = Entity("Alice", "person")
    tracked = track_entity_mentions(messages, entity)

    # Alice should be mentioned multiple times
    assert len(tracked) >= 2
    assert all(isinstance(t, tuple) for t in tracked)


def test_extract_entity_relationships():
    """Test entity relationship extraction."""
    messages = [
        Message("user", "Alice works with Bob at TechCorp"),
        Message("user", "Alice and Bob collaborate on Python"),
    ]
    entities = [Entity("Alice", "person"), Entity("Bob", "person")]
    relationships = extract_entity_relationships(messages, entities)

    # Should find relationships
    assert isinstance(relationships, dict)
    assert len(relationships) > 0


def test_merge_entities():
    """Test entity merging."""
    entity1 = Entity("Alice", "person", mentions=2)
    entity2 = Entity("alice", "person", mentions=1)

    merged = merge_entities(entity1, entity2, prefer="most_mentioned")

    # Should merge based on most mentions
    assert merged.mentions == 3
    assert merged.type == "person"


# ============================================================================
# ConversationBuffer Tests
# ============================================================================


def test_conversation_buffer_creation():
    """Test ConversationBuffer creation."""
    buffer = ConversationBuffer(max_messages=10)
    assert buffer.max_messages == 10
    assert len(buffer.messages) == 0


def test_add_message(conversation_buffer):
    """Test adding messages."""
    initial_count = len(conversation_buffer.messages)
    conversation_buffer.add_message("user", "Test message")
    assert len(conversation_buffer.messages) == initial_count + 1


def test_get_recent_messages(conversation_buffer):
    """Test getting recent messages."""
    recent = conversation_buffer.get_recent_messages(count=1)
    assert len(recent) == 1
    assert recent[0].role == "assistant"


def test_get_context(conversation_buffer):
    """Test getting formatted context."""
    context = conversation_buffer.get_context(max_tokens=1000)
    assert isinstance(context, str)
    assert len(context) > 0


def test_buffer_max_messages():
    """Test buffer respects max_messages."""
    buffer = ConversationBuffer(max_messages=3)
    for i in range(5):
        buffer.add_message("user", f"Message {i}")

    assert len(buffer.messages) == 3


def test_buffer_entity_tracking():
    """Test entity tracking in buffer."""
    buffer = ConversationBuffer(enable_entity_tracking=True)
    buffer.add_message("user", "I'm Alice from TechCorp")
    buffer.add_message("assistant", "Hello Alice!")

    entities = buffer.get_entities()
    assert len(entities) > 0


def test_search_messages(conversation_buffer):
    """Test message search."""
    conversation_buffer.add_message("user", "Tell me about Python")
    results = conversation_buffer.search_messages("Python")
    assert len(results) > 0


def test_buffer_clear(conversation_buffer):
    """Test clearing buffer."""
    conversation_buffer.clear()
    assert len(conversation_buffer.messages) == 0


def test_buffer_serialization():
    """Test buffer to_dict/from_dict."""
    buffer = ConversationBuffer()
    buffer.add_message("user", "Test")

    data = buffer.to_dict()
    assert "messages" in data
    assert "config" in data
    assert data["config"]["max_messages"] == 100

    restored = ConversationBuffer()
    restored.from_dict(data)
    assert len(restored.messages) == len(buffer.messages)


# ============================================================================
# Formatting Tests
# ============================================================================


def test_format_messages_simple(sample_messages):
    """Test simple message formatting."""
    result = format_messages(sample_messages, format_style="simple")
    assert isinstance(result, str)
    assert "user:" in result or "assistant:" in result


def test_format_messages_chat(sample_messages):
    """Test chat-style formatting."""
    result = format_messages(sample_messages, format_style="chat")
    assert "👤" in result or "🤖" in result


def test_format_messages_detailed(sample_messages):
    """Test detailed formatting."""
    result = format_messages(sample_messages, format_style="detailed")
    assert len(result) > 0


def test_format_messages_json(sample_messages):
    """Test JSON formatting."""
    result = format_messages(sample_messages, format_style="json")
    # Should be valid JSON
    parsed = json.loads(result)
    assert isinstance(parsed, list)


# ============================================================================
# Filtering Tests
# ============================================================================


def test_filter_messages_by_role(sample_messages):
    """Test filtering by role."""
    user_msgs = filter_messages(sample_messages, role="user")
    assert all(m.role == "user" for m in user_msgs)


def test_filter_messages_by_content(sample_messages):
    """Test filtering by content."""
    python_msgs = filter_messages(sample_messages, contains="Python")
    assert all("Python" in m.content for m in python_msgs)


def test_filter_messages_combined(sample_messages):
    """Test combined filtering."""
    result = filter_messages(sample_messages, role="user", contains="Alice")
    assert len(result) > 0
    assert all(m.role == "user" for m in result)


def test_filter_messages_empty_result(sample_messages):
    """Test filtering with no matches."""
    result = filter_messages(sample_messages, contains="nonexistent")
    assert len(result) == 0


# ============================================================================
# Utility Tests
# ============================================================================


def test_merge_conversations():
    """Test merging conversations."""
    buffer1 = ConversationBuffer()
    buffer1.add_message("user", "Hello")

    buffer2 = ConversationBuffer()
    buffer2.add_message("user", "Goodbye")

    merged = merge_conversations(buffer1, buffer2)
    assert len(merged.messages) == 2


def test_prune_buffer_simple(sample_messages):
    """Test simple buffer pruning."""
    buffer = ConversationBuffer()
    for msg in sample_messages:
        buffer.add_message(msg.role, msg.content)

    pruned = prune_buffer(buffer, strategy="oldest", keep_count=3)
    assert len(pruned.messages) == 3


def test_prune_buffer_alternating(sample_messages):
    """Test alternating pruning strategy."""
    buffer = ConversationBuffer()
    for msg in sample_messages:
        buffer.add_message(msg.role, msg.content)

    pruned = prune_buffer(buffer, strategy="oldest", keep_count=4)
    assert len(pruned.messages) == 4


# ============================================================================
# Persistence Tests
# ============================================================================


def test_save_and_load_conversation(conversation_buffer):
    """Test saving and loading conversation."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        filepath = f.name

    try:
        # Save
        save_conversation(conversation_buffer, filepath)
        assert os.path.exists(filepath)

        # Load
        loaded = load_conversation(filepath)
        assert len(loaded.messages) == len(conversation_buffer.messages)
        assert loaded.max_messages == conversation_buffer.max_messages
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# ============================================================================
# Advanced Pattern Tests
# ============================================================================


def test_create_semantic_memory(sample_messages):
    """Test semantic memory creation."""
    memory = create_semantic_memory(sample_messages)
    assert isinstance(memory, dict)
    assert len(memory) > 0


def test_create_episodic_memory(sample_messages):
    """Test episodic memory creation."""
    episodes = create_episodic_memory(sample_messages, episode_duration=2)
    assert len(episodes) > 0
    assert all(isinstance(ep, list) for ep in episodes)


def test_get_relevant_memory(conversation_buffer):
    """Test relevant memory retrieval."""
    conversation_buffer.add_message("user", "Tell me about Python")
    relevant = get_relevant_memory("Python", conversation_buffer, top_k=2)
    assert len(relevant) <= 2
    assert all(isinstance(m, Message) for m in relevant)


# ============================================================================
# Edge Cases
# ============================================================================


def test_empty_message_list():
    """Test handling empty message list."""
    result = create_sliding_window([], window_size=5)
    assert len(result) == 0


def test_single_message():
    """Test handling single message."""
    msgs = [Message("user", "Hello")]
    result = create_sliding_window(msgs, window_size=5)
    assert len(result) == 1


def test_large_conversation():
    """Test handling large conversations."""
    buffer = ConversationBuffer(max_messages=100)
    for i in range(150):
        buffer.add_message("user", f"Message {i}")

    assert len(buffer.messages) == 100


def test_special_characters_in_content():
    """Test handling special characters."""
    msg = Message("user", "Hello! @#$%^&*()")
    assert msg.content == "Hello! @#$%^&*()"


def test_very_long_message():
    """Test handling very long messages."""
    long_content = "x" * 10000
    msg = Message("user", long_content)
    assert len(msg.content) == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
