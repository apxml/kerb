"""Summary-based memory management functions.

This module provides functions for creating and managing conversation summaries:
- create_progressive_summary: Progressive conversation summary
- summarize_conversation: Comprehensive conversation summary
- create_hierarchical_summary: Hierarchical chunk summaries
"""

from typing import List, Union, TYPE_CHECKING
from datetime import datetime

from kerb.core.types import Message
from .classes import ConversationSummary
from .entities import extract_entities

if TYPE_CHECKING:
    from kerb.core.enums import SummaryStrategy


def create_progressive_summary(
    messages: List[Message],
    existing_summary: str = None,
    summary_length: str = "medium"
) -> str:
    """Create a progressive summary of conversation messages.
    
    Args:
        messages: Messages to summarize
        existing_summary: Previous summary to build upon
        summary_length: "short", "medium", or "long"
        
    Returns:
        str: Summary of the messages
        
    Example:
        >>> summary = create_progressive_summary(messages, summary_length="short")
    """
    if not messages:
        return existing_summary or ""
    
    # Extract key information
    topics = set()
    user_questions = []
    assistant_responses = []
    
    for msg in messages:
        if msg.role == "user":
            # Extract questions and topics
            if "?" in msg.content:
                user_questions.append(msg.content)
            # Simple topic extraction (nouns/important words)
            words = msg.content.lower().split()
            topics.update(w for w in words if len(w) > 4)
        elif msg.role == "assistant":
            assistant_responses.append(msg.content)
    
    # Build summary based on length
    if summary_length == "short":
        # Just key topics
        summary_parts = []
        if existing_summary:
            summary_parts.append(existing_summary)
        if topics:
            topic_list = list(topics)[:5]
            summary_parts.append(f"Discussed: {', '.join(topic_list)}")
        summary = ". ".join(summary_parts)
        
    elif summary_length == "medium":
        # Topics + brief exchange summary
        summary_parts = []
        if existing_summary:
            summary_parts.append(existing_summary)
        
        if user_questions:
            summary_parts.append(f"User asked {len(user_questions)} question(s)")
        
        if assistant_responses:
            summary_parts.append(f"Assistant provided {len(assistant_responses)} response(s)")
        
        if topics:
            topic_list = list(topics)[:8]
            summary_parts.append(f"Topics: {', '.join(topic_list)}")
        
        summary = ". ".join(summary_parts)
        
    else:  # "long"
        # Detailed summary
        summary_parts = []
        if existing_summary:
            summary_parts.append("Previous: " + existing_summary)
        
        summary_parts.append(f"Messages: {len(messages)}")
        
        if user_questions:
            summary_parts.append(f"Questions ({len(user_questions)}): " + 
                                "; ".join(q[:50] for q in user_questions[:3]))
        
        if topics:
            topic_list = list(topics)[:10]
            summary_parts.append(f"Topics discussed: {', '.join(topic_list)}")
        
        summary = ". ".join(summary_parts)
    
    return summary


def summarize_conversation(
    messages: List[Message],
    summary_strategy: Union['SummaryStrategy', str] = "extractive",
    key_points: int = 5
) -> ConversationSummary:
    """Create a comprehensive conversation summary.
    
    Args:
        messages: Messages to summarize
        summary_strategy: Summary strategy (SummaryStrategy enum or string: "extractive", "abstractive", "combined")
        key_points: Number of key points to extract
        
    Returns:
        ConversationSummary: Structured summary of conversation
        
    Examples:
        >>> # Using enum (recommended)
        >>> from kerb.core.enums import SummaryStrategy
        >>> summary = summarize_conversation(messages, summary_strategy=SummaryStrategy.EXTRACTIVE)
        
        >>> # Using string (for backward compatibility)
        >>> summary = summarize_conversation(messages, key_points=3)
    """
    from kerb.core.enums import SummaryStrategy, validate_enum_or_string
    
    if not messages:
        return ConversationSummary(
            summary="No messages",
            message_count=0,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat()
        )
    
    # Validate and normalize strategy
    strategy_val = validate_enum_or_string(summary_strategy, SummaryStrategy, "summary_strategy")
    if isinstance(strategy_val, SummaryStrategy):
        strategy_str = strategy_val.value
    else:
        strategy_str = strategy_val
    
    start_time = messages[0].timestamp
    end_time = messages[-1].timestamp
    
    # Extract entities
    entities = extract_entities([m.content for m in messages])
    entity_names = [e.name for e in entities[:10]]
    
    # Extract key points (simple: important sentences/questions)
    key_points_list = []
    for msg in messages:
        if msg.role == "user" and "?" in msg.content:
            key_points_list.append(msg.content[:100])
        elif len(msg.content) > 50 and len(key_points_list) < key_points:
            # Add longer, potentially important messages
            key_points_list.append(msg.content[:100])
    
    key_points_list = key_points_list[:key_points]
    
    # Create summary text
    if strategy_str == "extractive":
        # Extract representative messages
        summary_text = create_progressive_summary(messages, summary_length="medium")
    elif strategy_str == "combined":
        # Combination of extractive and abstractive
        summary_text = create_progressive_summary(messages, summary_length="long")
    else:  # abstractive
        # More detailed summary
        summary_text = create_progressive_summary(messages, summary_length="long")
    
    return ConversationSummary(
        summary=summary_text,
        message_count=len(messages),
        start_time=start_time,
        end_time=end_time,
        key_points=key_points_list,
        entities=entity_names
    )


def create_hierarchical_summary(
    messages: List[Message],
    chunk_size: int = 10
) -> List[ConversationSummary]:
    """Create hierarchical summaries of conversation chunks.
    
    Args:
        messages: Messages to summarize
        chunk_size: Number of messages per chunk
        
    Returns:
        List[ConversationSummary]: Summary for each chunk
        
    Example:
        >>> summaries = create_hierarchical_summary(messages, chunk_size=5)
    """
    if not messages:
        return []
    
    summaries = []
    
    for i in range(0, len(messages), chunk_size):
        chunk = messages[i:i + chunk_size]
        summary = summarize_conversation(chunk, summary_strategy="extractive")
        summaries.append(summary)
    
    return summaries
