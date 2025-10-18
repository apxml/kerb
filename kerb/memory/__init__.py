"""Memory management utilities for LLM applications.

This module provides comprehensive memory management for conversational AI:

Core Classes:
    ConversationBuffer - Main class for managing conversation memory
    Entity - Represents an extracted entity
    ConversationSummary - Represents a conversation summary

Common Functions:
    create_sliding_window() - Recent messages window
    create_token_limited_window() - Token-limited window
    create_alternating_window() - Alternating user/assistant pairs

Submodules:
    buffers - Conversation buffer management and sliding windows
    summaries - Summary-based memory functions
    entities - Entity extraction and tracking
    utils - Utility functions (format, filter, merge, persistence)
    patterns - Advanced memory patterns (semantic, episodic)
    classes - Data classes (Entity, ConversationSummary)

Examples:
    >>> # Common usage - top-level imports
    >>> from kerb.memory import ConversationBuffer, Entity
    >>> buffer = ConversationBuffer()
    >>> buffer.add_message("user", "Hello!")

    >>> # Specialized usage - submodule imports
    >>> from kerb.memory.summaries import create_progressive_summary
    >>> from kerb.memory.entities import extract_entities
    >>> from kerb.memory.patterns import create_semantic_memory
"""

# Submodule imports for specialized usage
from . import buffers, classes, entities, patterns, summaries, utils
from .buffers import (ConversationBuffer, create_alternating_window,
                      create_sliding_window, create_token_limited_window)
# Top-level imports: Core classes and most common functions
from .classes import ConversationSummary, Entity
from .summaries import (create_hierarchical_summary,
                        create_progressive_summary, summarize_conversation)
from .utils import load_conversation, save_conversation

__all__ = [
    # Core classes
    "ConversationBuffer",
    "Entity",
    "ConversationSummary",
    # Common buffer functions
    "create_sliding_window",
    "create_token_limited_window",
    "create_alternating_window",
    # Common summary functions
    "create_progressive_summary",
    "summarize_conversation",
    "create_hierarchical_summary",
    # Common utility functions
    "save_conversation",
    "load_conversation",
    # Submodules
    "buffers",
    "summaries",
    "entities",
    "utils",
    "patterns",
    "classes",
]
