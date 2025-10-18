"""Context management utilities for LLM applications.

This module provides comprehensive tools for managing LLM context windows:

Data Classes:
    ContextItem - Represents a single item in context
    ContextWindow - Represents a managed context window
    CompressionResult - Result of context compression
    TruncationStrategy - Enum for truncation strategies
    CompressionMethod - Enum for compression methods

Token Management:
    For token counting, use the tokenizer module:
        from kerb.tokenizer import count_tokens, batch_count_tokens

Context Window Management:
    create_context_window() - Create managed context window
    truncate_context_window() - Truncate window to fit token limits

Sliding Window Utilities:
    create_sliding_window() - Create sliding windows over items
    create_token_sliding_window() - Create token-based sliding windows
    create_adaptive_window() - Create adaptive window balancing recency and priority

Context Compression:
    compress_context() - Compress context to target token count
    auto_compress_window() - Automatically compress window items

Priority Management:
    assign_priorities() - Assign priorities using custom function
    priority_by_recency() - Assign priorities based on recency
    priority_by_relevance() - Assign priorities based on query relevance
    priority_by_diversity() - Assign priorities to maximize diversity

Context Optimization:
    deduplicate_context() - Remove duplicate or similar items
    reorder_context() - Reorder items using specified strategy
    merge_context_windows() - Merge multiple windows into one
    optimize_context_for_query() - Optimize window for specific query

Context Formatting:
    format_context_window() - Format window for LLM consumption
    context_to_messages() - Convert window to chat message format
    extract_context_summary() - Extract summary of window contents
"""

# Submodules for specialized functionality
from . import compression, formatting, optimization, priority, sliding
# Convenience imports for common operations
from .compression import auto_compress_window, compress_context
from .formatting import (context_to_messages, extract_context_summary,
                         format_context_window)
from .optimization import (deduplicate_context, merge_context_windows,
                           optimize_context_for_query, reorder_context)
from .priority import (assign_priorities, priority_by_diversity,
                       priority_by_recency, priority_by_relevance)
from .sliding import (create_adaptive_window, create_sliding_window,
                      create_token_sliding_window)
# Core data classes and enums
from .types import (CompressionMethod, CompressionResult, ContextItem,
                    ContextWindow, TruncationStrategy)
# Core window management (most common)
from .window import create_context_window, truncate_context_window

__all__ = [
    # Data classes and enums
    "ContextItem",
    "ContextWindow",
    "CompressionResult",
    "TruncationStrategy",
    "CompressionMethod",
    # Core window management
    "create_context_window",
    "truncate_context_window",
    # Submodules
    "compression",
    "sliding",
    "optimization",
    "formatting",
    "priority",
    # Compression
    "compress_context",
    "auto_compress_window",
    # Sliding windows
    "create_sliding_window",
    "create_token_sliding_window",
    "create_adaptive_window",
    # Optimization
    "deduplicate_context",
    "reorder_context",
    "merge_context_windows",
    "optimize_context_for_query",
    # Formatting
    "format_context_window",
    "context_to_messages",
    "extract_context_summary",
    # Priority management
    "assign_priorities",
    "priority_by_recency",
    "priority_by_relevance",
    "priority_by_diversity",
]
