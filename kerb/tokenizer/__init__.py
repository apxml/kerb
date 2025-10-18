"""Tokenizer utilities for counting tokens across different LLM models.

This module provides comprehensive token counting support for:
- OpenAI models (GPT-3.5, GPT-4, etc.) via tiktoken
- HuggingFace models (BERT, Llama, etc.) via transformers
- Fast approximation methods for quick estimates

Key features:
- Tokenizer enum for explicit, type-safe tokenizer specification
- count_tokens: Count tokens for a single text
- batch_count_tokens: Count tokens for multiple texts
- count_tokens_for_messages: Count tokens in chat message format
- truncate_to_token_limit: Truncate text to fit token limits
- tokens_to_chars / chars_to_tokens: Convert between tokens and characters
"""

from . import utils
from .tokenizer import (Tokenizer, batch_count_tokens, count_tokens,
                        count_tokens_for_messages, truncate_to_token_limit)
from .utils import (chars_to_tokens, estimate_cost, optimize_token_usage,
                    tokens_to_chars)

__all__ = [
    # Core functions
    "count_tokens",
    "batch_count_tokens",
    "count_tokens_for_messages",
    "truncate_to_token_limit",
    # Tokenizer enum
    "Tokenizer",
    # Utils submodule
    "utils",
    # Utility functions (also accessible via utils submodule)
    "tokens_to_chars",
    "chars_to_tokens",
    "estimate_cost",
    "optimize_token_usage",
]
