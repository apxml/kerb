"""LLM generation utilities for unified API access and response management.

This module provides comprehensive tools for LLM generation across multiple providers.

Usage:
    # Common
    from kerb.generation import generate, generate_stream, Generator

    # Providers
    from kerb.generation.providers import (
        OpenAIGenerator,
        AnthropicGenerator,
        GoogleGenerator,
        CohereGenerator,
        MistralGenerator,
    )

    # Utilities
    from kerb.generation.utils import retry_with_exponential_backoff, batch_generate
"""

# Import from core
from kerb.core import Message
from kerb.core.types import MessageRole

# Submodule imports: Specialized implementations
from . import base, providers, utils
from .base import BaseProvider, get_provider, list_providers, register_provider
from .config import GenerationConfig, GenerationResponse, StreamChunk, Usage
from .enums import MODEL_PRICING, LLMProvider, ModelName
# Top-level imports: Core classes and most common functions
from .generator import (Generator, generate, generate_async, generate_batch,
                        generate_stream)
# Import provider classes for convenient access
from .providers import (AnthropicGenerator, CohereGenerator, GoogleGenerator,
                        MistralGenerator, OpenAIGenerator)
from .utils import (CostTracker, RateLimiter, ResponseCache,
                    async_retry_with_exponential_backoff, calculate_cost,
                    format_messages, get_cost_summary, global_cost_tracker,
                    parse_json_response, reset_cost_tracking,
                    retry_with_exponential_backoff, validate_response)

__all__ = [
    # Core generation functions
    "generate",
    "generate_stream",
    "generate_batch",
    "generate_async",
    "Generator",
    # Configuration classes
    "GenerationConfig",
    "GenerationResponse",
    "StreamChunk",
    "Usage",
    # Enums and constants
    "LLMProvider",
    "ModelName",
    "MODEL_PRICING",
    "MessageRole",
    "Message",
    # Utility functions
    "retry_with_exponential_backoff",
    "async_retry_with_exponential_backoff",
    "parse_json_response",
    "validate_response",
    "format_messages",
    "calculate_cost",
    "batch_generate",
    # Utility classes
    "RateLimiter",
    "ResponseCache",
    "CostTracker",
    "global_cost_tracker",
    "get_cost_summary",
    "reset_cost_tracking",
    # Provider base and registry
    "BaseProvider",
    "register_provider",
    "get_provider",
    "list_providers",
    # Provider implementations
    "OpenAIGenerator",
    "AnthropicGenerator",
    "GoogleGenerator",
    "CohereGenerator",
    "MistralGenerator",
    # Submodules
    "providers",
    "utils",
    "base",
]
