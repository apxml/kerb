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

# Top-level imports: Core classes and most common functions
from .generator import (
    generate,
    generate_stream,
    generate_batch,
    generate_async,
    Generator,
)

from .config import (
    GenerationConfig,
    GenerationResponse,
    StreamChunk,
    Usage,
)

from .enums import (
    LLMProvider,
    ModelName,
    MODEL_PRICING,
)

from .utils import (
    calculate_cost,
    parse_json_response,
    validate_response,
    format_messages,
    RateLimiter,
    ResponseCache,
    CostTracker,
    retry_with_exponential_backoff,
    async_retry_with_exponential_backoff,
    get_cost_summary,
    reset_cost_tracking,
    global_cost_tracker,
)

from .base import (
    BaseProvider,
    register_provider,
    get_provider,
    list_providers,
)

# Submodule imports: Specialized implementations
from . import providers, utils, base

# Import provider classes for convenient access
from .providers import (
    OpenAIGenerator,
    AnthropicGenerator,
    GoogleGenerator,
    CohereGenerator,
    MistralGenerator,
)

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
