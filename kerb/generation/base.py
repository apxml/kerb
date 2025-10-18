"""Base provider abstraction and registry for LLM generation.

This module provides the base class for all LLM providers and a registry
system for managing custom providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

from kerb.core.types import Message

from .config import GenerationConfig, GenerationResponse, StreamChunk

# ============================================================================
# Provider Registry
# ============================================================================

_provider_registry: Dict[str, "BaseProvider"] = {}


def register_provider(name: str, provider: "BaseProvider") -> None:
    """Register a custom provider.

    Args:
        name: Provider name (used in model strings like "custom::model-name")
        provider: Provider instance

    Examples:
        >>> from kerb.generation.base import register_provider
        >>> provider = MyCustomProvider(api_key="...")
        >>> register_provider("mycustom", provider)
        >>> # Now can use: generate(messages, model="mycustom::my-model")
    """
    _provider_registry[name] = provider


def get_provider(name: str) -> Optional["BaseProvider"]:
    """Get a registered provider by name.

    Args:
        name: Provider name

    Returns:
        Provider instance or None if not found
    """
    return _provider_registry.get(name)


def list_providers() -> List[str]:
    """List all registered provider names."""
    return list(_provider_registry.keys())


# ============================================================================
# Base Provider Interface
# ============================================================================


class BaseProvider(ABC):
    """Base class for LLM providers.

    Custom providers should inherit from this class and implement
    the required methods.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize provider.

        Args:
            api_key: API key (if None, will try to get from environment)
            **kwargs: Provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    def generate(
        self, messages: List[Message], config: GenerationConfig
    ) -> GenerationResponse:
        """Generate a response.

        Args:
            messages: List of conversation messages
            config: Generation configuration

        Returns:
            GenerationResponse
        """
        pass

    @abstractmethod
    def generate_stream(
        self, messages: List[Message], config: GenerationConfig
    ) -> Iterator[StreamChunk]:
        """Generate a streaming response.

        Args:
            messages: List of conversation messages
            config: Generation configuration

        Yields:
            StreamChunk
        """
        pass

    @abstractmethod
    async def generate_async(
        self, messages: List[Message], config: GenerationConfig
    ) -> GenerationResponse:
        """Generate a response asynchronously.

        Args:
            messages: List of conversation messages
            config: Generation configuration

        Returns:
            GenerationResponse
        """
        pass

    def validate_config(self, config: GenerationConfig) -> bool:
        """Validate configuration for this provider.

        Args:
            config: Generation configuration

        Returns:
            bool: True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        return True
