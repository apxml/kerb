"""Configuration classes for LLM generation.

This module contains data classes for configuring generation requests and responses.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""

    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    n: int = 1  # Number of completions
    logprobs: Optional[int] = None
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None  # For JSON mode
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


@dataclass
class Usage:
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def cost(self) -> float:
        """Calculate cost based on token usage (requires model pricing)."""
        return 0.0  # Will be set by the response


@dataclass
class GenerationResponse:
    """Response from LLM generation.

    Note: LLMProvider is imported lazily to avoid circular imports.
    """

    content: str
    model: str
    provider: Any  # Will be LLMProvider at runtime
    usage: Usage
    finish_reason: Optional[str] = None
    latency: float = 0.0
    cost: float = 0.0
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": (
                self.provider.value
                if hasattr(self.provider, "value")
                else str(self.provider)
            ),
            "usage": asdict(self.usage),
            "finish_reason": self.finish_reason,
            "latency": self.latency,
            "cost": self.cost,
            "cached": self.cached,
            "metadata": self.metadata,
        }


@dataclass
class StreamChunk:
    """Represents a chunk from streaming generation."""

    content: str
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
