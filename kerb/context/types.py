"""Data classes and enums for context management.

This module defines core data structures used across the context package.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class TruncationStrategy(Enum):
    """Strategies for truncating context when exceeding limits."""

    FIRST = "first"  # Keep first N tokens
    LAST = "last"  # Keep last N tokens
    MIDDLE = "middle"  # Keep start and end, remove middle
    PRIORITY = "priority"  # Keep highest priority items
    SEMANTIC = "semantic"  # Keep semantically most relevant


class CompressionMethod(Enum):
    """Methods for compressing context."""

    SUMMARIZE = "summarize"  # Summarize content
    EXTRACT_KEY_INFO = "extract_key_info"  # Extract key information
    REMOVE_REDUNDANCY = "remove_redundancy"  # Remove duplicate info
    ABBREVIATE = "abbreviate"  # Use abbreviations and shorter forms
    HYBRID = "hybrid"  # Combine multiple methods


@dataclass
class ContextItem:
    """Represents a single item in the context window."""

    content: str
    priority: float = 1.0
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    item_type: str = "text"  # text, code, conversation, document, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextItem":
        """Create from dictionary."""
        return cls(**data)

    def __lt__(self, other: "ContextItem") -> bool:
        """Compare by priority (for heap operations)."""
        return self.priority < other.priority


@dataclass
class ContextWindow:
    """Represents a managed context window."""

    items: list[ContextItem] = field(default_factory=list)
    max_tokens: Optional[int] = None
    current_tokens: int = 0
    strategy: TruncationStrategy = TruncationStrategy.LAST
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_item(self, item: ContextItem) -> None:
        """Add item to context window."""
        self.items.append(item)
        if item.token_count:
            self.current_tokens += item.token_count

    def get_content(self) -> str:
        """Get concatenated content from all items."""
        return "\n\n".join(item.content for item in self.items)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "items": [item.to_dict() for item in self.items],
            "max_tokens": self.max_tokens,
            "current_tokens": self.current_tokens,
            "strategy": self.strategy.value,
            "metadata": self.metadata,
        }


@dataclass
class CompressionResult:
    """Result of context compression."""

    compressed_content: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    method: CompressionMethod
    metadata: Dict[str, Any] = field(default_factory=dict)
