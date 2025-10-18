"""Enumeration types for text preprocessing."""

from enum import Enum


class NormalizationLevel(Enum):
    """Text normalization intensity."""
    MINIMAL = "minimal"  # Only basic whitespace normalization
    STANDARD = "standard"  # Standard cleaning (recommended)
    AGGRESSIVE = "aggressive"  # Remove most non-alphanumeric content


class LanguageDetectionMode(Enum):
    """Language detection strategy."""
    FAST = "fast"  # Fast heuristic-based detection
    ACCURATE = "accurate"  # More accurate but slower
    SIMPLE = "simple"  # Simple character-based detection


class DeduplicationMode(Enum):
    """Deduplication strategy."""
    EXACT = "exact"  # Exact string matching
    FUZZY = "fuzzy"  # Fuzzy matching (similar strings)
    SEMANTIC = "semantic"  # Semantic similarity (requires embeddings)


class ContentType(Enum):
    """Text content type classification."""
    PLAIN_TEXT = "plain_text"
    CODE = "code"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    MIXED = "mixed"
    UNKNOWN = "unknown"
