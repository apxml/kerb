"""Data types for text preprocessing."""

from dataclasses import dataclass, field
from typing import List, Tuple

from .enums import NormalizationLevel


@dataclass
class LanguageResult:
    """Language detection result."""
    language: str
    confidence: float
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Text quality metrics."""
    length: int
    word_count: int
    avg_word_length: float
    sentence_count: int
    avg_sentence_length: float
    special_char_ratio: float
    digit_ratio: float
    uppercase_ratio: float
    readability_score: float


@dataclass
class NormalizationConfig:
    """Configuration for text normalization operations.
    
    Attributes:
        level: Normalization intensity level
        lowercase: Convert to lowercase
        remove_urls: Remove URLs from text
        remove_emails: Remove email addresses
        remove_extra_spaces: Remove redundant whitespace
    """
    level: NormalizationLevel = NormalizationLevel.STANDARD
    lowercase: bool = False
    remove_urls: bool = True
    remove_emails: bool = True
    remove_extra_spaces: bool = True
