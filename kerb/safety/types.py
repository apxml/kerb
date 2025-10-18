"""Data types and classes for safety operations.

This module defines all data classes and type definitions used in the safety subpackage.
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

from .enums import ContentCategory, PIIType, ToxicityLevel


@dataclass
class SafetyResult:
    """Result from safety check."""
    safe: bool
    score: float  # 0.0 (unsafe) to 1.0 (safe)
    category: ContentCategory = ContentCategory.SAFE
    confidence: float = 1.0
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PIIMatch:
    """Detected PII with metadata."""
    pii_type: PIIType
    text: str
    start: int
    end: int
    confidence: float = 1.0
    context: Optional[str] = None


@dataclass
class ModerationResult:
    """Comprehensive moderation check result."""
    safe: bool
    categories: Dict[ContentCategory, float] = field(default_factory=dict)
    flagged_categories: List[ContentCategory] = field(default_factory=list)
    overall_score: float = 1.0
    toxicity_level: ToxicityLevel = ToxicityLevel.NONE
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Guardrail:
    """Custom safety guardrail."""
    name: str
    check_function: Callable[[str], SafetyResult]
    description: Optional[str] = None
    enabled: bool = True
