"""Core types for parsing utilities.

This module defines the fundamental enums and data classes used across
the parsing subpackage.
"""

from enum import Enum
from typing import Any, List, Optional
from dataclasses import dataclass, field


class ParseMode(Enum):
    """Parsing mode for extracting structured data."""
    STRICT = "strict"  # Fail on any parsing error
    LENIENT = "lenient"  # Try to fix common issues
    BEST_EFFORT = "best_effort"  # Extract what's possible


class ValidationLevel(Enum):
    """Validation strictness level."""
    NONE = "none"  # No validation
    BASIC = "basic"  # Basic type checking
    SCHEMA = "schema"  # Full schema validation
    STRICT = "strict"  # Strict schema + additional constraints


@dataclass
class ParseResult:
    """Result from parsing operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    fixed: bool = False  # Whether output was auto-fixed
    original: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result from validation operation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data: Any = None
