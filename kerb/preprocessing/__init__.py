"""Text preprocessing utilities for LLM applications.

This module provides comprehensive text preprocessing tools for cleaning,
normalizing, and preparing text data for LLM processing.

Usage Examples:
    >>> # Common usage - normalize text
    >>> from kerb.preprocessing import normalize_text
    >>> clean = normalize_text("  Hello   World!  ", lowercase=True)
    
    >>> # Text operations
    >>> from kerb.preprocessing.text import (
    ...     normalize_whitespace,
    ...     remove_special_chars,
    ...     truncate_text
    ... )
    
    >>> # Language detection
    >>> from kerb.preprocessing.language import detect_language
    >>> result = detect_language("Bonjour le monde")
    
    >>> # Content filtering
    >>> from kerb.preprocessing.filtering import filter_by_length
    >>> filtered = filter_by_length(["hi", "hello world"], min_length=5)
    
    >>> # Batch processing
    >>> from kerb.preprocessing.batch import preprocess_batch
    >>> processed = preprocess_batch(["  text1  ", "  text2  "])

Organization:
    - Top-level: Core functions and most common operations
    - Submodules: Specialized implementations organized by functionality
        - text: Text normalization, cleaning, case handling
        - language: Language detection and filtering
        - deduplication: Text deduplication operations
        - filtering: Content filtering and quality control
        - analysis: Content analysis and classification
        - transforms: Advanced text transformations
        - batch: Batch processing utilities
        - enums: Enumeration types
        - types: Data classes and type definitions
"""

# Enums
from .enums import (
    NormalizationLevel,
    LanguageDetectionMode,
    DeduplicationMode,
    ContentType,
)

# Data classes
from .types import (
    LanguageResult,
    QualityMetrics,
    NormalizationConfig,
)

# Core text operations (most commonly used)
from .text import (
    normalize_text,
    normalize_whitespace,
    normalize_unicode,
    normalize_quotes,
    normalize_dashes,
    remove_accents,
    clean_html,
    clean_markdown,
    remove_urls,
    remove_emails,
    remove_phone_numbers,
    remove_special_chars,
    remove_extra_whitespace,
    remove_control_chars,
    strip_punctuation,
    normalize_case,
    to_title_case,
    to_sentence_case,
    preserve_acronyms,
    truncate_text,
    split_long_text,
)

# Language detection
from .language import (
    detect_language,
    detect_language_batch,
    is_language,
    filter_by_language,
    get_supported_languages,
)

# Deduplication
from .deduplication import (
    deduplicate_exact,
    deduplicate_fuzzy,
    deduplicate_semantic,
    deduplicate_lines,
    deduplicate_sentences,
    find_duplicates,
    compute_text_hash,
)

# Content filtering
from .filtering import (
    filter_by_length,
    filter_by_pattern,
    filter_profanity,
    filter_pii,
    detect_spam,
    filter_by_quality,
    filter_non_ascii,
)

# Content analysis
from .analysis import (
    classify_content_type,
    detect_code,
    detect_sentiment,
    measure_readability,
    count_words,
    count_sentences,
    count_paragraphs,
)

# Advanced transformations
from .transforms import (
    expand_contractions,
    standardize_numbers,
    standardize_dates,
    extract_entities,
    segment_sentences,
    segment_words,
)

# Batch processing
from .batch import (
    preprocess_batch,
    preprocess_pipeline,
)

# Import submodules for explicit access
from . import text
from . import language
from . import deduplication
from . import filtering
from . import analysis
from . import transforms
from . import batch


__all__ = [
    # Enums
    "NormalizationLevel",
    "LanguageDetectionMode",
    "DeduplicationMode",
    "ContentType",
    
    # Data classes
    "LanguageResult",
    "QualityMetrics",
    "NormalizationConfig",
    
    # Text normalization & cleaning
    "normalize_text",
    "normalize_whitespace",
    "normalize_unicode",
    "normalize_quotes",
    "normalize_dashes",
    "remove_accents",
    "clean_html",
    "clean_markdown",
    "remove_urls",
    "remove_emails",
    "remove_phone_numbers",
    "remove_special_chars",
    "remove_extra_whitespace",
    "remove_control_chars",
    "strip_punctuation",
    
    # Case handling
    "normalize_case",
    "to_title_case",
    "to_sentence_case",
    "preserve_acronyms",
    
    # Language detection
    "detect_language",
    "detect_language_batch",
    "is_language",
    "filter_by_language",
    "get_supported_languages",
    
    # Deduplication
    "deduplicate_exact",
    "deduplicate_fuzzy",
    "deduplicate_semantic",
    "deduplicate_lines",
    "deduplicate_sentences",
    "find_duplicates",
    "compute_text_hash",
    
    # Content filtering
    "filter_by_length",
    "filter_by_pattern",
    "filter_profanity",
    "filter_pii",
    "detect_spam",
    "filter_by_quality",
    "filter_non_ascii",
    
    # Content analysis
    "classify_content_type",
    "detect_code",
    "detect_sentiment",
    "measure_readability",
    "count_words",
    "count_sentences",
    "count_paragraphs",
    
    # Advanced features
    "expand_contractions",
    "standardize_numbers",
    "standardize_dates",
    "extract_entities",
    "segment_sentences",
    "segment_words",
    
    # Batch processing
    "preprocess_batch",
    "preprocess_pipeline",
    
    # Utilities
    "truncate_text",
    "split_long_text",
    
    # Submodules
    "text",
    "language",
    "deduplication",
    "filtering",
    "analysis",
    "transforms",
    "batch",
]

