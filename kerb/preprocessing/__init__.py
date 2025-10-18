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

# Import submodules for explicit access
from . import (analysis, batch, deduplication, filtering, language, text,
               transforms)
# Content analysis
from .analysis import (classify_content_type, count_paragraphs,
                       count_sentences, count_words, detect_code,
                       detect_sentiment, measure_readability)
# Batch processing
from .batch import preprocess_batch, preprocess_pipeline
# Deduplication
from .deduplication import (compute_text_hash, deduplicate_exact,
                            deduplicate_fuzzy, deduplicate_lines,
                            deduplicate_semantic, deduplicate_sentences,
                            find_duplicates)
# Enums
from .enums import (ContentType, DeduplicationMode, LanguageDetectionMode,
                    NormalizationLevel)
# Content filtering
from .filtering import (detect_spam, filter_by_length, filter_by_pattern,
                        filter_by_quality, filter_non_ascii, filter_pii,
                        filter_profanity)
# Language detection
from .language import (detect_language, detect_language_batch,
                       filter_by_language, get_supported_languages,
                       is_language)
# Core text operations (most commonly used)
from .text import (clean_html, clean_markdown, normalize_case,
                   normalize_dashes, normalize_quotes, normalize_text,
                   normalize_unicode, normalize_whitespace, preserve_acronyms,
                   remove_accents, remove_control_chars, remove_emails,
                   remove_extra_whitespace, remove_phone_numbers,
                   remove_special_chars, remove_urls, split_long_text,
                   strip_punctuation, to_sentence_case, to_title_case,
                   truncate_text)
# Advanced transformations
from .transforms import (expand_contractions, extract_entities,
                         segment_sentences, segment_words, standardize_dates,
                         standardize_numbers)
# Data classes
from .types import LanguageResult, NormalizationConfig, QualityMetrics

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
