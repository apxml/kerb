"""Text normalization, cleaning, and manipulation operations."""

import re
import unicodedata
import html
from typing import List, Optional, Union

from kerb.core.enums import TruncateStrategy, CaseMode, validate_enum_or_string
from .enums import NormalizationLevel
from .types import NormalizationConfig


# ============================================================================
# Text Normalization & Cleaning
# ============================================================================

def normalize_text(
    text: str,
    level: NormalizationLevel = NormalizationLevel.STANDARD,
    lowercase: bool = False,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_extra_spaces: bool = True,
    config: Optional[NormalizationConfig] = None
) -> str:
    """Comprehensive text normalization with configurable intensity.
    
    Args:
        text: Input text to normalize
        level: Normalization intensity level (ignored if config is provided)
        lowercase: Convert to lowercase (ignored if config is provided)
        remove_urls: Remove URLs from text (ignored if config is provided)
        remove_emails: Remove email addresses (ignored if config is provided)
        remove_extra_spaces: Remove redundant whitespace (ignored if config is provided)
        config: NormalizationConfig object with all parameters (recommended)
        
    Returns:
        Normalized text
        
    Examples:
        >>> # Using config object (recommended)
        >>> from kerb.preprocessing import NormalizationConfig, NormalizationLevel
        >>> config = NormalizationConfig(
        ...     level=NormalizationLevel.STANDARD,
        ...     lowercase=True,
        ...     remove_urls=True
        ... )
        >>> normalized = normalize_text("Check this: https://example.com", config=config)
        
        >>> # Using individual parameters (backward compatible)
        >>> normalized = normalize_text("HELLO WORLD", lowercase=True)
    """
    # Use config if provided, otherwise use individual parameters
    if config is not None:
        level = config.level
        lowercase = config.lowercase
        remove_urls = config.remove_urls
        remove_emails = config.remove_emails
        remove_extra_spaces = config.remove_extra_spaces
    
    if not text:
        return text
    
    result = text
    
    # Always normalize unicode
    result = normalize_unicode(result)
    
    # URLs and emails
    if remove_urls:
        result = _remove_urls(result)
    if remove_emails:
        result = _remove_emails(result)
    
    # Level-specific processing
    if level == NormalizationLevel.MINIMAL:
        if remove_extra_spaces:
            result = normalize_whitespace(result)
    
    elif level == NormalizationLevel.STANDARD:
        result = normalize_quotes(result)
        result = normalize_dashes(result)
        if remove_extra_spaces:
            result = normalize_whitespace(result)
        result = remove_control_chars(result)
    
    elif level == NormalizationLevel.AGGRESSIVE:
        result = normalize_quotes(result)
        result = normalize_dashes(result)
        result = remove_special_chars(result, keep_basic=True)
        if remove_extra_spaces:
            result = normalize_whitespace(result)
        result = remove_control_chars(result)
    
    # Case normalization
    if lowercase:
        result = result.lower()
    
    return result.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and newlines.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
        
    Examples:
        >>> normalize_whitespace("Hello   world\\n\\n\\ntest")
        'Hello world\\n\\ntest'
    """
    if not text:
        return text
    
    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace multiple newlines with double newline (preserve paragraphs)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove trailing/leading whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Normalize unicode characters.
    
    Args:
        text: Input text
        form: Unicode normalization form (NFC, NFD, NFKC, NFKD)
        
    Returns:
        Unicode-normalized text
        
    Examples:
        >>> normalize_unicode("café")  # Normalizes different accent representations
        'café'
    """
    if not text:
        return text
    return unicodedata.normalize(form, text)


def normalize_quotes(text: str) -> str:
    """Convert smart quotes to standard quotes.
    
    Args:
        text: Input text
        
    Returns:
        Text with standard quotes
        
    Examples:
        >>> normalize_quotes('"Hello" and 'world'")
        '"Hello" and \\'world\\''
    """
    if not text:
        return text
    
    # Smart double quotes to standard
    text = text.replace('"', '"').replace('"', '"')
    
    # Smart single quotes to standard
    text = text.replace(''', "'").replace(''', "'")
    
    # Prime and backtick variations
    text = text.replace('‛', "'").replace('‚', "'")
    text = text.replace('„', '"').replace('‟', '"')
    
    return text


def normalize_dashes(text: str) -> str:
    """Convert various dashes to standard forms.
    
    Args:
        text: Input text
        
    Returns:
        Text with standard dashes
        
    Examples:
        >>> normalize_dashes("em—dash and en–dash")
        'em-dash and en-dash'
    """
    if not text:
        return text
    
    # Convert em and en dashes to hyphen
    text = text.replace('—', '-').replace('–', '-')
    
    # Other dash variants
    text = text.replace('―', '-').replace('‐', '-')
    text = text.replace('‑', '-').replace('⁃', '-')
    
    return text


def remove_accents(text: str) -> str:
    """Remove diacritical marks from text.
    
    Args:
        text: Input text
        
    Returns:
        Text without accents
        
    Examples:
        >>> remove_accents("café résumé")
        'cafe resume'
    """
    if not text:
        return text
    
    # Decompose unicode characters
    nfkd = unicodedata.normalize('NFKD', text)
    
    # Filter out combining marks
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])


def clean_html(text: str, keep_newlines: bool = True) -> str:
    """Remove HTML tags and entities.
    
    Args:
        text: Input text with HTML
        keep_newlines: Keep newlines from <br> and <p> tags
        
    Returns:
        Plain text without HTML
        
    Examples:
        >>> clean_html("<p>Hello <b>world</b></p>")
        'Hello world'
    """
    if not text:
        return text
    
    # Convert common tags to newlines if requested
    if keep_newlines:
        text = re.sub(r'<br\s*/?>|</p>|</div>|</li>', '\n', text, flags=re.IGNORECASE)
    
    # Remove all HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Clean up whitespace
    text = normalize_whitespace(text)
    
    return text.strip()


def clean_markdown(text: str, keep_structure: bool = False) -> str:
    """Remove or normalize markdown formatting.
    
    Args:
        text: Input markdown text
        keep_structure: Keep basic structure (headings, lists)
        
    Returns:
        Plain or lightly formatted text
        
    Examples:
        >>> clean_markdown("# Hello **world**")
        'Hello world'
    """
    if not text:
        return text
    
    result = text
    
    # Remove code blocks
    result = re.sub(r'```[\s\S]*?```', '', result)
    result = re.sub(r'`[^`]+`', '', result)
    
    # Remove links but keep text
    result = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', result)
    
    # Remove images
    result = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', result)
    
    if not keep_structure:
        # Remove headings markers
        result = re.sub(r'^#+\s+', '', result, flags=re.MULTILINE)
        
        # Remove list markers
        result = re.sub(r'^\s*[-*+]\s+', '', result, flags=re.MULTILINE)
        result = re.sub(r'^\s*\d+\.\s+', '', result, flags=re.MULTILINE)
        
        # Remove emphasis
        result = re.sub(r'\*\*([^*]+)\*\*', r'\1', result)
        result = re.sub(r'\*([^*]+)\*', r'\1', result)
        result = re.sub(r'__([^_]+)__', r'\1', result)
        result = re.sub(r'_([^_]+)_', r'\1', result)
        
        # Remove strikethrough
        result = re.sub(r'~~([^~]+)~~', r'\1', result)
    
    # Clean up whitespace
    result = normalize_whitespace(result)
    
    return result.strip()


def remove_urls(text: str, replacement: str = "") -> str:
    """Remove or replace URLs.
    
    Args:
        text: Input text
        replacement: String to replace URLs with
        
    Returns:
        Text without URLs
        
    Examples:
        >>> remove_urls("Check https://example.com for info")
        'Check  for info'
    """
    return _remove_urls(text, replacement)


def _remove_urls(text: str, replacement: str = "") -> str:
    """Internal URL removal."""
    if not text:
        return text
    
    # Match http(s) URLs
    text = re.sub(
        r'https?://[^\s<>"{}|\\^`\[\]]+',
        replacement,
        text,
        flags=re.IGNORECASE
    )
    
    # Match www URLs
    text = re.sub(
        r'www\.[^\s<>"{}|\\^`\[\]]+',
        replacement,
        text,
        flags=re.IGNORECASE
    )
    
    return text


def remove_emails(text: str, replacement: str = "") -> str:
    """Remove or replace email addresses.
    
    Args:
        text: Input text
        replacement: String to replace emails with
        
    Returns:
        Text without email addresses
        
    Examples:
        >>> remove_emails("Contact me@example.com")
        'Contact '
    """
    return _remove_emails(text, replacement)


def _remove_emails(text: str, replacement: str = "") -> str:
    """Internal email removal."""
    if not text:
        return text
    
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        replacement,
        text
    )
    
    return text


def remove_phone_numbers(text: str, replacement: str = "") -> str:
    """Remove or replace phone numbers.
    
    Args:
        text: Input text
        replacement: String to replace phone numbers with
        
    Returns:
        Text without phone numbers
        
    Examples:
        >>> remove_phone_numbers("Call 555-123-4567")
        'Call '
    """
    if not text:
        return text
    
    # Various phone number formats
    patterns = [
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # 555-123-4567
        r'\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b',  # (555) 123-4567
        r'\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b',  # International
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, replacement, text)
    
    return text


def remove_special_chars(text: str, keep_basic: bool = True) -> str:
    """Remove special characters with options.
    
    Args:
        text: Input text
        keep_basic: Keep basic punctuation (.,!?;:)
        
    Returns:
        Text with special characters removed
        
    Examples:
        >>> remove_special_chars("Hello@#$world!")
        'Hello world!'
    """
    if not text:
        return text
    
    if keep_basic:
        # Keep letters, numbers, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'-]', ' ', text)
    else:
        # Keep only letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def remove_extra_whitespace(text: str) -> str:
    """Remove redundant whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Text with single spaces only
        
    Examples:
        >>> remove_extra_whitespace("Hello    world")
        'Hello world'
    """
    if not text:
        return text
    
    # Replace all whitespace sequences with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def remove_control_chars(text: str) -> str:
    """Remove control characters.
    
    Args:
        text: Input text
        
    Returns:
        Text without control characters
        
    Examples:
        >>> remove_control_chars("Hello\\x00world\\x01")
        'Helloworld'
    """
    if not text:
        return text
    
    # Keep newlines and tabs, remove other control characters
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r')
    
    return text


def strip_punctuation(text: str, keep_internal: bool = True) -> str:
    """Remove punctuation with options.
    
    Args:
        text: Input text
        keep_internal: Keep punctuation within words (e.g., apostrophes)
        
    Returns:
        Text with punctuation removed
        
    Examples:
        >>> strip_punctuation("Hello, world!")
        'Hello world'
    """
    if not text:
        return text
    
    if keep_internal:
        # Remove punctuation at word boundaries
        text = re.sub(r'(?<!\w)[^\w\s]+|[^\w\s]+(?!\w)', ' ', text)
    else:
        # Remove all punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
    
    # Clean up spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# ============================================================================
# Case Handling
# ============================================================================

def normalize_case(text: str, mode: Union[CaseMode, str] = "sentence") -> str:
    """Smart case normalization.
    
    Args:
        text: Input text
        mode: Case mode (CaseMode enum or string: "lower", "upper", "title", "sentence")
        
    Returns:
        Case-normalized text
        
    Examples:
        >>> normalize_case("HELLO WORLD", mode=CaseMode.SENTENCE)
        'Hello world'
        
        >>> normalize_case("hello world", mode="title")
        'Hello World'
    """
    if not text:
        return text
    
    # Validate and normalize mode
    mode_val = validate_enum_or_string(mode, CaseMode, "mode")
    if isinstance(mode_val, CaseMode):
        mode_str = mode_val.value
    else:
        mode_str = mode_val
    
    if mode_str == "lower":
        return text.lower()
    elif mode_str == "upper":
        return text.upper()
    elif mode_str == "title":
        return to_title_case(text)
    elif mode_str == "sentence":
        return to_sentence_case(text)
    else:
        return text


def to_title_case(text: str) -> str:
    """Convert to title case.
    
    Args:
        text: Input text
        
    Returns:
        Title-cased text
        
    Examples:
        >>> to_title_case("hello world from python")
        'Hello World From Python'
    """
    if not text:
        return text
    
    # Simple title case
    return text.title()


def to_sentence_case(text: str) -> str:
    """Convert to sentence case.
    
    Args:
        text: Input text
        
    Returns:
        Sentence-cased text
        
    Examples:
        >>> to_sentence_case("hello world. this is a test.")
        'Hello world. This is a test.'
    """
    if not text:
        return text
    
    # Split into sentences and capitalize first letter of each
    sentences = re.split(r'([.!?]+\s+)', text)
    result = []
    
    for i, part in enumerate(sentences):
        if i % 2 == 0 and part:  # Actual sentence text
            result.append(part[0].upper() + part[1:].lower() if len(part) > 0 else part)
        else:
            result.append(part)
    
    return ''.join(result)


def preserve_acronyms(text: str, acronyms: Optional[List[str]] = None) -> str:
    """Smart case conversion preserving acronyms.
    
    Args:
        text: Input text
        acronyms: List of acronyms to preserve (default: common ones)
        
    Returns:
        Text with preserved acronyms
        
    Examples:
        >>> preserve_acronyms("nasa and fbi are agencies", ["NASA", "FBI"])
        'NASA and FBI are agencies'
    """
    if not text:
        return text
    
    if acronyms is None:
        acronyms = ["NASA", "FBI", "CIA", "USA", "UK", "UN", "EU", "WHO", "NATO"]
    
    result = text
    for acronym in acronyms:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(acronym), re.IGNORECASE)
        result = pattern.sub(acronym, result)
    
    return result


# ============================================================================
# Utilities
# ============================================================================

def truncate_text(
    text: str,
    max_length: int,
    strategy: Union[TruncateStrategy, str] = "end",
    suffix: str = "..."
) -> str:
    """Truncate text intelligently.
    
    Args:
        text: Input text
        max_length: Maximum length
        strategy: Truncation strategy (TruncateStrategy enum or string: "end", "middle", "start", "smart")
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
        
    Examples:
        >>> truncate_text("Hello world", max_length=8)
        'Hello...'
        
        >>> truncate_text("Hello world", max_length=8, strategy=TruncateStrategy.MIDDLE)
        'He...ld'
        
        >>> truncate_text("This is a sentence. And another one.", max_length=20, strategy="smart")
        'This is a sentence....'
    """
    if not text or len(text) <= max_length:
        return text
    
    # Validate and normalize strategy
    strategy_val = validate_enum_or_string(strategy, TruncateStrategy, "strategy")
    if isinstance(strategy_val, TruncateStrategy):
        strategy_str = strategy_val.value
    else:
        strategy_str = strategy_val
    
    # Account for suffix length
    available_length = max_length - len(suffix)
    
    if available_length <= 0:
        return text[:max_length]
    
    if strategy_str == "end":
        return text[:available_length] + suffix
    
    elif strategy_str == "start":
        return suffix + text[-available_length:]
    
    elif strategy_str == "middle":
        half = available_length // 2
        return text[:half] + suffix + text[-(available_length - half):]
    
    elif strategy_str == "smart":
        # Try to truncate at sentence boundary
        truncated = text[:available_length]
        
        # Find last sentence ending
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclamation = truncated.rfind('!')
        
        last_sentence_end = max(last_period, last_question, last_exclamation)
        
        if last_sentence_end > available_length * 0.7:
            # Good sentence boundary found
            return text[:last_sentence_end + 1] + suffix
        
        # Fall back to word boundary
        last_space = truncated.rfind(' ')
        if last_space > available_length * 0.8:
            return text[:last_space] + suffix
        
        # No good boundary, just truncate
        return truncated + suffix
    
    else:
        return text[:available_length] + suffix


def split_long_text(
    text: str,
    max_length: int,
    overlap: int = 0,
    preserve_words: bool = True
) -> List[str]:
    """Split text exceeding length limit.
    
    Args:
        text: Input text
        max_length: Maximum length per chunk
        overlap: Overlap between chunks
        preserve_words: Don't split words
        
    Returns:
        List of text chunks
        
    Examples:
        >>> split_long_text("Hello world test", max_length=8)
        ['Hello', 'world', 'test']
    """
    if not text or len(text) <= max_length:
        return [text] if text else []
    
    chunks = []
    
    if preserve_words:
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + (1 if current_chunk else 0)  # +1 for space
            
            if current_length + word_length > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Handle overlap
                if overlap > 0:
                    overlap_words = []
                    overlap_length = 0
                    for w in reversed(current_chunk):
                        if overlap_length + len(w) + 1 <= overlap:
                            overlap_words.insert(0, w)
                            overlap_length += len(w) + 1
                        else:
                            break
                    current_chunk = overlap_words
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(word)
            current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
    else:
        # Character-based splitting
        start = 0
        while start < len(text):
            end = start + max_length
            chunks.append(text[start:end])
            start = end - overlap
    
    return chunks
