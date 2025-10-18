"""Content filtering operations."""

import re
from typing import List, Optional

from .text import remove_emails, remove_phone_numbers


def filter_by_length(
    texts: List[str],
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    unit: str = "chars"
) -> List[str]:
    """Filter texts by length constraints.
    
    Args:
        texts: List of texts
        min_length: Minimum length
        max_length: Maximum length
        unit: Length unit - "chars", "words", "sentences"
        
    Returns:
        Filtered list of texts
        
    Examples:
        >>> filter_by_length(["hi", "hello world", ""], min_length=3)
        ['hello world']
    """
    # Import here to avoid circular dependency
    from .analysis import count_words, count_sentences
    
    result = []
    
    for text in texts:
        if unit == "chars":
            length = len(text)
        elif unit == "words":
            length = count_words(text)
        elif unit == "sentences":
            length = count_sentences(text)
        else:
            raise ValueError(f"Invalid unit: {unit}")
        
        if min_length is not None and length < min_length:
            continue
        if max_length is not None and length > max_length:
            continue
        
        result.append(text)
    
    return result


def filter_by_pattern(
    texts: List[str],
    pattern: str,
    keep_matches: bool = True,
    flags: int = 0
) -> List[str]:
    """Filter texts by regex pattern.
    
    Args:
        texts: List of texts
        pattern: Regex pattern
        keep_matches: Keep matching texts (False to keep non-matching)
        flags: Regex flags
        
    Returns:
        Filtered list of texts
        
    Examples:
        >>> filter_by_pattern(["hello", "world", "hi"], r"^h", keep_matches=True)
        ['hello', 'hi']
    """
    regex = re.compile(pattern, flags)
    
    if keep_matches:
        return [text for text in texts if regex.search(text)]
    else:
        return [text for text in texts if not regex.search(text)]


def filter_profanity(text: str, replacement: str = "***") -> str:
    """Remove or mask profane content.
    
    Args:
        text: Input text
        replacement: Replacement string for profanity
        
    Returns:
        Filtered text
        
    Examples:
        >>> filter_profanity("This is clean text")
        'This is clean text'
    """
    if not text:
        return text
    
    # Basic profanity list (minimal for demonstration)
    profanity_list = ["damn", "hell", "crap"]
    
    result = text
    for word in profanity_list:
        # Case-insensitive word boundary replacement
        pattern = r'\b' + re.escape(word) + r'\b'
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


def filter_pii(text: str, replacement: str = "[REDACTED]") -> str:
    """Remove or mask personally identifiable information.
    
    Args:
        text: Input text
        replacement: Replacement string for PII
        
    Returns:
        Text with PII removed
        
    Examples:
        >>> filter_pii("Email me@example.com or call 555-1234")
        'Email [REDACTED] or call [REDACTED]'
    """
    if not text:
        return text
    
    result = text
    
    # Remove emails
    result = remove_emails(result, replacement)
    
    # Remove phone numbers
    result = remove_phone_numbers(result, replacement)
    
    # Remove SSN-like patterns
    result = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', replacement, result)
    
    # Remove credit card-like patterns
    result = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', replacement, result)
    
    return result


def detect_spam(text: str, threshold: float = 0.5) -> bool:
    """Detect spam or low-quality content.
    
    Args:
        text: Input text
        threshold: Spam score threshold (0-1)
        
    Returns:
        True if text is likely spam
        
    Examples:
        >>> detect_spam("BUY NOW!!! CLICK HERE!!!")
        True
    """
    if not text:
        return True
    
    spam_score = 0.0
    
    # Check for excessive caps
    if len(text) > 10:
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        if caps_ratio > 0.5:
            spam_score += 0.3
    
    # Check for excessive punctuation
    punct_count = len(re.findall(r'[!?]{2,}', text))
    if punct_count > 2:
        spam_score += 0.2
    
    # Check for spam keywords
    spam_keywords = ["buy now", "click here", "limited time", "act now", "free money"]
    text_lower = text.lower()
    for keyword in spam_keywords:
        if keyword in text_lower:
            spam_score += 0.15
    
    # Check for excessive URLs
    url_count = len(re.findall(r'https?://', text, re.IGNORECASE))
    if url_count > 3:
        spam_score += 0.2
    
    return spam_score >= threshold


def filter_by_quality(texts: List[str], min_score: float = 0.5) -> List[str]:
    """Filter by quality metrics.
    
    Args:
        texts: List of texts
        min_score: Minimum quality score (0-1)
        
    Returns:
        List of high-quality texts
        
    Examples:
        >>> filter_by_quality(["Good text here.", "x", "Another good one."])
        ['Good text here.', 'Another good one.']
    """
    # Import here to avoid circular dependency
    from .analysis import measure_readability, count_words
    
    result = []
    
    for text in texts:
        metrics = measure_readability(text)
        
        # Calculate quality score
        score = 1.0
        
        # Penalize very short texts
        if len(text) < 10:
            score -= 0.5
        
        # Penalize spam
        if detect_spam(text):
            score -= 0.4
        
        # Penalize low word count
        word_count = count_words(text)
        if word_count < 3:
            score -= 0.3
        
        if score >= min_score:
            result.append(text)
    
    return result


def filter_non_ascii(text: str, replacement: str = "", keep_extended: bool = True) -> str:
    """Filter or replace non-ASCII characters.
    
    Args:
        text: Input text
        replacement: Replacement for non-ASCII chars
        keep_extended: Keep extended ASCII (128-255)
        
    Returns:
        ASCII-filtered text
        
    Examples:
        >>> filter_non_ascii("Hello 世界")
        'Hello '
    """
    from .text import normalize_whitespace
    
    if not text:
        return text
    
    if keep_extended:
        # Keep ASCII + extended ASCII
        result = ''.join(c if ord(c) < 256 else replacement for c in text)
    else:
        # Keep only standard ASCII
        result = ''.join(c if ord(c) < 128 else replacement for c in text)
    
    # Clean up extra spaces if replacement was empty
    if not replacement:
        result = normalize_whitespace(result)
    
    return result
