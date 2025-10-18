"""Content analysis and classification."""

import re
from typing import List

from .enums import ContentType


def classify_content_type(text: str) -> ContentType:
    """Classify text content type.

    Args:
        text: Input text

    Returns:
        ContentType enum value

    Examples:
        >>> classify_content_type("def foo():\\n    pass")
        <ContentType.CODE: 'code'>
    """
    if not text:
        return ContentType.UNKNOWN

    # Check for code
    if detect_code(text):
        return ContentType.CODE

    # Check for JSON
    if text.strip().startswith("{") or text.strip().startswith("["):
        try:
            import json

            json.loads(text)
            return ContentType.JSON
        except:
            pass

    # Check for HTML
    if re.search(r"<[a-z][\s\S]*>", text, re.IGNORECASE):
        return ContentType.HTML

    # Check for Markdown
    if re.search(r"^#{1,6}\s|```|\[.+\]\(.+\)", text, re.MULTILINE):
        return ContentType.MARKDOWN

    # Default to plain text
    return ContentType.PLAIN_TEXT


def detect_code(text: str) -> bool:
    """Detect if text contains code.

    Args:
        text: Input text

    Returns:
        True if text appears to be code

    Examples:
        >>> detect_code("def foo(): return True")
        True
    """
    if not text:
        return False

    # Check for code patterns
    code_patterns = [
        r"\bdef\s+\w+\s*\(",  # Python functions
        r"\bclass\s+\w+",  # Class definitions
        r"\bimport\s+\w+",  # Imports
        r"\bfunction\s+\w+\s*\(",  # JavaScript functions
        r"=>",  # Arrow functions
        r"{\s*\n\s+",  # Code blocks
        r";\s*\n",  # Statement terminators
    ]

    for pattern in code_patterns:
        if re.search(pattern, text):
            return True

    return False


def detect_sentiment(text: str) -> str:
    """Basic sentiment detection.

    Args:
        text: Input text

    Returns:
        Sentiment: "positive", "negative", or "neutral"

    Examples:
        >>> detect_sentiment("I love this!")
        'positive'
    """
    if not text:
        return "neutral"

    text_lower = text.lower()

    # Simple keyword-based sentiment
    positive_words = [
        "love",
        "great",
        "excellent",
        "awesome",
        "wonderful",
        "good",
        "happy",
    ]
    negative_words = ["hate", "bad", "terrible", "awful", "horrible", "poor", "sad"]

    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


def measure_readability(text: str) -> float:
    """Calculate readability score (0-1, higher is more readable).

    Args:
        text: Input text

    Returns:
        Readability score

    Examples:
        >>> score = measure_readability("This is simple text.")
        >>> score > 0.5
        True
    """
    if not text or len(text) < 10:
        return 0.0

    words = count_words(text)
    sentences = count_sentences(text)

    if sentences == 0 or words == 0:
        return 0.0

    # Average word length
    avg_word_length = len(text.replace(" ", "")) / words

    # Average sentence length
    avg_sentence_length = words / sentences

    # Simple readability score
    # Penalize long words and long sentences
    word_score = max(0, 1 - (avg_word_length - 5) / 10)
    sentence_score = max(0, 1 - (avg_sentence_length - 15) / 20)

    return (word_score + sentence_score) / 2


def count_words(text: str) -> int:
    """Smart word counting.

    Args:
        text: Input text

    Returns:
        Word count

    Examples:
        >>> count_words("Hello world, this is a test")
        6
    """
    if not text:
        return 0

    # Split on whitespace and filter empty strings
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    return len(words)


def count_sentences(text: str) -> int:
    """Smart sentence counting.

    Args:
        text: Input text

    Returns:
        Sentence count

    Examples:
        >>> count_sentences("Hello. World! How are you?")
        3
    """
    if not text:
        return 0

    # Split on sentence terminators
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def count_paragraphs(text: str) -> int:
    """Count paragraphs.

    Args:
        text: Input text

    Returns:
        Paragraph count

    Examples:
        >>> count_paragraphs("Para 1\\n\\nPara 2\\n\\nPara 3")
        3
    """
    if not text:
        return 0

    # Split on double newlines
    paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = [p for p in paragraphs if p.strip()]
    return len(paragraphs)
